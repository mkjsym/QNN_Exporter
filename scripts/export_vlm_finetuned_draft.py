"""Export a validated full-context VLM-finetuned DFlash with aligned train calibration."""
import copy
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils._pytree import tree_leaves

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import llama, SUPPORTED_LLM_MODELS
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import DFlashDecoderInference
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import Mode
from executorch.examples.qualcomm.oss_scripts.llama.dataset import builders
from prepare_vlm_finetuned_draft import OUT, base, sha256, atomic_json

CALIBRATION_CALLS = []


class AlignedVLMCalibration(DFlashDecoderInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads((OUT/'calibration/manifest.json').read_text())
        self.sources = manifest['records']
        assert len(self.sources) == 16
        assert all(sha256(record['path']) == record['sha256'] for record in self.sources)
        self.cursor = 0

    def predict_step(self, module, input_ids, attn_mask=None, hidden_states=(), tok_embedding=None):
        if not hasattr(self, '_aligned_forward'):
            self._aligned_forward = self.make_forward_fn()
        return self._aligned_forward(module, dict(input_ids=input_ids, attention_mask=attn_mask))

    def make_forward_fn(self):
        @torch.inference_mode()
        def call(model, inputs):
            index = self.cursor
            self.cursor += 1
            record = self.sources[index % len(self.sources)]
            item = torch.load(record['path'], map_location='cpu', weights_only=False, mmap=True)
            assert item['case']['split'] == 'train' and item['layer_ids'] == [1, 9, 17, 25, 33]
            features, ids = item['target_features'], item['input_ids']
            prefix = min(item['prompt_length']+[0, 8, 32, 64][index % 4], len(ids)-1)
            keys = [torch.zeros_like(k) for k in self._k_caches]
            values = [torch.zeros_like(v) for v in self._v_caches]
            assert len(keys) == 5 and keys[0].shape == (1, 8, 128, 1000)
            for begin in range(0, prefix, 16):
                end = min(prefix, begin+16)
                take = end-begin
                hidden = [torch.zeros(1, 16, 2560) for _ in range(5)]
                for layer in range(5):
                    hidden[layer][0, :take] = features[begin:end, layer].float()
                block = torch.full((1, 8), 151669, dtype=torch.long)
                block[0, 0] = ids[end]
                mask = torch.full((1, 1, 8, 1024), -255.)
                mask[..., :begin] = 0
                mask[..., 1000:1000+take] = 0
                mask[..., -8:] = 0
                result = tree_leaves(model(*hidden, self._embed(block), mask,
                    torch.arange(end, end+8, dtype=torch.int32)[None],
                    torch.arange(begin, begin+16, dtype=torch.int32)[None], *keys, *values))
                assert len(result) == 11 and result[0].shape == (1, 8, 151936)
                for layer in range(5):
                    keys[layer][..., begin:end] = result[1+layer][..., :take]
                    values[layer][:, :, begin:end, :] = result[6+layer][:, :, :take, :]
            row = dict(id=record['id'], prefix=prefix, source_sha256=record['sha256'], call=index)
            CALIBRATION_CALLS.append(row)
            print('VLM_ALIGNED_CALIBRATION', json.dumps(row), flush=True)
            return result[0]
        return call


def main():
    args = llama._build_parser().parse_args()
    assert args.decoder_model == 'qwen3-4b-dflash-w4'
    assert args.compile_only and not args.qat and not args.use_fp16
    assert args.prefill_ar_len == 8 and args.max_context_len == 1024
    asset = Path(args.checkpoint).parent
    manifest = json.loads((asset/'manifest.json').read_text())
    audit = json.loads((asset/'fp32-audit.json').read_text())
    assert audit['passed'] and audit['checkpoint_sha256'] == manifest['checkpoint_sha256']
    assert sha256(args.checkpoint) == manifest['outputs']['draft-meta.pth']
    calibration = OUT/'calibration'
    calibration_manifest = json.loads((calibration/'manifest.json').read_text())
    assert sha256(args.dflash_hidden) == calibration_manifest['outputs']['calibration-hidden.pt']
    assert len(args.calib_samples) == 1
    assert sha256(args.calib_samples[0]) == calibration_manifest['outputs']['calibration-prompts.json']
    artifact = Path(args.artifact).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    assert not list(artifact.iterdir())
    config = copy.copy(SUPPORTED_LLM_MODELS[args.decoder_model])
    object.__setattr__(config, 'tokenizer_repo_id', str(base.TEXT_TARGET))
    SUPPORTED_LLM_MODELS[args.decoder_model] = config
    llm_wrappers.DFlashDecoderInference = AlignedVLMCalibration
    original = llm_wrappers.process_model_args
    def geometry(*positional, **keywords):
        result = original(*positional, **keywords)
        mode = keywords.get('mode', positional[4] if len(positional) > 4 else None)
        if mode in (Mode.CALIBRATE, Mode.DECODE, Mode.PREFILL):
            result.ar_len = 8
        return result
    llm_wrappers.process_model_args = geometry
    builders.draft_calibrate_ar_len = lambda max_context_len, n_inject: 8
    atomic_json(artifact/'vlm-ft-export-manifest.json', dict(arguments=vars(args),
        asset_manifest_sha256=sha256(asset/'manifest.json'), calibration_manifest_sha256=sha256(calibration/'manifest.json'),
        source_sha256=sha256(Path(__file__)), physical_rows=8, proposal_rows=7, injection_rows=16, cache_rows=1000,
        tokenizer_source=str(base.TEXT_TARGET),
        precision='Unchanged Qwen3_DFlashDraftW4A16 recipe; mixed W4/W8, A16, KV8; original frozen embedding/head',
        scope='Full-context VLM fine-tuning control; no image pooling or novel SD policy. Train-only aligned visual hidden calibration.'))
    args.build_folder = os.path.realpath(args.build_folder)
    llama.export_llama(args)
    assert {r['id'] for r in CALIBRATION_CALLS} == {r['id'] for r in calibration_manifest['records']}
    atomic_json(artifact/'aligned-calibration-calls.json', CALIBRATION_CALLS)


if __name__ == '__main__':
    main()
