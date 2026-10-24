"""Mixed W4/W8 A16 export of a register-study drafter's inject-only prefill graph (mobile-plan-20261019.md section 11.2, M4).

Same checkpoint and quantization recipe as export_vlm_register_mobile_draft.py, same 16 calibration pages packed per arm, but the graph is
DFlashInjectModel (vlm_register_mobile_inject_model.py): fc + hidden_norm + per-layer wk/wv (+k_norm, RoPE) for --inject_rows committed rows
per call and no block path. Context = 8 (unused cache) + inject rows + 8. Calibration feeds each page in inject-row chunks.
Arguments: --inject_rows N (default 256) plus the llama.py arguments; --max_context_len and --max_seq_len must equal 8 + N + 8.
"""
import argparse, copy, json, os, sys
from pathlib import Path
import torch
from torch.utils._pytree import tree_leaves
ET = Path(os.environ['EXECUTORCH_ROOT']).resolve(); sys.path.insert(0, str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import llama, SUPPORTED_LLM_MODELS, LLM_VARIANT_ARCHS
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import DFlashDecoderInference
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import Mode
from executorch.examples.qualcomm.oss_scripts.llama.dataset import builders
from executorch.examples.qualcomm.oss_scripts.llama.model import dflash_draft
from vlm_register_mobile_export_common import OUT, TEXT, ARMS, ROPE_POSITIONS, pack, extend_rope_tables, sha as sha256, write as atomic_json
from vlm_register_mobile_inject_model import DFlashInjectModel, inject_context, CACHE_ROWS, BLOCK_ROWS
ASSET = None; ARM = None; INJECT = 256; CONTEXT = inject_context(256); CACHE = CACHE_ROWS   # set from --inject_rows in main()
_original_draft_init = dflash_draft.DFlashDraftModel.__init__


def _draft_init_with_long_rope(self, config, *args, **kwargs):
    _original_draft_init(self, config, *args, **kwargs)
    extend_rope_tables(self, config)


dflash_draft.DFlashDraftModel.__init__ = _draft_init_with_long_rope   # DFlashInjectModel inherits it
CALIBRATION_CALLS = []


class RegisterMobileInjectCalibration(DFlashDecoderInference):
    """The 16 calibration pages, packed per arm exactly as for the step graph, fed INJECT rows per call; no cache is kept (the graph reads none)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_inject == INJECT, (self.n_inject, INJECT)
        manifest = json.loads((OUT/'calibration/manifest.json').read_text())
        self.sources = manifest['records']; assert len(self.sources) == 16
        assert all(sha256(r['path']) == r['sha256'] for r in self.sources)
        self.cursor = 0
        self.spec = ARMS[ARM]; self.mode = self.spec['mode']
        self.assets = dict(arm=ARM, summary=torch.load(ASSET/'summary.pt', weights_only=True) if (ASSET/'summary.pt').exists() else None)

    def predict_step(self, module, input_ids, attn_mask=None, hidden_states=(), tok_embedding=None):
        if not hasattr(self, '_aligned_forward'): self._aligned_forward = self.make_forward_fn()
        return self._aligned_forward(module, dict(input_ids=input_ids, attention_mask=attn_mask))

    def make_forward_fn(self):
        @torch.inference_mode()
        def call(model, inputs):
            index = self.cursor; self.cursor += 1
            record = self.sources[index % len(self.sources)]
            sample = torch.load(record['path'], map_location='cpu', weights_only=True)
            ids = sample['input_ids'].long()
            features, positions = pack(sample, self.mode, self.assets, self.spec)
            prefix = min(int(sample['prompt_length'])+[0, 8, 32, 64][index % 4], len(ids)-1)
            valid = positions < prefix
            features, positions = features[valid].reshape(-1, 5, 2560), positions[valid]
            keys = [torch.zeros_like(k) for k in self._k_caches]; values = [torch.zeros_like(v) for v in self._v_caches]
            assert len(keys) == 5 and keys[0].shape == (1, 8, 128, CACHE), (keys[0].shape, CACHE)
            calls = 0
            for begin in range(0, len(features), INJECT):
                end = min(len(features), begin+INJECT); take = end-begin
                hidden = [torch.zeros(1, INJECT, 2560) for _ in range(5)]
                for layer in range(5): hidden[layer][0, :take] = features[begin:end, layer].float()
                block = torch.full((1, BLOCK_ROWS), 151669, dtype=torch.long); block[0, 0] = ids[prefix]
                mask = torch.full((1, 1, BLOCK_ROWS, CONTEXT), -255.); mask[..., CACHE:CACHE+take] = 0; mask[..., -BLOCK_ROWS:-1] = 0
                ctx_positions = torch.zeros((1, INJECT), dtype=torch.int32); ctx_positions[0, :take] = positions[begin:end].int()
                result = tree_leaves(model(*hidden, self._embed(block), mask, torch.arange(prefix, prefix+BLOCK_ROWS, dtype=torch.int32)[None], ctx_positions, *keys, *values))
                assert len(result) == 11 and result[0].shape == (1, BLOCK_ROWS, 2560) and result[1].shape == (1, 8, 128, INJECT) and result[6].shape == (1, 8, INJECT, 128), [tuple(r.shape) for r in result]
                calls += 1
            row = dict(id=record['id'], prefix=prefix, source_sha256=record['sha256'], call=index, packed_rows=int(len(features)), graph_calls=calls)
            CALIBRATION_CALLS.append(row); print('REGISTER_MOBILE_INJECT_CALIBRATION_CALL', json.dumps(row), flush=True)
            return result[0]
        return call


def main():
    global ASSET, ARM, INJECT, CONTEXT, CACHE
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument('--inject_rows', type=int, default=256)
    own, rest = pre.parse_known_args()
    args = llama._build_parser().parse_args(rest)
    INJECT = own.inject_rows; CONTEXT = inject_context(INJECT); CACHE = CACHE_ROWS
    assert args.decoder_model == 'qwen3-4b-dflash-w4' and args.compile_only and not args.qat and not args.use_fp16
    assert args.prefill_ar_len == BLOCK_ROWS and args.max_context_len == CONTEXT and args.max_seq_len == CONTEXT, (args.max_context_len, args.max_seq_len, CONTEXT)
    asset = Path(args.checkpoint).parent; ASSET = asset
    manifest = json.loads((asset/'manifest.json').read_text()); ARM = manifest['arm']
    audit = json.loads((asset/'fp32-inject-audit.json').read_text())
    assert audit['passed'] and audit['inject_rows'] == INJECT and audit['checkpoint_sha256'] == manifest['checkpoint_sha256']
    assert sha256(args.checkpoint) == manifest['outputs']['draft-meta.pth']
    calibration = OUT/'calibration'; calibration_manifest = json.loads((calibration/'manifest.json').read_text())
    assert sha256(args.dflash_hidden) == calibration_manifest['outputs']['calibration-hidden.pt']
    assert len(args.calib_samples) == 1 and sha256(args.calib_samples[0]) == calibration_manifest['outputs']['calibration-prompts.json']
    artifact = Path(args.artifact).resolve(); artifact.mkdir(parents=True, exist_ok=True); assert not list(artifact.iterdir())
    config = copy.copy(SUPPORTED_LLM_MODELS[args.decoder_model]); object.__setattr__(config, 'tokenizer_repo_id', str(TEXT)); object.__setattr__(config, 'n_inject', INJECT)
    SUPPORTED_LLM_MODELS[args.decoder_model] = config
    LLM_VARIANT_ARCHS[args.decoder_model] = DFlashInjectModel
    llm_wrappers.DFlashDecoderInference = RegisterMobileInjectCalibration
    original = llm_wrappers.process_model_args
    def geometry(*positional, **keywords):
        result = original(*positional, **keywords)
        mode = keywords.get('mode', positional[4] if len(positional) > 4 else None)
        if mode in (Mode.CALIBRATE, Mode.DECODE, Mode.PREFILL): result.ar_len = BLOCK_ROWS
        return result
    llm_wrappers.process_model_args = geometry
    builders.draft_calibrate_ar_len = lambda max_context_len, n_inject: BLOCK_ROWS
    atomic_json(artifact/'vlm-register-mobile-inject-export-manifest.json', dict(arguments=vars(args), arm=ARM, mode=manifest['mode'], graph='inject-only draft prefill (DFlashInjectModel)',
        asset_manifest_sha256=sha256(asset/'manifest.json'), calibration_manifest_sha256=sha256(calibration/'manifest.json'), source_sha256=sha256(Path(__file__)),
        model_sha256=sha256(Path(__file__).with_name('vlm_register_mobile_inject_model.py')), inject_audit_sha256=sha256(asset/'fp32-inject-audit.json'),
        physical_rows=BLOCK_ROWS, injection_rows=INJECT, cache_rows=CACHE, max_context_len=CONTEXT, rope_positions=ROPE_POSITIONS, tokenizer_source=str(TEXT),
        precision='Unchanged Qwen3_DFlashDraftW4A16 recipe; mixed W4/W8, A16; K/V outputs float; no block path, no LM head',
        scope='Register study doc20k drafters, M4 inject-only prefill graphs; document-page calibration (16 train pages, cap 3072) packed per visual mode, chunked by the inject width'))
    args.build_folder = os.path.realpath(args.build_folder)
    llama.export_llama(args)
    assert {r['id'] for r in CALIBRATION_CALLS} == {r['id'] for r in calibration_manifest['records']}
    atomic_json(artifact/'aligned-calibration-calls.json', CALIBRATION_CALLS)


if __name__ == '__main__':
    main()
