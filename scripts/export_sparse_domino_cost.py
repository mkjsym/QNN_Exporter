"""Export the eight-query Domino operator for mobile component measurements."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS, LLM_VARIANT_ARCHS, llama
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import DFlashDecoderInference
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
from executorch.examples.qualcomm.oss_scripts.llama.dataset import builders as dataset_builders
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import Mode
from export_domino_w4 import DominoPackedModel
from export_draft_head_w4 import DraftHeadW4Recipe, sha

KNOTS = [0, 1, 2, 3, 4, 5, 6, 15]


class SparseCalibration(DFlashDecoderInference):
    def make_forward_fn(self):
        original = super().make_forward_fn()
        def forward(model, inputs):
            assert inputs['input_ids'].shape[-1] == 8
            copied = dict(inputs)
            copied['attention_mask'] = [m.clone() for m in inputs['attention_mask']]
            for m in copied['attention_mask']:
                m[..., -8:] = 0
            def positioned(*args):
                args = list(args)
                assert args[7].shape == (1, 8) and args[8].shape == (1, 32)
                offsets = torch.tensor(KNOTS, dtype=args[7].dtype, device=args[7].device)[None]
                args[7] = args[7][:, :1] + offsets
                return model(*args)
            return original(positioned, copied)[..., :151936]
        return forward


def main():
    args = llama._build_parser().parse_args()
    assert args.decoder_model == 'qwen3-4b-dspark-w4' and args.compile_only and args.model_mode == 'hybrid'
    assert args.prefill_ar_len == 8 and args.max_context_len == 1024 and not args.qat and not args.use_fp16
    out = Path(args.artifact).resolve(); out.mkdir(parents=True); assert not list(out.iterdir())
    torch.set_num_threads(8); torch.manual_seed(260922401); np.random.seed(260922401); random.seed(260922401)
    config = copy.copy(SUPPORTED_LLM_MODELS[args.decoder_model])
    object.__setattr__(config, 'n_inject', 32); object.__setattr__(config, 'quant_recipe', DraftHeadW4Recipe)
    SUPPORTED_LLM_MODELS[args.decoder_model] = config; LLM_VARIANT_ARCHS[args.decoder_model] = DominoPackedModel
    llm_wrappers.DFlashDecoderInference = SparseCalibration
    old_args = llm_wrappers.process_model_args
    def geometry(*positional, **keywords):
        value = old_args(*positional, **keywords)
        if keywords.get('mode', positional[4] if len(positional) > 4 else None) == Mode.CALIBRATE:
            value.ar_len = 8
        return value
    llm_wrappers.process_model_args = geometry
    dataset_builders.draft_calibrate_ar_len = lambda max_context_len, n_inject: 8
    original_lower = llm_wrappers.to_edge_transform_and_lower_to_qnn
    def freeze(**kwargs):
        model = kwargs['module']['prefill_forward']; inputs = kwargs['inputs']['prefill_forward']
        ep = torch.export.export(model, inputs, strict=True)
        destination = out / 'prefill-frozen.pt2'; torch.export.save(ep, destination)
        (out / 'frozen.json').write_text(json.dumps(dict(file=destination.name, sha256=sha(destination),
            inputs=[dict(shape=list(x.shape), dtype=str(x.dtype)) for x in inputs],
            scope='Post-PTQ CPU graph, prior to QNN lowering. Runtime equivalence not presumed.'), indent=2) + '\n')
        print('SPARSE_DOMINO_FROZEN', flush=True)
        return original_lower(**kwargs)
    llm_wrappers.to_edge_transform_and_lower_to_qnn = freeze
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args),
        physical_queries=8, query_offsets=KNOTS, logical_horizon=16, inject_rows=32, packed_width=152192,
        checkpoint_sha256=sha(Path(args.checkpoint)), hidden_sha256=sha(Path(args.dflash_hidden)),
        sources={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('export_domino_w4.py'),
            Path(__file__).with_name('export_draft_body.py'), Path(__file__).with_name('export_draft_head_w4.py')]},
        scope='Component cost/contract calibration of shared sparse8 operator. Same W4/A16 recipe and checkpoint as native17; new PTQ activation scales are not claimed identical. No new SD algorithm, AL or TPS result.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    args.build_folder = os.path.realpath(args.build_folder); llama.export_llama(args)
    (out / 'complete.json').write_text(json.dumps(dict(pte_sha256=sha(out / 'hybrid_llama_qnn.pte')), indent=2) + '\n')
    print('SPARSE_DOMINO_EXPORTED', flush=True)


if __name__ == '__main__':
    main()
