#!/usr/bin/env python3
"""Export a shared ten-query graph for matched horizon/mask measurements."""
import copy
import json
import os
from pathlib import Path
import sys

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS, llama
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import DFlashDecoderInference, PADDING_MASK_VALUE
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
from executorch.examples.qualcomm.oss_scripts.llama.dataset import builders as dataset_builders
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import Mode
from export_draft_head_w4 import DraftHeadW4Recipe, sha


def horizon_mask(mask, physical, mode):
    assert physical == 10 and mode in ['dense', 'shield', 'readonly']
    mask = mask.clone()
    mask[..., -physical:] = 0
    if mode == 'shield':
        mask[..., :7, -physical+7:] = PADDING_MASK_VALUE
    elif mode == 'readonly':
        mask[..., -physical+7:] = PADDING_MASK_VALUE
    return mask


class HorizonCalibration(DFlashDecoderInference):
    def make_forward_fn(self):
        forward = super().make_forward_fn()
        count = 0

        def wrapped(model, inputs):
            nonlocal count
            physical = inputs['input_ids'].shape[-1]
            mode = ['dense', 'shield', 'readonly'][count % 3]
            copied = dict(inputs)
            copied['attention_mask'] = [horizon_mask(m, physical, mode) for m in inputs['attention_mask']]
            count += 1
            return forward(model, copied)
        return wrapped


def main():
    args = llama._build_parser().parse_args()
    assert args.decoder_model == 'qwen3-4b-dspark-w4' and args.compile_only and not args.qat and not args.use_fp16
    assert args.prefill_ar_len == 10 and args.max_context_len == 1024
    artifact = Path(args.artifact).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    assert not list(artifact.iterdir())
    config = copy.copy(SUPPORTED_LLM_MODELS[args.decoder_model])
    object.__setattr__(config, 'n_inject', 16)
    object.__setattr__(config, 'quant_recipe', DraftHeadW4Recipe)
    SUPPORTED_LLM_MODELS[args.decoder_model] = config
    llm_wrappers.DFlashDecoderInference = HorizonCalibration
    original_args = llm_wrappers.process_model_args

    def geometry_args(*positional, **keywords):
        result = original_args(*positional, **keywords)
        mode = keywords.get('mode', positional[4] if len(positional) > 4 else None)
        if mode == Mode.CALIBRATE:
            result.ar_len = 10
        return result

    llm_wrappers.process_model_args = geometry_args
    dataset_builders.draft_calibrate_ar_len = lambda max_context_len, n_inject: 10
    manifest = dict(arguments=vars(args), source_sha256=sha(Path(__file__)),
        recipe_sha256=sha(Path(__file__).with_name('export_draft_head_w4.py')),
        checkpoint_sha256=sha(Path(args.checkpoint)), hidden_sha256=sha(Path(args.dflash_hidden)),
        physical_queries=10, logical_horizon=10, core=7, n_inject=16,
        calibration='Original hidden dump and calibration prompts, mask cycle dense/shield/readonly. '
                    'One shared W4/A16 artifact for all three policies and same-graph core7 control.',
        scope='Frozen originalB7 weights, no head/backbone training or precision change. '
              'Allten outputs are logits. Actual runtime masks must match named policies. '
              'Longer horizon alone is a stronger control, not a proven novel algorithm. No speed claim.')
    (artifact/'horizon-export-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    args.build_folder = os.path.realpath(args.build_folder)
    llama.export_llama(args)


if __name__ == '__main__':
    main()
