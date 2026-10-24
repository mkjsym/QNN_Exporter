#!/usr/bin/env python3
"""Export the common b16 control with explicit physical padding and injection size."""
import copy
import json
import torch
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
from export_draft_head_w4 import sha, DraftHeadW4Recipe
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import Qwen3_DFlashDraftW4A16QuantRecipe
from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
from export_domino_w4 import DominoPackedModel


class LongDraftCalibration(DFlashDecoderInference):
    def make_forward_fn(self):
        forward = super().make_forward_fn()

        def wrapped(model, inputs):
            copied = dict(inputs)
            physical = inputs['input_ids'].shape[-1]
            masks = []
            for mask in inputs['attention_mask']:
                mask = mask.clone()
                mask[..., -physical:] = PADDING_MASK_VALUE
                # Runtime opens the first 16 proposal input rows and closes row 17.
                opened = min(16, physical)
                mask[..., -physical:-physical + opened if opened < physical else None] = 0
                masks.append(mask)
            copied['attention_mask'] = masks
            return forward(model, copied)
        return wrapped


class Native8BCalibration(LongDraftCalibration):
    def make_forward_fn(self):
        forward = super().make_forward_fn()
        return lambda model, inputs: forward(model, inputs)[..., :151936]

def main():
    torch.set_num_threads(8)
    args = llama._build_parser().parse_args()
    assert args.decoder_model == 'qwen3-8b-dspark-w4' and args.compile_only and not args.qat and not args.use_fp16
    assert args.prefill_ar_len == 17 and args.max_context_len == 1024
    artifact = Path(args.artifact).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    assert not list(artifact.iterdir())
    config = copy.copy(SUPPORTED_LLM_MODELS[args.decoder_model])
    object.__setattr__(config, 'n_inject', 32)
    object.__setattr__(config, 'quant_recipe', DraftHeadW4Recipe)
    SUPPORTED_LLM_MODELS[args.decoder_model] = config
    LLM_VARIANT_ARCHS[args.decoder_model] = DominoPackedModel
    llm_wrappers.DFlashDecoderInference = Native8BCalibration
    original_args = llm_wrappers.process_model_args
    def geometry_args(*positional, **keywords):
        result = original_args(*positional, **keywords)
        mode = keywords.get('mode', positional[4] if len(positional) > 4 else None)
        if mode == Mode.CALIBRATE:
            result.ar_len = 17
        return result
    llm_wrappers.process_model_args = geometry_args
    dataset_builders.draft_calibrate_ar_len = lambda max_context_len, n_inject: 17
    manifest = dict(arguments=vars(args), source_sha256=sha(Path(__file__)),
                    recipe_sha256=sha(Path(__file__).with_name('export_draft_head_w4.py')),
                    checkpoint_sha256=sha(Path(args.checkpoint)), hidden_sha256=sha(Path(args.dflash_hidden)),
                    physical_rows=17, logical_block_size=16, n_inject=32,
                    scope='Official Domino 8B, W4 body / W4 output-head group16 matching the new DSpark8B control. '
                          'Calibration opens all 16 logical block keys; the extra physical row stays masked. '
                          'No split/conditional new algorithm is represented by this export.')
    (artifact / 'long-draft-export-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    args.build_folder = os.path.realpath(args.build_folder)
    llama.export_llama(args)


if __name__ == '__main__':
    main()
