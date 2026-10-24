"""Export an untrained two-lane DSpark workload for FABT cost feasibility."""
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


class BridgeProbeCalibration(DFlashDecoderInference):
    def make_forward_fn(self):
        forward = super().make_forward_fn()

        def wrapped(model, inputs):
            copied = dict(inputs)
            assert inputs['input_ids'].shape[-1] == 33
            masks = []
            for value in inputs['attention_mask']:
                mask = value.clone()
                mask[..., -33:] = PADDING_MASK_VALUE
                mask[..., :16, -33:-17] = 0
                mask[..., 16:32, -17:-1] = 0
                mask[..., 32:, -33:-17] = 0
                masks.append(mask)
            copied['attention_mask'] = masks
            return forward(model, copied)
        return wrapped


def main():
    args = llama._build_parser().parse_args()
    assert args.decoder_model == 'qwen3-4b-dspark-w4' and args.compile_only
    assert not args.qat and not args.use_fp16
    assert args.prefill_ar_len == 33 and args.max_context_len == 1024
    artifact = Path(args.artifact).resolve()
    artifact.mkdir(parents=True)
    config = copy.copy(SUPPORTED_LLM_MODELS[args.decoder_model])
    object.__setattr__(config, 'n_inject', 32)
    object.__setattr__(config, 'quant_recipe', DraftHeadW4Recipe)
    SUPPORTED_LLM_MODELS[args.decoder_model] = config
    llm_wrappers.DFlashDecoderInference = BridgeProbeCalibration
    original = llm_wrappers.process_model_args

    def geometry_args(*positional, **keywords):
        result = original(*positional, **keywords)
        mode = keywords.get('mode', positional[4] if len(positional) > 4 else None)
        if mode == Mode.CALIBRATE:
            result.ar_len = 33
        return result
    llm_wrappers.process_model_args = geometry_args
    dataset_builders.draft_calibrate_ar_len = lambda max_context_len, n_inject: 33
    manifest = dict(arguments=vars(args), source_sha256=sha(Path(__file__)),
                    recipe_sha256=sha(Path(__file__).with_name('export_draft_head_w4.py')),
                    checkpoint_sha256=sha(Path(args.checkpoint)), hidden_sha256=sha(Path(args.dflash_hidden)),
                    physical_rows=33, logical_rows=32, lane_size=16, n_inject=32,
                    scope='Unmodified B16 weights; two isolated lanes plus one pad row. '
                          'Calibration uses original train-only hidden/prompts. Runtime repeats lane positions. '
                          'Cost and mask isolation probe, NOT trained FABT quality or SD performance.')
    (artifact/'future-anchor-probe-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    args.build_folder = os.path.realpath(args.build_folder)
    llama.export_llama(args)


if __name__ == '__main__':
    main()
