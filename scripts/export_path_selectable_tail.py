"""Export two-layer PSSD cost probe, using existing B16 calibration geometry."""
import copy
import json
import os
from pathlib import Path
from export_long_draft_w4 import (SUPPORTED_LLM_MODELS, llama, llm_wrappers,
    LongDraftCalibration, dataset_builders, Mode, DraftHeadW4Recipe, sha)


def main():
    args=llama._build_parser().parse_args()
    assert args.decoder_model=='qwen3-4b-dspark-w4' and args.compile_only
    assert not args.qat and not args.use_fp16 and args.prefill_ar_len==17
    assert json.loads(Path(args.params).read_text())['n_layers']==2
    artifact=Path(args.artifact).resolve();artifact.mkdir()
    config=copy.copy(SUPPORTED_LLM_MODELS[args.decoder_model])
    object.__setattr__(config,'n_inject',32);object.__setattr__(config,'quant_recipe',DraftHeadW4Recipe)
    SUPPORTED_LLM_MODELS[args.decoder_model]=config
    llm_wrappers.DFlashDecoderInference=LongDraftCalibration
    original=llm_wrappers.process_model_args
    def geometry(*positional,**keywords):
        result=original(*positional,**keywords)
        mode=keywords.get('mode',positional[4] if len(positional)>4 else None)
        if mode==Mode.CALIBRATE:result.ar_len=17
        return result
    llm_wrappers.process_model_args=geometry
    dataset_builders.draft_calibrate_ar_len=lambda max_context_len,n_inject:17
    manifest=dict(arguments=vars(args),source_sha256=sha(Path(__file__)),
        checkpoint_sha256=sha(Path(args.checkpoint)),hidden_sha256=sha(Path(args.dflash_hidden)),
        scope='Untrained PSSD last-two-layer workload with full fc/context injection and full W4 LM head. '
              'Train-only original embedding calibration, NOT calibrated learned latent quality.',
        physical_rows=17,logical_rows=16,n_layers=2,n_inject=32)
    (artifact/'path-selectable-tail-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    args.build_folder=os.path.realpath(args.build_folder);llama.export_llama(args)


if __name__=='__main__':main()
