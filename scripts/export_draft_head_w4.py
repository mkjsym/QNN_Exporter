#!/usr/bin/env python3
"""Export a full-vocabulary DSpark control with only the output head changed to W4.

All model/export changes stay in this process; ExecuTorch files are untouched.
This is a quantization control, not a new speculative-decoding algorithm.
"""
import copy
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import sys
ET=Path(os.environ['EXECUTORCH_ROOT']).resolve();sys.path.insert(0,str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS,llama
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    Qwen3_DFlashDraftW4A16QuantRecipe,MarginMinMaxObserver,QuantDtype,QuantGranularity)


class DraftHeadW4Recipe(Qwen3_DFlashDraftW4A16QuantRecipe):
    def __init__(self,verbose=False):
        super().__init__(verbose=verbose)
        self.recipe.add_regex({r'output\.conv'},QuantDtype.use_16a4w_block,False,
            act_observer=MarginMinMaxObserver,granularity=QuantGranularity.PER_BLOCK,
            extra_kwargs={'block_size':(1,16,1,1)},priority=2,
            note='Experimental full-vocabulary LM head W4, group16; body recipe unchanged.')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    args=llama._build_parser().parse_args()
    if args.decoder_model!='qwen3-4b-dspark-w4' or not args.compile_only or args.qat or args.use_fp16:
        raise ValueError('Require compile-only DSpark4B W4/A16 PTQ')
    artifact=Path(args.artifact).resolve();artifact.mkdir(parents=True,exist_ok=True)
    if list(artifact.iterdir()):raise FileExistsError('Use a new empty artifact directory')
    old=SUPPORTED_LLM_MODELS[args.decoder_model]
    config=copy.copy(old);object.__setattr__(config,'quant_recipe',DraftHeadW4Recipe)
    assert old.quant_recipe is Qwen3_DFlashDraftW4A16QuantRecipe
    SUPPORTED_LLM_MODELS[args.decoder_model]=config
    source=ET/'examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py'
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Only output.conv weights: W8 per-channel to W4 per-block group16; full vocabulary, unchanged body architecture/checkpoint/calibration/margins/KV.',
        arguments=vars(args),source_sha256=sha(Path(__file__)),upstream_recipe_sha256=sha(source),
        checkpoint_sha256=sha(Path(args.checkpoint)),hidden_sha256=sha(Path(args.dflash_hidden)),
        device_inference_run=False,sd_performance_established=False,novelty_established=False)
    (artifact/'head-w4-export-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    args.build_folder=os.path.realpath(args.build_folder)
    llama.export_llama(args)


if __name__=='__main__':main()
