"""Export native B9 DFlash2 as one QNN draft call, with packed logits and gate."""
import copy
import json
import os
from pathlib import Path
import sys

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0,str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS, SUPPORTED_LLM_MODELS, llama
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
from executorch.examples.qualcomm.oss_scripts.llama.dataset import builders
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import Mode
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import MarginMinMaxObserver, QuantDtype, QuantGranularity
from export_draft_head_w4 import DraftHeadW4Recipe
from dflash2_model import DFlash2PackedModel
from dflash2_aligned_calibration import AlignedDFlash2Calibration
from prepare_dflash2 import OUT, sha


class DFlash2Recipe(DraftHeadW4Recipe):
    def __init__(self,verbose=False):
        super().__init__(verbose)
        self.recipe.add_regex({r'.*(attention_conv|mlp_conv)\.kernel_projection\.conv$',r'selector_gate\.conv$'},
            QuantDtype.use_16a8w,False,act_observer=MarginMinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,priority=3,
            note='DFlash2 dynamic kernel and selector gate projection W8/A16; original body/head W4.')


def main():
    args=llama._build_parser().parse_args()
    assert args.decoder_model=='qwen3-4b-dspark-w4' and args.compile_only and not args.qat and not args.use_fp16
    assert args.prefill_ar_len==9 and args.max_context_len==1024
    assets=OUT/'npu-assets';manifest=json.loads((assets/'manifest.json').read_text())
    assert sha(args.checkpoint)==manifest['outputs']['draft-meta.pth']
    assert sha(args.dflash_hidden)==manifest['outputs']['calibration-hidden.pt']
    artifact=Path(args.artifact).resolve();artifact.mkdir(parents=True,exist_ok=True)
    assert not list(artifact.iterdir())
    config=copy.copy(SUPPORTED_LLM_MODELS[args.decoder_model])
    object.__setattr__(config,'n_inject',16);object.__setattr__(config,'quant_recipe',DFlash2Recipe)
    SUPPORTED_LLM_MODELS[args.decoder_model]=config
    LLM_VARIANT_ARCHS[args.decoder_model]=DFlash2PackedModel
    llm_wrappers.DFlashDecoderInference=AlignedDFlash2Calibration
    original=llm_wrappers.process_model_args
    def geometry(*positional,**keywords):
        result=original(*positional,**keywords)
        mode=keywords.get('mode',positional[4] if len(positional)>4 else None)
        # Only B9 draft calls are used. Identical shapes also keep PTQ encoding
        # transfer exact despite shape-dependent slice canonicalization.
        if mode in (Mode.CALIBRATE,Mode.DECODE,Mode.PREFILL):result.ar_len=9
        return result
    llm_wrappers.process_model_args=geometry
    builders.draft_calibrate_ar_len=lambda max_context_len,n_inject:9
    if os.environ.get('DFLASH2_QDQ_REFERENCE')=='1':
        from dflash2_qdq_reference import capture
        compile_original=llm_wrappers.HybridTextDecoder.compile
        def compile_with_reference(self,request):
            self._encoding_override(self.calibration_prefill.decoder,self.decode.decoder,True)
            capture(self.decode.decoder,artifact)
            return compile_original(self,request)
        llm_wrappers.HybridTextDecoder.compile=compile_with_reference
    plan=dict(arguments=vars(args),class_name='DFlash2PackedModel',physical_rows=9,proposal_rows=[1,9],
              injection_rows=16,cache_rows=999,output_shape=[1,9,152192],vocabulary_columns=[0,151936],
              selector_gate_columns=[151936,152192],selector_rank=256,selector_top_k=16,
              target_block_ids=[1,9,17,25,33],target_hidden_indices=[2,10,18,26,34],
              source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('dflash2_model.py')]},
              assets_manifest_sha256=sha(assets/'manifest.json'),
              scope='Actual DFlash2; the historical decoder_model registry key supplies existing Qwen3 geometry only. '
                    'All 9 block keys open, no padding/truncation; SWA full history valid within max context1024.')
    plan['aligned_calibration']=True
    plan['source_sha256']['dflash2_aligned_calibration.py']=sha(Path(__file__).with_name('dflash2_aligned_calibration.py'))
    (artifact/'dflash2-export-manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    args.build_folder=os.path.realpath(args.build_folder)
    llama.export_llama(args)


if __name__=='__main__':main()
