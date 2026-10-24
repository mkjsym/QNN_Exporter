"""Reconstruct the existing target recipe and retain its frozen QDQ graph.

This is a separate baseline until device comparisons establish its relationship
to the deployed target. No ExecuTorch source files are modified.
"""
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ET=Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0,str(ET.parent))
import numpy as np
import torch
from executorch.examples.qualcomm.oss_scripts.llama import llama
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    args=llama._build_parser().parse_args()
    assert args.decoder_model=='qwen3-4b-sqnr' and args.compile_only
    assert args.model_mode=='hybrid' and args.prefill_ar_len==32 and args.max_context_len==1024
    assert not args.qat and not args.use_fp16
    out=Path(args.artifact).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not list(out.iterdir()),'Use a new empty directory'
    torch.manual_seed(26091824);np.random.seed(26091824);random.seed(26091824)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=26091824,
        scope='Reconstructed Qwen3-4B SQNR baseline; NOT presumed equal to deployed canonical target.',
        args=vars(args),source_sha256=sha(__file__),checkpoint_sha256=sha(args.checkpoint),
        calibration_sha256={str(p):sha(p) for p in args.calib_samples},
        upstream_sha256={str(p.relative_to(ET)):sha(p) for p in [
            ET/'examples/qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py',
            ET/'examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py',
            ET/'examples/qualcomm/oss_scripts/llama/model/static_llama.py',
            ET/'examples/qualcomm/oss_scripts/llama/__init__.py']},
        device_equivalence_established=False,performance_established=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    original=llm_wrappers.to_edge_transform_and_lower_to_qnn

    def freeze_then_lower(*positional,**kw):
        assert not positional and set(kw['module'])=={'kv_forward','prefill_forward'}
        saved={}
        for name in ('kv_forward','prefill_forward'):
            model=kw['module'][name];inputs=kw['inputs'][name]
            ep=torch.export.export(model,inputs,strict=True)
            dest=out/(name+'-frozen.pt2')
            print('EFPV_SAVE_FROZEN',name,flush=True)
            torch.export.save(ep,dest)
            saved[name]=dict(file=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),
                user_input_shapes=[dict(shape=list(t.shape),dtype=str(t.dtype)) for t in inputs],
                graph_nodes=len(list(model.graph.nodes)))
            (out/(name+'-graph.txt')).write_text(model.code)
            nodes=[]
            for n in model.graph.nodes:
                value=n.meta.get('val')
                nodes.append(dict(name=n.name,op=n.op,target=str(n.target),
                    shape=list(value.shape) if isinstance(value,torch.Tensor) else None,
                    modules=list(n.meta.get('nn_module_stack',{}).values())))
            (out/(name+'-nodes.json')).write_text(json.dumps(nodes,indent=2,default=str)+'\n')
            del ep;gc.collect()
        saved['constant_methods']=kw.get('constant_methods',{})
        (out/'frozen-manifest.json').write_text(json.dumps(saved,indent=2)+'\n')
        print('EFPV_FROZEN_READY',flush=True)
        return original(**kw)

    llm_wrappers.to_edge_transform_and_lower_to_qnn=freeze_then_lower
    args.build_folder=os.path.realpath(args.build_folder)
    llama.export_llama(args)
    manifest['export_complete']=True
    manifest['pte_sha256']=sha(out/'hybrid_llama_qnn.pte')
    (out/'complete.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('EFPV_FROZEN_EXPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
