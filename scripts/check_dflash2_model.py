"""Check the export model against saved, independent upstream FP32 outputs."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from dflash2_model import DFlash2PackedModel
from executorch.examples.models.llama.model_args import ModelArgs
from prepare_dflash2 import OUT, sha

@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    root=Path(__file__).resolve().parents[2]/'MobiSpec.cpp'
    raw=root/'docs/results/raw/dflash2-20260919'
    cfg=json.loads((Path(os.environ['EXECUTORCH_ROOT'])/'examples/models/qwen3_dflash/config/4b_dflash_config.json').read_text())
    cfg.update(max_seq_len=1024,max_context_len=1024,use_kv_cache=True)
    model=DFlash2PackedModel(ModelArgs(**cfg),ar_len=9,n_inject=16).eval()
    sd=torch.load(OUT/'npu-assets/draft-meta.pth',weights_only=True,mmap=True)
    model.load_state_dict(sd,strict=True)
    results=[]
    for prefix in (1,128,512):
        fixture=raw/'reference'/f'p{prefix}'
        features=torch.from_numpy(np.fromfile(fixture/'features.f32',dtype=np.float32).reshape(1,prefix,5,2560))
        tokens=torch.from_numpy(np.fromfile(fixture/'tokens.i32',dtype=np.int32).astype('int64')).unsqueeze(0)
        cache_len=1024-prefix-9
        keys=[torch.zeros(1,8,128,cache_len) for _ in range(5)]
        values=[torch.zeros(1,8,cache_len,128) for _ in range(5)]
        mask=torch.zeros(1,1,9,1024);mask[...,:cache_len]=-float('inf')
        packed,_,_=model(*[features[:,:,i,:] for i in range(5)],model.tok_embeddings(tokens),mask,
            torch.arange(prefix,prefix+9).unsqueeze(0),torch.arange(prefix).unsqueeze(0),*keys,*values)
        out=raw/'npu-f32'/f'p{prefix}';out.mkdir(parents=True,exist_ok=True)
        for name,data in [('logits',packed[...,:151936]),('dflash2_gate',packed[...,151936:])]:
            actual=data.numpy().reshape(-1);expected=np.fromfile(fixture/(name+'.f32'),dtype=np.float32)
            actual.tofile(out/(name+'.f32'))
            error=actual.astype('float64')-expected
            record=dict(prefix=prefix,name=name,max_abs=float(np.max(np.abs(error))),
                relative_l2=float(np.linalg.norm(error)/np.linalg.norm(expected)),
                passed=bool(np.allclose(actual,expected,atol=.002,rtol=.0002)))
            print(record,flush=True);results.append(record)
    report=dict(script_sha256=sha(__file__),model_sha256=sha(Path(__file__).with_name('dflash2_model.py')),
        strict_state_dict_keys=len(sd),criteria=dict(atol=.002,rtol=.0002),results=results,
        passed=all(r['passed'] for r in results))
    (raw/'npu-f32-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['passed']

if __name__=='__main__':main()
