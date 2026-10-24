"""Record the exact calibrated QDQ model before QNN lowering."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils._pytree import tree_leaves
from prepare_dflash2 import OUT,sha

@torch.inference_mode()
def capture(model,artifact):
    torch.set_num_threads(8)
    source=Path(__file__).resolve().parents[2]/'MobiSpec.cpp/docs/results/raw/dflash2-20260919/reference'
    embedding=np.memmap(OUT/'npu-assets/embedding-f16.bin',dtype=np.float16,mode='r',shape=(151936,2560))
    out=artifact/'qdq-reference';out.mkdir(exist_ok=True)
    records=[]
    for prefix in (1,128,512):
        fixture=source/f'p{prefix}'
        features=np.fromfile(fixture/'features.f32',dtype=np.float32).reshape(prefix,5,2560)
        tokens=np.fromfile(fixture/'tokens.i32',dtype=np.int32)
        block=torch.from_numpy(embedding[tokens].astype('float32')).unsqueeze(0)
        pos=torch.arange(prefix,prefix+9,dtype=torch.int32).unsqueeze(0)
        keys=[torch.zeros(1,8,128,999) for _ in range(5)]
        values=[torch.zeros(1,8,999,128) for _ in range(5)]
        for begin in range(0,prefix,16):
            take=min(16,prefix-begin)
            hidden=[]
            for l in range(5):
                h=torch.zeros(1,16,2560);h[:,:take]=torch.from_numpy(features[begin:begin+take,l].copy());hidden.append(h)
            mask=torch.full((1,1,9,1024),-255.)
            mask[...,:begin]=0;mask[...,999:999+take]=0;mask[...,-9:]=0
            pc=torch.arange(begin,begin+16,dtype=torch.int32).unsqueeze(0)
            result=tree_leaves(model(*hidden,block,mask,pos,pc,*keys,*values));assert len(result)==11
            for l in range(5):
                keys[l][...,begin:begin+take]=result[1+l][...,:take]
                values[l][:,:,begin:begin+take,:]=result[6+l][:,:,:take,:]
        packed=result[0].float().numpy();assert packed.shape==(1,9,152192) and np.isfinite(packed).all()
        path=out/f'p{prefix}.f32';packed.tofile(path)
        records.append(dict(prefix=prefix,sha256=sha(path),shape=list(packed.shape)))
        print('DFLASH2_QDQ_REFERENCE',prefix,flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(records=records,source_sha256=sha(__file__),
        scope='Actual calibrated QDQ decode graph after cache-encoding override. Same B9/inject16/cache999 and FP16 embedding as device; -255/0 mask, fullblock9, repeated causal commits.'),indent=2)+'\n')
