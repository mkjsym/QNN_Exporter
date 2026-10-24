#!/usr/bin/env python3
"""Compare one fixed-input device output with body/full export references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(8)
    done=json.loads((args.probe/'result.json').read_text());assert done['complete']
    def load(path,width):
        x=np.fromfile(path,dtype='<f4');assert x.size==8*width and np.isfinite(x).all()
        return torch.from_numpy(x.reshape(8,width))
    body=load(args.probe/'graph-body-r0/output.f32',2560)
    full=load(args.probe/'graph-matched-full-r0/output.f32',151936)
    old=load(args.probe/'graph-existing-full-r0/output.f32',151936)
    reference=load(args.artifact/'head-probe-reference.f32',151936)
    body_reference=load(args.artifact/'head-probe-body-reference.f32',2560)
    sd=torch.load(args.checkpoint,mmap=True,weights_only=True,map_location='cpu')
    weight=sd['output.weight'].half().float()
    with torch.no_grad():projected=torch.nn.functional.linear(body,weight)
    def compare(a,b):
        result=dict(cosine_by_row=torch.nn.functional.cosine_similarity(a,b,dim=1).tolist(),
            relative_rms=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)))
        if a.shape[1]==151936:
            aa,bb=a.argmax(1),b.argmax(1);at=torch.topk(a,256,dim=1).indices;bt=torch.topk(b,256,dim=1).indices
            result.update(argmax_a=aa.tolist(),argmax_b=bb.tolist(),argmax_equal_rows=int((aa==bb).sum()),
                top256_overlap_by_row=[len(set(x.tolist())&set(y.tolist()))/256 for x,y in zip(at,bt)])
        return result
    result=dict(scope='One fixed calibration input, eight rows, all block rows open. Export/precision smoke check only; no AR acceptance or SD throughput inference.',
        body_vs_fp32_body=compare(body,body_reference),matched_full_vs_fp32=compare(full,reference),
        existing_full_vs_fp32=compare(old,reference),body_fp16_cpu_head_vs_matched_full=compare(projected,full),
        body_fp16_cpu_head_vs_fp32=compare(projected,reference),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest=args.artifact/'head-probe-output-check.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    projected.numpy().tofile(args.artifact/'head-probe-body-cpu-logits.f32')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
