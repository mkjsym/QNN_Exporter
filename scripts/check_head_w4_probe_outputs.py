#!/usr/bin/env python3
"""Compare saved first-call logits of matched W8/W4 full-vocabulary QNN graphs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    arrays={};hashes={}
    for name in ['matched-full','head-w4']:
        runs=[]
        for repeat in range(3):
            path=a.probe/f'{name}-r{repeat}'/'output.f32';x=np.fromfile(path,dtype='<f4')
            assert x.size==8*151936 and np.isfinite(x).all();runs.append(x.reshape(8,151936));hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert all(np.array_equal(runs[0],x) for x in runs[1:]);arrays[name]=runs[0].astype(np.float64)
    full=arrays['matched-full'];w4=arrays['head-w4'];rows=[]
    for x,y in zip(full,w4):
        top=lambda z:set(np.argpartition(z,-256)[-256:].tolist())
        rows.append(dict(argmax_equal=bool(x.argmax()==y.argmax()),top256_overlap=len(top(x)&top(y))/256,
            relative_rms=float(np.linalg.norm(x-y)/np.linalg.norm(x)),max_abs=float(np.abs(x-y).max())))
    result=dict(passed=True,all_repeat_outputs_exact=True,rows=rows,
        argmax_equal_rows=sum(r['argmax_equal'] for r in rows),mean_top256_overlap=float(np.mean([r['top256_overlap'] for r in rows])),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes=hashes,
        scope='One calibration hidden input, eight output rows. Repeat determinism and quantization differences only; no SD acceptance or performance claim.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
