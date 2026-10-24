"""Reuse fixed training inputs for exact no-image context packing checks."""
import json,struct
from pathlib import Path
import numpy as np
from vlm_noimage_export_common import ROOT,MOBI,STUDY,OUT,sha,write

def main():
    previous=ROOT/'artifacts/vlm-residual8-20261001/summary-fixtures'
    plan=json.loads((previous/'manifest.json').read_text());dest=OUT/'summary-fixtures';dest.mkdir(exist_ok=False)
    checks=[]
    for old in plan['checks']:
        if old['mode']!='norm8':continue
        p=Path(old['input']);data=p.read_bytes();n,dim,first,height,width=struct.unpack('5i',data[:20])
        features=np.frombuffer(data[20:],dtype=np.float32).reshape(5,n,dim)
        positions=np.array([i for i in range(n) if not first<=i<first+height*width],dtype=np.int32)
        target=dest/(old['id']+'-expected.bin')
        target.write_bytes(struct.pack('i',len(positions))+positions.tobytes()+features[:,positions,:].copy().tobytes())
        checks.append(dict(id=old['id'],mode='none',input=str(p),input_sha256=sha(p),expected=str(target),expected_sha256=sha(target),removed_image_rows=height*width))
    assert len(checks)==6
    write(dest/'manifest.json',dict(checks=checks,source_sha256=sha(__file__),old_manifest_sha256=sha(previous/'manifest.json')))
    print('NOIMAGE_FIXTURES_READY',flush=True)
if __name__=='__main__':main()
