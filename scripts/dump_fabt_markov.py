"""Write unmodified DSpark Markov tensors for the FABT CPU cost probe."""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    source=a.checkpoint/'model.safetensors';outputs={}
    with safe_open(str(source),framework='pt',device='cpu') as f:
        for name in ('markov_w1','markov_w2'):
            value=f.get_tensor('markov_head.'+name+'.weight').float().contiguous()
            assert tuple(value.shape)==(151936,256)
            dest=a.output/(name+'.f32');dest.write_bytes(value.numpy().tobytes())
            outputs[dest.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    digest=hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    source_hash=digest.hexdigest()
    (a.output/'manifest.json').write_text(json.dumps(dict(source_sha256=source_hash,outputs=outputs,
        scope='Unmodified checkpoint tensors converted to float32; no fitting or precision search'),indent=2)+'\n')


if __name__=='__main__':main()
