"""Keep DSpark layers 3/4 for an untrained PSSD tail cost probe."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('params',type=Path)
    p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir()
    original=torch.load(a.source,map_location='cpu',mmap=True,weights_only=True)
    selected={}
    for key,value in original.items():
        if key.startswith('layers.'):
            parts=key.split('.');layer=int(parts[1])
            if layer not in (3,4):continue
            parts[1]=str(layer-3);key='.'.join(parts)
        selected[key]=value
    assert sum(k.endswith('attention.wq.weight') for k in selected)==2
    torch.save(selected,a.output/'tail.pth')
    params={k:v for k,v in json.loads(a.params.read_text()).items() if not k.startswith('draft_')}
    assert params['n_layers']==5;params['n_layers']=2
    (a.output/'params.json').write_text(json.dumps(params,indent=2)+'\n')
    (a.output/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(a.source),
        checkpoint_sha256=sha(a.output/'tail.pth'),source_layers=[3,4],trained=False,
        scope='Untrained architecture shape/cost probe. Original full-context fc and full LM head retained; '
              'embedding inputs will be overridden at runtime. Calibration is a cost probe only.'),indent=2)+'\n')


if __name__=='__main__':main()
