"""Shared export-only image-packing and provenance utilities."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
MOBI=ROOT.parent/'MobiSpec.cpp'
STUDY=MOBI/'docs/results/raw/vlm-deepspec-20260930'
OUT=ROOT/'artifacts/vlm-deepspec-20260930/export'
TEXT=ROOT.parent/'dflash-models/Qwen3-4B'
INITIAL=ROOT/'artifacts/vlm-deepspec-20260930/initial/backbone.safetensors'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def pack(features,groups,mode,folded):
    n=len(features);pos=torch.arange(n)
    if mode=='full' or not (groups>=0).any():return features,pos
    keep=groups<0;features=features.clone()
    for region in range(8):
        index=(groups==region).nonzero().flatten()
        if not len(index):continue
        raw=features[index].float()
        weights=torch.full((len(index),),1/len(index)) if mode=='pool8' else (raw@folded[region]/2560**.5).softmax(0)
        location=index[len(index)//2];keep[location]=True
        features[location]=(raw*weights[:,None]).sum(0).to(features.dtype)
    return features[keep],pos[keep]
