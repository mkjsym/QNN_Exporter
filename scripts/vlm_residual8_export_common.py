"""Export-only packing for normalized question summaries."""
from pathlib import Path
import torch
import torch.nn.functional as F
from vlm_deepspec_export_common import ROOT,MOBI,TEXT,INITIAL,sha,write
STUDY=MOBI/'docs/results/raw/vlm-residual8-20261001'
BASE=MOBI/'docs/results/raw/vlm-deepspec-20260930'
OUT=ROOT/'artifacts/vlm-residual8-20261001/export'


def pack(features,groups,mode,summary,prompt_length):
    pos=torch.arange(len(features),device=features.device)
    if not (groups>=0).any():return features,pos
    assert 0<prompt_length<=len(features)
    deep=features[:,-2560:].float()
    query=summary['summary_queries'].float()
    if mode!='staticres8':query=query+F.linear(deep[prompt_length-1],summary['summary_condition.weight'].float())[None]
    query=F.normalize(query,dim=-1,eps=.01)
    keep=groups<0;output=features.clone()
    for region in range(8):
        index=(groups==region).nonzero().flatten()
        if not len(index):continue
        keys=F.normalize(F.linear(deep[index],summary['summary_key.weight'].float()),dim=-1,eps=.01)
        attention=(keys@query[region]/.5).softmax(0)
        alpha=1. if mode=='norm8' else .25
        weights=(1-alpha)/len(index)+alpha*attention
        location=index[len(index)//2];keep[location]=True
        output[location]=(features[index].float()*weights[:,None]).sum(0).to(features.dtype)
    return output[keep],pos[keep]
