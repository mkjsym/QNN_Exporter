"""Reuse the validated static DFlash graph for a selected data-scale checkpoint."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
MOBI=ROOT.parent/'MobiSpec.cpp'
sys.path.insert(0,str(MOBI/'tools/training'))
from vlm_data_scale import STUDY,TEXT,BASE,checkpoint,sha,write
from vlm_residual8_export_common import pack as normalized_pack
OUT=ROOT/'artifacts/vlm-data-scale-20261006/export'

def candidate():
    value=json.loads((STUDY/'mobile-candidate.json').read_text())
    assert value['selected']
    assert sha(checkpoint(value['arm'],value['step'])/'model.safetensors')==value['checkpoint_sha256']
    return value

def pack(features,groups,mode,summary,prompt_length):
    if mode=='norm8':return normalized_pack(features,groups,mode,summary,prompt_length)
    assert mode in ('full','none')
    positions=torch.arange(len(features),device=features.device)
    if mode=='full':return features,positions
    keep=groups<0
    assert keep.any()
    return features[keep],positions[keep]
