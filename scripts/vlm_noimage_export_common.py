"""No-visual-token packing with the same trained checkpoint and QNN recipe."""
from vlm_deepspec_export_common import ROOT,MOBI,TEXT,INITIAL,sha,write
import torch
BASE=MOBI/'docs/results/raw/vlm-deepspec-20260930'
STUDY=MOBI/'docs/results/raw/vlm-noimage-20261004'
OUT=ROOT/'artifacts/vlm-noimage-20261004/export'

def pack(features,groups,mode,folded=None):
    assert mode=='none'
    keep=groups<0
    return features[keep],torch.arange(len(features))[keep]
