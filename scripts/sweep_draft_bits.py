#!/usr/bin/env python3
"""Fake-quantize the draft's weights at several bit widths and see what survives, before paying
for an export.

Run this first when considering a new weight setting -- an export is ~6 minutes and this is
seconds. It measures WEIGHTS ONLY, though: the real pipeline also quantizes activations and the
KV cache, and that is where this draft's damage actually came from. Treat the numbers as an upper
bound (8-bit weights predict cos 0.9997 here; the full pipeline delivered 0.889).

  ./sweep_draft_bits.py --ckpt dflash4b.pth --hidden hidden_4b.pt
"""
import argparse, json, os, sys, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="draft checkpoint (.pth)")
ap.add_argument("--hidden", required=True, help="target-hidden dump from dump_target_hidden.py")
ap.add_argument("--config", default=None, help="draft config json (default: from $EXECUTORCH_ROOT)")
args = ap.parse_args()

ET = os.environ.get("EXECUTORCH_ROOT")
if ET:
    sys.path.insert(0, os.path.dirname(os.path.abspath(ET)))
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDraftModel

cfg_path = args.config or os.path.join(
    ET or ".", "examples/models/qwen3_dflash/config/4b_dflash_config.json")
cfg=json.load(open(cfg_path))
DIM,NCTX,NBLK,NINJ,CTX=cfg["dim"],5,8,16,1024
SD=torch.load(args.ckpt,weights_only=True)
def build():
    a=ModelArgs(dim=DIM,n_layers=cfg["n_layers"],n_heads=cfg["n_heads"],n_kv_heads=cfg["n_kv_heads"],
      head_dim=cfg["head_dim"],hidden_dim=cfg["hidden_dim"],vocab_size=cfg["vocab_size"],
      norm_eps=cfg["norm_eps"],rope_freq_base=cfg["rope_theta"],use_hf_rope=True,use_qk_norm=True,
      qk_norm_before_rope=True,max_seq_len=CTX,max_context_len=CTX,use_kv_cache=True,max_batch_size=1)
    m=DFlashDraftModel(a,ar_len=NBLK,n_inject=NINJ,n_target_layers=5).eval()
    m.load_state_dict(SD,strict=True); return m,a
def q_chan(t,bits):
    s=t.abs().amax(1,keepdim=True).clamp(min=1e-12)/(2**(bits-1)-1)
    return torch.round(t/s).clamp(-(2**(bits-1)),2**(bits-1)-1)*s
def q_block(t,bits,bs=16):
    o,i=t.shape
    if i%bs: return q_chan(t,bits)
    x=t.reshape(o,i//bs,bs)
    s=x.abs().amax(-1,keepdim=True).clamp(min=1e-12)/(2**(bits-1)-1)
    return (torch.round(x/s).clamp(-(2**(bits-1)),2**(bits-1)-1)*s).reshape(o,i)
m,a=build()
H=torch.load(args.hidden)["hidden"][:NCTX]
h=[torch.zeros(1,NINJ,DIM) for _ in range(5)]
for i in range(5): h[i][0,:NCTX]=H[:,i*DIM:(i+1)*DIM]
tok=torch.full((1,NBLK),151669,dtype=torch.int32); tok[0,0]=26194
emb=m.tok_embeddings(tok.long())
pos=torch.arange(NCTX,NCTX+NBLK,dtype=torch.int32)[None]
pc=torch.zeros(1,NINJ,dtype=torch.int32); pc[0,:NCTX]=torch.arange(NCTX)
cl=CTX-NINJ-NBLK
k=[torch.zeros(1,a.n_kv_heads,a.head_dim,cl) for _ in range(a.n_layers)]
v=[torch.zeros(1,a.n_kv_heads,cl,a.head_dim) for _ in range(a.n_layers)]
mask=torch.full((1,1,NBLK,CTX),-255.0); mask[0,0,:,cl:cl+NCTX]=0.0; mask[0,0,:,cl+NINJ:cl+NINJ+NBLK]=0.0
run=lambda mm: mm(*h,emb,mask,pos,pc,*k,*v)[0][0]
with torch.no_grad(): ref=run(m)
def ev(fn):
    mm,_=build()
    for n,p_ in mm.named_parameters():
        if n.startswith("layers.") and p_.dim()==2: p_.data=fn(n,p_.data)
    with torch.no_grad(): o=run(mm)
    return (float(torch.nn.functional.cosine_similarity(ref.flatten(),o.flatten(),0)),
            int((ref.argmax(-1)==o.argmax(-1)).sum()))
B=0.537e9   # draft body parameter count, for the MB column
print(f"{'setting':<38}{'cos':>8}{'argmax':>9}{'body MB':>9}", flush=True)
print("-"*64, flush=True)
for b in (4,5,6,8):
    c,g=ev(lambda n,t,b=b: q_block(t,b))
    print(f"{'all layers block-16 %d-bit'%b:<38}{c:>8.4f}{g:>7}/8{B*b/8/2**20:>9.0f}", flush=True)
print("-"*64, flush=True)
for L in range(5):
    c,g=ev(lambda n,t,L=L: t if n.startswith(f"layers.{L}.") else q_block(t,4))
    print(f"{'all 4-bit except layer %d kept fp32'%L:<38}{c:>8.4f}{g:>7}/8", flush=True)
print("-"*64, flush=True)
for L in range(5):
    c,g=ev(lambda n,t,L=L: q_block(t,8) if n.startswith(f"layers.{L}.") else q_block(t,4))
    print(f"{'layer %d 8-bit, rest 4-bit'%L:<38}{c:>8.4f}{g:>7}/8{B*(0.2*8+0.8*4)/8/2**20:>9.0f}", flush=True)
