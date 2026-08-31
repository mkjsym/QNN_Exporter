#!/usr/bin/env python3
"""Write a DFlash draft's token embedding table as a raw fp16 file for the runtime to mmap.

Why not reuse the target GGUF's copy
------------------------------------
The QNN drivers already load a `*-outq4` GGUF that carries `token_embd.weight`, but it carries it
as Q4_0, and the draft is far too sensitive for that: substituting the Q4_0 table into the fp32
reference costs 5 of 8 argmax on a fixed input (logit cosine 0.9371). Q8_0 is 7 of 8 and fp16 is
exact, so this writes fp16 -- 778 MB for a 151936x2560 table. It is mmap'd and only the rows a step
actually touches are faulted in, so the resident cost is a few pages, not the file.

Layout: row-major [vocab, dim], little-endian IEEE half. No header -- the runner knows both extents
from params.json, and a header would be one more thing to keep in sync.

Usage:
  dump_draft_embd.py --ckpt dflash4b.pth --out draft4b_embd_f16.bin
"""
import argparse

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="draft checkpoint holding tok_embeddings.weight")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="tok_embeddings.weight")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, weights_only=True, map_location="cpu")
    if args.key not in sd:
        raise SystemExit(f"{args.key} not in checkpoint; have {sorted(sd)[:8]}...")
    w = sd[args.key]
    if w.dim() != 2:
        raise SystemExit(f"expected a 2-D table, got {tuple(w.shape)}")
    w.to(torch.float16).contiguous().numpy().tofile(args.out)
    print(f"{args.out}: {tuple(w.shape)} fp16, {w.numel() * 2 / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
