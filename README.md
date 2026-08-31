# QNN_Exporter

Build the Hexagon-NPU context binaries that
[MobiSpec.cpp](https://github.com/mkjsym/MobiSpec.cpp) runs speculative decoding on: a Qwen3
**target** and a DFlash/DSpark **draft**, both as QNN contexts, plus the `params.json` and the
host-side embedding table the runtime needs beside them.

Clone this repo alone and you have everything on the build side. It carries a patch against a
pinned ExecuTorch commit (model registrations, quantization recipes, a DFlash draft model, the
calibration path for hidden-state inputs), the driver scripts, and the calibration set the
reference numbers were measured with.

```
patch/     the ExecuTorch patch + apply.sh
scripts/   export / extract / params / embedding-dump drivers
data/      calibration prompts (39 chat-templated Spec-Bench samples)
docs/      full export guide, patch notes, quantization rationale, peak-memory notes
```

---

## What you need

| | version | note |
|---|---|---|
| ExecuTorch | pinned `88a5f60` | `apply.sh` checks this and refuses otherwise |
| QAIRT (QNN SDK) | 2.37.1 | plus a **v81 skel** for SM8850 — see [docs/export-guide.md](docs/export-guide.md) §1 |
| Android NDK | r26d | for `build-android` |
| Host RAM | **≥ 128 GiB** | a 4B export peaks at 102 GiB, an 8B at 127 GiB ([docs/peak-memory.md](docs/peak-memory.md)) |
| Disk | ~40 GiB | fp32 checkpoint + `.pte` + extracted context |

The RAM figure is not negotiable and is the single most common way an export dies. It is also why
the patch shares one fp32 weight copy across the three model instances — without that, an 8B
export needs 154 GiB and gets killed.

## Setup

```bash
cp env.sh.example env.sh && $EDITOR env.sh    # four paths
source env.sh
./patch/apply.sh                              # applies to $EXECUTORCH_ROOT
python -c "import executorch.examples.qualcomm.oss_scripts.llama as m; print(m.__file__)"
# must print a path under $EXECUTORCH_ROOT -- if it says site-packages, PYTHONPATH is wrong and
# your newly registered models will either KeyError or silently run stale code
```

## Build a target context

```bash
source env.sh
./scripts/export_target.sh qwen3-4b-sqnr /work/art_q4b 32 1024      # ~43 min, 102 GiB peak
./scripts/extract_context.sh /work/art_q4b /work/ctx_ar32 6
./scripts/make_params.py --hf Qwen/Qwen3-4B --out /work/ctx_ar32/params.json
```

`32` is `prefill_ar_len` — the verify width, fixed at compile time. It is the one export
parameter a speculative-decoding user must think about, and 32 is the measured optimum: it fits a
31-node tree plus its root in a single graph call, while a re-export at 64 lost 4–7% (NPU draft)
and 13–18% (GPU draft) to padding that a wider tree never repaid.

Registered targets: `qwen3-4b`, `qwen3-4b-sqnr` (deployed), `qwen3-4b-w8`, `qwen3-4b-kv16`,
`qwen3-4b-r2`, `qwen3-4b-lc`, `qwen3-8b`. Recipes live in `static_llm_quant_recipe.py`.

## Build a draft context

A draft head cannot be calibrated from prompts — its input is the *target's* hidden states, so
they get replayed from a dump taken offline:

```bash
python $EXECUTORCH_ROOT/examples/qualcomm/oss_scripts/llama/dump_target_hidden.py \
  --target Qwen/Qwen3-4B --calib data/calib_specbench.json --out /work/hidden_4b.pt
./scripts/export_draft.sh qwen3-4b-dflash /work/art_dflash /work/hidden_4b.pt
./scripts/extract_context.sh /work/art_dflash /work/ctx_dflash 1
./scripts/make_params.py --hf Qwen/Qwen3-4B --out /work/ctx_dflash/params.json --draft --n-layers 5
./scripts/dump_draft_embd.py --ckpt dflash4b.pth --out /work/draft4b_embd_f16.bin
```

Before paying for an export, `scripts/sweep_draft_bits.py --ckpt … --hidden …` fake-quantizes the
weights at 4/5/6/8 bits in seconds. It sees weights only, so read it as an upper bound.

Registered drafts: `qwen3-4b-dflash` (deployed, 16a8w), `qwen3-4b-dflash-w4` (w4a16),
`qwen3-4b-dflash-w4a8`, `qwen3-4b-dspark` (DSpark-b7, needed by PCTree/DARTree), `qwen3-8b-dflash`.

The embedding table is deliberately outside the graph: fp16 on the host, mmapped, 8 rows touched
per step. Reusing the target GGUF's Q4_0 table instead costs 5 of 8 top-1 agreements.

## Deploy

```
ctx_ar32/     forward_{0..5}.bin  forward_{0..5}_json.json  params.json     # 13 files
ctx_dflash/   forward_0.bin       forward_0_json.json       params.json
draft4b_embd_f16.bin
```

`params.json` is **not** an export artifact — nothing produces it, and a context missing it loads
and then emits nothing at all. `make_params.py` writes it; `head_dim` must be explicit, since
derived as `dim/n_heads` it is wrong for Qwen3 (4B gives 80, the real value is 128).

Running these is MobiSpec.cpp's side: see `examples/speculative-dflash-qnn-npu/README.md` there.

---

## Reference numbers

Galaxy S26 (Snapdragon 8 Elite Gen 5), Spec-Bench 13 categories, `-n 128`, greedy, chat-templated.
`×AR` is against autoregressive decoding on the same NPU context measured in the same window
(4B: 17.55 t/s). These are cross-checked against a token-identity gate — the speculative output
must equal the AR output token for token.

**Qwen3-4B target (`qwen3-4b-sqnr`, ar_len 32) + NPU draft**

| technique | draft context | AL | draft ms | verify ms | t/s | ×AR |
|---|---|---:|---:|---:|---:|---:|
| PCTree | `qwen3-4b-dspark` | 4.77 | 18.9 | 60.3 | 60.0 | **3.42** |
| DARTree | `qwen3-4b-dspark` | 4.67 | 17.9 | 60.4 | 59.4 | 3.39 |
| TileMenu | `qwen3-4b-dflash` | 4.28 | 17.9 | 56.9 | 57.1 | 3.25 |
| DDTree | `qwen3-4b-dflash` | 4.13 | 18.2 | 60.0 | 52.6 | 3.00 |
| DSpark | `qwen3-4b-dspark` | 3.76 | 16.8 | 55.7 | 51.8 | 2.95 |
| DFlash | `qwen3-4b-dflash` | 3.46 | 16.6 | 55.8 | 47.6 | 2.71 |
| *AR* | — | — | — | — | *17.55* | *1.00* |

**Qwen3-8B target (`qwen3-8b`, ar_len 32) + DFlash-b7 draft**, 4 categories

| category | AR t/s | SD t/s | ×AR | AL | verify ms | draft ms |
|---|---:|---:|---:|---:|---:|---:|
| QA | 11.48 | 36.35 | 3.17 | 5.05 | 95.6 | 21.3 |
| math | 12.00 | 19.84 | 1.65 | 2.91 | 99.5 | 22.1 |
| coding | 12.24 | 19.26 | 1.57 | 2.83 | 100.4 | 22.0 |
| reasoning | 12.59 | 16.30 | 1.29 | 2.23 | 98.6 | 22.2 |

**Export cost and artifact size**

| artifact | recipe | wall | peak RSS | `.pte` | context |
|---|---|---:|---:|---:|---:|
| Qwen3-4B target | `16a4w` + 13 groups at 8b + KV8 | 43 min | 102.0 GiB | 3.6 GB | 3.6 GB |
| Qwen3-8B target | same | 49 min | 126.6 GiB | 5.5 GB | 5.5 GB |
| DFlash-4B draft | `16a8w` per-channel + margin ×2 | ~6 min | — | 1.6 GB | 1.6 GB |
| DSpark-b7 draft | same, embedding-free | ~6 min | — | 894 MB | 894 MB |
| DFlash-4B w4a16 | `16a4w` per-block-16 + margin ×2 | ~6 min | — | 1.4 GB | 1.4 GB |

## Docs

| | |
|---|---|
| [docs/export-guide.md](docs/export-guide.md) | checkpoint → device, every step and what it does |
| [docs/patch-notes.md](docs/patch-notes.md) | what the ExecuTorch patch changes, and what was deliberately left out |
| [docs/quantization.md](docs/quantization.md) | why these recipes; every draft setting tried |
| [docs/peak-memory.md](docs/peak-memory.md) | why an 8B export needs 127 GiB and how it got there from 154 |

## Why these recipes

The full account is [docs/quantization.md](docs/quantization.md). Three things decide most of it:

- **The draft's failure mode is clipping, not rounding.** Replaying its QDQ graph with rounding
  kept but clipping removed scores cos 0.9911; the reverse scores 0.7866. The 0.002% of
  activations that got clipped were the extremes attention runs on. Widening the observer's range
  ×2 took device acceptance from 1.03 to 3.62. A `HistogramObserver` makes it *worse* — MSE
  minimization narrows the range, which is the wrong direction here.
- **The draft's activations cannot be narrowed.** w4a16 is accuracy-neutral (context 1.60 → 1.38
  GB), w4a8 collapses to AL 1.04.
- **`ar_len=32`.** Wider graphs pay padding on every call; a wider tree does not repay it.

## Gotchas that cost real time

1. `params.json` missing → silent no-output. Not an error message anywhere.
2. `head_dim` omitted → wrong for Qwen3, plausible-looking garbage.
3. `PYTHONPATH` not overriding site-packages → stale code runs silently.
4. Mixed QNN `.so` versions on device → `Stub lib id mismatch`.
5. Calibrating a draft on token embeddings instead of target hiddens → observers set from a
   distribution the graph never sees (the target's layer-33 hidden reaches 16384 against a mean
   |x| of 4.5).
6. Exporting on a machine under 128 GiB → OOM kill with no useful traceback.
