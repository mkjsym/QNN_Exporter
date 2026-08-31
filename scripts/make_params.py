#!/usr/bin/env python3
"""Write the params.json the runtime reads next to a context directory.

params.json is NOT an export artifact -- nothing produces it, and without it the runner loads
and then emits nothing at all. head_dim in particular must be written explicitly: derived as
dim/n_heads it is wrong for Qwen3 (4B gives 2560/32 = 80, the real value is 128).

  ./make_params.py --hf Qwen/Qwen3-4B --out ctx_ar32/params.json
  ./make_params.py --hf Qwen/Qwen3-4B --out ctx_dflash/params.json --draft --n-layers 5
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF repo id or local dir holding config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft", action="store_true", help="add the DFlash draft fields")
    ap.add_argument("--n-layers", type=int, default=0, help="override (a draft has its own depth)")
    ap.add_argument("--n-inject", type=int, default=16)
    ap.add_argument("--n-target-layers", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--mask-token-id", type=int, default=151669)
    a = ap.parse_args()

    path = os.path.join(a.hf, "config.json")
    if os.path.exists(path):
        cfg = json.load(open(path))
    else:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(a.hf, "config.json")))

    dim = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    p = {
        "dim": dim,
        "n_layers": a.n_layers or cfg["num_hidden_layers"],
        "n_heads": n_heads,
        "n_kv_heads": cfg.get("num_key_value_heads", n_heads),
        "head_dim": cfg.get("head_dim") or dim // n_heads,
        "vocab_size": cfg["vocab_size"],
        "ffn_dim_multiplier": 1,
        "multiple_of": 256,
        "norm_eps": cfg.get("rms_norm_eps", 1e-6),
        "rope_theta": cfg.get("rope_theta", 1000000.0),
        "use_scaled_rope": False,
    }
    if a.draft:
        p.update({
            "n_inject": a.n_inject,
            "n_target_layers": a.n_target_layers,
            "block_size": a.block_size,
            "mask_token_id": a.mask_token_id,
        })
    if cfg.get("head_dim") is None:
        print(f"warning: config.json has no head_dim; derived {p['head_dim']} from dim/n_heads. "
              "Check it against the checkpoint -- Qwen3 sets it independently.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(p, open(a.out, "w"), indent=1)
    print(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
