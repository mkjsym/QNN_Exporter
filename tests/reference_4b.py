#!/usr/bin/env python3
"""Greedy FP32 decoder reference using the exact GGUF weights and mtmd image fixture."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from executorch.examples.qualcomm.oss_scripts.llama.model.qwen3_vl import load_qwen3_vl_gguf
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tokens", type=int, default=32)
args = parser.parse_args()
torch.set_num_threads(8)
meta = json.loads(args.fixture.read_text())
config = Qwen3VLTextConfig(**json.loads((args.tokenizer / "config.json").read_text())["text_config"])
config._attn_implementation = "eager"
if not config.tie_word_embeddings:
    raise ValueError("this reference expects the tied Qwen3-VL-4B LM head")
weights = load_qwen3_vl_gguf(args.checkpoint)
mapping = {"tok_embeddings": "embed_tokens", "attention.wq": "self_attn.q_proj",
    "attention.wk": "self_attn.k_proj", "attention.wv": "self_attn.v_proj",
    "attention.wo": "self_attn.o_proj", "attention.q_norm_fn": "self_attn.q_norm",
    "attention.k_norm_fn": "self_attn.k_norm", "attention_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm", "feed_forward.w1": "mlp.gate_proj",
    "feed_forward.w2": "mlp.down_proj", "feed_forward.w3": "mlp.up_proj"}
hf_weights = {}
for key, value in weights.items():
    if key == "output.weight":
        continue
    for old, new in mapping.items():
        key = key.replace(old, new)
    hf_weights[key] = value
with torch.device("meta"):
    model = Qwen3VLTextModel(config)
model.load_state_dict(hf_weights, strict=True, assign=True)
model.rotary_emb = Qwen3VLTextRotaryEmbedding(config, device="cpu")
model.eval()
prefix = args.fixture.with_suffix("")
def read(role):
    return torch.from_numpy(np.fromfile(str(prefix) + f".{role}.f32", dtype="<f4").reshape(1,meta["rows"],2560))
embeds = read("vlm_embd")
ds = [read(f"vlm_deepstack_{i}") for i in range(3)]
visual = ds[0].ne(0).any(-1)
if visual.sum() != meta["image_rows"]:
    raise ValueError("fixture visual row count mismatch")
pos = torch.from_numpy(np.fromfile(str(prefix) + ".positions.i32", dtype="<i4").reshape(-1,3)).T[:,None,:]
pos4 = torch.cat((torch.arange(meta["rows"]).view(1,1,-1), pos))
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
tokens = []
with torch.no_grad():
    out = model(inputs_embeds=embeds, position_ids=pos4, visual_pos_masks=visual,
        deepstack_visual_embeds=[d[visual] for d in ds], use_cache=True)
    for i in range(args.tokens):
        logits = out.last_hidden_state[0,-1] @ weights["output.weight"].T
        if i == 0:
            np.save(str(args.output) + ".first_logits.npy", logits.numpy())
        token = logits.argmax().item()
        tokens.append(token)
        if token == tokenizer.eos_token_id or i + 1 == args.tokens:
            break
        seq = meta["rows"] + i
        p = seq + meta["rope_delta"]
        pos4 = torch.tensor([seq,p,p,p]).view(4,1,1)
        out = model(inputs_embeds=weights["tok_embeddings.weight"][token].view(1,1,-1),
            position_ids=pos4, past_key_values=out.past_key_values, use_cache=True)
result = {"reference": "HF FP32 text decoder + supplied mtmd vision fixture", "tokens": tokens,
          "text": tokenizer.decode(tokens), "fixture": meta}
args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(result, ensure_ascii=False), flush=True)
