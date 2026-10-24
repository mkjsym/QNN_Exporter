#!/usr/bin/env python3
"""Download only Qwen3-VL tokenizer/config files, never model weights."""
import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("output")
args = parser.parse_args()
snapshot_download("Qwen/Qwen3-VL-4B-Instruct", local_dir=args.output,
    allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "special_tokens_map.json", "chat_template.jinja"])
