#!/usr/bin/env python3
"""Apply the Qwen3-VL extension after the pinned base export patch."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("executorch", type=Path)
parser.add_argument("--check", action="store_true")
parser.add_argument("--reverse", action="store_true")
args = parser.parse_args()
patch = Path(__file__).with_name("qwen3-vl.patch")
base = ["git", "-C", str(args.executorch), "apply"]
flags = ["--reverse"] if args.reverse else []
check = subprocess.run([*base, *flags, "--check", str(patch)], capture_output=True, text=True)
if check.returncode:
    opposite = [] if args.reverse else ["--reverse"]
    if subprocess.run([*base, *opposite, "--check", str(patch)], capture_output=True).returncode == 0:
        print("Qwen3-VL extension already in the requested state")
    else:
        raise SystemExit("Apply patch/apply.sh first; the Qwen3-VL extension did not apply cleanly:\n" + check.stderr)
elif args.check:
    print("Qwen3-VL extension applies cleanly")
else:
    subprocess.run([*base, *flags, str(patch)], check=True)
    print("Qwen3-VL extension applied" if not args.reverse else "Qwen3-VL extension reverted")
