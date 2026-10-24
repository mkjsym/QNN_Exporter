#!/usr/bin/env python3
"""Extract each multifunction QNN context once, preserving existing output directories."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess

from executorch.backends.qualcomm.python import PyQnnManagerAdaptor
from executorch.exir._serialize._program import deserialize_pte_binary
from validate_vlm_context import validate

parser = argparse.ArgumentParser()
parser.add_argument("artifact", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--compiled-contexts", action="store_true",
    help="use incrementally saved raw contexts instead of the packaged PTE")
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("output already exists; choose a new directory")
params = json.loads((args.artifact / "params.json").read_text())
if params.get("vlm_input_version") != 1:
    raise SystemExit("missing explicit VLM input metadata")
n_shards = len(params.get("shard_layers", [2,10,18,26,34])) + 1
if args.compiled_contexts:
    contexts = args.artifact / "compiled_contexts"
    if {p.name for p in contexts.glob("*.bin")} != {f"forward_{i}.bin" for i in range(n_shards)} or list(contexts.glob("*.tmp")):
        raise SystemExit(f"expected {n_shards} complete compiled contexts; export may still be running")
    delegates = [None] * n_shards
else:
    program = deserialize_pte_binary((args.artifact / "hybrid_llama_qnn.pte").read_bytes()).program
    plans = [p for p in program.execution_plan if p.name == "kv_forward"]
    if len(plans) != 1:
        raise SystemExit("expected a kv_forward execution plan")
    delegates = plans[0].delegates
    if len(delegates) != n_shards or any(d.id != "QnnBackend" for d in delegates):
        raise SystemExit(f"expected {n_shards} QNN target shards")
args.output.mkdir(parents=True)
sdk = Path(os.environ["QNN_SDK_ROOT"])
manifest = {}
graphs = {"kv_forward": [], "prefill_forward": []}
for i, delegate in enumerate(delegates):
    if args.compiled_contexts:
        binary = (contexts / f"forward_{i}.bin").read_bytes()
    else:
        data = program.backend_delegate_data[delegate.processed.index].data
        binary = PyQnnManagerAdaptor.StripProtocol(data)
    if len(binary) == 0:
        raise RuntimeError("expected a compiled QNN context, got a DLC")
    path = args.output / f"forward_{i}.bin"
    try:
        if not args.compiled_contexts:
            raise OSError(errno.EXDEV, "PTE extraction requires a write")
        os.link(contexts / path.name, path)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        with path.open("xb") as file:
            file.write(binary)
    manifest[path.name] = hashlib.sha256(binary).hexdigest()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(sdk / "lib/x86_64-linux-clang")
    subprocess.run([str(sdk / "bin/x86_64-linux-clang/qnn-context-binary-utility"),
        "--context_binary", str(path), "--json_file", str(args.output / f"forward_{i}_json.json")],
        env=env, check=True)
    raw = json.loads((args.output / f"forward_{i}_json.json").read_text())
    validate(raw, params["context_length"], params["prefill_ar_len"], params.get("kv_cache_bits",8))
    shard_graphs = {g["info"]["graphName"]: g["info"] for g in raw["info"]["graphs"]}
    for method in graphs:
        if method not in shard_graphs:
            raise RuntimeError(f"shard {i} is missing {method}")
        inputs = [t["info"] for t in shard_graphs[method]["graphInputs"]]
        graphs[method].append({"shard": i, "inputs": [{"name": t["name"],
            "dims": t.get("dimensions", t.get("currentDimensions")), "dtype": t["dataType"]}
            for t in inputs if "vlm_" in t["name"]]})
for method, shards in graphs.items():
    for role in ("vlm_embd", "vlm_rope_cos", "vlm_rope_sin", "vlm_deepstack_0", "vlm_deepstack_1", "vlm_deepstack_2"):
        if not any(role in t["name"] for s in shards for t in s["inputs"]):
            raise RuntimeError(f"{method} is missing explicit input {role}")
(args.output / "params.json").write_text(json.dumps(params, indent=2) + "\n")
(args.output / "io_contract.json").write_text(json.dumps(graphs, indent=2) + "\n")
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Extracted {n_shards} target shards to {args.output}")
