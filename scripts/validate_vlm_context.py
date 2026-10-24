#!/usr/bin/env python3
"""Reject QNN graphs whose actual IO cannot be bound by the MobiSpec VLM runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def validate(raw, context, ar, kv_bits=8):
    if kv_bits not in (8,16):
        raise ValueError("KV precision must be 8 or 16 bits")
    kv_dtype = f"QNN_DATATYPE_UFIXED_POINT_{kv_bits}"
    graphs = {g["info"]["graphName"]: g["info"] for g in raw["info"]["graphs"]}
    if set(graphs) != {"kv_forward", "prefill_forward"}:
        raise ValueError("expected kv_forward and prefill_forward")
    for name, a in (("kv_forward",1),("prefill_forward",ar)):
        graph = graphs[name]
        masks = []
        for tensor in graph["graphInputs"]:
            t = tensor["info"]
            n, dims, dtype = t["name"], t["dimensions"], t["dataType"]
            if "atten_mask" in n:
                masks.append(t)
                if dims != [1,1,a,context] or dtype != "QNN_DATATYPE_UFIXED_POINT_16":
                    raise ValueError(f"{name}: incompatible attention mask {dims}, {dtype}")
            elif "_args_" in n:
                if dims not in ([1,8,128,context-a],[1,8,context-a,128]) or dtype != kv_dtype:
                    raise ValueError(f"{name}: incompatible KV input {n}: {dims}, {dtype}")
            elif "fallback" in n:
                if dims != [1,a,2560] or dtype != "QNN_DATATYPE_UFIXED_POINT_16":
                    raise ValueError(f"{name}: incompatible shard boundary {n}")
            elif "vlm_" in n:
                expected = [a,64] if "rope_" in n else [1,a,8] if "prompt_slot" in n else [1,a,2560]   # prompt_slot: deep-register variant, one-hot register index per row
                if dims != expected or dtype not in ("QNN_DATATYPE_FLOAT_32", "QNN_DATATYPE_UFIXED_POINT_16", "QNN_DATATYPE_UFIXED_POINT_8"):
                    raise ValueError(f"{name}: incompatible VLM input {n}: {dims}, {dtype}")
            else:
                raise ValueError(f"{name}: unrecognized target input {n}")
        if len(masks) != 1:
            raise ValueError(f"{name}: missing/duplicate attention mask")
        for tensor in graph["graphOutputs"]:
            t=tensor["info"]
            dims=t["dimensions"]
            if len(dims) == 4 and (dims[-1] == 128 or dims[-2] == 128):
                if dims not in ([1,8,128,a],[1,8,a,128]) or t["dataType"] != kv_dtype:
                    raise ValueError(f"{name}: incompatible KV output {t['name']}")
    return graphs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("binary",type=Path)
    parser.add_argument("--context",type=int,default=int(os.environ.get("QNN_VLM_CONTEXT_LENGTH","1024")))
    parser.add_argument("--ar",type=int,default=int(os.environ.get("QNN_VLM_PREFILL_AR","32")))
    parser.add_argument("--kv-bits",type=int,choices=(8,16),default=int(os.environ.get("QNN_VLM_KV_BITS","8")))
    args=parser.parse_args()
    sdk=Path(os.environ["QNN_SDK_ROOT"])
    destination=args.binary.with_name(args.binary.stem + "_json.json")
    env=os.environ.copy();env["LD_LIBRARY_PATH"]=str(sdk/"lib/x86_64-linux-clang")
    subprocess.run([str(sdk/"bin/x86_64-linux-clang/qnn-context-binary-utility"),
        "--context_binary",str(args.binary),"--json_file",str(destination)],env=env,check=True)
    validate(json.loads(destination.read_text()),args.context,args.ar,args.kv_bits)
    print(f"Validated QNN IO: {args.binary.name}, C={args.context}, A={args.ar}, KV={args.kv_bits}",flush=True)


if __name__ == "__main__":
    main()
