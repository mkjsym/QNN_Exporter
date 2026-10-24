#!/usr/bin/env python3
"""Compile the explicit Qwen3-VL target; draft artifacts remain unchanged."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ar", type=int, default=32)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--method", choices=("dflash", "eagle3", "dflash-deep"), default="dflash")
    parser.add_argument("--layer-prompts", type=Path, default=None, help="dflash-deep: f32 [33, 8, dim] per-layer register prompts baked into the graph")
    parser.add_argument("--calib-registers", type=Path, default=None, help="f32 [8, dim] register table appended to every calibration fixture: dflash-deep with one-hot prompt slots; dflash (plain) without slots, so the plain target's PTQ encodings see the dynamic-register side rows it serves on the device (M6, 2026-10-22)")
    parser.add_argument("--kv-bits", type=int, choices=(8,16), default=8)
    parser.add_argument("--calib-ar", type=int, default=None, help="rows per PTQ calibration step (default: --ar). Calibrating a 40-row target at 32 gives it the encodings of the 32-row target built from the same fixtures (shared encodings across verification widths)")
    parser.add_argument("--context-only", action="store_true", help="save native contexts without a duplicate PTE container")
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    et = Path(os.environ["EXECUTORCH_ROOT"]).resolve()
    sdk = Path(os.environ["QNN_SDK_ROOT"]).resolve()
    if not 1 < args.ar < args.context:
        parser.error("require 1 < ar < context")
    if args.calib_ar is not None and not 1 < args.calib_ar < args.context:
        parser.error("require 1 < calib-ar < context")
    if not args.checkpoint.is_file() or not args.tokenizer.is_dir():
        parser.error("checkpoint/tokenizer path does not exist")
    for name in ("program.fbs", "scalar_type.fbs"):
        if not (et / "exir/_serialize" / name).is_file():
            raise RuntimeError(f"ExecuTorch package resource {name} is missing; prepare/install this ExecuTorch checkout before export")
    if (args.method == "dflash-deep") != (args.layer_prompts is not None):
        parser.error("--layer-prompts goes with --method dflash-deep")
    if args.layer_prompts is not None:
        if not args.layer_prompts.is_file() or args.layer_prompts.stat().st_size != 33 * 8 * 2560 * 4:
            parser.error("--layer-prompts must be a [33, 8, 2560] float32 file")
        os.environ["QNN_VLM_LAYER_PROMPTS"] = str(args.layer_prompts.resolve())
        if args.calib_registers is None or not args.calib_registers.is_file() or args.calib_registers.stat().st_size != 8 * 2560 * 4:
            parser.error("--calib-registers must be the arm's [8, 2560] float32 register table (deep calibration needs real register rows)")
        os.environ["QNN_VLM_CALIB_REGISTERS"] = str(args.calib_registers.resolve())
    elif args.calib_registers is not None:
        # plain dflash target calibrated with the dynamic-register rows of a dynreg arm (M6 re-calibration, 2026-10-22)
        if not args.calib_registers.is_file() or args.calib_registers.stat().st_size != 8 * 2560 * 4:
            parser.error("--calib-registers must be the arm's [8, 2560] float32 register table")
        os.environ["QNN_VLM_CALIB_REGISTERS"] = str(args.calib_registers.resolve())
    args.artifact.mkdir(parents=True, exist_ok=True)
    required_gib = 4 if args.context_only else 10
    if shutil.disk_usage(args.artifact).free < required_gib * 2**30:
        raise RuntimeError(f"need at least {required_gib} GiB free for this export mode")
    for fixture in args.fixture:
        meta = json.loads(fixture.read_text())
        if meta.get("image_rows", 0) <= 0 or meta["rows"] > args.context - args.ar:
            raise ValueError(f"incompatible image fixture: {fixture}")
    # The PTQ loop uses these only to schedule calls; actual inputs are the image fixtures.
    samples = args.artifact / "calibration_schedule.json"
    samples.write_text(json.dumps([{"messages": [{"role": "user", "content": "Describe this image."}]} for _ in args.fixture]))
    command = [sys.executable, "-m", "executorch.examples.qualcomm.oss_scripts.llama.llama",
        "--decoder_model", "qwen3-vl-4b-" + args.method, "--model_mode", "hybrid",
        "--prefill_ar_len", str(args.ar), "--max_context_len", str(args.context),
        "--max_seq_len", str(args.context),
        "--soc_model", "SM8850", "--build_folder", str(args.build.resolve()),
        "--artifact", str(args.artifact.resolve()), "--compile_only",
        "--checkpoint", str(args.checkpoint.resolve()), "--tokenizer_model", str(args.tokenizer.resolve()),
        "--vlm_calibration", *[str(p.resolve()) for p in args.fixture],
        "--calib_samples", str(samples.resolve()), "--calib_limit", str(len(args.fixture)),
        "--prompt", "Describe this image."]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(et.parent) + os.pathsep + str(et),
        LD_LIBRARY_PATH=str(sdk / "lib/x86_64-linux-clang"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    # Exercise the exact serializer before loading the 4B checkpoint.
    subprocess.run([sys.executable, "-c", "import torch; from executorch.exir import to_edge; "
        "from executorch.exir._serialize._program import deserialize_pte_binary; "
        "m=torch.nn.ReLU(); p=to_edge(torch.export.export(m, (torch.ones(1),))).to_executorch(); "
        "assert deserialize_pte_binary(p.buffer).program.execution_plan; "
        "print('PTE serialization preflight passed', flush=True)"], env=env, check=True)
    contexts = args.artifact.resolve() / "compiled_contexts"
    contexts.mkdir()  # Never mix contexts from different exports.
    env["QNN_VLM_CONTEXT_DUMP_DIR"] = str(contexts)
    env["QNN_VLM_CONTEXT_VALIDATOR"] = str(Path(__file__).with_name("validate_vlm_context.py").resolve())
    env["QNN_VLM_CONTEXT_LENGTH"] = str(args.context)
    env["QNN_VLM_PREFILL_AR"] = str(args.ar)
    env["QNN_VLM_KV_BITS"] = str(args.kv_bits)
    if args.calib_ar is not None:
        env["QNN_VLM_CALIB_AR"] = str(args.calib_ar)   # read by the patched base_component.py (patch/qwen3-vl.patch)
    env["QNN_VLM_CONTEXT_ONLY"] = "1" if args.context_only else "0"
    (args.artifact / "command.json").write_text(json.dumps({"argv": command,
        "executorch": str(et), "sdk": str(sdk), "vision_backend": "CPU",
        "draft": "unchanged Qwen3-4B " + args.method, "calibration_fixtures": len(args.fixture),
        "layer_prompts": str(args.layer_prompts.resolve()) if args.layer_prompts else None,
        "calib_registers": str(args.calib_registers.resolve()) if args.calib_registers else None,
        "kv_bits": args.kv_bits, "context_only": args.context_only, "calib_ar": args.calib_ar or args.ar}, indent=2) + "\n")
    params = json.loads((et / "examples/models/qwen3/config/vl_4b_config.json").read_text())
    params.update(vlm_input_version=1, n_deepstack=3, mrope_sections=[24,20,20])
    params.update(speculative_method=args.method,
        shard_layers=[2,10,18,26,34] if args.method.startswith("dflash") else [2,18,33])   # dflash and dflash-deep share the six-shard layout
    params.update(context_length=args.context, prefill_ar_len=args.ar)
    params.update(kv_cache_bits=args.kv_bits)
    (args.artifact / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
