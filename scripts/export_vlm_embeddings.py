#!/usr/bin/env python3
"""Write the target's F16 embedding table without materializing FP32 model weights."""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = gguf.GGUFReader(str(args.gguf))
    if reader.fields["general.architecture"].contents() != "qwen3vl":
        raise ValueError("expected Qwen3-VL target GGUF")
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    if list(tensor.shape) != [2560, 151936]:
        raise ValueError("expected Qwen3-VL-4B embedding shape")
    if tensor.tensor_type not in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        raise ValueError("use an unquantized target GGUF")
    digest = hashlib.sha256()
    with args.output.open("xb") as file:
        for row in range(0, 151936, 512):
            chunk = gguf.quants.dequantize(tensor.data[row:row + 512], tensor.tensor_type).astype("<f2")
            if not np.isfinite(chunk).all():
                raise ValueError("embedding conversion overflowed")
            data = chunk.tobytes()
            digest.update(data)
            file.write(data)
    metadata = {"source": str(args.gguf.resolve()), "tensor": tensor.name,
                "source_type": tensor.tensor_type.name, "shape": [151936, 2560], "dtype": "float16-le",
                "bytes": args.output.stat().st_size, "sha256": digest.hexdigest()}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
