# Qwen3-VL-4B target export for DFlash and EAGLE-3

The target is Qwen3-VL-4B-Instruct. The draft remains the existing Qwen3-4B DFlash checkpoint,
including its own embedding table, output head, mask token and one-dimensional positions.
Runtime code belongs to MobiSpec.cpp; this repository owns export scripts and the ExecuTorch patch.
EAGLE-3 uses the existing Qwen3-4B head with a separate target export: pass `--method eagle3`
to `export_vlm_target.py` for four shards at `[2,18,33]`. DFlash remains the default, with six shards.

`--kv-bits 16` keeps the existing SQNR weight/activation recipe and removes only its KV8
annotation. The default remains KV8. Actual context IO validation checks the requested precision.
This controlled comparison was added after a VLM A1/A32 trace first differed by one KV8 unit at
position 130 in shard 3, despite identical live inputs and preceding output histories.
Both KV16 targets compiled and passed actual IO checks. The DFlash device control did not
resolve A1/A32 agreement: all four short SD cells differed from A1, and differences appeared
inside the first native call with identical live inputs. These cells are excluded from performance
results. The benchmark uses KV8 and a fixed A32 target for both sequential AR and verification;
A1 throughput is measured separately. Evidence belongs to MobiSpec.cpp
`docs/results/vlm-kv16-control.json` and `docs/results/vlm-kv8-precision/`.

`--context-only` retains all validated native context binaries and skips the duplicate PTE
container. Use `extract_vlm_context.py --compiled-contexts` for this mode. The normal mode still
packages a PTE. Container hashes are recorded before removing any redundant experiment copies.

## Official support and implementation choice

Qualcomm's [QIDK release notes](https://qualcomm.github.io/qidk/release-notes/solutions.html)
list Qwen3-VL-4B-Instruct w4a16 on Snapdragon 8 Elite Gen 5, Android 16 and QAIRT 2.45.0.
The [AI Hub model](https://aihub.qualcomm.com/models/qwen3_vl_4b_instruct) provides the
`geniex_qairt-w4a16` deployment path. This includes vision, embedding lookup and the text decoder.

The [GenieX public interface](https://github.com/qualcomm/GenieX/blob/main/sdk/include/geniex.h)
says its speculative options apply to `llama_cpp` and are ignored by `qairt`; it lists no
DFlash API. Our DFlash runner also needs intermediate layer outputs, batched verification and KV
rollback. Therefore this extension exports explicit QNN decoder graphs using our existing QAIRT
2.37.1/v81 toolchain. Compatibility with the official 2.45 bundle is a separate matter.

Initial scope: CPU mtmd vision encoder; NPU target decoder; existing NPU or Adreno draft.
N/N and N/G describe the two decoders. One still image and greedy generation are supported.
This is an experimental transfer of a text-trained draft; acceptance and speed must be measured.

## Build

Apply the base patch once, then the VLM extension:

```bash
./patch/apply.sh "$EXECUTORCH_ROOT"
python patch/apply_vlm.py "$EXECUTORCH_ROOT"
python scripts/download_vlm_tokenizer.py /work/qwen3vl-tokenizer
python scripts/export_vlm_embeddings.py /models/Qwen3-VL-4B-Instruct-BF16.gguf /work/target_embeddings.f16
```

Use MobiSpec.cpp's `llama-qnn-vlm-prepare` to encode a real image and dump calibration inputs:

```bash
QNN_VLM_EMBD=/work/target_embeddings.f16 QNN_VLM_DUMP=/work/image64 \
  /path/to/MobiSpec.cpp/build/bin/llama-qnn-vlm-prepare \
  -m /models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --mmproj /models/mmproj-Qwen3VL-4B-F16.gguf --image /images/example.jpg \
  --image-max-tokens 64 -p 'Describe this image in one sentence.' -t 8

python scripts/export_vlm_target.py \
  --checkpoint /models/Qwen3-VL-4B-Instruct-BF16.gguf --tokenizer /work/qwen3vl-tokenizer \
  --fixture /work/image64.json --artifact /work/vlm-a32-c1024 \
  --ar 32 --context 1024 --build "$EXECUTORCH_ROOT/build-android"
python scripts/extract_vlm_context.py /work/vlm-a32-c1024 /work/ctx_vlm_a32
```

Multiple `--fixture` arguments calibrate multiple images. One image is for bringing up the path,
not evidence of quantization quality across workloads. No new draft training/export is required.
`--calib-registers table.bin` (f32 `[8, 2560]`, the drafter arm's register table) appends the register rows to every
calibration fixture: with `--method dflash-deep` they carry one-hot prompt slots, with the plain `dflash` method they are
plain side rows, so a target that will serve dynamic-register verification calls is calibrated on them (the first plain
export was not and the dynreg8 arm lost about 2.6 percent AL on the device relative to the host; the re-export of
2026-10-22 is `ctx_vlm_a32_c4096_r1`).
The checkpoint loader accepts BF16/F16/F32 GGUF weights and loads FP32 directly into RAM; it
rejects quantized checkpoints. It avoids writing a second full FP32 checkpoint to disk.

The driver checks ExecuTorch's installed `program.fbs` / `scalar_type.fbs` resources and runs a
small PTE serialization round trip before loading weights. A source-only clone needs the same
package resources as an installed checkout. Each compiled shard is saved immediately under
`compiled_contexts/`; choose a fresh artifact directory for every export. If final PTE packaging
fails after all six contexts are saved, extraction remains possible:

```bash
python scripts/extract_vlm_context.py /work/vlm-a32-c1024 /work/ctx_vlm_a32 --compiled-contexts
```

This recovery path validates graph names and VLM inputs with the SDK utility exactly as the PTE
path does. Partially written files use `.tmp` names and are never accepted as complete contexts.
Raw-context extraction uses hard links on the same filesystem to avoid another full copy of weights.
Each shard is also checked immediately after compilation: attention-mask dimensions must match the
requested context, KV inputs/outputs must match the selected uint8/uint16 precision and shard
boundaries must be uint16. The runtime does not
support float KV caches mixed with quantized KV outputs.

The local recipe is `Qwen3_4B_SqnrQuantRecipe`, a mixed 4/8-bit weight recipe with 16-bit
activations, masked softmax and R3 enabled. Most projections use block-16 4-bit weights;
selected sensitive projections and the output head use per-channel 8-bit weights. This is not
the official uniform w4a16 package. KV16 changes only the KV annotation, not those weight rules.

## Graph contract

| Input | Shape | Meaning |
|---|---|---|
| `vlm_embd` | `[1,A,2560]` | Target text embeddings or projected image embeddings |
| `atten_mask` | `[1,1,A,C]` | Causal/ancestor visibility in sequence order |
| `vlm_rope_cos`, `vlm_rope_sin` | `[A,64]` each | Interleaved MRoPE, theta 5e6, sections 24/20/20 |
| `vlm_deepstack_0..2` | `[1,A,2560]` each | Add after layers 0..2; zero at text/padded rows |
| `args_*` | existing packed KV layout | Persistent cache; no rotary coordinates in slot metadata |

The six shards split at layers `[2,10,18,26,34]`. The first capture includes the layer-1 DeepStack
addition. The extension keeps that addition inside the decoder-layer module so the splitting pass
places its boundary correctly. Calibration enters through `predict_step`, replays every fixture
chunk, and writes real KV between chunks. Text-only calibration is rejected.

## Validation

```bash
VLM_TEST_ARTIFACTS=/work/fixtures python tests/test_qwen3_vl.py
```

The fixture directory must contain `image64.json` and its binary companions. Tests compare against
Hugging Face Qwen3-VL, check chunked KV, verify the DeepStack capture boundary and exercise PTQ's
actual entry point. See MobiSpec.cpp's `docs/vlm-dflash-qwen3vl.md` for runtime commands and device status.
The cache-capacity regression deliberately uses unequal `max_seq_len` and `max_context_len`.
The embedding-free upstream model's example-input helper used the former while IO tagging used
the latter; the VLM helper now constructs caches using `max_context_len` and the driver sets both
CLI lengths explicitly. Compile success alone did not detect that incompatible IO contract.

## Chat-template control for LLM/VLM comparisons

The native Qwen3-4B no-thinking template adds an empty thinking block before generation;
Qwen3-VL-4B-Instruct uses a different assistant prefix. The speedup investigation uses a
metadata-only tokenizer to measure that prefix change with fixed QNN target and draft contexts:

```bash
PYTHONPATH=/path/to/MobiSpec.cpp/gguf-py python scripts/make_template_control_tokenizer.py \
  --source /models/Qwen3-VL-4B-Instruct-Q4_0.gguf \
  --template-source /models/Qwen3-4B-q4_0-outq4.gguf \
  --output /work/qwen3vl-llm-template-tokenizer.gguf
```

The helper copies the VLM metadata, replaces `tokenizer.chat_template`, and checks that this is
the only metadata change. It writes zero weight tensors and records source/output SHA-256 hashes
in a companion JSON file. Pass the result as the QNN runtime's `--tokenizer`, with the existing
VLM context binaries and target embedding table. MobiSpec.cpp's prepare tool should verify the
actual rendered token IDs for each evaluation input before device measurements.
