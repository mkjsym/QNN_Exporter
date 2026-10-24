#!/usr/bin/env bash
# Build the NPU draft context for a DFlash- or DSpark-style draft head.
#
#   ./export_draft.sh <decoder_model> <artifact_dir> <target_hidden.pt>
#
# A draft head cannot be calibrated from token prompts: its input is the target's hidden states,
# so those are replayed from a dump taken offline against the bf16 target (dump_target_hidden.py,
# shipped inside the patch). Feed it token embeddings instead and `fc` learns a range it will
# never see -- on Qwen3-4B the target's layer-33 hidden reaches 16384 while its mean |x| is 4.5.
#
# Serves both draft families. They differ in three inputs and nothing else here:
#
#   DFlash   BLOCK=8  ./export_draft.sh qwen3-4b-dflash-w4 out hidden_5layer.pt
#   EAGLE-3  BLOCK=32 CKPT=eagle3_4b.pth \
#                     ./export_draft.sh qwen3-4b-eagle3-w4 out hidden_3layer.pt
#
# CKPT is required for EAGLE-3 and not for DFlash. An EAGLE-3 draft owns no embedding table -- it
# borrows the target's -- so its checkpoint has to be converted with --target-dir to carry one in,
# and the pipeline's own download path cannot pass that. Convert first, then hand the .pth over:
#
#   python examples/models/qwen3_eagle3/convert_weights.py <draft> eagle3_4b.pth --target-dir <target>
#
# BLOCK is the graph's row count. For DFlash it is the proposal block (8); for EAGLE-3 it is the
# widest call the driver makes -- the recompute over accepted tokens, or a tree depth's frontier --
# so 32. Padding is close to free on this backend: cost is weight streaming, and the rows ride along.
set -euo pipefail
MODEL=${1:?decoder_model, e.g. qwen3-4b-dflash}
ART=${2:?artifact dir}
HID=${3:?target hidden dump (.pt)}
: "${EXECUTORCH_ROOT:?set EXECUTORCH_ROOT}"
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT}"
HERE=$(cd "$(dirname "$0")" && pwd)
CALIB=${CALIB:-$HERE/../data/calib_specbench.json}

cd "$EXECUTORCH_ROOT"
python examples/qualcomm/oss_scripts/llama/llama.py \
  --decoder_model "$MODEL" --model_mode hybrid \
  --prefill_ar_len "${BLOCK:-8}" --max_context_len "${CTX:-1024}" \
  --soc_model "${SOC_MODEL:-SM8850}" --build_folder "${ET_BUILD:-build-android}" \
  --artifact "$ART" --compile_only \
  --calib_samples "$CALIB" --dflash_hidden "$HID" \
  ${CKPT:+--checkpoint "$CKPT"} \
  --prompt "Once upon a time"
echo "draft .pte: $(ls -lh "$ART"/*.pte | awk '{print $5}')"
