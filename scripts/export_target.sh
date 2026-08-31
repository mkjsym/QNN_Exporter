#!/usr/bin/env bash
# Build the NPU target context for a Qwen3 model.
#
#   ./export_target.sh <decoder_model> <artifact_dir> [prefill_ar_len] [max_context_len]
#
# prefill_ar_len is the verify width, fixed at compile time. 32 fits a 31-node tree plus its root
# and is what the reference numbers use; 64 was measured and lost -- the padding a half-filled
# graph carries costs 4-7% on an NPU draft and 13-18% on a GPU one, which a wider tree never repays.
set -euo pipefail
MODEL=${1:?decoder_model, e.g. qwen3-4b-sqnr}
ART=${2:?artifact dir}
AR=${3:-32}
CTX=${4:-1024}
: "${EXECUTORCH_ROOT:?set EXECUTORCH_ROOT to your patched ExecuTorch checkout}"
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the QAIRT install}"
HERE=$(cd "$(dirname "$0")" && pwd)
CALIB=${CALIB:-$HERE/../data/calib_specbench.json}

cd "$EXECUTORCH_ROOT"
python examples/qualcomm/oss_scripts/llama/llama.py \
  --decoder_model "$MODEL" --model_mode hybrid \
  --prefill_ar_len "$AR" --max_context_len "$CTX" \
  --soc_model "${SOC_MODEL:-SM8850}" --build_folder "${ET_BUILD:-build-android}" \
  --artifact "$ART" --compile_only \
  --calib_samples "$CALIB" \
  --prompt "Once upon a time"
echo "target .pte: $(ls -lh "$ART"/*.pte | awk '{print $5}')"
