#!/bin/bash
# M0: the two 4096-context Qwen3-VL-4B targets (ar_len 32 for AR / none / norm / B-protocol registers, ar_len 40 for the A-protocol register arms),
# calibrated on the 16 document-page fixtures (same pages as the drafter calibration). Sequential; each ~35-60 min, peak RAM measured by /usr/bin/time.
set -o pipefail
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
F=/home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/fixtures
A=/data/youngmin/qnn-artifacts/vlm-register-mobile-20261019
FIX=$(sed "s/^/--fixture /" /tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad/fixtures16.list | tr "
" " ")
cd /home/youngmin/dev/llm/QNN_Exporter
for AR in 32 40; do
  OUT=$A/target-a${AR}-c4096-v2
  [ -e $OUT ] && { echo "SKIP existing $OUT"; continue; }
  echo "$(date '+%H:%M') START ar=$AR"
  /usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
    --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
    --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
    $FIX --artifact $OUT --ar $AR --context 4096 --context-only --build /home/youngmin/dev/llm/executorch/build-android
  echo "$(date '+%H:%M') EXPORT_EXIT ar=$AR $?"
  /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/extract_vlm_context.py $OUT $A/ctx_vlm_a${AR}_c4096 --compiled-contexts
  echo "$(date '+%H:%M') EXTRACT_EXIT ar=$AR $?"
done
echo TARGETS_C4096_DONE
