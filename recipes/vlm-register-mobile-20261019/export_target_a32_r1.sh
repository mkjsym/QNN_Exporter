#!/bin/bash
# M6 (plan section 13, user approval 2026-10-22): plain Qwen3-VL-4B a32 target re-exported with the dynreg8-20k-seed42 register rows appended to every
# PTQ calibration fixture (as the deep v2 target was on 2026-10-19). Context 4096, ar_len 32, same 16 fixtures. Outputs on the root disk (/data is full).
set -o pipefail
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
S=/tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad
A=/home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019
FIX=$(sed "s/^/--fixture /" $S/fixtures16.list | tr "\n" " ")
cd /home/youngmin/dev/llm/QNN_Exporter
OUT=$A/target-a32-c4096-r1
[ -e $OUT ] && { echo "SKIP existing $OUT"; exit 1; }
echo "$(date '+%H:%M') START a32 r1 (plain target, calibration fixtures + dynreg8 register rows), THP disabled"
if ! /usr/bin/python3 $S/nothp.py /usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
  --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
  $FIX --artifact $OUT --ar 32 --context 4096 --context-only \
  --calib-registers $A/export/dynreg8-20k-seed42/registers-f32.bin \
  --build /home/youngmin/dev/llm/executorch/build-android; then echo "$(date '+%H:%M') A32R_EXPORT_FAILED"; exit 1; fi
echo "$(date '+%H:%M') EXPORT_EXIT a32r 0"
if ! /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/extract_vlm_context.py $OUT $A/ctx_vlm_a32_c4096_r1 --compiled-contexts; then echo "$(date '+%H:%M') A32R_EXTRACT_FAILED"; exit 1; fi
[ -e $A/ctx_vlm_a32_c4096_r1/params.json ] || { echo A32R_EXTRACT_FAILED no params; exit 1; }
echo "$(date '+%H:%M') EXTRACT_EXIT a32r 0"; du -sh $OUT $A/ctx_vlm_a32_c4096_r1
echo A32R_TARGET_DONE
