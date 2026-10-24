#!/bin/bash
# M2: deep-register Qwen3-VL-4B target, second export (2026-10-19 18:05): calibration fixtures now carry the 8 register rows with one-hot
# prompt slots (--calib-registers), so the baked prompt path is quantized on real prompt values. Context 4096, ar_len 32.
# Attempt 2 (22:2x): the first attempt (18:04) starved in kernel huge-page compaction on the full server; this one runs under nothp.py (PR_SET_THP_DISABLE).
set -o pipefail
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
A=/data/youngmin/qnn-artifacts/vlm-register-mobile-20261019
FIX=$(sed "s/^/--fixture /" /tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad/fixtures16.list | tr "\n" " ")
cd /home/youngmin/dev/llm/QNN_Exporter
OUT=$A/target-deep-a32-c4096-v3
[ -e $OUT ] && { echo "SKIP existing $OUT"; exit 1; }
echo "$(date '+%H:%M') START deep v2 ar=32 (calibrated prompt slots), attempt 2, THP disabled"
if ! /usr/bin/python3 /tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad/nothp.py /usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
  --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
  $FIX --artifact $OUT --ar 32 --context 4096 --context-only --method dflash-deep \
  --layer-prompts /home/youngmin/dev/llm/MobiSpec.cpp/docs/results/raw/vlm-register-mobile-20261019/layer-prompts-deep-dynreg8-20k-seed42-f32.bin \
  --calib-registers /home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/export/deep-dynreg8-20k-seed42/registers-f32.bin \
  --build /home/youngmin/dev/llm/executorch/build-android; then echo "$(date '+%H:%M') DEEP2_EXPORT_FAILED"; echo DEEP2_CHAIN_EXIT 1; exit 1; fi
echo "$(date '+%H:%M') EXPORT_EXIT deep2 0"
if ! /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/extract_vlm_context.py $OUT $A/ctx_vlm_deep_a32_c4096_v2 --compiled-contexts; then echo "$(date '+%H:%M') DEEP2_EXTRACT_FAILED"; echo DEEP2_CHAIN_EXIT 1; exit 1; fi
[ -e $A/ctx_vlm_deep_a32_c4096_v2/params.json ] || { echo DEEP2_EXTRACT_FAILED no params; echo DEEP2_CHAIN_EXIT 1; exit 1; }
echo "$(date '+%H:%M') EXTRACT_EXIT deep2 0"
echo DEEP2_TARGET_DONE
echo DEEP2_CHAIN_EXIT 0
