#!/bin/bash
# Fair single table, A protocol (non-iso, 40 verify rows): the plain a40 target re-exported with the dynreg8-20k-seed42 register rows appended to every PTQ
# calibration fixture, exactly as a32r (export_target_a32_r1.sh, M6) but ar_len 40. Context 4096, same 16 fixtures, root disk, THP disabled. About 90 minutes, RSS about 70 GB.
set -o pipefail
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
S=/tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad
A=/home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019
FIX=$(sed "s/^/--fixture /" $S/fixtures16.list | tr "\n" " ")
cd /home/youngmin/dev/llm/QNN_Exporter
OUT=$A/target-a40-c4096-r1
[ -e $OUT ] && { echo "SKIP existing $OUT"; exit 1; }
echo "$(date '+%H:%M') START a40 r1 (plain 40-row target, calibration fixtures + dynreg8 register rows), THP disabled"
if ! /usr/bin/python3 $S/nothp.py /usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
  --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
  $FIX --artifact $OUT --ar 40 --context 4096 --context-only \
  --calib-registers $A/export/dynreg8-20k-seed42/registers-f32.bin \
  --build /home/youngmin/dev/llm/executorch/build-android; then echo "$(date '+%H:%M') A40R_EXPORT_FAILED"; exit 1; fi
echo "$(date '+%H:%M') EXPORT_EXIT a40r 0"
if ! /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/extract_vlm_context.py $OUT $A/ctx_vlm_a40_c4096_r1 --compiled-contexts; then echo "$(date '+%H:%M') A40R_EXTRACT_FAILED"; exit 1; fi
[ -e $A/ctx_vlm_a40_c4096_r1/params.json ] || { echo A40R_EXTRACT_FAILED no params; exit 1; }
echo "$(date '+%H:%M') EXTRACT_EXIT a40r 0"; du -sh $OUT $A/ctx_vlm_a40_c4096_r1
echo A40R_TARGET_DONE
