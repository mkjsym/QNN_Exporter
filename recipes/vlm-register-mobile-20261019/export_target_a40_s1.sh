#!/bin/bash
# D1 feasibility (mobile plan 16.3, 2026-10-24): the plain a40 target calibrated exactly like a32r (export_target_a32_r1.sh: same 16 fixtures,
# dynreg8-20k-seed42 register rows, 32 rows per calibration step via --calib-ar 32) but compiled at ar_len 40. If calibration is deterministic its
# encodings equal a32r's bit for bit, which separates "different encodings" from "different 40-row graph numerics" as the cause of a32r/a40r
# output differences. Context 4096, root disk, THP disabled. About 90 minutes, RSS about 70 GB.
set -o pipefail
R=$(cd "$(dirname "$0")" && pwd)
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
A=/home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019
FIX=$(sed "s/^/--fixture /" $R/fixtures16.list | tr "\n" " ")
cd /home/youngmin/dev/llm/QNN_Exporter
OUT=$A/target-a40-c4096-s1
[ -e $OUT ] && { echo "SKIP existing $OUT"; exit 1; }
echo "$(date '+%H:%M') START a40s (40-row target, calibrated at 32 rows like a32r), THP disabled"
if ! /usr/bin/python3 $R/nothp.py /usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
  --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
  $FIX --artifact $OUT --ar 40 --calib-ar 32 --context 4096 --context-only \
  --calib-registers $A/export/dynreg8-20k-seed42/registers-f32.bin \
  --build /home/youngmin/dev/llm/executorch/build-android; then echo "$(date '+%H:%M') A40S_EXPORT_FAILED"; exit 1; fi
echo "$(date '+%H:%M') EXPORT_EXIT a40s 0"
if ! /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/extract_vlm_context.py $OUT $A/ctx_vlm_a40_c4096_s1 --compiled-contexts; then echo "$(date '+%H:%M') A40S_EXTRACT_FAILED"; exit 1; fi
[ -e $A/ctx_vlm_a40_c4096_s1/params.json ] || { echo A40S_EXTRACT_FAILED no params; exit 1; }
echo "$(date '+%H:%M') EXTRACT_EXIT a40s 0"; du -sh $OUT $A/ctx_vlm_a40_c4096_s1
echo A40S_TARGET_DONE
