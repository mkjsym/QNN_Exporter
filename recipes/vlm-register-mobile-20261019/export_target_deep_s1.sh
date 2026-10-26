#!/bin/bash
# D1b (mobile plan 16.5, user 2026-10-26 "D1b, B만"): the deep-register target (per-layer prompts baked in) at ar_len 32 with encoding import.
# Same 16 fixtures in the same order as a32r (export_target_a32_r1.sh). Each calibration fixture is replayed twice: once on a32r's exact
# rows (dynreg8-20k-seed42 register table, every prompt slot zero; the deep graph then computes the plain graph's values bit for bit) and
# once on the deep rows (deep-dynreg8-20k-seed42 registers, one-hot slots). Every observer outside the prompt path keeps the first pass's
# state (--calib-common-registers), so the deep target should share a32r's encodings. Context 4096, root disk, THP disabled.
# About 110 minutes (calibration runs twice), RSS about 72 GB. Stops the export if the shared host's MemAvailable falls under 20 GiB.
set -o pipefail
R=$(cd "$(dirname "$0")" && pwd)
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
A=/home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019
FIX=$(sed "s/^/--fixture /" $R/fixtures16.list | tr "\n" " ")
cd /home/youngmin/dev/llm/QNN_Exporter
OUT=$A/target-deep-a32-c4096-s1
[ -e $OUT ] && { echo "SKIP existing $OUT"; exit 1; }
echo "$(date '+%H:%M') START deep32s (deep target, a32r encodings imported outside the prompt path), THP disabled"
setsid /usr/bin/python3 $R/nothp.py /usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
  --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
  $FIX --artifact $OUT --ar 32 --context 4096 --context-only --method dflash-deep \
  --layer-prompts /home/youngmin/dev/llm/MobiSpec.cpp/docs/results/raw/vlm-register-mobile-20261019/layer-prompts-deep-dynreg8-20k-seed42-f32.bin \
  --calib-registers $A/export/deep-dynreg8-20k-seed42/registers-f32.bin \
  --calib-common-registers $A/export/dynreg8-20k-seed42/registers-f32.bin \
  --build /home/youngmin/dev/llm/executorch/build-android &
PID=$!
while kill -0 $PID 2>/dev/null; do
  m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$m" -lt 20 ]; then echo "$(date '+%T') MemAvailable ${m} GiB < 20: stopping the export"; kill -TERM -$PID; fi
  sleep 10
done
wait $PID; rc=$?
[ $rc -eq 0 ] || { echo "$(date '+%H:%M') DEEP32S_EXPORT_FAILED rc=$rc"; exit 1; }
echo "$(date '+%H:%M') EXPORT_EXIT deep32s 0"
if ! /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/extract_vlm_context.py $OUT $A/ctx_vlm_deep_a32_c4096_s1 --compiled-contexts; then echo "$(date '+%H:%M') DEEP32S_EXTRACT_FAILED"; exit 1; fi
[ -e $A/ctx_vlm_deep_a32_c4096_s1/params.json ] || { echo DEEP32S_EXTRACT_FAILED no params; exit 1; }
echo "$(date '+%H:%M') EXTRACT_EXIT deep32s 0"; du -sh $OUT $A/ctx_vlm_deep_a32_c4096_s1
echo DEEP32S_TARGET_DONE
