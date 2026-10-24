#!/bin/bash
# Fair single table, A protocol (non-iso, 40 verify rows): the deep-register target (per-layer prompts baked in) at ar_len 40, the v2 recipe
# (export_target_deep_v3.sh: --calib-registers with the deep arm's register rows, THP disabled) with --ar 40. Context 4096, root disk. About 95 minutes, RSS about 72 GB.
set -o pipefail
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
S=/tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad
A=/home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019
FIX=$(sed "s/^/--fixture /" $S/fixtures16.list | tr "\n" " ")
cd /home/youngmin/dev/llm/QNN_Exporter
OUT=$A/target-deep-a40-c4096-v2
[ -e $OUT ] && { echo "SKIP existing $OUT"; exit 1; }
echo "$(date '+%H:%M') START deep ar=40 (layer prompts + register-row calibration), THP disabled"
if ! /usr/bin/python3 $S/nothp.py /usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
  --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
  $FIX --artifact $OUT --ar 40 --context 4096 --context-only --method dflash-deep \
  --layer-prompts /home/youngmin/dev/llm/MobiSpec.cpp/docs/results/raw/vlm-register-mobile-20261019/layer-prompts-deep-dynreg8-20k-seed42-f32.bin \
  --calib-registers $A/export/deep-dynreg8-20k-seed42/registers-f32.bin \
  --build /home/youngmin/dev/llm/executorch/build-android; then echo "$(date '+%H:%M') DEEP40_EXPORT_FAILED"; exit 1; fi
echo "$(date '+%H:%M') EXPORT_EXIT deep40 0"
if ! /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/extract_vlm_context.py $OUT $A/ctx_vlm_deep_a40_c4096_v2 --compiled-contexts; then echo "$(date '+%H:%M') DEEP40_EXTRACT_FAILED"; exit 1; fi
[ -e $A/ctx_vlm_deep_a40_c4096_v2/params.json ] || { echo DEEP40_EXTRACT_FAILED no params; exit 1; }
echo "$(date '+%H:%M') EXTRACT_EXIT deep40 0"; du -sh $OUT $A/ctx_vlm_deep_a40_c4096_v2
echo DEEP40_TARGET_DONE
