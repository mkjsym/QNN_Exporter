#!/bin/bash
# M2: the deep-register Qwen3-VL-4B target (per-layer prompts baked in, vlm_prompt_slot input), context 4096, ar_len 32 (B protocol deep@T23).
set -o pipefail
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
A=/data/youngmin/qnn-artifacts/vlm-register-mobile-20261019
FIX=$(sed "s/^/--fixture /" /tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad/fixtures16.list | tr "\n" " ")
cd /home/youngmin/dev/llm/QNN_Exporter
OUT=$A/target-deep-a32-c4096-v2
[ -e $OUT ] && { echo "SKIP existing $OUT"; exit 1; }
echo "$(date '+%H:%M') START deep ar=32"
if ! /usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
  --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
  $FIX --artifact $OUT --ar 32 --context 4096 --context-only --method dflash-deep \
  --layer-prompts /home/youngmin/dev/llm/MobiSpec.cpp/docs/results/raw/vlm-register-mobile-20261019/layer-prompts-deep-dynreg8-20k-seed42-f32.bin \
  --build /home/youngmin/dev/llm/executorch/build-android; then echo "$(date '+%H:%M') DEEP_EXPORT_FAILED"; exit 1; fi
echo "$(date '+%H:%M') EXPORT_EXIT deep 0"
if ! /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/extract_vlm_context.py $OUT $A/ctx_vlm_deep_a32_c4096 --compiled-contexts; then echo "$(date '+%H:%M') DEEP_EXTRACT_FAILED"; exit 1; fi
[ -e $A/ctx_vlm_deep_a32_c4096/params.json ] || { echo DEEP_EXTRACT_FAILED no params; exit 1; }
echo "$(date '+%H:%M') EXTRACT_EXIT deep 0"
echo DEEP_TARGET_DONE
