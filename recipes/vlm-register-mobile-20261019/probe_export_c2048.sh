#!/bin/bash
# M0 probe: Qwen3-VL-4B target export at context 2048 / ar_len 32 with one document fixture, to measure wall time and peak RAM before the 4096 exports.
set -o pipefail
export EXECUTORCH_ROOT=/home/youngmin/dev/llm/executorch
export QNN_SDK_ROOT=/home/youngmin/qairt/2.37.1-v81
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/youngmin/dev/llm/QNN_Exporter
/usr/bin/time -v /home/youngmin/anaconda3/envs/etqnn/bin/python scripts/export_vlm_target.py \
  --checkpoint /data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf \
  --tokenizer /home/youngmin/dev/llm/specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer \
  --fixture /home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/fixtures/doc2-chartqa-00-1886.json \
  --artifact /data/youngmin/qnn-artifacts/vlm-register-mobile-20261019/target-probe-a32-c2048-r3 \
  --ar 32 --context 2048 --context-only --build /home/youngmin/dev/llm/executorch/build-android
echo PROBE_EXPORT_EXIT $?
