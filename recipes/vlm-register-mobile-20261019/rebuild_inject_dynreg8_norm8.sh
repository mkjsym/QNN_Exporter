#!/bin/bash
# Rebuild on the root disk what the 10-22 server cleanup (S1) removed and the inject256 exports need: calibration/ (derived from the hash-checked page dump)
# and draft-meta.pth of dynreg8/norm8 (convert from the training checkpoints; outputs must equal the hashes in the surviving /data manifests), then the queue's
# inject256 audit/export/extract with OUT=export-root, then copy ctx-<arm>-inject256 into the harness EXPORT dir (/data, 35 MB each).
set -e
export REGISTER_MOBILE_EXPORT_OUT=/home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/export-root
ET=/home/youngmin/dev/llm/executorch; SDK=/home/youngmin/qairt/2.37.1-v81; ROOT=/home/youngmin/dev/llm/QNN_Exporter; MOBI=/home/youngmin/dev/llm/MobiSpec.cpp
export EXECUTORCH_ROOT=$ET QNN_SDK_ROOT=$SDK PYTHONPATH=$ET/..:$ET:$MOBI/tools/training:$ROOT/scripts LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 REGISTER_CAP=doc20k
AW=/home/youngmin/anaconda3/envs/awbench/bin/python; PY=/home/youngmin/anaconda3/envs/etqnn/bin/python; E=$ROOT/artifacts/vlm-register-mobile-20261019/export
mkdir -p /home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/export-root; cd $ET
echo "$(date '+%T') calibration"; [ -e /home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/export-root/calibration/manifest.json ] || $AW $ROOT/scripts/prepare_vlm_register_mobile_export.py calibration
for arm in dynreg8-20k-seed42 norm8-20k-seed42; do
  echo "$(date '+%T') convert $arm"; [ -e /home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/export-root/$arm/manifest.json ] || $PY $ROOT/scripts/prepare_vlm_register_mobile_export.py convert --arm $arm
  $PY - $E/$arm/manifest.json /home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/export-root/$arm/manifest.json <<'PYEOF'
import json, sys
old, new = (json.load(open(p))['outputs'] for p in sys.argv[1:3])
bad = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
assert not bad, ('regenerated outputs differ from the recorded hashes', bad)
print('CONVERT_IDENTICAL', sys.argv[2], sorted(old))
PYEOF
done
echo "$(date '+%T') inject256 queue"; $PY $ROOT/scripts/queue_vlm_register_mobile_export.py --inject 256 dynreg8-20k-seed42 norm8-20k-seed42
for arm in dynreg8-20k-seed42 norm8-20k-seed42; do
  c=ctx-$arm-inject256; [ -e $E/$c ] && { echo "EXISTS $E/$c"; exit 1; }
  cp -a /home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/export-root/$c $E/$c.copying && (cd /home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019/export-root/$c && sha256sum *) > /tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad/$c.sha && (cd $E/$c.copying && sha256sum -c --quiet /tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad/$c.sha) && mv $E/$c.copying $E/$c
  echo "COPIED $c"
done
echo "$(date '+%T') INJECT_REBUILD_OK"
