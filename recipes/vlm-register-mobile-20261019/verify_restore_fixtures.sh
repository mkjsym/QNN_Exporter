#!/bin/bash
# Verify the fixture files pulled back from the phone (count, sizes vs the deleted-file list, sha256 vs the device), then put them at the harness FIXTURES path.
set -u
S=/tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad
L=/home/youngmin/dev/llm/QNN_Exporter/artifacts/vlm-register-mobile-20261019; R=$L/fixtures-m1.restoring
REMOTE=/data/local/tmp/vlm-register-mobile-20261019/fixtures
n=$(ls $R | wc -l); echo "local files: $n"
python3 - "$R" <<'PY' || exit 1
import sys, pathlib
S='/tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad'
R=pathlib.Path(sys.argv[1]); keep=('.json','.tokens.i32','.positions.i32','.vlm_embd.f32','.vlm_deepstack_0.f32','.vlm_deepstack_1.f32','.vlm_deepstack_2.f32')
want={n:int(b) for n,b in (l.split('\t') for l in open(f'{S}/fixtures_m1_names.txt').read().splitlines() if l) if n.endswith(keep)}
have={p.name:p.stat().st_size for p in R.iterdir()}
bad=[n for n in want if have.get(n)!=want[n]]; extra=[n for n in have if n not in want]
print('expected', len(want), 'present', len(have), 'size mismatch or missing', len(bad), bad[:3], 'extra', len(extra), extra[:3])
sys.exit(0 if not bad and not extra and len(want)==896 else 1)
PY
(cd $R && find . -type f -exec sha256sum {} + | sed 's#\./##' | sort -k2) > $S/fixtures_local.sha
adb -s R5KL20FMLCN shell "cd $REMOTE && find . -type f -exec sha256sum {} +" | tr -d '\r' | sed 's#\./##' | sort -k2 > $S/fixtures_device.sha
echo "sha lines local $(wc -l < $S/fixtures_local.sha) device $(wc -l < $S/fixtures_device.sha)"
if cmp -s $S/fixtures_local.sha $S/fixtures_device.sha && [ "$(wc -l < $S/fixtures_local.sha)" = 896 ]; then
  [ -L $L/fixtures-m1 ] && [ ! -e $L/fixtures-m1 ] && rm $L/fixtures-m1 && echo "dangling symlink removed"
  [ ! -e $L/fixtures-m1 ] && mv $R $L/fixtures-m1 && echo "FIXTURES_RESTORED $(date '+%H:%M') $(ls $L/fixtures-m1 | wc -l) files $(du -sh $L/fixtures-m1 | cut -f1)"
else
  echo "FIXTURES_VERIFY_FAILED"; diff $S/fixtures_local.sha $S/fixtures_device.sha | head -5
fi
