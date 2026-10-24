#!/usr/bin/env python3
"""Freeze the horizon export invocation and keep all generated files under /data."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ET = Path('/home/youngmin/dev/llm/executorch')
SDK = Path('/home/youngmin/qairt/2.37.1-v81')
PY = '/home/youngmin/anaconda3/envs/etqnn/bin/python'
OUT = ROOT/'artifacts/horizon-20260917'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    backing = Path('/data/youngmin/qnn-artifacts/horizon-20260917')
    backing.mkdir()
    OUT.symlink_to(backing, target_is_directory=True)
    (OUT/'tmp').mkdir()
    old = json.loads((ROOT/'artifacts/mobile-head-20260915/head-w4-control-v2/head-w4-export-manifest.json').read_text())
    checkpoint, hidden = [Path(old['arguments'][key]) for key in ['checkpoint', 'dflash_hidden']]
    assert sha(checkpoint) == old['checkpoint_sha256'] and sha(hidden) == old['hidden_sha256']
    args = [PY, str(ROOT/'scripts/export_horizon_draft_w4.py'), '--decoder_model', 'qwen3-4b-dspark-w4',
        '--model_mode', 'hybrid', '--prefill_ar_len', '10', '--max_context_len', '1024',
        '--soc_model', 'SM8850', '--build_folder', str(ET/'build-android'),
        '--artifact', str(OUT/'export-q10-v1'), '--compile_only', '--checkpoint', str(checkpoint),
        '--calib_samples', str(ROOT/'data/calib_specbench.json'), '--dflash_hidden', str(hidden),
        '--prompt', 'Once upon a time']
    overrides = dict(EXECUTORCH_ROOT=str(ET), QNN_SDK_ROOT=str(SDK),
        ANDROID_NDK_ROOT='/home/youngmin/android-ndk-r26d', PYTHONPATH=str(ET.parent)+':'+str(ET),
        LD_LIBRARY_PATH=str(SDK/'lib/x86_64-linux-clang'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        CUDA_VISIBLE_DEVICES='0', TMPDIR=str(OUT/'tmp'))
    names = ['export_horizon_draft_w4.py', 'launch_horizon_export.py', 'export_draft_head_w4.py']
    (OUT/'sources').mkdir()
    for n in names:
        (OUT/'sources'/n).write_bytes((ROOT/'scripts'/n).read_bytes())
    plan = dict(utc=datetime.now(timezone.utc).isoformat(), argv=args, env_overrides=overrides,
                sources={n: sha(OUT/'sources'/n) for n in names},
                checkpoint_sha256=sha(checkpoint), hidden_sha256=sha(hidden),
                calibration_sha256=sha(ROOT/'data/calib_specbench.json'))
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    with (OUT/'export.log').open('x') as f:
        subprocess.run(args, cwd=ROOT, env=dict(os.environ, **overrides), stdout=f, stderr=subprocess.STDOUT, check=True)
    for n, digest in plan['sources'].items():
        assert sha(ROOT/'scripts'/n) == digest
    pte = OUT/'export-q10-v1/hybrid_llama_qnn.pte'
    assert pte.exists()
    (OUT/'complete.json').write_text(json.dumps(dict(pte_sha256=sha(pte), sources_unchanged=True))+'\n')
    print('HORIZON_EXPORT_COMPLETE', pte, flush=True)


if __name__ == '__main__':
    main()
