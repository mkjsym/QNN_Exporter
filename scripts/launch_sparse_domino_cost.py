"""Guard the isolated CPU-only sparse Domino export and preserve provenance."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/sparse-domino-cost-20260922'
ET = Path('/home/youngmin/dev/llm/executorch')
SDK = Path('/home/youngmin/qairt/2.37.1-v81')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    OUT.mkdir(); (OUT / 'tmp').mkdir(); (OUT / 'sources').mkdir()
    old = ROOT / 'artifacts/long-draft-20260917/export-domino-w4-v1/long-draft-export-manifest.json'
    old_manifest = json.loads(old.read_text())
    checkpoint = ROOT / old_manifest['arguments']['checkpoint']; hidden = ROOT / old_manifest['arguments']['dflash_hidden']
    assert sha(checkpoint) == old_manifest['checkpoint_sha256'] and sha(hidden) == old_manifest['hidden_sha256']
    calibration = ROOT / old_manifest['arguments']['calib_samples'][0]
    names = ['launch_sparse_domino_cost.py', 'export_sparse_domino_cost.py', 'export_domino_w4.py',
        'export_long_draft_w4.py', 'export_draft_body.py', 'export_draft_head_w4.py']
    for name in names:
        shutil.copyfile(ROOT / 'scripts' / name, OUT / 'sources' / name)
    overrides = dict(EXECUTORCH_ROOT=str(ET), QNN_SDK_ROOT=str(SDK), ANDROID_NDK_ROOT='/home/youngmin/android-ndk-r26d',
        PYTHONPATH=str(ET.parent) + ':' + str(ET), LD_LIBRARY_PATH=str(SDK / 'lib/x86_64-linux-clang'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TMPDIR=str(OUT / 'tmp'))
    argv = ['/home/youngmin/anaconda3/envs/etqnn/bin/python', '-u', str(ROOT / 'scripts/export_sparse_domino_cost.py'),
        '--decoder_model', 'qwen3-4b-dspark-w4', '--model_mode', 'hybrid', '--prefill_ar_len', '8', '--max_context_len', '1024',
        '--soc_model', 'SM8850', '--build_folder', str(ET / 'build-android'), '--artifact', str(OUT / 'export-v1'),
        '--compile_only', '--checkpoint', str(checkpoint), '--calib_samples', str(calibration), '--dflash_hidden', str(hidden),
        '--prompt', 'Once upon a time']
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), argv=argv, env_overrides=overrides,
        sources={name: sha(OUT / 'sources' / name) for name in names},
        inputs={str(p): sha(p) for p in [old, checkpoint, hidden, calibration]},
        limits=dict(cpu_threads=8, gpu=False, minimum_available_GiB=24, minimum_disk_GiB=20, timeout_seconds=3600))
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n'); samples = []; start = time.monotonic()
    with (OUT / 'export.log').open('x') as log:
        job = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, **overrides), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while job.poll() is None:
                mem = {x.split(':')[0]: int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
                status = dict(pid=job.pid, utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                    available_GiB=mem['MemAvailable']/2**20, disk_GiB=shutil.disk_usage(OUT).free/2**30)
                samples.append(status); (OUT / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
                if status['available_GiB'] < 24 or status['disk_GiB'] < 20 or status['elapsed_seconds'] > 3600:
                    raise RuntimeError('Export resource/time guard')
                time.sleep(5)
            assert job.returncode == 0, 'Export failed; preserve log and inspect before any retry'
        except BaseException:
            if job.poll() is None:
                os.killpg(job.pid, signal.SIGTERM)
                try:
                    job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL); job.wait()
            raise
        finally:
            (OUT / 'host-resources.json').write_text(json.dumps(samples, indent=2) + '\n')
            (OUT / 'terminal.json').write_text(json.dumps(dict(returncode=job.returncode, elapsed_seconds=time.monotonic()-start), indent=2) + '\n')
    for name, digest in plan['sources'].items():
        assert sha(ROOT / 'scripts' / name) == digest
    for path, digest in plan['inputs'].items():
        assert sha(path) == digest
    print('SPARSE_DOMINO_COST_EXPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
