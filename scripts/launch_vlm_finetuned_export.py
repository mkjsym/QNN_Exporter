"""Promote the Full-FT control only after its complete, exact online validation."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/vlm-ft-20260929'
FOLLOWUP = ROOT.parent/'MobiSpec.cpp/docs/results/raw/vlm-learned8-followup-20260929'


def write(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main(attempt):
    run = OUT/f'full42-export-attempt{attempt}'
    run.mkdir(exist_ok=False)
    status = dict(status='waiting_for_complete_seed42_rollout', started_utc=now(), pid=os.getpid(),
                  proc_start_ticks=Path('/proc/self/stat').read_text().split()[21])
    write(run/'status.json', status)
    started = time.monotonic()
    while True:
        records = []
        for folder, arm in [('original-learned8', 'untrained_full'), ('full', 'full_seed42')]:
            path = FOLLOWUP/f'rollout-{folder}-fp32/status.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            if state.get('status') == 'failed':
                raise RuntimeError('Preserve failed rollout before any promotion')
            cases = [r for r in state.get('cases', []) if r['arm'] == arm and r['policy'] == 'ddtree32']
            records.append(cases)
        if all(len(cases) == 256 for cases in records):
            break
        assert time.monotonic()-started < 10800
        time.sleep(10)
    assert all(c['reference_equal'] for cases in records for c in cases)
    records = [sorted(cases, key=lambda x:x['id']) for cases in records]
    assert [(c['id'], c['decode_tokens']) for c in records[0]] == [(c['id'], c['decode_tokens']) for c in records[1]]
    old_calls, new_calls = [sum(c['verification_calls'] for c in cases) for cases in records]
    gain = old_calls/new_calls-1
    status.update(validation_al_gain=gain, reference_exact=True, total_inputs=256, decision_utc=now())
    if gain <= 0:
        status.update(status='not_promoted_no_online_al_gain')
        write(run/'status.json', status)
        return
    et = ROOT.parent/'executorch'
    sdk = Path('/home/youngmin/qairt/2.37.1-v81')
    (run/'tmp').mkdir()
    overrides = dict(EXECUTORCH_ROOT=str(et), QNN_SDK_ROOT=str(sdk),
        PYTHONPATH=str(et.parent)+':'+str(et), LD_LIBRARY_PATH=str(sdk/'lib/x86_64-linux-clang'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TMPDIR=str(run/'tmp'))
    artifact = OUT/f'npu-export-full42-v{attempt}'
    command = ['/home/youngmin/anaconda3/envs/etqnn/bin/python', '-u', str(ROOT/'scripts/export_vlm_finetuned_draft.py'),
        '--decoder_model', 'qwen3-4b-dflash-w4', '--model_mode', 'hybrid', '--prefill_ar_len', '8',
        '--max_context_len', '1024', '--soc_model', 'SM8850', '--build_folder', str(et/'build-android'),
        '--artifact', str(artifact), '--compile_only', '--checkpoint', str(OUT/'full_seed42/draft-meta.pth'),
        '--calib_samples', str(OUT/'calibration/calibration-prompts.json'),
        '--dflash_hidden', str(OUT/'calibration/calibration-hidden.pt'), '--prompt', 'Once upon a time']
    write(run/'command.json', dict(argv=command, env_overrides=overrides, disk_reserve_gib=50, memory_reserve_gib=24,
                                  timeout_seconds=5400, decision='Positive paired DDTree32 AL on all256 validation inputs; an existing full-context FT control, not a novel method'))
    assert shutil.disk_usage(OUT).free > 70*(1 << 30)
    with (run/'export.log').open('x') as log:
        child = subprocess.Popen(command, cwd=et, env=dict(os.environ, **overrides),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status.update(status='export_running', child_pid=child.pid, export_started_utc=now())
        write(run/'status.json', status)
        started = time.monotonic()
        samples = []
        try:
            while child.poll() is None:
                mem = {line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                sample = dict(utc=now(), free_disk_gib=shutil.disk_usage(OUT).free/(1 << 30),
                              available_memory_gib=mem['MemAvailable']/(1 << 20), elapsed_seconds=time.monotonic()-started)
                samples.append(sample)
                write(run/'resources.json', sample)
                assert sample['free_disk_gib'] > 50 and sample['available_memory_gib'] > 24, 'Resource guard'
                assert sample['elapsed_seconds'] < 5400, 'Export timeout'
                time.sleep(5)
            assert child.returncode == 0, 'Export failure; inspect export.log'
            status.update(status='export_complete', completed_utc=now())
        except BaseException as error:
            status.update(status='failed', failure=str(error), failed_utc=now())
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            write(run/'status.json', status)
            write(run/'resource-history.json', samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert args.attempt > 0
    try:
        main(args.attempt)
    except BaseException as error:
        path = OUT/f'full42-export-attempt{args.attempt}/status.json'
        if path.exists():
            value = json.loads(path.read_text())
            value.update(status='failed', failure=str(error), failed_utc=now())
            write(path, value)
        raise
