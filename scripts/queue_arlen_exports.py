"""Sequential width exports with bounded disk/RAM and an overnight deadline."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', help='PRECISION:WIDTH, for a controlled recovery')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'artifacts/arlen-ddtree-20260928'
    out.mkdir(parents=True, exist_ok=True)
    et = root.parent / 'executorch'
    sdk = Path('/home/youngmin/qairt/2.37.1-v81')
    env = os.environ.copy()
    env.update(EXECUTORCH_ROOT=str(et), QNN_SDK_ROOT=str(sdk),
               PYTHONPATH=str(et.parent)+':'+str(et),
               LD_LIBRARY_PATH=str(sdk / 'lib/x86_64-linux-clang'),
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    jobs = [(p, a) for p in ['W4A16', 'W4A8', 'W8A16'] for a in [32, 64]]
    if args.only:
        precision, width = args.only.split(':')
        jobs = [(precision, int(width))]
    for precision, width in jobs:
        name = f'{precision}-a{width}'
        target = out / name
        if (target / 'complete.json').exists():
            continue
        assert not target.exists(), f'Inspect prior incomplete export: {target}'
        argv = ['/home/youngmin/anaconda3/envs/etqnn/bin/python',
                str(root / 'scripts/export_arlen_target.py'), '--ar', str(width),
                '--precision', precision, '--output', str(target)]
        started = time.monotonic()
        with (out / f'{name}.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=et, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    memory = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
                    status = dict(precision=precision, width=width, pid=process.pid,
                                  utc=datetime.now(timezone.utc).isoformat(),
                                  seconds=time.monotonic()-started,
                                  free_disk_gib=shutil.disk_usage(out).free/(1<<30),
                                  free_ram_gib=memory['MemAvailable']/(1<<20))
                    (out / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
                    if status['free_disk_gib'] < 6 or status['free_ram_gib'] < 20:
                        raise RuntimeError('Export resource guard')
                    if status['seconds'] > 7200:
                        raise TimeoutError('Export exceeded two hours')
                    time.sleep(10)
                assert process.returncode == 0, f'Export {name} failed; inspect log'
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        print('WIDTH_READY', name, round(time.monotonic()-started), flush=True)
    if not args.only:
        (out / 'queue-complete.json').write_text(json.dumps(dict(complete=True,
            jobs=[f'{p}-a{a}' for p, a in jobs]), indent=2)+'\n')


if __name__ == '__main__':
    main()
