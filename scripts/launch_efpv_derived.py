"""Run a bounded lowering/extraction stage on the frozen reconstructed target."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from launch_efpv_export import ROOT,OUT,ET,SDK


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['split','extract-full','extract-split'])
    p.add_argument('--revision',type=int,default=1);a=p.parse_args()
    attempt=OUT/(a.stage+'-attempt-v'+str(a.revision));attempt.mkdir()
    env=os.environ.copy();overrides=dict(EXECUTORCH_ROOT=str(ET),QNN_SDK_ROOT=str(SDK),
        PYTHONPATH=str(ET.parent)+':'+str(ET),LD_LIBRARY_PATH=str(SDK/'lib/x86_64-linux-clang'),
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TMPDIR=str(attempt/'tmp'))
    (attempt/'tmp').mkdir();env.update(overrides)
    python='/home/youngmin/anaconda3/envs/etqnn/bin/python'
    if a.stage=='split':
        argv=[python,str(ROOT/'scripts/export_efpv_split_target.py'),str(OUT/'full-v2'),str(OUT/('split-v'+str(a.revision)))]
    else:
        source='full-v2' if a.stage=='extract-full' else 'split-v1'
        argv=[python,str(ROOT/'scripts/extract_efpv_context.py'),str(OUT/source),str(OUT/(source+'-context')),
            '--shards','6' if a.stage=='extract-full' else '7','--params',
            str(ROOT.parent/'MobiSpec.cpp/docs/results/raw/mobile-heterogeneous-sd-20260918/efpv-split-v1/canonical-params.json')]
    (attempt/'command.json').write_text(json.dumps(dict(argv=argv,env_overrides=overrides,
        minimum_available_memory_gib=24,minimum_free_disk_gib=12,timeout_seconds=5400),indent=2)+'\n')
    samples=[];started=time.monotonic()
    with (attempt/'run.log').open('w') as log:
        job=subprocess.Popen(argv,cwd=ET,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while job.poll() is None:
                mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
                sample=dict(utc=datetime.now(timezone.utc).isoformat(),pid=job.pid,
                    available_memory_gib=mem['MemAvailable']/(1<<20),free_disk_gib=shutil.disk_usage(OUT).free/(1<<30),
                    elapsed_seconds=time.monotonic()-started)
                samples.append(sample);(attempt/'status.json').write_text(json.dumps(sample,indent=2)+'\n')
                if sample['available_memory_gib']<24 or sample['free_disk_gib']<12:raise RuntimeError('host resource guard')
                if sample['elapsed_seconds']>5400:raise TimeoutError('stage exceeded 90min')
                time.sleep(5)
            assert job.returncode==0,'Stage failed; see run.log'
        except BaseException:
            if job.poll() is None:
                os.killpg(job.pid,signal.SIGTERM)
                try:job.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait()
            raise
        finally:(attempt/'host-resource-log.json').write_text(json.dumps(samples,indent=2)+'\n')
    print('EFPV_DERIVED_DONE',a.stage,round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':main()
