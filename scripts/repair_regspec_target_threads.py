"""Repeat only deep targets with the baseline's eight-thread calibration recipe."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_regspec_exports import OUT, ROOT, ENV, PY, ARMS

MOBI=ROOT.parent/'MobiSpec.cpp'
STUDY=MOBI/'docs/results/raw/regspec-s26-20261109'


def main():
    env=dict(ENV,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    original=json.loads((OUT/'targets-state-before-adoption.json').read_text())['command']
    status=OUT/'targets-status.json'
    def record(**kw):
        pending=status.with_suffix('.pending')
        pending.write_text(json.dumps(dict(pid=os.getpid(),utc=time.time(),variant='t8',**kw),indent=2)+'\n');pending.replace(status)
    def run(name,command,log):
        record(status='running',step=name,command=command)
        print('START',name,flush=True)
        with log.open('a') as f:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic()
        while child.poll() is None:
            mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
            if mem<24*1024**2 or shutil.disk_usage(OUT).free<50*2**30 or time.monotonic()-start>12*3600:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=40);raise RuntimeError(name+': resource/time guard')
            time.sleep(5)
        assert child.returncode==0,(name,child.returncode,str(log))
        print('DONE',name,flush=True)
    try:
        for arm,width,key in [(ARMS[1],40,'steering40'),(ARMS[2],56,'copy56')]:
            while True:
                mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
                if mem>=105*1024**2:break
                record(status='waiting',step='target repair memory',available_kib=mem);time.sleep(30)
            name='target-'+arm+'-a'+str(width)+'-t8';artifact=OUT/name;context=OUT/('ctx-'+name)
            assert not artifact.exists() and not context.exists(), 'Do not overwrite existing attempts'
            command=[v.replace(ARMS[1],arm) for v in original]
            command[command.index('--ar')+1]=str(width)
            command[command.index('--artifact')+1]=str(artifact)
            run(name,command,OUT/(name+'.log'))
            run('extract-'+name,[PY,str(ROOT/'scripts/extract_vlm_context.py'),str(artifact),str(context),'--compiled-contexts'],OUT/('extract-'+name+'.log'))
            run('encoding-check-'+key+'-t8',[PY,str(MOBI/'tools/research/compare_target_encodings.py'),
                str(ROOT/'artifacts/vlm-register-mobile-20261019/ctx_vlm_a32_c4096_r1'),str(context),'--extra-in-b','prompt_slot'],
                STUDY/('encoding-check-'+key+'-t8.log'))
        record(status='complete');print('TARGET_REPAIR_COMPLETE',flush=True)
    except BaseException as e:record(status='failed',error=repr(e));raise


if __name__=='__main__':main()
