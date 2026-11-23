"""Adopt an in-flight target export and finish both widths without restarting PTQ."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_regspec_exports import OUT, ROOT, ENV, PY, ARMS


def process_state(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:return None


def memory_kib():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--adopt-pid',type=int,required=True);p.add_argument('--original-state',type=Path,required=True)
    a=p.parse_args();original=json.loads(a.original_state.read_text());status=OUT/'targets-status.json'
    assert original['step']=='target-'+ARMS[1]+'-a40'
    def record(**kw):
        pending=status.with_suffix('.pending');pending.write_text(json.dumps(dict(pid=os.getpid(),utc=time.time(),**kw),indent=2)+'\n');pending.replace(status)
    def watch(pid,child=None):
        birth=process_state(pid)
        start=time.monotonic()
        while True:
            state=process_state(pid)
            if child is not None:
                if child.poll() is not None:assert child.returncode==0,('export exit',child.returncode);break
            elif state is None or state[0]=='Z' or (birth and state[1]!=birth[1]):break
            if memory_kib()<24*1024**2 or time.monotonic()-start>12*3600:
                os.killpg(pid,signal.SIGTERM);raise RuntimeError('resource/time guard')
            time.sleep(5)
    def extract(arm,width):
        name='target-'+arm+'-a'+str(width)
        record(status='running',step='extract-'+name)
        with (OUT/('extract-'+name+'.log')).open('a') as f:
            subprocess.run([PY,str(ROOT/'scripts/extract_vlm_context.py'),str(OUT/name),str(OUT/('ctx-'+name)),
                            '--compiled-contexts'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
        context=OUT/('ctx-'+name)
        assert (context/'params.json').exists()
        for i in range(6):
            assert (context/f'forward_{i}.bin').stat().st_size>0
            assert json.loads((context/f'forward_{i}_json.json').read_text())['info']['graphs']
        record(status='running',step='validated-'+name)
    try:
        record(status='running',step=original['step'],adopted_pid=a.adopt_pid,command=original['command'],
               note='Existing PTQ process continues; 12-hour per-process guard, same 24-GiB memory guard')
        print('ADOPT',a.adopt_pid,original['step'],flush=True)
        watch(a.adopt_pid)
        extract(ARMS[1],40)
        while memory_kib()<105*1024**2:
            record(status='waiting',step='target56 memory',available_kib=memory_kib());time.sleep(30)
        cmd=[v.replace(ARMS[1],ARMS[2]).replace('-a40','-a56') for v in original['command']]
        cmd[cmd.index('--ar')+1]='56'
        name='target-'+ARMS[2]+'-a56'
        record(status='running',step=name,command=cmd)
        with (OUT/(name+'.log')).open('a') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print('START',name,child.pid,flush=True);watch(child.pid,child);extract(ARMS[2],56)
        record(status='complete');print('TARGETS_COMPLETE',flush=True)
    except BaseException as e:record(status='failed',error=repr(e));raise


if __name__=='__main__':main()
