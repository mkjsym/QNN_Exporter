"""Bounded sequential conversion, numerical audit, and QAIRT export for each fixed arm."""
import json,os,subprocess,shutil,time,signal
from pathlib import Path
from vlm_residual8_export_common import ROOT,MOBI,STUDY,OUT,write,sha
ARMS=['norm8-seed42']
ET=ROOT.parent/'executorch';SDK=Path('/home/youngmin/qairt/2.37.1-v81')

def main():
 assert json.loads((STUDY/'holdout-summary.json').read_text())['mobile_gate_passed']
 micro=json.loads((STUDY/'mobile-summary/complete.json').read_text());assert micro['passed']
 gate=json.loads((STUDY/'measured-summary-cost-gate.json').read_text());assert gate['passed']
 assert gate['cpu_sha256']==sha(STUDY/'mobile-summary/complete.json') and gate['holdout_sha256']==sha(STUDY/'holdout-summary.json')
 OUT.mkdir(parents=True,exist_ok=True)
 state=dict(status='running',pid=os.getpid(),proc_start_ticks=Path('/proc/self/stat').read_text().split()[21],source_sha256=sha(__file__),arms=ARMS)
 status=OUT/'queue-status.json'
 env=dict(os.environ,EXECUTORCH_ROOT=str(ET),QNN_SDK_ROOT=str(SDK),PYTHONPATH=f'{ET.parent}:{ET}:{MOBI}/tools/training',
     LD_LIBRARY_PATH=str(SDK/'lib/x86_64-linux-clang'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
 py='/home/youngmin/anaconda3/envs/etqnn/bin/python';aw='/home/youngmin/anaconda3/envs/awbench/bin/python'
 try:
  if not (OUT/'calibration/manifest.json').exists():
   subprocess.run([aw,str(ROOT/'scripts/prepare_vlm_residual8_export.py'),'calibration'],env=env,check=True)
  for arm in ARMS:
   asset=OUT/arm;artifact=OUT/f'npu-export-{arm}'
   commands=[('convert',[py,str(ROOT/'scripts/prepare_vlm_residual8_export.py'),'convert','--arm',arm]),
     ('reference',[aw,str(ROOT/'scripts/prepare_vlm_residual8_export_references.py'),'--arms',arm]),
     ('audit',[py,str(ROOT/'scripts/check_vlm_residual8_export.py'),'--arm',arm]),
     ('export',[py,'-u',str(ROOT/'scripts/export_vlm_residual8_draft.py'),'--decoder_model','qwen3-4b-dflash-w4','--model_mode','hybrid','--prefill_ar_len','8',
       '--max_context_len','1024','--soc_model','SM8850','--build_folder',str(ET/'build-android'),'--artifact',str(artifact),'--compile_only',
       '--checkpoint',str(asset/'draft-meta.pth'),'--calib_samples',str(OUT/'calibration/calibration-prompts.json'),
       '--dflash_hidden',str(OUT/'calibration/calibration-hidden.pt'),'--prompt','Once upon a time'])]
   for phase,cmd in commands:
    state.update(arm=arm,phase=phase,command=cmd);write(status,state)
    assert shutil.disk_usage(OUT).free>50*(1<<30)
    log=OUT/f'{arm}-{phase}.log';started=time.monotonic()
    with log.open('x') as f:child=subprocess.Popen(cmd,env=env,cwd=ET,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    state['child_pid']=child.pid;write(status,state)
    while child.poll() is None:
     mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
     if shutil.disk_usage(OUT).free<50*(1<<30) or mem['MemAvailable']<24*(1<<20) or time.monotonic()-started>5400:
      os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise RuntimeError('Resource/time guard; no deletion')
     time.sleep(5)
    assert child.returncode==0,f'{arm}/{phase} failed; inspect {log}'
   write(OUT/f'{arm}-export-complete.json',dict(arm=arm,artifact=str(artifact)))
  state.update(status='complete')
 except BaseException as e:state.update(status='failed',error=repr(e));raise
 finally:write(status,state)
if __name__=='__main__':main()
