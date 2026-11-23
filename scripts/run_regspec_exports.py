"""Run the frozen S26 request exports with independent artifact directories."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = Path('/data/youngmin/qnn-artifacts/regspec-s26-20261109')
PY = '/home/youngmin/anaconda3/envs/etqnn/bin/python'
AW = '/home/youngmin/anaconda3/envs/awbench/bin/python'
ARMS = ['full-s1000-seed43', 'dynreg8-deep-taps-s1000-seed43', 'dynreg8-deep-taps-copy024-s1000-seed43']
ENV = dict(os.environ, EXECUTORCH_ROOT=str(BASE/'executorch'), QNN_SDK_ROOT='/home/youngmin/qairt/2.37.1-v81',
    REGISTER_MOBILE_STUDY='/data/regspec/mobile-request-20261109/drafters',
    REGISTER_MOBILE_CALIB=str(OUT/'calibration'), REGISTER_MOBILE_EXPORT_OUT=str(OUT/'export'),
    PYTHONPATH=f'{BASE}:{BASE}/executorch:{BASE}/MobiSpec.cpp/tools/training:{ROOT}/scripts',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6')

def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['drafts','targets']); a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    status=OUT/(a.phase+'-status.json')
    def record(**kw):
        tmp=status.with_suffix('.pending'); tmp.write_text(json.dumps(dict(pid=os.getpid(),utc=time.time(),**kw),indent=2)+'\n'); tmp.replace(status)
    def run(name,cmd):
        record(status='running',step=name,command=cmd)
        print('START',name,flush=True)
        with (OUT/(name+'.log')).open('a') as f:
            child=subprocess.Popen(cmd,env=ENV,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic()
        while child.poll() is None:
            mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
            if mem < 24*1024**2 or time.monotonic()-start > 14400:
                os.killpg(child.pid,signal.SIGTERM); child.wait(timeout=40); raise RuntimeError(name+': resource/time guard')
            time.sleep(5)
        if child.returncode: raise RuntimeError(name+' failed: '+str(OUT/(name+'.log')))
        print('DONE',name,flush=True)
    try:
        if a.phase=='drafts':
            run('prepare-calibration',[AW,str(ROOT/'scripts/prepare_vlm_register_mobile_export.py'),'calibration'])
            for arm in ARMS:
                run('convert-'+arm,[PY,str(ROOT/'scripts/prepare_vlm_register_mobile_export.py'),'convert','--arm',arm])
            for arm in ARMS:
                ctx='4096' if arm.startswith('full-') else '1024'
                run('step-'+arm,[PY,str(ROOT/'scripts/queue_vlm_register_mobile_export.py'),arm,'--context',ctx,'--phase-limit-s','14400'])
                run('inject-'+arm,[PY,str(ROOT/'scripts/queue_vlm_register_mobile_export.py'),arm,'--inject','256','--phase-limit-s','14400'])
        else:
            fixtures=(ROOT/'recipes/vlm-register-mobile-20261019/fixtures16.list').read_text().splitlines()
            old=ROOT/'artifacts/vlm-register-mobile-20261019/export'
            for arm,width in zip(ARMS[1:],[40,56]):
                asset=OUT/'export'/arm
                while not (asset/'manifest.json').exists():
                    record(status='waiting',step='converted assets',arm=arm)
                    other=OUT/'drafts-status.json'
                    if other.exists() and json.loads(other.read_text()).get('status')=='failed': raise RuntimeError('draft conversion failed')
                    time.sleep(10)
                while True:
                    mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
                    if mem>=105*1024**2: break
                    record(status='waiting',step='target export memory',available_kib=mem); time.sleep(30)
                name='target-'+arm+'-a'+str(width)
                cmd=[PY,str(ROOT/'recipes/vlm-register-mobile-20261019/nothp.py'),PY,str(ROOT/'scripts/export_vlm_target.py'),
                    '--checkpoint','/data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf',
                    '--tokenizer',str(BASE/'specinfer_QNN/experiments/vlm-dflash-20260913/tokenizer'),
                    '--artifact',str(OUT/name),'--ar',str(width),'--context','4096','--context-only','--method','dflash-deep',
                    '--layer-prompts',str(asset/'layer-prompts-f32.bin'),'--calib-registers',str(asset/'registers-f32.bin'),
                    '--calib-common-registers',str(old/'dynreg8-20k-seed42/registers-f32.bin'),'--calib-ar','32',
                    '--build',str(BASE/'executorch/build-android')]
                for f in fixtures: cmd+=['--fixture',f]
                run(name,cmd)
                run('extract-'+name,[PY,str(ROOT/'scripts/extract_vlm_context.py'),str(OUT/name),str(OUT/('ctx-'+name)),'--compiled-contexts'])
        record(status='complete')
    except BaseException as e:
        record(status='failed',error=repr(e)); raise

if __name__=='__main__':main()
