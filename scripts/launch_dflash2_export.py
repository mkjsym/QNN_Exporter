"""Launch the bounded DFlash2 export with an owned-process guard."""
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/dflash2-20260919'
ET=ROOT.parent/'executorch'
SDK=Path('/home/youngmin/qairt/2.37.1-v81')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,default=1);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    run=OUT/('attempt-v'+str(args.revision));run.mkdir()
    env=os.environ.copy()
    overrides=dict(EXECUTORCH_ROOT=str(ET),QNN_SDK_ROOT=str(SDK),
        PYTHONPATH=str(ET.parent)+':'+str(ET),LD_LIBRARY_PATH=str(SDK/'lib/x86_64-linux-clang'),
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TMPDIR=str(run/'tmp'))
    (run/'tmp').mkdir(exist_ok=True);env.update(overrides)
    for name in ['DFLASH2_ALIGNED_CALIBRATION','DFLASH2_QDQ_REFERENCE']:
        if name in env:overrides[name]=env[name]
    audit=json.loads((ROOT.parent/'MobiSpec.cpp/docs/results/raw/dflash2-20260919/npu-f32-audit.json').read_text())
    assert audit['passed']
    assets=OUT/'npu-assets'
    argv=['/home/youngmin/anaconda3/envs/etqnn/bin/python',str(ROOT/'scripts/export_dflash2_draft.py'),
        '--decoder_model','qwen3-4b-dspark-w4','--model_mode','hybrid','--prefill_ar_len','9','--max_context_len','1024',
        '--soc_model','SM8850','--build_folder',str(ET/'build-android'),'--artifact',str(OUT/('npu-export-v'+str(args.revision))),
        '--compile_only','--checkpoint',str(assets/'draft-meta.pth'),
        '--calib_samples',str(assets/'calibration-prompts.json'),
        '--dflash_hidden',str(assets/'calibration-hidden.pt'),'--prompt','Once upon a time']
    (run/'command.json').write_text(json.dumps(dict(argv=argv,env_overrides=overrides,
        minimum_available_memory_gib=24,minimum_free_disk_gib=12,timeout_seconds=5400),indent=2)+'\n')
    samples=[];started=time.monotonic()
    with (run/'export.log').open('w') as log:
        process=subprocess.Popen(argv,cwd=ET,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                sample=dict(utc=datetime.now(timezone.utc).isoformat(),pid=process.pid,
                    available_memory_gib=mem['MemAvailable']/(1<<20),free_disk_gib=shutil.disk_usage(OUT).free/(1<<30),
                    elapsed_seconds=time.monotonic()-started)
                samples.append(sample)
                (run/'status.json').write_text(json.dumps(sample,indent=2)+'\n')
                if sample['available_memory_gib']<24 or sample['free_disk_gib']<12:raise RuntimeError('host resource guard')
                if sample['elapsed_seconds']>5400:raise TimeoutError('export exceeded 90min')
                time.sleep(5)
            assert process.returncode==0,'Export failed; see export.log'
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            (run/'host-resource-log.json').write_text(json.dumps(samples,indent=2)+'\n')
    print('DFLASH2_HOST_EXPORT_DONE',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':main()
