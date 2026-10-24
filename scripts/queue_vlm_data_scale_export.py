"""Convert, audit, export and extract one selected model with resource guards."""
import json,os,shutil,signal,subprocess,time
from pathlib import Path
from vlm_data_scale_export_common import ROOT,MOBI,STUDY,OUT,candidate,sha,write
ET=ROOT.parent/'executorch';SDK=Path('/home/youngmin/qairt/2.37.1-v81')
AW='/home/youngmin/anaconda3/envs/awbench/bin/python';PY='/home/youngmin/anaconda3/envs/etqnn/bin/python'

def main():
    c=candidate();key=c['key'];OUT.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',pid=os.getpid(),proc_start_ticks=Path('/proc/self/stat').read_text().split()[21],key=key,source_sha256=sha(__file__))
    helpers=['vlm_data_scale_export_common.py','prepare_vlm_data_scale_export.py','prepare_vlm_data_scale_export_references.py',
        'check_vlm_data_scale_export.py','export_vlm_data_scale_draft.py','check_vlm_residual8_export.py','export_vlm_residual8_draft.py','vlm_residual8_export_common.py']
    state['source_hashes']={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in helpers}
    env=dict(os.environ,EXECUTORCH_ROOT=str(ET),QNN_SDK_ROOT=str(SDK),PYTHONPATH=f'{ET.parent}:{ET}:{MOBI}/tools/training',
        LD_LIBRARY_PATH=str(SDK/'lib/x86_64-linux-clang'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
    asset=OUT/key;artifact=OUT/('npu-export-'+key);context=OUT/('ctx-'+key)
    jobs=[('calibration',[AW,str(ROOT/'scripts/prepare_vlm_data_scale_export.py'),'calibration']),
        ('convert',[PY,str(ROOT/'scripts/prepare_vlm_data_scale_export.py'),'convert']),
        ('reference',[AW,str(ROOT/'scripts/prepare_vlm_data_scale_export_references.py')]),
        ('audit',[PY,str(ROOT/'scripts/check_vlm_data_scale_export.py'),'--arm',key]),
        ('cpu-summary',[AW,str(MOBI/'tools/training/check_vlm_data_scale_cpu_summary.py')]),
        ('export',[PY,'-u',str(ROOT/'scripts/export_vlm_data_scale_draft.py'),'--decoder_model','qwen3-4b-dflash-w4','--model_mode','hybrid',
            '--prefill_ar_len','8','--max_context_len','1024','--soc_model','SM8850','--build_folder',str(ET/'build-android'),
            '--artifact',str(artifact),'--compile_only','--checkpoint',str(asset/'draft-meta.pth'),
            '--calib_samples',str(OUT/'calibration/calibration-prompts.json'),'--dflash_hidden',str(OUT/'calibration/calibration-hidden.pt'),'--prompt','Once upon a time']),
        ('extract',[PY,str(ROOT/'scripts/extract_variable_draft_context.py'),str(artifact),str(context),'--width','151936',
            '--params',str(ET/'examples/models/qwen3_dflash/config/4b_dflash_config.json'),'--physical-block','8','--logical-block','7','--inject-rows','16'])]
    try:
        for phase,cmd in jobs:
            assert shutil.disk_usage(OUT).free>50*(1<<30)
            state.update(phase=phase,command=cmd);write(OUT/'queue-status.json',state)
            with (OUT/(key+'-'+phase+'.log')).open('x') as out:
                child=subprocess.Popen(cmd,cwd=ET,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;write(OUT/'queue-status.json',state);start=time.monotonic()
            while child.poll() is None:
                mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                if shutil.disk_usage(OUT).free<50*(1<<30) or mem['MemAvailable']<24*(1<<20) or time.monotonic()-start>5400:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise RuntimeError('Resource/time guard; no deletion')
                time.sleep(5)
            assert child.returncode==0,f'{phase} failed; inspect saved log'
        manifest=json.loads((asset/'manifest.json').read_text())
        write(context/'runtime-contract.json',dict(context_sha256=sha(context/'forward_0.bin'),params_sha256=sha(context/'params.json'),
            asset_manifest_sha256=sha(asset/'manifest.json'),checkpoint_sha256=manifest['checkpoint_sha256'],mode=c['mode'],physical_block=8,open_block=7,
            output_rows=list(range(7)),target_ar_len=32,ddtree_budget=31,ddtree_k=4,
            summary_sha256=sha(asset/'summary-f32.bin') if c['mode']=='norm8' else None))
        assert all(sha(p)==h for p,h in state['source_hashes'].items())
        state.update(status='complete',context=str(context),context_sha256=sha(context/'forward_0.bin'),root_free_GiB=shutil.disk_usage(OUT).free/2**30)
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:write(OUT/'queue-status.json',state)
    print('DATA_SCALE_NPU_EXPORT_COMPLETE',key,flush=True)

if __name__=='__main__':main()
