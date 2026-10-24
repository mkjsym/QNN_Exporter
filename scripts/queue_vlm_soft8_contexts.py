"""Extract completed exports and pin each b7 runtime contract without changing weights."""
import json,os,subprocess,time
from pathlib import Path
from vlm_soft8_export_common import ROOT,OUT,write,sha

def main():
 state=dict(status='running',pid=os.getpid());status=OUT/'context-queue.json'
 try:
  et=ROOT.parent/'executorch';sdk=Path('/home/youngmin/qairt/2.37.1-v81')
  env=dict(os.environ,EXECUTORCH_ROOT=str(et),QNN_SDK_ROOT=str(sdk),PYTHONPATH=f'{et.parent}:{et}',LD_LIBRARY_PATH=str(sdk/'lib/x86_64-linux-clang'))
  for arm in ['norm8-seed42']:
   state.update(arm=arm,phase='waiting');write(status,state);start=time.monotonic()
   while not (OUT/f'{arm}-export-complete.json').exists():
    queue=json.loads((OUT/'queue-status.json').read_text());assert queue['status']!='failed',queue.get('error')
    assert time.monotonic()-start<10800;time.sleep(10)
   artifact=OUT/f'npu-export-{arm}';dest=OUT/f'ctx-{arm}'
   command=['/home/youngmin/anaconda3/envs/etqnn/bin/python',str(ROOT/'scripts/extract_variable_draft_context.py'),str(artifact),str(dest),
      '--width','151936','--params',str(et/'examples/models/qwen3_dflash/config/4b_dflash_config.json'),
      '--physical-block','8','--logical-block','7','--inject-rows','16']
   state.update(phase='extracting',command=command);write(status,state)
   with (OUT/f'{arm}-extract.log').open('x') as out:subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
   asset=OUT/arm;manifest=json.loads((asset/'manifest.json').read_text())
   write(dest/'runtime-contract.json',dict(context_sha256=sha(dest/'forward_0.bin'),params_sha256=sha(dest/'params.json'),
       asset_manifest_sha256=sha(asset/'manifest.json'),mode=manifest['mode'],physical_block=8,open_block=7,output_rows=list(range(7)),
       target_ar_len=32,ddtree_budget=31,ddtree_k=4,summary_sha256=sha(asset/'summary-f32.bin')))
  state.update(status='complete')
 except BaseException as e:state.update(status='failed',error=repr(e));raise
 finally:write(status,state)
if __name__=='__main__':main()
