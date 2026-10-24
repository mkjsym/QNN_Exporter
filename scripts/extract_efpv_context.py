"""Extract immutable EFPV contexts, keeping logical head/body outputs explicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from executorch.backends.qualcomm.python import PyQnnManagerAdaptor
from executorch.exir._serialize._program import deserialize_pte_binary


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--shards',type=int,required=True);p.add_argument('--params',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    pte=args.artifact/'hybrid_llama_qnn.pte';program=deserialize_pte_binary(pte.read_bytes()).program
    plan=next(x for x in program.execution_plan if x.name=='prefill_forward')
    assert len(plan.delegates)==args.shards
    args.output.mkdir();sdk=Path(os.environ['QNN_SDK_ROOT']);records=[]
    for i,delegate in enumerate(plan.delegates):
        assert delegate.id=='QnnBackend'
        data=program.backend_delegate_data[delegate.processed.index].data
        binary=PyQnnManagerAdaptor.StripProtocol(data);assert len(binary)>0
        dest=args.output/f'forward_{i}.bin';dest.write_bytes(binary)
        j=args.output/f'forward_{i}_json.json'
        subprocess.run([str(sdk/'bin/x86_64-linux-clang/qnn-context-binary-utility'),
            '--context_binary',str(dest),'--json_file',str(j)],check=True)
        raw=json.loads(j.read_text());graph=next(g['info'] for g in raw['info']['graphs'] if g['info']['graphName']=='prefill_forward')
        outputs=[t['info'] for t in graph['graphOutputs']]
        records.append(dict(shard=i,sha256=sha(dest),bytes=len(binary),outputs=outputs))
    params=json.loads(args.params.read_text());params['efpv_reconstructed_target']=True
    params['efpv_head_split']=args.shards==7
    (args.output/'params.json').write_text(json.dumps(params,indent=2)+'\n')
    contract=dict(pte_sha256=sha(pte),shards=records,extractor_sha256=sha(Path(__file__)),canonical_equivalent=False)
    (args.output/'io-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print('EFPV_EXTRACT_COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
