#!/usr/bin/env python3
"""Extract a variable-block draft context with explicit physical/logical geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from executorch.backends.qualcomm.python import PyQnnManagerAdaptor
from executorch.exir._serialize._program import deserialize_pte_binary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('artifact',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--width',type=int,choices=[2560,151936],required=True)
    parser.add_argument('--params',type=Path,required=True)
    parser.add_argument('--resume-incomplete',action='store_true')
    parser.add_argument('--physical-block',type=int,required=True)
    parser.add_argument('--logical-block',type=int,required=True)
    parser.add_argument('--inject-rows',type=int,default=32)
    parser.add_argument('--layers',type=int,default=5)
    parser.add_argument('--inject-only',action='store_true',help='inject-only draft-prefill graph (M4): no KV cache inputs; K/V outputs for --inject-rows committed rows, primary [1, block, 2560]')
    args=parser.parse_args()
    assert args.physical_block == args.logical_block + 1 and args.logical_block > 0
    if args.output.exists() and (not args.resume_incomplete or (args.output/'io-contract.json').exists()):
        raise FileExistsError(args.output)
    pte=args.artifact/'hybrid_llama_qnn.pte'
    program=deserialize_pte_binary(pte.read_bytes()).program
    plans=[p for p in program.execution_plan if p.name=='prefill_forward']
    assert len(plans)==1 and len(plans[0].delegates)==1
    delegate=plans[0].delegates[0];assert delegate.id=='QnnBackend'
    binary=PyQnnManagerAdaptor.StripProtocol(program.backend_delegate_data[delegate.processed.index].data)
    assert len(binary)>0
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'forward_0.bin'
    if path.exists(): assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(binary).digest()
    else: path.write_bytes(binary)
    sdk=Path(os.environ['QNN_SDK_ROOT'])
    dest=args.output/'forward_0_json.json'
    subprocess.run([str(sdk/'bin/x86_64-linux-clang/qnn-context-binary-utility'),
        '--context_binary',str(path),'--json_file',str(dest)],check=True,
        env=dict(os.environ,LD_LIBRARY_PATH=str(sdk/'lib/x86_64-linux-clang')))
    info=json.loads(dest.read_text())
    graphs=[g['info'] for g in info['info']['graphs'] if g['info']['graphName']=='prefill_forward']
    assert len(graphs)==1;graph=graphs[0]
    inputs=[t['info'] for t in graph['graphInputs']];outputs=[t['info'] for t in graph['graphOutputs']]
    dims=lambda t:t.get('dimensions',t.get('currentDimensions'))
    primary=[t for t in outputs if dims(t)==[1,args.physical_block,args.width]]
    assert len(primary)==1
    if args.width==2560:
        assert primary[0]['dataType']=='QNN_DATATYPE_FLOAT_32'
    else:
        assert any(s in primary[0]['name'] for s in ['squeeze','logit'])
    caches=[t for t in inputs if 'args_' in t['name']]
    if args.inject_only:
        assert not caches, [t['name'] for t in caches]
        hidden=[t for t in inputs if dims(t)==[1,args.inject_rows,2560]]; ctx_pos=[t for t in inputs if 'input_pos_ctx' in t['name']]
        assert len(hidden)==args.layers and len(ctx_pos)==1 and dims(ctx_pos[0])==[1,args.inject_rows], (len(hidden), ctx_pos)
        kout=[t for t in outputs if dims(t)==[1,8,128,args.inject_rows]]; vout=[t for t in outputs if dims(t)==[1,8,args.inject_rows,128]]
        assert len(kout)==args.layers and len(vout)==args.layers and all(t['dataType']=='QNN_DATATYPE_FLOAT_32' for t in kout+vout), (len(kout), len(vout))
        assert any('atten_mask' in t['name'].lower() for t in inputs), [t['name'] for t in inputs]
    else:
        assert args.layers>0 and len(caches)==2*args.layers and all('8' in str(t['dataType']) for t in caches)
    params=json.loads(args.params.read_text())
    assert params['dim']==2560 and params['vocab_size']==151936 and params['n_layers']==args.layers
    params['draft_output_kind']='normalized_hidden' if args.width==2560 else 'logits'
    params['draft_output_width']=args.width
    params['draft_physical_block']=args.physical_block
    params['draft_logical_block']=args.logical_block
    params['draft_inject_rows']=args.inject_rows
    params['draft_graph_kind']='inject' if args.inject_only else 'step'
    (args.output/'params.json').write_text(json.dumps(params,indent=2)+'\n')
    result=dict(width=args.width,physical_block=args.physical_block,logical_block=args.logical_block,inject_rows=args.inject_rows,graph_kind='inject' if args.inject_only else 'step',bytes=len(binary),graph=graph['graphName'],primary=primary,
        inputs=inputs,outputs=outputs,source_pte_sha256=hashlib.sha256(pte.read_bytes()).hexdigest(),
        context_sha256=hashlib.sha256(binary).hexdigest(),
        source_params_sha256=hashlib.sha256(args.params.read_bytes()).hexdigest(),
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'io-contract.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['width','bytes','graph','primary']},indent=2))


if __name__=='__main__':main()
