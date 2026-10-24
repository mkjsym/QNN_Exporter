"""Extract and check the sparse8 Domino QNN component artifact."""
import json
import os
from pathlib import Path
import subprocess
from extract_horizon_context import sha
from executorch.backends.qualcomm.python import PyQnnManagerAdaptor
from executorch.exir._serialize._program import deserialize_pte_binary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/sparse-domino-cost-20260922'


def main():
    dest = OUT / 'context-v1'; dest.mkdir(exist_ok=True)
    assert not (dest / 'io-contract.json').exists()
    pte = OUT / 'export-v1/hybrid_llama_qnn.pte'
    program = deserialize_pte_binary(pte.read_bytes()).program
    plans = [p for p in program.execution_plan if p.name == 'prefill_forward']
    assert len(plans) == 1 and len(plans[0].delegates) == 1
    delegate = plans[0].delegates[0]; assert delegate.id == 'QnnBackend'
    binary = PyQnnManagerAdaptor.StripProtocol(program.backend_delegate_data[delegate.processed.index].data)
    path = dest / 'forward_0.bin'
    if path.exists():
        import hashlib
        assert sha(path) == hashlib.sha256(binary).hexdigest()
    else:
        path.write_bytes(binary)
    sdk = Path(os.environ['QNN_SDK_ROOT']); meta = dest / 'forward_0_json.json'
    subprocess.run([str(sdk / 'bin/x86_64-linux-clang/qnn-context-binary-utility'), '--context_binary',
        str(path), '--json_file', str(meta)], check=True,
        env=dict(os.environ, LD_LIBRARY_PATH=str(sdk / 'lib/x86_64-linux-clang')))
    all_graphs = [g['info'] for g in json.loads(meta.read_text())['info']['graphs']]
    graphs = [g for g in all_graphs if g['graphName'] == 'prefill_forward']
    assert len(graphs) == 1 and graphs[0]['graphName'] == 'prefill_forward'
    inputs = [t['info'] for t in graphs[0]['graphInputs']]
    outputs = [t['info'] for t in graphs[0]['graphOutputs']]
    dims = lambda t: t.get('dimensions', t.get('currentDimensions'))
    assert len(inputs) == 19 and len(outputs) == 11
    expected = [(5, [1,32,2560], 'FLOAT_32'), (1,[1,8,2560],'FLOAT_32'),
        (1,[1,1,8,1024],'UFIXED_POINT_16'), (1,[1,8],'INT_32'), (1,[1,32],'INT_32'),
        (5,[1,8,128,984],'UFIXED_POINT_8'), (5,[1,8,984,128],'UFIXED_POINT_8')]
    for count, shape, dtype in expected:
        matches = [t for t in inputs if dims(t) == shape]
        assert len(matches) == count, (shape, len(matches))
        assert all(t['dataType'] == 'QNN_DATATYPE_' + dtype for t in matches), matches
    for count, shape in [(1,[1,8,152192]),(5,[1,8,128,32]),(5,[1,8,32,128])]:
        matches = [t for t in outputs if dims(t) == shape]
        assert len(matches) == count and all(t['dataType'] == 'QNN_DATATYPE_FLOAT_32' for t in matches)
    params_source = ROOT / 'artifacts/long-draft-20260917/context-domino-w4-v1/params.json'
    if not params_source.exists():
        candidates = list((ROOT / 'artifacts/long-draft-20260917').glob('**/params.json'))
        candidates = [p for p in candidates if json.loads(p.read_text()).get('draft_output_kind') == 'domino_packed']
        assert len(candidates) == 1, candidates
        params_source = candidates[0]
    params = json.loads(params_source.read_text())
    assert (params['dim'], params['n_layers'], params['vocab_size']) == (2560,5,151936)
    params.update(draft_physical_block=8, draft_logical_block=16, draft_inject_rows=32,
        draft_query_offsets=[0,1,2,3,4,5,6,15])
    (dest / 'params.json').write_text(json.dumps(params, indent=2)+'\n')
    contract = dict(physical_queries=8, logical_horizon=16, query_offsets=params['draft_query_offsets'],
        injection_rows=32, cache_rows=984, inputs=inputs, outputs=outputs, source_pte_sha256=sha(pte),
        context_sha256=sha(path), source_params=str(params_source), source_params_sha256=sha(params_source),
        extractor_sha256=sha(__file__), single_qnn_delegate=True,
        context_graphs=[g['graphName'] for g in all_graphs], measured_graph='prefill_forward')
    (dest / 'io-contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    print('SPARSE_DOMINO_CONTEXT_COMPLETE', sha(path), flush=True)


if __name__ == '__main__':
    main()
