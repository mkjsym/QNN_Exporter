"""Extract and validate the shared ten-query W4 draft context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from executorch.backends.qualcomm.python import PyQnnManagerAdaptor
from executorch.exir._serialize._program import deserialize_pte_binary


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--params', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    pte = a.artifact/'hybrid_llama_qnn.pte'
    program = deserialize_pte_binary(pte.read_bytes()).program
    plans = [x for x in program.execution_plan if x.name == 'prefill_forward']
    assert len(plans) == 1 and len(plans[0].delegates) == 1
    delegate = plans[0].delegates[0]
    assert delegate.id == 'QnnBackend'
    binary = PyQnnManagerAdaptor.StripProtocol(program.backend_delegate_data[delegate.processed.index].data)
    assert len(binary) > 0
    a.output.mkdir(parents=True)
    context = a.output/'forward_0.bin'
    context.write_bytes(binary)
    sdk = Path(os.environ['QNN_SDK_ROOT'])
    metadata = a.output/'forward_0_json.json'
    subprocess.run([str(sdk/'bin/x86_64-linux-clang/qnn-context-binary-utility'),
        '--context_binary', str(context), '--json_file', str(metadata)], check=True,
        env=dict(os.environ, LD_LIBRARY_PATH=str(sdk/'lib/x86_64-linux-clang')))
    graphs = [g['info'] for g in json.loads(metadata.read_text())['info']['graphs']
              if g['info']['graphName'] == 'prefill_forward']
    assert len(graphs) == 1
    graph = graphs[0]
    inputs = [x['info'] for x in graph['graphInputs']]
    outputs = [x['info'] for x in graph['graphOutputs']]
    dims = lambda x: x.get('dimensions', x.get('currentDimensions'))
    primary = [x for x in outputs if dims(x) == [1, 10, 151936]]
    assert len(primary) == 1 and primary[0]['dataType'] == 'QNN_DATATYPE_UFIXED_POINT_16'
    hidden = [x for x in inputs if dims(x) == [1, 16, 2560]]
    assert len(hidden) == 5 and all(x['dataType'] == 'QNN_DATATYPE_FLOAT_32' for x in hidden)
    mask = [x for x in inputs if dims(x) == [1, 1, 10, 1024]]
    assert len(mask) == 1 and mask[0]['dataType'] == 'QNN_DATATYPE_UFIXED_POINT_16'
    caches = [x for x in inputs if 'args_' in x['name']]
    assert len(caches) == 10 and all(x['dataType'] == 'QNN_DATATYPE_UFIXED_POINT_8' for x in caches)
    assert sum(dims(x) == [1, 8, 128, 998] for x in caches) == 5
    assert sum(dims(x) == [1, 8, 998, 128] for x in caches) == 5
    kvout = [x for x in outputs if x not in primary]
    assert len(kvout) == 10 and all(x['dataType'] == 'QNN_DATATYPE_FLOAT_32' for x in kvout)
    assert sum(dims(x) == [1, 8, 128, 16] for x in kvout) == 5
    assert sum(dims(x) == [1, 8, 16, 128] for x in kvout) == 5
    params = json.loads(a.params.read_text())
    assert (params['dim'], params['vocab_size'], params['n_layers']) == (2560, 151936, 5)
    assert params['draft_output_kind'] == 'logits' and params['draft_output_width'] == 151936
    (a.output/'params.json').write_bytes(a.params.read_bytes())
    contract = dict(queries=10, injection_rows=16, cache_rows=998, graph=graph['graphName'],
        bytes=len(binary), source_pte_sha256=sha(pte), source_params_sha256=sha(a.params),
        extractor_sha256=sha(__file__), context_sha256=sha(context), inputs=inputs, outputs=outputs,
        scope='One shared W4/A16 artifact for dense, shield, readonly and core7 policies; no auxiliary outputs.')
    (a.output/'io-contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(json.dumps({k: contract[k] for k in ['queries', 'injection_rows', 'bytes', 'context_sha256']}))


if __name__ == '__main__':
    main()
