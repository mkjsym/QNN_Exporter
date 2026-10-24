"""Original Python forward using exact dequantized GPU GGUF values."""
import json
from pathlib import Path
import sys
import torch
from export_dflash2_gpu import OUT,MOBI,gguf,sha
sys.path.insert(0,str(MOBI/'tools/research'))
import reference_dflash2 as reference

def load_dequantized(_):
    path=OUT/'gpu/dflash2-q4_0.gguf'
    manifest=json.loads(path.with_suffix('.json').read_text())
    assert sha(path)==manifest['output_sha256']
    reader=gguf.GGUFReader(path)
    records={r['output']:r for r in manifest['tensors']}
    state={}
    for t in reader.tensors:
        r=records[t.name]
        values=gguf.quants.dequantize(t.data,t.tensor_type).reshape(r['shape'])
        state[r['source']]=torch.from_numpy(values.copy()).float()
    assert len(state)==83
    return state

if __name__=='__main__':
    reference.load_file=load_dequantized
    sys.argv += ['--output-label','reference-gpu-q4']
    reference.main()
    p=reference.RAW/'reference-gpu-q4/manifest.json'
    result=json.loads(p.read_text());result.update(gguf_sha256=sha(OUT/'gpu/dflash2-q4_0.gguf'),
        dequantization_source_sha256=sha(__file__),dtype='FP32 original Python operators on exact GGUF dequantized weights; not CPU ggml Q8 activation dot kernels')
    p.write_text(json.dumps(result,indent=2)+'\n')
