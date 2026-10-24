"""Convert official vision sample inputs using the exported QNN tensor ABI."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARTIFACT=ROOT/'artifacts/vlm-vision-backends-20261008/official'
info=json.loads((ARTIFACT/'vision-info.json').read_text())
def tensors(value):
    if isinstance(value,dict):
        if 'name' in value and 'dataType' in value and 'dimensions' in value:yield value
        else:
            for item in value.values():yield from tensors(item)
    elif isinstance(value,list):
        for item in value:yield from tensors(item)
target=ARTIFACT/'inputs';target.mkdir(exist_ok=True)
manifest={}
for t in tensors(info):
    if t['type']!='QNN_TENSOR_TYPE_APP_WRITE':continue
    name=t['name'];values=np.fromfile(ARTIFACT/(name+'.raw'),'<f4')
    assert values.size==int(np.prod(t['dimensions']))
    dtype=t['dataType']
    if dtype in ['QNN_DATATYPE_UFIXED_POINT_8','QNN_DATATYPE_UFIXED_POINT_16']:
        quant=t['quantizeParams'];assert quant['quantizationEncoding']=='QNN_QUANTIZATION_ENCODING_SCALE_OFFSET'
        scale=quant['scaleOffset']['scale'];offset=quant['scaleOffset']['offset'];bits=8 if dtype.endswith('_8') else 16
        encoded=np.clip(np.rint(values/np.float32(scale)-np.float32(offset)),0,2**bits-1).astype('u1' if bits==8 else '<u2')
    elif dtype=='QNN_DATATYPE_FLOAT_16':encoded=values.astype('<f2')
    elif dtype=='QNN_DATATYPE_FLOAT_32':encoded=values.astype('<f4')
    else:raise ValueError(dtype)
    path=target/(name+'.bin');encoded.tofile(path)
    manifest[name]=dict(dimensions=t['dimensions'],data_type=dtype,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
(target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('VISION_CONTEXT_INPUTS_READY')
