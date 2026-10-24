"""Export the unchanged public B16 checkpoint using the established GPU tensor types."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
MOBI = ROOT.parent / 'MobiSpec.cpp'
sys.path.insert(0, str(MOBI / 'gguf-py'))
import gguf


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    old = ROOT / 'artifacts/long-draft-20260917/assets-b16-v1/manifest.json'
    provenance = json.loads(old.read_text()); ck = Path(provenance['checkpoint'])
    assert sha(ck / 'model.safetensors') == provenance['checkpoint_sha256']
    cfg = json.loads((ck / 'config.json').read_text())
    assert cfg['block_size'] == 16 and cfg['num_hidden_layers'] == 5
    template = ROOT.parent / 'specinfer_QNN/experiments/vlm-dflash-20260913/DSpark-device-emb.gguf'
    out = ROOT / 'artifacts/commit-pipeline-20260919/gpu-v1'; out.mkdir(parents=True, exist_ok=True)
    target = out / 'dspark-b16-q4_0.gguf'
    reader = gguf.GGUFReader(template)
    writer = gguf.GGUFWriter(target, 'dflash', use_temp_file=True)
    for name, f in reader.fields.items():
        if name.startswith('GGUF.') or name == 'general.architecture': continue
        value = f.contents()
        if name == 'dflash.block_size': value = 16
        if name in ('general.name', 'general.finetune'): value = 'Qwen3-4B-DSpark-b16'
        writer.add_key_value(name, value, f.types[0], f.types[1] if len(f.types) > 1 else None)
    tensors = load_file(str(ck / 'model.safetensors'))
    mapper = gguf.get_tensor_name_map(gguf.MODEL_ARCH.DFLASH, 5)
    named = {}
    for name, tensor in tensors.items():
        if name == 'fc.weight': dest = name
        elif name == 'hidden_norm.weight': dest = 'enc.output_norm.weight'
        elif name == 'embed_tokens.weight': dest = 'token_embd.weight'
        elif name == 'lm_head.weight': dest = 'output.weight'
        else: dest = mapper.get_name(name if name.startswith('model.') or name == 'lm_head.weight' else 'model.'+name, try_suffixes=('.weight', '.bias'))
        assert dest, name
        assert dest not in named, dest
        named[dest] = (name, tensor)
    assert set(named) == {t.name for t in reader.tensors}, set(named) ^ {t.name for t in reader.tensors}
    records = []
    for t in reader.tensors:
        name, tensor = named[t.name]
        value = tensor.float().numpy()
        if value.ndim == 2 and value.shape[0] == 1 and len(t.shape) == 1: value = value[0]
        assert list(value.shape) == t.shape.tolist()[::-1], name
        data = gguf.quants.quantize(value, t.tensor_type)
        writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
        records.append(dict(source=name, output=t.name, shape=list(value.shape), dtype=t.tensor_type.name,
                            packed_sha256=hashlib.sha256(data.tobytes()).hexdigest()))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    check = gguf.GGUFReader(target)
    assert check.fields['dflash.block_size'].contents() == 16
    for t, r in zip(check.tensors, records):
        assert t.name == r['output'] and hashlib.sha256(t.data.tobytes()).hexdigest() == r['packed_sha256']
    manifest = dict(checkpoint=str(ck), checkpoint_sha256=sha(ck / 'model.safetensors'),
        config_sha256=sha(ck / 'config.json'), template_sha256=sha(template), source_sha256=sha(__file__),
        output_sha256=sha(target), tensors=records, trained=False,
        scope='Full original B16 backbone, original embedding and full LM head. Established Q4_0 GPU '
              'body/head/embedding, Q8_0 Markov, FP32 norms. Different quantizer from QNN W4; no new quantization algorithm.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('GPU_EXPORT_COMPLETE', target, target.stat().st_size, flush=True)


if __name__ == '__main__': main()
