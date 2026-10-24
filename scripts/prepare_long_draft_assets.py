#!/usr/bin/env python3
"""Prepare one frozen DSpark checkpoint and train-only QNN calibration assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'teachers', 'output', 'gguf-python']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    et = Path(os.environ['EXECUTORCH_ROOT']).resolve()
    sys.path.insert(0, str(et.parent))
    sys.path.insert(0, str(args.gguf_python.resolve()))
    from executorch.examples.models.qwen3_dflash.convert_weights import dflash_hf_to_meta
    import gguf
    weights = load_file(str(args.checkpoint / 'model.safetensors'))
    config = json.loads((args.checkpoint / 'config.json').read_text())
    assert config['block_size'] == 16 and config['hidden_size'] == 2560
    meta = dflash_hf_to_meta(weights)
    torch.save(meta, args.output / 'draft-meta.pth')
    writer = gguf.GGUFWriter(str(args.output / 'markov-head-f16.gguf'), 'qwen3')
    writer.add_name('DSpark b16 Markov head only')
    for name in ['markov_w1', 'markov_w2']:
        value = weights['markov_head.' + name + '.weight'].half().contiguous()
        assert torch.isfinite(value).all()
        writer.add_tensor(name + '.weight', value.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    embedding_sha = hashlib.sha256()
    for chunk in weights['embed_tokens.weight'].split(1024):
        embedding_sha.update(chunk.half().contiguous().numpy().tobytes())
    hidden, cases, inputs = [], [], {}
    paths = sorted(args.teachers.glob('train-*.pt'))
    assert len(paths) >= 24
    for path in paths:
        item = torch.load(path, map_location='cpu', weights_only=False)
        assert item['case']['split'] == 'train'
        features = item['target_features']
        pick = torch.linspace(0, len(features) - 1, min(64, len(features))).round().long()
        hidden.append(features[pick].reshape(len(pick), -1).float())
        cases.append(dict(messages=[dict(role='user', content=item['case']['prompt'])]))
        inputs[str(path)] = sha(path)
    hidden = torch.cat(hidden)
    assert hidden.shape[1] == 12800 and torch.isfinite(hidden).all()
    torch.save(dict(hidden=hidden, source='Actual QNN train-prefix captures; BF16 cache converted to FP32'), args.output / 'calibration-hidden.pt')
    (args.output / 'calibration-prompts.json').write_text(json.dumps(cases, indent=2) + '\n')
    manifest = dict(source_sha256=sha(__file__), checkpoint=str(args.checkpoint),
                    checkpoint_sha256=sha(args.checkpoint / 'model.safetensors'),
                    config_sha256=sha(args.checkpoint / 'config.json'),
                    converter_sha256=sha(et / 'examples/models/qwen3_dflash/convert_weights.py'),
                    embedding_raw_f16_sha256=embedding_sha.hexdigest(), hidden_rows=len(hidden),
                    calibration_train_cases=len(cases), calibration_inputs=inputs,
                    outputs={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
                    scope='Unmodified public b16 checkpoint control. Neither a new algorithm nor a performance result. '
                          'No dev/test teacher states enter export calibration.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'calibration_inputs'}, indent=2))


if __name__ == '__main__':
    main()
