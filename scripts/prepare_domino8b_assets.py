#!/usr/bin/env python3
"""Prepare native Domino graph weights and its unmodified host correction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from prepare_long_draft_assets import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'target', 'output', 'gguf-python']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    et = Path(os.environ['EXECUTORCH_ROOT']).resolve()
    sys.path.insert(0, str(et.parent))
    sys.path.insert(0, str(args.gguf_python.resolve()))
    from executorch.examples.models.qwen3_dflash.convert_weights import dflash_hf_to_meta
    import gguf
    weights = load_file(str(args.checkpoint / 'model.safetensors'))
    cfg = json.loads((args.checkpoint / 'config.json').read_text())
    target_cfg = json.loads((args.target / 'config.json').read_text())
    assert cfg['block_size'] == 16 and cfg['hidden_size'] == 4096
    assert not target_cfg['tie_word_embeddings']
    index = json.loads((args.target / 'model.safetensors.index.json').read_text())['weight_map']
    name = 'model.embed_tokens.weight'
    with safe_open(str(args.target / index[name]), framework='pt') as f:
        embedding = f.get_tensor(name)
    with safe_open(str(args.target / index['lm_head.weight']), framework='pt') as f:
        output = f.get_tensor('lm_head.weight')
    backbone = {k: v for k, v in weights.items() if not k.startswith(('prefix_gru.', 'embed_proj.'))}
    backbone.update({'embed_tokens.weight': embedding, 'lm_head.weight': output})
    meta = dflash_hf_to_meta(backbone)
    meta['domino_base.weight'] = weights['embed_proj.0.weight'][:, :4096].contiguous()
    torch.save(meta, args.output / 'draft-meta.pth')
    writer = gguf.GGUFWriter(str(args.output / 'domino-head-f16.gguf'), 'qwen3')
    writer.add_name('Native Domino full-prefix correction')
    for name, value in [('domino_hh.weight', weights['prefix_gru.weight_hh_l0']),
                        ('domino_state.weight', weights['embed_proj.0.weight'][:, 4096:]),
                        ('domino_output.weight', weights['embed_proj.2.weight'])]:
        writer.add_tensor(name, value.half().contiguous().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    emb_sha = hashlib.sha256()
    gi_sha = hashlib.sha256()
    ih = weights['prefix_gru.weight_ih_l0'].float()
    maximum = 0.
    with (args.output / 'domino-gi-f16.bin').open('wb') as f:
        for chunk in embedding.split(1024):
            emb_sha.update(chunk.half().contiguous().numpy().tobytes())
            gi = torch.nn.functional.linear(chunk.float(), ih)
            packed = gi.half()
            assert torch.isfinite(packed).all()
            maximum = max(maximum, float((gi - packed.float()).abs().max()))
            raw = packed.contiguous().numpy().tobytes()
            f.write(raw)
            gi_sha.update(raw)
    manifest = dict(source_sha256=sha(__file__), checkpoint=str(args.checkpoint),
                    checkpoint_sha256=sha(args.checkpoint / 'model.safetensors'),
                    config_sha256=sha(args.checkpoint / 'config.json'),
                    converter_sha256=sha(et / 'examples/models/qwen3_dflash/convert_weights.py'),
                    embedding_raw_f16_sha256=emb_sha.hexdigest(),
                    gi_shape=[151936, 3072], gi_dtype='float16', gi_sha256=gi_sha.hexdigest(),
                    gi_fp16_max_abs_error=maximum,
                    outputs={p.name: sha(p) for p in args.output.iterdir()},
                    scope='Native public Domino baseline, not a new algorithm. Input-to-GRU projection '
                          'is precomputed from original weights and rounded to FP16; recurrent/state '
                          'correction weights retain checkpoint values in FP16 storage. Static base '
                          'projection is exported alongside full vocabulary logits.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
