"""Convert the pinned DFlash2 checkpoint without dropping trained tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from prepare_dflash2 import OUT, WEIGHT_SHA, sha

MOBI = Path(__file__).resolve().parents[2] / 'MobiSpec.cpp'
sys.path.insert(0, str(MOBI / 'gguf-py'))
import gguf


def tensor_name(name, mapper):
    explicit = {
        'fc.weight': 'fc.weight', 'hidden_norm.weight': 'enc.output_norm.weight',
        'embed_tokens.weight': 'token_embd.weight', 'lm_head.weight': 'output.weight',
        'candidate_selector.predecessor_codebook': 'selector_prev.weight',
        'candidate_selector.successor_codebook': 'selector_next.weight',
        'candidate_selector.hidden_projection.weight': 'selector_hidden.weight',
    }
    if name in explicit:
        return explicit[name]
    if name.startswith('layers.'):
        _, layer, tail = name.split('.', 2)
        conv = {'attention_conv.base_kernel': 'attn_conv_base.weight',
                'attention_conv.kernel_projection.weight': 'attn_conv_proj.weight',
                'mlp_conv.base_kernel': 'ffn_conv_base.weight',
                'mlp_conv.kernel_projection.weight': 'ffn_conv_proj.weight'}
        if tail in conv:
            return f'blk.{layer}.{conv[tail]}'
    result = mapper.get_name('model.' + name, try_suffixes=('.weight', '.bias'))
    if not result:
        raise KeyError('Unmapped tensor: ' + name)
    return result


def add_metadata(writer, cfg, template, dtype):
    skip = {'general.architecture', 'general.name', 'general.finetune', 'general.file_type',
            'dflash.block_size', 'dflash.anchor_first', 'dflash.target_layers',
            'dflash.attention.sliding_window', 'dflash.attention.sliding_window_pattern'}
    for name, field in template.fields.items():
        if name.startswith('GGUF.') or name in skip:
            continue
        writer.add_key_value(name, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.add_name('RedHatAI Qwen3-4B DFlash2')
    writer.add_uint32('general.file_type', 0 if dtype == 'f32' else 2)
    for key in ('block_size', 'conv_kernel_size', 'conv_group_size', 'selector_rank', 'selector_top_k'):
        writer.add_uint32('dflash.' + key, cfg[key])
    writer.add_uint32('dflash.anchor_first', 0)
    writer.add_array('dflash.target_layers', [x + 1 for x in cfg['aux_hidden_state_layer_ids']])
    writer.add_uint32('dflash.attention.sliding_window', cfg['transformer_layer_config']['sliding_window'])
    writer.add_array('dflash.attention.sliding_window_pattern', [True] * 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dtype', choices=['f32', 'q4_0'], required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    ckpt = OUT / 'checkpoint'
    assert sha(ckpt / 'model.safetensors') == WEIGHT_SHA
    cfg = json.loads((ckpt / 'config.json').read_text())
    sd = load_file(str(ckpt / 'model.safetensors'))
    mapper = gguf.get_tensor_name_map(gguf.MODEL_ARCH.DFLASH, 5)
    template_path = MOBI.parent / 'specinfer_QNN/experiments/vlm-dflash-20260913/DSpark-device-emb.gguf'
    template = gguf.GGUFReader(template_path)
    out = OUT / 'gpu'; out.mkdir(exist_ok=True)
    path = out / f'dflash2-{args.dtype}.gguf'
    if path.exists():
        raise FileExistsError(path)
    writer = gguf.GGUFWriter(path, 'dflash', use_temp_file=True)
    add_metadata(writer, cfg, template, args.dtype)
    records = []; names = set()
    for name, t in sd.items():
        dest = tensor_name(name, mapper)
        assert dest not in names, dest
        names.add(dest)
        value = t.float().numpy()
        kind = gguf.GGMLQuantizationType.F32
        if dest.startswith(('selector_prev', 'selector_next')):
            kind = gguf.GGMLQuantizationType.BF16
        elif args.dtype == 'q4_0' and value.ndim == 2:
            kind = gguf.GGMLQuantizationType.F16 if 'conv_proj' in dest or dest == 'selector_hidden.weight' else gguf.GGMLQuantizationType.Q4_0
        data = gguf.quants.quantize(value, kind)
        writer.add_tensor(dest, data, raw_dtype=kind)
        records.append(dict(source=name, output=dest, shape=list(t.shape), dtype=kind.name,
                            packed_sha256=hashlib.sha256(data.tobytes()).hexdigest()))
    assert len(records) == 83
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    check = gguf.GGUFReader(path)
    for tensor, record in zip(check.tensors, records):
        assert tensor.name == record['output']
        assert hashlib.sha256(tensor.data.tobytes()).hexdigest() == record['packed_sha256']
    manifest = dict(checkpoint_sha256=WEIGHT_SHA, config_sha256=sha(ckpt / 'config.json'),
                    source_sha256=sha(__file__), template_sha256=sha(template_path),
                    output_sha256=sha(path), tensors=records,
                    scope='All 83 trained tensors mapped; original B9 and non-causal SWA contract. '
                          'GPU quantization follows existing Q4_0 body/head; new conv projections and selector gate use F16.')
    path.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('DFLASH2_GGUF_READY', path, path.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
