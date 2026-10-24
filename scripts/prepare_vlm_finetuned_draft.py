"""Prepare a full-context VLM-finetuned DFlash checkpoint and train-only calibration."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
MOBI = ROOT.parent/'MobiSpec.cpp'
sys.path.insert(0, str(MOBI/'tools/training'))
import train_vlm_learned8 as base
from vlm_learned8 import FeatureTeacher, TAPS, atomic_json, sha256
from vlm_visual_followup import FOLLOWUP, provenance

OUT = ROOT/'artifacts/vlm-ft-20260929'


def capture():
    output = OUT/'calibration'
    output.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(output).free > 50*(1 << 30)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.65, 0)
    cache = output/'response-cache'
    cache.symlink_to(base.STUDY/'responses', target_is_directory=True)
    teacher = FeatureTeacher(base.TARGET, 'cuda:0', cache)
    rows = base.rows('train')
    chosen = [r for domain, n in [('general', 6), ('ocr', 6), ('text', 4)]
              for r in [item for item in rows if item['domain'] == domain][:n]]
    assert len(chosen) == 16
    records, hidden = [], []
    for item in chosen:
        source = cache/(item['id']+'.json')
        assert source.exists()
        digest = sha256(source)
        teacher.model.model.rope_deltas = None
        example = teacher.example(item)
        assert sha256(source) == digest
        assert example['features'].shape[1] == 12800
        assert torch.isfinite(example['features']).all()
        path = output/(item['id']+'.pt')
        torch.save(dict(case=dict(item, split='train'), layer_ids=list(TAPS),
                        target_features=example['features'].reshape(-1, 5, 2560),
                        input_ids=example['tokens'], prompt_length=example['prompt_length'],
                        image_positions=example['image_positions'], grid_hw=example['grid_hw']), path)
        # Include image, prompt text and generated-response contexts, only from training inputs.
        hidden.append(example['features'])
        records.append(dict(path=str(path), sha256=sha256(path), id=item['id'], domain=item['domain'],
                            response_sha256=digest, rows=len(example['tokens']),
                            prompt_length=example['prompt_length'], image_tokens=len(example['image_positions'])))
        print('CAPTURE', item['id'], records[-1]['rows'], flush=True)
    torch.save(dict(hidden=torch.cat(hidden).float()), output/'calibration-hidden.pt')
    prompts = [dict(messages=[dict(role='user', content=item['prompt'])]) for item in chosen]
    atomic_json(output/'calibration-prompts.json', prompts)
    atomic_json(output/'manifest.json', dict(completed_utc=base.now(), records=records,
                source_hashes=provenance(Path(__file__)), teacher='BF16 frozen training teacher; real visual DeepStack features',
                split='train only; no validation or heldout features',
                outputs={name:sha256(output/name) for name in ['calibration-hidden.pt', 'calibration-prompts.json']}))


def convert(arm):
    mode, seed = arm.split('_seed')
    assert mode == 'full' and int(seed) in [42, 43]
    checkpoint = FOLLOWUP/f'full-seed{seed}/best.safetensors'
    complete = json.loads(checkpoint.with_name('training-complete.json').read_text())
    assert complete['step'] == 250 and complete['seen'] == 4000
    output = OUT/arm
    output.mkdir(parents=True, exist_ok=True)
    assert not list(output.iterdir()), 'Refuse to overwrite conversion outputs'
    assert shutil.disk_usage(output).free > 60*(1 << 30)
    torch.set_num_threads(4)
    et = Path(os.environ['EXECUTORCH_ROOT']).resolve()
    sys.path.insert(0, str(et.parent))
    from executorch.examples.models.qwen3_dflash.convert_weights import dflash_hf_to_meta
    source = load_file(str(checkpoint))
    assert torch.count_nonzero(source.pop('visual_queries')) == 0
    normal = {key.removeprefix('model.'):value for key, value in source.items()}
    assert all(key.startswith('model.') for key in source)
    embedding_index = json.loads((base.TEXT_TARGET/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(base.TEXT_TARGET/embedding_index['model.embed_tokens.weight'], framework='pt') as tensor_file:
        embedding = tensor_file.get_tensor('model.embed_tokens.weight').float()
    normal['embed_tokens.weight'] = embedding
    normal['lm_head.weight'] = embedding
    converted = dflash_hf_to_meta(normal)
    # Conversion must preserve every FC column exactly; only the graph summation order changes.
    assert torch.equal(torch.cat([converted[f'fc.{i}.weight'] for i in range(5)], 1), normal['fc.weight'])
    assert all(torch.isfinite(tensor).all() for tensor in converted.values())
    torch.save(converted, output/'draft-meta.pth')
    embedding.half().numpy().tofile(output/'embedding-f16.bin')
    atomic_json(output/'manifest.json', dict(completed_utc=base.now(), arm=arm,
                checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                converter_sha256=sha256(et/'examples/models/qwen3_dflash/convert_weights.py'),
                source_hashes=provenance(Path(__file__)),
                weights='FP32 trained masters; frozen original Qwen3-4B embedding/head; full visual context',
                fc_reconstruction_exact=True, target_layer_ids=list(TAPS), graph_keys=sorted(converted),
                outputs={name:sha256(output/name) for name in ['draft-meta.pth', 'embedding-f16.bin']}))
    print('ASSETS_READY', arm, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['capture', 'convert'])
    parser.add_argument('--arm', default='full_seed42')
    args = parser.parse_args()
    if args.action == 'capture':
        capture()
    else:
        convert(args.arm)
