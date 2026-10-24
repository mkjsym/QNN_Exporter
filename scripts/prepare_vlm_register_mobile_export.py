"""Calibration manifest and per-arm weight packing for the register study's mobile drafter exports."""
import argparse, json, os, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from vlm_register_mobile_export_common import ROOT, MOBI, STUDY, CALIB, OUT, TEXT, ARMS, sha, write


def calibration():
    """Reuse the document-page dump as the PTQ calibration set; the two files the exporter's parser wants are derived from it."""
    manifest = json.loads((CALIB/'manifest.json').read_text()); assert len(manifest['records']) == 16
    for r in manifest['records']: assert sha(r['path']) == r['sha256'], r['id']
    dest = OUT/'calibration'; dest.mkdir(parents=True, exist_ok=False)
    hidden = []; prompts = []
    for r in manifest['records']:
        sample = torch.load(r['path'], weights_only=True, mmap=True)
        hidden.append(sample['target_hidden_states'][:int(sample['prompt_length'])].float())
        prompts.append(dict(messages=[dict(role='user', content=f"calibration page {r['id']}")]))
    torch.save(dict(hidden=torch.cat(hidden)), dest/'calibration-hidden.pt')
    write(dest/'calibration-prompts.json', prompts)
    write(dest/'manifest.json', dict(records=manifest['records'], dump_manifest_sha256=sha(CALIB/'manifest.json'), source_sha256=sha(__file__),
                                     outputs={n: sha(dest/n) for n in ['calibration-hidden.pt', 'calibration-prompts.json']}))
    print('REGISTER_MOBILE_CALIBRATION', len(prompts), flush=True)


def convert(arm):
    assert shutil.disk_usage(OUT).free > 50*(1 << 30)
    spec = ARMS[arm]; target = OUT/arm; target.mkdir(parents=True, exist_ok=False)
    complete = json.loads((STUDY/arm/'complete.json').read_text())
    cp = Path(complete['checkpoint'])/'model.safetensors'; assert sha(cp) == complete['model_sha256']
    state = load_file(str(cp)); state.pop('visual_queries', None)
    side = {}
    if spec['mode'] in ('norm8', 'norm32'):
        side['summary.pt'] = {k: state.pop(k).float() for k in ('summary_key.weight', 'summary_condition.weight', 'summary_queries')}
        torch.save(side['summary.pt'], target/'summary.pt')
        torch.cat([side['summary.pt'][k].flatten() for k in ('summary_key.weight', 'summary_condition.weight', 'summary_queries')]).numpy().tofile(target/'summary-f32.bin')
    if spec['mode'] == 'dynreg':
        regs = state.pop('register_embeddings').float(); regs.numpy().tofile(target/'registers-f32.bin')
        if 'register_layer_prompts' in state: state.pop('register_layer_prompts').float().numpy().tofile(target/'layer-prompts-f32.bin')
    weights = {k: v.float() for k, v in state.items()}
    index = json.loads((TEXT/'model.safetensors.index.json').read_text())['weight_map']
    from safetensors import safe_open
    with safe_open(TEXT/index['model.embed_tokens.weight'], framework='pt') as f: embedding = f.get_tensor('model.embed_tokens.weight').float()
    for name in ('embed_tokens.weight', 'lm_head.weight'):
        assert torch.equal(weights[name], embedding), name
        weights[name] = embedding
    et = Path(os.environ['EXECUTORCH_ROOT']); sys.path.insert(0, str(et.parent))
    from executorch.examples.models.qwen3_dflash.convert_weights import dflash_hf_to_meta
    converted = dflash_hf_to_meta(weights)
    assert torch.equal(torch.cat([converted[f'fc.{i}.weight'] for i in range(5)], 1), weights['fc.weight'])
    torch.save(converted, target/'draft-meta.pth'); embedding.half().numpy().tofile(target/'embedding-f16.bin')
    names = ['draft-meta.pth', 'embedding-f16.bin'] + [n for n in ('summary.pt', 'summary-f32.bin', 'registers-f32.bin', 'layer-prompts-f32.bin') if (target/n).exists()]
    write(target/'manifest.json', dict(arm=arm, mode=spec['mode'], spec={k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}, step=complete['step'],
                                       checkpoint=str(cp), checkpoint_sha256=complete['model_sha256'], source_sha256=sha(__file__),
                                       converter_sha256=sha(et/'examples/models/qwen3_dflash/convert_weights.py'),
                                       contract='B7 root+6mask padded to physical B8; open7; original scalar RoPE with physical context packing',
                                       outputs={name: sha(target/name) for name in names}))
    print('REGISTER_MOBILE_CONVERTED', arm, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['calibration', 'convert']); p.add_argument('--arm'); a = p.parse_args()
    calibration() if a.phase == 'calibration' else convert(a.arm)
