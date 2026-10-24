"""Fetch the pinned serving checkpoint and record its complete tensor inventory."""
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
REPO = 'RedHatAI/Qwen3-4B-speculator.dflash2'
REVISION = '27aa911ddb040c61adeeade08e92949e24c9e17b'
WEIGHT_SHA = '7f3dc8c73f86489bf8c4d65ed4b0139eb1be3f115c7cd2e5779e4835f5908dfe'
OUT = ROOT / 'artifacts/dflash2-20260919'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    physical = Path('/data/youngmin/mobispec-export/dflash2-20260919')
    physical.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.symlink_to(physical, target_is_directory=True)
    assert OUT.resolve() == physical.resolve()
    if shutil.disk_usage(physical).free < 12 * (1 << 30):
        raise RuntimeError('Insufficient host disk reserve')
    ckpt = OUT / 'checkpoint'
    ckpt.mkdir(exist_ok=True)
    for name in ('config.json', 'README.md', 'model.safetensors'):
        print('FETCH', name, flush=True)
        hf_hub_download(REPO, name, revision=REVISION, local_dir=ckpt)
    cfg = json.loads((ckpt / 'config.json').read_text())
    assert cfg['speculators_config']['verifier']['name_or_path'] == 'Qwen/Qwen3-4B'
    assert cfg['block_size'] == 9 and not cfg['sample_from_anchor']
    assert cfg['aux_hidden_state_layer_ids'] == [1, 9, 17, 25, 33]
    assert sha(ckpt / 'model.safetensors') == WEIGHT_SHA
    tensors = []
    with safe_open(ckpt / 'model.safetensors', framework='pt', device='cpu') as f:
        for key in f.keys():
            t = f.get_slice(key)
            tensors.append(dict(name=key, shape=t.get_shape(), dtype=t.get_dtype()))
    result = dict(repo=REPO, revision=REVISION, source_sha256=sha(__file__),
                  files={n:sha(ckpt / n) for n in ('config.json', 'README.md', 'model.safetensors')},
                  tensors=tensors, weight_hash_verified=True)
    (OUT / 'checkpoint-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('CHECKPOINT_READY', len(tensors), flush=True)


if __name__ == '__main__':
    main()
