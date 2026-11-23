"""Capture new RegSpec register taps on the existing 16 training calibration pages."""
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
MOBI = ROOT.parent/'MobiSpec.cpp'
sys.path.insert(0, str(MOBI/'tools/training'))
os.environ.setdefault('REGISTER_CAP', 'doc20k')
from vlm_register256_paths import STUDY, TARGET
from vlm_register import RegisterTeacher
from vlm_register_mobile_export_common import sha, write

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    old = MOBI/'docs/results/raw/vlm-register-mobile-20261019/calibration/manifest.json'
    source = json.loads(old.read_text())
    assert len(source['records']) == 16
    items = {r['id']:r for r in map(json.loads, (STUDY/'data/train.jsonl').read_text().splitlines())}
    specs = {}
    for arm in ('dynreg8-s1000-seed43', 'dynreg8-deep-taps-s1000-seed43', 'dynreg8-deep-taps-copy024-s1000-seed43'):
        cp = args.bundle/'drafters'/arm/'model.safetensors'
        with safe_open(cp, framework='pt', device='cpu') as f:
            specs[arm] = dict(embeddings=f.get_tensor('register_embeddings').float(),
                prompts=f.get_tensor('register_layer_prompts') if 'register_layer_prompts' in f.keys() else None,
                checkpoint_sha256=sha(cp))
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = RegisterTeacher(str(TARGET), args.device, args.output/'unused-responses', max_tokens=3072)
    records = []
    for r in source['records']:
        assert sha(r['path']) == r['sha256']
        dest = args.output/(r['id']+'.pt')
        if not dest.exists():
            sample = torch.load(r['path'], weights_only=True, mmap=True)
            inputs, _, _ = teacher.inputs(items[r['id']])
            n = int(sample['prompt_length'])
            assert inputs['input_ids'][0].cpu().equal(sample['input_ids'][:n]), r['id']
            outputs, delta, _ = teacher.capture_prefill(inputs)
            cache = outputs.past_key_values
            registers = {}
            for arm, spec in specs.items():
                teacher.layer_prompts = None if spec['prompts'] is None else spec['prompts'].to(device=teacher.device, dtype=torch.bfloat16)
                taps = teacher.register_sets_taps(cache, [n], delta, spec['embeddings'], False)[0]
                assert cache.get_seq_length() == n and taps.shape == (8, 12800)
                registers[arm] = taps.cpu().to(torch.bfloat16)
                teacher.layer_prompts = None
            sample['registers'] = registers
            tmp = dest.with_suffix('.pt.partial')
            torch.save(sample, tmp)
            tmp.replace(dest)
            del outputs, cache
            if args.device.startswith('cuda'): torch.cuda.empty_cache()
        records.append(dict(r, path=str(dest), sha256=sha(dest)))
        print('REGSPEC_CALIB', r['id'], flush=True)
    write(args.output/'manifest.json', dict(records=records, source_manifest_sha256=sha(old),
        source_sha256=sha(__file__), target=str(TARGET),
        register_arms={k:v['checkpoint_sha256'] for k,v in specs.items()},
        calibration_scope='same 16 training pages; new checkpoint register embeddings and layer prompts; no evaluation images'))
    print('REGSPEC_CALIBRATION_COMPLETE', flush=True)

if __name__ == '__main__': main()
