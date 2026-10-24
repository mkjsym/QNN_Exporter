#!/usr/bin/env python3
"""Fold a normalized-context bridge into the original five-tap draft interface."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'adapter', 'output', 'gguf-python']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(args.gguf_python.resolve()))
    import gguf
    state = load_file(str(args.checkpoint / 'model.safetensors'), device='cuda')
    model = torch.load(args.adapter, map_location='cuda', weights_only=False)
    assert model['mode'] == 'full' and 'left' in model
    fc = state['fc.weight'].float()
    gamma = state['hidden_norm.weight'].float()
    assert fc.shape == (2560, 12800) and torch.all(gamma.abs() > .001)
    gram = fc @ fc.T
    inverse = torch.linalg.solve(gram, fc)
    eye = torch.eye(2560, device='cuda')
    preimage_error = float((inverse @ fc.T - eye).abs().max())
    assert preimage_error < 1e-3, preimage_error
    left = model['left'] / model['std'][:, None]
    latent_bias = -model['mean'] @ left
    raw_right = (model['right'] / gamma[None]) @ inverse
    raw_bias = (model['bias'] / gamma) @ inverse
    # Preserve the actual committed raw feature vector and add a preimage of the
    # predicted normalized-context innovation, scaled by committed FC RMS.
    assets = {'frontier_left.weight': left.T.contiguous(),
              'frontier_right.weight': raw_right.T.contiguous(),
              'frontier_latent_bias.weight': latent_bias[None],
              'frontier_raw_bias.weight': raw_bias[None],
              'frontier_gamma.weight': gamma[None]}
    writer = gguf.GGUFWriter(str(args.output / 'frontier-bridge-f32.gguf'), 'qwen3')
    writer.add_name('Frozen DSpark frontier target-context bridge')
    for name, value in assets.items():
        assert torch.isfinite(value).all()
        writer.add_tensor(name, value.float().cpu().contiguous().numpy())
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    generator = torch.Generator(device='cuda').manual_seed(260917)
    x = torch.randn(32, left.shape[0], generator=generator, device='cuda')
    expected = (((x-model['mean'])/model['std'])@model['left'])@model['right']+model['bias']
    actual = (((x@left+latent_bias)@raw_right+raw_bias)@fc.T)*gamma
    error = float((actual-expected).abs().max())
    relative = float((actual-expected).square().mean().sqrt()/expected.square().mean().sqrt())
    assert relative < 1e-3, relative
    manifest = dict(adapter=str(args.adapter), adapter_sha256=sha(args.adapter), source_sha256=sha(__file__),
                    checkpoint_sha256=sha(args.checkpoint/'model.safetensors'), rank=model['rank'],
                    shapes={k: list(v.shape) for k,v in assets.items()}, dtype='float32',
                    preimage_identity_max_error=preimage_error, folded_projection_max_error=error,
                    folded_projection_relative_rms_error=relative,
                    outputs={p.name: sha(p) for p in args.output.iterdir()},
                    scope='No draft or target weight changes. FP32 bridge artifact. FC preimage is an algebraic '
                          'deployment transform; actual QNN quantization and proposed raw-feature distribution '
                          'require separate on-device validation. Synthetic projected-context algebra test only.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
