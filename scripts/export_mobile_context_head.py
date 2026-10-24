#!/usr/bin/env python3
"""Export a frozen CPU context head; no target/draft NPU graph is modified."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert model['schema'] == 1 and model['mode'] in ['compact', 'full', 'pca32', 'pca128']
    assert model['hidden'] == 16 and model['rank'] == 256 and 0 < model['clip'] <= 8
    original_mode = model['mode']
    if original_mode.startswith('pca'):
        # Fold the learned training-only projection into the deployed first layer.
        components = int(original_mode[3:]); projection = model['projection'].double()
        assert projection.shape == (12800, components) and torch.isfinite(projection).all()
        assert model['input_dim'] == components+261 and model['raw_input_dim'] == 13061
        mean, std = model['mean'].double(), model['std'].double()
        assert torch.isfinite(mean).all() and torch.isfinite(std).all() and torch.all(std > 0)
        state = model['state_dict']; weight = state['0.weight'].double()
        assert weight.shape == (16, components+261+7)
        scaled = weight[:, :components+261]/std
        folded = torch.cat([scaled[:, :components]@projection.T, scaled[:, components:], weight[:, -7:]], dim=1)
        bias = state['0.bias'].double()-scaled@mean
        state = dict(state, **{'0.weight': folded.float(), '0.bias': bias.float()})
        model = dict(model, mode='full', input_dim=13061, mean=torch.zeros(13061), std=torch.ones(13061), state_dict=state)
    mode = int(model['mode'] == 'full'); size = model['input_dim']; dim = 2560
    assert size == (5*dim+5+256 if mode else 85+256)
    state = model['state_dict']
    tensors = [model['mean'], model['std'], model['inverse_basis_rms'],
               state['0.weight'], state['0.bias'], state['2.weight'], state['2.bias']]
    shapes = [(size,), (size,), (256,), (16, size+7), (16,), (256, 16), (256,)]
    assert all(tuple(t.shape) == shape and torch.isfinite(t).all() for t, shape in zip(tensors, shapes))
    assert torch.all(model['std'] > 0) and torch.all(model['inverse_basis_rms'] > 0)
    payload = struct.pack('<8if', 0x4843544d, 1, mode, size, 16, 256, 5, dim, model['clip'])
    payload += b''.join(t.detach().numpy().astype('<f4').tobytes() for t in tensors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError('Refusing to overwrite a different deployment artifact')
    args.output.write_bytes(payload)
    provenance = dict(format='MTCH_V1', backend='CPU context head', mode=model['mode'], training_mode=original_mode,
                      checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      output_sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                      trainable_parameters=sum(t.numel() for t in tensors[3:]))
    args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(args.output, provenance['output_sha256'])


if __name__ == '__main__':
    main()
