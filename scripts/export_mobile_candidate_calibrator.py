#!/usr/bin/env python3
"""Export a research candidate-score checkpoint to the mobile CPU runtime format."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.checkpoint.read_text())
    assert model['schema'] == 1 and model['features'] == 8 and model['hidden'] == 16
    assert 0 < model['clip'] <= 8 and len(model['mean']) == len(model['std']) == 8
    assert len(model['first_weight']) == len(model['first_bias']) == len(model['second_weight']) == 16
    assert all(len(row) == 8 for row in model['first_weight'])
    assert all(value > 0 for value in model['std'])
    values = model['mean']+model['std']+sum(model['first_weight'], [])+model['first_bias']+model['second_weight']+[model['second_bias']]
    assert all(math.isfinite(value) for value in values)
    text = f"MOBILE_SCORE_V1 8 16 {model['clip']:.9g}\n"+' '.join(f'{v:.9g}' for v in values)+'\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError('Refusing to overwrite a different deployment artifact')
    args.output.write_text(text)
    provenance = dict(format='MOBILE_SCORE_V1', backend='CPU candidate scoring; NPU graphs unchanged',
                      checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      trainable_parameters=161, normalization_values=16)
    args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(args.output, provenance['output_sha256'])


if __name__ == '__main__':
    main()
