#!/usr/bin/env python3
"""Export native Domino logits and the static part of prefix correction together."""
import os
import json
from pathlib import Path
import sys

import torch
from torch import nn

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
from export_draft_body import DraftBodyModel
import export_long_draft_w4 as long_export


class DominoPackedModel(DraftBodyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domino_base = nn.Linear(self.dim, 256, bias=False)

    def forward(self, h_0, h_1, h_2, h_3, h_4, block_embd,
                atten_mask, input_pos, input_pos_ctx, *args):
        hidden, keys, values = super().forward(h_0, h_1, h_2, h_3, h_4, block_embd,
                                              atten_mask, input_pos, input_pos_ctx, *args)
        assert self.output_cache and not self.logits_scaling
        logits = self.output(hidden)
        base = self.domino_base(hidden)
        packed = torch.cat([logits, base], dim=-1).unsqueeze(0).squeeze(0)
        return packed, keys, values


class DominoCalibration(long_export.LongDraftCalibration):
    def make_forward_fn(self):
        forward = super().make_forward_fn()
        return lambda model, inputs: forward(model, inputs)[..., :151936]


if __name__ == '__main__':
    artifact = Path(sys.argv[sys.argv.index('--artifact') + 1]).resolve()
    provenance = dict(sources={p.name: long_export.sha(p) for p in
        [Path(__file__), Path(long_export.__file__), Path(__file__).with_name('export_draft_body.py'),
         Path(__file__).with_name('export_draft_head_w4.py')]},
        packed_output_shape=[1, 17, 152192], vocabulary_columns=151936, static_correction_columns=256,
        scope='Native Domino baseline. Same W4 full vocabulary and body recipe as DSpark control; '
              'packed suffix contains the 256-dimensional static prefix-correction projection. '
              'The inherited long-draft manifest describes shared geometry, not this checkpoint identity.')
    plan = artifact.with_name(artifact.name + '-plan.json')
    assert not plan.exists()
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(provenance, indent=2) + '\n')
    LLM_VARIANT_ARCHS['qwen3-4b-dspark-w4'] = DominoPackedModel
    long_export.LongDraftCalibration = DominoCalibration
    long_export.main()
    (artifact / 'domino-export-manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
