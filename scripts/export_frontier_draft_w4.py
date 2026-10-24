#!/usr/bin/env python3
"""Expose frozen draft hidden and injected raw FC states for frontier synthesis."""
import json
import os
from pathlib import Path
import sys
import torch

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDraftModel
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import DFlashDecoderInference
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
import export_draft_head_w4 as head_export


class FrontierPackedModel(DFlashDraftModel):
    def forward(self, h_0, h_1, h_2, h_3, h_4, block_embd,
                atten_mask, input_pos, input_pos_ctx, *args):
        assert self.n_target_layers == 5 and self.output_cache and not self.logits_scaling
        hidden_states = block_embd
        output_k_cache, output_v_cache = [], []
        freqs_cos, freqs_sin = self.freqs_cos[input_pos][0], self.freqs_sin[input_pos][0]
        freqs_cos_ctx = self.freqs_cos[input_pos_ctx][0]
        freqs_sin_ctx = self.freqs_sin[input_pos_ctx][0]
        raw_context = sum(fc(h) for fc, h in zip(self.fc, [h_0, h_1, h_2, h_3, h_4]))
        g_ctx = self.hidden_norm(raw_context)
        for ind, layer in enumerate(self.layers):
            hidden_states, k, v = layer(hidden_states, g_ctx=g_ctx,
                freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                freqs_cos_ctx=freqs_cos_ctx, freqs_sin_ctx=freqs_sin_ctx,
                atten_mask=atten_mask, k_caches=args[ind], v_caches=args[self.n_layers+ind])
            output_k_cache.extend(k)
            output_v_cache.extend(v)
        logits = self.output(self.norm(hidden_states))
        # Broadcast the sixteen FC rows through a shape-independent floating operation.
        packed_context = raw_context.reshape(1, 1, 16 * self.dim) + hidden_states[..., :1] * 0.0
        packed = torch.cat([logits, hidden_states, packed_context], dim=-1).unsqueeze(0).squeeze(0)
        return packed, output_k_cache, output_v_cache


class FrontierCalibration(DFlashDecoderInference):
    def make_forward_fn(self):
        forward = super().make_forward_fn()
        return lambda model, inputs: forward(model, inputs)[..., :151936]


if __name__ == '__main__':
    artifact = Path(sys.argv[sys.argv.index('--artifact') + 1]).resolve()
    provenance = dict(sources={p.name: head_export.sha(p) for p in [Path(__file__), Path(head_export.__file__)]},
        packed_output_shape=[1, 8, 195456], vocabulary_columns=151936,
        draft_hidden_columns=2560, context_columns=40960, physical_rows=8, logical_rows=7, inject_rows=16,
        scope='Original DSpark7 weights and W4/A16 full-head recipe. Extra output carries H5 before final norm '
              'and all sixteen raw projected target context rows. The original logits remain the first151936 columns. '
              'Any graph-induced difference must be measured with a same-graph PC31 control.')
    plan = artifact.with_name(artifact.name + '-plan.json')
    assert not plan.exists()
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(provenance, indent=2) + '\n')
    LLM_VARIANT_ARCHS['qwen3-4b-dspark-w4'] = FrontierPackedModel
    llm_wrappers.DFlashDecoderInference = FrontierCalibration
    head_export.main()
    (artifact / 'frontier-export-manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
