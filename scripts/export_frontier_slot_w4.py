#!/usr/bin/env python3
"""Place frontier auxiliaries in the unused eighth physical draft output row."""
import json
import os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDraftModel
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
import export_draft_head_w4 as head_export


def pack_slot(logits, hidden, context):
    # The calibration graph has 504 rows, the unused decode graph has one.
    # Keep an identical operator sequence for the three encoding maps.
    h7 = F.pad(hidden, (0, 0, 0, 7))[:, :7]
    aux = torch.cat([h7.reshape(1, 1, -1), context.reshape(1, 1, -1)], dim=-1)
    aux = F.pad(aux, (0, logits.shape[-1] - aux.shape[-1]))
    padded = F.pad(logits, (0, 0, 0, 8))
    packed = torch.cat([padded[:, :7], aux, padded[:, 8:]], dim=1)[:, :logits.shape[1]]
    return packed.unsqueeze(0).squeeze(0)


class FrontierSlotModel(DFlashDraftModel):
    def forward(self, h_0, h_1, h_2, h_3, h_4, block_embd,
                atten_mask, input_pos, input_pos_ctx, *args):
        assert self.n_target_layers == 5 and self.output_cache and not self.logits_scaling
        hidden = block_embd
        output_k_cache, output_v_cache = [], []
        cos, sin = self.freqs_cos[input_pos][0], self.freqs_sin[input_pos][0]
        cos_ctx, sin_ctx = self.freqs_cos[input_pos_ctx][0], self.freqs_sin[input_pos_ctx][0]
        raw_context = sum(fc(h) for fc, h in zip(self.fc, [h_0, h_1, h_2, h_3, h_4]))
        context = self.hidden_norm(raw_context)
        for ind, layer in enumerate(self.layers):
            hidden, k, v = layer(hidden, g_ctx=context, freqs_cos=cos, freqs_sin=sin,
                freqs_cos_ctx=cos_ctx, freqs_sin_ctx=sin_ctx, atten_mask=atten_mask,
                k_caches=args[ind], v_caches=args[self.n_layers + ind])
            output_k_cache.extend(k)
            output_v_cache.extend(v)
        logits = self.output(self.norm(hidden))
        return pack_slot(logits, hidden, raw_context), output_k_cache, output_v_cache


original_tag_ios = llm_wrappers.TextDecoder._tag_ios


def preserve_frontier_output(self, node, fixed_point_type):
    if llm_wrappers.is_graph_output(node):
        shape = node.meta['val'].size()
        if len(shape) == 3 and shape[-1] == 151936:
            return None
    return original_tag_ios(self, node, fixed_point_type)


if __name__ == '__main__':
    artifact = Path(sys.argv[sys.argv.index('--artifact') + 1]).resolve()
    provenance = dict(sources={p.name: head_export.sha(p) for p in [Path(__file__), Path(head_export.__file__)]},
        packed_output_shape=[1, 8, 151936], auxiliary_row=7, auxiliary_hidden_elements=7*2560,
        auxiliary_context_elements=16*2560, physical_rows=8, logical_rows=7, inject_rows=16,
        scope='Deployment of the same FCS algorithm and frozen W4/A16 draft. First seven logits rows remain '
              'unchanged before PTQ; the unused eighth row carries auxiliary states. This packaging is not '
              'an algorithm contribution. Same-graph PC31 control and exact AR validation are required.')
    plan = artifact.with_name(artifact.name + '-plan.json')
    assert not plan.exists()
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(provenance, indent=2) + '\n')
    LLM_VARIANT_ARCHS['qwen3-4b-dspark-w4'] = FrontierSlotModel
    llm_wrappers.TextDecoder._tag_ios = preserve_frontier_output
    head_export.main()
    (artifact / 'frontier-export-manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
