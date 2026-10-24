#!/usr/bin/env python3
"""Export an isolated DFlash/DSpark body graph for sparse-head cost experiments.

Use the normal llama.py arguments. Only qwen3-4b-dspark-w4 is accepted.
The first output is normalized draft hidden state, not vocabulary logits.
This is an export-process override; no ExecuTorch source files are changed.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))

from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDraftModel
from executorch.examples.qualcomm.oss_scripts.llama import llama


class DraftBodyModel(DFlashDraftModel):
    def forward(self, h_0, h_1, h_2, h_3, h_4, block_embd,
                atten_mask, input_pos, input_pos_ctx, *args):
        assert self.n_target_layers == 5
        hidden_states = block_embd
        output_k_cache, output_v_cache = [], []
        freqs_cos, freqs_sin = self.freqs_cos[input_pos][0], self.freqs_sin[input_pos][0]
        freqs_cos_ctx = self.freqs_cos[input_pos_ctx][0]
        freqs_sin_ctx = self.freqs_sin[input_pos_ctx][0]
        g_ctx = self.hidden_norm(sum(fc(h) for fc, h in zip(
            self.fc, [h_0, h_1, h_2, h_3, h_4])))
        for ind, layer in enumerate(self.layers):
            hidden_states, k, v = layer(hidden_states, g_ctx=g_ctx,
                freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                freqs_cos_ctx=freqs_cos_ctx, freqs_sin_ctx=freqs_sin_ctx,
                atten_mask=atten_mask, k_caches=args[ind],
                v_caches=args[self.n_layers+ind])
            output_k_cache.extend(k)
            output_v_cache.extend(v)
        hidden = self.norm(hidden_states)
        if self.logits_scaling:
            hidden = hidden / self.logits_scaling
        # Preserve an explicit rank-three output and a recognizable output node.
        hidden = hidden.unsqueeze(0).squeeze(0)
        if self.output_cache:
            return hidden, output_k_cache, output_v_cache
        return hidden


def main():
    args = llama._build_parser().parse_args()
    if args.decoder_model != 'qwen3-4b-dspark-w4':
        raise ValueError('This experimental export is restricted to qwen3-4b-dspark-w4')
    if not args.compile_only or args.qat or args.use_fp16:
        raise ValueError('Require compile_only, existing W4/A16 PTQ, and no QAT')
    artifact = Path(args.artifact).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    if list(artifact.glob('*.pte')):
        raise FileExistsError('Refuse to replace an existing compiled graph')
    upstream = ET/'examples/qualcomm/oss_scripts/llama/model/dflash_draft.py'
    manifest = dict(output='post-norm hidden [1, block, 2560], float IO; not vocabulary logits',
        full_vocab=151936, head_weight_source='unchanged checkpoint output.weight',
        checkpoint=args.checkpoint, hidden_dump=args.dflash_hidden,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        upstream_sha256=hashlib.sha256(upstream.read_bytes()).hexdigest(),
        arguments=vars(args), device_inference_run=False)
    (artifact/'body-export-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    LLM_VARIANT_ARCHS[args.decoder_model] = DraftBodyModel
    args.build_folder = os.path.realpath(args.build_folder)
    llama.export_llama(args)


if __name__ == '__main__':
    main()
