"""Inject-only DFlash draft graph for the drafter's prompt prefill (mobile-plan-20261019.md section 11.2, M4).

A DFlash draft step carries two branches: the block rows (attention over the cache, MLP, LM head) and the committed rows, which never
join the residual stream: g_ctx = hidden_norm(sum_i fc_i(h_i)) once, then per layer k_ctx = RoPE(k_norm(wk(g_ctx))), v_ctx = wv(g_ctx).
The register study's drafter contexts run both branches for every 16 committed rows, so a Full prompt of 1,677 rows costs about 104 graph
calls that each stream the whole drafter (about 14 ms). This module keeps only the committed-row branch for n_inject rows per call; the
block path is absent, so the call streams fc + wk/wv (59M parameters) instead of the 1.3B of the full graph.

Graph I/O (same names and layouts as the full draft graph, so the runtime's input filler and K/V write-back apply unchanged):
  inputs  h_0..h_4 [1, n_inject, 2560] float (target hidden per drafter layer), block_embd [1, 8, 2560] float, atten_mask [1, 1, 8, context],
          input_pos [1, 8] int32 (unused, dropped by the lowering), input_pos_ctx [1, n_inject] int32, args_0..args_9 (unused caches, dropped)
  outputs primary [1, 8, 2560] float (block_embd plus the mask's last column: it keeps atten_mask a graph input because the runtime reads
          context_len and ar_len off it; no model compute), then k_ctx x5 [1, 8, 128, n_inject] float, v_ctx x5 [1, 8, n_inject, 128] float.
Context = CACHE_ROWS (8, never read) + n_inject + 8 block rows.
"""
import torch
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDraftModel
from vlm_register_mobile_export_common import BLOCK_ROWS, INJECT_CACHE_ROWS as CACHE_ROWS, inject_context, inject_suffix


def _rope_tiled(x, freqs_cos, freqs_sin):
    """apply_rotary_emb_single (model/apply_rope.py) with the per-row cos/sin materialized over the heads instead of broadcast.
    The HTP lowering of the broadcast multiply `[1, H, n, 64] * [1, 1, n, 64]` reuses the first 32 rows of the broadcast operand
    for every later row when n = 256 (found 2026-10-22 with qnn-net-run: row r is RoPE'd at the position of row r mod 32; the
    16-row step graph never crosses that boundary), so the committed-row keys came out wrong past row 32 of every call."""
    x_r, x_i = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    c = freqs_cos[None, None, :, :].repeat(1, x.shape[1], 1, 1)
    s = freqs_sin[None, None, :, :].repeat(1, x.shape[1], 1, 1)
    return torch.cat([x_r * c - x_i * s, x_r * s + x_i * c], dim=-1)


class DFlashInjectModel(DFlashDraftModel):
    """DFlashDraftModel with the block path removed: g_ctx and every layer's k_ctx/v_ctx for the n_inject committed rows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.layers:
            layer.attention.apply_rope_emb = _rope_tiled

    def forward(self, h_0, h_1, h_2, h_3, h_4, block_embd, atten_mask, input_pos, input_pos_ctx, *args):
        assert self.n_target_layers == 5
        freqs_cos_ctx = self.freqs_cos[input_pos_ctx][0]
        freqs_sin_ctx = self.freqs_sin[input_pos_ctx][0]
        g_ctx = self.hidden_norm(sum(fc(h) for fc, h in zip(self.fc, (h_0, h_1, h_2, h_3, h_4))))
        output_k_cache, output_v_cache = [], []
        for layer in self.layers:
            attention = layer.attention
            # after prepare_attention_conv (export) the projections are wk_conv/wv_conv; before it (FP32 audit) they are wk/wv
            k_ctx, v_ctx = attention._ctx_kv(g_ctx, freqs_cos_ctx, freqs_sin_ctx, hasattr(attention, 'wk_conv'))
            output_k_cache.append(k_ctx); output_v_cache.append(v_ctx)
        primary = block_embd + atten_mask[0, 0, :, -1:]
        if self.output_cache:
            return primary, output_k_cache, output_v_cache
        return primary
