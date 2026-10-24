"""G0 for the inject-only draft-prefill graph (mobile-plan-20261019.md section 11.2): on the 16 calibration pages packed per arm, the FP32
DFlashInjectModel's k_ctx/v_ctx on n_inject-row chunks must equal the FP32 draft graph's committed-row K/V on 16-row chunks (the same
fc, hidden_norm, wk/wv, k_norm and RoPE; the block path contributes nothing to those outputs). Writes <asset>/fp32-inject-audit.json.
Usage: check_vlm_register_mobile_inject.py --arm full-20k-seed42 [--inject-rows 256] [--main-context 4096]
"""
import argparse, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from vlm_register_mobile_export_common import OUT, ARMS, ROPE_POSITIONS, pack, extend_rope_tables, cache_rows, sha, write
ET = Path(os.environ['EXECUTORCH_ROOT']); sys.path.insert(0, str(ET.parent))
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDraftModel
from vlm_register_mobile_inject_model import DFlashInjectModel, inject_context, CACHE_ROWS


def model_args(cfg, L):
    return ModelArgs(dim=cfg['dim'], n_layers=cfg['n_layers'], n_heads=cfg['n_heads'], n_kv_heads=cfg['n_kv_heads'], head_dim=cfg['head_dim'], hidden_dim=cfg['hidden_dim'],
        vocab_size=cfg['vocab_size'], norm_eps=cfg['norm_eps'], rope_freq_base=cfg['rope_theta'], use_hf_rope=True, use_qk_norm=True, qk_norm_before_rope=True,
        max_seq_len=L, max_context_len=L, use_kv_cache=True, max_batch_size=1)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--arm', required=True); p.add_argument('--inject-rows', type=int, default=256); p.add_argument('--main-context', type=int, default=4096)
    a = p.parse_args(); torch.set_num_threads(6); N = a.inject_rows; L = a.main_context; cache = cache_rows(L); LI = inject_context(N)
    asset = OUT/a.arm; manifest = json.loads((asset/'manifest.json').read_text()); spec = ARMS[a.arm]
    cfg = json.loads((ET/'examples/models/qwen3_dflash/config/4b_dflash_config.json').read_text())
    state = torch.load(asset/'draft-meta.pth', weights_only=True, mmap=True)
    main_args, inject_args = model_args(cfg, L), model_args(cfg, LI)
    main = DFlashDraftModel(main_args, ar_len=8, n_inject=16, n_target_layers=5).eval(); extend_rope_tables(main, main_args)
    main.load_state_dict(state, strict=True, assign=True)
    inject = DFlashInjectModel(inject_args, ar_len=8, n_inject=N, n_target_layers=5).eval(); extend_rope_tables(inject, inject_args)
    inject.load_state_dict(state, strict=True, assign=True)
    assets = dict(arm=a.arm, summary=torch.load(asset/'summary.pt', weights_only=True) if (asset/'summary.pt').exists() else None)
    calibration = json.loads((OUT/'calibration/manifest.json').read_text()); checks = []; worst = dict(cosine=1.0, max_abs=0.0)
    ex = inject.get_example_inputs(); ikeys, ivalues = ex[5], ex[6]
    assert ikeys[0].shape == (1, 8, 128, CACHE_ROWS), ikeys[0].shape
    for record in calibration['records']:
        assert sha(record['path']) == record['sha256']; sample = torch.load(record['path'], map_location='cpu', weights_only=True)
        ids = sample['input_ids'].long(); features, positions = pack(sample, spec['mode'], assets, spec)
        prefix = min(int(sample['prompt_length']), len(ids)-1); valid = positions < prefix
        features, positions = features[valid].reshape(-1, 5, 2560).float(), positions[valid]; n = len(features)
        assert n <= cache and int(positions.max()) < ROPE_POSITIONS, (n, cache)
        # main graph, 16 rows per call, cache filling as the device does
        keys, values = main.get_example_inputs()[5:]
        K_main = [torch.zeros(1, 8, 128, n) for _ in range(5)]; V_main = [torch.zeros(1, 8, n, 128) for _ in range(5)]
        for begin in range(0, n, 16):
            end = min(n, begin+16); take = end-begin
            hidden = [torch.zeros(1, 16, 2560) for _ in range(5)]
            for layer in range(5): hidden[layer][0, :take] = features[begin:end, layer]
            block = torch.full((1, 8), 151669, dtype=torch.long); block[0, 0] = ids[prefix]
            mask = torch.full((1, 1, 8, L), -255.); mask[..., :begin] = 0; mask[..., cache:cache+take] = 0; mask[..., -8:-1] = 0
            ctx_pos = torch.zeros((1, 16), dtype=torch.int32); ctx_pos[0, :take] = positions[begin:end].int()
            block_pos = torch.arange(prefix, prefix+8, dtype=torch.int32)[None]
            _, new_keys, new_values = main(*hidden, main.tok_embeddings(block), mask, block_pos, ctx_pos, *keys, *values)
            for layer in range(5):
                keys[layer][..., begin:end] = new_keys[layer][..., :take]; values[layer][:, :, begin:end, :] = new_values[layer][:, :, :take, :]
                K_main[layer][..., begin:end] = new_keys[layer][..., :take]; V_main[layer][:, :, begin:end, :] = new_values[layer][:, :, :take, :]
        # inject graph, N rows per call, no cache
        K_inj = [torch.zeros(1, 8, 128, n) for _ in range(5)]; V_inj = [torch.zeros(1, 8, n, 128) for _ in range(5)]; calls = 0
        for begin in range(0, n, N):
            end = min(n, begin+N); take = end-begin
            hidden = [torch.zeros(1, N, 2560) for _ in range(5)]
            for layer in range(5): hidden[layer][0, :take] = features[begin:end, layer]
            block = torch.full((1, 8), 151669, dtype=torch.long); block[0, 0] = ids[prefix]
            mask = torch.full((1, 1, 8, LI), -255.); mask[..., CACHE_ROWS:CACHE_ROWS+take] = 0; mask[..., -8:-1] = 0
            ctx_pos = torch.zeros((1, N), dtype=torch.int32); ctx_pos[0, :take] = positions[begin:end].int()
            block_pos = torch.arange(prefix, prefix+8, dtype=torch.int32)[None]
            primary, new_keys, new_values = inject(*hidden, inject.tok_embeddings(block), mask, block_pos, ctx_pos, *ikeys, *ivalues)
            assert primary.shape == (1, 8, 2560) and new_keys[0].shape == (1, 8, 128, N) and new_values[0].shape == (1, 8, N, 128)
            for layer in range(5):
                K_inj[layer][..., begin:end] = new_keys[layer][..., :take]; V_inj[layer][:, :, begin:end, :] = new_values[layer][:, :, :take, :]
            calls += 1
        per_layer = []
        for layer in range(5):
            for name, x, y in (('k', K_main[layer], K_inj[layer]), ('v', V_main[layer], V_inj[layer])):
                cos = F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item(); mad = (x-y).abs().max().item()
                per_layer.append(dict(layer=layer, tensor=name, cosine=cos, max_abs=mad)); worst = dict(cosine=min(worst['cosine'], cos), max_abs=max(worst['max_abs'], mad))
        check = dict(id=record['id'], prefix=prefix, packed_rows=n, inject_calls=calls, main_calls=(n+15)//16, cosine_min=min(r['cosine'] for r in per_layer), max_abs=max(r['max_abs'] for r in per_layer))
        checks.append(check); print('INJECT_CHECK', json.dumps(check), flush=True)
    passed = worst['cosine'] >= 0.9999
    write(asset/'fp32-inject-audit.json', dict(passed=passed, inject_rows=N, max_context_len=inject_context(N), cache_rows=CACHE_ROWS, main_context=L, rule='cosine >= 0.9999 on every layer K and V',
                                               worst=worst, checks=checks, rope_positions=ROPE_POSITIONS, source_sha256=sha(__file__), model_sha256=sha(Path(__file__).with_name('vlm_register_mobile_inject_model.py')),
                                               checkpoint_sha256=manifest['checkpoint_sha256'], graph_sha256=manifest['outputs']['draft-meta.pth']))
    print('INJECT_AUDIT', json.dumps(dict(arm=a.arm, passed=passed, **worst)), flush=True)
    assert passed, worst


if __name__ == '__main__': main()
