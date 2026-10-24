"""Check converted full-context DFlash logits and committed KV against the training model."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDraftModel
from prepare_vlm_finetuned_draft import OUT, base, sha256, atomic_json
from vlm_learned8 import VisualDraft
from vlm_visual_followup import load_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', default='full_seed42')
    args = parser.parse_args()
    torch.set_num_threads(4)
    asset = OUT/args.arm
    manifest = json.loads((asset/'manifest.json').read_text())
    assert sha256(asset/'draft-meta.pth') == manifest['outputs']['draft-meta.pth']
    cfg = json.loads((ET/'examples/models/qwen3_dflash/config/4b_dflash_config.json').read_text())
    config = ModelArgs(dim=cfg['dim'], n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        n_kv_heads=cfg['n_kv_heads'], head_dim=cfg['head_dim'], hidden_dim=cfg['hidden_dim'],
        vocab_size=cfg['vocab_size'], norm_eps=cfg['norm_eps'], rope_freq_base=cfg['rope_theta'],
        use_hf_rope=True, use_qk_norm=True, qk_norm_before_rope=True,
        max_seq_len=1024, max_context_len=1024, use_kv_cache=True, max_batch_size=1)
    weights = torch.load(asset/'draft-meta.pth', weights_only=True, mmap=True)
    graph = DFlashDraftModel(config, ar_len=8, n_inject=16, n_target_layers=5).eval()
    graph.load_state_dict(weights, strict=True, assign=True)
    upstream = VisualDraft.load(base.DRAFT, base.TEXT_TARGET, 'cpu').float().eval()
    load_weights(upstream, manifest['checkpoint'])
    sources = json.loads((OUT/'calibration/manifest.json').read_text())['records']
    chosen = [next(record for record in sources if record['domain'] == d) for d in ['general', 'ocr', 'text']]
    checks = []
    with torch.inference_mode():
        for record in chosen:
            assert sha256(record['path']) == record['sha256']
            item = torch.load(record['path'], weights_only=False)
            features = item['target_features'].float()
            ids = item['input_ids']
            for prefix in sorted({item['prompt_length'], min(item['prompt_length']+16, len(ids)-1)}):
                reference, _ = upstream(features[:prefix].reshape(prefix, -1), ids,
                                        [prefix], item['image_positions'], item['grid_hw'], 'full')
                example = graph.get_example_inputs()
                keys, values = example[5:]
                for begin in range(0, prefix, 16):
                    end = min(prefix, begin+16)
                    take = end-begin
                    hidden = [torch.zeros(1, 16, 2560) for _ in range(5)]
                    for layer in range(5):
                        hidden[layer][0, :take] = features[begin:end, layer]
                    block_ids = torch.full((1, 8), 151669, dtype=torch.long)
                    block_ids[0, 0] = ids[end]
                    mask = torch.full((1, 1, 8, 1024), -255.)
                    mask[..., :begin] = 0
                    mask[..., 1000:1000+take] = 0
                    mask[..., -8:] = 0
                    result, new_keys, new_values = graph(*hidden, graph.tok_embeddings(block_ids), mask,
                        torch.arange(end, end+8, dtype=torch.int32)[None],
                        torch.arange(begin, begin+16, dtype=torch.int32)[None], *keys, *values)
                    assert len(new_keys) == len(new_values) == 5
                    for layer in range(5):
                        keys[layer][..., begin:end] = new_keys[layer][..., :take]
                        values[layer][:, :, begin:end, :] = new_values[layer][:, :, :take, :]
                actual = result[:, 1:]
                delta = (actual-reference).abs()
                same = actual.argmax(-1).equal(reference.argmax(-1))
                checks.append(dict(id=item['case']['id'], domain=item['case']['domain'], prefix=prefix,
                                   max_abs=float(delta.max()), mean_abs=float(delta.mean()), top1_exact=same))
                print('CHECK', checks[-1], flush=True)
                assert same
                torch.testing.assert_close(actual, reference, atol=.003, rtol=.001)
    atomic_json(asset/'fp32-audit.json', dict(passed=True, checks=checks, source_sha256=sha256(Path(__file__)),
                scope='Converted split-FC static graph with real multi-step committed KV vs frozen training implementation; FP32 compute before PTQ',
                checkpoint_sha256=manifest['checkpoint_sha256'], graph_sha256=manifest['outputs']['draft-meta.pth']))


if __name__ == '__main__':
    main()
