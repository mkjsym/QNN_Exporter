#!/usr/bin/env python3
"""Check body-only export against the unchanged full-head module and its KV outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from export_draft_body import DraftBodyModel, DFlashDraftModel, ET
from executorch.examples.models.llama.model_args import ModelArgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--hidden', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(8)
    cfg = json.loads((ET/'examples/models/qwen3_dflash/config/4b_dflash_config.json').read_text())
    a = ModelArgs(dim=cfg['dim'], n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        n_kv_heads=cfg['n_kv_heads'], head_dim=cfg['head_dim'], hidden_dim=cfg['hidden_dim'],
        vocab_size=cfg['vocab_size'], norm_eps=cfg['norm_eps'], rope_freq_base=cfg['rope_theta'],
        use_hf_rope=True, use_qk_norm=True, qk_norm_before_rope=True,
        max_seq_len=1024, max_context_len=1024, use_kv_cache=True, max_batch_size=1)
    sd = {k:v.float() for k,v in torch.load(args.checkpoint, mmap=True, weights_only=True).items()}
    full = DFlashDraftModel(a, ar_len=8, n_inject=16, n_target_layers=5).eval()
    body = DraftBodyModel(a, ar_len=8, n_inject=16, n_target_layers=5).eval()
    full.load_state_dict(sd, strict=True, assign=True); body.load_state_dict(sd, strict=True, assign=True)
    hidden = torch.load(args.hidden, weights_only=True)['hidden'].float()
    assert hidden.shape[1] == 12800 and len(hidden) >= 16
    data = full.get_example_inputs()
    hh = [hidden[:16, i*2560:(i+1)*2560][None].contiguous() for i in range(5)]
    ids = torch.full((1,8), 151669, dtype=torch.long); ids[0,0] = 26194
    emb = full.tok_embeddings(ids)
    mask = torch.full((1,1,8,1024), -255.); mask[:,:,:,-24:] = 0
    pos = torch.arange(16,24,dtype=torch.int32)[None]
    pc = torch.arange(16,dtype=torch.int32)[None]
    inputs = (*hh, emb, mask, pos, pc, *data[5], *data[6])
    saved = []
    handle = full.norm.register_forward_hook(lambda m, i, o: saved.append(o.detach()))
    with torch.no_grad():
        reference = full(*inputs); actual = body(*inputs)
    handle.remove()
    assert torch.equal(saved[0], actual[0])
    assert len(reference[1]) == len(actual[1]) and len(reference[2]) == len(actual[2])
    for x, y in zip(reference[1]+reference[2], actual[1]+actual[2]):
        assert torch.equal(x,y)
    with torch.no_grad():
        projected = torch.nn.functional.linear(actual[0], sd['output.weight'])
    assert torch.equal(reference[0], projected)
    ep = torch.export.export(body, inputs, strict=True)
    used = [str(s.target) for s in ep.graph_signature.input_specs if s.kind.name == 'PARAMETER']
    assert 'output.weight' not in used and 'tok_embeddings.weight' not in used
    result = dict(passed=True, hidden_exact=True, kv_tensors_exact=len(actual[1])+len(actual[2]),
        full_projection_exact=True, exported_head_parameters=0,
        shape=list(actual[0].shape), dtype=str(actual[0].dtype),
        head_embedding_bit_equal=torch.equal(sd['output.weight'], sd['tok_embeddings.weight']),
        head_elements=sd['output.weight'].numel(), used_graph_parameters=used,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, tensor in [('head-probe-hidden.f32', torch.stack(hh).flatten()),
                         ('head-probe-reference.f32', reference[0]),
                         ('head-probe-body-reference.f32', actual[0])]:
        path = args.out.parent/name
        tensor.detach().contiguous().numpy().tofile(path)
        result[name] = dict(bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result['embedding_f16_sha256'] = hashlib.sha256(sd['tok_embeddings.weight'].half().contiguous().numpy().tobytes()).hexdigest()
    args.out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
