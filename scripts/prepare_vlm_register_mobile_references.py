"""Native references for the FP32 export audit: the host drafter's logits on the export-packed context of three calibration pages."""
import json, os, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from vlm_register_mobile_export_common import OUT, CALIB, ARMS, MOBI, pack, sha, write
sys.path.insert(0, str(MOBI/'tools/training'))
os.environ.setdefault('REGISTER_CAP', 'doc20k')


@torch.inference_mode()
def main(arm):
    torch.set_num_threads(4); spec = ARMS[arm]; asset = OUT/arm; manifest = json.loads((asset/'manifest.json').read_text())
    from run_vlm_deepspec_rollout import MODEL_ARGS
    from vlm_register_online import build_online_model
    draft = build_online_model(spec['mode'], MODEL_ARGS, 'cpu', torch.float32, register_count=int(spec.get('register_count', 0)), refresh='previous',
                               deep=bool(spec.get('deep', False))).eval()
    draft.load_state_dict(load_file(manifest['checkpoint']), strict=True)
    assets = dict(arm=arm, summary=torch.load(asset/'summary.pt', weights_only=True) if (asset/'summary.pt').exists() else None)
    sources = json.loads((OUT/'calibration/manifest.json').read_text())['records']; records = []
    for record in [next(r for r in sources if r['domain'] == d) for d in ('docvqa', 'chartqa', 'infographic')]:
        assert sha(record['path']) == record['sha256']
        sample = torch.load(record['path'], weights_only=True); ids = sample['input_ids'].long()
        packed, positions = pack(sample, spec['mode'], assets, spec)
        for prefix in (int(sample['prompt_length']), min(int(sample['prompt_length'])+16, len(ids)-1)):
            valid = positions < prefix; context = packed[valid].float()[None]; ctx_pos = positions[valid][None]
            block = torch.full((1, 7), 151669, dtype=torch.long); block[0, 0] = ids[prefix]
            allpos = torch.cat((ctx_pos, torch.arange(prefix, prefix+7)[None]), 1)
            mask = torch.ones((1, 1, 7, len(context[0])+7), dtype=torch.bool)
            hidden = draft._forward_backbone(position_ids=allpos, attention_mask=mask, noise_embedding=draft.embed_tokens(block), target_hidden_states=context)
            path = asset/f"reference-{record['id']}-{prefix}.pt"
            torch.save(dict(context=context[0], context_positions=ctx_pos[0], root=int(ids[prefix]), original_prefix=prefix, reference_logits=draft.compute_logits(hidden)), path)
            records.append(dict(path=str(path), sha256=sha(path), id=record['id'], domain=record['domain'], prefix=prefix, packed_rows=int(context.shape[1])))
    write(asset/'native-references.json', dict(records=records, source_sha256=sha(__file__), checkpoint_sha256=manifest['checkpoint_sha256'], packing='vlm_register_mobile_export_common.pack'))
    print('REGISTER_MOBILE_REFERENCES', arm, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
