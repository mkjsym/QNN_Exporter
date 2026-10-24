"""Compare mode-specific native packing and save six training-only graph fixtures."""
import torch
from safetensors.torch import load_file
from vlm_data_scale_export_common import OUT,candidate,pack,sha,write
from vlm_data_scale import build_model, VENDOR
import sys
sys.path.insert(0,str(VENDOR))
from deepspec.utils.config import ConfigNode
import json

@torch.inference_mode()
def main():
    torch.set_num_threads(4);chosen=candidate();asset=OUT/chosen['key'];mode=chosen['mode']
    args=ConfigNode(dict(num_draft_layers=5,target_layer_ids=[1,9,17,25,33],mask_token_id=151669,num_anchors=8,block_size=7,markov_rank=0,confidence_head_alpha=0.))
    draft=build_model(args,mode,'cpu',torch.float32).eval()
    manifest=json.loads((asset/'manifest.json').read_text());draft.load_state_dict(load_file(manifest['checkpoint']),strict=True)
    summary=torch.load(asset/'summary.pt',weights_only=True);sources=json.loads((OUT/'calibration/manifest.json').read_text())['records'];records=[]
    for record in [next(r for r in sources if r['domain']==d) for d in ('general','ocr','text')]:
        assert sha(record['path'])==record['sha256']
        sample=torch.load(record['path'],weights_only=True,mmap=True);features=sample['target_hidden_states'];ids=sample['input_ids'].long()
        packed,positions=pack(features,sample['image_groups'],mode,summary,sample['prompt_length'])
        args=(features[None],sample['image_groups'][None],torch.ones_like(sample['image_groups'][None]))
        if mode=='norm8':native,native_positions,_=draft.pack_context(*args,[sample['prompt_length']])
        else:native,native_positions,_=draft.pack_context(*args)
        assert torch.equal(native[0],packed) and torch.equal(native_positions[0],positions)
        for prefix in (sample['prompt_length'],min(sample['prompt_length']+16,len(ids)-1)):
            valid=positions<prefix;context=packed[valid].float()[None];ctx_pos=positions[valid][None]
            block=torch.full((1,7),151669,dtype=torch.long);block[0,0]=ids[prefix]
            allpos=torch.cat((ctx_pos,torch.arange(prefix,prefix+7)[None]),1)
            mask=torch.ones((1,1,7,len(context[0])+7),dtype=torch.bool)
            hidden=draft._forward_backbone(position_ids=allpos,attention_mask=mask,noise_embedding=draft.embed_tokens(block),target_hidden_states=context)
            path=asset/f"reference-{record['id']}-{prefix}.pt"
            torch.save(dict(context=context[0],context_positions=ctx_pos[0],root=int(ids[prefix]),original_prefix=prefix,reference_logits=draft.compute_logits(hidden)),path)
            records.append(dict(path=str(path),sha256=sha(path),id=record['id'],domain=record['domain'],prefix=prefix,packed_rows=context.shape[1]))
    write(asset/'native-references.json',dict(records=records,source_sha256=sha(__file__),checkpoint_sha256=manifest['checkpoint_sha256'],pooling_training_export_exact=True))
    print('DATA_SCALE_NATIVE_REFERENCES',chosen['key'],flush=True)

if __name__=='__main__':main()
