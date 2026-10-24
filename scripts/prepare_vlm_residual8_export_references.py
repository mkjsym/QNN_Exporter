"""Native-model references for static graph, physical cache, and B7-to-B8 padding checks."""
import argparse,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from vlm_residual8_paths import ROOT,STUDY,INITIAL
from vlm_residual8 import build_model
from deepspec.utils.config import ConfigNode
sys.path.insert(0,str(ROOT.parent/'QNN_Exporter/scripts'))
from vlm_residual8_export_common import OUT,pack,sha,write

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',required=True);args=p.parse_args();torch.set_num_threads(4)
    model_args=ConfigNode(dict(num_draft_layers=5,target_layer_ids=[1,9,17,25,33],mask_token_id=151669,num_anchors=8,block_size=7,markov_rank=0,confidence_head_alpha=0.))
    draft=build_model(model_args,'norm8','cpu',torch.float32).eval()
    sources=json.loads((OUT/'calibration/manifest.json').read_text())['records']
    selected=[next(r for r in sources if r['domain']==d) for d in ['general','ocr','text']]
    for arm in args.arms:
        asset=OUT/arm;manifest=json.loads((asset/'manifest.json').read_text());mode=manifest['mode'];draft.visual_mode=mode
        draft.load_state_dict(load_file(manifest['checkpoint']),strict=True)
        summary=torch.load(asset/'summary.pt',weights_only=True);records=[]
        for record in selected:
            sample=torch.load(record['path'],weights_only=True,mmap=True);features=sample['target_hidden_states'];ids=sample['input_ids'].long()
            # Exercise BF16 pooling identically to the training adapter before FP32 body comparison.
            packed,positions=pack(features,sample['image_groups'],mode,summary,sample['prompt_length'])
            test_input=features[None];groups=sample['image_groups'][None]
            native,native_positions,valid=draft.pack_context(test_input,groups,torch.ones_like(groups),[sample['prompt_length']])
            assert torch.equal(native[0],packed) and torch.equal(native_positions[0],positions)
            for prefix in [sample['prompt_length'],min(sample['prompt_length']+16,len(ids)-1)]:
                valid=positions<prefix;context=packed[valid].float()[None];ctx_pos=positions[valid][None]
                block=torch.full((1,7),151669,dtype=torch.long);block[0,0]=ids[prefix]
                allpos=torch.cat((ctx_pos,torch.arange(prefix,prefix+7)[None]),1)
                mask=torch.ones((1,1,7,len(context[0])+7),dtype=torch.bool)
                hidden=draft._forward_backbone(position_ids=allpos,attention_mask=mask,noise_embedding=draft.embed_tokens(block),target_hidden_states=context)
                logits=draft.compute_logits(hidden)
                path=asset/f"reference-{record['id']}-{prefix}.pt"
                torch.save(dict(context=context[0],context_positions=ctx_pos[0],root=int(ids[prefix]),original_prefix=prefix,reference_logits=logits),path)
                records.append(dict(path=str(path),sha256=sha(path),id=record['id'],domain=record['domain'],prefix=prefix,packed_rows=context.shape[1]))
        write(asset/'native-references.json',dict(records=records,source_sha256=sha(__file__),checkpoint_sha256=manifest['checkpoint_sha256'],pooling_training_export_exact=True))
        print('NATIVE_REFERENCES',arm,flush=True)

if __name__=='__main__':main()
