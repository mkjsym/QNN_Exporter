"""Convert matched DeepSpec b7 weights and training-only calibration for NPU export."""
import argparse,json,os,shutil,sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from vlm_noimage_export_common import ROOT,MOBI,STUDY,BASE,OUT,TEXT,INITIAL,sha,write

def calibration():
    dest=OUT/'calibration';dest.mkdir(exist_ok=False)
    rows=[json.loads(x) for x in (MOBI/'docs/results/raw/vlm-learned8-20260929/data/train.jsonl').read_text().splitlines()]
    selected=[x for d,n in [('general',6),('ocr',6),('text',4)] for x in [r for r in rows if r['domain']==d][:n]]
    records=[];hiddens=[]
    for item in selected:
        source=BASE/'cache/train'/(item['id']+'.pt');sample=torch.load(source,weights_only=True,mmap=True)
        records.append(dict(path=str(source),sha256=sha(source),id=item['id'],domain=item['domain'],split='train'))
        hiddens.append(sample['target_hidden_states'][sample['image_groups']<0])
    torch.save(dict(hidden=torch.cat(hiddens).float()),dest/'calibration-hidden.pt')
    write(dest/'calibration-prompts.json',[dict(messages=[dict(role='user',content=x['prompt'])]) for x in selected])
    write(dest/'manifest.json',dict(records=records,source_sha256=sha(__file__),outputs={n:sha(dest/n) for n in ['calibration-hidden.pt','calibration-prompts.json']}))

def convert(arm):
    assert shutil.disk_usage(OUT).free>50*(1<<30)
    target=OUT/arm;target.mkdir(exist_ok=False)
    if arm=='original':
        checkpoint=INITIAL;state=load_file(str(checkpoint));queries=torch.zeros(8,2560);mode='full'
    else:
        mode,seed=arm.split('-seed');complete=json.loads((STUDY/arm/'complete.json').read_text())
        checkpoint=Path(complete['checkpoint'])/'model.safetensors';assert sha(checkpoint)==complete['model_sha256']
        state=load_file(str(checkpoint));queries=state.pop('visual_queries')
    weights={k:v.float() for k,v in state.items()}
    index=json.loads((TEXT/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(TEXT/index['model.embed_tokens.weight'],framework='pt') as f:embedding=f.get_tensor('model.embed_tokens.weight').float()
    for name in ['embed_tokens.weight','lm_head.weight']:
        if name in weights:assert torch.equal(weights[name],embedding),'Frozen text head changed'
        weights[name]=embedding
    folded=queries.float()@weights['fc.weight'];folded.numpy().tofile(target/'folded-queries-f32.bin')
    et=Path(os.environ['EXECUTORCH_ROOT']);sys.path.insert(0,str(et.parent))
    from executorch.examples.models.qwen3_dflash.convert_weights import dflash_hf_to_meta
    converted=dflash_hf_to_meta(weights)
    assert torch.equal(torch.cat([converted[f'fc.{i}.weight'] for i in range(5)],1),weights['fc.weight'])
    torch.save(converted,target/'draft-meta.pth');embedding.half().numpy().tofile(target/'embedding-f16.bin')
    write(target/'manifest.json',dict(arm=arm,mode=mode,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        source_sha256=sha(__file__),converter_sha256=sha(et/'examples/models/qwen3_dflash/convert_weights.py'),
        contract='B7 root+6mask padded to physical B8; open7; prediction rows0..6; original scalar RoPE with physical context packing',
        weights='BF16 deployed native training checkpoint converted to FP32 for PTQ; frozen original text embedding/head',
        outputs={n:sha(target/n) for n in ['draft-meta.pth','embedding-f16.bin','folded-queries-f32.bin']}))
    print('CONVERTED',arm,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['calibration','convert']);p.add_argument('--arm',default='original');a=p.parse_args();torch.set_num_threads(4)
    calibration() if a.action=='calibration' else convert(a.arm)
