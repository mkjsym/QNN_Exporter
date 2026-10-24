"""Pack native checkpoint weights without changing the established PTQ recipe."""
import argparse,json,os,shutil,sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from vlm_data_scale_export_common import ROOT,MOBI,STUDY,OUT,TEXT,checkpoint,candidate,sha,write

def calibration():
    old=ROOT/'artifacts/vlm-residual8-20261001/export/calibration'
    manifest=json.loads((old/'manifest.json').read_text())
    assert len(manifest['records'])==16
    train={json.loads(line)['id'] for line in (STUDY/'data/train20k.jsonl').read_text().splitlines()}
    for r in manifest['records']:assert r['id'] in train and r['split']=='train' and sha(r['path'])==r['sha256']
    dest=OUT/'calibration';dest.mkdir(parents=True,exist_ok=False)
    for name,digest in manifest['outputs'].items():
        assert sha(old/name)==digest
        shutil.copyfile(old/name,dest/name)
    manifest.update(reused_manifest_sha256=sha(old/'manifest.json'),source_sha256=sha(__file__))
    write(dest/'manifest.json',manifest)

def convert():
    chosen=candidate();key=chosen['key'];mode=chosen['mode']
    assert shutil.disk_usage(OUT).free>50*(1<<30)
    target=OUT/key;target.mkdir(parents=True,exist_ok=False)
    cp=checkpoint(chosen['arm'],chosen['step'])/'model.safetensors';state=load_file(str(cp));state.pop('visual_queries')
    summary={k:state.pop(k).float() for k in ('summary_key.weight','summary_condition.weight','summary_queries')} if mode=='norm8' else None
    torch.save(summary,target/'summary.pt')
    if summary:torch.cat([summary[k].flatten() for k in ('summary_key.weight','summary_condition.weight','summary_queries')]).numpy().tofile(target/'summary-f32.bin')
    weights={k:v.float() for k,v in state.items()}
    index=json.loads((TEXT/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(TEXT/index['model.embed_tokens.weight'],framework='pt') as f:embedding=f.get_tensor('model.embed_tokens.weight').float()
    for name in ('embed_tokens.weight','lm_head.weight'):
        assert torch.equal(weights[name],embedding)
        weights[name]=embedding
    et=Path(os.environ['EXECUTORCH_ROOT']);sys.path.insert(0,str(et.parent))
    from executorch.examples.models.qwen3_dflash.convert_weights import dflash_hf_to_meta
    converted=dflash_hf_to_meta(weights)
    assert torch.equal(torch.cat([converted[f'fc.{i}.weight'] for i in range(5)],1),weights['fc.weight'])
    torch.save(converted,target/'draft-meta.pth');embedding.half().numpy().tofile(target/'embedding-f16.bin')
    names=['draft-meta.pth','embedding-f16.bin','summary.pt']+(['summary-f32.bin'] if summary else [])
    write(target/'manifest.json',dict(arm=chosen['arm'],step=chosen['step'],mode=mode,key=key,checkpoint=str(cp),checkpoint_sha256=sha(cp),
        selection_sha256=sha(STUDY/'mobile-candidate.json'),source_sha256=sha(__file__),converter_sha256=sha(et/'examples/models/qwen3_dflash/convert_weights.py'),
        contract='B7 root+6mask padded to physical B8; open7; original scalar RoPE with physical context packing',
        outputs={name:sha(target/name) for name in names}))
    print('DATA_SCALE_CONVERTED',key,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['calibration','convert']);a=p.parse_args();torch.set_num_threads(4)
    calibration() if a.action=='calibration' else convert()
