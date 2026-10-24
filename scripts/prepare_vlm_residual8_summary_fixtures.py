"""Small CPU summary assets and train-only native-format packing fixtures."""
import json,struct
import torch
from safetensors import safe_open
from vlm_residual8_export_common import ROOT,MOBI,STUDY,BASE,OUT,pack,sha,write


def main():
    torch.set_num_threads(2)
    out=OUT.parent/'summary-fixtures';out.mkdir(parents=True,exist_ok=False)
    modes=('norm8','residual8','staticres8');assets={}
    for mode in modes:
        done=json.loads((STUDY/f'{mode}-seed42/complete.json').read_text())
        with safe_open(done['checkpoint']+'/model.safetensors',framework='pt') as f:
            summary={k:f.get_tensor(k).float() for k in ('summary_key.weight','summary_condition.weight','summary_queries')}
        path=out/(mode+'.bin');torch.cat([summary[k].flatten() for k in summary]).numpy().tofile(path)
        assets[mode]=summary
    records=json.loads((BASE/'cache/train/manifest.json').read_text())['records']
    selected=[r for d in ('general','ocr') for r in [r for r in records if r['domain']==d][:3]]
    checks=[]
    for r in selected:
        sample=torch.load(BASE/'cache/train'/(r['id']+'.pt'),weights_only=True,mmap=True)
        n=sample['prompt_length'];features=sample['target_hidden_states'][:n];groups=sample['image_groups'][:n]
        first=int((groups>=0).nonzero()[0]);h,w=sample['grid_hw']
        fixture=out/(r['id']+'.bin')
        with fixture.open('wb') as f:f.write(struct.pack('5i',n,2560,first,h,w));f.write(features.reshape(n,5,2560).transpose(0,1).float().contiguous().numpy().tobytes())
        for mode,summary in assets.items():
            expected,pos=pack(features,groups,mode,summary,n)
            path=out/f"{r['id']}-{mode}.pt";torch.save(dict(hidden=expected.float(),positions=pos,text=groups[pos]<0),path)
            checks.append(dict(id=r['id'],mode=mode,input=str(fixture),assets=str(out/(mode+'.bin')),expected=str(path),input_rows=n,grid_hw=[h,w]))
    write(out/'manifest.json',dict(checks=checks,source_sha256=sha(__file__),pack_source_sha256=sha(ROOT/'scripts/vlm_residual8_export_common.py'),files={p.name:sha(p) for p in out.iterdir()}))
    print('SUMMARY_FIXTURES_READY',len(checks))

if __name__=='__main__':main()
