"""Freeze rep5 and the request's ood10, and make host vision fixtures."""
import json, os, subprocess, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MOBI=ROOT.parent/'MobiSpec.cpp'
BUNDLE=Path('/data/regspec/mobile-request-20261109')
OUT=Path('/data/youngmin/qnn-artifacts/regspec-s26-20261109/fixtures')
STUDY=MOBI/'docs/results/raw/regspec-s26-20261109'
OLD=MOBI/'docs/results/raw/vlm-register-mobile-20261019'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ids=json.loads((OLD/'inputs-rep5.json').read_text())['ids']
    data=MOBI/'docs/results/raw/vlm-register-doc20k-20261016/data/validation.jsonl'
    val={r['id']:r for r in map(json.loads,data.read_text().splitlines())}
    rows=[]
    for i in ids: rows.append(dict(val[i],subset='rep5'))
    for r in map(json.loads,(BUNDLE/'ood10/ood10.jsonl').read_text().splitlines()):
        image=BUNDLE/'ood10/img'/Path(r['image']).name
        assert sha(image)==r['image_sha256']
        rows.append(dict(r,image=str(image),subset='ood10'))
    cases=[]
    suffixes=['.json','.tokens.i32','.positions.i32','.vlm_embd.f32','.vlm_deepstack_0.f32','.vlm_deepstack_1.f32','.vlm_deepstack_2.f32']
    for r in rows:
        name=r['id']; dest=OUT/name
        if r['subset']=='rep5':
            old=ROOT/'artifacts/vlm-register-mobile-20261019/fixtures-m1'
            for suffix in suffixes:
                p=Path(str(dest)+suffix)
                if not p.exists():p.symlink_to(old/(name+suffix))
        elif not Path(str(dest)+'.json').exists():
            env=dict(os.environ,QNN_VLM_DUMP=str(dest),QNN_VLM_EMBD=str(ROOT.parent/'specinfer_QNN/experiments/vlm-dflash-20260913/target_embeddings.f16'))
            cmd=[str(MOBI/'build-host-vlm/bin/llama-qnn-vlm-prepare'),'-m','/data/llamacpp_bench/models/Qwen3-VL-4B-Instruct-BF16.gguf',
                '--mmproj','/data/llamacpp_bench/models/mmproj/mmproj-Qwen3VL-4B-F16.gguf','--image',r['image'],
                '--image-max-tokens','3072','-p',r['prompt'],'-t','8']
            with (OUT/(name+'.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        meta=json.loads(Path(str(dest)+'.json').read_text())
        cases.append(dict(id=name,subset=r['subset'],domain=r['domain'],prompt=r['prompt'],local_image=r['image'],
            image=name+Path(r['image']).suffix,sha256=sha(r['image']),cap=3072,fixtures=str(OUT),image_tokens_host=meta['image_rows']))
        print('FIXTURE',name,meta['image_rows'],flush=True)
    (STUDY/'inputs.json').write_text(json.dumps(dict(cases=cases,rule='existing rep5; first two inputs of each requested OOD subset, no reselection'),indent=2)+'\n')
    print('REGSPEC_INPUTS_COMPLETE',len(cases),flush=True)

if __name__=='__main__':main()
