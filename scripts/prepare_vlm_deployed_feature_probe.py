"""Prepare FP32 draft references using actual deployed target hidden features."""
import json
from pathlib import Path
import struct
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
MOBI=ROOT.parent/'MobiSpec.cpp'
sys.path.insert(0,str(MOBI/'tools/training'))
import train_vlm_learned8 as base
from vlm_learned8 import VisualDraft,sha256,atomic_json
from vlm_visual_followup import load_weights

RAW=MOBI/'docs/results/raw/vlm-learned8-followup-20260929/target-feature-diagnostic-v1'
EXPORT=ROOT/'artifacts/vlm-ft-20260929'


def comparison(a,b):
    a,b=a.double().flatten(),b.double().flatten()
    delta=a-b
    return dict(cosine=float(torch.dot(a,b)/(a.norm()*b.norm())),
        rms_ratio=float(a.norm()/b.norm()),rms_difference=float(delta.square().mean().sqrt()),
        mean_abs=float(delta.abs().mean()),max_abs=float(delta.abs().max()))


def main():
    torch.set_num_threads(4)
    capture=json.loads((RAW/'capture.json').read_text());assert capture['complete']
    asset=json.loads((EXPORT/'full_seed42/manifest.json').read_text())
    assert sha256(asset['checkpoint'])==asset['checkpoint_sha256']
    model=VisualDraft.load(base.DRAFT,base.TEXT_TARGET,'cpu').float().eval()
    load_weights(model,asset['checkpoint'])
    output=RAW/'fixtures';output.mkdir(exist_ok=False)
    sources={r['id']:r for r in json.loads((EXPORT/'calibration/manifest.json').read_text())['records']}
    records=[]
    with torch.inference_mode():
        for record in capture['records']:
            folder=Path(record['path'])
            assert sha256(folder/'prompt.bin')==record['sha256']['prompt.bin']
            blob=(folder/'prompt.bin').read_bytes();magic,version,layers,dim,prefix=struct.unpack('<5i',blob[:20])
            assert (magic,version,layers,dim)==(0x4c494654,1,5,2560)
            assert len(blob)==20+4*prefix+4*layers*prefix*dim
            ids=torch.from_numpy(np.frombuffer(blob,dtype='<i4',count=prefix,offset=20).astype('int64'))
            features=torch.from_numpy(np.frombuffer(blob,dtype='<f4',offset=20+4*prefix).copy().reshape(5,prefix,2560).transpose(1,0,2)).contiguous()
            assert torch.isfinite(features).all()
            root=json.loads((folder/'tokens.json').read_text())[0]
            block=torch.full((8,),151669,dtype=torch.long);block[0]=root
            tokens=torch.cat([ids,block])
            result,_=model(features.reshape(prefix,-1),tokens,[prefix],[],None,'full')
            assert tuple(result.shape)==(1,7,151936)
            fixture=output/record['case']['id'];fixture.mkdir()
            features.numpy().tofile(fixture/'features.f32');block.int().numpy().tofile(fixture/'tokens.i32')
            result.float().numpy().tofile(fixture/'proposal-reference.f32')
            source=sources[record['case']['id']];assert sha256(source['path'])==source['sha256']
            teacher=torch.load(source['path'],map_location='cpu',weights_only=False)
            same_ids=prefix==teacher['prompt_length'] and ids.equal(teacher['input_ids'][:prefix].long())
            row=dict(id=record['case']['id'],domain=record['case']['domain'],prefix=prefix,root=root,
                fixture=str(fixture),training_teacher_prompt_tokens_exact=same_ids,
                training_teacher_prompt_length=teacher['prompt_length'],
                training_teacher_root_equal=root==int(teacher['input_ids'][teacher['prompt_length']]),
                fp32_top1=result.argmax(-1).flatten().tolist(),
                sha256={n:sha256(fixture/n) for n in ['features.f32','tokens.i32','proposal-reference.f32']})
            if same_ids:
                ref=teacher['target_features'][:prefix].float()
                images=set(teacher['image_positions'])
                image_rows=torch.tensor(sorted(images),dtype=torch.long)
                text_rows=torch.tensor([i for i in range(prefix) if i not in images],dtype=torch.long)
                row['feature_comparison']=[dict(tap=layer,**comparison(features[:,j],ref[:,j]),
                    image=comparison(features[image_rows,j],ref[image_rows,j]),
                    text=comparison(features[text_rows,j],ref[text_rows,j])) for j,layer in enumerate([1,9,17,25,33])]
            records.append(row);print('DEPLOYED_FEATURE_FIXTURE',row['id'],same_ids,flush=True)
    atomic_json(output/'manifest.json',dict(complete=True,records=records,
        checkpoint_sha256=asset['checkpoint_sha256'],capture_sha256=sha256(RAW/'capture.json'),
        source_sha256=sha256(Path(__file__)),
        scope='FP32 full-FT draft given byte-identical captured NPU target hidden and actual NPU first token',
        limitation='Teacher-vs-device hidden includes vision implementation, target quantization and numerical changes; not an isolated quantization test'))


if __name__=='__main__':main()
