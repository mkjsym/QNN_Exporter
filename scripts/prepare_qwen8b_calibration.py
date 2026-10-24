"""Capture train-only target states on CPU for the native 8B export."""
import json, hashlib, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/qwen8b-transfer-20260926/calibration'
TARGET=ROOT.parent/'hf-models/Qwen3-8B'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'artifacts/long-draft-20260917/assets-b16-v1/calibration-prompts.json'
    cases=json.loads(source.read_text())[:12]
    assert len(cases)==12
    plan=dict(source=str(source),sha256=sha(source),cases=cases,hidden_indices=[2,10,18,26,34],
        target=str(TARGET),device='cpu',dtype='bfloat16',max_input_tokens=384,
        scope='First 12 frozen train-only calibration inputs from the prior export; no benchmark inputs.')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    tok=AutoTokenizer.from_pretrained(TARGET,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(TARGET,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval()
    blocks=[]
    with torch.inference_mode():
        for i,case in enumerate(cases):
            ids=tok.apply_chat_template(case['messages'],add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False)[:,:384]
            out=model.model(ids,output_hidden_states=True,use_cache=False)
            h=torch.cat([out.hidden_states[j][0] for j in plan['hidden_indices']],dim=-1)
            pick=torch.linspace(0,len(h)-1,min(64,len(h))).round().long()
            blocks.append(h[pick].float());print('CALIBRATION',i,len(ids[0]),flush=True)
    hidden=torch.cat(blocks);assert hidden.shape[1]==20480 and torch.isfinite(hidden).all()
    torch.save(dict(hidden=hidden,source='CPU BF16 official Qwen3-8B train-only calibration'),OUT/'hidden.pt')
    (OUT/'prompts.json').write_text(json.dumps(cases,indent=2)+'\n')
    (OUT/'complete.json').write_text(json.dumps(dict(rows=len(hidden),outputs={n:sha(OUT/n) for n in ['hidden.pt','prompts.json','plan.json']}),indent=2)+'\n')
if __name__=='__main__':main()
