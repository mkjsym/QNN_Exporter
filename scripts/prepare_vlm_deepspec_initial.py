"""Recover the retained BF16 b7 backbone for matched DeepSpec initialization."""
import hashlib,json,sys
from pathlib import Path
import torch
from safetensors.torch import save_file
Q=Path(__file__).resolve().parents[1]
M=Q.parent/'MobiSpec.cpp'
sys.path.insert(0,str(M/'gguf-py'))
import gguf

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 out=Q/'artifacts/vlm-deepspec-20260930/initial';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'backbone.safetensors').exists() and not (out/'manifest.json').exists()
 source=Q.parent/'dflash-models/Qwen3-4B-DFlash-b7-bf16.gguf'
 rename={'fc.weight':'fc.weight','enc.output_norm.weight':'hidden_norm.weight','output_norm.weight':'norm.weight'}
 suffixes={'attn_norm.weight':'input_layernorm.weight','ffn_norm.weight':'post_attention_layernorm.weight',
  'ffn_gate.weight':'mlp.gate_proj.weight','ffn_up.weight':'mlp.up_proj.weight','ffn_down.weight':'mlp.down_proj.weight',
  'attn_q.weight':'self_attn.q_proj.weight','attn_k.weight':'self_attn.k_proj.weight','attn_v.weight':'self_attn.v_proj.weight',
  'attn_output.weight':'self_attn.o_proj.weight','attn_q_norm.weight':'self_attn.q_norm.weight','attn_k_norm.weight':'self_attn.k_norm.weight'}
 for i in range(5):
  for a,b in suffixes.items():rename[f'blk.{i}.{a}']=f'layers.{i}.{b}'
 reader=gguf.GGUFReader(str(source));assert {t.name for t in reader.tensors}==rename.keys()
 state={}
 for t in reader.tensors:
  assert t.tensor_type in [gguf.GGMLQuantizationType.BF16,gguf.GGMLQuantizationType.F32],(t.name,t.tensor_type)
  value=torch.from_numpy(gguf.quants.dequantize(t.data,t.tensor_type).copy())
  assert torch.equal(value,value.to(torch.bfloat16).float()), 'Source tensor is not exactly BF16 representable: '+t.name
  state[rename[t.name]]=value.to(torch.bfloat16).contiguous()
 assert len(state)==58
 save_file(state,str(out/'backbone.safetensors'))
 record=dict(source=str(source),source_sha256=sha(source),backbone_sha256=sha(out/'backbone.safetensors'),
             source_general_name=reader.fields['general.name'].contents(),tensors=58,initialization='Same retained mobile DFlash b7 backbone; no new weights downloaded',
             train_contract='DeepSpec DFlash B7: anchor + six masks; rows0..6 predict next seven tokens; open7',
             shapes={k:list(v.shape) for k,v in state.items()})
 (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 print('INITIAL_B7_READY',record['backbone_sha256'])
if __name__=='__main__':main()
