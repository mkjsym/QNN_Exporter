"""FP32 verification of original-position packed KV and physical B8/open7 export."""
import argparse,json,os,sys
from pathlib import Path
import torch
from vlm_residual8_export_common import OUT,sha,write
ET=Path(os.environ['EXECUTORCH_ROOT']);sys.path.insert(0,str(ET.parent))
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDraftModel

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args();torch.set_num_threads(4)
    asset=OUT/a.arm;manifest=json.loads((asset/'manifest.json').read_text())
    cfg=json.loads((ET/'examples/models/qwen3_dflash/config/4b_dflash_config.json').read_text())
    config=ModelArgs(dim=cfg['dim'],n_layers=cfg['n_layers'],n_heads=cfg['n_heads'],n_kv_heads=cfg['n_kv_heads'],head_dim=cfg['head_dim'],hidden_dim=cfg['hidden_dim'],
        vocab_size=cfg['vocab_size'],norm_eps=cfg['norm_eps'],rope_freq_base=cfg['rope_theta'],use_hf_rope=True,use_qk_norm=True,qk_norm_before_rope=True,
        max_seq_len=1024,max_context_len=1024,use_kv_cache=True,max_batch_size=1)
    graph=DFlashDraftModel(config,ar_len=8,n_inject=16,n_target_layers=5).eval()
    graph.load_state_dict(torch.load(asset/'draft-meta.pth',weights_only=True,mmap=True),strict=True,assign=True)
    refs=json.loads((asset/'native-references.json').read_text());checks=[]
    for record in refs['records']:
        assert sha(record['path'])==record['sha256'];item=torch.load(record['path'],weights_only=True)
        features=item['context'].reshape(-1,5,2560);positions=item['context_positions'];prefix=len(features)
        keys,values=graph.get_example_inputs()[5:]
        for begin in range(0,prefix,16):
            end=min(prefix,begin+16);take=end-begin
            hidden=[torch.zeros(1,16,2560) for _ in range(5)]
            for layer in range(5):hidden[layer][0,:take]=features[begin:end,layer]
            block=torch.full((1,8),151669,dtype=torch.long);block[0,0]=item['root']
            mask=torch.full((1,1,8,1024),-255.);mask[...,:begin]=0;mask[...,1000:1000+take]=0;mask[...,-8:-1]=0
            ctx_pos=torch.zeros((1,16),dtype=torch.int32);ctx_pos[0,:take]=positions[begin:end].int()
            block_pos=torch.arange(item['original_prefix'],item['original_prefix']+8,dtype=torch.int32)[None]
            result,new_keys,new_values=graph(*hidden,graph.tok_embeddings(block),mask,block_pos,ctx_pos,*keys,*values)
            for layer in range(5):
                keys[layer][...,begin:end]=new_keys[layer][...,:take];values[layer][:,:,begin:end,:]=new_values[layer][:,:,:take,:]
        actual=result[:,:7];reference=item['reference_logits'];diff=(actual-reference).abs();equal=actual.argmax(-1).equal(reference.argmax(-1))
        checks.append(dict(id=record['id'],prefix=record['prefix'],packed_rows=prefix,max_abs=diff.max().item(),top1_exact=equal))
        assert equal,checks[-1];torch.testing.assert_close(actual,reference,atol=.003,rtol=.001);print('CHECK',checks[-1],flush=True)
    write(asset/'fp32-audit.json',dict(passed=True,checks=checks,source_sha256=sha(__file__),checkpoint_sha256=manifest['checkpoint_sha256'],graph_sha256=manifest['outputs']['draft-meta.pth']))

if __name__=='__main__':main()
