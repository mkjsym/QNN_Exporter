"""Fused QNN DFlash2 body with an explicit packed unary/gate output contract."""
import os
from pathlib import Path
import sys

import torch
from torch import nn

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0, str(ET.parent))
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import DFlashDecoderLayer
from export_draft_body import DraftBodyModel


class DFlash2Conv(nn.Module):
    def __init__(self, hidden_size=2560, group_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.group_size = group_size
        self.groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2,2,hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 4*self.groups, bias=False)

    def convolve(self, hidden, dynamic, side):
        b, n, _ = hidden.shape
        values = hidden.reshape(b,n,self.groups,self.group_size)
        base = self.base_kernel[side].reshape(2,self.groups,self.group_size)
        # Keep the same nonempty operator sequence for decode and prefill encoding transfer.
        previous = torch.cat([torch.zeros_like(values[:,:1]),values],dim=1)[:,:n]
        return (values * (base[0] + dynamic[:,:,0,:,None]) +
                previous * (base[1] + dynamic[:,:,1,:,None])).reshape(b,n,self.hidden_size)

    def prepare(self, hidden):
        b,n,_ = hidden.shape
        dynamic = self.kernel_projection(hidden).reshape(b,n,2,2,self.groups)
        return self.convolve(hidden,dynamic[:,:,0],0), dynamic[:,:,1]

    def finish(self, hidden, dynamic):
        return self.convolve(hidden,dynamic,1)


class DFlash2Layer(DFlashDecoderLayer):
    def __init__(self, layer_idx, config, output_new_cache_only=True):
        super().__init__(layer_idx,config,output_new_cache_only)
        self.attention_conv = DFlash2Conv(config.dim)
        self.mlp_conv = DFlash2Conv(config.dim)

    def forward(self,x,g_ctx,freqs_cos,freqs_sin,freqs_cos_ctx,freqs_sin_ctx,
                atten_mask,k_caches,v_caches):
        h, dynamic = self.attention_conv.prepare(self.attention_norm(x))
        h,k,v = self.attention(hidden_states=h,g_ctx=g_ctx,freqs_cos=freqs_cos,
                               freqs_sin=freqs_sin,freqs_cos_ctx=freqs_cos_ctx,
                               freqs_sin_ctx=freqs_sin_ctx,atten_mask=atten_mask,
                               k_caches=k_caches,v_caches=v_caches)
        h = x + self.attention_conv.finish(h,dynamic)
        m,dynamic = self.mlp_conv.prepare(self.ffn_norm(h))
        return h + self.mlp_conv.finish(self.feed_forward(m),dynamic),k,v


class DFlash2PackedModel(DraftBodyModel):
    def __init__(self,config,*args,**kwargs):
        super().__init__(config,*args,**kwargs)
        assert config.dim == 2560 and config.n_layers == 5
        assert self.max_context_len <= 2048
        self.layers = nn.ModuleList([DFlash2Layer(i,config,kwargs.get('output_new_cache_only',True)) for i in range(config.n_layers)])
        self.selector_gate = nn.Linear(config.dim,256,bias=False)

    def forward(self,h_0,h_1,h_2,h_3,h_4,block_embd,atten_mask,input_pos,input_pos_ctx,*args):
        hidden,keys,values = super().forward(h_0,h_1,h_2,h_3,h_4,block_embd,
                                           atten_mask,input_pos,input_pos_ctx,*args)
        assert self.output_cache and not self.logits_scaling
        packed = torch.cat([self.output(hidden),self.selector_gate(hidden)],dim=-1)
        return packed.unsqueeze(0).squeeze(0),keys,values
