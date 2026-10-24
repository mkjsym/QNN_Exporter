"""Archive the frozen head weights and QDQ arithmetic, without recalibration."""
import argparse
import json
import os
from pathlib import Path
import sys

ET=Path(os.environ['EXECUTORCH_ROOT']).resolve();sys.path.insert(0,str(ET.parent))
import torch
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers  # register QDQ ops
from export_efpv_split_target import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('frozen',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    assert not a.output.exists();torch.set_num_threads(8)
    manifest=json.loads((a.frozen.parent/'frozen-manifest.json').read_text())
    assert sha(a.frozen)==manifest['prefill_forward']['sha256']
    ep=torch.export.load(a.frozen);model=ep.module()
    def paths(n):return [x[0] for x in n.meta.get('nn_module_stack',{}).values()]
    norm,=[n for n in model.graph.nodes if n.op=='call_function' and 'norm' in paths(n) and 'rms_norm' in str(n.target)]
    conv,=[n for n in model.graph.nodes if n.op=='call_function' and 'output.conv' in paths(n) and 'conv2d' in str(n.target)]
    weight=conv.args[1];assert str(weight.target)=='quantized_decomposed.dequantize_per_channel.default'
    def value(n):
        assert n.op=='get_attr';v=model
        for part in n.target.split('.'):v=getattr(v,part)
        return v.detach().contiguous().cpu()
    w,scales,zeros=map(value,weight.args[:3])
    assert w.shape==(151936,2560,1,1) and w.dtype==torch.int8
    assert scales.numel()==zeros.numel()==151936
    assert weight.args[3:]==(0,-127,127,torch.int8),weight.args[3:]
    normq,=[n for n in norm.users if 'quantize_per_tensor.default' in str(n.target)]
    convq,=[n for n in conv.users if 'quantize_per_tensor.default' in str(n.target)]
    def quant(node):
        return dict(scale=node.args[1],zero_point=node.args[2],qmin=node.args[3],qmax=node.args[4],dtype=str(node.args[5]))
    chain=[];node=conv.args[0]
    while node is not norm:
        chain.append(dict(name=node.name,target=str(node.target),args=[x.name if isinstance(x,torch.fx.Node) else str(x) for x in node.args]))
        node=node.args[0];assert isinstance(node,torch.fx.Node)
        assert len(chain)<20
    a.output.mkdir();files={}
    for name,tensor in [('weight-i8.bin',w),('scale-f32.bin',scales.float()),('zero-i32.bin',zeros.int())]:
        path=a.output/name;tensor.numpy().tofile(path)
        files[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),bytes=path.stat().st_size,sha256=sha(path))
    contract=dict(frozen_sha256=sha(a.frozen),source_sha256=sha(__file__),files=files,
        norm_quantization=quant(normq),logit_quantization=quant(convq),norm_to_conv_chain=chain,
        weight_quantization=dict(axis=0,qmin=-127,qmax=127,zero_min=int(zeros.min()),zero_max=int(zeros.max())),
        qnn_integer_rounding_or_accumulation_proven=False,
        scope='Frozen mathematical QDQ contract. Does not certify QNN hardware implementation or GPU agreement.')
    (a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n');print(json.dumps(contract,indent=2))


if __name__=='__main__':main()
