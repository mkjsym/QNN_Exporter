"""Derive a body/head split from a saved target QDQ graph, without PTQ.

The split is inserted after the final normalized activation in the QNN capture
pipeline. Both sides retain that activation's existing quantization encoding.
"""
import argparse
import copy
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ET=Path(os.environ['EXECUTORCH_ROOT']).resolve();sys.path.insert(0,str(ET.parent))
import torch
from executorch.backends.qualcomm._passes.fold_qdq import FoldQDQ
from executorch.backends.qualcomm._passes.tag_quant_io import TagQuantIO
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,QCOM_QUANT_ATTRS)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,to_edge_transform_and_lower_to_qnn,update_spill_fill_size)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import TextDecoder,get_qnn_pass_manager_cls
from executorch.extension.llm.custom_ops.model_sharding import get_split_graph_pass,SplitGraph
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass,PassResult
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


class SplitFinalNorm(ExportPass):
    def __init__(self,audit_path):
        super().__init__();self.audit_path=Path(audit_path)

    def call(self,graph_module):
        candidates=[]
        for node in graph_module.graph.nodes:
            paths=[v[0] for v in node.meta.get('nn_module_stack',{}).values()]
            if node.op=='call_function' and 'norm' in paths and 'rms_norm' in str(node.target):
                candidates.append(node)
        assert len(candidates)==1,[(n.name,str(n.target)) for n in candidates]
        source=candidates[0]
        assert list(source.meta['val'].shape) in ([1,1,2560],[1,32,2560])
        assert QCOM_QUANT_ATTRS in source.meta,'Missing existing final-norm encoding'
        users=list(source.users)
        assert users
        with graph_module.graph.inserting_after(source):
            cut=graph_module.graph.call_function(exir_ops.edge.llama.fallback.default,(source,))
            cut.meta=copy.copy(source.meta)
        for user in users:user.replace_input_with(source,cut)
        graph_module.recompile()
        info=dict(source=source.name,cut=cut.name,shape=list(source.meta['val'].shape),
            quantization={k:str(v) for k,v in source.meta[QCOM_QUANT_ATTRS].items()},
            original_users=[n.name for n in users],new_quantization_observers=False)
        self.audit_path.write_text(json.dumps(info,indent=2)+'\n')
        return PassResult(graph_module,True)


def main():
    p=argparse.ArgumentParser();p.add_argument('frozen_dir',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--no-head-split',action='store_true',help='Replay A1/A32 lowering as a full control')
    args=p.parse_args();args.output.mkdir()
    manifest=json.loads((args.frozen_dir/'frozen-manifest.json').read_text())
    torch.set_num_threads(8)
    models={};inputs={};all_passes={};all_deps={};source_hashes={}
    for name,ar in [('kv_forward',1),('prefill_forward',32)]:
        source=args.frozen_dir/(name+'-frozen.pt2')
        assert sha(source)==manifest[name]['sha256'];source_hashes[name]=sha(source)
        ep=torch.export.load(source);models[name]=ep.module();inputs[name]=ep.example_inputs[0]
        tagger=TextDecoder.__new__(TextDecoder)
        tagger.control_args=SimpleNamespace(decoder_model='qwen3-4b-sqnr')
        tagger.meta=copy.deepcopy(manifest['constant_methods']);tagger.meta['get_ar_len']=ar
        tagger.io_shape={(1,ar,151936)}
        tagger.kv_cache_shape={(128,1024),(1024,128),(128,ar),(ar,128)}
        cls=get_qnn_pass_manager_cls();passes=cls.get_capture_program_passes();deps=cls.get_passes_dependency_for_capture_program()
        _,setting=get_split_graph_pass(36,6,pattern=r'^(?:layers)\.(\d+)',shard_layers=[2,10,18,26,34])
        passes[SplitGraph]=setting;deps[SplitGraph]=[FoldQDQ]
        if not args.no_head_split:
            passes[SplitFinalNorm]={QCOM_PASS_ACTIVATE_KEY:True,
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY:{'audit_path':str(args.output/(name+'-head-cut.json'))}}
            deps[SplitFinalNorm]=[SplitGraph];deps[TagQuantIO]=[SplitFinalNorm]
        else:deps[TagQuantIO]=[SplitGraph]
        passes[TagQuantIO][QCOM_PASS_ACTIVATE_KEY]=True
        passes[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]['get_quant_io_dtype_fn']=partial(
            tagger._tag_ios,fixed_point_type={'kv_type':torch.uint8,'io_type':torch.uint16})
        all_passes[name]=passes;all_deps[name]=deps
        del ep;gc.collect()
    spec=generate_qnn_executorch_compiler_spec(soc_model=get_soc_to_chipset_map()['SM8850'],
        backend_options=generate_htp_compiler_spec(use_fp16=False,use_multi_contexts=True,use_weight_sharing=True),
        shared_buffer=True,use_mha2sha=True)
    plan=dict(frozen_source_sha256=source_hashes,source_sha256=sha(__file__),
        mode='full-A1/A32' if args.no_head_split else 'body-A1/A32+head-A1/A32',
        new_calibration=False,device_equivalence_established=False)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    edge=to_edge_transform_and_lower_to_qnn(module=models,inputs=inputs,
        compiler_specs={name:spec for name in models},constant_methods=manifest['constant_methods'],
        passes_job=all_passes,dep_table=all_deps,skip_node_op_set={'llama.fallback.default'})
    for name in models:update_spill_fill_size(edge.exported_program(name))
    program=edge.to_executorch(ExecutorchBackendConfig(memory_planning_pass=MemoryPlanningPass(
        alloc_graph_input=False,alloc_graph_output=False),passes=[BuildQuantIo()]))
    with (args.output/'hybrid_llama_qnn.pte').open('wb') as f:program.write_to_file(f)
    plan['pte_sha256']=sha(args.output/'hybrid_llama_qnn.pte');plan['export_complete']=True
    (args.output/'complete.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('EFPV_SPLIT_EXPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
