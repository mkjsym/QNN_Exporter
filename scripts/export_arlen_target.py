"""Export fixed Qwen3-4B verify widths with unchanged PTQ inputs and recipe."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ar', type=int, required=True)
    parser.add_argument('--precision', choices=['W4A16', 'W4A8', 'W8A16', 'W4A16-deployed', 'W4A8-protected'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.ar in (1, 8, 16, 32, 64)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    et = Path(os.environ['EXECUTORCH_ROOT'])
    sdk = Path(os.environ['QNN_SDK_ROOT'])
    sys.path[:0] = [str(et.parent), str(et)]
    import numpy as np
    import torch
    import subprocess
    from executorch.examples.qualcomm.oss_scripts.llama import llama, SUPPORTED_LLM_MODELS
    from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
        StaticLLMQuantRecipe, QuantRecipe, QuantGranularity, QuantDtype,
        MinMaxObserver, annotate_kv_8bit, Qwen3_4B_SqnrQuantRecipe)
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
    from executorch.backends.qualcomm.utils.utils import PyQnnManagerAdaptor
    seed = 260928
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    dtype = {'W4A16': QuantDtype.use_16a4w, 'W4A8': QuantDtype.use_8a4w,
             'W8A16': QuantDtype.use_16a8w, 'W4A16-deployed': QuantDtype.use_16a4w,
             'W4A8-protected': QuantDtype.use_16a4w}[args.precision]

    class ControlledRecipe(StaticLLMQuantRecipe):
        default_quant_dtype = dtype

        def __init__(self, verbose=False):
            super().__init__()
            self.recipe = QuantRecipe(dtype, False, act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR, verbose=verbose)
            self.recipe.add_node_target({torch.ops.aten.conv2d.default}, dtype, False,
                act_observer=MinMaxObserver, granularity=QuantGranularity.PER_CHANNEL)
            self.recipe.custom_quant_annotations.append(annotate_kv_8bit)

    class ProtectedA8Recipe(StaticLLMQuantRecipe):
        default_quant_dtype = QuantDtype.use_16a4w

        def __init__(self, verbose=False):
            super().__init__()
            self.recipe = QuantRecipe(self.default_quant_dtype, False,
                act_observer=MinMaxObserver, granularity=QuantGranularity.PER_TENSOR, verbose=verbose)
            self.recipe.add_node_target({torch.ops.aten.conv2d.default}, QuantDtype.use_8a4w, False,
                act_observer=MinMaxObserver, granularity=QuantGranularity.PER_CHANNEL)
            for layers, submodule, projection in Qwen3_4B_SqnrQuantRecipe._SENSITIVE:
                ids = "|".join(str(i) for i in layers)
                self.recipe.add_regex({rf"layers\.({ids})\.{submodule}\.forward\.__self__\.{projection}_conv"},
                    QuantDtype.use_16a8w, False, act_observer=MinMaxObserver,
                    granularity=QuantGranularity.PER_CHANNEL)
            self.recipe.add_regex({r"output\.conv"}, QuantDtype.use_16a8w, False,
                act_observer=MinMaxObserver, granularity=QuantGranularity.PER_CHANNEL)
            self.recipe.custom_quant_annotations.append(annotate_kv_8bit)

    config = copy.copy(SUPPORTED_LLM_MODELS['qwen3-4b-sqnr'])
    deployed = args.precision == 'W4A16-deployed'
    if not deployed:
        object.__setattr__(config, 'quant_recipe', ProtectedA8Recipe if args.precision == 'W4A8-protected' else ControlledRecipe)
    SUPPORTED_LLM_MODELS['qwen3-4b-sqnr'] = config
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path('/home/youngmin/.cache/meta_checkpoints/Qwen_Qwen3-4B.pth')
    calibration = root / 'data/calib_specbench.json'
    argv = ['--decoder_model', 'qwen3-4b-sqnr', '--model_mode', 'hybrid',
            '--prefill_ar_len', str(args.ar), '--max_context_len', '1024',
            '--max_seq_len', '1024', '--soc_model', 'SM8850',
            '--build_folder', str(et / 'build-android'), '--artifact', str(out),
            '--compile_only', '--checkpoint', str(checkpoint),
            '--calib_samples', str(calibration), '--prompt', 'Once upon a time']
    settings = llama._build_parser().parse_args(argv)
    plan = dict(ar=args.ar, context=1024, seed=seed, argv=argv,
                checkpoint_sha256=sha(checkpoint), calibration_sha256=sha(calibration),
                exporter_sha256=sha(__file__),
                precision=args.precision, weight_granularity=(
                    'deployed mixed SQNR recipe: W4 block16, sensitive projections and LM head W8 per-channel'
                    if deployed else ('mixed W4A8 per-channel linears; SQNR-sensitive projections and LM head W8A16; non-linear/residual defaults A16'
                                     if args.precision == 'W4A8-protected' else 'per-channel for all conv/linear, including LM head')), 
                precision_exceptions='KV8 common; embedding/norm and graph boundary/logit storage audited separately',
                source_sha256={str(p.relative_to(et)): sha(p) for p in [
                    et / 'examples/qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py',
                    et / 'examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py',
                    et / 'examples/qualcomm/oss_scripts/llama/model/static_llama.py']})
    write(out / 'plan.json', plan)
    original = llm_wrappers.to_edge_transform_and_lower_to_qnn

    class ContextsSaved(Exception):
        pass

    def lower(*positional, **kw):
        assert not positional and set(kw['module']) == {'kv_forward', 'prefill_forward'}
        encodings = {}
        for name, model in kw['module'].items():
            records = []
            def encode(value):
                if isinstance(value, torch.fx.Node):
                    if value.op == 'get_attr':
                        tensor = model
                        for part in value.target.split('.'):
                            tensor = getattr(tensor, part)
                        if isinstance(tensor, torch.Tensor):
                            array = tensor.detach().cpu().contiguous().numpy()
                            return dict(shape=list(array.shape), dtype=str(array.dtype),
                                        sha256=hashlib.sha256(array.tobytes()).hexdigest())
                    return dict(op=value.op, target=str(value.target))
                if isinstance(value, (list, tuple)):
                    return [encode(x) for x in value]
                return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
            for node in model.graph.nodes:
                if 'quantized_decomposed' in str(node.target) and 'quantize' in str(node.target):
                    records.append(dict(op=str(node.target), parameters=encode(node.args[1:])))
            assert records
            digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            encodings[name] = dict(records=len(records), sha256=digest)
            write(out / (name + '-encodings.json'), records)
        write(out / 'encoding-digests.json', encodings)
        edge = original(**kw)
        contexts = out / 'context'
        contexts.mkdir()
        by_method = {}
        for name in ('kv_forward', 'prefill_forward'):
            modules = [m for m in edge.exported_program(name).graph_module._modules.values()
                       if hasattr(m, 'processed_bytes')]
            assert len(modules) == 6, (name, len(modules))
            digests = []
            for i, module in enumerate(modules):
                raw = PyQnnManagerAdaptor.StripProtocol(module.processed_bytes)
                assert len(raw) > 0
                digest = hashlib.sha256(raw).hexdigest()
                path = contexts / f'forward_{i}.bin'
                if path.exists():
                    assert sha(path) == digest, 'Hybrid graph contexts must share the same compiled binary'
                else:
                    temp = path.with_suffix('.tmp')
                    temp.write_bytes(raw)
                    temp.replace(path)
                    with (out / f'context-{i}.log').open('w') as log:
                        subprocess.run([str(sdk / 'bin/x86_64-linux-clang/qnn-context-binary-utility'),
                                        '--context_binary', str(path), '--json_file',
                                        str(contexts / f'forward_{i}_json.json')], check=True,
                                       stdout=log, stderr=subprocess.STDOUT)
                digests.append(digest)
                del raw
            by_method[name] = digests
        io = []
        for i in range(6):
            raw = json.loads((contexts / f'forward_{i}_json.json').read_text())
            graphs = {g['info']['graphName']: g['info'] for g in raw['info']['graphs']}
            assert set(graphs) == {'kv_forward', 'prefill_forward'}
            for name, width in [('kv_forward', 1), ('prefill_forward', args.ar)]:
                tensors = [t['info'] for t in graphs[name]['graphInputs']]
                masks = [t for t in tensors if 'atten_mask' in t['name']]
                assert len(masks) == 1 and masks[0]['dimensions'] == [1, 1, width, 1024]
                kv = [t for t in tensors if len(t['dimensions']) == 4 and t not in masks]
                assert kv and all(t['dataType'] == 'QNN_DATATYPE_UFIXED_POINT_8' for t in kv)
                assert all(t['dimensions'] in ([1, 8, 128, 1024-width], [1, 8, 1024-width, 128]) for t in kv)
                io.append(dict(shard=i, graph=name, width=width, mask=masks[0]))
        params = json.loads((et / 'examples/models/qwen3/config/4b_config.json').read_text())
        params.update(prefill_ar_len=args.ar, context_length=1024,
                      shard_layers=[2, 10, 18, 26, 34], kv_cache_bits=8)
        write(contexts / 'params.json', params)
        write(out / 'io-audit.json', io)
        write(out / 'complete.json', dict(**plan, elapsed_seconds=time.monotonic()-started,
              encoding_digests=encodings, files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
              for p in contexts.iterdir() if p.is_file()}, context_only=True,
              hybrid_contexts_identical=True))
        print('ARLEN_EXPORT_COMPLETE', args.precision, args.ar, flush=True)
        raise ContextsSaved()

    llm_wrappers.to_edge_transform_and_lower_to_qnn = lower
    started = time.monotonic()
    try:
        llama.export_llama(settings)
    except ContextsSaved:
        pass
    assert (out / 'complete.json').is_file()


if __name__ == '__main__':
    main()
