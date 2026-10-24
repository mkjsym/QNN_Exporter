"""Map all DFlash2 weights to their QNN-graph or CPU-selector owner."""
import json
import os
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file
from prepare_dflash2 import OUT, WEIGHT_SHA, sha

ET = Path(os.environ['EXECUTORCH_ROOT']).resolve()
sys.path.insert(0,str(ET.parent))
MOBI = Path(__file__).resolve().parents[2]/'MobiSpec.cpp'
sys.path.insert(0,str(MOBI/'gguf-py'))
import gguf
from executorch.examples.models.qwen3_dflash.convert_weights import dflash_hf_to_meta


def main():
    torch.set_num_threads(4)
    assert sha(OUT/'checkpoint/model.safetensors') == WEIGHT_SHA
    sd = load_file(str(OUT/'checkpoint/model.safetensors'))
    out = OUT/'npu-assets';out.mkdir(exist_ok=True)
    converted = {}; owners = {}; normal = {}
    writer = gguf.GGUFWriter(out/'selector-bf16.gguf','dflash')
    writer.add_uint32('dflash.selector_rank',256);writer.add_uint32('dflash.selector_top_k',16)
    writer.add_uint32('dflash.block_size',9)
    for name,value in sd.items():
        if name.endswith(('predecessor_codebook','successor_codebook')):
            dest = 'selector_prev.weight' if 'predecessor' in name else 'selector_next.weight'
            data = gguf.quants.quantize(value.float().numpy(),gguf.GGMLQuantizationType.BF16)
            writer.add_tensor(dest,data,raw_dtype=gguf.GGMLQuantizationType.BF16)
            owners[name] = 'CPU selector: '+dest
        elif name == 'candidate_selector.hidden_projection.weight':
            converted['selector_gate.weight'] = value;owners[name] = 'QNN: selector_gate.weight'
        elif '.attention_conv.' in name or '.mlp_conv.' in name:
            converted[name] = value;owners[name] = 'QNN: '+name
        else:
            normal[name] = value;owners[name] = 'Standard DFlash mapping'
    converted.update(dflash_hf_to_meta(normal))
    torch.save(converted,out/'draft-meta.pth')
    embedding = sd['embed_tokens.weight'].half().contiguous()
    assert torch.isfinite(embedding).all()
    embedding.numpy().tofile(out/'embedding-f16.bin')
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    base = Path(__file__).resolve().parents[1]/'artifacts/long-draft-20260917/assets-b16-v1'
    calibration = json.loads((base/'manifest.json').read_text())
    for name in ('calibration-hidden.pt','calibration-prompts.json'):
        assert sha(base/name)==calibration['outputs'][name]
        dest=out/name
        if not dest.exists():dest.symlink_to((base/name).resolve())
    # Explicit ownership permits two host tensors to be absent from the graph checkpoint.
    assert len(owners)==83 and len(converted)==85
    result=dict(checkpoint_sha256=WEIGHT_SHA,source_sha256=sha(__file__),owners=owners,
                graph_keys=sorted(converted),calibration_manifest_sha256=sha(base/'manifest.json'),
                calibration_train_cases=calibration['calibration_train_cases'],
                outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DFLASH2_NPU_ASSETS_READY',len(converted),flush=True)


if __name__=='__main__':main()
