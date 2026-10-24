"""Prepare unchanged FP32 DSpark Markov weights for the full-input PC audit."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sparse-domino-cost-20260922/pc-audit-assets'
CHECKPOINT=Path('/home/youngmin/.cache/huggingface/hub/models--deepseek-ai--dspark_qwen3_4b_block7/snapshots/3457dff1417cb84927f6098a5fcb7cee85c934b7/model.safetensors')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);tensors={}
    assert not (OUT/'manifest.json').exists()
    with safe_open(str(CHECKPOINT),framework='pt') as f:
        for short,name in [('w1','markov_head.markov_w1.weight'),('w2','markov_head.markov_w2.weight')]:
            x=f.get_tensor(name);original_dtype=str(x.dtype);x=x.float().contiguous()
            assert x.shape==(151936,256) and torch.isfinite(x).all()
            path=OUT/(short+'.f32')
            if path.exists():assert sha(path)==hashlib.sha256(x.numpy().tobytes()).hexdigest()
            else:x.numpy().tofile(path)
            tensors[name]=dict(path=str(path),sha256=sha(path),shape=list(x.shape),source_dtype=original_dtype)
    (OUT/'manifest.json').write_text(json.dumps(dict(tensors=tensors,checkpoint=str(CHECKPOINT),
        checkpoint_sha256=sha(CHECKPOINT),source_sha256=sha(__file__),
        scope='Original pretrained Markov values converted losslessly to FP32 for the unchanged deployed PCTree equation. No weight tuning or new precision method.'),indent=2)+'\n')
    print('FULL_PC_ASSETS_COMPLETE',flush=True)


if __name__=='__main__':main()
