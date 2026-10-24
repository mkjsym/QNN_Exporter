"""Paths and export-only context packing for the register study's mobile (S26) drafter exports (doc20k arms, cap 3072).

Modes: none (text rows only), norm8 / norm32 (text rows + pooled rows at the region centres, the device CPU summary
vlm-image-summary.h kind 3 with 2x4 / 4x8 bands), dynreg (text rows + K register taps at the drafter slots, the device
dynreg mode). Calibration samples come from MobiSpec.cpp/tools/training/dump_vlm_mobile_calibration.py.
"""
import hashlib, json, os
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
MOBI = ROOT.parent/'MobiSpec.cpp'
STUDY = MOBI/'docs/results/raw/vlm-register-doc20k-20261016'
CALIB = MOBI/'docs/results/raw/vlm-register-mobile-20261019/calibration'
OUT = Path(os.environ.get('REGISTER_MOBILE_EXPORT_OUT') or ROOT/'artifacts/vlm-register-mobile-20261019/export')   # 2026-10-23: override to rebuild on the root disk (/data is full; the 10-22 cleanup removed calibration/ and draft-meta.pth)
TEXT = ROOT.parent/'dflash-models/Qwen3-4B'
ARMS = {'none-20k-seed42': dict(mode='none'), 'norm8-20k-seed42': dict(mode='norm8', regions=(2, 4)), 'norm32-20k-seed42': dict(mode='norm32', regions=(4, 8)),
        'dynreg8-20k-seed42': dict(mode='dynreg', register_count=8), 'deep-dynreg8-20k-seed42': dict(mode='dynreg', register_count=8, deep=True),
        'full-20k-seed42': dict(mode='full')}   # GLANCE-style: every prompt row, image rows included (M3, 2026-10-21; needs the 4,096-row drafter graph)
# Drafter graph context: 1,024 for the M2 arms (cache 1,000 rows), 4,096 for the M3 Full arm and its iso-graph none control (cache 4,072).
# The cache is the context minus the 16 inject rows and the 8 block rows; artifacts of a non-default context carry the suffix -c<L>.
BLOCK_ROWS, INJECT_ROWS = 8, 16
def cache_rows(context_len): return context_len - INJECT_ROWS - BLOCK_ROWS
def context_suffix(context_len): return '' if context_len == 1024 else f'-c{context_len}'
# M4 inject-only draft-prefill graphs (plan 11.2): artifact npu-export-<arm>-inject<N>, context ctx-<arm>-inject<N>; the graph reads no cache,
# so its context is 8 unused cache rows + N inject rows + the 8 block rows (model: vlm_register_mobile_inject_model.py).
INJECT_CACHE_ROWS = 8
def inject_context(n_inject): return INJECT_CACHE_ROWS + n_inject + BLOCK_ROWS
def inject_suffix(n_inject): return f'-inject{n_inject}'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''): h.update(data)
    return h.hexdigest()


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.pending'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def spatial_groups(image_positions, grid_hw, shape):
    """Region id per image token (host rule tools/training/vlm_learned8.spatial_groups; device vlm-image-summary.h bands)."""
    h, w = grid_hw; rows, cols = shape
    assert len(image_positions) == h*w
    groups = {}
    for offset, position in enumerate(image_positions):
        row, col = divmod(offset, w)
        groups[int(position)] = min(row*rows//h, rows-1)*cols + min(col*cols//w, cols-1)
    return groups


def register_slots(image_positions, count):
    n = len(image_positions)
    return [int(image_positions[((2*k+1)*n)//(2*count)]) for k in range(count)]


def pack(sample, mode, assets, arm_spec):
    """Drafter context rows [n, 5*2560] (float32) and their RoPE positions for one calibration sample.

    sample: the dump record (target_hidden_states [rows, 12800], image_positions, grid_hw, prompt_length, registers{arm}).
    """
    features = sample['target_hidden_states'].float(); total = len(features)
    image = [int(p) for p in sample['image_positions'].tolist()]; prompt = int(sample['prompt_length'])
    pos = torch.arange(total)
    visual = torch.zeros(total, dtype=torch.bool); visual[image] = True
    if mode == 'full':
        return features, pos   # every row of the prompt in sequence order (the host OnlineFullDraft and the device's summary=full path)
    if mode == 'none':
        keep = ~visual
        return features[keep], pos[keep]
    if mode == 'dynreg':
        keep = ~visual
        regs = sample['registers'][assets['arm']].float(); count = int(arm_spec['register_count']); assert regs.shape == (count, features.shape[1])
        slots = torch.tensor(register_slots(image, count))
        text_prompt = keep.clone(); text_prompt[prompt:] = False; after = keep.clone(); after[:prompt] = False
        # text rows of the prompt, then the register rows, then the response rows (device: registers committed after the prompt tail)
        return torch.cat((features[text_prompt], regs, features[after]), 0), torch.cat((pos[text_prompt], slots, pos[after]), 0)
    if mode in ('norm8', 'norm32'):
        groups = spatial_groups(image, tuple(sample['grid_hw']), tuple(arm_spec['regions'])); G = arm_spec['regions'][0]*arm_spec['regions'][1]
        summary = assets['summary']; deep = features[:, -2560:]
        question = deep[prompt-1]
        queries = summary['summary_queries'].float() + F.linear(question, summary['summary_condition.weight'].float())[None]
        keep = ~visual; output = features.clone()
        for region in range(G):
            index = torch.tensor([p for p in image if groups[p] == region], dtype=torch.long)
            if not len(index): continue
            key = F.normalize(F.linear(deep[index], summary['summary_key.weight'].float()), dim=-1, eps=.01)
            query = F.normalize(queries[region], dim=-1, eps=.01)
            weights = (key @ query/.5).softmax(0)
            location = index[len(index)//2]; keep[location] = True
            output[location] = (features[index]*weights[:, None]).sum(0)
        return output[keep], pos[keep]
    raise ValueError(mode)


ROPE_POSITIONS = 4096   # document pages put drafter rows at sequence positions up to ~3,600; the cache (1024) is unchanged


def extend_rope_tables(model, config, end=ROPE_POSITIONS):
    """Replace the graph's RoPE tables (sized by max_context_len) with tables for `end` positions, same formula as static_llama."""
    from executorch.examples.models.llama.rope import hf_precompute_freqs_cis, precompute_freqs_cis
    if config.use_hf_rope:
        cos, sin = hf_precompute_freqs_cis(config.head_dim, end, config.rope_freq_base, config.partial_rotary_factor)
        cos = cos[:, : cos.shape[-1] // 2]; sin = sin[:, : sin.shape[-1] // 2]
    else:
        cos, sin = precompute_freqs_cis(config.head_dim, end, config.rope_freq_base, config.use_scaled_rope, config.rope_scale_factor)
    old = model.freqs_cos
    # float32 argument rounding alone moves cos/sin by up to ~2.4e-4 at 4,096 rad (ulp of the angle), so a 1e-6 tolerance fails
    # intermittently for 4,096-row defaults (seen 2026-10-22 at row 2731, max|diff| 1.5e-4); 1e-3 still catches a different formula or base.
    if not (cos.shape[1:] == old.shape[1:] and torch.allclose(cos[:old.shape[0]], old, atol=1e-3, rtol=0)):
        d = (cos[:old.shape[0]].float() - old.float()).abs() if cos.shape[1:] == old.shape[1:] else None
        raise AssertionError(f'RoPE table prefix must match the graph default: model {type(model).__name__}, default table {tuple(old.shape)} {old.dtype}, '
                             f'new table {tuple(cos.shape)} {cos.dtype}, max_context_len {getattr(config, "max_context_len", None)}, '
                             f'max|diff| {d.max().item() if d is not None else "shape mismatch"}, first row over 1e-6 {int((d.max(1).values > 1e-6).nonzero()[0]) if d is not None and (d.max(1).values > 1e-6).any() else None}')
    model.freqs_cos = cos.to(old.dtype); model.freqs_sin = sin.to(model.freqs_sin.dtype)
    return model
