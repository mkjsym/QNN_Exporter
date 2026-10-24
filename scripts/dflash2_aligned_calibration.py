"""Replay causally aligned train features, anchor IDs and cache positions."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils._pytree import tree_leaves
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import DFlashDecoderInference
from prepare_dflash2 import OUT, sha


class AlignedDFlash2Calibration(DFlashDecoderInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base = Path(__file__).resolve().parents[1] / 'artifacts/long-draft-20260917/assets-b16-v1/manifest.json'
        manifest = json.loads(base.read_text())
        self._aligned_sources = sorted(manifest['calibration_inputs'].items())
        assert len(self._aligned_sources) == 48
        self._aligned_cursor = 0
        self._aligned_embedding = np.memmap(OUT/'npu-assets/embedding-f16.bin', dtype=np.float16,
                                           mode='r', shape=(151936, 2560))
        for path, digest in self._aligned_sources:
            assert sha(path) == digest

    def predict_step(self, module, input_ids, attn_mask=None, hidden_states=(), tok_embedding=None):
        if not hasattr(self, '_aligned_forward'):
            self._aligned_forward = self.make_forward_fn()
        return self._aligned_forward(module, dict(input_ids=input_ids, attention_mask=attn_mask))

    def make_forward_fn(self):
        @torch.inference_mode()
        def call(model, inputs):
            index = self._aligned_cursor
            assert index < len(self._aligned_sources)
            self._aligned_cursor += 1
            path, digest = self._aligned_sources[index]
            item = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            assert item['case']['split'] == 'train'
            assert item['layer_ids'] == [1, 9, 17, 25, 33]
            features, ids = item['target_features'], item['input_ids']
            prefix = min([1, 16, 37, 64, 128, 256][index % 6], len(features), len(ids)-1)
            assert features.shape[1:] == (5, 2560) and prefix > 0
            keys = [torch.zeros_like(k) for k in self._k_caches]
            values = [torch.zeros_like(v) for v in self._v_caches]
            assert len(keys) == 5 and keys[0].shape == (1, 8, 128, 999)
            for begin in range(0, prefix, 16):
                end = min(prefix, begin+16)
                take = end-begin
                hidden = []
                for layer in range(5):
                    value = torch.zeros(1, 16, 2560)
                    value[:, :take] = features[begin:end, layer].float()
                    hidden.append(value)
                block_ids = np.full(9, 151669, dtype=np.int32)
                block_ids[0] = int(ids[end])
                block = torch.from_numpy(self._aligned_embedding[block_ids].astype('float32')).unsqueeze(0)
                mask = torch.full((1, 1, 9, 1024), -255.)
                mask[..., :begin] = 0
                mask[..., 999:999+take] = 0
                mask[..., -9:] = 0
                pos = torch.arange(end, end+9, dtype=torch.int32).unsqueeze(0)
                pc = torch.arange(begin, begin+16, dtype=torch.int32).unsqueeze(0)
                result = tree_leaves(model(*hidden, block, mask, pos, pc, *keys, *values))
                assert len(result) == 11 and result[0].shape == (1, 9, 152192)
                for layer in range(5):
                    keys[layer][..., begin:end] = result[1+layer][..., :take]
                    values[layer][:, :, begin:end, :] = result[6+layer][:, :, :take, :]
            print('DFLASH2_ALIGNED_CALIBRATION', index, item['case']['id'], prefix,
                  'anchor', int(ids[prefix]), 'source', digest, flush=True)
            return result[0][..., :151936]
        return call
