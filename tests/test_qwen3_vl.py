#!/usr/bin/env python3
"""Reference checks for the multimodal target interface; no NPU required."""
import json
import os
from pathlib import Path
import unittest
import tempfile

import numpy as np
import torch
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.qwen3_vl import Qwen3VLTargetModel, interleaved_mrope
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

ROOT = Path(os.environ["VLM_TEST_ARTIFACTS"]).resolve()
METRICS = {}
torch.set_num_threads(4)


def hf_config():
    return Qwen3VLTextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128, vocab_size=64,
        max_position_embeddings=128, rope_parameters={"rope_type": "default", "rope_theta": 5_000_000., "mrope_section": [24, 20, 20]},
        _attn_implementation="eager")


def model_args():
    return ModelArgs(dim=64, hidden_dim=128, n_layers=4, n_heads=2, n_kv_heads=1,
        head_dim=128, vocab_size=64, max_seq_len=24, max_context_len=24,
        use_kv_cache=True, use_hf_rope=True, use_qk_norm=True, qk_norm_before_rope=True,
        norm_eps=1e-6, rope_theta=5_000_000.)


def load_hf_weights(target, hf):
    mapping = {"embed_tokens": "tok_embeddings", "self_attn.q_proj": "attention.wq",
        "self_attn.k_proj": "attention.wk", "self_attn.v_proj": "attention.wv",
        "self_attn.o_proj": "attention.wo", "self_attn.q_norm": "attention.q_norm_fn",
        "self_attn.k_norm": "attention.k_norm_fn", "input_layernorm": "attention_norm",
        "post_attention_layernorm": "ffn_norm", "mlp.gate_proj": "feed_forward.w1",
        "mlp.down_proj": "feed_forward.w2", "mlp.up_proj": "feed_forward.w3"}
    weights = {}
    for key, value in hf.state_dict().items():
        for old, new in mapping.items():
            key = key.replace(old, new)
        weights[key] = value
    weights["output.weight"] = torch.eye(64)
    target.load_state_dict(weights, strict=True)


def forward_chunk(model, embeddings, positions, deepstack, offset, keys, values):
    a = embeddings.shape[1]
    cache_len = keys[0].shape[-1]
    mask = torch.full((1, 1, a, cache_len + a), -255.)
    mask[..., :offset] = 0
    mask[..., cache_len:] = torch.triu(torch.full((a, a), -255.), diagonal=1)
    cos, sin = interleaved_mrope(positions)
    out = model(embeddings, mask, cos, sin, *deepstack, *keys, *values)
    for i, (k, v) in enumerate(zip(out[1], out[2])):
        keys[i][..., offset:offset + a] = k.detach()
        values[i][..., offset:offset + a, :] = v.detach()
    return out[0]


class VLMTests(unittest.TestCase):
    def test_cache_shape_uses_context_capacity(self):
        args = model_args()
        args.max_seq_len = 16
        args.max_context_len = 40
        model = Qwen3VLTargetModel(args, ar_len=4)
        ex = model.get_example_inputs()
        self.assertEqual(next(iter(ex[1])).shape[-1], 40)
        self.assertEqual(ex[-2][0].shape[-1], 36)
        self.assertEqual(ex[-1][0].shape[-2], 36)
        self.assertEqual(model.get_metadata()["get_max_context_len"], 40)

    def test_rotary_reference_and_real_cpp_fixture(self):
        pos = torch.tensor([[7, 11, 19], [17, 17, 17], [0, 0, 0], [2047, 300, 400]])
        rotary = Qwen3VLTextRotaryEmbedding(hf_config())
        expected = rotary(torch.zeros(1, len(pos), 64), pos.T[:, None, :])
        actual = interleaved_mrope(pos)
        for value, ref in zip(actual, expected):
            torch.testing.assert_close(value, ref[0, :, :64], atol=2e-5, rtol=2e-5)
        # The real mtmd fixture exercises the C++ builder, including time frequencies 60..63.
        path = ROOT / "image64.json"
        self.assertTrue(path.exists(), "generate image64 fixture before running the reference suite")
        meta = json.loads(path.read_text())
        positions = torch.from_numpy(np.fromfile(ROOT / "image64.positions.i32", dtype="<i4").reshape(-1, 3))
        expected = rotary(torch.zeros(1, len(positions), 64), positions.T[:, None, :])
        errors = []
        for name, ref in zip(("cos", "sin"), expected):
            cpp = torch.from_numpy(np.fromfile(ROOT / f"image64.vlm_rope_{name}.f32", dtype="<f4").reshape(-1, 64))
            errors.append((cpp - ref[0, :, :64]).abs().max().item())
            torch.testing.assert_close(cpp, ref[0, :, :64], atol=2e-5, rtol=2e-5)
        self.assertLess(meta["rope_delta"], 0)
        METRICS["cpp_mrope_vs_hf_max_abs"] = max(errors)
        METRICS["fixture"] = meta

    @torch.no_grad()
    def test_hf_deepstack_and_chunked_kv(self):
        torch.manual_seed(17)
        hf = Qwen3VLTextModel(hf_config()).eval()
        full, chunked = Qwen3VLTargetModel(model_args(), ar_len=8).eval(), Qwen3VLTargetModel(model_args(), ar_len=4).eval()
        for m in (full, chunked):
            load_hf_weights(m, hf)
        embeds = torch.randn(1, 8, 64) * .25
        positions = torch.tensor([[0,0,0], [1,1,1], [2,2,2], [2,2,3], [2,3,2], [2,3,3], [4,4,4], [5,5,5]])
        visual = torch.tensor([[False,False,True,True,True,True,False,False]])
        ds = [torch.zeros_like(embeds) for _ in range(3)]
        for value in ds:
            value[visual] = torch.randn(4, 64) * .3
        # HF's four-axis interface explicitly separates causal sequence order from MRoPE.
        pos4 = torch.cat((torch.arange(8).view(1, 1, 8), positions.T[:, None, :]))
        ref = hf(inputs_embeds=embeds, position_ids=pos4,
            visual_pos_masks=visual, deepstack_visual_embeds=[d[visual] for d in ds], use_cache=False).last_hidden_state
        ex = full.get_example_inputs()
        actual = forward_chunk(full, embeds, positions, ds, 0, ex[-2], ex[-1])
        torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
        ex = chunked.get_example_inputs()
        pieces = [forward_chunk(chunked, embeds[:, o:o+4], positions[o:o+4],
            [d[:, o:o+4] for d in ds], o, ex[-2], ex[-1]) for o in (0, 4)]
        split = torch.cat(pieces, dim=1)
        torch.testing.assert_close(split, ref, atol=2e-5, rtol=2e-5)
        METRICS["decoder_vs_hf_max_abs"] = (actual-ref).abs().max().item()
        METRICS["chunked_decoder_vs_hf_max_abs"] = (split-ref).abs().max().item()

    def test_export_capture_after_deepstack(self):
        from executorch.extension.llm.custom_ops.model_sharding import SplitGraph
        model = Qwen3VLTargetModel(model_args(), ar_len=4).eval()
        ex = model.get_example_inputs()
        program = torch.export.export(model, (ex[0], *ex[1], *ex[2:7], *ex[7], *ex[8]))
        graph = program.module()
        SplitGraph([2])(graph)
        cuts = [node for node in graph.graph.nodes if "llama.fallback" in str(node.target)]
        self.assertEqual(len(cuts), 1)
        source = cuts[0].args[0]
        self.assertIn("add", str(source.target))
        self.assertTrue(any("vlm_deepstack_1" in str(arg) for arg in source.args))
        METRICS["first_shard_capture"] = "after layer 1 DeepStack addition"

    def test_calibration_rejects_text_only_fixture(self):
        from executorch.examples.qualcomm.oss_scripts.llama.inference.qwen3_vl import Qwen3VLDecoderInference
        with self.assertRaisesRegex(ValueError, "real image"):
            Qwen3VLDecoderInference(Qwen3VLTargetModel(model_args(), ar_len=4).get_example_inputs, [])

    def test_ptq_entry_replays_every_image_chunk(self):
        from executorch.examples.qualcomm.oss_scripts.llama.inference.qwen3_vl import Qwen3VLDecoderInference
        model = Qwen3VLTargetModel(model_args(), ar_len=4)
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / "fixture"
            prefix.with_suffix(".json").write_text(json.dumps({"format_version": 1, "rows": 7,
                "image_rows": 2, "dim": 64}))
            for role in ("vlm_embd", "vlm_rope_cos", "vlm_rope_sin", "vlm_deepstack_0", "vlm_deepstack_1", "vlm_deepstack_2"):
                x = np.zeros((7, 64), dtype="<f4")
                x[5:7] = 3
                x.tofile(str(prefix) + f".{role}.f32")
            inference = Qwen3VLDecoderInference(model.get_example_inputs, [prefix.with_suffix(".json")])
            def probe(*args):
                calls.append((args[1].clone(), args[6].clone()))
                return torch.zeros(1,4,64), [torch.ones(1,1,128,4) for _ in range(4)], [torch.ones(1,1,4,128) for _ in range(4)]
            output = inference.predict_step(probe, torch.zeros(1,4,dtype=torch.long))
            self.assertEqual(len(calls), 2)
            self.assertEqual(output.shape[1], 7)
            self.assertTrue((calls[1][0][..., :4] == 0).all())
            self.assertEqual(calls[1][1][0,1,0].item(), 3)
            self.assertEqual(calls[1][1][0,3,0].item(), 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(VLMTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT / "reference_results.json").write_text(json.dumps({"passed": result.wasSuccessful(),
        "tests_run": result.testsRun, **METRICS}, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
