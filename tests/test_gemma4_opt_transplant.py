from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.gemma4_opt_transplant import (
    _rewrite_embed_outputs_to_float32,
    _state_key_candidates_for_matmul,
    _state_key_candidates_for_plain_initializer,
)


class Gemma4OptimizedTransplantMappingTests(unittest.TestCase):
    def test_maps_reference_matmul_nodes_to_gemma4_state_keys(self) -> None:
        self.assertEqual(
            _state_key_candidates_for_matmul("/model/layers.4/attn/q_proj/MatMul_Quant"),
            ["model.language_model.layers.4.self_attn.q_proj.weight"],
        )
        self.assertEqual(
            _state_key_candidates_for_matmul("/model/layers.7/per_layer/per_layer_projection/MatMul_Quant"),
            ["model.language_model.layers.7.per_layer_projection.weight"],
        )
        self.assertEqual(
            _state_key_candidates_for_matmul("/lm_head/MatMul_Quant"),
            ["lm_head.weight", "model.lm_head.weight", "model.language_model.embed_tokens.weight"],
        )

    def test_maps_reference_norm_initializers_to_gemma4_state_keys(self) -> None:
        self.assertEqual(
            _state_key_candidates_for_plain_initializer("model.layers.2.attn.q_norm.layernorm.weight"),
            ["model.language_model.layers.2.self_attn.q_norm.weight"],
        )
        self.assertEqual(
            _state_key_candidates_for_plain_initializer("model.layers.35.final_norm_layernorm.weight"),
            ["model.language_model.norm.weight"],
        )


@unittest.skipUnless(importlib.util.find_spec("onnx") is not None, "onnx is required")
class Gemma4OptimizedTransplantOnnxTests(unittest.TestCase):
    def test_rewrite_embed_outputs_to_float32_adds_output_casts(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embed_tokens_q4f16.onnx"
            graph = helper.make_graph(
                [helper.make_node("Identity", ["input"], ["inputs_embeds"])],
                "embed",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1])],
                [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
            onnx.save_model(model, path)

            patched = _rewrite_embed_outputs_to_float32(path)
            reloaded = onnx.load(str(path), load_external_data=False)

        self.assertTrue(patched)
        self.assertEqual(reloaded.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(reloaded.graph.node[-1].op_type, "Cast")


if __name__ == "__main__":
    unittest.main()
