from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.gemma4_opt_transplant import (
    _rewrite_embed_outputs_to_float32,
    _state_key_candidates_for_matmul,
    _state_key_candidates_for_plain_initializer,
    _tensor_for_matmul_node,
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
            _state_key_candidates_for_matmul("/model/per_layer_projection/MatMul_Quant"),
            [
                "model.language_model.per_layer_projection.weight",
                "model.language_model.per_layer_model_projection.weight",
            ],
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

    def test_concatenates_gemma4_gate_and_up_proj_for_fused_matmul(self) -> None:
        class FakeSafeFile:
            def get_tensor(self, key: str):
                lookup = {
                    "model.language_model.layers.3.mlp.gate_proj.weight": [[1.0, 2.0]],
                    "model.language_model.layers.3.mlp.up_proj.weight": [[3.0, 4.0]],
                }
                if key not in lookup:
                    raise KeyError(key)
                return lookup[key]

        tensor, source_key = _tensor_for_matmul_node(
            FakeSafeFile(),
            "/model/layers.3/mlp/gate_up_proj/MatMul_Quant",
        )
        self.assertEqual(source_key, "model.language_model.layers.3.mlp.{gate,up}_proj.weight")
        self.assertEqual(tensor.tolist(), [[1.0, 2.0], [3.0, 4.0]])


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

    def test_rewrite_embed_outputs_to_float32_patches_bfloat16_outputs(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embed_tokens_q4f16.onnx"
            graph = helper.make_graph(
                [helper.make_node("Identity", ["input"], ["per_layer_inputs"])],
                "embed",
                [helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1])],
                [helper.make_tensor_value_info("per_layer_inputs", TensorProto.BFLOAT16, [1])],
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
