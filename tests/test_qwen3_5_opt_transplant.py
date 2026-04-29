from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.qwen3_5_opt_transplant import (
    _needs_qwen35_rmsnorm_offset,
    _open_source_tensors,
    _resolve_vision_dtype,
    _source_checkpoint_path,
    _state_key_for_matmul,
    _state_key_for_plain_initializer,
    _vision_artifact_stem,
    _write_tokenizer_config,
    _write_browser_config,
)


class Qwen35OptTransplantTests(unittest.TestCase):
    def test_matmul_node_names_map_to_hf_state_keys(self) -> None:
        self.assertEqual(
            _state_key_for_matmul("/model/layers.3/gdn/in_proj_qkv/MatMul_Quant"),
            "model.language_model.layers.3.linear_attn.in_proj_qkv.weight",
        )
        self.assertEqual(
            _state_key_for_matmul("/model/layers.3/attn/q_proj/MatMul_Quant"),
            "model.language_model.layers.3.self_attn.q_proj.weight",
        )
        self.assertEqual(
            _state_key_for_matmul("/model/layers.3/mlp/gate_proj/MatMul_Quant"),
            "model.language_model.layers.3.mlp.gate_proj.weight",
        )
        self.assertEqual(
            _state_key_for_matmul("/lm_head/MatMul_Quant"),
            "model.language_model.embed_tokens.weight",
        )

    def test_plain_initializer_names_map_to_hf_state_keys(self) -> None:
        self.assertEqual(
            _state_key_for_plain_initializer("model.layers.4.attn.q_norm.layernorm.weight"),
            "model.language_model.layers.4.self_attn.q_norm.weight",
        )
        self.assertEqual(
            _state_key_for_plain_initializer("model.layers.4.gdn.conv1d.weight_3d"),
            "model.language_model.layers.4.linear_attn.conv1d.weight",
        )
        self.assertEqual(
            _state_key_for_plain_initializer("model.layers.24.final_norm_layernorm.weight"),
            "model.language_model.norm.weight",
        )
        self.assertEqual(
            _state_key_for_plain_initializer("model.layers.35.final_norm_layernorm.weight"),
            "model.language_model.norm.weight",
        )

    def test_only_plain_qwen_rmsnorm_initializers_need_offset_gamma(self) -> None:
        self.assertTrue(_needs_qwen35_rmsnorm_offset("model.layers.0.input_layernorm.weight"))
        self.assertTrue(_needs_qwen35_rmsnorm_offset("model.layers.0.post_attention_layernorm.weight"))
        self.assertTrue(_needs_qwen35_rmsnorm_offset("model.layers.3.attn.q_norm.layernorm.weight"))
        self.assertTrue(_needs_qwen35_rmsnorm_offset("model.layers.3.attn.k_norm.layernorm.weight"))
        self.assertTrue(_needs_qwen35_rmsnorm_offset("model.layers.24.final_norm_layernorm.weight"))
        self.assertTrue(_needs_qwen35_rmsnorm_offset("model.layers.35.final_norm_layernorm.weight"))
        self.assertFalse(_needs_qwen35_rmsnorm_offset("model.layers.0.gdn.norm.weight"))
        self.assertFalse(_needs_qwen35_rmsnorm_offset("model.layers.0.gdn.dt_bias"))

    def test_browser_config_normalizes_bfloat16_and_uses_official_external_data_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "config.json"
            output = root / "out.json"
            source.write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "torch_dtype": "bfloat16",
                        "text_config": {"dtype": "bfloat16"},
                    }
                ),
                encoding="utf-8",
            )

            _write_browser_config(source, output)

            config = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(config["torch_dtype"], "float16")
        self.assertEqual(config["text_config"]["dtype"], "float16")
        self.assertEqual(
            config["transformers.js_config"]["use_external_data_format"],
            {
                "decoder_model_merged.onnx": 2,
                "vision_encoder": 1,
                "embed_tokens": 1,
                "decoder_model_merged": 1,
            },
        )
        self.assertEqual(config["transformers.js_config"]["kv_cache_dtype"]["q4"], "float16")

    def test_text_only_browser_config_omits_vision_external_data_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "config.json"
            output = root / "out.json"
            source.write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "torch_dtype": "bfloat16",
                    }
                ),
                encoding="utf-8",
            )

            _write_browser_config(source, output, include_vision=False)

            config = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            config["transformers.js_config"]["use_external_data_format"],
            {
                "decoder_model_merged.onnx": 2,
                "embed_tokens": 1,
                "decoder_model_merged": 1,
            },
        )

    def test_tokenizer_config_uses_template_metadata_and_chat_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_tokenizer = root / "tokenizer_config.json"
            output = root / "out_tokenizer_config.json"
            template_tokenizer.write_text(
                json.dumps(
                    {
                        "bos_token_id": 248044,
                        "pad_token_id": 248044,
                        "model_max_length": 32768,
                        "chat_template": "official {{ messages[0].content }}",
                    }
                ),
                encoding="utf-8",
            )

            _write_tokenizer_config(
                template_tokenizer_config=template_tokenizer,
                output_tokenizer_config=output,
            )

            tokenizer_config = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(tokenizer_config["bos_token_id"], 248044)
        self.assertEqual(tokenizer_config["pad_token_id"], 248044)
        self.assertEqual(tokenizer_config["chat_template"], "official {{ messages[0].content }}")

    def test_vision_dtype_auto_preserves_existing_defaults(self) -> None:
        self.assertEqual(_resolve_vision_dtype(decoder_dtype="q4", vision_dtype="auto"), "fp16")
        self.assertEqual(_resolve_vision_dtype(decoder_dtype="q8", vision_dtype="auto"), "q8")
        self.assertEqual(_vision_artifact_stem("q4"), "vision_encoder_q4")

    def test_sharded_safetensors_source_is_supported(self) -> None:
        try:
            import numpy as np
            from safetensors.numpy import save_file
        except ImportError as exc:
            self.skipTest(f"optional safetensors test dependency unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_a = root / "model-00001-of-00002.safetensors"
            shard_b = root / "model-00002-of-00002.safetensors"
            save_file({"model.language_model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32)}, shard_a)
            save_file({"model.language_model.norm.weight": np.full((2,), 2.0, dtype=np.float32)}, shard_b)
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 24},
                        "weight_map": {
                            "model.language_model.embed_tokens.weight": shard_a.name,
                            "model.language_model.norm.weight": shard_b.name,
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(_source_checkpoint_path(root), index)
            with _open_source_tensors(index) as tensors:
                self.assertEqual(tensors.get_tensor("model.language_model.embed_tokens.weight").shape, (2, 2))
                self.assertEqual(float(tensors.get_tensor("model.language_model.norm.weight")[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
