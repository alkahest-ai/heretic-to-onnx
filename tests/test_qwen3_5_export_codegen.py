from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.config import InheritAssets, Manifest, ValidationConfig
from tools.heretic_to_onnx.qwen3_5_export_codegen import (
    DecoderCacheEntry,
    ExportContract,
    SessionSpec,
    build_qwen3_5_export_contract,
    render_qwen3_5_export_runner,
)
from tools.heretic_to_onnx.qwen3_5_quantize_codegen import render_qwen3_5_quantize_runner


def _sample_manifest(*, include_video: bool = False) -> Manifest:
    return Manifest(
        source_model_id="Qwen/Qwen3.5-0.8B",
        base_model_id="Qwen/Qwen3.5-0.8B",
        architecture="qwen3_5_conditional_generation",
        target_repo_id="alkahest-ai/alkahest-0.8b",
        target_dtype="q4f16",
        target_device="webgpu",
        modalities=["text", "image", "video"] if include_video else ["text", "image"],
        inherit_assets=InheritAssets(),
        expected_architecture="Qwen3_5ForConditionalGeneration",
        expected_onnx_files=[
            "onnx/vision_encoder_q4f16.onnx",
            "onnx/embed_tokens_q4f16.onnx",
            "onnx/decoder_model_merged_q4f16.onnx",
        ],
        validation=ValidationConfig(),
        manifest_path=Path("/tmp/heretic-qwen3_5-test.yaml"),
    )


def _sample_q4_webgpu_manifest() -> Manifest:
    return Manifest(
        source_model_id="Qwen/Qwen3.5-0.8B",
        base_model_id="Qwen/Qwen3.5-0.8B",
        architecture="qwen3_5_conditional_generation",
        target_repo_id="thomasjvu/alkahest-0.8b-q4-webgpu",
        target_dtype="q4",
        target_device="webgpu",
        modalities=["text", "image"],
        inherit_assets=InheritAssets(),
        expected_architecture="Qwen3_5ForConditionalGeneration",
        expected_onnx_files=[
            "onnx/vision_encoder_fp16.onnx",
            "onnx/embed_tokens_q4.onnx",
            "onnx/decoder_model_merged_q4.onnx",
        ],
        validation=ValidationConfig(),
        manifest_path=Path("/tmp/heretic-qwen3_5-q4-test.yaml"),
    )


def _write_source_config(
    root: Path,
    *,
    layer_types: list[str] | None,
    full_attention_interval: int = 4,
) -> None:
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5",
        "text_config": {
            "full_attention_interval": full_attention_interval,
            "head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 96,
            "num_hidden_layers": len(layer_types or []),
            "num_key_value_heads": 4,
        },
        "video_token_id": 248057,
    }
    if layer_types is not None:
        config["text_config"]["layer_types"] = layer_types
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _sample_contract() -> ExportContract:
    return ExportContract(
        ok=True,
        model_type="qwen3_5",
        architecture="Qwen3_5ForConditionalGeneration",
        supports_video=True,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_key_value_heads=4,
        head_dim=128,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
        linear_key_head_dim=64,
        linear_value_head_dim=96,
        decoder_cache_entries=[
            DecoderCacheEntry(
                layer_index=0,
                layer_type="linear_attention",
                input_names=[
                    "past_key_values.0.conv_state",
                    "past_key_values.0.recurrent_state",
                ],
                output_names=[
                    "present.0.conv_state",
                    "present.0.recurrent_state",
                ],
            ),
            DecoderCacheEntry(
                layer_index=1,
                layer_type="full_attention",
                input_names=[
                    "past_key_values.1.key",
                    "past_key_values.1.value",
                ],
                output_names=[
                    "present.1.key",
                    "present.1.value",
                ],
            ),
        ],
        image_token_id=151655,
        video_token_id=151656,
        sessions=[
            SessionSpec(
                name="vision_encoder",
                raw_filename="vision_encoder.onnx",
                package_filename="vision_encoder_q4f16.onnx",
                inputs=["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"],
                outputs=["image_features", "video_features"],
                dynamic_axes={},
            ),
            SessionSpec(
                name="embed_tokens",
                raw_filename="embed_tokens.onnx",
                package_filename="embed_tokens_q4f16.onnx",
                inputs=["input_ids"],
                outputs=["inputs_embeds"],
                dynamic_axes={},
            ),
            SessionSpec(
                name="decoder_model_merged",
                raw_filename="decoder_model_merged.onnx",
                package_filename="decoder_model_merged_q4f16.onnx",
                inputs=[
                    "inputs_embeds",
                    "image_features",
                    "video_features",
                    "image_grid_thw",
                    "video_grid_thw",
                    "mm_token_type_ids",
                    "attention_mask",
                    "position_ids",
                    "past_key_values.0.conv_state",
                    "past_key_values.0.recurrent_state",
                    "past_key_values.1.key",
                    "past_key_values.1.value",
                ],
                outputs=[
                    "logits",
                    "present.0.conv_state",
                    "present.0.recurrent_state",
                    "present.1.key",
                    "present.1.value",
                ],
                dynamic_axes={},
            ),
        ],
    )


class Qwen35ExportCodegenTests(unittest.TestCase):
    def test_build_contract_mixed_layers_produce_typed_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(
                source_path,
                layer_types=["linear_attention", "full_attention", "linear_attention"],
            )

            contract = build_qwen3_5_export_contract(_sample_manifest(), source_path)

        self.assertTrue(contract.ok)
        self.assertEqual(
            contract.layer_types,
            ["linear_attention", "full_attention", "linear_attention"],
        )
        self.assertEqual(
            [entry.input_names for entry in contract.decoder_cache_entries],
            [
                ["past_key_values.0.conv_state", "past_key_values.0.recurrent_state"],
                ["past_key_values.1.key", "past_key_values.1.value"],
                ["past_key_values.2.conv_state", "past_key_values.2.recurrent_state"],
            ],
        )
        decoder_session = contract.sessions[-1]
        self.assertIn("past_key_values.0.conv_state", decoder_session.inputs)
        self.assertIn("past_key_values.1.key", decoder_session.inputs)
        self.assertEqual(decoder_session.dynamic_axes["past_key_values.0.conv_state"], {0: "batch"})
        self.assertEqual(decoder_session.dynamic_axes["past_key_values.1.key"], {0: "batch", 2: "past_sequence"})

    def test_build_contract_full_attention_only_uses_kv_cache_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(
                source_path,
                layer_types=["full_attention", "full_attention"],
            )

            contract = build_qwen3_5_export_contract(_sample_manifest(), source_path)

        self.assertTrue(contract.ok)
        self.assertEqual(
            [entry.layer_type for entry in contract.decoder_cache_entries],
            ["full_attention", "full_attention"],
        )
        self.assertEqual(
            contract.decoder_cache_entries[0].input_names,
            ["past_key_values.0.key", "past_key_values.0.value"],
        )

    def test_build_contract_linear_attention_only_uses_linear_state_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(
                source_path,
                layer_types=["linear_attention", "linear_attention"],
            )

            contract = build_qwen3_5_export_contract(_sample_manifest(), source_path)

        self.assertTrue(contract.ok)
        self.assertEqual(
            [entry.layer_type for entry in contract.decoder_cache_entries],
            ["linear_attention", "linear_attention"],
        )
        self.assertEqual(
            contract.decoder_cache_entries[1].output_names,
            ["present.1.conv_state", "present.1.recurrent_state"],
        )
        self.assertEqual(contract.warnings, [])

    def test_quantize_runner_converts_full_graph_to_float16(self) -> None:
        runner = render_qwen3_5_quantize_runner(
            _sample_contract(),
            input_dir="/tmp/input",
            output_dir="/tmp/output",
            report_path="/tmp/report.json",
            block_size=32,
        )

        self.assertIn("keep_io_types=False", runner)
        self.assertIn("disable_shape_infer=True", runner)
        self.assertIn('conversion_mode = "converted_to_fp16"', runner)
        self.assertIn('if "already converted to float16" not in str(exc):', runner)
        self.assertIn('"conversion_mode": conversion_mode', runner)

    def test_build_contract_maps_official_qwen_webgpu_mixed_dtype_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(
                source_path,
                layer_types=["linear_attention", "full_attention"],
            )

            contract = build_qwen3_5_export_contract(_sample_q4_webgpu_manifest(), source_path)

        self.assertTrue(contract.ok)
        self.assertEqual(contract.sessions[0].package_filename, "vision_encoder_fp16.onnx")
        self.assertEqual(contract.sessions[1].package_filename, "embed_tokens_q4.onnx")
        self.assertEqual(contract.sessions[2].package_filename, "decoder_model_merged_q4.onnx")

    def test_quantize_runner_routes_official_qwen_webgpu_dtypes_per_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(
                source_path,
                layer_types=["linear_attention", "full_attention"],
            )
            contract = build_qwen3_5_export_contract(_sample_q4_webgpu_manifest(), source_path)

        runner = render_qwen3_5_quantize_runner(
            contract,
            input_dir="/tmp/input",
            output_dir="/tmp/output",
            report_path="/tmp/report.json",
            block_size=32,
        )

        self.assertIn('if package_dtype == "q4":', runner)
        self.assertIn('GatherBlockQuantized', runner)
        self.assertIn('return _quantize_gather_block_q4(input_path, output_path, block_size)', runner)
        self.assertIn('return _quantize_q4(input_path, output_path, block_size)', runner)
        self.assertIn('if package_dtype == "fp16":', runner)
        self.assertIn('return _quantize_fp16(input_path, output_path)', runner)
        self.assertIn('keep_io_types=True', runner)
        self.assertIn('_quantize_session(raw_path, quantized_path, args.block_size, session["package_filename"])', runner)
        compile(runner, "<qwen3_5_quantize_runner>", "exec")

    def test_build_contract_derives_layer_types_from_full_attention_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(
                source_path,
                layer_types=None,
                full_attention_interval=3,
            )

            config = json.loads((source_path / "config.json").read_text(encoding="utf-8"))
            config["text_config"]["num_hidden_layers"] = 6
            (source_path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

            contract = build_qwen3_5_export_contract(_sample_manifest(), source_path)

        self.assertTrue(contract.ok)
        self.assertEqual(
            contract.layer_types,
            [
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )

    def test_build_contract_adds_video_visual_io_for_v2_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(
                source_path,
                layer_types=["linear_attention", "full_attention"],
            )

            contract = build_qwen3_5_export_contract(_sample_manifest(include_video=True), source_path)

        self.assertTrue(contract.supports_video)
        vision_session = contract.sessions[0]
        self.assertEqual(
            vision_session.inputs,
            ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"],
        )
        self.assertEqual(vision_session.outputs, ["image_features", "video_features"])
        decoder_session = contract.sessions[-1]
        self.assertIn("video_features", decoder_session.inputs)
        self.assertIn("video_grid_thw", decoder_session.inputs)

    def test_runner_handles_visual_pooling_and_placeholder_scatter(self) -> None:
        runner = render_qwen3_5_export_runner(
            _sample_contract(),
            source_path="/tmp/source",
            base_path="/tmp/base",
            output_dir="/tmp/output",
            report_path="/tmp/report.json",
            opset_version=17,
        )

        self.assertIn('pooled = _merge_visual_tensors(getattr(outputs, "pooler_output", None))', runner)
        self.assertIn('pooled = _merge_visual_tensors(outputs.get("pooler_output"))', runner)
        self.assertIn('mm_token_type_ids == 1', runner)
        self.assertIn("masked_scatter(", runner)

    def test_runner_supports_mixed_cache_methods_and_sample_builder(self) -> None:
        runner = render_qwen3_5_export_runner(
            _sample_contract(),
            source_path="/tmp/source",
            base_path="/tmp/base",
            output_dir="/tmp/output",
            report_path="/tmp/report.json",
            opset_version=17,
        )

        self.assertIn("class FlatQwen35CacheLayer(AttrDict):", runner)
        self.assertIn("def update_conv_state(self, conv_state, layer_idx: int):", runner)
        self.assertIn("def update_recurrent_state(self, recurrent_state, layer_idx: int):", runner)
        self.assertIn('FlatQwen35Cache(CONTRACT["decoder_cache_entries"], past_key_values)', runner)
        self.assertIn('for cache_entry in CONTRACT["decoder_cache_entries"]:', runner)
        self.assertIn('conv_state_shape = (1, linear_conv_dim, CONTRACT["linear_conv_kernel_dim"])', runner)
        self.assertIn('recurrent_state_shape = (', runner)
        self.assertIn('model.model.get_video_features(', runner)
        self.assertIn('mm_token_type_ids == 2', runner)
        self.assertIn('pixel_values_videos', runner)
        self.assertIn('repeat_shape = [frames] + [1] * max(pixel_values.ndim - 1, 0)', runner)
        self.assertIn("def _resolve_pad_token_id(config):", runner)
        self.assertIn("pad_token_id = _resolve_pad_token_id(config)", runner)
        self.assertIn("torch.full((1, total_sequence), pad_token_id", runner)
        self.assertIn("repeat_shape = [1] * key.dim()", runner)
        self.assertIn("key = key.repeat(*repeat_shape)", runner)
        self.assertIn("value = value.repeat(*repeat_shape)", runner)
        self.assertNotIn("repeat_interleave(", runner)

    def test_runner_uses_supported_export_kwargs_without_legacy_fallback_chain(self) -> None:
        runner = render_qwen3_5_export_runner(
            _sample_contract(),
            source_path="/tmp/source",
            base_path="/tmp/base",
            output_dir="/tmp/output",
            report_path="/tmp/report.json",
            opset_version=17,
        )

        self.assertIn("def _supports_onnx_export_kwarg(name: str):", runner)
        self.assertIn("def _run_torch_onnx_export(module, export_kwargs):", runner)
        self.assertIn('if _supports_onnx_export_kwarg("external_data"):', runner)
        self.assertNotIn("use_external_data_format=True", runner)


if __name__ == "__main__":
    unittest.main()
