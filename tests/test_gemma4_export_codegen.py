from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.config import InheritAssets, Manifest, ValidationConfig
from tools.heretic_to_onnx.gemma4_export_codegen import (
    ExportContract,
    SessionSpec,
    build_gemma4_export_contract,
    render_gemma4_export_runner,
)
from tools.heretic_to_onnx.gemma4_quantize_codegen import render_gemma4_quantize_runner


def _sample_manifest(*, include_audio: bool = False, include_video: bool = False) -> Manifest:
    modalities = ["text", "image"]
    if include_audio:
        modalities.append("audio")
    if include_video:
        modalities.append("video")
    expected_onnx_files = [
        "onnx/vision_encoder_q4f16.onnx",
        "onnx/embed_tokens_q4f16.onnx",
        "onnx/decoder_model_merged_q4f16.onnx",
    ]
    if include_audio:
        expected_onnx_files.insert(1, "onnx/audio_encoder_q4f16.onnx")
    return Manifest(
        source_model_id="google/gemma-4-E2B-it",
        base_model_id="google/gemma-4-E2B-it",
        architecture="gemma4_conditional_generation",
        target_repo_id="alkahest-ai/rally-2b-v2" if include_audio or include_video else "alkahest-ai/rally-2b",
        target_dtype="q4f16",
        target_device="webgpu",
        modalities=modalities,
        inherit_assets=InheritAssets(),
        expected_architecture="Gemma4ForConditionalGeneration",
        expected_onnx_files=expected_onnx_files,
        validation=ValidationConfig(),
        manifest_path=Path("/tmp/heretic-gemma4-test.yaml"),
    )


def _sample_text_manifest() -> Manifest:
    return Manifest(
        source_model_id="google/gemma-4-E2B-it",
        base_model_id="google/gemma-4-E2B-it",
        architecture="gemma4_conditional_generation",
        target_repo_id="alkahest-ai/rally-2b-text",
        target_dtype="q4f16",
        target_device="webgpu",
        modalities=["text"],
        inherit_assets=InheritAssets(),
        expected_architecture="Gemma4ForConditionalGeneration",
        expected_onnx_files=[
            "onnx/embed_tokens_q4f16.onnx",
            "onnx/decoder_model_merged_q4f16.onnx",
        ],
        validation=ValidationConfig(),
        manifest_path=Path("/tmp/heretic-gemma4-text-test.yaml"),
    )


def _write_source_config(root: Path) -> None:
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "hidden_size_per_layer_input": 512,
            "num_hidden_layers": 6,
            "num_kv_shared_layers": 2,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "full_attention"],
            "use_bidirectional_attention": False,
        },
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _sample_contract() -> ExportContract:
    return ExportContract(
        ok=True,
        model_type="gemma4",
        architecture="Gemma4ForConditionalGeneration",
        use_bidirectional_attention="False",
        supports_audio=True,
        supports_video=True,
        num_hidden_layers=6,
        num_kv_shared_layers=2,
        num_cache_layers=4,
        cache_layer_types=["sliding_attention", "sliding_attention", "full_attention", "full_attention"],
        hidden_size_per_layer_input=512,
        sessions=[
            SessionSpec(
                name="vision_encoder",
                raw_filename="vision_encoder.onnx",
                package_filename="vision_encoder_q4f16.onnx",
                inputs=["pixel_values", "pixel_position_ids", "pixel_values_videos", "video_position_ids"],
                outputs=["image_features", "video_features"],
                dynamic_axes={},
            ),
            SessionSpec(
                name="audio_encoder",
                raw_filename="audio_encoder.onnx",
                package_filename="audio_encoder_q4f16.onnx",
                inputs=["input_features", "input_features_mask"],
                outputs=["audio_features"],
                dynamic_axes={},
            ),
            SessionSpec(
                name="embed_tokens",
                raw_filename="embed_tokens.onnx",
                package_filename="embed_tokens_q4f16.onnx",
                inputs=["input_ids"],
                outputs=["inputs_embeds", "per_layer_inputs"],
                dynamic_axes={},
            ),
            SessionSpec(
                name="decoder_model_merged",
                raw_filename="decoder_model_merged.onnx",
                package_filename="decoder_model_merged_q4f16.onnx",
                inputs=["inputs_embeds", "per_layer_inputs", "attention_mask", "position_ids"],
                outputs=["logits"],
                dynamic_axes={},
            ),
        ],
    )


class Gemma4ExportCodegenTests(unittest.TestCase):
    def test_build_contract_v1_stays_text_image_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(source_path)
            contract = build_gemma4_export_contract(_sample_manifest(), source_path)

        self.assertFalse(contract.supports_audio)
        self.assertFalse(contract.supports_video)
        self.assertEqual(contract.sessions[0].inputs, ["pixel_values", "pixel_position_ids"])
        self.assertTrue(all(session.name != "audio_encoder" for session in contract.sessions))

    def test_build_contract_v2_adds_audio_and_video_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(source_path)
            contract = build_gemma4_export_contract(
                _sample_manifest(include_audio=True, include_video=True),
                source_path,
            )

        self.assertTrue(contract.supports_audio)
        self.assertTrue(contract.supports_video)
        self.assertEqual(
            contract.sessions[0].inputs,
            ["pixel_values", "pixel_position_ids", "pixel_values_videos", "video_position_ids"],
        )
        self.assertEqual(contract.sessions[0].outputs, ["image_features", "video_features"])
        self.assertEqual(contract.sessions[1].name, "audio_encoder")

    def test_build_contract_text_only_omits_vision_and_audio_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(source_path)
            contract = build_gemma4_export_contract(_sample_text_manifest(), source_path)

        self.assertEqual([session.name for session in contract.sessions], ["embed_tokens", "decoder_model_merged"])

    def test_runner_loads_image_processor_only_when_vision_session_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            _write_source_config(source_path)
            contract = build_gemma4_export_contract(_sample_text_manifest(), source_path)
            runner = render_gemma4_export_runner(
                contract,
                source_path="/tmp/source",
                base_path="/tmp/base",
                output_dir="/tmp/output",
                report_path="/tmp/report.json",
                opset_version=17,
            )

        self.assertIn('has_vision_session = any(session["name"] == "vision_encoder"', runner)
        self.assertIn("if has_vision_session else None", runner)

    def test_runner_contains_v2_multimodal_helpers(self) -> None:
        runner = render_gemma4_export_runner(
            _sample_contract(),
            source_path="/tmp/source",
            base_path="/tmp/base",
            output_dir="/tmp/output",
            report_path="/tmp/report.json",
            opset_version=17,
        )

        self.assertIn("def _prepare_audio_core_inputs(input_features, input_features_mask):", runner)
        self.assertIn("model.model.get_video_features(", runner)
        self.assertIn("Gemma4VideoProcessor", runner)
        self.assertIn("def _load_video_processor(source_path: Path, base_path: Path):", runner)
        self.assertIn('candidate = root / "video_preprocessor_config.json"', runner)
        self.assertIn('processor_candidate = root / "processor_config.json"', runner)
        self.assertIn('video_processor = processor_config.get("video_processor")', runner)
        self.assertIn("def _build_video_sample_inputs(video_processor, image_height: int, image_width: int, device):", runner)
        self.assertIn('processed.get("pixel_values_videos")', runner)
        self.assertIn("def _patch_audio_attention_for_onnx_export(model):", runner)
        self.assertIn("def _patch_gemma4_attention_for_onnx_export():", runner)
        self.assertIn('arguments["attention_mask"] = attention_mask.to(device=query.device)', runner)
        self.assertIn("def _extract_block_context_export_safe(self, hidden_states: torch.Tensor) -> torch.Tensor:", runner)
        self.assertIn("self_attn._extract_block_context = types.MethodType(_extract_block_context_export_safe, self_attn)", runner)
        self.assertIn('if CONTRACT.get("supports_video"):', runner)
        self.assertIn('if CONTRACT.get("supports_audio"):', runner)
        self.assertIn("_patch_gemma4_attention_for_onnx_export()", runner)
        self.assertIn('_patch_audio_attention_for_onnx_export(model)', runner)
        self.assertIn('wrappers["audio_encoder"] = Gemma4AudioEncoderWrapper(model)', runner)

    def test_quantize_runner_converts_full_graph_to_float16(self) -> None:
        runner = render_gemma4_quantize_runner(
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
        self.assertIn("def _harmonize_float16_elementwise_inputs(model) -> int:", runner)
        self.assertIn('if node.op_type not in {"Add", "Sub", "Mul", "Div"}:', runner)
        self.assertIn('"fixed_elementwise_inputs": fixed_elementwise_inputs', runner)
        self.assertIn('"conversion_mode": conversion_mode', runner)


if __name__ == "__main__":
    unittest.main()
