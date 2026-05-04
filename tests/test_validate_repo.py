from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.config import InheritAssets, Manifest, ValidationConfig
from tools.heretic_to_onnx.validate_repo import validate_package


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _manifest(root: Path, *, expected_onnx_files: list[str]) -> Manifest:
    return Manifest(
        source_model_id="example/source",
        base_model_id="example/base",
        architecture="gemma4_conditional_generation",
        target_repo_id="alkahest-ai/runtime-smoke",
        target_dtype="q4f16",
        target_device="webgpu",
        modalities=["text"],
        inherit_assets=InheritAssets(),
        expected_architecture="Gemma4ForConditionalGeneration",
        expected_onnx_files=expected_onnx_files,
        validation=ValidationConfig(),
        manifest_path=root / "manifest.yaml",
    )


def _qwen_q4_manifest(root: Path, *, expected_onnx_files: list[str]) -> Manifest:
    return Manifest(
        source_model_id="example/source",
        base_model_id="example/base",
        architecture="qwen3_5_conditional_generation",
        target_repo_id="thomasjvu/alkahest-0.8b-heretic-onnx-opt",
        target_dtype="q4",
        target_device="webgpu",
        modalities=["text", "image"],
        inherit_assets=InheritAssets(),
        expected_architecture="Qwen3_5ForConditionalGeneration",
        expected_onnx_files=expected_onnx_files,
        validation=ValidationConfig(),
        manifest_path=root / "manifest.yaml",
    )


@unittest.skipUnless(
    importlib.util.find_spec("onnx") is not None,
    "onnx is required for Gemma4 WebGPU graph contract tests",
)
class ValidateRepoGemma4WebgpuContractTests(unittest.TestCase):
    def _write_gemma_package_config(self, package_dir: Path) -> None:
        _write_json(
            package_dir / "config.json",
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "dtype": "float16",
                "transformers.js_config": {
                    "use_external_data_format": True,
                    "kv_cache_dtype": {"q4f16": "float16"},
                },
            },
        )

    def _write_custom_op_model(self, path: Path, *, opset: int, op_type: str) -> None:
        import onnx
        from onnx import TensorProto, helper

        path.parent.mkdir(parents=True, exist_ok=True)
        graph = helper.make_graph(
            [helper.make_node(op_type, ["x"], ["y"], domain="com.microsoft")],
            "gemma4_contract_graph",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", opset),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )
        onnx.save_model(model, path)

    def test_gemma4_q4f16_contract_accepts_reference_style_text_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            self._write_gemma_package_config(package_dir)
            self._write_custom_op_model(
                package_dir / "onnx" / "decoder_model_merged_q4f16.onnx",
                opset=21,
                op_type="MatMulNBits",
            )
            self._write_custom_op_model(
                package_dir / "onnx" / "embed_tokens_q4f16.onnx",
                opset=21,
                op_type="GatherBlockQuantized",
            )

            report = validate_package(
                _manifest(
                    root,
                    expected_onnx_files=[
                        "onnx/embed_tokens_q4f16.onnx",
                        "onnx/decoder_model_merged_q4f16.onnx",
                    ],
                ),
                package_dir,
                runtime_smoke=False,
            )

        self.assertTrue(report.ok, report.errors)

    def test_gemma4_q4f16_contract_rejects_legacy_opset_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            self._write_gemma_package_config(package_dir)
            self._write_custom_op_model(
                package_dir / "onnx" / "decoder_model_merged_q4f16.onnx",
                opset=17,
                op_type="MatMulNBits",
            )

            report = validate_package(
                _manifest(
                    root,
                    expected_onnx_files=[
                        "onnx/embed_tokens_q4f16.onnx",
                        "onnx/decoder_model_merged_q4f16.onnx",
                    ],
                ),
                package_dir,
                runtime_smoke=False,
            )

        self.assertFalse(report.ok)
        self.assertTrue(any("must use ONNX opset >= 21" in error for error in report.errors))


class ValidateRepoQwenWebgpuContractTests(unittest.TestCase):
    def _write_qwen_package_config(self, package_dir: Path) -> None:
        _write_json(
            package_dir / "config.json",
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "dtype": "float16",
                "transformers.js_config": {
                    "use_external_data_format": {
                        "vision_encoder": 1,
                        "embed_tokens": 1,
                        "decoder_model_merged": 1,
                    },
                    "kv_cache_dtype": {"q4": "float16"},
                },
            },
        )

    def test_qwen_q4_webgpu_contract_rejects_q4f16_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            self._write_qwen_package_config(package_dir)

            report = validate_package(
                _qwen_q4_manifest(
                    root,
                    expected_onnx_files=[
                        "onnx/vision_encoder_q4f16.onnx",
                        "onnx/embed_tokens_q4f16.onnx",
                        "onnx/decoder_model_merged_q4f16.onnx",
                    ],
                ),
                package_dir,
                runtime_smoke=False,
            )

        self.assertFalse(report.ok)
        self.assertTrue(any("must not use legacy q4f16" in error for error in report.errors))
        self.assertTrue(any("must declare official-style text sessions" in error for error in report.errors))

    def test_qwen_q4_webgpu_contract_rejects_inflated_08b_embed_external_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            self._write_qwen_package_config(package_dir)
            data_path = package_dir / "onnx" / "embed_tokens_q4.onnx_data"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            with data_path.open("wb") as file:
                file.truncate(351 * 1024 * 1024)

            report = validate_package(
                _qwen_q4_manifest(
                    root,
                    expected_onnx_files=[
                        "onnx/vision_encoder_fp16.onnx",
                        "onnx/embed_tokens_q4.onnx",
                        "onnx/decoder_model_merged_q4.onnx",
                    ],
                ),
                package_dir,
                runtime_smoke=False,
            )

        self.assertFalse(report.ok)
        self.assertTrue(any("too large for the 0.8B q4 WebGPU contract" in error for error in report.errors))

    def test_qwen_q4_webgpu_contract_allows_q4_vision_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            self._write_qwen_package_config(package_dir)

            report = validate_package(
                _qwen_q4_manifest(
                    root,
                    expected_onnx_files=[
                        "onnx/vision_encoder_q4.onnx",
                        "onnx/embed_tokens_q4.onnx",
                        "onnx/decoder_model_merged_q4.onnx",
                    ],
                ),
                package_dir,
                runtime_smoke=False,
            )

        self.assertTrue(report.ok)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None,
        "onnx is required for Qwen WebGPU graph contract tests",
    )
    def test_qwen_q4_webgpu_contract_rejects_unoptimized_decoder_graph(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            self._write_qwen_package_config(package_dir)
            decoder_path = package_dir / "onnx" / "decoder_model_merged_q4.onnx"
            decoder_path.parent.mkdir(parents=True, exist_ok=True)
            input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
            output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
            graph = helper.make_graph(
                [helper.make_node("Identity", ["input"], ["output"])],
                "unoptimized_qwen_decoder",
                [input_info],
                [output_info],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            onnx.save_model(model, decoder_path)

            report = validate_package(
                _qwen_q4_manifest(
                    root,
                    expected_onnx_files=[
                        "onnx/vision_encoder_fp16.onnx",
                        "onnx/embed_tokens_q4.onnx",
                        "onnx/decoder_model_merged_q4.onnx",
                    ],
                ),
                package_dir,
                runtime_smoke=False,
            )

        self.assertFalse(report.ok)
        self.assertTrue(any("missing required optimized custom ops" in error for error in report.errors))

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None,
        "onnx + onnxruntime are required for Qwen optimized runtime smoke tests",
    )
    def test_runtime_smoke_skips_official_qwen_optimized_decoder(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            self._write_qwen_package_config(package_dir)
            decoder_path = package_dir / "onnx" / "decoder_model_merged_q4.onnx"
            decoder_path.parent.mkdir(parents=True, exist_ok=True)
            graph = helper.make_graph(
                [
                    helper.make_node(
                        "MatMulNBits",
                        ["x"],
                        ["matmul_out"],
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "SkipSimplifiedLayerNormalization",
                        ["matmul_out"],
                        ["y"],
                        domain="com.microsoft",
                    )
                ],
                "optimized_qwen_decoder",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1])],
            )
            model = helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", 21),
                    helper.make_opsetid("com.microsoft", 1),
                ],
            )
            onnx.save_model(model, decoder_path)

            report = validate_package(
                _qwen_q4_manifest(
                    root,
                    expected_onnx_files=[
                        "onnx/vision_encoder_fp16.onnx",
                        "onnx/embed_tokens_q4.onnx",
                        "onnx/decoder_model_merged_q4.onnx",
                    ],
                ),
                package_dir,
                runtime_smoke=True,
            )

        self.assertTrue(report.ok)
        self.assertEqual(len(report.runtime_smoke), 1)
        self.assertTrue(report.runtime_smoke[0].ok)
        self.assertTrue(report.runtime_smoke[0].skipped)
        self.assertTrue(any("runtime smoke skipped for optimized Qwen decoder" in warning for warning in report.warnings))


@unittest.skipUnless(
    importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None,
    "onnx + onnxruntime are required for packaged runtime smoke tests",
)
class ValidateRepoRuntimeSmokeTests(unittest.TestCase):
    def _write_package_config(self, package_dir: Path) -> None:
        _write_json(
            package_dir / "config.json",
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "dtype": "float16",
                "transformers.js_config": {
                    "use_external_data_format": True,
                    "kv_cache_dtype": {"q4f16": "float16"},
                },
            },
        )

    def _write_add_model(self, path: Path, *, input_type: int, weight_type: int, output_type: int) -> None:
        import onnx
        from onnx import TensorProto, helper

        def make_initializer(name: str, data_type: int) -> onnx.TensorProto:
            if data_type == TensorProto.FLOAT:
                return helper.make_tensor(name, data_type, [1], [1.0])
            if data_type == TensorProto.FLOAT16:
                return helper.make_tensor(name, data_type, [1], struct.pack("<e", 1.0), raw=True)
            raise AssertionError(f"unsupported data type for test initializer: {data_type}")

        path.parent.mkdir(parents=True, exist_ok=True)
        input_info = helper.make_tensor_value_info("input", input_type, [1])
        output_info = helper.make_tensor_value_info("output", output_type, [1])
        node = helper.make_node("Add", ["input", "weight"], ["output"])
        graph = helper.make_graph(
            [node],
            "runtime_smoke_graph",
            [input_info],
            [output_info],
            [make_initializer("weight", weight_type)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.save_model(model, path)

    def test_runtime_smoke_passes_for_a_valid_packaged_session(self) -> None:
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            onnx_path = package_dir / "onnx" / "decoder_model_merged_q4f16.onnx"
            self._write_package_config(package_dir)
            self._write_add_model(
                onnx_path,
                input_type=onnx.TensorProto.FLOAT,
                weight_type=onnx.TensorProto.FLOAT,
                output_type=onnx.TensorProto.FLOAT,
            )

            report = validate_package(
                _manifest(root, expected_onnx_files=["onnx/decoder_model_merged_q4f16.onnx"]),
                package_dir,
                strict_onnx=True,
                runtime_smoke=True,
            )

        self.assertTrue(report.ok, report.errors)
        self.assertTrue(report.runtime_smoke_enabled)
        self.assertEqual(len(report.runtime_smoke), 1)
        self.assertTrue(report.runtime_smoke[0].ok)
        self.assertEqual(report.runtime_smoke[0].onnx_path, "onnx/decoder_model_merged_q4f16.onnx")

    def test_runtime_smoke_fails_for_mixed_float_and_float16_add(self) -> None:
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            onnx_path = package_dir / "onnx" / "decoder_model_merged_q4f16.onnx"
            self._write_package_config(package_dir)
            self._write_add_model(
                onnx_path,
                input_type=onnx.TensorProto.FLOAT16,
                weight_type=onnx.TensorProto.FLOAT,
                output_type=onnx.TensorProto.FLOAT16,
            )

            report = validate_package(
                _manifest(root, expected_onnx_files=["onnx/decoder_model_merged_q4f16.onnx"]),
                package_dir,
                strict_onnx=True,
                runtime_smoke=True,
            )

        self.assertFalse(report.ok)
        self.assertEqual(len(report.runtime_smoke), 1)
        self.assertFalse(report.runtime_smoke[0].ok)
        self.assertIn("decoder_model_merged_q4f16.onnx", report.runtime_smoke[0].onnx_path)
        self.assertIn("float", report.runtime_smoke[0].error.lower())

    def test_runtime_smoke_skips_optimized_gemma4_decoder(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            decoder_path = package_dir / "onnx" / "decoder_model_merged_q4f16.onnx"
            self._write_package_config(package_dir)
            decoder_path.parent.mkdir(parents=True, exist_ok=True)
            graph = helper.make_graph(
                [helper.make_node("MatMulNBits", ["x"], ["y"], domain="com.microsoft")],
                "optimized_gemma4_decoder",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1])],
            )
            model = helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", 21),
                    helper.make_opsetid("com.microsoft", 1),
                ],
            )
            onnx.save_model(model, decoder_path)

            report = validate_package(
                _manifest(root, expected_onnx_files=["onnx/decoder_model_merged_q4f16.onnx"]),
                package_dir,
                strict_onnx=True,
                runtime_smoke=True,
            )

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(len(report.runtime_smoke), 1)
        self.assertTrue(report.runtime_smoke[0].ok)
        self.assertTrue(report.runtime_smoke[0].skipped)
        self.assertTrue(any("runtime smoke skipped for optimized Gemma4 decoder" in warning for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()
