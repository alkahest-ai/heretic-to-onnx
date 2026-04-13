from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any

from .config import Manifest


@dataclass(slots=True)
class SessionSpec:
    name: str
    raw_filename: str
    package_filename: str
    inputs: list[str]
    outputs: list[str]
    dynamic_axes: dict[str, dict[int, str]]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExportContract:
    ok: bool
    model_type: str
    architecture: str
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    image_token_id: int | None
    video_token_id: int | None
    sessions: list[SessionSpec]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "sessions": [session.to_dict() for session in self.sessions],
            "warnings": list(self.warnings),
        }


def _session_filename_map(manifest: Manifest) -> dict[str, str]:
    mapping: dict[str, str] = {}
    suffix = f"_{manifest.target_dtype}.onnx"
    for relative_path in manifest.expected_onnx_files:
        path = Path(relative_path)
        if path.suffix != ".onnx":
            continue
        if path.stem.endswith(f"_{manifest.target_dtype}"):
            session_name = path.stem[: -len(f"_{manifest.target_dtype}")]
        else:
            session_name = path.stem
        mapping[session_name] = path.name
    return mapping


def _load_source_config(source_path: str | Path) -> dict[str, Any]:
    config_path = Path(source_path).expanduser().resolve() / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_qwen3_5_export_contract(manifest: Manifest, source_path: str | Path) -> ExportContract:
    source_config = _load_source_config(source_path)
    text_config = source_config.get("text_config", {})
    package_filenames = _session_filename_map(manifest)

    architecture = ""
    architectures = source_config.get("architectures")
    if isinstance(architectures, list) and architectures:
        architecture = str(architectures[0])

    num_hidden_layers = int(text_config.get("num_hidden_layers", 0))
    num_key_value_heads = int(text_config.get("num_key_value_heads", 0))
    head_dim = int(text_config.get("head_dim", 0))
    image_token_id = source_config.get("image_token_id")
    video_token_id = source_config.get("video_token_id")

    warnings: list[str] = []
    ok = True

    if source_config.get("model_type") != "qwen3_5":
        ok = False
        warnings.append(f"expected model_type=qwen3_5, got {source_config.get('model_type')!r}")
    if architecture != manifest.expected_architecture:
        ok = False
        warnings.append(f"expected architecture {manifest.expected_architecture}, got {architecture!r}")
    if num_hidden_layers <= 0:
        ok = False
        warnings.append("Qwen3.5 export currently expects num_hidden_layers > 0")
    if num_key_value_heads <= 0:
        ok = False
        warnings.append("Qwen3.5 export currently expects num_key_value_heads > 0")
    if head_dim <= 0:
        ok = False
        warnings.append("Qwen3.5 export currently expects head_dim > 0")
    if image_token_id is None:
        warnings.append("config.json does not define image_token_id; sample multimodal trace may be incomplete")

    sessions = [
        SessionSpec(
            name="vision_encoder",
            raw_filename="vision_encoder.onnx",
            package_filename=package_filenames.get("vision_encoder", "vision_encoder_q4f16.onnx"),
            inputs=["pixel_values", "image_grid_thw"],
            outputs=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "image_batch", 2: "image_height", 3: "image_width"},
                "image_grid_thw": {0: "image_batch"},
                "image_features": {0: "image_tokens"},
            },
            notes=["This session extracts LM-ready visual features for Qwen3.5 browser inference."],
        ),
        SessionSpec(
            name="embed_tokens",
            raw_filename="embed_tokens.onnx",
            package_filename=package_filenames.get("embed_tokens", "embed_tokens_q4f16.onnx"),
            inputs=["input_ids"],
            outputs=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "inputs_embeds": {0: "batch", 1: "sequence"},
            },
            notes=["Image/video placeholder token IDs are masked to PAD before token embedding."],
        ),
    ]

    decoder_inputs = [
        "inputs_embeds",
        "image_features",
        "image_grid_thw",
        "mm_token_type_ids",
        "attention_mask",
        "position_ids",
    ]
    decoder_outputs = ["logits"]
    decoder_dynamic_axes: dict[str, dict[int, str]] = {
        "inputs_embeds": {0: "batch", 1: "sequence"},
        "image_features": {0: "image_tokens"},
        "image_grid_thw": {0: "image_batch"},
        "mm_token_type_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "attention_sequence"},
        "position_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }

    for layer_index in range(num_hidden_layers):
        key_name = f"past_key_values.{layer_index}.key"
        value_name = f"past_key_values.{layer_index}.value"
        present_key_name = f"present.{layer_index}.key"
        present_value_name = f"present.{layer_index}.value"
        decoder_inputs.extend([key_name, value_name])
        decoder_outputs.extend([present_key_name, present_value_name])
        decoder_dynamic_axes[key_name] = {0: "batch", 2: "past_sequence"}
        decoder_dynamic_axes[value_name] = {0: "batch", 2: "past_sequence"}
        decoder_dynamic_axes[present_key_name] = {0: "batch", 2: "present_sequence"}
        decoder_dynamic_axes[present_value_name] = {0: "batch", 2: "present_sequence"}

    sessions.append(
        SessionSpec(
            name="decoder_model_merged",
            raw_filename="decoder_model_merged.onnx",
            package_filename=package_filenames.get("decoder_model_merged", "decoder_model_merged_q4f16.onnx"),
            inputs=decoder_inputs,
            outputs=decoder_outputs,
            dynamic_axes=decoder_dynamic_axes,
            notes=[
                "Cache names follow the flat Transformers.js convention (`past_key_values.*` in, `present.*` out).",
                "The merged decoder uses precomputed image features plus mm_token_type_ids to preserve the multimodal path.",
            ],
        )
    )

    return ExportContract(
        ok=ok,
        model_type=str(source_config.get("model_type", "")),
        architecture=architecture,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        image_token_id=int(image_token_id) if image_token_id is not None else None,
        video_token_id=int(video_token_id) if video_token_id is not None else None,
        sessions=sessions,
        warnings=warnings,
    )


def render_qwen3_5_export_runner(
    contract: ExportContract,
    *,
    source_path: str,
    base_path: str,
    output_dir: str,
    report_path: str,
    opset_version: int,
) -> str:
    contract_literal = repr(contract.to_dict())
    default_source_path = json.dumps(str(Path(source_path).expanduser().resolve()))
    default_base_path = json.dumps(str(Path(base_path).expanduser().resolve()))
    default_output_dir = json.dumps(str(Path(output_dir).expanduser().resolve()))
    default_report_path = json.dumps(str(Path(report_path).expanduser().resolve()))
    template = dedent(
        f"""\
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse
        import json
        import math
        import types
        from pathlib import Path

        import onnx
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, Qwen3_5ForConditionalGeneration

        CONTRACT = __CONTRACT_JSON__


        def _load_json(path: Path) -> dict:
            return json.loads(path.read_text(encoding="utf-8"))


        def _resolve_processor_config(source_path: Path, base_path: Path) -> dict:
            for root in (source_path, base_path):
                for filename in ("processor_config.json", "preprocessor_config.json"):
                    candidate = root / filename
                    if candidate.exists():
                        return _load_json(candidate)
            return {{}}


        def _load_image_processor(source_path: Path, base_path: Path):
            for root in (source_path, base_path):
                try:
                    return AutoImageProcessor.from_pretrained(str(root), trust_remote_code=False)
                except Exception:
                    continue
            return None


        class _PatchedScaledDotProductAttention:
            def __enter__(self):
                self.original = torch.nn.functional.scaled_dot_product_attention
                torch.nn.functional.scaled_dot_product_attention = _onnx_safe_scaled_dot_product_attention
                return self

            def __exit__(self, exc_type, exc, tb):
                torch.nn.functional.scaled_dot_product_attention = self.original
                return False


        def _onnx_safe_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            if scale is None:
                scale = 1.0 / math.sqrt(query.shape[-1])

            if enable_gqa and query.shape[-3] != key.shape[-3]:
                if query.shape[-3] % key.shape[-3] != 0:
                    raise ValueError("query heads must be divisible by key heads when enable_gqa=True")
                repeat_factor = query.shape[-3] // key.shape[-3]
                key = key.repeat_interleave(repeat_factor, dim=-3)
                value = value.repeat_interleave(repeat_factor, dim=-3)

            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    fill_value = torch.finfo(attn_scores.dtype).min
                    attn_scores = attn_scores.masked_fill(~attn_mask, fill_value)
                else:
                    attn_scores = attn_scores + attn_mask.to(dtype=attn_scores.dtype)

            if is_causal:
                query_length = query.shape[-2]
                key_length = key.shape[-2]
                causal_mask = torch.ones((query_length, key_length), dtype=torch.bool, device=query.device).tril(
                    diagonal=key_length - query_length
                )
                fill_value = torch.finfo(attn_scores.dtype).min
                attn_scores = attn_scores.masked_fill(~causal_mask, fill_value)

            attn_probs = torch.softmax(attn_scores, dim=-1)
            if dropout_p and dropout_p > 0:
                attn_probs = torch.dropout(attn_probs, dropout_p, train=False)
            return torch.matmul(attn_probs, value)


        class AttrDict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__


        class FlatQwen35Cache:
            def __init__(self, entries):
                self.keys = [key for key, _ in entries]
                self.values = [value for _, value in entries]
                self.is_initialized = True

            def get_seq_length(self, layer_idx: int = 0):
                if not self.keys:
                    return 0
                return self.keys[layer_idx].shape[-2]

            def update(self, key_states, value_states, layer_idx: int):
                next_key_states = torch.cat([self.keys[layer_idx], key_states], dim=-2)
                next_value_states = torch.cat([self.values[layer_idx], value_states], dim=-2)
                self.keys[layer_idx] = next_key_states
                self.values[layer_idx] = next_value_states
                return next_key_states, next_value_states

            def flatten(self):
                values = []
                for key_states, value_states in zip(self.keys, self.values):
                    values.extend([key_states, value_states])
                return tuple(values)


        def _extract_visual_tensor(outputs):
            if torch.is_tensor(outputs):
                return outputs
            if hasattr(outputs, "image_features") and torch.is_tensor(outputs.image_features):
                return outputs.image_features
            if hasattr(outputs, "last_hidden_state") and torch.is_tensor(outputs.last_hidden_state):
                return outputs.last_hidden_state
            if isinstance(outputs, dict):
                if "image_features" in outputs and torch.is_tensor(outputs["image_features"]):
                    return outputs["image_features"]
                if "last_hidden_state" in outputs and torch.is_tensor(outputs["last_hidden_state"]):
                    return outputs["last_hidden_state"]
                tensors = [value for value in outputs.values() if torch.is_tensor(value)]
                if len(tensors) == 1:
                    return tensors[0]
            raise TypeError(f"unsupported get_image_features output type: {{type(outputs)!r}}")


        def _pack_visual_outputs(image_features):
            return AttrDict({{
                "image_features": image_features,
                "last_hidden_state": image_features,
            }})


        class Qwen35VisionEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values, image_grid_thw):
                outputs = self.model.model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    return_dict=True,
                )
                return _extract_visual_tensor(outputs)


        class Qwen35EmbedTokensWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                config = self.model.config
                pad_token_id = config.pad_token_id
                multimodal_mask = input_ids == config.image_token_id
                if getattr(config, "video_token_id", None) is not None:
                    multimodal_mask = multimodal_mask | (input_ids == config.video_token_id)
                llm_input_ids = input_ids.clone()
                llm_input_ids[multimodal_mask] = pad_token_id
                return self.model.get_input_embeddings()(llm_input_ids)


        class Qwen35MergedDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(
                self,
                inputs_embeds,
                image_features,
                image_grid_thw,
                mm_token_type_ids,
                attention_mask,
                position_ids,
                *past_key_values,
            ):
                cache_entries = []
                for index in range(0, len(past_key_values), 2):
                    cache_entries.append((past_key_values[index], past_key_values[index + 1]))
                flat_cache = FlatQwen35Cache(cache_entries)

                owner = self.model.model
                original_get_image_features = owner.get_image_features

                def patched_get_image_features(_self, pixel_values=None, image_grid_thw=None, **kwargs):
                    return _pack_visual_outputs(image_features)

                owner.get_image_features = types.MethodType(patched_get_image_features, owner)
                dummy_pixel_values = torch.zeros((1,), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                try:
                    outputs = self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=flat_cache,
                        pixel_values=dummy_pixel_values,
                        image_grid_thw=image_grid_thw,
                        mm_token_type_ids=mm_token_type_ids,
                        use_cache=True,
                        return_dict=True,
                    )
                finally:
                    owner.get_image_features = original_get_image_features

                return (outputs.logits, *flat_cache.flatten())


        def _parse_dtype(name: str):
            if name == "auto":
                return "auto"
            if not hasattr(torch, name):
                raise ValueError(f"unsupported torch dtype: {{name}}")
            return getattr(torch, name)


        def _load_model(source_path: Path, dtype_name: str, device: str):
            torch_dtype = _parse_dtype(dtype_name)
            model_kwargs = {{"trust_remote_code": False}}
            if torch_dtype != "auto":
                model_kwargs["torch_dtype"] = torch_dtype
            model = Qwen3_5ForConditionalGeneration.from_pretrained(str(source_path), **model_kwargs)
            model.to(device)
            model.eval()
            return model


        def _resolve_export_image_budget(processor_config: dict):
            image_processor = processor_config.get("image_processor", processor_config)
            patch_size = int(image_processor.get("patch_size", 16)) if isinstance(image_processor, dict) else 16
            merge_size = int(image_processor.get("merge_size", 2)) if isinstance(image_processor, dict) else 2
            spatial_compression = max(patch_size * merge_size, 1)

            # Qwen's shortest/longest_edge values are pixel budgets, not literal dimensions.
            # Use a small fixed token budget for export so the ONNX trace stays bounded.
            target_visual_tokens = 256
            target_pixels = target_visual_tokens * spatial_compression * spatial_compression

            if not isinstance(image_processor, dict):
                return target_pixels

            size = image_processor.get("size")
            if isinstance(size, dict):
                min_pixels = size.get("shortest_edge")
                max_pixels = size.get("longest_edge")
                if isinstance(min_pixels, int):
                    target_pixels = max(target_pixels, min_pixels)
                if isinstance(max_pixels, int):
                    target_pixels = min(target_pixels, max_pixels)
            return target_pixels


        def _resolve_export_image_size(processor_config: dict):
            image_processor = processor_config.get("image_processor", processor_config)
            patch_size = int(image_processor.get("patch_size", 16)) if isinstance(image_processor, dict) else 16
            merge_size = int(image_processor.get("merge_size", 2)) if isinstance(image_processor, dict) else 2
            multiple = max(patch_size * merge_size, 1)
            target_pixels = _resolve_export_image_budget(processor_config)
            image_size = max(int(math.sqrt(target_pixels)), multiple)
            image_size = max((image_size // multiple) * multiple, multiple)
            return image_size


        def _build_sample_inputs(model, processor_config: dict, image_processor):
            config = model.config
            hidden_dtype = model.get_input_embeddings().weight.dtype
            device = next(model.parameters()).device

            image_size = _resolve_export_image_size(processor_config)
            if image_processor is not None:
                if hasattr(image_processor, "size"):
                    export_budget = _resolve_export_image_budget(processor_config)
                    image_processor.size = {{
                        "shortest_edge": export_budget,
                        "longest_edge": export_budget,
                    }}
                blank_image = Image.new("RGB", (image_size, image_size), color=0)
                processed = image_processor(images=blank_image, return_tensors="pt")
                pixel_values = processed["pixel_values"].to(device=device, dtype=hidden_dtype)
                image_grid_thw = processed["image_grid_thw"].to(device=device, dtype=torch.long)
            else:
                vision_config = config.vision_config
                patch_size = int(getattr(vision_config, "patch_size", 16))
                grid_size = max(image_size // patch_size, 1)
                pixel_values = torch.zeros((1, 3, image_size, image_size), dtype=hidden_dtype, device=device)
                image_grid_thw = torch.tensor([[1, grid_size, grid_size]], dtype=torch.long, device=device)

            input_ids = torch.full((1, 16), config.pad_token_id, dtype=torch.long, device=device)
            if getattr(config, "image_token_id", None) is not None:
                input_ids[:, 1] = config.image_token_id
            mm_token_type_ids = torch.zeros((1, 16), dtype=torch.int32, device=device)
            mm_token_type_ids[:, 1] = 1
            attention_mask = torch.ones((1, 16), dtype=torch.bool, device=device)
            position_ids = torch.arange(16, dtype=torch.long, device=device).unsqueeze(0)

            embed_wrapper = Qwen35EmbedTokensWrapper(model).to(device)
            vision_wrapper = Qwen35VisionEncoderWrapper(model).to(device)
            inputs_embeds = embed_wrapper(input_ids)
            image_features = vision_wrapper(pixel_values, image_grid_thw)

            cache_tensors = []
            for _ in range(CONTRACT["num_hidden_layers"]):
                cache_shape = (1, CONTRACT["num_key_value_heads"], 1, CONTRACT["head_dim"])
                cache_tensors.append(torch.zeros(cache_shape, dtype=hidden_dtype, device=device))
                cache_tensors.append(torch.zeros(cache_shape, dtype=hidden_dtype, device=device))

            return {{
                "vision_encoder": (pixel_values, image_grid_thw),
                "embed_tokens": (input_ids,),
                "decoder_model_merged": (
                    inputs_embeds,
                    image_features,
                    image_grid_thw,
                    mm_token_type_ids,
                    attention_mask,
                    position_ids,
                    *cache_tensors,
                ),
            }}


        def _normalize_external_data(output_path: Path):
            model = onnx.load(str(output_path), load_external_data=False)
            onnx.save_model(
                model,
                str(output_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{{output_path.name}}_data",
                size_threshold=0,
                convert_attribute=False,
            )


        def _export_onnx(module, sample_inputs, session_spec: dict, output_path: Path, opset_version: int):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dynamic_axes = session_spec["dynamic_axes"]
            if session_spec["name"] == "vision_encoder":
                # The vision encoder trace is more reliable with fixed shapes here.
                dynamic_axes = None
            export_kwargs = {{
                "args": sample_inputs,
                "f": str(output_path),
                "input_names": session_spec["inputs"],
                "output_names": session_spec["outputs"],
                "opset_version": opset_version,
                "do_constant_folding": True,
                "dynamo": False,
            }}
            if dynamic_axes is not None:
                export_kwargs["dynamic_axes"] = dynamic_axes
            try:
                with _PatchedScaledDotProductAttention():
                    torch.onnx.export(module, **export_kwargs, external_data=True)
            except TypeError:
                export_kwargs.pop("dynamo", None)
                try:
                    with _PatchedScaledDotProductAttention():
                        torch.onnx.export(module, **export_kwargs, external_data=True)
                except TypeError:
                    with _PatchedScaledDotProductAttention():
                        torch.onnx.export(module, **export_kwargs, use_external_data_format=True)
            _normalize_external_data(output_path)


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(prog="qwen3_5-export-runner")
            parser.add_argument("--source-path", default={default_source_path})
            parser.add_argument("--base-path", default={default_base_path})
            parser.add_argument("--output-dir", default={default_output_dir})
            parser.add_argument("--report-path", default={default_report_path})
            parser.add_argument("--device", default="cpu")
            parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
            parser.add_argument("--opset-version", default={opset_version}, type=int)
            args = parser.parse_args(argv)

            source_path = Path(args.source_path).expanduser().resolve()
            base_path = Path(args.base_path).expanduser().resolve()
            output_dir = Path(args.output_dir).expanduser().resolve()
            report_path = Path(args.report_path).expanduser().resolve()

            if not source_path.exists():
                raise FileNotFoundError(f"source path does not exist: {{source_path}}")
            if not CONTRACT["ok"]:
                raise RuntimeError("contract validation failed: " + "; ".join(CONTRACT["warnings"]))

            processor_config = _resolve_processor_config(source_path, base_path)
            image_processor = _load_image_processor(source_path, base_path)
            model = _load_model(source_path, args.torch_dtype, args.device)
            sample_inputs = _build_sample_inputs(model, processor_config, image_processor)

            wrappers = {{
                "vision_encoder": Qwen35VisionEncoderWrapper(model),
                "embed_tokens": Qwen35EmbedTokensWrapper(model),
                "decoder_model_merged": Qwen35MergedDecoderWrapper(model),
            }}

            results = {{}}
            for session_spec in CONTRACT["sessions"]:
                session_name = session_spec["name"]
                output_path = output_dir / session_spec["raw_filename"]
                wrapper = wrappers[session_name].to(args.device)
                wrapper.eval()
                _export_onnx(
                    wrapper,
                    sample_inputs[session_name],
                    session_spec,
                    output_path,
                    args.opset_version,
                )
                results[session_name] = {{
                    "raw_path": str(output_path),
                    "external_data_path": str(output_path.with_name(f"{{output_path.name}}_data")),
                    "package_filename": session_spec["package_filename"],
                }}

            report = {{
                "ok": True,
                "contract": CONTRACT,
                "output_dir": str(output_dir),
                "results": results,
            }}
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )
    return template.replace("__CONTRACT_JSON__", contract_literal)
