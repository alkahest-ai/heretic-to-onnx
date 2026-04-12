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
    use_bidirectional_attention: str | None
    num_hidden_layers: int
    num_kv_shared_layers: int
    num_cache_layers: int
    cache_layer_types: list[str]
    hidden_size_per_layer_input: int
    sessions: list[SessionSpec]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "use_bidirectional_attention": self.use_bidirectional_attention,
            "num_hidden_layers": self.num_hidden_layers,
            "num_kv_shared_layers": self.num_kv_shared_layers,
            "num_cache_layers": self.num_cache_layers,
            "cache_layer_types": list(self.cache_layer_types),
            "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
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


def build_gemma4_export_contract(
    manifest: Manifest,
    source_path: str | Path,
) -> ExportContract:
    source_config = _load_source_config(source_path)
    text_config = source_config.get("text_config", {})
    package_filenames = _session_filename_map(manifest)

    architecture = ""
    architectures = source_config.get("architectures")
    if isinstance(architectures, list) and architectures:
        architecture = str(architectures[0])

    num_hidden_layers = int(text_config.get("num_hidden_layers", 0))
    num_kv_shared_layers = int(text_config.get("num_kv_shared_layers", 0))
    num_cache_layers = max(num_hidden_layers - num_kv_shared_layers, 0)
    layer_types = [str(value) for value in text_config.get("layer_types", [])[:num_cache_layers]]
    hidden_size_per_layer_input = int(text_config.get("hidden_size_per_layer_input", 0))
    use_bidirectional_attention = text_config.get("use_bidirectional_attention")

    warnings: list[str] = []
    ok = True

    if source_config.get("model_type") != "gemma4":
        ok = False
        warnings.append(f"expected model_type=gemma4, got {source_config.get('model_type')!r}")
    if architecture != manifest.expected_architecture:
        ok = False
        warnings.append(f"expected architecture {manifest.expected_architecture}, got {architecture!r}")
    if hidden_size_per_layer_input <= 0:
        ok = False
        warnings.append("Gemma 4 export currently expects hidden_size_per_layer_input > 0")
    if use_bidirectional_attention == "vision":
        ok = False
        warnings.append(
            "Gemma 4 configs with use_bidirectional_attention='vision' are not supported by the current browser contract"
        )

    sessions = [
        SessionSpec(
            name="vision_encoder",
            raw_filename="vision_encoder.onnx",
            package_filename=package_filenames.get("vision_encoder", "vision_encoder_q4f16.onnx"),
            inputs=["pixel_values", "pixel_position_ids"],
            outputs=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "image_batch", 2: "image_height", 3: "image_width"},
                "pixel_position_ids": {0: "image_batch", 1: "image_patches"},
                "image_features": {0: "image_tokens"},
            },
            notes=["Matches the Transformers.js Gemma 4 vision session input name `pixel_position_ids`."],
        ),
        SessionSpec(
            name="audio_encoder",
            raw_filename="audio_encoder.onnx",
            package_filename=package_filenames.get("audio_encoder", "audio_encoder_q4f16.onnx"),
            inputs=["input_features", "input_features_mask"],
            outputs=["audio_features"],
            dynamic_axes={
                "input_features": {0: "audio_batch", 1: "audio_frames"},
                "input_features_mask": {0: "audio_batch", 1: "audio_frames"},
                "audio_features": {0: "audio_tokens"},
            },
            notes=["Audio padding is stripped in the wrapper so the output matches browser merge semantics."],
        ),
        SessionSpec(
            name="embed_tokens",
            raw_filename="embed_tokens.onnx",
            package_filename=package_filenames.get("embed_tokens", "embed_tokens_q4f16.onnx"),
            inputs=["input_ids"],
            outputs=["inputs_embeds", "per_layer_inputs"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "inputs_embeds": {0: "batch", 1: "sequence"},
                "per_layer_inputs": {0: "batch", 1: "sequence"},
            },
            notes=["Multimodal placeholder token IDs are replaced with PAD before embedding, matching HF Gemma 4."],
        ),
    ]

    decoder_inputs = ["inputs_embeds", "per_layer_inputs", "attention_mask", "position_ids"]
    decoder_outputs = ["logits"]
    decoder_dynamic_axes: dict[str, dict[int, str]] = {
        "inputs_embeds": {0: "batch", 1: "sequence"},
        "per_layer_inputs": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "attention_sequence"},
        "position_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }

    for layer_index in range(num_cache_layers):
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
                "Only the non-shared Gemma 4 KV layers are exposed as ONNX cache tensors.",
            ],
        )
    )

    return ExportContract(
        ok=ok,
        model_type=str(source_config.get("model_type", "")),
        architecture=architecture,
        use_bidirectional_attention=str(use_bidirectional_attention)
        if use_bidirectional_attention is not None
        else None,
        num_hidden_layers=num_hidden_layers,
        num_kv_shared_layers=num_kv_shared_layers,
        num_cache_layers=num_cache_layers,
        cache_layer_types=layer_types,
        hidden_size_per_layer_input=hidden_size_per_layer_input,
        sessions=sessions,
        warnings=warnings,
    )


def render_gemma4_export_runner(
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
        import inspect
        import json
        from pathlib import Path

        import numpy as np
        import onnx
        import torch
        from transformers import AutoFeatureExtractor, AutoImageProcessor, Gemma4ForConditionalGeneration

        CONTRACT = __CONTRACT_JSON__


        def _load_json(path: Path) -> dict:
            return json.loads(path.read_text(encoding="utf-8"))


        def _resolve_processor_config(source_path: Path, base_path: Path) -> dict:
            for root in (source_path, base_path):
                candidate = root / "processor_config.json"
                if candidate.exists():
                    return _load_json(candidate)
            return {{}}


        def _patch_masking_utils_for_onnx_export():
            try:
                from transformers import masking_utils
            except Exception:
                return

            original_sdpa_mask = getattr(masking_utils, "sdpa_mask", None)
            if original_sdpa_mask is None or getattr(original_sdpa_mask, "__name__", "") == "_patched_sdpa_mask":
                return

            try:
                original_signature = inspect.signature(original_sdpa_mask)
            except (TypeError, ValueError):
                original_signature = None

            def _normalize_cache_position(q_length, q_offset, attention_mask):
                device = torch.device("cpu")
                if torch.is_tensor(q_length):
                    device = q_length.device
                elif torch.is_tensor(attention_mask):
                    device = attention_mask.device

                if q_length is None:
                    return None
                if torch.is_tensor(q_length):
                    if q_length.ndim == 0:
                        cache_position = torch.arange(int(q_length.item()), device=device, dtype=torch.long)
                    else:
                        cache_position = q_length.to(device=device, dtype=torch.long)
                else:
                    cache_position = torch.arange(int(q_length), device=device, dtype=torch.long)

                if torch.is_tensor(q_offset):
                    cache_position = cache_position + q_offset.to(device=device, dtype=torch.long)
                elif q_offset:
                    cache_position = cache_position + int(q_offset)
                return cache_position

            def _patched_sdpa_mask(*args, **kwargs):
                if original_signature is not None:
                    bound = original_signature.bind_partial(*args, **kwargs)
                    arguments = dict(bound.arguments)
                else:
                    arguments = dict(kwargs)

                attention_mask = arguments.get("attention_mask")
                q_offset = arguments.get("q_offset", 0)
                cache_position = arguments.get("cache_position")
                if cache_position is None:
                    cache_position = _normalize_cache_position(arguments.get("q_length"), q_offset, attention_mask)

                if cache_position is None:
                    return original_sdpa_mask(*args, **kwargs)

                if not torch.is_tensor(cache_position):
                    cache_position = torch.as_tensor(cache_position, dtype=torch.long)
                cache_position = cache_position.to(dtype=torch.long)
                if cache_position.ndim == 0:
                    cache_position = torch.arange(int(cache_position.item()), device=cache_position.device, dtype=torch.long)

                batch_size = arguments.get("batch_size")
                if batch_size is None:
                    if torch.is_tensor(attention_mask):
                        batch_size = attention_mask.shape[0]
                    else:
                        batch_size = 1

                kv_length = arguments.get("kv_length")
                if kv_length is None:
                    return original_sdpa_mask(*args, **kwargs)

                kv_offset = arguments.get("kv_offset", 0)
                mask_function = arguments.get("mask_function")
                if mask_function is None:
                    mask_function = getattr(masking_utils, "causal_mask_function", None)
                if mask_function is None:
                    return original_sdpa_mask(*args, **kwargs)

                local_size = arguments.get("local_size")
                allow_is_causal_skip = bool(arguments.get("allow_is_causal_skip", True))
                allow_torch_fix = bool(arguments.get("allow_torch_fix", True))

                padding_mask = attention_mask
                if padding_mask is not None:
                    required_length = kv_length + (int(kv_offset.item()) if torch.is_tensor(kv_offset) else int(kv_offset))
                    if padding_mask.shape[-1] < required_length:
                        padding_mask = torch.nn.functional.pad(padding_mask, (0, required_length - padding_mask.shape[-1]))

                    mask_indices = torch.arange(kv_length, device=padding_mask.device)
                    if torch.is_tensor(kv_offset):
                        mask_indices = mask_indices + kv_offset.to(device=padding_mask.device, dtype=mask_indices.dtype)
                    elif kv_offset:
                        mask_indices = mask_indices + int(kv_offset)
                    padding_mask = padding_mask[:, mask_indices].to(dtype=torch.bool)

                if allow_is_causal_skip:
                    query_length = cache_position.shape[0]
                    can_skip = query_length == 1 or kv_length == query_length
                    if local_size is not None:
                        can_skip = can_skip and kv_length < local_size
                    if can_skip and (padding_mask is None or padding_mask.all()):
                        return None

                vmap_for_bhqkv = getattr(masking_utils, "_vmap_for_bhqkv", None)
                if vmap_for_bhqkv is None:
                    return original_sdpa_mask(*args, **kwargs)

                kv_arange = torch.arange(kv_length, device=cache_position.device)
                if torch.is_tensor(kv_offset):
                    kv_arange = kv_arange + kv_offset.to(device=cache_position.device, dtype=kv_arange.dtype)
                elif kv_offset:
                    kv_arange = kv_arange + int(kv_offset)

                mask = vmap_for_bhqkv(mask_function, bh_indices=False)(None, None, cache_position, kv_arange)
                mask = mask[None, None, :, :].expand(batch_size, -1, -1, -1)

                if padding_mask is not None:
                    mask = mask & padding_mask[:, None, None, :]

                if allow_torch_fix and not getattr(masking_utils, "_is_torch_greater_or_equal_than_2_5", True):
                    mask = mask | torch.all(~mask, dim=-1, keepdim=True)

                return mask

            _patched_sdpa_mask.__name__ = "_patched_sdpa_mask"
            masking_utils.sdpa_mask = _patched_sdpa_mask
            try:
                masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = masking_utils.sdpa_mask
            except Exception:
                pass


        def _load_image_processor(source_path: Path, base_path: Path):
            last_error = None
            for root in (source_path, base_path):
                candidate = root / "processor_config.json"
                if not candidate.exists():
                    continue
                try:
                    return AutoImageProcessor.from_pretrained(str(root), trust_remote_code=False)
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise RuntimeError("failed to load Gemma 4 image processor from source/base assets") from last_error
            raise FileNotFoundError("processor_config.json was not found in the source or base model assets")


        def _load_feature_extractor(source_path: Path, base_path: Path):
            last_error = None
            for root in (source_path, base_path):
                candidate = root / "processor_config.json"
                fallback = root / "preprocessor_config.json"
                if not candidate.exists() and not fallback.exists():
                    continue
                try:
                    return AutoFeatureExtractor.from_pretrained(str(root), trust_remote_code=False)
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise RuntimeError("failed to load Gemma 4 feature extractor from source/base assets") from last_error
            raise FileNotFoundError("processor_config.json/preprocessor_config.json was not found in the source or base model assets")


        class FlatGemma4Cache:
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


        class Gemma4VisionEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values, pixel_position_ids):
                outputs = self.model.model.get_image_features(
                    pixel_values=pixel_values,
                    image_position_ids=pixel_position_ids,
                    return_dict=True,
                )
                return outputs.pooler_output


        class Gemma4AudioEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_features, input_features_mask):
                outputs = self.model.model.get_audio_features(
                    input_features=input_features,
                    input_features_mask=input_features_mask,
                    return_dict=True,
                )
                audio_features = outputs.pooler_output
                audio_mask = outputs.attention_mask.to(torch.bool)
                return audio_features[audio_mask]


        class Gemma4EmbedTokensWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                config = self.model.config
                multimodal_mask = (
                    (input_ids == config.image_token_id)
                    | (input_ids == config.video_token_id)
                    | (input_ids == config.audio_token_id)
                )
                llm_input_ids = input_ids.clone()
                llm_input_ids[multimodal_mask] = config.text_config.pad_token_id
                inputs_embeds = self.model.model.get_input_embeddings()(llm_input_ids)

                pad_embedding = self.model.model.language_model.embed_tokens.weight[config.text_config.pad_token_id, :]
                llm_inputs_embeds = torch.where(
                    multimodal_mask[..., None],
                    pad_embedding.view(1, 1, -1),
                    inputs_embeds,
                )
                per_layer_inputs = self.model.model.language_model.get_per_layer_inputs(
                    llm_input_ids,
                    llm_inputs_embeds,
                )
                return inputs_embeds, per_layer_inputs


        class Gemma4MergedDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inputs_embeds, per_layer_inputs, attention_mask, position_ids, *past_key_values):
                cache_entries = []
                for index in range(0, len(past_key_values), 2):
                    cache_entries.append((past_key_values[index], past_key_values[index + 1]))
                flat_cache = FlatGemma4Cache(cache_entries)
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    per_layer_inputs=per_layer_inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=flat_cache,
                    use_cache=True,
                    return_dict=True,
                )
                return (outputs.logits, *flat_cache.flatten())


        def _parse_dtype(name: str):
            if name == "auto":
                return "auto"
            if not hasattr(torch, name):
                raise ValueError(f"unsupported torch dtype: {{name}}")
            return getattr(torch, name)


        def _position_grid(height_patches: int, width_patches: int):
            y_coords = torch.arange(height_patches, dtype=torch.long)
            x_coords = torch.arange(width_patches, dtype=torch.long)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).unsqueeze(0)


        def _resolve_image_size(processor_config: dict, patch_size: int, pooling_kernel_size: int):
            image_processor = processor_config.get("image_processor", processor_config)
            size = image_processor.get("size") if isinstance(image_processor, dict) else None
            multiple = max(patch_size * pooling_kernel_size, 1)

            def _normalize(value):
                if not isinstance(value, int) or value <= 0:
                    return multiple
                return max(((value + multiple - 1) // multiple) * multiple, multiple)

            if isinstance(size, dict):
                height = size.get("height")
                width = size.get("width")
                if not isinstance(height, int):
                    fallback = size.get("shortest_edge") or size.get("longest_edge")
                    height = fallback if isinstance(fallback, int) else None
                if not isinstance(width, int):
                    width = height
                return _normalize(height), _normalize(width)
            if isinstance(size, int):
                normalized = _normalize(size)
                return normalized, normalized
            return multiple, multiple


        def _load_model(source_path: Path, dtype_name: str, device: str):
            torch_dtype = _parse_dtype(dtype_name)
            model_kwargs = {{
                "trust_remote_code": False,
                # Eager attention avoids the traced SDPA mask path that currently fails
                # for Gemma 4 multimodal encoder export under this torch/transformers stack.
                "attn_implementation": "eager",
            }}
            if torch_dtype != "auto":
                model_kwargs["torch_dtype"] = torch_dtype
            model = Gemma4ForConditionalGeneration.from_pretrained(str(source_path), **model_kwargs)
            model.to(device)
            model.eval()
            return model


        def _build_sample_inputs(model, image_processor, feature_extractor, processor_config: dict):
            text_config = model.config.text_config
            hidden_dtype = model.model.language_model.embed_tokens.weight.dtype
            device = next(model.parameters()).device

            vision_config = model.model.vision_tower.config
            patch_size = int(getattr(vision_config, "patch_size", 16))
            pooling_kernel_size = int(model.model.vision_tower.config.pooling_kernel_size)
            image_height, image_width = _resolve_image_size(processor_config, patch_size, pooling_kernel_size)
            dummy_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            vision_batch = image_processor(images=[dummy_image], return_tensors="pt")
            pixel_values = vision_batch["pixel_values"].to(device)
            pixel_position_ids = vision_batch.get("image_position_ids", vision_batch.get("pixel_position_ids"))
            if pixel_position_ids is None:
                raise RuntimeError("Gemma 4 image processor did not return image_position_ids/pixel_position_ids")
            pixel_position_ids = pixel_position_ids.to(device)

            sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
            dummy_audio = np.zeros(sampling_rate, dtype=np.float32)
            audio_batch = feature_extractor(
                [dummy_audio],
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            input_features = audio_batch["input_features"].to(device)
            input_features_mask = audio_batch.get("input_features_mask", audio_batch.get("attention_mask"))
            if input_features_mask is None:
                raise RuntimeError("Gemma 4 feature extractor did not return an audio mask")
            input_features_mask = input_features_mask.to(device=device, dtype=torch.bool)

            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
            inputs_embeds = torch.zeros((1, 1, text_config.hidden_size), dtype=hidden_dtype, device=device)
            per_layer_inputs = torch.zeros(
                (1, 1, text_config.num_hidden_layers, text_config.hidden_size_per_layer_input),
                dtype=hidden_dtype,
                device=device,
            )
            attention_mask = torch.ones((1, 2), dtype=torch.bool, device=device)
            position_ids = torch.tensor([[1]], dtype=torch.long, device=device)

            cache_tensors = []
            layer_types = CONTRACT["cache_layer_types"]
            for layer_index in range(CONTRACT["num_cache_layers"]):
                layer_type = layer_types[layer_index] if layer_index < len(layer_types) else "sliding_attention"
                head_dim = text_config.global_head_dim if layer_type == "full_attention" else text_config.head_dim
                cache_shape = (1, text_config.num_key_value_heads, 1, head_dim)
                cache_tensors.append(torch.zeros(cache_shape, dtype=hidden_dtype, device=device))
                cache_tensors.append(torch.zeros(cache_shape, dtype=hidden_dtype, device=device))

            return {{
                "vision_encoder": (pixel_values, pixel_position_ids),
                "audio_encoder": (input_features, input_features_mask),
                "embed_tokens": (input_ids,),
                "decoder_model_merged": (
                    inputs_embeds,
                    per_layer_inputs,
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
            if session_spec["name"] in {{"vision_encoder", "audio_encoder"}}:
                # The multimodal encoder traces become unstable with symbolic sequence axes
                # under current torch/transformers versions, so export them with fixed shapes.
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
                torch.onnx.export(module, **export_kwargs, external_data=True)
            except TypeError:
                export_kwargs.pop("dynamo", None)
                try:
                    torch.onnx.export(module, **export_kwargs, external_data=True)
                except TypeError:
                    torch.onnx.export(module, **export_kwargs, use_external_data_format=True)
            _normalize_external_data(output_path)


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(prog="gemma4-export-runner")
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

            _patch_masking_utils_for_onnx_export()
            image_processor = _load_image_processor(source_path, base_path)
            feature_extractor = _load_feature_extractor(source_path, base_path)
            processor_config = _resolve_processor_config(source_path, base_path)
            model = _load_model(source_path, args.torch_dtype, args.device)
            sample_inputs = _build_sample_inputs(model, image_processor, feature_extractor, processor_config)

            wrappers = {{
                "vision_encoder": Gemma4VisionEncoderWrapper(model),
                "audio_encoder": Gemma4AudioEncoderWrapper(model),
                "embed_tokens": Gemma4EmbedTokensWrapper(model),
                "decoder_model_merged": Gemma4MergedDecoderWrapper(model),
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
