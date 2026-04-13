# Alkahest Qwen Browser Export Postmortem

Date context: April 12, 2026.

This is the running write-up for the direct `alkahest-*` browser export lane built from the Qwen 3.5 Heretic family.

Unlike the Rally 2B postmortem, this one documents an export lane that is **not fully shipped yet**. The point is to record what is true upstream, what the current browser package actually exposes, and what is still blocking a successful direct Qwen browser export.

## Current Status

Current public naming:

- `alkahest-*` = Qwen-based browser packages
- `rally-*` = Gemma-based browser packages

Current practical status:

- the Qwen ONNX packaging lane exists in this repo
- the current intended shipped browser surface is **text + image**
- direct `alkahest-0.8b` export is now past the unbounded vision sample issue, but is still blocked during merged decoder export tracing

So this is not a published success case yet. It is a bring-up postmortem.

## Upstream Capability vs Shipped Browser Capability

Important distinction:

- upstream Qwen 3.5 supports image understanding
- upstream Qwen 3.5 also supports video understanding
- current browser export work in this repo does **not** automatically retain every upstream multimodal path

That is the same core lesson as Rally/Gemma audio:

- if a modality is not exported, packaged, and wired through the browser runtime, it is not a real shipped capability even if the original checkpoint supports it

### What the current browser lane actually targets

For Qwen in this repo, the current browser package target is:

- `vision_encoder`
- `embed_tokens`
- `decoder_model_merged`

That means the actively targeted browser interface today is:

- text
- image

Not:

- video
- audio

## Is Qwen Video Possible Here?

Yes, **in principle**.

But not by accident, and not for free.

It would need a real export lane for:

- video-side processor outputs
- video token / placeholder handling
- temporal grid metadata
- session contracts that preserve the video path
- packaging rules for the additional processor metadata
- browser runtime support that knows how to prepare and feed video inputs

So the correct statement is:

- upstream Qwen video understanding exists
- current `alkahest-*` browser export does not yet ship that path
- shipping it would be a new implementation effort, not a small toggle

## Is Qwen Image Understanding Different From Gemma Image Understanding?

Yes.

At the browser-package level both families are currently treated as:

- text + image

But the underlying model strengths differ.

Gemma 4 is positioned more strongly around:

- OCR
- documents / PDFs
- UI / screen understanding
- chart comprehension

Qwen 3.5 is positioned more strongly around:

- broader multimodal benchmark coverage
- explicit video support upstream
- general visual-language capability across image and video

So the interface is similar, but the base-model strengths are not identical.

## What Broke So Far

### 1. Base repo asset assumptions were too strict

The first failure was not the model graph. It was the asset contract.

The official Qwen base repos do not always provide:

- `processor_config.json`

They do provide:

- `preprocessor_config.json`
- in some cases `video_preprocessor_config.json`

The original Qwen manifests and validation logic incorrectly required `processor_config.json`, which caused the direct run to fail during inspect before export even started.

The fix was:

- make the Qwen manifests depend on `preprocessor_config.json`
- make the package validator respect the manifest’s required asset list instead of hard-coding `processor_config.json`

### 2. The Qwen export runner was reading only one processor filename

The generated runner originally tried to resolve only:

- `processor_config.json`

That was too narrow for the real upstream asset layout.

The fix was:

- allow the export runner to resolve either `processor_config.json` or `preprocessor_config.json`

### 3. Synthetic vision sample construction is still wrong

The current blocker is inside the vision sample path used for export tracing.

The failing error is:

- `RuntimeError: The size of tensor a (8388608) must match the size of tensor b (4194304) at non-singleton dimension 0`

That happens when the export runner tries to produce image features from synthetic sample inputs for the Qwen visual stack.

The useful interpretation is:

- the synthetic `pixel_values` / `image_grid_thw` pair does not match what the upstream Qwen visual model expects

This is not a packaging failure and not a quantization failure.

It is a sample-input construction failure during raw ONNX export.

### 4. Hand-built image tensors were not reliable enough

The original Qwen sample path built vision inputs manually from:

- `image_size`
- `patch_size`
- `spatial_merge_size`

That was too brittle.

The next attempted fix was:

- generate the visual sample through Hugging Face `AutoImageProcessor`

That moved the export lane forward, but it exposed the next blocker in the visual path instead of fully solving the lane.

### 5. ONNX export currently fails on the Qwen vision SDPA / GQA path

Once the visual sample path became more faithful to upstream preprocessing, the export advanced into the actual Qwen vision attention stack.

The next failure was:

- `AssertionError: conversion of scaled_dot_product_attention not implemented if enable_gqa is True`

This is not a Qwen packaging issue. It is a PyTorch ONNX exporter limitation on the attention operator emitted by the current Qwen vision stack.

The practical implication is:

- the current direct Qwen browser lane needs an ONNX-safe fallback attention implementation during export

That is a model-export compatibility problem, not evidence that Qwen image understanding is impossible.

### 6. Qwen `shortest_edge` / `longest_edge` are pixel budgets, not literal dimensions

The next issue was subtler and more operational:

- the official Qwen processor config uses `size["shortest_edge"]` and `size["longest_edge"]` as `min_pixels` / `max_pixels`
- the exporter initially treated those values as literal image edge lengths
- for Qwen 3.5 0.8B, that meant a synthetic export image sized from `65536`, which exploded the traced vision sequence and got the export process killed with exit code `-9`

The fix is to keep using the official processor, but override it with a small export-only visual token budget before tracing ONNX.

This matters because Qwen's public processor semantics are unusual if you expect `size` to mean `height` / `width`.

### 7. The merged decoder export path patched the right idea with the wrong Qwen contract

After bounding the export image budget, the next traceback moved further forward into the merged decoder trace itself.

The failing error was:

- `TypeError: cat() received an invalid combination of arguments - got (NoneType, dim=int)`

The useful interpretation is:

- the export wrapper did successfully reach the multimodal reinjection path in Qwen
- but the export-only patch was returning the wrong structure from the temporary `get_image_features` hook
- current Qwen expects a `pooler_output` list that it later concatenates with `torch.cat(...)`
- the export shim was returning `image_features` / `last_hidden_state`, so Qwen saw `pooler_output=None` and failed

There was a second, related issue in the same area:

- the merged decoder export uses `inputs_embeds`, not raw `input_ids`
- the embed session intentionally masks image placeholder IDs to PAD before embedding
- current Qwen reconstructs the placeholder mask from `input_ids`, or from literal image-token embeddings when `input_ids` are absent
- that means the export wrapper also has to reconstruct the multimodal placeholder mask from `mm_token_type_ids` during tracing, otherwise Qwen cannot find the image slots to refill

So this blocker is not about the visual tower anymore.

It is about matching the current Hugging Face Qwen multimodal forward contract closely enough during the merged decoder export shim.

### 8. The ONNX export retry path was masking the real traceback

The same run also exposed an exporter-control bug in our generated runner.

The runner originally treated any `TypeError` raised during `torch.onnx.export(...)` as if it meant:

- an ONNX exporter keyword was unsupported

That was too broad.

In this case, the real `TypeError` came from inside the traced Qwen forward path, but the runner caught it and retried export with older keyword variants. That produced a misleading final error:

- `TypeError: export() got an unexpected keyword argument 'use_external_data_format'`

The practical fix is:

- detect supported `torch.onnx.export` kwargs up front
- retry only for true unexpected-keyword cases
- let model-forward `TypeError`s surface directly so the traceback stays useful

### 9. The decoder export sample needs one placeholder slot per merged image feature

The next refinement was in the decoder export sample itself.

Even after the export image budget was fixed, the synthetic decoder sample still only reserved one image placeholder token.

That is not how the Qwen multimodal path behaves in practice:

- one image can expand into many merged visual feature tokens
- so the decoder export sample must reserve that many image placeholder slots
- and the export wrapper should merge `image_features` back into `inputs_embeds` directly using `mm_token_type_ids`

That is more stable than depending on private Hugging Face reinjection hooks during ONNX export.

### 10. The flat cache shim also has to implement the newer masking helper API

Once the merged decoder sample moved far enough into the text stack, the next traceback came from the cache shim itself:

- `AttributeError: 'FlatQwen35Cache' object has no attribute 'get_mask_sizes'`

That is not a model-graph issue.

It means newer Transformers masking utilities expect the export-time KV cache adapter to provide:

- `get_seq_length(...)`
- `get_mask_sizes(...)`
- `update(...)`

So the direct Qwen export shim has to match not just the cache tensor shapes, but the cache helper interface expected by current masking code.

### 11. Transformers masking utils also need an ONNX-safe SDPA mask shim

After the cache adapter gained `get_mask_sizes(...)`, the next failure moved one level deeper into the newer Transformers masking utilities:

- `IndexError: tuple index out of range`
- inside `transformers.masking_utils.sdpa_mask`

The important detail is that the current masking path can receive scalar-form `q_length` / `cache_position` values during export tracing.

That is already handled in the Gemma export lane with a compatibility shim that:

- normalizes scalar `q_length` into a real `cache_position` tensor
- materializes the boolean causal mask explicitly
- keeps the ONNX trace away from the brittle branch in the default SDPA masking helper

So the Qwen export lane needs the same style of masking-utils patch, not a different model-specific workaround.

## Operational Nuance: Failed Pulls Can Make New Logs Look Old

One concrete trap showed up during debugging:

- `git fetch` and `git pull` failed on the Phala box with `Could not resolve host: github.com`
- `git rev-parse --short HEAD` therefore stayed on the older local commit
- rerunning the same export generated the same old runner and the same old tensor error

That matters because it can look like a new fix failed when the box never actually received the fix.

For this lane, always verify:

- `git fetch origin`
- `git pull --ff-only`
- `git rev-parse --short HEAD`

before trusting a rerun as evidence against a new patch.

## Why Video Does Not “Just Work” Even If Image Eventually Works

Image export success would still not imply video export success.

Video would need extra work for:

- processor metadata
- frame batching
- temporal grid layout
- placeholder token handling
- browser-side ingestion

So the right roadmap is:

1. get direct text + image Qwen export stable
2. validate browser packaging and runtime
3. decide whether video is worth a dedicated second multimodal lane

## Practical Conclusion

What is true right now:

- Qwen video understanding exists upstream
- Qwen image understanding exists upstream
- current `alkahest-*` browser export is intended to ship only `text + image`
- the direct Qwen export lane is still blocked, but the blocker has moved from synthetic vision sample sizing into the merged decoder export shim
- the direct Qwen export lane is still blocked, but the blocker is now in narrow compatibility shims around the merged decoder export path
- the direct Qwen export lane is still blocked, but the remaining blockers are now narrow compatibility shims around the merged decoder and masking helper paths

What is likely true next:

- direct Qwen browser export is still feasible
- but it needs a faithful processor-driven sample path, a sane export-only image budget, an ONNX-safe attention path for the visual tower, and a merged decoder shim that explicitly matches Qwen's merged visual token count

The next debug target is not quantization.

It is the merged decoder export shim around the decoder.

It is:

- keep the Qwen export trace on a bounded image-token budget instead of inheriting the full processor pixel range
- patch or bypass the Qwen vision SDPA/GQA path so ONNX export can lower it safely
- reserve one image placeholder slot per merged image feature token
- merge image features back into `inputs_embeds` directly from `mm_token_type_ids`
- keep the flat KV cache shim aligned with the current Transformers masking helper interface
- patch the newer Transformers SDPA masking helper for ONNX export just like the Gemma lane already does
- then resume ONNX export from there
