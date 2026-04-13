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
- direct `alkahest-0.8b` export is still blocked during the vision sample/export step

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
- the direct Qwen export lane is still blocked in the vision sample step

What is likely true next:

- direct Qwen browser export is still feasible
- but it needs a faithful processor-driven sample path, a sane export-only image budget, and an ONNX-safe attention path for the visual tower

The next debug target is not the decoder and not quantization.

It is:

- build a vision sample that exactly matches the upstream Qwen processor/model expectation
- keep the Qwen export trace on a bounded image-token budget instead of inheriting the full processor pixel range
- patch or bypass the Qwen vision SDPA/GQA path so ONNX export can lower it safely
- then resume ONNX export from there
