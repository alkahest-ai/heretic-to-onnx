# Heretic to ONNX Plan

As of 2026-04-11, a reusable converter is realistic for Gemma 4 "Heretic" models that are still standard Hugging Face Transformers checkpoints.

`p-e-w/gemma-4-E2B-it-heretic-ara` fits that requirement:

- it is published as `Transformers` + `Safetensors`
- the repo exposes `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja`
- the model page shows it as a `Gemma4ForConditionalGeneration` checkpoint

## Short answer

Yes, this should work for `p-e-w/gemma-4-E2B-it-heretic-ara`.

But the converter should be designed as:

- architecture-specific
- browser-targeted
- manifest-driven

Not as a single generic "convert any heretic model" script.

## Why it should work

The source model appears to be a full Gemma 4 checkpoint, not:

- GGUF only
- LoRA adapter only
- custom `trust_remote_code` architecture

That matters because the ONNX export path depends on the source model still matching the native Gemma 4 graph expected by Hugging Face and Transformers.js.

## The one important gap

`p-e-w/gemma-4-E2B-it-heretic-ara` does not appear to publish the full processor assets needed for a browser-ready ONNX repo.

In practice, the converter needs to inherit these from the base model:

- `processor_config.json`
- `preprocessor_config.json`

For this target, the base should be `google/gemma-4-E2B-it`.

## Product goal

Build a tool that takes a Gemma 4 Heretic-style Hugging Face repo and produces a new repo that can be loaded in the browser like:

```js
const processor = await AutoProcessor.from_pretrained("your-org/heretic-gemma-4-e2b-it-onnx");
const model = await Gemma4ForConditionalGeneration.from_pretrained(
  "your-org/heretic-gemma-4-e2b-it-onnx",
  { device: "webgpu", dtype: "q4f16" },
);
```

## Scope

Phase 1 scope:

- Gemma 4 E2B instruct-derived models
- full HF checkpoints only
- browser-ready ONNX package
- `q4f16` as the primary WebGPU target

Phase 2 scope:

- Gemma 4 E4B
- additional dtypes like `fp16` and `q4`
- optional CPU ONNX packaging

Out of scope for the first version:

- GGUF input repos
- adapter-only repos
- arbitrary architectures outside Gemma 4

## Required output contract

To behave like the ONNX Community Gemma 4 browser models, the output repo should contain:

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `processor_config.json`
- `preprocessor_config.json`
- `onnx/audio_encoder_q4f16.onnx`
- `onnx/audio_encoder_q4f16.onnx_data`
- `onnx/vision_encoder_q4f16.onnx`
- `onnx/vision_encoder_q4f16.onnx_data`
- `onnx/embed_tokens_q4f16.onnx`
- `onnx/embed_tokens_q4f16.onnx_data`
- `onnx/decoder_model_merged_q4f16.onnx`
- `onnx/decoder_model_merged_q4f16.onnx_data`

For the ONNX Community E2B build, the q4f16 files are approximately:

- audio encoder: 171 MB
- vision encoder: 99 MB
- embed tokens: 1.59 GB
- decoder merged: 1.52 GB

That size profile is one reason E2B is the right first target.

## Config requirement

The output `config.json` also needs the browser metadata used by Transformers.js.

At minimum, the Gemma 4 ONNX Community package adds a `transformers.js_config` block with:

- `use_external_data_format`
- `kv_cache_dtype`

If this block is missing or wrong, browser loading may fail or silently choose the wrong assumptions.

## Proposed repo layout

```text
tools/
  heretic_to_onnx/
    convert.py
    inspect.py
    export_gemma4.py
    quantize_gemma4.py
    package_repo.py
    validate_repo.py
configs/
  heretic-to-onnx.gemma4-e2b-heretic-ara.yaml
docs/
  heretic-to-onnx-plan.md
```

## Converter design

The converter should be manifest-driven.

Example inputs:

- source model id
- base model id
- architecture family
- target dtype
- target modalities
- output repo id or output directory

Example command:

```bash
python -m tools.heretic_to_onnx.convert \
  --config configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml
```

## Pipeline

### Step 1: inspect source

Validate that the source repo:

- has `model.safetensors`
- has `config.json`
- does not require custom remote code
- matches a supported architecture

For Gemma 4, reject anything that is not:

- `Gemma4ForConditionalGeneration`
- or a directly compatible Gemma 4 text variant we explicitly support later

### Step 2: resolve inherited assets

Pull text assets from the Heretic repo:

- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `generation_config.json`

Pull missing processor assets from the base Gemma repo:

- `processor_config.json`
- `preprocessor_config.json`

If the source repo already provides equivalent processor files, prefer the source.

### Step 3: export ONNX graphs

Export the Gemma 4 model into the same split used by the ONNX Community package:

- `audio_encoder`
- `vision_encoder`
- `embed_tokens`
- `decoder_model_merged`

Do not try to export one giant model file for browser use.

The exporter should preserve:

- Gemma 4 cache layout
- PLE behavior
- multimodal token wiring
- external data sharding

### Step 4: quantize for browser

Primary target:

- `q4f16`

Secondary targets later:

- `fp16`
- `q4`

Keep the quantization step separate from export so you can debug graph correctness before compression.

### Step 5: patch browser metadata

Patch the final `config.json` to include the ONNX-specific `transformers.js_config` block.

Also ensure:

- `architectures` remains `Gemma4ForConditionalGeneration`
- token ids match the source/base model pair
- the chat template stays aligned with the Heretic model behavior

### Step 6: package repo

Write the final repo structure with:

- small JSON/text metadata files
- ONNX files under `onnx/`
- external-data shards named exactly the way the ONNX files reference them

### Step 7: validate

Run validation in this order:

1. PyTorch source checkpoint generates a smoke-test answer
2. ONNX Runtime Python load succeeds for each exported session
3. ONNX Runtime Python generation succeeds for one short text prompt
4. Transformers.js can load the packaged repo with `device: "webgpu"` and `dtype: "q4f16"`

## Fastest implementation strategy

Do not try to support every Heretic model at once.

Build the first version specifically for:

- `p-e-w/gemma-4-E2B-it-heretic-ara`
- base: `google/gemma-4-E2B-it`
- target dtype: `q4f16`

Once that works, generalize to other Gemma 4 Heretic repos by swapping only the manifest.

## Risks

### Export logic risk

The hardest part is not downloading the weights.

The hardest part is reproducing the exact split-graph export contract that Transformers.js expects for Gemma 4.

### Asset mismatch risk

If the Heretic repo changes tokenizer, chat template, or generation config in ways that do not match the base processor files, browser behavior can drift even if export succeeds.

### Browser memory risk

E2B is browser-feasible.

E4B is possible, but it is much less forgiving on consumer hardware.

## Decision

The right build plan is:

1. create a Gemma 4 E2B-only converter first
2. target `p-e-w/gemma-4-E2B-it-heretic-ara` first
3. inherit missing processor assets from `google/gemma-4-E2B-it`
4. match the ONNX Community repo contract exactly
5. validate in Python ONNX Runtime before testing WebGPU in the browser

If that first target works, then turning other Gemma 4 Heretic checkpoints into WebGPU-ready ONNX packages becomes mostly a packaging and validation problem rather than a new exporter problem.
