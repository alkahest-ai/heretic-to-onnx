# Model Portfolio Roadmap

This document organizes the model work for this repo into execution waves.

## North Star

Build an **Alkahest** model portfolio that is:

- private by default
- roleplay-capable
- compatible with browser/WebGPU deployment where practical
- explicit about provenance even when the public-facing model family name is Alkahest

## Named Line

Current naming decision:

- `alkahest-ai/rally-2b` = Gemma 4 E2B Heretic ONNX
- `alkahest-ai/rally-4b` = Gemma 4 E4B Heretic ONNX
- `alkahest-ai/alkahest-4b` = Qwen 3.5 4B Heretic ONNX
- `alkahest-ai/alkahest-2b` = Qwen 3.5 2B Heretic ONNX
- `alkahest-ai/alkahest-0.8b` = Qwen 3.5 0.8B Heretic ONNX

Direct provenance should still live in model cards and internal docs.

## Wave 1: Direct Heretic -> ONNX

These are the fastest wins because they avoid training in the first pass.

### Gemma 4 E2B

- Source: `p-e-w/gemma-4-E2B-it-heretic-ara`
- Goal: convert to browser/WebGPU ONNX
- Public name: `alkahest-ai/rally-2b`
- Status: current primary converter target

### Gemma 4 E4B

- Source: `coder3101/gemma-4-E4B-it-heretic`
- Goal: convert to browser/WebGPU ONNX
- Public name: `alkahest-ai/rally-4b`
- Status: next Gemma 4 target

Reason to do these first:

- proves the converter on two sizes
- creates immediate portfolio assets
- avoids spending the first H200 window only on training experiments

## Wave 2: Roleplay-Tuned Rally

After direct conversions work, move to your own tuned checkpoints.

### Target variants

- `alkahest-ai/rally-2b-rp`
- `alkahest-ai/rally-4b-rp`
- `alkahest-ai/alkahest-4b-rp`
- `alkahest-ai/alkahest-2b-rp`
- `alkahest-ai/alkahest-0.8b-rp`

The direct conversion repos stay size-explicit, and the tuned repos also stay size-explicit so the portfolio stays operationally clear.

### Training shape

Best practical path:

1. start from the Heretic checkpoint
2. fine-tune with a roleplay/chat dataset
3. merge the adapter back to a normal Hugging Face checkpoint
4. convert the merged checkpoint to ONNX

### Why this works

The ONNX target should be the **merged final HF checkpoint**, not the training adapter artifact.

That keeps the deployment story consistent:

- train with Unsloth or equivalent
- merge to standard HF model
- export to ONNX
- run in browser/WebGPU

## Wave 3: Qwen Alkahest Lane

### Qwen 3.5 4B Heretic

- Source: `tvall43/Qwen3.5-4B-heretic`
- Goal: add a non-Gemma multimodal browser family to the Alkahest portfolio
- Public name: `alkahest-ai/alkahest-4b`
- Status: Qwen3.5 export scaffold is now present, but it is less proven than the Gemma 4 path

### Qwen 3.5 2B Heretic

- Source: `tvall43/Qwen3.5-2B-heretic-v3b`
- Goal: smaller Alkahest desktop/browser lane with better consumer feasibility than 4B
- Public name: `alkahest-ai/alkahest-2b`

### Qwen 3.5 0.8B Heretic

- Source: `tvall43/Qwen3.5-0.8B-heretic-v3`
- Goal: lowest-cost Alkahest browser lane and best default candidate for broad free chat
- Public name: `alkahest-ai/alkahest-0.8b`

## Roleplay Dataset Plan

Start with a high-signal text dataset first.

Good first dataset characteristics:

- multi-turn chat
- clear persona or scene control
- strong style and boundary control
- plenty of assistant-side long-form replies
- minimal formatting noise

Use the existing example as the shape reference:

- `/Users/area/heretic/examples/roleplay-dataset.example.jsonl`

The actual dataset decision is documented in:

- `/Users/area/heretic/docs/roleplay-training-data.md`

## Multimodal Guidance

### Gemma 4

Gemma 4 is the better multimodal lane for this repo right now.

If you fine-tune on text roleplay data:

- you still keep the multimodal architecture
- but you must regression-test image/audio behavior afterward

### Qwen

Qwen is useful for portfolio variance and now has matching exporter scaffolds in this repo.

Important distinction:

- `rally` lanes are currently shipped as `text + image`
- `alkahest` Qwen lanes are currently shipped as `text + image`

Audio remains outside the working browser export path right now, so the two public families are aligned on modality today.

## Recommended H200 Work Order

For the first paid 24-hour GPU TEE window:

1. convert Gemma 4 E2B Heretic
2. convert Gemma 4 E4B Heretic
3. convert Qwen3.5 4B Heretic
4. convert Qwen3.5 2B Heretic
5. convert Qwen3.5 0.8B Heretic
6. begin roleplay fine-tune on E2B
7. queue E4B roleplay fine-tune
8. queue Qwen3.5 2B and 0.8B roleplay fine-tunes before 4B if browser-first deployment is the priority

That ordering minimizes the chance that the whole window disappears into training/debug overhead without any shipped model.

See `/Users/area/heretic/docs/alkahest-model-branding.md` for the branding rule.
