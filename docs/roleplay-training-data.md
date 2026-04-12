# Roleplay Training Data

This document defines the roleplay dataset strategy for the Alkahest model line.

Date context: April 11, 2026.

## Short Answer

We are **not** using a single off-the-shelf public dataset as the canonical Alkahest roleplay corpus.

The current plan is:

1. ship the direct ONNX conversions first
2. build an **Alkahest-owned roleplay dataset**
3. use a small amount of public Apache-licensed roleplay data only as bootstrap material
4. avoid non-commercial or gated datasets as the foundation of the branded line

## Why

You want these models to be:

- private
- open-source-able
- commercially usable later
- branded under Alkahest

That means the dataset story has to be clean.

## Dataset Decision

### Canonical training dataset

The canonical roleplay dataset should be:

- `alkahest/roleplay-v1` as an internal dataset first
- made of **original characters, scenarios, voice patterns, and interaction formats**
- structured for chat-style SFT

This is the dataset that should shape:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`
- later tuned portfolio variants

### Bootstrap public data

For initial experimentation, use only Apache-licensed public roleplay data.

Best current bootstrap candidates:

- `chimbiwide/NPC-Dialogue_v2`
- `chimbiwide/NPC-Quest-Dialogue`

Why these two:

- both are explicitly marked `apache-2.0`
- both are clearly intended for SFT
- both are recent, structured, and easy to ingest

## Datasets We Are Not Using As The Foundation

### `taozi555/rp-opus`

Not the foundation dataset.

Why:

- license is `CC BY-NC 4.0`
- gated access
- non-commercial restriction is a bad fit for the core Alkahest line

### mixed PIPPA-heavy RP corpora

Not the foundation dataset.

Why:

- provenance is noisier
- copyright/IP exposure is higher
- harder to defend as the canonical branded training source

That does not mean they are impossible to experiment with privately, but they should not be the clean public backbone of the line.

## Recommended Data Strategy

Use a two-part dataset.

### Part A: Apache bootstrap set

Start with:

- `chimbiwide/NPC-Dialogue_v2`
- `chimbiwide/NPC-Quest-Dialogue`

But filter them before training:

- remove rows with obvious franchise IP you do not want in the final model voice
- remove malformed or excessively generic exchanges
- normalize to a single chat format

### Part B: Alkahest original set

Create `roleplay-v1` with:

- original companion personas
- original scenario prompts
- flirt / intimacy / comfort / conflict lanes if desired
- desired reply style, pacing, and boundaries
- examples of what “good Rally / Sheena behavior” actually looks like

This original set should become the majority signal over time.

## Recommended Training Split

For the first tuned run:

- `60-80%` Alkahest original data
- `20-40%` Apache bootstrap data

The point of the public data is to help conversational depth and pacing, not to define the identity of the line.

## Format

Use multi-turn chat JSONL matching the structure already scaffolded in:

- `/Users/area/heretic/examples/roleplay-dataset.example.jsonl`

Preferred pattern:

- `system` or instruction seed
- `user` turn
- `assistant` turn
- repeated multi-turn conversations

## Training Sequence

For Gemma 4 or Qwen:

1. start from the Heretic checkpoint
2. fine-tune with Unsloth using the cleaned dataset
3. merge the adapter back into a normal Hugging Face checkpoint
4. run the ONNX conversion pipeline on the merged checkpoint

So yes, the fine-tuned model can still become an ONNX/WebGPU model. The ONNX target is the **merged final HF checkpoint**, not the LoRA adapter alone.

## Immediate Plan

### Phase 1

Ship these direct conversions first:

- `alkahest/rally-2b`
- `alkahest/rally-4b`
- `alkahest/sheena-4b`

### Phase 2

Build `roleplay-v1` and tune:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`

## Sources

- Unsloth fine-tuning guide: [Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- NPC-Dialogue_v2: [Hugging Face](https://huggingface.co/datasets/chimbiwide/NPC-Dialogue_v2)
- NPC-Quest-Dialogue: [Hugging Face](https://huggingface.co/datasets/chimbiwide/NPC-Quest-Dialogue)
- RP-Opus: [Hugging Face](https://huggingface.co/datasets/taozi555/rp-opus)
