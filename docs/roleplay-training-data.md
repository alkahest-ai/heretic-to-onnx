# Roleplay Training Data

This document defines the roleplay dataset strategy for the Alkahest model line.

Date context: April 11, 2026.

## Short Answer

We are **not** using a single off-the-shelf public dataset as the canonical Alkahest roleplay corpus.

The current plan is:

1. ship the direct ONNX conversions first
2. build an **Alkahest-owned original dataset**
3. keep Hugging Face public datasets out of the foundation
4. use a spreadsheet-first review workflow before any row reaches training

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

- `roleplay_v2` as the active internal dataset
- made of **original characters, scenarios, voice patterns, and interaction formats**
- structured for chat-style SFT
- promoted from review tables into approved JSONL before training

This is the dataset that should shape:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`
- `alkahest/sheena-2b-rp`
- `alkahest/sheena-0.8b-rp`
- later tuned portfolio variants

### Public data

Public HF datasets are now optional reference material only.

They are not the foundation of the branded line.

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

Build `roleplay_v2` from original Alkahest-authored material only.

Create it with:

- original companion personas
- original scenario prompts
- flirt / intimacy / comfort / praise / slow-burn lanes
- desired reply style, pacing, and boundaries
- examples of what “good Rally / Sheena behavior” actually looks like
- TSV review tables that let humans rewrite and approve rows before promotion

## Format

Use spreadsheet review tables as the editable source and compile them into multi-turn chat JSONL before training.

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
- `alkahest/sheena-2b`
- `alkahest/sheena-0.8b`

### Phase 2

Build `roleplay_v2` and tune:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`
- `alkahest/sheena-2b-rp`
- `alkahest/sheena-0.8b-rp`

## Sources

- Unsloth fine-tuning guide: [Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- NPC-Dialogue_v2: [Hugging Face](https://huggingface.co/datasets/chimbiwide/NPC-Dialogue_v2)
- NPC-Quest-Dialogue: [Hugging Face](https://huggingface.co/datasets/chimbiwide/NPC-Quest-Dialogue)
- RP-Opus: [Hugging Face](https://huggingface.co/datasets/taozi555/rp-opus)
