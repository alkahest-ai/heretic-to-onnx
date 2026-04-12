# Alkahest Model Branding

This document aligns model naming with the portfolio rules in the second-brain.

## Portfolio Alignment

Your second-brain frames `Alkahest` as:

- the private generative AI substrate
- the routing and inference posture layer
- the shared rails under Phantasy, Rally, and related products

That means the model line should read like **platform substrate**, not like a product story.

## Branding Rule

Use `Alkahest` as the outward-facing family name.

But always keep real provenance in:

- model cards
- internal docs
- release notes
- any compliance or attribution fields required by the source license

So the pattern is:

- public family name: `Alkahest`
- internal or model-card provenance: original upstream base + fine-tune source

## Recommended Naming Pattern

Use concise names that encode family, size, and tune lane.

Examples:

- `Alkahest-G4-E2B-Heretic-ONNX`
- `Alkahest-G4-E4B-Heretic-ONNX`
- `Alkahest-G4-E2B-Roleplay-v1`
- `Alkahest-G4-E4B-Roleplay-v1`
- `Alkahest-Q35-4B-Heretic-v1`

If you want a softer external name, you can shorten public labels while keeping the technical repo name explicit.

## Current Naming Decision

Use the companion names directly.

Current line:

- `alkahest/rally-2b` = Gemma 4 E2B Heretic ONNX
- `alkahest/rally-4b` = Gemma 4 E4B Heretic ONNX
- `alkahest/sheena-4b` = Qwen 3.5 4B Heretic ONNX
- `alkahest/sheena-2b` = Qwen 3.5 2B Heretic ONNX
- `alkahest/sheena-0.8b` = Qwen 3.5 0.8B Heretic ONNX

Future tuned line:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`
- `alkahest/sheena-2b-rp`
- `alkahest/sheena-0.8b-rp`

This is cleaner than repeating `alkahest` in every slug.

## Recommended Two-Layer Naming System

Use **two separate naming modes**:

- descriptive names for direct upstream conversions
- Alkahest family names for your own tuned models

### 1. Direct conversion repos

These should stay size-explicit so operators know what they are:

- `alkahest/rally-2b`
- `alkahest/rally-4b`
- `alkahest/sheena-4b`

### 2. Alkahest-tuned family repos

These are the models you want to read as your own portfolio line.

The goal is:

- same family language across Gemma and Qwen
- no base-family name in the repo slug
- enough structure that operators still know which size/tier they are dealing with

## Recommended Naming Decision

The cleanest compromise is:

- use the companion names directly
- use size suffixes for the raw/direct variants
- use explicit `-rp` suffixes for the tuned variants so operators can distinguish them instantly

Recommended pattern:

- direct conversion:
  - `alkahest/rally-2b`
  - `alkahest/rally-4b`
  - `alkahest/sheena-4b`
  - `alkahest/sheena-2b`
  - `alkahest/sheena-0.8b`
- tuned family:
  - `alkahest/rally-2b-rp`
  - `alkahest/rally-4b-rp`
  - `alkahest/sheena-4b-rp`
  - `alkahest/sheena-2b-rp`
  - `alkahest/sheena-0.8b-rp`

## Provenance Pattern

Each model card should include a section like:

- `Brand family`: Alkahest
- `Derived from`: original upstream repo id
- `Tuning lineage`: Heretic / roleplay tune / merge details
- `Export lineage`: ONNX q4f16 browser package if applicable

This gives you branding without pretending the base architecture came from nowhere.

## What To Avoid

- do not hide provenance completely
- do not imply you authored the original base architecture if you did not
- do not let the repo name become so abstract that operators cannot tell what family it belongs to

## Repo Naming Recommendation

Use a split between brand and technical artifact.

Examples:

- Hugging Face repo: `alkahest/rally-2b`
- model card title: `Rally 2B`
- provenance section:
  - source model: `p-e-w/gemma-4-E2B-it-heretic-ara`
  - base architecture lineage: `google/gemma-4-E2B-it`

## Why This Matches The Second-Brain

Your second-brain consistently treats Alkahest as:

- substrate
- private inference posture
- shared rails

So an Alkahest-branded model family is coherent as long as:

- Phantasy stays the runtime/OS story
- Rally stays the proof-point product story
- Alkahest remains the underlying model/inference layer
