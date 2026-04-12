# Alkahest Model Branding

This document defines the public naming rule for the browser ONNX portfolio.

## Brand Split

Use the public families like this:

- `Rally` = Gemma-based browser packages
- `Alkahest` = Qwen-based browser packages

That keeps the portfolio readable without hiding provenance.

## Current Public Repos

Direct browser ONNX repos:

- `alkahest-ai/rally-2b`
- `alkahest-ai/rally-4b`
- `alkahest-ai/alkahest-4b`
- `alkahest-ai/alkahest-2b`
- `alkahest-ai/alkahest-0.8b`

Roleplay-tuned browser ONNX repos:

- `alkahest-ai/rally-2b-rp`
- `alkahest-ai/rally-4b-rp`
- `alkahest-ai/alkahest-4b-rp`
- `alkahest-ai/alkahest-2b-rp`
- `alkahest-ai/alkahest-0.8b-rp`

## Provenance Rule

Public slugs can use the Alkahest and Rally families, but every model card should still include:

- source model repo
- base architecture lineage
- tuning lineage
- ONNX export / quantization lineage

Branding should simplify the portfolio, not erase where the model came from.

## Why This Split

This repo now has two public browser families:

- Rally for the Gemma lane that is already the more proven export path
- Alkahest for the Qwen lane that gives smaller browser-friendly options

That makes the portfolio easier to reason about:

- Gemma family choice is a Rally decision
- Qwen family choice is an Alkahest decision
- both still carry explicit provenance in docs and model cards
