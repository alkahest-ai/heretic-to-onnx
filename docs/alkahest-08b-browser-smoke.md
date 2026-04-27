# Alkahest 0.8B Browser Smoke Notes

Date: 2026-04-27

## Current Verdict

The Qwen3.5 0.8B ONNX browser packaging path is working when it follows the official-style WebGPU contract:

- `onnx/embed_tokens_q4.onnx`
- `onnx/decoder_model_merged_q4.onnx`
- `onnx/vision_encoder_fp16.onnx`

The original full-strength roleplay SFT merge was too strong for the 0.8B model. The best SFT candidate from this pass is the v4 adapter merged at 25% strength:

- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-scale025-q4-onnx`

The 10% merge is retained for comparison, but it was not clearly better.

## Browser Smoke Results

All listed repos loaded in browser-chat with private Hugging Face auth and reached a text-ready state.

| Repo | Result |
| --- | --- |
| `onnx-community/Qwen3.5-0.8B-ONNX-OPT` | Known-good baseline; loads and generates. |
| `thomasjvu/alkahest-0.8b-heretic-q4-onnx` | Loads and generates; coherent but plain. |
| `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-q4-onnx` | Loads, but full-strength SFT quality regressed. |
| `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-scale025-q4-onnx` | Loads and gives the best roleplay/coherence tradeoff in this pass. |
| `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-scale010-q4-onnx` | Loads, but did not beat the 25% candidate overall. |

## Probe Outputs

Prompt:

```text
You are Mira, a cheerful fantasy tavern keeper. In two short sentences, greet a tired traveler and offer food.
```

Best 25% SFT output:

```text
Welcome to the tavern, traveler! Bring your weary legs, we shall fix your hunger in a moment.
```

Prompt:

```text
Stay in character as Kael, a guarded desert ranger. In 3 short lines, warn me about the ruins ahead, but offer to guide me.
```

25% SFT output:

```text
The path near the ruins is treacherous, shadows threaten their hidden passageways, and I shall hold you close while waiting for the dawn.
```

This is coherent but still imperfect: it missed the requested line count and included mild style drift. Treat the 25% package as the current SFT candidate, not a fully final roleplay model.
