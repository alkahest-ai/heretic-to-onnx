# Browser Free Chat

This is the actual browser chat integration for running a public ONNX LLM entirely on the user’s device.

## What It Uses

- Transformers.js
- WebGPU
- Hugging Face ONNX model repos

Current default model:

- `onnx-community/Qwen3.5-0.8B-ONNX`

The browser app lives in:

- `browser-chat/index.html`
- `browser-chat/app.js`
- `browser-chat/styles.css`
- `examples/browser-loader.mjs`

Current scope:

- the sample browser runtime now supports both Gemma and Qwen presets
- the browser UI now supports text chat plus a single image input for Gemma and Qwen presets
- the browser UI now supports a single audio input for Rally / Gemma presets

## What It Does

This app gives you:

- a real multi-turn chat UI
- model loading with progress updates
- streaming token output
- a model picker for public Gemma and Qwen ONNX repos plus the planned Alkahest Rally and Sheena repos
- image upload for multimodal prompts
- audio upload for Rally / Gemma multimodal prompts
- a browser-cache clear action so users can remove downloaded model files from the web UI and unload the active model
- browser-only inference with no server-side inference bill
- size and modality notes for each preset

## Important Constraint

This only works cleanly with:

- public Hugging Face model repos
- or self-hosted public model files

Do not plan around direct browser loading from private Hugging Face repos. That would require shipping a client-side token, which defeats the point.

So for user-facing free chat:

1. publish the ONNX repo publicly
2. or self-host the ONNX files on your own CDN/domain

## Local Run

From the repo root:

```bash
bash scripts/serve_browser_chat.sh
```

Then open:

- `http://localhost:4173/browser-chat/`

You can override the model from the URL:

- `http://localhost:4173/browser-chat/?model=onnx-community/Qwen3.5-0.8B-ONNX`
- `http://localhost:4173/browser-chat/?model=onnx-community/Qwen3.5-2B-ONNX`
- `http://localhost:4173/browser-chat/?model=alkahest-ai/rally-2b`
- `http://localhost:4173/browser-chat/?model=alkahest-ai/alkahest-0.8b`

## Recommended Deployment Shape

For the first real deployment:

1. ship the static app
2. point the default at `onnx-community/Qwen3.5-0.8B-ONNX` or `alkahest-ai/alkahest-0.8b`
3. keep `alkahest-ai/alkahest-2b` as the next desktop-quality step up
4. offer `alkahest-ai/rally-2b` and `alkahest-ai/alkahest-4b` as heavier desktop tiers
5. keep `alkahest-ai/rally-4b` as an explicit high-memory option, not the default

If you want a broader consumer browser tier, prioritize:

- `alkahest-ai/alkahest-0.8b`
- `alkahest-ai/alkahest-2b`

Those are better default browser-chat candidates than the larger Gemma and Qwen 4B lanes.

## Device Guidance

This is desktop-first.

Use this messaging:

- “Runs in-browser with WebGPU. Best experience on desktop-class devices.”

Do not promise:

- “works on phones”

The model packages remain multimodal at the architecture level after text-only fine-tuning, but the current shipped browser exports are only validated for text + image. Use image prompts to regression-test Gemma and Qwen; do not treat browser audio as part of the supported Rally path yet.
