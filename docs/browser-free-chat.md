# Browser Free Chat

This is the actual browser chat integration for running a public ONNX LLM entirely on the user’s device.

## What It Uses

- Transformers.js
- WebGPU
- Hugging Face ONNX model repos

Current default model:

- `thomasjvu/rally-2b-rp`

The browser app lives in:

- `browser-chat/index.html`
- `browser-chat/app.js`
- `browser-chat/styles.css`
- `examples/browser-loader.mjs`

Current scope:

- the sample browser runtime now supports both Gemma 4 and Qwen 3.5 presets
- the browser UI now supports text chat plus a single image input for all shipped presets
- the browser UI now supports a single audio input for Rally / Gemma multimodal presets
- the browser UI now supports a single video input for shipped `v2` presets

## Release Gate

Do not treat a manifest capability, model card, or browser preset label as a real shipped feature until the package has passed:

- packaged ONNX validation with runtime smoke enabled
- a manual browser-chat smoke against the published repo

This matters because metadata-only validation can still miss broken ONNX session graphs.

## What It Does

This app gives you:

- a real multi-turn chat UI
- model loading with progress updates
- a worker-backed runtime so downloads and WebGPU session setup stay off the main UI thread when the browser supports it
- streaming token output
- a model picker for the published public Alkahest and Rally ONNX repos
- image upload for multimodal prompts
- audio upload for Rally / Gemma multimodal prompts
- a browser-cache clear action so users can remove downloaded model files from the web UI and unload the active model
- browser-only inference with no server-side inference bill
- size and modality notes for each preset
- text-first lazy session loading so plain chat warms lighter decoder sessions before full multimodal encoders are fetched

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

- `http://localhost:4173/`

You can override the model from the URL:

- `http://localhost:4173/browser-chat/?model=thomasjvu/alkahest-0.8b`
- `http://localhost:4173/browser-chat/?model=thomasjvu/alkahest-0.8b-v2`
- `http://localhost:4173/browser-chat/?model=thomasjvu/alkahest-2b`
- `http://localhost:4173/browser-chat/?model=thomasjvu/alkahest-2b-v2`
- `http://localhost:4173/browser-chat/?model=thomasjvu/rally-2b`
- `http://localhost:4173/browser-chat/?model=thomasjvu/rally-2b-v2`
- `http://localhost:4173/browser-chat/?model=thomasjvu/rally-2b-rp`
- `http://localhost:4173/browser-chat/?model=thomasjvu/rally-4b`
- `http://localhost:4173/browser-chat/?model=thomasjvu/rally-4b-v2`

## Recommended Deployment Shape

For the first real deployment:

1. ship the static app
2. point the default at `thomasjvu/rally-2b-rp`, `thomasjvu/rally-2b-v2`, or `thomasjvu/alkahest-2b-v2`
3. keep `thomasjvu/rally-2b` and `thomasjvu/alkahest-2b` as lighter direct tiers
4. offer `thomasjvu/rally-4b`, `thomasjvu/rally-4b-v2`, `thomasjvu/alkahest-4b`, and `thomasjvu/alkahest-4b-v2` as heavier desktop tiers
5. move the owner namespace constant when repo ownership moves from the personal account to the org

For the current browser tester, prioritize:

- `thomasjvu/rally-2b-rp`
- `thomasjvu/rally-2b-v2`
- `thomasjvu/alkahest-2b-v2`
- `thomasjvu/rally-2b`
- `thomasjvu/alkahest-2b`

Those are better default browser-chat candidates than the larger 4B lanes.

Plain text chat is now intentionally cheaper than full multimodal use:

- clicking `Load model` warms text sessions first
- sending a text-only prompt without preloading also stays on the text-only path
- attaching image, audio, or video upgrades the same model to its full multimodal session set on demand

The preset modality claims in this doc assume that release gate has passed for the repo being tested or published.

## Device Guidance

This is desktop-first.

Use this messaging:

- “Runs in-browser with WebGPU. Best experience on desktop-class devices.”

Do not promise:

- “works on phones”

The model packages remain multimodal at the architecture level after text-only fine-tuning. Use image, audio, and video prompts against the shipped preset set according to each preset’s advertised inputs, and regression-test direct lanes as text + image first.
