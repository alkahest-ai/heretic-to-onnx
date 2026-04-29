# Browser Free Chat

This is the actual browser chat integration for running a public ONNX LLM entirely on the user’s device.

## What It Uses

- Transformers.js
- WebGPU
- Hugging Face ONNX model repos

Current default model:

- `thomasjvu/alkahest-0.8b-heretic-q4-onnx`

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
- a model picker for the current Alkahest Qwen3.5 ONNX smoke targets
- image upload for multimodal prompts
- audio upload for Rally / Gemma multimodal prompts
- a browser-cache clear action so users can remove downloaded model files from the web UI and unload the active model
- browser-only inference with no server-side inference bill
- size and modality notes for each preset
- text-first lazy session loading so plain chat warms lighter decoder sessions before full multimodal encoders are fetched

## Important Constraint

For end-user deployment, this only works cleanly with:

- public Hugging Face model repos
- or self-hosted public model files

Local/private smoke can use the browser-chat HF token field or the local `/__hf_token` helper, but do not ship a client-side token in production. Private repo smoke is for validation only.

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

- `http://localhost:4173/browser-chat/?model=onnx-community/Qwen3.5-0.8B-ONNX-OPT`
- `http://localhost:4173/browser-chat/?model=thomasjvu/alkahest-0.8b-heretic-q4-onnx`
- `http://localhost:4173/browser-chat/?model=thomasjvu/alkahest-0.8b-heretic-q4-onnx-rp`
- `http://localhost:4173/browser-chat/?model=thomasjvu/alkahest-2b-heretic-q4-onnx`

## Recommended Deployment Shape

For the first real deployment:

1. ship the static app
2. point the default at the best validated public Alkahest q4 ONNX repo
3. keep the 0.8B Heretic repo as the stable fallback while SFT quality is still moving
4. expose the 2B Heretic q4 repo only after a cold browser load and first-generation smoke pass
5. move the owner namespace constant when repo ownership moves from the personal account to the org

For the current browser tester, prioritize:

- `onnx-community/Qwen3.5-0.8B-ONNX-OPT`
- `thomasjvu/alkahest-0.8b-heretic-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-q4-onnx-rp`
- `thomasjvu/alkahest-2b-heretic-q4-onnx`

Older q4f16 exports and rejected SFT experiments are intentionally hidden from the picker to avoid confusing smoke results.

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
