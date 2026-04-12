# Browser Compatibility

This document covers the realistic browser target for the ONNX/WebGPU packages produced by this repo.

Date context: April 11, 2026.

## Short Answer

### Will these browser models work on phones?

Sometimes.

### Will they work in Safari on iPhone?

Technically, **Safari now has WebGPU support** on iPhone and iPad in current Apple releases.

But the more important answer is:

- **small models**: maybe yes
- **Gemma 4 E2B / E4B multimodal browser packages**: do **not** assume yes for phones

The browser support question and the memory/performance question are separate.

## Current Browser Support

Apple’s current official WebKit material says:

- Safari `26.0` adds support for `WebGPU`
- it ships on `macOS`, `iOS`, `iPadOS`, and `visionOS`
- frameworks including `Transformers.js` and `ONNX Runtime` work in Safari `26.0`

At the same time, the current Hugging Face Transformers.js docs and ONNX Runtime Web docs still contain older compatibility wording:

- Transformers.js warns Safari may require a `WebGPU` feature flag
- ONNX Runtime Web still mentions `Safari Technology Preview`

Inference:

- Apple’s platform support has moved forward
- some framework docs lag behind current Safari shipping status

## Practical Limiter: Model Size

ONNX Runtime Web’s official large-model docs say:

- browser `ArrayBuffer` handling is around `2GB` in Chrome as an example
- protobuf files have a `2GB` limit, requiring external data for larger models
- WebAssembly memory has a `4GB` limit

Those constraints are why these Gemma browser packages are split into multiple ONNX sessions with external data.

Even if Safari supports WebGPU, phone-class devices still have tighter real-world limits:

- less available RAM
- tighter thermal budget
- lower sustained GPU throughput
- higher chance of browser reloads or tab eviction

## Recommendation By Device Class

### Desktop Safari on Apple Silicon

This is a realistic target for the Gemma browser packages.

### iPad Safari

Possible, but still test-driven. More realistic than iPhone for larger models.

### iPhone Safari

Do not make Gemma 4 E2B or E4B multimodal ONNX your primary promise here.

If it works on high-end devices, treat that as a bonus, not the baseline contract.

## Product Guidance

For `Alkahest`:

- market the browser ONNX line as **desktop-first**
- treat mobile Safari as an **experimental / best-effort lane**
- if mobile matters, plan a smaller model lane specifically for it

That smaller mobile lane could later be:

- a smaller Gemma variant
- a smaller Qwen variant
- a distilled text-first model

## Safe Messaging

Good message:

- “Runs in-browser with WebGPU. Best experience on desktop-class devices.”

Risky message:

- “Runs on phones in Safari.”

The second statement is too broad for Gemma 4-sized browser packages.

## Sources

- WebKit Safari 26 WebGPU announcement: [WebKit](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/)
- Apple WebGPU session: [WWDC25](https://developer.apple.com/videos/play/wwdc2025/236/)
- Transformers.js WebGPU guide: [Hugging Face](https://huggingface.co/docs/transformers.js/guides/webgpu)
- ONNX Runtime WebGPU guide: [ONNX Runtime](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- ONNX Runtime large-model guide: [ONNX Runtime](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
