# Alkahest Definitive Browser Model List

Date: 2026-04-29

## Final Alkahest Targets

These are the definitive Qwen3.5 browser model slots for the Alkahest line:

| Slot | Final repo target | Current source/status |
| --- | --- | --- |
| 0.8B Heretic | `thomasjvu/alkahest-0.8b-heretic-q4-onnx` | Exists and browser-smoked. Stable fallback. |
| 0.8B Heretic RP | `thomasjvu/alkahest-0.8b-heretic-q4-onnx-rp` | Exists. Promoted from `thomasjvu/alkahest-0.8b-heretic-rp-sft-two-stage-a100-b100-q4-onnx`; needs one final smoke under the definitive repo name. |
| 2B Heretic | `thomasjvu/alkahest-2b-heretic-q4-onnx` | Exists and browser-smoked. Technically working, but slow on local Mac. |
| 2B Heretic RP | `thomasjvu/alkahest-2b-heretic-q4-onnx-rp` | Not built yet. Train/apply the same A100+B100 two-stage RP method to the 2B Heretic checkpoint, then export q4 ONNX and smoke. |
| 4B Heretic Text | `thomasjvu/alkahest-4b-heretic-q4-onnx-text` | Planned text-only q4 package from `tvall43/Qwen3.5-4B-heretic` using `onnx-community/Qwen3.5-4B-ONNX-OPT`. |
| 4B Heretic RP Text | `thomasjvu/alkahest-4b-heretic-q4-onnx-rp-text` | Planned text-only q4 package after 4B two-stage RP SFT produces `thomasjvu/alkahest-4b-heretic-rp-merged`. |

Text-only variants are separate speed targets for chat-only browser use:

| Slot | Final repo target | Current source/status |
| --- | --- | --- |
| 2B Heretic Text | `thomasjvu/alkahest-2b-heretic-q4-onnx-text` | Built and validated on Kaggle at ~1.44 GB; upload pending HF secret availability in the Kaggle job. |
| 2B Heretic RP Text | `thomasjvu/alkahest-2b-heretic-q4-onnx-rp-text` | Build after the 2B RP merged checkpoint exists. |
| 4B Heretic Text | `thomasjvu/alkahest-4b-heretic-q4-onnx-text` | Build from `tvall43/Qwen3.5-4B-heretic`; expected to be materially heavier than 2B and desktop-only for browser use. |
| 4B Heretic RP Text | `thomasjvu/alkahest-4b-heretic-q4-onnx-rp-text` | Build after the 4B RP merged checkpoint exists. |

## Quantization Status

The browser packages are q4 for the text path:

- `onnx/embed_tokens_q4.onnx`
- `onnx/decoder_model_merged_q4.onnx`

Current multimodal packages still include:

- `onnx/vision_encoder_fp16.onnx`

The train/export source checkpoints are normal Hugging Face checkpoints. The browser ONNX packages are the quantized artifacts.

`fp16` means 16-bit floating point weights. It is roughly half the size of fp32, but still much larger than q4/int4-style quantized weights. For our Qwen browser packages, q4 is used for the text model and fp16 is used for the vision encoder because that matches the known-working upstream-style Qwen WebGPU package contract.

## 2B Speed Plan

The 2B q4 package is already quantized, so the next speed wins are packaging/runtime changes, not simply "more quantization":

- Build a text-only 2B q4 package that omits the fp16 vision encoder when the target is chat-only.
- Use `thomasjvu/alkahest-2b-heretic-q4-onnx-text` as the first text-only test repo.
- Use `thomasjvu/alkahest-2b-heretic-q4-onnx-rp-text` for the RP text-only counterpart after the 2B RP checkpoint is built.
- Expected text-only package size is about 1.45-1.55 GB, based on the current 2B package shards: about 1.12 GB decoder data plus about 311 MB embedding data plus small ONNX/config/tokenizer files.
- Current full 2B browser package is about 2.2 GB because it also carries a roughly 637 MB fp16 vision shard.
- Keep max tokens low by default for 2B browser smoke, because generation latency dominates after load.
- Keep browser cache warm and avoid clearing cache during normal testing.
- Treat 2B as desktop-only; use 0.8B for fast iteration and weaker client hardware.

## 4B Text-Only Plan

4B should be treated as a desktop/browser stress target, not the default consumer model. The immediate 4B scope is text-only:

- Source Heretic checkpoint: `tvall43/Qwen3.5-4B-heretic`.
- Base model metadata: `Qwen/Qwen3.5-4B`.
- ONNX template: `onnx-community/Qwen3.5-4B-ONNX-OPT`.
- Direct text target: `thomasjvu/alkahest-4b-heretic-q4-onnx-text`.
- RP source target: `thomasjvu/alkahest-4b-heretic-rp-merged`.
- RP text target: `thomasjvu/alkahest-4b-heretic-q4-onnx-rp-text`.
- First 4B SFT pass uses the same two-stage recipe, but shorter/lower-impact defaults on T4: 512 sequence length, 60 Stage A steps, and 50 Stage B steps. Raise only after the first smoke shows quality is worth the extra GPU time.

Vision q4 is possible and now has an explicit exporter switch:

- `--include-vision --vision-dtype fp16` keeps the known-good current full package contract.
- `--include-vision --vision-dtype q4` copies upstream `vision_encoder_q4` artifacts for a smaller experimental full package.

The 2B upstream q4 vision shard is roughly 218 MB versus roughly 668 MB for fp16, so a q4-vision full package could save about 450 MB. Treat q4 vision as experimental until it passes browser cold load, first generation, and image prompt smoke; do not block text-only chat packages on it.

## Promotion Order

1. Final-smoke existing `0.8B Heretic`.
2. Final-smoke `thomasjvu/alkahest-0.8b-heretic-q4-onnx-rp` under its definitive repo name.
3. Build and smoke `thomasjvu/alkahest-2b-heretic-q4-onnx-text`; if it passes, prefer it as the practical 2B text-chat target.
4. Final-smoke existing multimodal `2B Heretic` only if image support is required.
5. Train 2B RP using the same two-stage A100+B100 method, export q4 ONNX, upload to `2B Heretic RP`, then browser-smoke.
6. Build and smoke `thomasjvu/alkahest-2b-heretic-q4-onnx-rp-text` from the 2B RP merged checkpoint.
7. Build and smoke `thomasjvu/alkahest-4b-heretic-q4-onnx-text` as the direct 4B chat-only target.
8. Train 4B RP, export `thomasjvu/alkahest-4b-heretic-q4-onnx-rp-text`, then smoke only if the direct 4B text target is usable on the browser machine.

## Kaggle Jobs

- `alkahestai/alkahest-qwen-text-export`: builds `thomasjvu/alkahest-2b-heretic-q4-onnx-text` from `thomasjvu/alkahest-2b-heretic-merged`.
- `alkahestai/alkahest-2b-two-stage-sft-t4`: trains 2B Stage A + Stage B and uploads `thomasjvu/alkahest-2b-heretic-rp-merged`.
- `alkahestai/alkahest-2b-rp-qwen-export`: builds both `thomasjvu/alkahest-2b-heretic-q4-onnx-rp` and `thomasjvu/alkahest-2b-heretic-q4-onnx-rp-text` from the 2B RP merged checkpoint.
- `alkahestai/alkahest-4b-qwen-text-export`: builds `thomasjvu/alkahest-4b-heretic-q4-onnx-text`.
- `alkahestai/alkahest-4b-two-stage-sft-t4`: trains 4B Stage A + Stage B and uploads `thomasjvu/alkahest-4b-heretic-rp-merged` when the HF token is valid.
- `alkahestai/alkahest-4b-rp-qwen-text-export`: builds `thomasjvu/alkahest-4b-heretic-q4-onnx-rp-text`.

## Cleanup Rule

After the four final repos exist and pass browser smoke, delete or archive the old failed/legacy repos from the cleanup plan. Do not delete source merged checkpoints or Kaggle reports until the final repos are reproducible from documented inputs.
