# Alkahest Hugging Face Cleanup Plan

Date: 2026-04-29

Deletion status: paused. Do not delete or archive any Hugging Face repos until the definitive 0.8B/2B Heretic and RP targets are published, browser-smoked, and explicitly approved again at action time.

## Active Browser Smoke Targets

Keep these visible in browser-chat until the next quality decision:

- `onnx-community/Qwen3.5-0.8B-ONNX-OPT` - upstream control for loader/runtime regressions.
- `thomasjvu/alkahest-0.8b-heretic-q4-onnx` - stable Heretic-only 0.8B q4 baseline.
- `thomasjvu/alkahest-0.8b-heretic-q4-onnx-rp` - definitive 0.8B RP baseline promoted from the A100+B100 two-stage SFT package.
- `thomasjvu/alkahest-2b-heretic-q4-onnx` - current 2B Heretic q4 browser target.
- `thomasjvu/alkahest-2b-heretic-q4-onnx-text` - built, validated, and uploaded as private; browser smoke pending.
- `thomasjvu/alkahest-2b-heretic-q4-onnx-rp` - planned 2B RP target after 2B two-stage SFT/export.
- `thomasjvu/alkahest-2b-heretic-q4-onnx-rp-text` - planned chat-only 2B RP speed target.
- `thomasjvu/alkahest-4b-heretic-q4-onnx-text` - planned chat-only 4B Heretic target.
- `thomasjvu/alkahest-4b-heretic-q4-onnx-rp-text` - planned chat-only 4B RP target.

## Hidden From Browser Picker

These are no longer default smoke targets because they are legacy, rejected, or superseded:

- `thomasjvu/alkahest-0.8b-v2`
- `thomasjvu/alkahest-2b`
- `thomasjvu/alkahest-2b-v2`
- `thomasjvu/alkahest-4b`
- `thomasjvu/alkahest-4b-v2`
- `thomasjvu/rally-2b`
- `thomasjvu/rally-2b-v2`
- `thomasjvu/rally-2b-rp`
- `thomasjvu/rally-4b`
- `thomasjvu/rally-4b-v2`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v2-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v3-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-scale025-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-scale010-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-scale025-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-safety-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-safety2-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-safety2-scale050-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-two-stage-a100-b50-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-two-stage-a100-b100-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-q8-onnx`

## Final Approval Deletion List

These repos are approved candidates for deletion after the four definitive Alkahest repos exist, pass browser smoke, and have replacement links in docs/model cards:

- `thomasjvu/alkahest-0.8b-v2`
- `thomasjvu/alkahest-2b`
- `thomasjvu/alkahest-2b-v2`
- `thomasjvu/alkahest-4b`
- `thomasjvu/alkahest-4b-v2`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v2-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v3-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-scale025-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v4-scale010-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-scale025-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-safety-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-safety2-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-v5-safety2-scale050-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-two-stage-a100-b50-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-rp-sft-two-stage-a100-b100-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-q8-onnx`

Keep these unless the Gemma/Rally plan is explicitly abandoned:

- `thomasjvu/rally-2b`
- `thomasjvu/rally-2b-v2`
- `thomasjvu/rally-2b-rp`
- `thomasjvu/rally-4b`
- `thomasjvu/rally-4b-v2`

Never delete without a separate action-time confirmation:

- `thomasjvu/alkahest-0.8b-heretic-q4-onnx`
- `thomasjvu/alkahest-0.8b-heretic-q4-onnx-rp`
- `thomasjvu/alkahest-2b-heretic-q4-onnx`
- `thomasjvu/alkahest-2b-heretic-q4-onnx-text`
- `thomasjvu/alkahest-2b-heretic-q4-onnx-rp`
- `thomasjvu/alkahest-2b-heretic-q4-onnx-rp-text`
- `thomasjvu/alkahest-4b-heretic-q4-onnx-text`
- `thomasjvu/alkahest-4b-heretic-q4-onnx-rp-text`
- source merged checkpoints such as `thomasjvu/alkahest-0.8b-heretic-merged` and `thomasjvu/alkahest-2b-heretic-merged`

## Removal Rule

Do not delete Hugging Face repos until:

- a final q4 browser target has passed cold load, first generation, and cache reload;
- the source merged checkpoint or Kaggle output artifact is still available for reproducibility;
- the repo is not referenced by a model card, doc, or test URL we still need.

For now, hide clutter from the local UI first. Delete or archive only after the final 0.8B and 2B winners are confirmed.
