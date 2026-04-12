# Model Execution Matrix

This is the actual model matrix for the first serious Phala H200 run.

## Direct Browser ONNX Repos

- `alkahest/rally-2b` = direct ONNX conversion of `p-e-w/gemma-4-E2B-it-heretic-ara`
- `alkahest/rally-4b` = direct ONNX conversion of `coder3101/gemma-4-E4B-it-heretic`
- `alkahest/sheena-4b` = direct ONNX conversion of `tvall43/Qwen3.5-4B-heretic`

## Roleplay-Tuned Browser ONNX Repos

- `alkahest/rally-2b-rp` = roleplay-tuned E2B Gemma 4
- `alkahest/rally-4b-rp` = roleplay-tuned E4B Gemma 4
- `alkahest/sheena-4b-rp` = roleplay-tuned Qwen3.5 4B

## Short Answer

Yes, the plan is to fine-tune all three base families:

- E2B becomes tuned `rally-2b-rp`
- E4B becomes tuned `rally-4b-rp`
- Qwen3.5 4B becomes tuned `sheena-4b-rp`

And separately, all three base checkpoints also get direct browser-compatible ONNX repos.
