# Model Execution Matrix

This is the actual model matrix for the first serious Phala H200 run.

## Direct Browser ONNX Repos

- `alkahest-ai/rally-2b` = direct ONNX conversion of `p-e-w/gemma-4-E2B-it-heretic-ara`
- `alkahest-ai/rally-4b` = direct ONNX conversion of `coder3101/gemma-4-E4B-it-heretic`
- `alkahest-ai/alkahest-4b` = direct ONNX conversion of `tvall43/Qwen3.5-4B-heretic`
- `alkahest-ai/alkahest-2b` = direct ONNX conversion of `tvall43/Qwen3.5-2B-heretic-v3b`
- `alkahest-ai/alkahest-0.8b` = direct ONNX conversion of `tvall43/Qwen3.5-0.8B-heretic-v3`

## Roleplay-Tuned Browser ONNX Repos

- `alkahest-ai/rally-2b-rp` = roleplay-tuned E2B Gemma 4
- `alkahest-ai/rally-4b-rp` = roleplay-tuned E4B Gemma 4
- `alkahest-ai/alkahest-4b-rp` = roleplay-tuned Qwen3.5 4B
- `alkahest-ai/alkahest-2b-rp` = roleplay-tuned Qwen3.5 2B
- `alkahest-ai/alkahest-0.8b-rp` = roleplay-tuned Qwen3.5 0.8B

## Short Answer

Yes, the plan is to fine-tune all five base families:

- E2B becomes tuned `rally-2b-rp`
- E4B becomes tuned `rally-4b-rp`
- Qwen3.5 4B becomes tuned `alkahest-4b-rp`
- Qwen3.5 2B becomes tuned `alkahest-2b-rp`
- Qwen3.5 0.8B becomes tuned `alkahest-0.8b-rp`

And separately, all five base checkpoints also get direct browser-compatible ONNX repos.
