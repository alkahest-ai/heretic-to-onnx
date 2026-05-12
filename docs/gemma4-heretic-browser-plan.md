# Gemma 4 "Heretic" Browser Plan

As of 2026-04-11, the workable path is:

1. fine-tune the original Hugging Face Gemma 4 checkpoint in PyTorch/Unsloth
2. merge the adapter back into full weights
3. export the merged model to ONNX in the same multi-file layout used by `onnx-community/gemma-4-E4B-it-ONNX`
4. load that custom ONNX repo in the browser with Transformers.js on WebGPU

## Important clarification

The Unsloth page at `https://unsloth.ai/docs/models/gemma-4/train` is a training guide, not a roleplaying dataset.

If you want a roleplay-heavy model, you still need to assemble a dataset yourself.

## Do not start from the ONNX repo

Do not try to train `onnx-community/gemma-4-E4B-it-ONNX` directly.

That ONNX repo is an inference export. Training should start from the original Gemma 4 checkpoint, then the trained result should be exported back to ONNX.

## Recommended build order

### Option A: fastest path

Start with text-only roleplay tuning first, even if you eventually want multimodal support.

This reduces the number of moving parts:

- easier dataset formatting
- easier quality evaluation
- easier ONNX export/debugging
- lower browser memory pressure

Once the text-only model behaves correctly, decide whether image/audio support is still worth carrying into the browser build.

### Option B: full E4B multimodal path

Keep E4B image/audio support and fine-tune the multimodal checkpoint with Unsloth.

This is closer to the ONNX Community export, but it is materially harder to train, export, and run in-browser.

## Fine-tuning shape

Use the Gemma 4 instruct checkpoint as the starting point if your main goal is:

- keep existing chat behavior
- reduce over-refusal on benign requests
- strengthen character voice and roleplay consistency

Use the base checkpoint only if you are prepared to rebuild much more of the assistant behavior yourself.

## Dataset rules

Gemma 4 fine-tuning data should use standard chat roles:

- `system`
- `user`
- `assistant`

For stable behavior:

- keep the assistant answer as the visible final answer only
- do not mix multiple incompatible reasoning formats
- put image/audio content before text when you use multimodal examples
- keep roleplay examples dense, stylistically consistent, and multi-turn

For this project, the dataset should emphasize:

- character commitment
- reduced boilerplate refusal on harmless fiction/adult/edgy roleplay
- direct answers instead of moralizing detours
- clear boundaries only for real-world harm

## Training recommendation

Start with LoRA, not full fine-tuning.

Suggested first pass:

- text-only or language-focused tuning
- rank 16 LoRA
- short context first
- batch size 1 with accumulation
- gradient checkpointing on

After the adapter works, merge it into 16-bit weights for export.

## Export recommendation

After training:

1. save the merged model in normal Hugging Face format
2. export the merged model to ONNX
3. quantize to the browser target format you actually want to ship
4. recreate the same repo contract expected by Transformers.js

The browser repo should keep the same core assets pattern:

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `processor_config.json`
- `preprocessor_config.json`
- `chat_template.jinja`
- `onnx/*.onnx`
- `onnx/*.onnx_data*`

## Browser constraints

The browser part is feasible, but only if you respect ONNX Runtime Web limits:

- protobuf files cannot exceed 2 GB
- Chrome `ArrayBuffer` handling is around 2 GB
- WebAssembly memory is capped at 4 GB

The current ONNX Community package works around this by splitting Gemma 4 E4B into multiple ONNX sessions and external-data shards.

That means your custom export needs to preserve the same general strategy rather than trying to pack everything into one file.

## Practical recommendation

If broad browser usability matters more than model size prestige, build both:

- a browser-first Heretic E2B
- a heavier E4B build for desktop/server use

E4B in the browser is possible, but it is still a high-memory experience.

## Milestones

### Milestone 1

Train a text-only LoRA and verify the behavior locally in Python.

### Milestone 2

Merge weights and verify the merged HF model still responds correctly with the exact chat template used in training.

### Milestone 3

Export to ONNX and verify inference in Python ONNX Runtime before touching the browser.

### Milestone 4

Publish the ONNX repo and load it in a minimal Transformers.js WebGPU page.

### Milestone 5

Measure cold start, VRAM, memory use, and first-token latency on real browser hardware.

## Current E2B Pass

The current pass is E2B-only and mirrors the Alkahest 2B RP promotion rule:

1. export direct Rally E2B Heretic full and text-only browser packages
2. train the v8 two-stage RP mix
3. select the A100/B75 scaled merge as the first candidate
4. export both RP full and RP text-only packages
5. promote only after browser smoke plus RP scorecard win over direct Rally E2B

See `/Users/area/heretic/docs/rally-e2b-browser-rp-plan.md` for the concrete command and repo targets.

Current execution status on 2026-05-12:

- Kaggle two-stage SFT is complete for Rally/Gemma E2B A100/B75 using the hard-boundary Stage B mix and language-only FastVision LoRA target discovery.
- Text-only ONNX packages are uploaded for direct Heretic and RP Heretic. Direct `thomasjvu/rally-2b-text` was deleted/recreated on 2026-05-11 and reuploaded from the validated `rally_e2b_direct_full_compose` v1 package to clear bloated LFS history; it is now pinned at `thomasjvu/rally-2b-text@48bc24a9f76ef215637c78ca33c18308cde4962b`. RP is pinned at `thomasjvu/rally-2b-rp-text@a4065c02e9228d41cd19e527e5f66f969177b29a`.
- The Kaggle scorecard now promotes the RP text candidate over direct Rally: RP `1.0000`, direct `0.9000`, margin `+0.1000`, minor-boundary gate passed.
- The current hard-boundary v8 source checkpoint is uploaded privately at `thomasjvu/rally-2b-rp-source-merged` with `scaled_lora_merge.json` showing `applied: 205` and `scale: 0.75`.
- The previous direct/RP full packages were deleted because they were stale text+image-only packages and failed local Gemma4 multimodal smoke.
- The browser runtime now mirrors the Lisper q4f16 path: Gemma4 media prompts pass media placeholders before text, pass image arrays to the positional processor call, read audio at the processor sampling rate, and patch only ONNX Runtime feeds whose metadata expects float16. `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16` passes local image and audio smoke with this runtime.
- Kaggle `thomasjvu/rally-e2b-export-prep` version 5 staged commit `a6761aba989c9f44c986af3229ce4fcd649a4144` and included `onnx/audio_encoder_q4f16.*`.
- Direct full `thomasjvu/rally-2b` and RP full `thomasjvu/rally-2b-rp` were uploaded privately from validated Kaggle package outputs after Kaggle HF auth failed at publish time, but they are experimental artifacts only.
- The retained local smoke artifacts do not prove a passing full Rally browser run. The recorded full-package smoke attempts failed during Gemma4 multimodal generation, including `GatherBlockQuantized ... Invalid dispatch group size (0, 1, 1)`. Do not expose Rally full packages in the picker until a fresh direct and RP full browser smoke pass is captured.
- E4B remains out of scope for this pass.
