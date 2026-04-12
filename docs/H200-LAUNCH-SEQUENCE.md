# H200 Launch Sequence

This is the single end-to-end sequence for getting every target model trained, converted to ONNX, uploaded to Hugging Face, and ready for browser/WebGPU consumption.

## Goal

At the end of this run, you want these ten model repos on Hugging Face:

Direct ONNX repos:

- `alkahest/rally-2b`
- `alkahest/rally-4b`
- `alkahest/sheena-4b`
- `alkahest/sheena-2b`
- `alkahest/sheena-0.8b`

Roleplay-tuned ONNX repos:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`
- `alkahest/sheena-2b-rp`
- `alkahest/sheena-0.8b-rp`

## Reality Check

You are ready to run this.

The one real caveat is still the same:

- Gemma is the more proven export path in this repo.
- Qwen is scaffolded end to end, but it has not been live-validated on a real GPU from this workspace yet.

So the safest sequence is:

1. direct Gemma
2. tuned Gemma
3. direct Qwen
4. tuned Qwen

## Phase 1: Local Preflight

Before spending money on the H200 window, confirm these locally:

1. The repo you want to run is this one:
   - `git@github.com:thomasjvu/gemma-4-E2B-it-heretic-ara-ONNX.git`
2. The main docs exist:
   - `docs/PRELAUNCH.md`
   - `docs/phala-gpu-tee-oneclick.md`
   - `docs/model-execution-matrix.md`
   - `docs/browser-free-chat.md`
   - `docs/H200-LAUNCH-SEQUENCE.md`
3. Your roleplay data has been reviewed enough for a first pass:
   - approved `roleplay_v2` rows exist in `data/roleplay_v2/approved_jsonl/`
4. You know the target Hugging Face repos you want to end with.

If you want a quick data spot-check before launch:

```bash
python3 scripts/sample_roleplay_rows.py --input data/roleplay_v2/approved_jsonl --count 5
python3 scripts/sample_roleplay_rows.py --input data/roleplay_v2/approved_jsonl --lane praise --count 5
```

If you manually edit a review table before launch, compile and rebuild locally:

```bash
python3 scripts/review_table_to_jsonl.py \
  --input data/roleplay_v2/review_table/<batch>.tsv \
  --output data/roleplay_v2/approved_jsonl/<batch>.jsonl
python3 scripts/build_roleplay_training_corpus.py
python3 scripts/prepare_roleplay_dataset.py --input data/roleplay_v2/corpus.jsonl
```

## Phase 2: Hugging Face Preconditions

Before opening the H200 runtime, make sure the account behind `HF_TOKEN` has:

Read access to source/base models:

- `p-e-w/gemma-4-E2B-it-heretic-ara`
- `coder3101/gemma-4-E4B-it-heretic`
- `tvall43/Qwen3.5-4B-heretic`
- `tvall43/Qwen3.5-2B-heretic-v3b`
- `tvall43/Qwen3.5-0.8B-heretic-v3`
- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-0.8B`

Write access to target repos:

- `alkahest/rally-2b`
- `alkahest/rally-4b`
- `alkahest/sheena-4b`
- `alkahest/sheena-2b`
- `alkahest/sheena-0.8b`
- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`
- `alkahest/sheena-2b-rp`
- `alkahest/sheena-0.8b-rp`

Notes:

- The uploader can create missing target repos automatically with `--exist-ok`.
- The token still needs permission to create/upload model repos.
- Accept any gated model terms in Hugging Face before the run.

## Phase 3: Launch The Phala Runtime

Use the Phala UI, not the CVM path.

Choose:

1. `GPU TEE`
2. `H200`
3. `1 GPU`
4. `Jupyter Notebook (PyTorch)`
5. `On-Demand`

Once the instance is up, open the terminal inside Jupyter.

## Phase 4: Bootstrap The Workspace On H200

Run this in the Jupyter terminal:

```bash
git clone git@github.com:thomasjvu/gemma-4-E2B-it-heretic-ara-ONNX.git heretic
cd heretic
export HF_TOKEN=...
bash scripts/phala_gpu_tee_bootstrap.sh
```

What this does:

- upgrades pip
- installs the repo package
- installs Unsloth, TRL, datasets, ONNX, ONNX Runtime GPU, HF CLI, and Transformers
- runs converter bootstrap/preflight

## Phase 5: Final Dataset Review On The GPU Box

Before training, make sure you have an approved `roleplay_v2` corpus:

```bash
python3 scripts/sample_roleplay_rows.py --input data/roleplay_v2/approved_jsonl --count 5
python3 scripts/sample_roleplay_rows.py --input data/roleplay_v2/approved_jsonl --lane praise --count 5
```

If you need a new raw review batch:

```bash
bash scripts/phala_gpu_tee_oneclick.sh dataset-batch
```

After review, compile it and rebuild the corpus:

```bash
python3 scripts/review_table_to_jsonl.py \
  --input data/roleplay_v2/review_table/batch-0001.tsv \
  --output data/roleplay_v2/approved_jsonl/batch-0001.jsonl
python3 scripts/build_roleplay_training_corpus.py
python3 scripts/prepare_roleplay_dataset.py --input data/roleplay_v2/corpus.jsonl
```

## Phase 6: Run The Direct ONNX Conversions

This is the lowest-risk order:

```bash
bash scripts/phala_run_rally_2b_direct.sh
bash scripts/phala_run_rally_4b_direct.sh
bash scripts/phala_run_sheena_4b_direct.sh
bash scripts/phala_run_sheena_2b_direct.sh
bash scripts/phala_run_sheena_0_8b_direct.sh
```

What each direct run does:

1. render a runtime manifest
2. prepare the source model and inherited processor assets
3. export raw ONNX sessions
4. quantize to q4f16
5. assemble the browser package layout
6. upload the final package to Hugging Face

There is no direct-only bundle today. `all-gemma`, `all-qwen`, and `all` all include tuned runs after the approved-corpus compile step.

For the first paid run, the per-model sequence is easier to recover from if one step fails.

## Phase 7: Run The Roleplay Fine-Tunes And Tuned ONNX Exports

After the direct repos are done, run the tuned models:

```bash
bash scripts/phala_run_rally_2b_rp.sh
bash scripts/phala_run_rally_4b_rp.sh
bash scripts/phala_run_sheena_4b_rp.sh
bash scripts/phala_run_sheena_2b_rp.sh
bash scripts/phala_run_sheena_0_8b_rp.sh
```

What each tuned run does:

1. rebuild dataset if needed
2. fine-tune the base checkpoint with Unsloth
3. save LoRA output
4. merge to a standard HF checkpoint
5. render a tuned manifest that points at the merged local checkpoint
6. export ONNX
7. quantize to q4f16
8. package for Transformers.js / WebGPU
9. upload to the final Hugging Face repo

## Phase 8: If You Want A Single Command Instead

This is the all-in path:

```bash
bash scripts/phala_run_all_models.sh
```

That is fine if you want the shortest operator path.

The tradeoff is that recovery is worse if one model fails halfway through the window.

For the first paid run, the safer choice is still the explicit per-model sequence.

## Phase 9: Verify The Hugging Face Outputs

After upload, check that each repo exists and has the expected ONNX files.

Gemma repos should contain:

- `onnx/audio_encoder_q4f16.onnx`
- `onnx/vision_encoder_q4f16.onnx`
- `onnx/embed_tokens_q4f16.onnx`
- `onnx/decoder_model_merged_q4f16.onnx`

Qwen repos should contain:

- `onnx/vision_encoder_q4f16.onnx`
- `onnx/embed_tokens_q4f16.onnx`
- `onnx/decoder_model_merged_q4f16.onnx`

Also confirm each repo has:

- `config.json`
- `processor_config.json`
- tokenizer files
- `README.md`
- `transformers.js_config` embedded in `config.json`

## Phase 10: Validate In The Browser

Run the local browser chat:

```bash
bash scripts/serve_browser_chat.sh
```

Then open:

```text
http://localhost:4173/browser-chat/
```

Validation order:

1. `onnx-community/Qwen3.5-0.8B-ONNX`
2. `alkahest/sheena-0.8b`
3. `alkahest/sheena-0.8b-rp`
4. `alkahest/rally-2b`
5. `alkahest/rally-2b-rp`

Smoke tests to run:

1. text-only chat
2. image prompt on Sheena and Rally
3. audio prompt on Rally
4. clear browser cache from the UI and reload the model

## Phase 11: Consume The Models In Other Apps

For your other browser/WebGPU apps:

1. use public Hugging Face ONNX repos or self-host the same files publicly
2. load them with Transformers.js
3. pick a default model by device class

Recommended product defaults:

- default browser tier: `alkahest/sheena-0.8b` or `alkahest/sheena-0.8b-rp`
- stronger desktop tier: `alkahest/sheena-2b` or `alkahest/sheena-2b-rp`
- premium desktop-only tier: `alkahest/rally-2b` or `alkahest/rally-2b-rp`
- do not make `rally-4b` the default consumer browser model

## Phase 12: What To Do If Something Fails

If a run fails:

1. stop and note which model failed
2. keep the successful Hugging Face repos
3. rerun only the failed wrapper script
4. if the failure is in Qwen, do not block Gemma on it

If you are short on the 24-hour window, the highest-value completion order is:

1. `rally-2b`
2. `rally-2b-rp`
3. `sheena-0.8b`
4. `sheena-0.8b-rp`
5. `sheena-2b`
6. `sheena-2b-rp`
7. everything else
