# Heretic to ONNX

Turn Heretic-style Hugging Face checkpoints into browser-oriented ONNX packages for WebGPU.

Supported lanes:

- Gemma 4 conditional generation
- Qwen3.5 conditional generation

Current status:

- `prepare`, `inspect`, `package`, and `validate` are implemented
- `convert` orchestrates inspect -> prepare -> export -> quantize -> package -> validate
- Gemma 4 export and q4f16 quantization support `plan`, `script`, and `execute`
- Qwen3.5 export and q4f16 quantization support `plan`, `script`, and `execute`
- Gemma 4 is the more proven lane today

## Quick Start

Run the local smoke fixture from the repo root:

```bash
python3 -m tools.heretic_to_onnx bootstrap

python3 -m tools.heretic_to_onnx inspect \
  --config configs/heretic-to-onnx.local-smoke.yaml

python3 -m tools.heretic_to_onnx convert \
  --config configs/heretic-to-onnx.local-smoke.yaml \
  --work-dir build/work/local-smoke \
  --output-dir build/local-smoke
```

Generate export and quantize helper scripts:

```bash
python3 -m tools.heretic_to_onnx convert \
  --config configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml \
  --work-dir build/work/rally-2b \
  --export-mode script \
  --quantize-mode script

python3 -m tools.heretic_to_onnx convert \
  --config configs/heretic-to-onnx.qwen3-5-4b-heretic.yaml \
  --work-dir build/work/alkahest-4b \
  --export-mode script \
  --quantize-mode script
```

## Portfolio

Direct ONNX repos:

- `alkahest-ai/rally-2b`
- `alkahest-ai/rally-4b`
- `alkahest-ai/alkahest-4b`
- `alkahest-ai/alkahest-2b`
- `alkahest-ai/alkahest-0.8b`

Roleplay-tuned ONNX repos:

- `alkahest-ai/rally-2b-rp`
- `alkahest-ai/rally-4b-rp`
- `alkahest-ai/alkahest-4b-rp`
- `alkahest-ai/alkahest-2b-rp`
- `alkahest-ai/alkahest-0.8b-rp`

## Dataset

The active roleplay dataset lives in `data/roleplay_v2`.

Active workflow:

- generate small raw batches into `data/roleplay_v2/generated_raw/`
- review them in spreadsheet form from `data/roleplay_v2/review_table/`
- compile approved rows into `data/roleplay_v2/approved_jsonl/`
- build train/val splits only from approved rows plus the gold seed set
- lint and round-trip check the workflow before tuning

`data/roleplay_v1` is now prototype material and archive context.

## Main Docs

- `docs/PRELAUNCH.md`: first paid H200 checklist
- `docs/H200-LAUNCH-SEQUENCE.md`: single end-to-end launch sequence
- `docs/browser-free-chat.md`: actual static browser chat app
- `docs/phala-gpu-tee-oneclick.md`: fastest operator path
- `docs/roleplay-data-flow.md`: spreadsheet-first `roleplay_v2` dataset flow
- `docs/rally-end-to-end-runbook.md`: tune -> merge -> ONNX flow
- `docs/rally-2b-browser-export-postmortem.md`: what it took to get direct Rally 2B browser export working
- `docs/model-execution-matrix.md`: source model to target repo mapping
- `docs/phala-gpu-tee-h200-runbook.md`: GPU TEE product notes and launch guidance
- `docs/colab-runbook.md`: fallback Colab flow

## Browser Chat

There is now a real static browser chat client in `browser-chat/`.

It uses:

- Transformers.js
- WebGPU
- a built-in model picker for Gemma and Qwen browser ONNX repos
- image input for Gemma and Qwen browser lanes
- `onnx-community/Qwen3.5-0.8B-ONNX` by default

Current shipped browser exports are `text + image`. Audio is not part of the working Rally/Gemma browser path right now.

Run it locally from the repo root:

```bash
bash scripts/serve_browser_chat.sh
```

Then open:

```text
http://localhost:4173/browser-chat/
```

Read `docs/browser-free-chat.md` before wiring private or custom model hosting.

## Phala

The main paid path is Phala GPU TEE with `H200 / 1 GPU / On-Demand / Jupyter Notebook (PyTorch)`.

Primary entrypoints:

- `scripts/phala_gpu_tee_bootstrap.sh`
- `scripts/phala_gpu_tee_oneclick.sh`
- `scripts/phala_run_all_models.sh`
- `scripts/synthesize_roleplay_batch.py`
- `scripts/jsonl_to_review_table.py`
- `scripts/review_table_to_jsonl.py`
- `scripts/lint_roleplay_dataset.py`

Per-model wrappers:

- `scripts/phala_run_rally_2b_direct.sh`
- `scripts/phala_run_rally_4b_direct.sh`
- `scripts/phala_run_alkahest_4b_direct.sh`
- `scripts/phala_run_alkahest_2b_direct.sh`
- `scripts/phala_run_alkahest_0_8b_direct.sh`
- `scripts/phala_run_rally_2b_rp.sh`
- `scripts/phala_run_rally_4b_rp.sh`
- `scripts/phala_run_alkahest_4b_rp.sh`
- `scripts/phala_run_alkahest_2b_rp.sh`
- `scripts/phala_run_alkahest_0_8b_rp.sh`

Legacy `sheena-*` wrapper names and one-click modes still work as compatibility aliases.

Run the full H200 portfolio flow from the repo root with:

```bash
bash scripts/phala_run_all_models.sh
```

Quick dataset verification:

```bash
python3 -m unittest tests.test_roleplay_dataset_v2
python3 scripts/lint_roleplay_dataset.py --input data/roleplay_v2/approved_jsonl
```
