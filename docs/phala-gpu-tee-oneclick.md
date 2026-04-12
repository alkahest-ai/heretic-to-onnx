# Phala GPU TEE One-Click

This is the shortest repeatable path for the first paid `H200 / 1 GPU / On-Demand` run on Phala.

## Runtime Choice

Use the Phala UI and choose:

1. `GPU TEE`
2. `H200`
3. `1 GPU`
4. `Jupyter Notebook (PyTorch)`
5. `On-Demand`

This repo does **not** depend on the CVM path for the main H200 run.

## One-Time Terminal Setup

Inside the Phala Jupyter terminal:

```bash
git clone <your repo url> heretic
cd heretic
export HF_TOKEN=...
```

Then bootstrap the environment:

```bash
bash scripts/phala_gpu_tee_bootstrap.sh
```

## One-Click Modes

The main operator entrypoint is:

```bash
bash scripts/phala_gpu_tee_oneclick.sh <mode>
```

Available modes:

- `bootstrap`: install Python dependencies and run converter preflight
- `dataset`: generate the `6k` synthetic roleplay batch and rebuild train/val splits
- `rally-2b-direct`: convert and publish `alkahest/rally-2b`
- `rally-4b-direct`: convert and publish `alkahest/rally-4b`
- `sheena-4b-direct`: convert and publish `alkahest/sheena-4b`
- `rally`: build dataset, train tuned `rally-2b-rp`, export tuned `rally-2b-rp` to ONNX, publish it
- `rally-4b`: build dataset, train tuned `rally-4b-rp`, export tuned `rally-4b-rp` to ONNX, publish it
- `sheena`: build dataset, train tuned `sheena-4b-rp`, export tuned `sheena-4b-rp` to ONNX, publish it
- `all-gemma`: bootstrap, dataset, both direct Gemma conversions, both tuned Gemma conversions
- `all`: `all-gemma` plus direct and tuned `sheena`

## Recommended First Paid Run

Use:

```bash
bash scripts/phala_gpu_tee_oneclick.sh all-gemma
```

That covers the main deliverables that are already implemented in this repo:

1. `alkahest/rally-2b`
2. `alkahest/rally-4b`
3. tuned `alkahest/rally-2b-rp`
4. tuned `alkahest/rally-4b-rp`

If you still have time left in the 24-hour window, then run:

```bash
bash scripts/phala_gpu_tee_oneclick.sh all
```

That adds:

5. direct `alkahest/sheena-4b`
6. tuned `alkahest/sheena-4b-rp`

## Important Env Vars

Defaults are already embedded, but these are the most useful overrides:

- `HF_TOKEN`: required for gated model access and Hub uploads
- `DATASET_VARIANTS`: default `25`, giving `6,000` synthetic rows
- `HF_PRIVATE`: default `1`
- `RALLY2_DIRECT_REPO`: default `alkahest/rally-2b`
- `RALLY4_DIRECT_REPO`: default `alkahest/rally-4b`
- `SHEENA_DIRECT_REPO`: default `alkahest/sheena-4b`
- `RALLY2_TUNED_REPO`: default `alkahest/rally-2b-rp`
- `RALLY4_TUNED_REPO`: default `alkahest/rally-4b-rp`
- `SHEENA_SOURCE_MODEL`: default `tvall43/Qwen3.5-4B-heretic`
- `SHEENA_TUNED_REPO`: default `alkahest/sheena-4b-rp`
- `RALLY_MAX_STEPS`, `RALLY4_MAX_STEPS`, `SHEENA_MAX_STEPS`

Example:

```bash
export HF_TOKEN=...
export DATASET_VARIANTS=30
export RALLY4_TUNED_REPO=alkahest/rally-4b-rp
bash scripts/phala_gpu_tee_oneclick.sh all
```

## Per-Model Wrapper Scripts

If you prefer one entrypoint per output, use:

- `scripts/phala_run_rally_2b_direct.sh`
- `scripts/phala_run_rally_4b_direct.sh`
- `scripts/phala_run_sheena_4b_direct.sh`
- `scripts/phala_run_rally_2b_rp.sh`
- `scripts/phala_run_rally_4b_rp.sh`
- `scripts/phala_run_sheena_4b_rp.sh`
- `scripts/phala_run_all_models.sh`

## Reviewing The Synthetic Dataset

Before training, spot-check the generated batch:

```bash
python3 scripts/sample_roleplay_rows.py --count 5
python3 scripts/sample_roleplay_rows.py --tag praise --count 5
```

The main generated file is:

- `data/roleplay_v1/generated/batch-0002.jsonl`

After manual edits, rebuild the corpus:

```bash
python3 scripts/build_roleplay_training_corpus.py
python3 scripts/prepare_roleplay_dataset.py \
  --input data/roleplay_v1/corpus.jsonl
```

## Current Limitation

Gemma 4 is still the more proven export lane in this repo.

Qwen3.5 now has the same plan/script/execute scaffold and package path, but it is less proven because it has not been executed on a real GPU runtime from this workspace yet.
