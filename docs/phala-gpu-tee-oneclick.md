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
- `dataset` / `dataset-batch`: generate a small `roleplay_v2` raw batch plus TSV review table
- `dataset-compile`: build the approved `roleplay_v2` corpus and train/val splits, enforcing `MIN_APPROVED_ROWS`
- `rally-2b-direct`: convert and publish `alkahest-ai/rally-2b`
- `rally-4b-direct`: convert and publish `alkahest-ai/rally-4b`
- `alkahest-4b-direct`: convert and publish `alkahest-ai/alkahest-4b`
- `alkahest-2b-direct`: convert and publish `alkahest-ai/alkahest-2b`
- `alkahest-0.8b-direct`: convert and publish `alkahest-ai/alkahest-0.8b`
- `rally`: compile approved `roleplay_v2`, train tuned `rally-2b-rp`, export tuned `rally-2b-rp` to ONNX, publish it
- `rally-4b`: compile approved `roleplay_v2`, train tuned `rally-4b-rp`, export tuned `rally-4b-rp` to ONNX, publish it
- `alkahest-4b`: compile approved `roleplay_v2`, train tuned `alkahest-4b-rp`, export tuned `alkahest-4b-rp` to ONNX, publish it
- `alkahest-2b`: compile approved `roleplay_v2`, train tuned `alkahest-2b-rp`, export tuned `alkahest-2b-rp` to ONNX, publish it
- `alkahest-0.8b`: compile approved `roleplay_v2`, train tuned `alkahest-0.8b-rp`, export tuned `alkahest-0.8b-rp` to ONNX, publish it
- `all-gemma`: bootstrap, both direct Gemma conversions, approved-corpus compile, both tuned Gemma conversions
- `all-qwen`: bootstrap, all direct Alkahest Qwen conversions, approved-corpus compile, all tuned Alkahest Qwen conversions
- `all`: direct conversions plus approved-corpus compile plus all tuned conversions

## Recommended First Paid Run

Use:

```bash
bash scripts/phala_gpu_tee_oneclick.sh all-gemma
```

That covers the main deliverables that are already implemented in this repo, but it now assumes you already have an approved `roleplay_v2` corpus:

1. `alkahest-ai/rally-2b`
2. `alkahest-ai/rally-4b`
3. tuned `alkahest-ai/rally-2b-rp`
4. tuned `alkahest-ai/rally-4b-rp`

If you still have time left in the 24-hour window, then run:

```bash
bash scripts/phala_gpu_tee_oneclick.sh all
```

That adds:

5. direct `alkahest-ai/alkahest-4b`
6. direct `alkahest-ai/alkahest-2b`
7. direct `alkahest-ai/alkahest-0.8b`
8. tuned `alkahest-ai/alkahest-4b-rp`
9. tuned `alkahest-ai/alkahest-2b-rp`
10. tuned `alkahest-ai/alkahest-0.8b-rp`

## Important Env Vars

Defaults are already embedded, but these are the most useful overrides:

- `HF_TOKEN`: required for gated model access and Hub uploads
- `DATASET_COUNT`: default `300` raw conversations per review batch
- `REVIEW_OUTPUT`: default `data/roleplay_v2/review_table/batch-0001.tsv`
- `MIN_APPROVED_ROWS`: default `5000`
- `HF_PRIVATE`: default `1`
- `RALLY2_DIRECT_REPO`: default `alkahest-ai/rally-2b`
- `RALLY4_DIRECT_REPO`: default `alkahest-ai/rally-4b`
- `ALKAHEST4_DIRECT_REPO`: default `alkahest-ai/alkahest-4b`
- `ALKAHEST2_DIRECT_REPO`: default `alkahest-ai/alkahest-2b`
- `ALKAHEST08_DIRECT_REPO`: default `alkahest-ai/alkahest-0.8b`
- `RALLY2_TUNED_REPO`: default `alkahest-ai/rally-2b-rp`
- `RALLY4_TUNED_REPO`: default `alkahest-ai/rally-4b-rp`
- `ALKAHEST4_SOURCE_MODEL`: default `tvall43/Qwen3.5-4B-heretic`
- `ALKAHEST2_SOURCE_MODEL`: default `tvall43/Qwen3.5-2B-heretic-v3b`
- `ALKAHEST08_SOURCE_MODEL`: default `tvall43/Qwen3.5-0.8B-heretic-v3`
- `ALKAHEST4_TUNED_REPO`: default `alkahest-ai/alkahest-4b-rp`
- `ALKAHEST2_TUNED_REPO`: default `alkahest-ai/alkahest-2b-rp`
- `ALKAHEST08_TUNED_REPO`: default `alkahest-ai/alkahest-0.8b-rp`
- `RALLY_MAX_STEPS`, `RALLY4_MAX_STEPS`, `ALKAHEST4_MAX_STEPS`, `ALKAHEST2_MAX_STEPS`, `ALKAHEST08_MAX_STEPS`

Example:

```bash
export HF_TOKEN=...
export DATASET_COUNT=300
export RALLY4_TUNED_REPO=alkahest-ai/rally-4b-rp
bash scripts/phala_gpu_tee_oneclick.sh dataset-batch
```

## Per-Model Wrapper Scripts

If you prefer one entrypoint per output, use:

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
- `scripts/phala_run_all_models.sh`

## Reviewing The Synthetic Dataset

Before training, spot-check the generated batch:

```bash
python3 scripts/sample_roleplay_rows.py --input data/roleplay_v2/approved_jsonl --count 5
python3 scripts/sample_roleplay_rows.py --input data/roleplay_v2/approved_jsonl --lane praise --count 5
```

Typical raw and review files are:

- `data/roleplay_v2/generated_raw/batch-0001.jsonl`
- `data/roleplay_v2/review_table/batch-0001.tsv`

After review, compile and rebuild:

```bash
python3 scripts/review_table_to_jsonl.py \
  --input data/roleplay_v2/review_table/batch-0001.tsv \
  --output data/roleplay_v2/approved_jsonl/batch-0001.jsonl
python3 scripts/build_roleplay_training_corpus.py
python3 scripts/prepare_roleplay_dataset.py \
  --input data/roleplay_v2/corpus.jsonl
```

## Current Limitation

Gemma 4 is still the more proven export lane in this repo.

Qwen3.5 now has the same plan/script/execute scaffold and package path, but it is less proven because it has not been executed on a real GPU runtime from this workspace yet.

For consumer browser chat, `alkahest-0.8b` and `alkahest-2b` are the better first Qwen deployment targets than `alkahest-4b`.
