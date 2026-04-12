# Prelaunch

This is the short checklist to run before you spend the first paid H200 window.

## Repo

1. Confirm `origin` is:
   `git@github.com:thomasjvu/gemma-4-E2B-it-heretic-ara-ONNX.git`
2. Confirm the main operator docs exist:
   - `docs/phala-gpu-tee-oneclick.md`
   - `docs/model-execution-matrix.md`
   - `docs/rally-end-to-end-runbook.md`

## Hugging Face

Before launch, make sure the account behind `HF_TOKEN` has:

1. read access to:
   - `p-e-w/gemma-4-E2B-it-heretic-ara`
   - `coder3101/gemma-4-E4B-it-heretic`
   - `tvall43/Qwen3.5-4B-heretic`
   - `google/gemma-4-E2B-it`
   - `google/gemma-4-E4B-it`
   - `Qwen/Qwen3.5-4B`
2. write access to:
   - `alkahest/rally-2b`
   - `alkahest/rally-4b`
   - `alkahest/sheena-4b`
   - `alkahest/rally-2b-rp`
   - `alkahest/rally-4b-rp`
   - `alkahest/sheena-4b-rp`

## Dataset

1. Review:
   - `data/roleplay_v1/generated/batch-0002.jsonl`
2. Spot-check with:
   - `python3 scripts/sample_roleplay_rows.py --count 5`
   - `python3 scripts/sample_roleplay_rows.py --tag praise --count 5`
3. If you edit rows manually, rebuild before training:
   - `python3 scripts/build_roleplay_training_corpus.py`
   - `python3 scripts/prepare_roleplay_dataset.py --input data/roleplay_v1/corpus.jsonl`

## Phala Runtime

Use the Phala UI and choose:

1. `GPU TEE`
2. `H200`
3. `1 GPU`
4. `Jupyter Notebook (PyTorch)`
5. `On-Demand`

Inside the terminal:

```bash
git clone git@github.com:thomasjvu/gemma-4-E2B-it-heretic-ara-ONNX.git heretic
cd heretic
export HF_TOKEN=...
bash scripts/phala_gpu_tee_bootstrap.sh
```

## Recommended Run Order

Safest first live order:

1. `bash scripts/phala_run_rally_2b_direct.sh`
2. `bash scripts/phala_run_rally_4b_direct.sh`
3. `bash scripts/phala_run_sheena_4b_direct.sh`
4. `bash scripts/phala_run_rally_2b_rp.sh`
5. `bash scripts/phala_run_rally_4b_rp.sh`
6. `bash scripts/phala_run_sheena_4b_rp.sh`

If you want the all-in-one portfolio run instead:

```bash
bash scripts/phala_run_all_models.sh
```

## Expected Outputs

Direct browser ONNX repos:

- `alkahest/rally-2b`
- `alkahest/rally-4b`
- `alkahest/sheena-4b`

Roleplay-tuned browser ONNX repos:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`

## Reality Check

Gemma 4 is the more proven export lane in this repo.

Qwen3.5 is scaffolded to the same plan/script/execute level, but it has not been executed on a real GPU runtime from this workspace yet. So if you want the lowest-risk launch order, do the direct Gemma runs first.
