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
   - `tvall43/Qwen3.5-2B-heretic-v3b`
   - `tvall43/Qwen3.5-0.8B-heretic-v3`
   - `google/gemma-4-E2B-it`
   - `google/gemma-4-E4B-it`
   - `Qwen/Qwen3.5-4B`
   - `Qwen/Qwen3.5-2B`
   - `Qwen/Qwen3.5-0.8B`
2. write access to:
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

## Dataset

1. Review:
   - `data/roleplay_v2/review_table/*.tsv`
   - `data/roleplay_v2/approved_jsonl/*.jsonl`
2. Spot-check with:
   - `python3 scripts/sample_roleplay_rows.py --input data/roleplay_v2/approved_jsonl --count 5`
   - `python3 scripts/sample_roleplay_rows.py --input data/roleplay_v2/approved_jsonl --lane praise --count 5`
   - `python3 -m unittest tests.test_roleplay_dataset_v2`
3. If you edit review tables manually, recompile before training:
   - `python3 scripts/review_table_to_jsonl.py --input data/roleplay_v2/review_table/<batch>.tsv --output data/roleplay_v2/approved_jsonl/<batch>.jsonl`
4. Rebuild before training:
   - `python3 scripts/build_roleplay_training_corpus.py`
   - `python3 scripts/prepare_roleplay_dataset.py --input data/roleplay_v2/corpus.jsonl`

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
4. `bash scripts/phala_run_sheena_2b_direct.sh`
5. `bash scripts/phala_run_sheena_0_8b_direct.sh`
6. `bash scripts/phala_run_rally_2b_rp.sh`
7. `bash scripts/phala_run_rally_4b_rp.sh`
8. `bash scripts/phala_run_sheena_2b_rp.sh`
9. `bash scripts/phala_run_sheena_0_8b_rp.sh`
10. `bash scripts/phala_run_sheena_4b_rp.sh`

If you want the all-in-one portfolio run instead:

```bash
bash scripts/phala_run_all_models.sh
```

That now assumes an approved `roleplay_v2` corpus already exists.

## Expected Outputs

Direct browser ONNX repos:

- `alkahest/rally-2b`
- `alkahest/rally-4b`
- `alkahest/sheena-4b`
- `alkahest/sheena-2b`
- `alkahest/sheena-0.8b`

Roleplay-tuned browser ONNX repos:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`
- `alkahest/sheena-2b-rp`
- `alkahest/sheena-0.8b-rp`

## Reality Check

Gemma 4 is the more proven export lane in this repo.

Qwen3.5 is scaffolded to the same plan/script/execute level, but it has not been executed on a real GPU runtime from this workspace yet. So if you want the lowest-risk launch order, do the direct Gemma runs first.
