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
  --work-dir build/work/sheena-4b \
  --export-mode script \
  --quantize-mode script
```

## Portfolio

Direct ONNX repos:

- `alkahest/rally-2b`
- `alkahest/rally-4b`
- `alkahest/sheena-4b`

Roleplay-tuned ONNX repos:

- `alkahest/rally-2b-rp`
- `alkahest/rally-4b-rp`
- `alkahest/sheena-4b-rp`

## Dataset

The seeded roleplay dataset lives in `data/roleplay_v1`.

Current synthetic pass:

- `6` personas
- `10` scenes
- `240` persona/scene/lane combinations
- `25` variants per combination
- `6,000` generated conversations in `data/roleplay_v1/generated/batch-0002.jsonl`

## Main Docs

- `docs/PRELAUNCH.md`: first paid H200 checklist
- `docs/phala-gpu-tee-oneclick.md`: fastest operator path
- `docs/rally-end-to-end-runbook.md`: tune -> merge -> ONNX flow
- `docs/model-execution-matrix.md`: source model to target repo mapping
- `docs/phala-gpu-tee-h200-runbook.md`: GPU TEE product notes and launch guidance
- `docs/colab-runbook.md`: fallback Colab flow

## Phala

The main paid path is Phala GPU TEE with `H200 / 1 GPU / On-Demand / Jupyter Notebook (PyTorch)`.

Primary entrypoints:

- `scripts/phala_gpu_tee_bootstrap.sh`
- `scripts/phala_gpu_tee_oneclick.sh`
- `scripts/phala_run_all_models.sh`

Per-model wrappers:

- `scripts/phala_run_rally_2b_direct.sh`
- `scripts/phala_run_rally_4b_direct.sh`
- `scripts/phala_run_sheena_4b_direct.sh`
- `scripts/phala_run_rally_2b_rp.sh`
- `scripts/phala_run_rally_4b_rp.sh`
- `scripts/phala_run_sheena_4b_rp.sh`

Run the full H200 portfolio flow from the repo root with:

```bash
bash scripts/phala_run_all_models.sh
```
