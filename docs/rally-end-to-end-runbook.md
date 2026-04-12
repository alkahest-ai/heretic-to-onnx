# Rally End-to-End Runbook

This is the single runbook for taking `rally` from dataset prep to a browser-compatible ONNX package.

## 1. Launch GPU TEE

Use the Phala UI:

1. open `cloud.phala.com`
2. go to `GPU TEE`
3. click `Start Building`
4. choose `H200`
5. choose `1 GPU`
6. choose `Jupyter Notebook (PyTorch)`
7. choose `On-Demand`
8. launch

## 2. Prepare The Workspace

In the Jupyter terminal:

```bash
git clone <your repo url> heretic
cd heretic
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m pip install \
  "unsloth" \
  "unsloth_zoo" \
  "trl" \
  "datasets" \
  "accelerate>=1.0.0" \
  "safetensors>=0.4.3" \
  "sentencepiece>=0.2.0" \
  "onnx>=1.17.0" \
  "onnxruntime-gpu>=1.20.0" \
  "onnxconverter-common>=1.14.0" \
  "huggingface_hub[cli]>=0.31.0" \
  "transformers>=4.57.0"
export HF_TOKEN=...
```

Or use the repo bootstrap helper:

```bash
bash scripts/phala_gpu_tee_bootstrap.sh
```

## 3. Generate And Review Dataset Rows

Render prompt jobs:

```bash
python3 scripts/render_roleplay_prompt_pack.py --limit 240
```

Bulk-generate actual message rows:

```bash
python3 scripts/synthesize_roleplay_batch.py \
  --variants 25 \
  --seed 111 \
  --id-prefix bulk25 \
  --output data/roleplay_v1/generated/batch-0002.jsonl
```

This writes:

- `data/roleplay_v1/generated/batch-0002.jsonl`

Review that file and edit any rows that are too weak, repetitive, or off-tone.

## 4. Build The Training Corpus

```bash
python3 scripts/build_roleplay_training_corpus.py
python3 scripts/prepare_roleplay_dataset.py \
  --input data/roleplay_v1/corpus.jsonl
```

This produces:

- `corpus.jsonl`
- `splits/train.jsonl`
- `splits/val.jsonl`

## 5. Fine-Tune Rally

```bash
python3 scripts/train_rally_unsloth.py --save-merged
```

This writes:

- LoRA output: `build/unsloth/rally`
- merged checkpoint: `build/unsloth/rally-merged`

## 6. Create A Tuned ONNX Manifest

Point the converter at the merged local checkpoint:

```bash
python3 -m tools.heretic_to_onnx render-manifest \
  --template configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml \
  --output build/rally-tuned.yaml \
  --source-model-id build/unsloth/rally-merged \
  --target-repo-id alkahest/rally-2b-rp
```

## 7. Convert Tuned Rally To ONNX

```bash
python3 -m tools.heretic_to_onnx convert \
  --config build/rally-tuned.yaml \
  --work-dir build/work/rally-tuned \
  --output-dir build/out/rally-tuned \
  --force \
  --strict-onnx \
  --export-mode execute \
  --quantize-mode execute
```

## 8. Publish To Hugging Face

```bash
python3 -m tools.heretic_to_onnx publish-hf \
  --config build/rally-tuned.yaml \
  --package-dir build/out/rally-tuned \
  --repo-id alkahest/rally-2b-rp
```

## 9. Browser Validation

After upload, validate the package contract in a browser/WebGPU environment using the existing loader scaffold:

- `examples/browser-loader.mjs`

Desktop Safari / desktop Chromium should be the main validation targets.

## 10. One-Click H200 Path

If you want the whole Gemma loop in one terminal command on Phala GPU TEE, use:

```bash
bash scripts/phala_gpu_tee_oneclick.sh all-gemma
```

That path will:

1. install dependencies
2. build the `6k` synthetic dataset pass
3. convert `rally-2b`
4. convert `rally-4b`
5. fine-tune E2B into tuned `rally-2b-rp`
6. convert tuned `rally-2b-rp` to ONNX
7. fine-tune E4B into tuned `rally-4b-rp`
8. convert tuned `rally-4b-rp` to ONNX

If you also want the Qwen training pass in the same window:

```bash
bash scripts/phala_gpu_tee_oneclick.sh all
```

That final step also converts direct `sheena-4b`, fine-tunes `sheena-4b-rp`, and exports tuned `sheena-4b-rp` to ONNX.
