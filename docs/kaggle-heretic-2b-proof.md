# Kaggle Heretic 2B Proof

This proof runs upstream `heretic-llm` against official base checkpoints, saves a merged Hugging Face checkpoint, then hands that checkpoint to the existing ONNX package/publish flow.

The first targets are:

- `rally-2b`: `google/gemma-4-E2B-it`
- `alkahest-2b`: `Qwen/Qwen3.5-2B`

Do not start a paid export machine until the merged checkpoint exists and passes the local browser package hard gate.

## Kaggle Notebook Cells

Clone or upload this repo into the notebook, then install the proof dependencies:

```python
!pip install -U heretic-llm transformers accelerate bitsandbytes peft datasets huggingface_hub hf_transfer
```

Authenticate with a Kaggle secret or environment variable:

```python
import os
from huggingface_hub import login

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ.setdefault("HF_TOKEN", secrets.get_secret("HF_TOKEN"))
    try:
        os.environ.setdefault("HF_MERGED_REPO_ID", secrets.get_secret("HF_MERGED_REPO_ID"))
    except Exception:
        pass
except Exception:
    pass

login(token=os.environ["HF_TOKEN"])
```

Run one model at a time. Start with Alkahest if the goal is proving the Qwen lane:

```python
!python scripts/kaggle_heretic_2b_proof.py \
  --label alkahest-2b \
  --accelerator t4x2 \
  --n-trials 20 \
  --n-startup-trials 8 \
  --prompt-rows 160 \
  --eval-rows 80 \
  --max-response-length 64 \
  --upload-merged-to "$HF_OWNER/alkahest-2b-heretic-merged"
```

For Rally:

```python
!python scripts/kaggle_heretic_2b_proof.py \
  --label rally-2b \
  --accelerator t4x2 \
  --n-trials 20 \
  --n-startup-trials 8 \
  --prompt-rows 160 \
  --eval-rows 80 \
  --max-response-length 64 \
  --upload-merged-to "$HF_OWNER/rally-2b-heretic-merged"
```

The script writes:

- `config.toml`
- `heretic_stdin_answers.txt`
- `heretic.log`
- `checkpoints/`
- `<label>-heretic-merged/`

The runner forces Kaggle/notebook prompt mode by setting `KAGGLE_KERNEL_RUN_TYPE=Interactive` unless `--native-terminal-mode` is passed. It feeds Heretic answers that select the first Pareto trial, save locally, and choose the full merge path.

Pass `--upload-merged-to owner/repo` to upload the merged checkpoint before the Kaggle session can expire. In the notebook, set `HF_MERGED_REPO_ID` to enable this without editing cells.

The default accelerator profile is `t4x2`, which writes:

```toml
device_map = "auto"
max_memory = { "0" = "14GiB", "1" = "14GiB", "cpu" = "24GiB" }
```

Use `--accelerator single-gpu` for P100 or single T4 sessions. Use `--accelerator auto` if you want Heretic/Accelerate to choose without explicit memory caps.

## Checkpoint Requirements

The merged checkpoint must contain:

- `config.json`
- `generation_config.json`
- `tokenizer_config.json`
- `tokenizer.json` or equivalent tokenizer files
- `*.safetensors` or `pytorch_model*.bin`

If Heretic optimization succeeds but merge fails, keep the whole work directory. Do not rerun optimization just to retry merge. Preserve `config.toml`, `heretic.log`, and `checkpoints/` so the failure can be diagnosed or resumed.

Current upstream `heretic-llm` exposes adapter-only saving for non-quantized runs, but the `bnb_4bit` save menu only offers full merge or cancel. If a quantized merge OOMs, the safe artifact set is currently the Heretic checkpoint/log/config set.

## Handoff To ONNX

Copy the merged checkpoint directories to the export machine under one root:

```text
<HERETIC_OUTPUT_ROOT>/alkahest-2b-heretic-merged
<HERETIC_OUTPUT_ROOT>/rally-2b-heretic-merged
```

Then run the one-click flow with the new source mode:

```bash
export SOURCE_MODE=base_plus_heretic
export HERETIC_OUTPUT_ROOT=/path/to/models/heretic
export RALLY2_BASE_MODEL=google/gemma-4-E2B-it
export ALKAHEST2_BASE_MODEL=Qwen/Qwen3.5-2B

bash scripts/phala_gpu_tee_oneclick.sh alkahest-2b-v2-direct
bash scripts/phala_gpu_tee_oneclick.sh rally-2b-v2-direct
```

`SOURCE_MODE=base_plus_heretic` sets `source_model_id` to the local merged checkpoint while keeping `base_model_id` on the official base repo for inherited processor assets.

The ONNX hard gate still applies. `convert` and `publish-hf` must instantiate packaged ONNX sessions locally before upload.

## Edge Cases

- Kaggle GPU can be P100 or T4-class. The runner prints Python, disk, PyTorch, CUDA, GPU name, and package versions before running Heretic.
- P100 can be brittle with newer CUDA/Torch stacks. If the environment report shows no CUDA or a broken Torch install, stop before optimization.
- Merge can require more CPU RAM and disk than optimization. Treat a merge failure as a separate retry problem; keep the checkpoint artifacts.
- Kaggle disk is limited. Do not keep duplicate full checkpoints unless explicitly needed.
- Sessions can end. Keep artifacts under `/kaggle/working` and upload intermediate outputs if the run is important.
- Heretic modifies language model weights. Multimodal processor assets still come from the official base model through the ONNX manifest.
