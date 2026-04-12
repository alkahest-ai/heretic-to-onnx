# Phala GPU TEE H200 Runbook

This is the operator guide for running the **paid H200 GPU TEE** path on Phala Cloud for this repo.

Date context: April 11, 2026.

## Short Answer

### Can we start GPU TEE H200 from the Phala CLI?

Not from the official workflow I could verify.

The current official Phala docs for **GPU TEE** describe launching it from the **GPU TEE dashboard wizard**:

1. sign in to `cloud.phala.com`
2. open `GPU TEE`
3. click `Start Building`
4. choose hardware, GPU count, template, and plan

The official `phala` CLI docs I verified are for **CVMs** and `phala ssh`, not for launching GPU TEE H100/H200/B200 instances.

So the working assumption for now is:

- `CVM + phala deploy + phala ssh`: yes, CLI-driven
- `GPU TEE H200 on-demand`: **UI-driven**

If Phala later adds documented CLI/API support for GPU TEE launch, we can add that path to this repo.

### Does CVM still matter if we use GPU TEE?

Not for the main paid H200 execution path.

For this project:

- **GPU TEE** is the main paid path
- **CVM** is now just a fallback/debug path

We mention CVM because earlier repo work added a CLI-driven Docker Compose lane, and it can still be useful if you later want:

- a cheaper debug sandbox
- `phala ssh`
- attested Docker/Compose experiments outside the H200 flow

But if the plan is “use H200 GPU TEE for the real work,” then you should mostly ignore the CVM path.

## Why Use GPU TEE Instead Of A CVM?

For this repo, GPU TEE is the better paid execution lane when you want to:

- export large Gemma 4 checkpoints to ONNX
- quantize them for browser/WebGPU
- fine-tune with Unsloth before conversion
- use H100/H200/B200 rather than a smaller general CVM

Official Phala GPU TEE docs say:

- supported GPUs include `H100`, `H200`, and `B200`
- each instance includes NVIDIA Driver `570.133.20`
- each instance includes CUDA `12.8`
- you can scale from `1` to `8` GPUs per instance

## Billing Reality

GPU TEE on-demand is not like regular CVMs.

Phala’s FAQ says:

- regular CVMs are billed by seconds
- GPU TEE is billed by hours with a **24-hour minimum**

For on-demand GPU TEE, the FAQ gives this specific H200 example:

- **`$84 upfront for the first 24 hours`**
- billing starts only when the instance is actually launched
- stopping and starting again later triggers the 24-hour minimum again

That means you should treat the first H200 day as a **planned work window**, not as a casual experiment.

## Recommended H200 Configuration

For the first paid run:

- GPU: `H200`
- GPU count: `1`
- plan: `On-Demand`
- template: `Jupyter Notebook (PyTorch)`

Why:

- it is the lowest commitment H200 path
- Jupyter gives you browser access plus terminal access
- it is the least friction path for both training and conversion work

Use `Custom Configuration` only if you specifically want a container-oriented custom workload inside GPU TEE from day one.

For this repo, `Jupyter Notebook (PyTorch)` is the cleaner first choice because it gives you a terminal and notebook environment without forcing the CVM/Docker path back into the critical path.

## What To Do Before Launching The Paid 24-Hour Window

Do all of this locally first.

### Local readiness

Run:

```bash
bash /Users/area/heretic/scripts/phala_doctor.sh
```

### Phala auth

Your local Phala CLI auth drifted and currently reports `Invalid API key`.

If you want the local CVM tooling available too, refresh it now:

```bash
phala login
phala status
```

This does **not** launch GPU TEE, but it keeps the rest of the Phala tooling ready.

### Hugging Face token

Your `HF_TOKEN` should have:

- read access to `p-e-w/gemma-4-E2B-it-heretic-ara`
- read access to `coder3101/gemma-4-E4B-it-heretic`
- read access to `google/gemma-4-E2B-it`
- read access to `google/gemma-4-E4B-it`
- write access to your target ONNX repos

Also make sure the account behind the token has accepted any gated Gemma terms.

### Repo prep

Before paying for H200 time, make sure the repo branch already contains:

- the current converter tooling
- the Gemma E2B manifest
- the Gemma E4B manifest
- the docs in this directory

## Recommended Use Of The First 24-Hour Window

Do not spend the window only on setup.

Use this order:

1. verify GPU TEE and install the repo/tooling
2. convert `Rally 2B`
3. convert `Rally 4B`
4. if time remains, start the first roleplay fine-tune + export loop

This gets you two direct portfolio assets before you risk the longer fine-tune loop.

## Suggested Portfolio Order

### Wave 1: direct ONNX compatibility

- `alkahest-ai/rally-2b`
- `alkahest-ai/rally-4b`
- `alkahest-ai/sheena-4b`
- `alkahest-ai/sheena-2b`
- `alkahest-ai/sheena-0.8b`

### Wave 2: Alkahest fine-tuned Gemma variants

- `alkahest-ai/rally-2b-rp`
- `alkahest-ai/rally-4b-rp`

### Wave 3: Qwen variance lane

- `alkahest-ai/sheena-4b-rp`
- `alkahest-ai/sheena-2b-rp`
- `alkahest-ai/sheena-0.8b-rp`

Wave 3 is now scaffolded in this repo, but it is still the less proven export lane compared to Gemma 4.

If browser-first consumer deployment is the main goal, prioritize:

1. `sheena-0.8b`
2. `sheena-2b`
3. `sheena-4b`

## GPU TEE Launch Flow

From the official docs:

1. open the Phala dashboard
2. go to `GPU TEE`
3. click `Start Building`
4. choose `H200`
5. choose `1 GPU`
6. choose `Jupyter Notebook (PyTorch)`
7. choose `On-Demand`
8. launch

Provisioning is documented as taking **about 1 day**, so do not plan around immediate access.

## What To Run Once The Instance Is Ready

Open the JupyterLab URL from the dashboard, then open a terminal.

### Verify the GPU TEE

Phala’s official verification commands are:

```bash
nvidia-smi
nvidia-smi conf-compute -q
pip install nv-local-gpu-verifier nv_attestation_sdk
python -m verifier.cc_admin
```

### Bring in this repo

Use whichever is simplest operationally:

- clone the repo from GitHub
- upload a zip and unpack it

Then:

```bash
cd /path/to/heretic
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m pip install \
  "accelerate>=1.0.0" \
  "safetensors>=0.4.3" \
  "sentencepiece>=0.2.0" \
  "onnx>=1.17.0" \
  "onnxruntime-gpu>=1.20.0" \
  "onnxconverter-common>=1.14.0" \
  "huggingface_hub[cli]>=0.31.0" \
  "transformers>=4.57.0"
```

Set your Hugging Face token in the notebook terminal:

```bash
export HF_TOKEN=...
```

## Convert Rally 2B

```bash
python3 -m tools.heretic_to_onnx convert \
  --config /path/to/heretic/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml \
  --work-dir /workspace/heretic/build/work/gemma4-e2b \
  --output-dir /workspace/heretic/build/out/gemma4-e2b \
  --force \
  --strict-onnx \
  --export-mode execute \
  --quantize-mode execute
```

Then upload:

```bash
python3 -m tools.heretic_to_onnx publish-hf \
  --config /path/to/heretic/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml \
  --package-dir /workspace/heretic/build/out/gemma4-e2b
```

## Convert Rally 4B

```bash
python3 -m tools.heretic_to_onnx convert \
  --config /path/to/heretic/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml \
  --work-dir /workspace/heretic/build/work/gemma4-e4b \
  --output-dir /workspace/heretic/build/out/gemma4-e4b \
  --force \
  --strict-onnx \
  --export-mode execute \
  --quantize-mode execute
```

Then upload:

```bash
python3 -m tools.heretic_to_onnx publish-hf \
  --config /path/to/heretic/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml \
  --package-dir /workspace/heretic/build/out/gemma4-e4b
```

## Fine-Tune Then Convert

Yes, this is possible.

The practical sequence is:

1. fine-tune the **HF checkpoint** with Unsloth
2. save or merge the tuned weights back into a normal Hugging Face repo
3. point the ONNX converter at that merged model repo
4. export and quantize the merged model

The key rule is:

- **train first**
- **merge back to a standard HF checkpoint**
- **convert that merged checkpoint to ONNX**

Do not make GGUF your primary artifact if browser ONNX is the target.

## Multimodal + Roleplay Fine-Tune Guidance

For Gemma 4:

- a text-only roleplay dataset can still be useful
- the model can remain multimodal even if the fine-tune data is mostly text
- but you should treat image/audio behavior as needing regression checks after tuning

So the safe framing is:

- text roleplay tuning adjusts behavior and style
- multimodal support is retained at the architecture level
- multimodal quality still needs validation after the tune

## What Not To Do In The First H200 Window

- do not burn the whole day only on Docker/registry friction
- do not start with Qwen unless Gemma E2B/E4B are already planned and staged
- do not assume `phala ssh` is your entry path for GPU TEE
- do not stop and restart casually once the 24-hour billing window starts

## Sources

- Phala GPU TEE deploy guide: [docs](https://docs.phala.com/phala-cloud/confidential-ai/confidential-gpu/deploy-and-verify)
- Phala FAQ billing notes: [docs](https://docs.phala.com/phala-cloud/faqs)
- Phala SSH guide: [docs](https://docs.phala.com/phala-cloud/networking/enable-ssh-access)
