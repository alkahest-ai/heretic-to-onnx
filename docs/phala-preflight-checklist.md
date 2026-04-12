# Phala Preflight Checklist

Use this before spending real money on a Phala deployment.

Date context for this checklist: April 11, 2026.

## 1. Choose The Right Phala Product

Phala currently exposes two relevant paths:

- `CVM` with `phala deploy` and optional `phala ssh`
- `GPU TEE` for H100/H200/B200 custom AI workloads

They are related, but they are **not** the same workflow.

### Direct answer

If you use **GPU TEE H200**, you are not also deploying the same workload through the repo’s `phala deploy` CVM path.

Think of it this way:

- `CVM`: Phala’s general Docker Compose TEE VM product
- `GPU TEE`: Phala’s Confidential AI GPU product for H100/H200/B200 workloads

Both are part of Phala Cloud, but they are different operator flows.

### CVM path

Use this when you want:

- `phala deploy` from the CLI
- `phala ssh` debugging
- Docker Compose driven deployment from this repo

Important: official SSH docs say `phala ssh` only works with **dev OS images**. Production images have SSH disabled.

This repo’s current `phala_deploy.sh`, `phala ssh`, and Docker Compose helpers are for this **CVM path**.

### GPU TEE path

Use this when you want:

- H100, H200, or B200 GPUs in TEE
- training, fine-tuning, or heavy export/quantization workloads
- Jupyter Notebook (PyTorch), vLLM, or Custom Configuration

Important: the official GPU TEE docs currently describe launching from the **GPU TEE dashboard wizard**, not from `phala deploy`.

For this repo, GPU TEE should be treated as the paid execution lane for:

- heavy Gemma export
- roleplay fine-tuning
- subsequent ONNX conversion

## 2. Can We Use H200 1-GPU On-Demand?

Yes.

Official Phala docs currently say GPU TEE supports:

- H200 in US and India
- 1 to 8 GPUs per instance
- On-Demand, 1-month, and 6-month plans

Current documented rates on the GPU TEE page:

- H200 US: `$2.56/GPU/hour`
- H200 India: `$2.30/GPU/hour`

Current documented plan examples on the same page:

- `6-month commitment`: about `$2.88/GPU/hour`
- `1-month commitment`: about `$3.20/GPU/hour`
- `On-Demand`: about `$3.50/GPU/hour + storage`

Important billing notes from the FAQ:

- GPU TEE on-demand is billed by hours with a **24-hour minimum**
- For on-demand H200, the docs give an example minimum of **`$84 upfront for the first 24 hours`**
- If you stop and restart later, the 24-hour minimum applies again

So the answer is:

- `Yes`, you can use a **1-GPU H200 on-demand** instance instead of a monthly commitment.
- `No`, it is not minute-by-minute like regular CVMs.
- Budget as if the first day is a committed spend.

## 3. Which Path Fits This Repo Best?

### Best path for lowest operational risk

For the current repo, the cleanest option is:

1. use **GPU TEE H200 on-demand**
2. choose **Jupyter Notebook (PyTorch)** or **Custom Configuration**
3. run the export/quantize/package pipeline there
4. upload the finished package to Hugging Face

Why:

- this ONNX conversion job is a heavy GPU workload
- H200 GPU TEE is explicitly documented for training, fine-tuning, and custom code
- the docs say it includes CUDA `12.8` and NVIDIA driver `570.133.20`

### Best path for SSH-style debugging

If you specifically want `phala ssh`, port forwarding, and shell access through the CLI, keep using the **CVM debug** path from this repo:

- `/Users/area/heretic/docker/phala-compose.debug.yml`
- `/Users/area/heretic/scripts/phala_deploy.sh --debug ...`

That is the right path for CVM debugging, but it is not the same thing as the GPU TEE H200 product page.

### Practical takeaway

- if you want Docker Compose + `phala ssh`, use **CVM**
- if you want the **H200 GPU TEE** product, use the **GPU TEE UI**
- do not assume one is a thin wrapper around the other

## 4. What Is Already Set Up Locally

On this machine, the local Phala operator prerequisites are mostly in place:

- `phala` CLI installed
- Docker installed
- Node/npm installed
- SSH public key present
- repo helper scripts present

Current caveat:

- the local `phala` session can drift; if `phala status` reports `Invalid API key`, rerun `phala login`

Run this any time to confirm:

```bash
/Users/area/heretic/scripts/phala_doctor.sh
```

## 5. Go / No-Go Checklist

Before you launch a paid GPU TEE instance, confirm all of these:

- your Hugging Face token has read access to the source and base Gemma repos
- your Hugging Face token has write access to the target ONNX repo
- the Hugging Face account has accepted any gated Gemma terms
- your Docker image is built and pushed to a registry Phala can pull from
- your target repo ID is correct
- you understand the GPU TEE 24-hour minimum billing model
- you know whether you are using the `CVM + SSH` path or the `GPU TEE` path

## 6. Recommended Decision

For this job, the practical recommendation is:

- use **GPU TEE H200, 1 GPU, On-Demand**
- assume the first 24 hours are billable
- use **Jupyter Notebook (PyTorch)** if you want the least friction
- use **Custom Configuration** only if you specifically want the Dockerized repo workflow inside GPU TEE

## Sources

- Phala SSH docs: `https://docs.phala.com/phala-cloud/networking/enable-ssh-access`
- Phala CLI overview: `https://docs.phala.com/phala-cloud/phala-cloud-cli/overview`
- Phala GPU TEE deploy/verify: `https://docs.phala.com/phala-cloud/confidential-ai/confidential-gpu/deploy-and-verify`
- Phala FAQ: `https://docs.phala.com/phala-cloud/faqs`
