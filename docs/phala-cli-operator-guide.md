# Phala CLI Operator Guide

This guide is the reproducible operator path for running this repo on Phala Cloud. It is written for the real `phala` CLI as of April 11, 2026 and covers:

- local CLI setup
- image build and push
- one-shot batch deployment for the ONNX conversion pipeline
- interactive debug deployment with SSH access
- how to rerun the pipeline from inside the CVM
- what to commit so other contributors can reuse the workflow

Read `/Users/area/heretic/docs/phala-preflight-checklist.md` first if you need to decide between the CVM + SSH path and the GPU TEE H200/B200 path before launching paid infrastructure.

## What This Repo Actually Runs

This repo currently automates:

- Gemma 4 Heretic repo preparation
- raw Gemma 4 ONNX export
- q4f16 quantization
- browser/WebGPU package assembly
- Hugging Face model repo upload

It does **not** currently automate SFT/RLHF training. If you later add training code, use the debug/interactive Phala path in this guide to run that code inside the CVM/container.

## Important Product Split

There are two different Phala workflows relevant to this repo:

- `CVM + phala deploy + phala ssh`
- `GPU TEE (H100/H200/B200)` launched from the GPU TEE product flow

This guide focuses on the **CVM CLI workflow** because that is what the current repo scripts automate directly.

If you want **H200 on-demand GPU TEE**, see `/Users/area/heretic/docs/phala-preflight-checklist.md` first. The official GPU TEE docs currently describe that flow through the dashboard wizard, with Jupyter Notebook, vLLM, or Custom Configuration as the starting template.

## Files To Know

- CLI wrapper: `/Users/area/heretic/scripts/phala_cli.sh`
- Local prerequisite check: `/Users/area/heretic/scripts/phala_doctor.sh`
- Deploy helper: `/Users/area/heretic/scripts/phala_deploy.sh`
- Batch compose: `/Users/area/heretic/docker/phala-compose.yml`
- Debug compose: `/Users/area/heretic/docker/phala-compose.debug.yml`
- Env template: `/Users/area/heretic/docker/phala.env.example`
- Env generator: `/Users/area/heretic/scripts/write_phala_env.sh`
- Image build/push helper: `/Users/area/heretic/scripts/build_phala_image.sh`
- Runtime container entrypoint: `/Users/area/heretic/docker/phala-entrypoint.sh`

## 1. Local Prerequisites

Phala’s current docs say the CLI supports:

- global install: `npm install -g phala`
- no-install use: `npx phala` or `bunx phala`

This machine now has:

- `phala` installed globally
- Docker installed
- Node/npm installed
- a working authenticated Phala CLI profile

Run the local repo check any time:

```bash
/Users/area/heretic/scripts/phala_doctor.sh
```

## 2. Login And Verify

If a new operator is setting this up on another machine:

```bash
phala login
phala status
```

The CLI also supports `phala login --manual` if you want to paste an API token instead of using device flow.

## 3. Build And Push The Container Image

Build and push the Phala runtime image:

```bash
/Users/area/heretic/scripts/build_phala_image.sh ghcr.io/your-org/heretic-to-onnx:latest --push
```

If your registry is private, configure registry access in Phala before deploy.

## 4. Generate The Phala Env File

Write the env file used by `phala deploy -e`:

```bash
/Users/area/heretic/scripts/write_phala_env.sh \
  /Users/area/heretic/build/phala.env \
  ghcr.io/your-org/heretic-to-onnx:latest \
  lightnolimit/gemma-4-E2B-it-heretic-ara-ONNX
```

Then replace the placeholder `HF_TOKEN` handling with a real secret strategy:

- for straightforward deploys, keep `HF_TOKEN` in the env file only if your operational model allows it
- for safer operations, use Phala sealed env updates after deploy with `phala envs update`

The Hugging Face token needs:

- read access to `p-e-w/gemma-4-E2B-it-heretic-ara`
- read access to `google/gemma-4-E2B-it`
- write access to your target ONNX repo

## 5. Batch Deploy: Run The Whole Pipeline Automatically

This is the normal path when you just want the CVM to do the conversion job and publish the result.

List GPU instance types first:

```bash
phala instance-types gpu
```

Then deploy:

```bash
/Users/area/heretic/scripts/phala_deploy.sh \
  --name heretic-onnx-h100 \
  --env-file /Users/area/heretic/build/phala.env \
  --instance-type <gpu-instance-type>
```

What this does:

- deploys `/Users/area/heretic/docker/phala-compose.yml`
- waits for completion
- disables public logs and public sysinfo by default
- links this repo to the deployed CVM with `phala link`

After linking, the repo gets a `phala.toml` for faster follow-up commands. Commit that file if the linked CVM is meant to be shared by the team.

Useful follow-up commands:

```bash
phala cvms get
phala ps
phala logs heretic-to-onnx -f
phala runtime-config --json
```

## 6. Debug Deploy: Keep The Container Alive And SSH In

Use this when you want to inspect the CVM, run the pipeline manually, or later run custom training commands.

First ensure your SSH public key exists:

```bash
ls ~/.ssh/id_ed25519.pub
```

If you want the key registered in your account explicitly:

```bash
phala ssh-keys add --key-file ~/.ssh/id_ed25519.pub
```

Then deploy the debug compose:

```bash
/Users/area/heretic/scripts/phala_deploy.sh \
  --name heretic-onnx-debug \
  --env-file /Users/area/heretic/build/phala.env \
  --instance-type <gpu-instance-type> \
  --debug \
  --ssh-pubkey ~/.ssh/id_ed25519.pub
```

What changes in debug mode:

- uses `/Users/area/heretic/docker/phala-compose.debug.yml`
- enables `--dev-os`
- keeps the container idle with `sleep infinity`
- leaves `UPLOAD_TO_HF=0` so nothing is published until you run it yourself

## 7. SSH Into The CVM And Run The Pipeline Manually

SSH into the linked CVM:

```bash
phala ssh
```

Or target by name:

```bash
phala ssh heretic-onnx-debug
```

Once inside the VM:

```bash
docker ps
docker exec -it <container-id> bash
```

Once inside the container:

```bash
env | grep -E 'HF_TOKEN|SOURCE_MODEL_ID|BASE_MODEL_ID|TARGET_REPO_ID|WORK_DIR|OUTPUT_DIR'
/app/docker/phala-entrypoint.sh
```

If you only want part of the flow:

```bash
cd /app
python3 -m tools.heretic_to_onnx render-manifest \
  --template /app/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml \
  --output /data/runtime-manifest.yaml \
  --source-model-id "${SOURCE_MODEL_ID}" \
  --base-model-id "${BASE_MODEL_ID}" \
  --target-repo-id "${TARGET_REPO_ID}"

python3 -m tools.heretic_to_onnx convert \
  --config /data/runtime-manifest.yaml \
  --work-dir /data/work \
  --output-dir /data/output \
  --force \
  --strict-onnx \
  --export-mode execute \
  --quantize-mode execute

python3 -m tools.heretic_to_onnx publish-hf \
  --config /data/runtime-manifest.yaml \
  --package-dir /data/output \
  --repo-id "${TARGET_REPO_ID}"
```

## 8. Updating Secrets After Deploy

Phala CLI supports sealed environment updates:

```bash
phala envs update -e .env
```

Or target a specific CVM:

```bash
phala envs update heretic-onnx-debug -e .env
```

Use this if you need to rotate `HF_TOKEN` or add more secrets without rebuilding the image.

## 9. Logs And Inspection

Container logs:

```bash
phala logs heretic-to-onnx -f
```

Serial console logs:

```bash
phala logs --serial -f
```

List containers:

```bash
phala ps
```

Get CVM metadata:

```bash
phala cvms get
phala runtime-config --json
phala cvms attestation
```

## 10. Cleanup

Pause without deleting:

```bash
phala cvms stop heretic-onnx-h100
```

Resume:

```bash
phala cvms start heretic-onnx-h100
```

Delete when done:

```bash
phala cvms delete heretic-onnx-h100
```

## Notes For Open Source Contributors

- keep `docker/phala-compose.yml` as the batch/default deployment path
- keep `docker/phala-compose.debug.yml` for interactive work
- do not commit real secrets
- commit `phala.toml` only if the linked CVM/project association is intentionally shared
- if you add training code later, document it as a separate Phala runtime path rather than silently overloading the ONNX export container
