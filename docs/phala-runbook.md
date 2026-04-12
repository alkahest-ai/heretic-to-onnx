# Phala H100 Runbook

This repo now includes a Phala-oriented container path for running the full Gemma 4 export, q4f16 quantization, package assembly, and Hugging Face upload in one job.

For the full CLI/operator workflow, including `phala` install, login, deploy, SSH, debug mode, and rerunning from inside the CVM, see `/Users/area/heretic/docs/phala-cli-operator-guide.md`.
For the paid go/no-go checklist, including whether to use H200 on-demand GPU TEE or the standard CVM path, see `/Users/area/heretic/docs/phala-preflight-checklist.md`.

## Files

- Container image: `/Users/area/heretic/docker/Dockerfile.phala`
- Runtime entrypoint: `/Users/area/heretic/docker/phala-entrypoint.sh`
- Compose template for Phala: `/Users/area/heretic/docker/phala-compose.yml`
- Debug compose template for Phala: `/Users/area/heretic/docker/phala-compose.debug.yml`
- Compose env template: `/Users/area/heretic/docker/phala.env.example`
- Build helper: `/Users/area/heretic/scripts/build_phala_image.sh`
- Env writer: `/Users/area/heretic/scripts/write_phala_env.sh`
- CLI wrapper: `/Users/area/heretic/scripts/phala_cli.sh`
- CLI doctor: `/Users/area/heretic/scripts/phala_doctor.sh`
- Deploy helper: `/Users/area/heretic/scripts/phala_deploy.sh`

## What The Container Does

On startup, the container:

1. renders a runtime manifest with the source model, base model, and target Hugging Face repo you provide
2. runs `convert --export-mode execute --quantize-mode execute --strict-onnx`
3. creates the target Hugging Face model repo if needed
4. uploads the packaged ONNX folder with `hf upload-large-folder`

The packaged repo is left on the persistent `/data/output` volume even after it is uploaded.

## Required Secrets

Set these in Phala encrypted secrets:

- `HF_TOKEN`

That token needs:

- read access to `p-e-w/gemma-4-E2B-it-heretic-ara`
- read access to `google/gemma-4-E2B-it`
- write access to your target ONNX repo

Also make sure the Hugging Face account behind the token has already accepted the Gemma model terms if the base model is gated.

## Recommended Environment Values

Set these env vars in the Phala deployment:

- `PHALA_IMAGE`: the published container image for this repo
- `SOURCE_MODEL_ID`: `p-e-w/gemma-4-E2B-it-heretic-ara`
- `BASE_MODEL_ID`: `google/gemma-4-E2B-it`
- `TARGET_REPO_ID`: `lightnolimit/gemma-4-E2B-it-heretic-ara-ONNX`

You can generate a starter env file locally with:

```bash
/Users/area/heretic/scripts/write_phala_env.sh /Users/area/heretic/build/phala.env your-registry/heretic-to-onnx:latest
```

The compose template already sets:

- `WORK_DIR=/data/work`
- `OUTPUT_DIR=/data/output`
- `EXPORT_MODE=execute`
- `QUANTIZE_MODE=execute`
- `STRICT_ONNX=1`
- `UPLOAD_TO_HF=1`
- `MIN_GPU_MEMORY_MB=70000`

## Build And Push The Image

Build locally:

```bash
/Users/area/heretic/scripts/build_phala_image.sh your-registry/heretic-to-onnx:latest
```

Push it to a registry that Phala can pull from:

```bash
/Users/area/heretic/scripts/build_phala_image.sh your-registry/heretic-to-onnx:latest --push
```

If the registry is private, configure the registry credentials in Phala before deployment.

## Deploy On Phala

Use `/Users/area/heretic/docker/phala-compose.yml` as the deployment compose file.
Use `/Users/area/heretic/build/phala.env` or `/Users/area/heretic/docker/phala.env.example` as the env source when filling in the deployment values.

The important operational points are:

- keep `/data` on a persistent volume so export artifacts survive restart
- keep `HF_TOKEN` in encrypted secrets, not hardcoded in the compose file
- use an H100 class GPU for the job
- the entrypoint checks GPU memory at startup and fails if the runtime is below the configured minimum

## Hugging Face Publish Flow

The entrypoint uses:

```bash
python3 -m tools.heretic_to_onnx publish-hf \
  --config /data/runtime-manifest.yaml \
  --package-dir /data/output \
  --repo-id lightnolimit/gemma-4-E2B-it-heretic-ara-ONNX
```

That stage also ensures the package contains a minimal `README.md` model card before upload.
