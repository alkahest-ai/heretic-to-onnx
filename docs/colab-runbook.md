# Colab Runbook

This repo can run on Google Colab, but the realistic target is a paid GPU runtime with a high-memory profile.

Why:

- Colab's official FAQ says free resources are not guaranteed, usage limits fluctuate, GPU types vary over time, and free notebooks can run for at most 12 hours depending on availability.
- The FAQ also says paid Colab plans can expose a high-memory system profile and longer runtimes.
- For guaranteed dedicated hardware, Google points users to GCP Marketplace or Colab Enterprise.

Recommended runtime for this repo:

- GPU runtime enabled
- Paid Colab plan if available
- High-memory runtime if offered in the UI

Files added for Colab:

- `/Users/area/heretic/colab/heretic_to_onnx_full_pipeline.ipynb`

What the notebook does:

1. Mounts Google Drive.
2. Materializes this repo either from GitHub or from a zip stored in Drive.
3. Installs the Python dependencies needed for export and q4f16 quantization.
4. Writes a runtime manifest with your source model, base model, and target repo ID.
5. Runs `convert --export-mode execute --quantize-mode execute --strict-onnx`.
6. Zips the final package and copies it back to Drive.

Inputs you need to fill in before running:

- `REPO_URL` or `REPO_ZIP_IN_DRIVE`
- `SOURCE_MODEL_ID`
- `BASE_MODEL_ID`
- `TARGET_REPO_ID`
- `HF_TOKEN`

Expected outputs:

- Packaged ONNX repo under `build/heretic-ara-colab-package`
- Zip archive copied back to your chosen Drive folder

If Colab still fails:

- First failure mode is usually RAM or runtime termination.
- Second failure mode is missing or incompatible wheel versions.
- If that happens, move to a larger GPU VM or a dedicated GCP instance and reuse the same CLI commands from the notebook.
