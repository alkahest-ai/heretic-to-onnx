# Rally E2B Browser RP Plan

Date: 2026-05-04

## Scope

This lane starts after the Alkahest 0.8B/2B checkpoint. It applies the same promotion discipline that selected Alkahest 2B RP A100/B75, but only to Gemma 4 E2B Rally. E4B stays parked.

## Target Repos

Use `HF_OWNER=thomasjvu` for private app-facing recovery work unless a later release explicitly moves the public namespace.

| Target | Default repo variable | Purpose |
| --- | --- | --- |
| Direct Heretic full | `RALLY2_DIRECT_REPO=thomasjvu/rally-2b` | Base Heretic text + image browser package. |
| Direct Heretic text | `RALLY2_DIRECT_TEXT_REPO=thomasjvu/rally-2b-text` | Text-only browser package with Gemma4 q4f16 embed + decoder sessions only. |
| RP merged HF checkpoint | `RALLY2_MERGED_REPO=thomasjvu/rally-2b-rp-a100-b75-merged` | Full merged PyTorch/HF checkpoint for provenance and re-export. |
| RP full browser package | `RALLY2_TUNED_REPO=thomasjvu/rally-2b-rp` | A100/B75 text + image browser package. |
| RP text browser package | `RALLY2_TUNED_TEXT_REPO=thomasjvu/rally-2b-rp-text` | A100/B75 text-only browser package. |

## Current Kaggle Status

As of 2026-05-04, the active Rally/Gemma E2B pass is staying on Kaggle instead of Phala:

| Artifact | Status | Notes |
| --- | --- | --- |
| Two-stage SFT | Complete | `alkahestai/rally-e2b-two-stage-sft-t4` completed Stage A and Stage B on Kaggle T4. |
| Direct Heretic text | Uploaded | `thomasjvu/rally-2b-text`, HF commit `0ed67b21d4a6aa23614451587dd0e48a06f93dc1`, package report `ok: true`. |
| RP A100/B75 text | Uploaded | `thomasjvu/rally-2b-rp-text`, HF commit `cf64df4088314cc96a98786de1a0a963bc87e1d4`, package report `ok: true`. |
| RP A100/B75 merged checkpoint | Uploaded | `thomasjvu/rally-2b-rp-a100-b75-merged`, HF commit `3f2f180e1abea16d236e43e79b1e8454a1a5f168`; `scaled_lora_merge.json` verifies `ok: true`, scale `0.75`, 148 applied LoRA targets. |
| Full text+image browser packages | Blocked on Kaggle resources | T4 export OOMed during Gemma4 vision export; CPU export avoided VRAM but raw full ONNX intermediates exceeded the persistent Kaggle disk budget. Text-only packages are the current browser-ready artifacts. |

## Training Shape

The `rally` mode in `scripts/phala_gpu_tee_oneclick.sh` now uses the Alkahest v8 two-stage SFT rows:

- Stage A: instruction and exact-format adherence.
- Stage B: boundary-dominant RP improvement rows.
- Selected first candidate: A100/B75, matching the passing Alkahest 2B browser result.
- Promotion still requires browser smoke and RP scorecard win over direct Rally E2B before picker exposure.

The legacy one-stage Rally tune is still available as `rally-legacy` for diagnostics.

## Export Shape

Two Gemma4 E2B text-only manifests were added:

- `configs/heretic-to-onnx.gemma4-e2b-heretic-ara-text.yaml`
- `configs/heretic-to-onnx.gemma4-e2b-rp-text.yaml`

Gemma4 export contract generation now omits `vision_encoder` and `audio_encoder` sessions when the manifest is text-only. Full packages still use the text + image manifest and keep `vision_encoder_q4f16.*`.

## Kaggle Execution

Use Kaggle instead of Phala for the active lane. The workflow mirrors the Alkahest notebooks:

1. push and run `kaggle/rally_e2b_two_stage_sft`
2. when it completes, run `kaggle/rally_e2b_two_stage_export`, which consumes the training kernel as a source

Kernel IDs:

- `alkahestai/rally-e2b-two-stage-sft-t4`
- `alkahestai/rally-e2b-browser-export`
- `alkahestai/rally-e2b-rp-merged-upload`

CLI launch:

```bash
kaggle kernels push -p kaggle/rally_e2b_two_stage_sft --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle/rally_e2b_two_stage_export --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle/rally_e2b_rp_merged_upload
```

The two-kernel workflow performs:

1. direct Heretic full export and upload
2. direct Heretic text-only export and upload
3. two-stage RP Stage A training
4. two-stage RP Stage B training
5. A100/B75 scaled merge
6. merged checkpoint upload
7. RP full export and upload
8. RP text-only export and upload

The equivalent Phala command remains available as a fallback, but it is not the active path for this lane.

The upload-only merged checkpoint kernel is retained for future Kaggle-side recovery, but version 2 could not read the `HF_TOKEN` Kaggle secret in this session. The completed checkpoint was therefore recovered by streaming the Kaggle output to local disk with resumable download support, uploading it to Hugging Face, and deleting the local staging copy afterward.

## Promotion Gate

Do not add Rally presets to the default app picker until all of these are true:

- direct full and direct text browser smoke pass
- RP full and RP text browser smoke pass
- RP scorecard passes the minor-boundary gate
- RP total is at least `0.70`
- RP beats direct Rally E2B by at least `0.05`
- image smoke passes for the full packages

Until then, Rally repos are URL-override diagnostics only.
