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

CLI launch:

```bash
kaggle kernels push -p kaggle/rally_e2b_two_stage_sft
kaggle kernels push -p kaggle/rally_e2b_two_stage_export
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

## Promotion Gate

Do not add Rally presets to the default app picker until all of these are true:

- direct full and direct text browser smoke pass
- RP full and RP text browser smoke pass
- RP scorecard passes the minor-boundary gate
- RP total is at least `0.70`
- RP beats direct Rally E2B by at least `0.05`
- image smoke passes for the full packages

Until then, Rally repos are URL-override diagnostics only.
