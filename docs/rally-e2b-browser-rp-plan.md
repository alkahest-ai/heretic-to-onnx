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

As of 2026-05-08, the active Rally/Gemma E2B pass is staying on Kaggle instead of Phala. Local browser smoke is paused because cold-loading the Rally q4f16 ONNX packages repeatedly stressed the desktop WebGPU/browser stack enough to crash the machine.

| Artifact | Status | Notes |
| --- | --- | --- |
| Two-stage SFT | Complete | `alkahestai/rally-e2b-two-stage-sft-t4` completed Stage A and Stage B on Kaggle T4. |
| Direct Heretic text | Re-exported, validated, uploaded | Kaggle export completed with upload disabled, then local HF upload published the fixed opset 21 q4f16 package at `thomasjvu/rally-2b-text`, HF commit `7451f62519eb7932266b3ec0d361f5937bf325c4`. Package validation is clean; browser promotion is still blocked by local WebGPU stability. |
| RP A100/B75 text | Re-exported, validated, staged | Kaggle export completed with upload disabled. Local upload to `thomasjvu/rally-2b-rp-text` hit the private HF storage quota, so the fixed package is staged privately at `alkahest-ai/rally-2b-rp-text`, HF commit `170b70033163f747bf7976625a79591980013f7c`. The old `thomasjvu/rally-2b-rp-text` revision remains legacy opset 17 and is not promotable. |
| RP A100/B75 merged checkpoint | Uploaded | `thomasjvu/rally-2b-rp-a100-b75-merged`, HF commit `3f2f180e1abea16d236e43e79b1e8454a1a5f168`; `scaled_lora_merge.json` verifies `ok: true`, scale `0.75`, 148 applied LoRA targets. |
| Full text+image browser packages | Blocked on Kaggle resources | T4 export OOMed during Gemma4 vision export; CPU export avoided VRAM but raw full ONNX intermediates exceeded the persistent Kaggle disk budget. Text-only packages are the current browser-ready artifacts. |
| Kaggle scorecard-only lane | Added | `kaggle/rally_e2b_scorecard` runs the direct-vs-RP scorecard on Kaggle from the completed SFT kernel source, without requiring local browser/WebGPU load. This is the next validation step before another desktop browser attempt. |

The confirmed browser failure on the old pinned diagnostic scorecard was:

```text
Provider type for MatMulNBits node with name '/language_model/per_layer_model_projection/MatMul_Q4' is not set.
```

Local graph inspection showed the failed Rally text decoders were generated as ONNX opset 17 / IR 8, while the Lisper-trained and ONNX Community Gemma4 q4f16 reference decoders use opset 21 with `com.microsoft` custom ops. The current fixed text packages transplant the optimized reference-style decoder, preserve the `com.microsoft` custom ops, and pass strict package validation. Browser scorecard promotion is still pending because the desktop crashed during local WebGPU session initialization after the direct package completed download.

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

Gemma4 Rally export now defaults to ONNX opset 21, matching the Lisper/reference q4f16 WebGPU contract. Package validation also inspects Gemma4 q4f16 text graphs when ONNX is available and rejects legacy decoder graphs that do not import `com.microsoft` opset 1 or that omit the required custom ops:

- decoder: `MatMulNBits`
- embed: `GatherBlockQuantized`

The next exporter path is a reference-template transplant instead of another generic Torch trace. The `optimize-gemma4-text-package` command takes the packaged text repo, a local merged Gemma4 checkpoint, and a reference Gemma4 q4f16 ONNX template, then replaces the decoder with the optimized WebGPU graph and re-quantized source weights. The Rally Kaggle export notebook now runs that transplant after text export and before upload/validation.

## Kaggle Execution

Use Kaggle instead of Phala for the active lane. The workflow mirrors the Alkahest notebooks:

1. push and run `kaggle/rally_e2b_two_stage_sft`
2. when it completes, run `kaggle/rally_e2b_two_stage_export`, which consumes the training kernel as a source

Kernel IDs:

- `alkahestai/rally-e2b-two-stage-sft-t4`
- `alkahestai/rally-e2b-export-prep`
- `alkahestai/rally-e2b-browser-export`
- `alkahestai/rally-e2b-rp-text-export`
- `alkahestai/rally-e2b-rp-merged-upload`
- `alkahestai/rally-e2b-scorecard`

CLI launch:

```bash
kaggle kernels push -p kaggle/rally_e2b_two_stage_sft --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle/rally_e2b_export_prep
kaggle kernels push -p kaggle/rally_e2b_two_stage_export
kaggle kernels push -p kaggle/rally_e2b_rp_text_export
kaggle kernels push -p kaggle/rally_e2b_rp_merged_upload
kaggle kernels push -p kaggle/rally_e2b_scorecard --accelerator NvidiaTeslaT4
```

Current recovery shape:

1. `rally_e2b_export_prep` stages the exact `heretic-to-onnx` branch checkout plus the optimized Gemma4 q4f16 template into a Kaggle kernel source.
2. `rally_e2b_two_stage_export` consumes the prep source plus the SFT kernel source and runs only the direct text export lane.
3. `rally_e2b_rp_text_export` consumes the same prep source plus the SFT kernel source and runs only the RP text export lane.
4. `rally_e2b_scorecard` consumes the SFT kernel source, merges the A100/B75 RP checkpoint, scores direct versus RP on Kaggle, and redacts the raw minor-boundary response from the report.
5. Full text+image export remains parked until text-only browser packaging is proven inside Kaggle time limits.

The legacy monolithic workflow performed:

1. direct Heretic full export and upload
2. direct Heretic text-only export and upload
3. two-stage RP Stage A training, only if the existing merged checkpoint is not reused
4. two-stage RP Stage B training, only if the existing merged checkpoint is not reused
5. A100/B75 scaled merge
6. merged checkpoint upload
7. RP full export and upload
8. RP text-only export and upload

For the immediate recovery pass, run export/upload only against the existing direct source and A100/B75 merged checkpoint. Do not start a new Rally RP sweep until the text-only browser runtime can load and complete the scorecard.

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

## Next Recovery Pass

1. Push the scorecard-only Kaggle kernel and run it on T4 against the completed two-stage SFT source.
2. Use the Kaggle scorecard result to decide whether the current A100/B75 RP quality is worth another browser attempt. It must reach `0.70`, pass the minor-boundary gate, avoid adult false-refusal, and beat direct Rally by at least `0.05`.
3. Keep the fixed text HF revisions pinned: direct `thomasjvu/rally-2b-text@7451f62519eb7932266b3ec0d361f5937bf325c4`; RP staging `alkahest-ai/rally-2b-rp-text@170b70033163f747bf7976625a79591980013f7c`.
4. Do not use the old `thomasjvu/rally-2b-rp-text` revision for smoke or promotion until the HF quota issue is resolved and the fixed package replaces it.
5. If Kaggle scorecard fails quality, run a small Rally RP sweep on Kaggle before doing any more desktop browser smoke.
6. If Kaggle scorecard passes, retry browser smoke in a clean isolated browser profile, one model at a time, with the worker-backed scorecard runner.
7. Only after the text-only scorecard passes, resume full text+image package export.
