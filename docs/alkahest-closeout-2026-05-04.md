# Alkahest 0.8B/2B Closeout - 2026-05-04

## Summary

The Alkahest Qwen 0.8B/2B browser lane is complete for this pass. The app picker exposes the direct 0.8B/2B browser packages, the promoted 0.8B RP v8 packages, and the promoted 2B RP v8 A100/B75 package. Older RP repos, the weaker 2B A100/B50 candidate, Rally/Gemma, and 4B remain hidden from the default picker.

Current checkpoint:

- Branch: `codex/kaggle-heretic-2b-run`
- Commit: `33a27b8 Promote passing 2B RP browser target`
- Server used for smoke: `http://localhost:4173/browser-chat/`

## Promoted App Targets

| Slot | Repo | Status |
| --- | --- | --- |
| 0.8B direct full | `thomasjvu/alkahest-0.8b-heretic-q4-onnx` | Visible stable fallback. |
| 0.8B direct text | `thomasjvu/alkahest-0.8b-heretic-q4-onnx-text` | Visible lightweight text target. |
| 0.8B RP | `thomasjvu/alkahest-0.8b-heretic-rp-sft-two-stage-a50-b100-q4-onnx` | Visible recommended 0.8B RP target. |
| 0.8B RP softer fallback | `thomasjvu/alkahest-0.8b-heretic-rp-sft-two-stage-a25-b100-q4-onnx` | Visible softer RP fallback. |
| 2B direct full | `thomasjvu/alkahest-2b-heretic-q4-onnx` | Visible desktop-class direct target. |
| 2B direct text | `thomasjvu/alkahest-2b-heretic-q4-onnx-text` | Visible desktop-class text target. |
| 2B RP | `thomasjvu/alkahest-2b-heretic-rp-sft-two-stage-a100-b75-q4-onnx` | Visible recommended 2B RP target. |

## 2B RP Decision

The 2B v8 selected packages split cleanly:

| Candidate | Image smoke | Browser scorecard | Decision |
| --- | --- | --- | --- |
| `a100-b50` | Passed | `0.5800`, minor gate failed, margin `+0.0125` | Keep hidden diagnostic-only. |
| `a100-b75` | Passed cold and warm | `0.8025`, minor gate passed, margin `+0.2350` | Promote as 2B RP app target. |

Raw minor-boundary responses remain redacted from docs and tracked files.

## Verification

Checks run before the promotion commit:

- `node --check` for `examples/browser-loader.mjs`, browser-chat runtime files, smoke runner, and scorecard runner.
- `python3 -m py_compile` for Kaggle/export/scorecard scripts.
- Targeted unittests for two-stage export, RP scorecard, Qwen transplant, package repo, and repo validation.
- `git diff --check`.

Browser validation:

- 2B A100/B75 image smoke passed.
- 2B A100/B75 warm-cache image smoke passed.
- 2B direct versus A100/B50 versus A100/B75 browser scorecard promoted A100/B75 only.

## Parked Work

- Rally/Gemma stays parked until this Alkahest checkpoint is merged or accepted.
- 4B remains a memory stress lane, not an app-default browser lane.
- The next active model lane should be Gemma E2B browser packaging. E4B stays out of scope.

## Cleanup Notes

Ignored temp artifacts are covered by `.gitignore` through `tmp/`. The remaining large ignored temp artifact after the 2B package-copy cleanup is `tmp/kaggle-alkahest-08b-two-stage-export-v8`, a local 0.8B Kaggle output snapshot containing reports/logs and a copied repo tree. It can be deleted after explicit local deletion confirmation because the promoted 0.8B results and final docs are already committed.
