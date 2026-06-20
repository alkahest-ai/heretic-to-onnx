# Rally E4B / 12B Closeout — June 2026

Date: 2026-06-20

## Promotion matrix

Gate validation and app/browser delivery are separate milestones.

| Lane | Minor gate | False-refusal | Inference merge path | Browser / serve | App picker |
| --- | --- | --- | --- | --- | --- |
| Rally E2B RP text | 1.0 | passed | fp16 disk merge | Chrome WebGPU smoke passed | Visible |
| Rally E2B full | n/a | n/a | n/a | Fails (`GatherBlockQuantized`) | Hidden |
| Rally E4B RP | 1.0 | ~5/100 | CPU single-pass disk merge | Export lane ready; smoke pending | Hidden until smoke |
| Rally E4B direct | 0.0 minor | n/a | n/a | Export lane ready; smoke pending | Hidden until smoke |
| Rally 12B RP | 1.0 | within gate | pre-merged HF + 4-bit | vLLM scaffold + smoke script | vLLM only (no WebGPU) |

Root-cause write-up: [rally-gemma-minor-gate-postmortem.md](rally-gemma-minor-gate-postmortem.md).

## Canonical Kaggle kernels

| Purpose | Kernel ID |
| --- | --- |
| E4B SFT | `thomasjvu/rally-e4b-sft-jun14v10` |
| E4B compare (gate) | `thomasjvu/rally-e4b-compare-jun14v10` |
| E4B export prep | `thomasjvu/rally-e4b-export-prep` |
| E4B direct text export | `thomasjvu/rally-e4b-direct-text-export` |
| E4B RP merged upload | `thomasjvu/rally-e4b-rp-merged-upload` |
| E4B RP text export | `thomasjvu/rally-e4b-rp-text-export-jun20` |
| 12B SFT | `thomasjvu/rally-12b-two-stage-sft-a100` |
| 12B merged upload | `thomasjvu/rally-12b-rp-merged-upload` |
| 12B scorecard | `thomasjvu/rally-12b-scorecard-jun15v8` |

Push E4B export lane:

```bash
bash scripts/kaggle_push_rally_e4b_export.sh
```

## HF targets

| Repo | Role |
| --- | --- |
| `thomasjvu/rally-4b-text` | Direct Heretic E4B text browser package (post-export) |
| `thomasjvu/rally-4b-rp-text` | RP A100/B75 E4B text browser package (post-export) |
| `thomasjvu/rally-4b-rp-source-merged` | Private merged provenance checkpoint |
| `thomasjvu/rally-12b-rp-a100-b75-merged` | Private 12B merged checkpoint for vLLM |

## 12B vLLM serve

```bash
bash scripts/serve_vllm_gemma4.sh
# separate terminal after server is up:
bash scripts/vllm_smoke_gemma4_12b.sh
```

Config: `configs/vllm-gemma4-12b-rp.yaml`. Requires GPU + Hugging Face access to merged weights.

## Remaining before E4B picker exposure

1. Push export kernels and wait for completion.
2. Set `RALLY_UPLOAD=1` on export notebooks (or upload locally from Kaggle output).
3. Run Chrome 148+ Metal/WebGPU browser smoke via `browser-chat/smoke-runner.js`.
4. Only then treat `rally-4b-text` / `rally-4b-rp-text` presets as production-ready.

## E2B full multimodal (parked)

`thomasjvu/rally-2b` and `thomasjvu/rally-2b-rp` remain private experimental artifacts. Full-package WebGPU generation still fails; text-only Rally presets stay the default Gemma browser offering.