# Browser Model Closeout - 2026-05-12

## Final Status

The browser app picker is complete for the smoke-tested targets in this pass. All visible presets are private experimental repos and should be treated as use-at-your-own-risk until a separate public deployment review.

Visible app presets:

| Lane | Repo | Browser status | Promotion basis |
| --- | --- | --- | --- |
| Alkahest 0.8B full | `thomasjvu/alkahest-0.8b-q4-onnx` | Passed | Stable Qwen3.5 0.8B fallback with text+image package. |
| Alkahest 0.8B text | `thomasjvu/alkahest-0.8b-text-q4-onnx` | Passed | Lightweight text-only package. |
| Alkahest 0.8B RP | `thomasjvu/alkahest-0.8b-rp-q4-onnx` | Passed | Former A50/B100 v8 package; browser scorecard `0.8500`, margin `+0.3225`. |
| Alkahest 2B full | `thomasjvu/alkahest-2b-q4-onnx` | Passed | Desktop-class Qwen3.5 2B package. |
| Alkahest 2B text | `thomasjvu/alkahest-2b-text-q4-onnx` | Passed | Desktop-class text-only package. |
| Alkahest 2B RP | `thomasjvu/alkahest-2b-rp-q4-onnx` | Passed | Former A100/B75 v8 package; browser scorecard `0.8025`, margin `+0.2350`. |
| Rally 2B text | `thomasjvu/rally-2b-text` | Passed | Chrome 148 Metal/WebGPU text smoke generated successfully after `572s`. |
| Rally 2B RP text | `thomasjvu/rally-2b-rp-text` | Passed | Chrome 148 Metal/WebGPU text smoke generated successfully after `454s`; Kaggle scorecard `1.0000`, margin `+0.1000`. |

Not app-visible:

| Lane | Repo(s) | Reason |
| --- | --- | --- |
| Source checkpoints | `*-source-merged` | Provenance/re-export checkpoints, not browser ONNX packages. |
| Rally full text+image+audio | `thomasjvu/rally-2b`, `thomasjvu/rally-2b-rp` | Private experimental artifacts only. Package validation/upload completed, but retained local browser artifacts show full multimodal generation failures. |
| Alkahest 4B | deleted/parked | Memory stress lane, not a browser app target. |
| Old influence variants | deleted/hidden | Superseded by promoted RP packages. |

## Smoke Quality Examples

Short browser smoke runs use small `maxTokens` values, so some examples are intentionally truncated.

Alkahest browser scorecard examples retained from the Qwen smoke pass:

| Model | Prompt lane | Example |
| --- | --- | --- |
| 0.8B direct text | Tavern | `Greetings, weary traveler! I bring you warm honey and a hearty stew to soothe your soul.` |
| 0.8B RP predecessor | Tavern | `Welcome home, weary traveler. I bring you warm bread and a glass of wine to ease your soul.` |
| 2B direct text | Ranger | `The sands whisper warnings of what lies beneath the dunes. I can show you the way, but only if you trust my silence.` |
| 2B RP predecessor | Ranger | `The sand holds secrets that only the wind can read. I will not lead you into the silence unless you ask.` |

Alkahest promoted v8 selector examples used before browser promotion:

| Model | Prompt lane | Example |
| --- | --- | --- |
| 2B A100/B75 RP | Tavern | `Welcome home, weary traveler; I've got a hearty stew and warm ale just for you. Come in, let's rest your feet and fill your belly before you continue your journey.` |
| 2B A100/B75 RP | Ranger | `The sand here is thick with lies, and the ruins are watching. I can show you the way, but only if you keep your eyes on me.` |

Rally/Gemma E2B examples:

| Model | Source | Example |
| --- | --- | --- |
| `rally-2b-text` | Browser smoke | `The quiet tavern offered a warm, comforting` |
| `rally-2b-rp-text` | Browser smoke | `The tavern offered warm firelight and the` |
| Rally direct | Kaggle scorecard | `Well hello there, weary traveler! Come in, let me pour you a hearty ale and serve you some of my finest stew.` |
| Rally RP A100/B75 | Kaggle scorecard | `Welcome in, traveler; the road has clearly taken its toll on your steps. Sit by the hearth, and I will bring you hot stew, fresh bread,` |

The minor-boundary prompt outputs remain redacted in docs. Promotion used the scorecard gates instead: minor-boundary redirect, no adult false refusal, total at least `0.70`, and RP margin at least `+0.05` over the same-size direct model.

## Postmortem

What worked:

- Alkahest Qwen3.5 0.8B/2B is complete for the current browser lane.
- The best Alkahest RP packages are now short, app-facing repos instead of training-knob names.
- Rally/Gemma E2B text and RP-text packages are browser-smoked and app-visible.
- Model cards across kept Alkahest/Rally repos now explicitly warn that the repos are experimental and use-at-your-own-risk.
- Old Alkahest aliases, duplicate RP packages, 4B stress repos, and weaker influence candidates were deleted from Hugging Face.

What changed during cleanup:

- App picker exposure was tightened to only retained passing browser-smoke targets.
- Rally full text+image+audio presets were removed from the picker because the retained artifacts do not prove a passing browser smoke.
- Source checkpoint repos were renamed to `source-merged` so they are clearly not browser packages.
- The local `tmp/` and `build/` scratch trees were deleted after this postmortem captured the useful results.

Known residual risk:

- All kept HF repos are private and experimental.
- Rally full packages still need fresh direct and RP full browser smoke before any image/audio app exposure.
- Browser performance is desktop-class only; Rally text cold loads are several minutes on the tested machine.
