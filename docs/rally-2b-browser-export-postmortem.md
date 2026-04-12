# Rally 2B Browser Export Postmortem

Date context: April 12, 2026.

This is the write-up for the first successful direct `rally-2b` browser package built from the Gemma 4 Heretic lane and published to Hugging Face.

Final published repo:

- [thomasjvu/rally-2b](https://huggingface.co/thomasjvu/rally-2b)

## What Actually Shipped

The working package is a **Gemma 4 E2B Heretic browser export** with these ONNX sessions:

- `vision_encoder_q4f16.onnx`
- `embed_tokens_q4f16.onnx`
- `decoder_model_merged_q4f16.onnx`

Important limitation:

- the current shipped package is **text + image only**
- audio export is disabled for now in the Gemma 4 ONNX lane

## Commands That Worked

On the Phala H200 box, the reliable path was:

```bash
bash scripts/phala_gpu_tee_bootstrap_export.sh
```

Then:

```bash
export HF_OWNER=thomasjvu
export HF_PRIVATE=0
export RALLY2_DIRECT_REPO=thomasjvu/rally-2b
PYTHON_BIN="$HOME/work/heretic-to-onnx/.venvs/phala-export/bin/python" \
bash scripts/phala_run_rally_2b_direct.sh
```

That path completed:

1. manifest rendering
2. export
3. q4f16 quantization
4. package assembly
5. package validation
6. Hugging Face upload

## Why An Isolated Export Venv Was Needed

The default Jupyter environment on the Phala GPU TEE box was not a stable base for the Gemma 4 ONNX path.

The working solution was an isolated export environment created by:

- `scripts/phala_gpu_tee_bootstrap_export.sh`

That environment pins the export stack around:

- `torch==2.7.0+cu128`
- `torchvision==0.22.0+cu128`
- `torchaudio==2.7.0+cu128`
- current ONNX / ONNX Runtime / onnxconverter-common
- current `transformers`

This matters because several of the Gemma 4 failures were version-sensitive and lived in the interaction between:

- PyTorch ONNX tracing
- Transformers masking helpers
- ONNX Runtime quantization

## What Broke On The Way

### 1. Gemma 4 processor and sample-generation mismatches

Early failures came from the Gemma 4 multimodal processor contract not matching the exporter assumptions.

The important fixes were:

- load Gemma 4 processors without forcing a tokenizer-only path
- accept Gemma 4 `image_position_ids` instead of assuming only `pixel_position_ids`
- fix multimodal sample generation for the encoder wrappers

Without those changes, the export runner could not even build correct image-side inputs.

### 2. Gemma 4 attention export could not stay on the default masking path

The next class of failures came from Gemma 4 masking and attention internals during ONNX tracing.

The practical fixes were:

- force the export lane onto eager attention
- patch or bypass the `sdpa_mask` path used by current Transformers
- replace version-fragile library hooks with local fallback logic in the generated runner
- inline bidirectional mask materialization for the vision path

This was necessary because the stock masking path kept falling into tracing or `vmap` failures that were stable in Python inference but not export-safe.

### 3. Gemma 4 audio export is not viable in the current ONNX pipeline

The biggest architectural blocker was audio.

The Gemma 4 audio tower eventually hit an ONNX export limitation around:

- `unfold`
- dynamic padding / block-context extraction

That was not a small compatibility bug. It was a real exporter limitation in the current PyTorch ONNX path for this model family.

The practical decision was:

- remove audio from the Gemma 4 browser manifest templates
- stop packaging an `audio_encoder`
- treat the direct Rally 2B browser package as **text + image only**

That is the right call for now. Trying to keep audio in the same lane would have blocked the whole deliverable.

## Vision Nuance

Vision did export successfully, but it required a more defensive wrapper than the original pipeline assumed.

The useful mental model is:

- Gemma 4 vision export is feasible
- Gemma 4 vision export is not “drop the HF model into `torch.onnx.export` and walk away”

The exporter needed to:

- normalize processor outputs
- stay off the fragile masking path
- materialize the mask locally when necessary

Once that was done, `vision_encoder_q4f16.onnx` became a stable part of the package.

## Decoder Nuance

The merged decoder export also needed model-specific handling.

The key fix was:

- call `self.model.model.language_model(...)` directly in the export wrapper
- do **not** call the full multimodal `self.model(...)` path

Why:

- the full multimodal `forward` path tries to reverse embeddings and reconstruct per-layer inputs
- that breaks when export is intentionally driven from `inputs_embeds`

After routing through the language model stack directly, the decoder export moved forward.

## Cache Nuance

Once the decoder wrapper was corrected, current Transformers expected a richer cache API than the original flat export cache shim exposed.

The generated `FlatGemma4Cache` needed to provide:

- `get_seq_length(...)`
- `get_mask_sizes(...)`
- `update(...)`

Without `get_mask_sizes(...)`, causal-mask construction in current Transformers failed during export.

## Quantization Nuances

Quantization worked, but two separate issues had to be fixed.

### Python literal emission

The generated quantize runners originally embedded the export contract as JSON directly into Python source.

That produced invalid Python like:

- `true`
- `false`

instead of:

- `True`
- `False`

The fix was to emit the contract as a Python literal with `repr(...)`, not raw JSON.

### q4 to q4f16 float16 conversion on large models

The quantize runner then hit a large-model failure in:

- `onnxconverter_common.float16.convert_float_to_float16(...)`

The issue was shape inference trying to serialize the large quantized model in memory.

The working fix was:

- call `convert_float_to_float16(..., disable_shape_infer=True)`

That was the difference between a failing quantizer and a successful q4f16 package.

## Publish Nuances

The main package upload succeeded, but the generated model card initially triggered this Hub warning:

- `YAML Metadata Warning: empty or missing yaml metadata in repo card`

The fix was straightforward:

- generate YAML front matter in the model card
- preserve existing card content and prepend metadata when missing

There was also an operational nuance on the H200 box:

- `hf upload-large-folder` internally calls `create_repo(..., exist_ok=True)`
- transient DNS failures on the VM could break that step even when the repo already existed

When that happened, the smaller targeted upload still worked:

```bash
hf upload \
  thomasjvu/rally-2b \
  /home/jovyan/work/heretic-to-onnx/build/phala_gpu_tee/packages/rally-2b-direct/README.md \
  README.md \
  --repo-type model
```

That is useful for card-only refreshes and metadata fixes.

## Warnings That Are Okay

The final successful export and quantize reports still recorded:

- `export runner produced stderr output`
- `quantize runner produced stderr output`

That is expected for this lane because PyTorch and ONNX export emit a lot of tracing warnings on stderr even when the run succeeds.

What matters is:

- the JSON report marks the stage `ok: true`
- the expected ONNX artifacts exist
- package validation passes

## Current State Of The Gemma 4 Browser Lane

What is proven now:

- direct Rally 2B Gemma 4 browser packaging works
- vision export works
- merged decoder export works
- q4f16 quantization works
- package validation works
- Hugging Face publish works

What is not yet part of the proven lane:

- Gemma 4 audio export

So the current supported contract for direct Rally 2B should be treated as:

- **text**
- **image**
- not audio

## What To Reuse For Rally 4B

For `rally-4b`, reuse the same shape of workflow:

1. use the isolated export venv
2. use the generated Gemma 4 export and quantize runners
3. keep the package as text + image unless audio is solved separately
4. expect the same decoder/cache/masking codepath to matter
5. keep q4f16 conversion on `disable_shape_infer=True`

The main expected differences for `rally-4b` are operational, not conceptual:

- larger raw ONNX artifacts
- longer export and quantize time
- more pressure on disk, memory, and upload time

## Short Version

The reason Rally 2B finally worked is not one magic fix. It required tightening the whole lane:

- correct processor handling
- defensive vision input handling
- local masking fallbacks
- audio removal from the Gemma browser contract
- direct decoder export through the language-model stack
- a cache shim that matches current Transformers expectations
- q4f16 conversion that avoids large-model shape inference failure
- model cards with proper Hugging Face YAML metadata

That is now the baseline for the Rally 4B pass.
