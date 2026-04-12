# Rally 2B Training Plan

This is the first concrete roleplay tuning target after the direct ONNX conversions.

## Recommendation

Start the paid H200 run as **Jupyter Notebook (PyTorch)**, not custom container configuration.

Reason:

- lowest setup friction
- easy terminal access
- easier dataset iteration
- easier manual validation during the 24-hour minimum billing window

Use custom container configuration later if you want a hardened repeatable pipeline after the first successful tuning run.

## Target

- public name: `alkahest/rally-2b-rp`
- base checkpoint for the first tune: `p-e-w/gemma-4-E2B-it-heretic-ara`
- direct conversion artifact before tuning: `alkahest/rally-2b`

## Dataset Size Recommendation

For the first useful run, do **not** try to invent a million-example corpus.

Recommended first milestone:

- `5,000` to `12,000` conversations
- average `6` to `12` turns
- roughly `2M` to `6M` tokens after formatting

That is large enough to materially move style and behavior, but small enough to curate.

## Dataset Mix

First practical target:

- `2,000` to `4,000` synthetic original Alkahest conversations
- `1,000` to `3,000` cleaned Apache bootstrap conversations
- a small hand-reviewed validation set, around `200` to `500` rows

The first paid GPU window should **not** be spent generating a perfect final corpus. It should be spent proving the loop:

1. prepare data
2. tune
3. merge
4. convert
5. package
6. validate in browser

## Synthetic Data Rules

The synthetic sexy roleplay data should be:

- adult-only
- consensual
- emotionally responsive
- sexy, flirty, or intimate
- original in characters and settings

It should **not** rely on:

- copyrighted characters
- age ambiguity
- coercion or force
- exploitative or non-consensual framing

## Current Scaffold

The dataset scaffold now lives in:

- `/Users/area/heretic/data/roleplay_v1/README.md`
- `/Users/area/heretic/data/roleplay_v1/personas.yaml`
- `/Users/area/heretic/data/roleplay_v1/scenes.yaml`
- `/Users/area/heretic/data/roleplay_v1/style-rules.md`
- `/Users/area/heretic/data/roleplay_v1/seed_conversations.jsonl`
- `/Users/area/heretic/data/roleplay_v1/generator_prompt.md`
- `/Users/area/heretic/data/roleplay_v1/generated/README.md`
- `/Users/area/heretic/scripts/prepare_roleplay_dataset.py`
- `/Users/area/heretic/scripts/render_roleplay_prompt_pack.py`
- `/Users/area/heretic/scripts/synthesize_roleplay_batch.py`
- `/Users/area/heretic/scripts/build_roleplay_training_corpus.py`
- `/Users/area/heretic/scripts/train_rally_unsloth.py`

The current synthetic geometry is:

- `6` personas
- `10` scenes
- `240` persona/scene/lane combinations
- `25` variants per combination
- `6,000` generated rows in the first large batch

## Immediate Local Prep

Split the seed set:

```bash
python3 /Users/area/heretic/scripts/prepare_roleplay_dataset.py
```

That will write:

- `/Users/area/heretic/data/roleplay_v1/splits/train.jsonl`
- `/Users/area/heretic/data/roleplay_v1/splits/val.jsonl`

Render a synthetic prompt pack:

```bash
python3 /Users/area/heretic/scripts/render_roleplay_prompt_pack.py --limit 240
```

That writes:

- `/Users/area/heretic/data/roleplay_v1/prompt_pack.jsonl`

Build the unified training corpus after you have reviewed generated rows:

```bash
python3 /Users/area/heretic/scripts/build_roleplay_training_corpus.py
python3 /Users/area/heretic/scripts/prepare_roleplay_dataset.py \
  --input /Users/area/heretic/data/roleplay_v1/corpus.jsonl
```

Train with Unsloth:

```bash
python3 /Users/area/heretic/scripts/train_rally_unsloth.py --save-merged
```

For the first full synthetic pass:

```bash
python3 /Users/area/heretic/scripts/synthesize_roleplay_batch.py \
  --variants 25 \
  --seed 111 \
  --id-prefix bulk25 \
  --output /Users/area/heretic/data/roleplay_v1/generated/batch-0002.jsonl
```

## H200 Work Sequence

Recommended order for the first paid window:

1. run direct conversion for `rally-2b`
2. run direct conversion for `rally-4b`
3. run direct conversion for `sheena-4b`
4. expand `roleplay_v1`
5. fine-tune `rally-2b-rp`
6. merge the tuned checkpoint
7. convert the tuned checkpoint to ONNX
