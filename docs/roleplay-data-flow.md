# Roleplay Data Flow

Yes: the **actual training dataset** is supposed to be in chat messages.

This repo currently has **two layers** of data:

## 1. Training rows

These are the real SFT examples.

They look like:

```json
{
  "id": "seed-0001",
  "tags": ["adult", "consensual", "flirt"],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Files in this category:

- `/Users/area/heretic/data/roleplay_v1/seed_conversations.jsonl`
- `/Users/area/heretic/data/roleplay_v1/generated/*.jsonl`
- `/Users/area/heretic/data/roleplay_v1/corpus.jsonl`
- `/Users/area/heretic/data/roleplay_v1/splits/train.jsonl`
- `/Users/area/heretic/data/roleplay_v1/splits/val.jsonl`

## 2. Generation scaffolding

These are **not** the training rows themselves.

They exist to help generate more training rows.

Files in this category:

- `/Users/area/heretic/data/roleplay_v1/personas.yaml`
- `/Users/area/heretic/data/roleplay_v1/scenes.yaml`
- `/Users/area/heretic/data/roleplay_v1/style-rules.md`
- `/Users/area/heretic/data/roleplay_v1/generator_prompt.md`
- `/Users/area/heretic/data/roleplay_v1/prompt_pack.jsonl`

`prompt_pack.jsonl` is just a list of prompt jobs to feed into another model or workflow to generate more conversations.

## Intended Workflow

1. write or generate more rows in message format
2. save them into `generated/*.jsonl`
3. merge everything into one corpus
4. split train/val
5. train Unsloth on the resulting message dataset

## Current Synthetic Scale

The current scaffold is intentionally set up to generate a full first-pass dataset instead of a toy batch:

- `6` personas
- `10` scenes
- `240` persona/scene/lane combinations
- `25` variants per combination
- `6,000` generated rows in `batch-0002.jsonl`

That is enough to get the corpus into the right order of magnitude before manual editing.

## Scripts

Use:

- `/Users/area/heretic/scripts/render_roleplay_prompt_pack.py` to generate prompt jobs
- `/Users/area/heretic/scripts/synthesize_roleplay_batch.py` to generate bulk message-format conversation rows
- `/Users/area/heretic/scripts/build_roleplay_training_corpus.py` to merge real conversation files
- `/Users/area/heretic/scripts/prepare_roleplay_dataset.py` to validate/split one message file
- `/Users/area/heretic/scripts/train_rally_unsloth.py` to fine-tune `rally` or any other supported chat checkpoint by overriding `--model-name`
