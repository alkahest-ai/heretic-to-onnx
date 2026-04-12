# Roleplay Data Flow

The final training data is still plain multi-turn chat JSONL with `system`, `user`, and `assistant` roles.

What changed is the editing and promotion workflow.

## Canonical Dataset

The active dataset is `roleplay_v2`.

Its stages are:

1. `generated_raw/`
2. `review_table/`
3. `approved_jsonl/`
4. `splits/`

Only stage 3 and stage 4 feed training.

## Editable Format

Humans now edit a spreadsheet-style CSV/TSV review table, not the main JSONL corpus.

Required columns:

- `conversation_id`
- `turn_index`
- `role`
- `persona_id`
- `scene_id`
- `lane`
- `content`
- `tags`
- `status`
- `rewrite_notes`

Optional columns:

- `keep`
- `quality_score`
- `repetition_flag`
- `needs_rewrite`
- `approved_by`

## Actual Workflow

1. generate a small raw batch
2. export it to TSV
3. review and rewrite in Sheets / Excel / Numbers
4. mark approved rows
5. compile approved rows back to JSONL
6. lint the approved batch
7. merge gold + approved rows into the corpus
8. split train/val
9. train Unsloth on that approved corpus

## Quick Checks

Use these before a real tune:

```bash
python3 -m unittest tests.test_roleplay_dataset_v2
python3 scripts/lint_roleplay_dataset.py --input data/roleplay_v2/review_table/batch-0001.tsv --approved-only
python3 scripts/lint_roleplay_dataset.py --input data/roleplay_v2/approved_jsonl
```

## Primary Files

- `/Users/area/heretic/data/roleplay_v2/README.md`
- `/Users/area/heretic/data/roleplay_v2/personas.yaml`
- `/Users/area/heretic/data/roleplay_v2/scenes.yaml`
- `/Users/area/heretic/data/roleplay_v2/variation_axes.yaml`
- `/Users/area/heretic/data/roleplay_v2/gold/seed_conversations.jsonl`

## Primary Scripts

- `/Users/area/heretic/scripts/synthesize_roleplay_batch.py`
- `/Users/area/heretic/scripts/jsonl_to_review_table.py`
- `/Users/area/heretic/scripts/review_table_to_jsonl.py`
- `/Users/area/heretic/scripts/lint_roleplay_dataset.py`
- `/Users/area/heretic/scripts/build_roleplay_training_corpus.py`
- `/Users/area/heretic/scripts/prepare_roleplay_dataset.py`
- `/Users/area/heretic/scripts/train_rally_unsloth.py`

## `roleplay_v1`

`roleplay_v1` is now prototype material and archive context.

It is still useful for salvage and comparison, but it is not the default RP training backbone anymore.
