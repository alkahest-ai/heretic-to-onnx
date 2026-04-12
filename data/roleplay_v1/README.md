# roleplay_v1

This folder is the archived first-pass scaffold for the roleplay dataset.

Status:

- archived prototype
- still useful for salvage and comparison
- no longer the canonical RP training backbone

Use `data/roleplay_v2/` for the active spreadsheet-first review flow.

Do not generate new training batches from `roleplay_v1`.

Use it only for:

- salvage
- comparison
- prototype archaeology

The active commands now live on the `roleplay_v2` path:

- `python3 scripts/synthesize_roleplay_batch.py`
- `python3 scripts/jsonl_to_review_table.py`
- `python3 scripts/review_table_to_jsonl.py`
- `python3 scripts/build_roleplay_training_corpus.py`
- `python3 scripts/prepare_roleplay_dataset.py`
