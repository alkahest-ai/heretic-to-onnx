# roleplay_v2

`roleplay_v2` is the new canonical dataset flow for tuned `*-rp` models.

This dataset is:

- original-only
- adult-only
- consensual
- intimate, flirtatious, and emotionally specific
- spreadsheet-first for human review

Stages:

1. `generated_raw/`
2. `review_table/`
3. `approved_jsonl/`
4. `splits/`

Rules:

- raw batches are never trained directly
- review happens in CSV/TSV form
- only approved conversations compile into the training corpus
- the small `gold/` seed set anchors style and tone
- JSONL is the compiled artifact, not the primary editing surface

Primary scripts:

- `python3 scripts/synthesize_roleplay_batch.py`
- `python3 scripts/jsonl_to_review_table.py`
- `python3 scripts/review_table_to_jsonl.py`
- `python3 scripts/lint_roleplay_dataset.py`
- `python3 scripts/build_roleplay_training_corpus.py`
- `python3 scripts/prepare_roleplay_dataset.py`

Recommended first loop:

```bash
python3 scripts/synthesize_roleplay_batch.py --count 300
python3 scripts/jsonl_to_review_table.py
# edit the TSV in Sheets / Excel / Numbers
python3 scripts/review_table_to_jsonl.py
python3 scripts/build_roleplay_training_corpus.py
python3 scripts/prepare_roleplay_dataset.py
```

Quick checks:

```bash
python3 -m unittest tests.test_roleplay_dataset_v2
python3 scripts/lint_roleplay_dataset.py --input data/roleplay_v2/approved_jsonl
```
