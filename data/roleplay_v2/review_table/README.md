# review_table

This folder contains the editable spreadsheet-style review exports.

Recommended workflow:

1. generate raw JSONL
2. export TSV
3. edit in Sheets / Excel / Numbers
4. set `status=approved` on rows you want to keep
5. compile back to JSONL with `review_table_to_jsonl.py`
