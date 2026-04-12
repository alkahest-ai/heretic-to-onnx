# roleplay_v1

This folder is the scaffold for the first Alkahest-owned roleplay dataset.

Goals:

- adult-only
- consensual
- flirty / intimate / romantic / sexy without relying on copyrighted characters
- structured for chat-style SFT
- suitable for later open-source publication after review

Contents:

- `personas.yaml`: original companion archetypes
- `scenes.yaml`: original scenario seeds
- `style-rules.md`: writing rules for synthetic generation
- `seed_conversations.jsonl`: hand-authored seed records
- `generator_prompt.md`: reusable synthetic generation prompt
- `prompt_pack.jsonl`: generated prompt jobs
- `splits/`: generated train/val files

Current generation geometry:

- `6` personas
- `10` scenes
- `24` lane groups across the personas
- `240` persona/scene/lane combinations
- `25` variants per combination = `6,000` synthetic conversations in one batch

Use `python3 scripts/prepare_roleplay_dataset.py` to validate and split the dataset.
Use `python3 scripts/render_roleplay_prompt_pack.py` to generate prompt jobs for larger synthetic batches.
Use `python3 scripts/synthesize_roleplay_batch.py --variants 25 --id-prefix bulk25 --output data/roleplay_v1/generated/batch-0002.jsonl` to produce the first real `6k` synthetic pass.
