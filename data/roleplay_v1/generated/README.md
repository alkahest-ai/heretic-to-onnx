# generated

Put additional synthetic conversations here after generation and review.

Important:

- these files must contain actual SFT rows
- each row must be JSON with a `messages` list
- this is **training data**
- this is **not** the same thing as `prompt_pack.jsonl`

`prompt_pack.jsonl` contains generation jobs.

Files in `generated/` should contain completed chat conversations such as:

```json
{
  "id": "generated-0001",
  "tags": ["adult", "consensual", "flirt"],
  "messages": [
    {"role": "system", "content": "Stay in character..."},
    {"role": "user", "content": "I like how close you're standing."},
    {"role": "assistant", "content": "Then I won't pretend it was accidental."}
  ]
}
```
