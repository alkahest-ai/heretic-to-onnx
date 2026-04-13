from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.heretic_to_onnx.text_sft_dataset import prepare_texting_sex_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="Maxx0/Texting_sex")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "external_text_sft" / "texting_sex"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-message-chars", type=int, default=80)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=True)
    args = parser.parse_args()

    report = prepare_texting_sex_dataset(
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        split=args.split,
        val_fraction=args.val_fraction,
        max_rows=args.max_rows,
        min_message_chars=args.min_message_chars,
        streaming=args.streaming,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
