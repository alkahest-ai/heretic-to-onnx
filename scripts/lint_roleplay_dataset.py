from __future__ import annotations

import argparse
import json
from pathlib import Path

from roleplay_dataset_v2 import ROLEPLAY_V2_DIR, lint_conversations, load_conversations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROLEPLAY_V2_DIR / "corpus.jsonl"),
        help="Input JSONL or CSV/TSV review table path",
    )
    parser.add_argument("--assistant-line-threshold", type=int, default=2)
    parser.add_argument("--assistant-skeleton-threshold", type=int, default=3)
    parser.add_argument("--conversation-shape-threshold", type=int, default=4)
    parser.add_argument("--approved-only", action="store_true", help="Require approved review-table rows")
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    conversations = load_conversations(input_path, approved_only=args.approved_only)
    report = lint_conversations(
        conversations,
        assistant_line_threshold=args.assistant_line_threshold,
        assistant_skeleton_threshold=args.assistant_skeleton_threshold,
        conversation_shape_threshold=args.conversation_shape_threshold,
    )
    print(json.dumps(report, indent=2))
    if report["errors"]:
        return 1
    if args.fail_on_warnings and report["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
