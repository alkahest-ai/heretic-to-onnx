from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import DatasetDict, load_dataset
from trl import SFTConfig, SFTTrainer

ROOT_DIR = Path(__file__).resolve().parents[1]


def _apply_thinking_prefix(messages: list[dict], enabled: bool) -> list[dict]:
    if not enabled or not messages:
        return messages
    patched = deepcopy(messages)
    first = patched[0]
    if first.get("role") == "system":
        content = first.get("content", "")
        if isinstance(content, str) and not content.startswith("<|think|>"):
            first["content"] = "<|think|>\n" + content
    return patched


def _format_row(row: dict, tokenizer, *, enable_thinking: bool) -> dict:
    messages = row["messages"]
    if not isinstance(messages, list):
        raise ValueError("row is missing messages list")
    rendered = tokenizer.apply_chat_template(
        _apply_thinking_prefix(messages, enable_thinking),
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": rendered}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="p-e-w/gemma-4-E2B-it-heretic-ara")
    parser.add_argument("--train-file", default=str(ROOT_DIR / "data/roleplay_v2/splits/train.jsonl"))
    parser.add_argument("--val-file", default=str(ROOT_DIR / "data/roleplay_v2/splits/val.jsonl"))
    parser.add_argument(
        "--dataset-manifest",
        default=str(ROOT_DIR / "data/roleplay_v2/splits/manifest.json"),
        help="Manifest recording the roleplay dataset build used for training",
    )
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "build/unsloth/rally"))
    parser.add_argument("--merged-output-dir", default=str(ROOT_DIR / "build/unsloth/rally-merged"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(args.train_file).expanduser().resolve()),
            "validation": str(Path(args.val_file).expanduser().resolve()),
        },
    )
    if not isinstance(dataset, DatasetDict):
        raise ValueError("expected DatasetDict from load_dataset")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    train_dataset = dataset["train"].map(
        lambda row: _format_row(row, tokenizer, enable_thinking=args.enable_thinking),
        desc="format train rows",
    )
    eval_dataset = dataset["validation"].map(
        lambda row: _format_row(row, tokenizer, enable_thinking=args.enable_thinking),
        desc="format validation rows",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=25,
            save_steps=50,
            output_dir=str(Path(args.output_dir).expanduser().resolve()),
            optim="adamw_8bit",
            seed=args.seed,
            dataset_num_proc=1,
            report_to=[],
        ),
    )

    trainer.train()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.save_merged:
        merged_dir = Path(args.merged_output_dir).expanduser().resolve()
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    dataset_manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    dataset_manifest = None
    if dataset_manifest_path.exists():
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))

    metadata = {
        "model_name": args.model_name,
        "train_file": str(Path(args.train_file).expanduser().resolve()),
        "val_file": str(Path(args.val_file).expanduser().resolve()),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest": dataset_manifest,
        "output_dir": str(output_dir),
        "merged_output_dir": str(Path(args.merged_output_dir).expanduser().resolve()),
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "enable_thinking": args.enable_thinking,
    }
    (output_dir / "training-run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
