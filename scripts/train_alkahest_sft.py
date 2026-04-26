#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _messages_to_text(row: dict[str, Any], tokenizer: Any) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT row is missing a messages list")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _load_multimodal_text_model(model_id: str, **kwargs: Any) -> Any:
    from transformers import AutoModelForCausalLM

    try:
        from transformers import AutoModelForImageTextToText
    except Exception:
        AutoModelForImageTextToText = None

    try:
        from transformers import AutoModelForVision2Seq
    except Exception:
        AutoModelForVision2Seq = None

    errors: list[str] = []
    for factory in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        if factory is None:
            continue
        try:
            return factory.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            errors.append(f"{factory.__name__}: {exc}")
    raise RuntimeError("failed to load model with available AutoModel classes:\n" + "\n".join(errors))


def _save_generation_config(model: Any, output_dir: Path) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(str(output_dir))


def _copy_missing_assets(source: str, output_dir: Path, tokenizer: Any) -> None:
    tokenizer.save_pretrained(str(output_dir))
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        hf_hub_download = None

    for name in ("chat_template.jinja", "generation_config.json"):
        dst = output_dir / name
        if dst.exists():
            continue
        source_path = Path(source) / name
        if source_path.exists():
            shutil.copy2(source_path, dst)
            continue
        if hf_hub_download is None:
            continue
        try:
            downloaded = hf_hub_download(source, name)
        except Exception:
            continue
        shutil.copy2(downloaded, dst)


def _upload_folder(repo_id: str, folder: Path, *, private: bool, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=message,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA SFT runner for Alkahest Qwen Heretic checkpoints.")
    parser.add_argument("--model-name", default="thomasjvu/alkahest-0.8b-heretic-merged")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True, help="Adapter/trainer output directory")
    parser.add_argument("--merged-output-dir", required=True, help="Merged HF checkpoint output directory")
    parser.add_argument("--dataset-manifest", default="")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--merge", action="store_true", default=True)
    parser.add_argument("--no-merge", dest="merge", action="store_false")
    parser.add_argument("--upload-merged-to", default="")
    parser.add_argument("--upload-adapter-to", default="")
    parser.add_argument("--upload-private", action="store_true", default=True)
    parser.add_argument("--no-upload-private", dest="upload_private", action="store_false")
    return parser


def main() -> int:
    args = _parser().parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir).expanduser().resolve()
    merged_output_dir = Path(args.merged_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = _read_jsonl(Path(args.train_file).expanduser().resolve(), limit=args.train_limit)
    val_rows = _read_jsonl(Path(args.val_file).expanduser().resolve(), limit=args.val_limit)
    if not train_rows:
        raise ValueError("training split is empty")
    if not val_rows:
        raise ValueError("validation split is empty")

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        text = _messages_to_text(row, tokenizer)
        encoded = tokenizer(text, truncation=True, max_length=args.max_seq_length)
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(tokenize, remove_columns=list(train_rows[0].keys()))
    eval_dataset = Dataset.from_list(val_rows).map(tokenize, remove_columns=list(val_rows[0].keys()))

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = _load_multimodal_text_model(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            fp16=True,
            optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
            report_to=[],
            seed=args.seed,
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    run_report: dict[str, Any] = {
        "ok": True,
        "model_name": args.model_name,
        "train_file": str(Path(args.train_file).expanduser().resolve()),
        "val_file": str(Path(args.val_file).expanduser().resolve()),
        "dataset_manifest": args.dataset_manifest,
        "output_dir": str(output_dir),
        "merged_output_dir": str(merged_output_dir),
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "merged": False,
        "uploaded_merged_to": "",
        "uploaded_adapter_to": "",
    }

    if args.upload_adapter_to:
        _upload_folder(
            args.upload_adapter_to,
            output_dir,
            private=args.upload_private,
            message="Upload Alkahest 0.8B Heretic RP SFT adapter",
        )
        run_report["uploaded_adapter_to"] = args.upload_adapter_to

    if args.merge:
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_model = _load_multimodal_text_model(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        merged_model = PeftModel.from_pretrained(base_model, str(output_dir))
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(str(merged_output_dir), safe_serialization=True, max_shard_size="4GB")
        _save_generation_config(merged_model, merged_output_dir)
        _copy_missing_assets(args.model_name, merged_output_dir, tokenizer)
        run_report["merged"] = True

        if args.upload_merged_to:
            card = merged_output_dir / "README.md"
            if not card.exists():
                card.write_text(
                    "\n".join(
                        [
                            "---",
                            "library_name: transformers",
                            "license: other",
                            "private: true",
                            "---",
                            "",
                            "# Alkahest 0.8B Heretic RP SFT",
                            "",
                            "Experimental roleplay SFT checkpoint produced from `thomasjvu/alkahest-0.8b-heretic-merged`.",
                            "This is a merged Hugging Face checkpoint intended for ONNX/WebGPU export after validation.",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            _upload_folder(
                args.upload_merged_to,
                merged_output_dir,
                private=args.upload_private,
                message="Upload Alkahest 0.8B Heretic RP SFT merged checkpoint",
            )
            run_report["uploaded_merged_to"] = args.upload_merged_to

    (output_dir / "training-run.json").write_text(json.dumps(run_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
