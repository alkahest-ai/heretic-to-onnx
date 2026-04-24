from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_KAGGLE_ROOT = Path("/kaggle/working/heretic-to-onnx")


@dataclass(frozen=True, slots=True)
class KaggleHereticPreset:
    label: str
    base_model_id: str
    merged_dir_name: str


PRESETS: dict[str, KaggleHereticPreset] = {
    "rally-2b": KaggleHereticPreset(
        label="rally-2b",
        base_model_id="google/gemma-4-E2B-it",
        merged_dir_name="rally-2b-heretic-merged",
    ),
    "alkahest-2b": KaggleHereticPreset(
        label="alkahest-2b",
        base_model_id="Qwen/Qwen3.5-2B",
        merged_dir_name="alkahest-2b-heretic-merged",
    ),
}


@dataclass(slots=True)
class KaggleHereticRunConfig:
    label: str
    base_model_id: str
    work_dir: Path
    merged_output_dir: Path
    checkpoint_dir: Path
    config_path: Path
    log_path: Path
    quantization: str = "bnb_4bit"
    n_trials: int = 20
    n_startup_trials: int = 8
    prompt_rows: int = 160
    eval_rows: int = 80
    max_response_length: int = 64
    max_batch_size: int = 32
    device_map: str = "auto"
    max_memory: dict[str, str] = field(default_factory=dict)
    good_dataset: str = "mlabonne/harmless_alpaca"
    bad_dataset: str = "mlabonne/harmful_behaviors"
    good_column: str = "text"
    bad_column: str = "text"


@dataclass(slots=True)
class MergedCheckpointReport:
    ok: bool
    path: str
    missing: list[str] = field(default_factory=list)
    weight_files: list[str] = field(default_factory=list)
    tokenizer_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KaggleHereticReport:
    ok: bool
    label: str
    base_model_id: str
    command: list[str]
    stdin_answers_path: str
    config_path: str
    log_path: str
    work_dir: str
    checkpoint_dir: str
    merged_output_dir: str
    environment: dict[str, Any]
    merged_checkpoint: dict[str, Any]
    returncode: int | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_for_label(label: str) -> KaggleHereticPreset:
    try:
        return PRESETS[label]
    except KeyError as exc:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unsupported Kaggle Heretic label: {label}; expected one of: {valid}") from exc


def build_run_config(
    *,
    label: str,
    base_model_id: str | None = None,
    work_root: str | Path | None = None,
    merged_output_dir: str | Path | None = None,
    n_trials: int = 20,
    n_startup_trials: int = 8,
    prompt_rows: int = 160,
    eval_rows: int = 80,
    max_response_length: int = 64,
    accelerator: str = "t4x2",
) -> KaggleHereticRunConfig:
    preset = preset_for_label(label)
    resolved_base = base_model_id or preset.base_model_id
    root = Path(work_root) if work_root else DEFAULT_KAGGLE_ROOT / "heretic" / label
    root = root.expanduser().resolve()
    merged = Path(merged_output_dir).expanduser().resolve() if merged_output_dir else root / preset.merged_dir_name

    max_memory: dict[str, str] = {}
    if accelerator == "t4x2":
        max_memory = {"0": "14GiB", "1": "14GiB", "cpu": "24GiB"}
    elif accelerator == "single-gpu":
        max_memory = {"0": "14GiB", "cpu": "24GiB"}
    elif accelerator != "auto":
        raise ValueError("accelerator must be one of: t4x2, single-gpu, auto")

    return KaggleHereticRunConfig(
        label=label,
        base_model_id=resolved_base,
        work_dir=root,
        merged_output_dir=merged,
        checkpoint_dir=root / "checkpoints",
        config_path=root / "config.toml",
        log_path=root / "heretic.log",
        n_trials=n_trials,
        n_startup_trials=n_startup_trials,
        prompt_rows=prompt_rows,
        eval_rows=eval_rows,
        max_response_length=max_response_length,
        max_memory=max_memory,
    )


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def render_config_toml(config: KaggleHereticRunConfig) -> str:
    good_split = f"train[:{config.prompt_rows}]"
    bad_split = f"train[:{config.prompt_rows}]"
    good_eval_split = f"test[:{config.eval_rows}]"
    bad_eval_split = f"test[:{config.eval_rows}]"

    return "\n".join(
        [
            f"model = {_toml_string(config.base_model_id)}",
            f"quantization = {_toml_string(config.quantization)}",
            'dtypes = ["auto", "float16", "bfloat16"]',
            f"device_map = {_toml_string(config.device_map)}",
            *(
                [
                    "max_memory = { "
                    + ", ".join(
                        f"{_toml_string(device)} = {_toml_string(memory)}"
                        for device, memory in config.max_memory.items()
                    )
                    + " }"
                ]
                if config.max_memory
                else []
            ),
            f"n_trials = {config.n_trials}",
            f"n_startup_trials = {config.n_startup_trials}",
            f"max_response_length = {config.max_response_length}",
            "batch_size = 0",
            f"max_batch_size = {config.max_batch_size}",
            f"study_checkpoint_dir = {_toml_string(str(config.checkpoint_dir))}",
            "",
            "[good_prompts]",
            f"dataset = {_toml_string(config.good_dataset)}",
            f"split = {_toml_string(good_split)}",
            f"column = {_toml_string(config.good_column)}",
            "",
            "[bad_prompts]",
            f"dataset = {_toml_string(config.bad_dataset)}",
            f"split = {_toml_string(bad_split)}",
            f"column = {_toml_string(config.bad_column)}",
            "",
            "[good_evaluation_prompts]",
            f"dataset = {_toml_string(config.good_dataset)}",
            f"split = {_toml_string(good_eval_split)}",
            f"column = {_toml_string(config.good_column)}",
            "",
            "[bad_evaluation_prompts]",
            f"dataset = {_toml_string(config.bad_dataset)}",
            f"split = {_toml_string(bad_eval_split)}",
            f"column = {_toml_string(config.bad_column)}",
            "",
        ]
    )


def write_run_files(config: KaggleHereticRunConfig) -> Path:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.config_path.write_text(render_config_toml(config), encoding="utf-8")
    stdin_path = config.work_dir / "heretic_stdin_answers.txt"
    stdin_path.write_text(build_stdin_answers(config), encoding="utf-8")
    return stdin_path


def build_stdin_answers(config: KaggleHereticRunConfig) -> str:
    # In Kaggle/notebook mode Heretic uses Python input() menus. The first proof
    # selects the first Pareto trial, saves locally, and chooses the full merge.
    return "\n".join(["1", "1", str(config.merged_output_dir), "1"]) + "\n"


def build_heretic_command(heretic_exec: str = "heretic") -> list[str]:
    return [heretic_exec]


def collect_environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    try:
        usage = shutil.disk_usage(Path.cwd())
        report["disk"] = {
            "cwd": str(Path.cwd()),
            "total_gb": round(usage.total / 1024**3, 2),
            "free_gb": round(usage.free / 1024**3, 2),
        }
    except OSError as exc:
        report["disk_error"] = str(exc)

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["torch"]["gpu_name"] = props.name
            report["torch"]["gpu_total_gb"] = round(props.total_memory / 1024**3, 2)
    except Exception as exc:  # pragma: no cover - depends on Kaggle image
        report["torch_error"] = str(exc)

    for module_name in ("transformers", "datasets", "peft", "bitsandbytes", "heretic"):
        try:
            module = __import__(module_name)
            report[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - depends on Kaggle image
            report[f"{module_name}_error"] = str(exc)

    return report


def validate_merged_checkpoint(path: str | Path) -> MergedCheckpointReport:
    root = Path(path)
    missing: list[str] = []
    if not (root / "config.json").is_file():
        missing.append("config.json")
    if not (root / "generation_config.json").is_file():
        missing.append("generation_config.json")
    if not (root / "tokenizer_config.json").is_file():
        missing.append("tokenizer_config.json")

    tokenizer_files = sorted(
        name
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")
        if (root / name).is_file()
    )
    if not tokenizer_files:
        missing.append("tokenizer.json|tokenizer.model|vocab.json")

    weight_files = sorted(p.name for p in root.glob("*.safetensors"))
    weight_files.extend(sorted(p.name for p in root.glob("pytorch_model*.bin")))
    if not weight_files:
        missing.append("*.safetensors|pytorch_model*.bin")

    return MergedCheckpointReport(
        ok=not missing,
        path=str(root),
        missing=missing,
        weight_files=weight_files,
        tokenizer_files=tokenizer_files,
    )


def ensure_generation_config(path: str | Path, *, base_model_id: str) -> bool:
    root = Path(path)
    generation_config_path = root / "generation_config.json"
    if generation_config_path.is_file():
        return False
    try:
        from transformers import GenerationConfig

        GenerationConfig.from_pretrained(base_model_id).save_pretrained(root)
        return True
    except Exception:
        return False


def run_kaggle_heretic(
    config: KaggleHereticRunConfig,
    *,
    heretic_exec: str = "heretic",
    dry_run: bool = False,
    force_notebook_mode: bool = True,
) -> KaggleHereticReport:
    stdin_path = write_run_files(config)
    command = build_heretic_command(heretic_exec)
    environment = collect_environment_report()
    warnings: list[str] = []
    errors: list[str] = []
    returncode: int | None = None

    if dry_run:
        merged = validate_merged_checkpoint(config.merged_output_dir)
        return KaggleHereticReport(
            ok=True,
            label=config.label,
            base_model_id=config.base_model_id,
            command=command,
            stdin_answers_path=str(stdin_path),
            config_path=str(config.config_path),
            log_path=str(config.log_path),
            work_dir=str(config.work_dir),
            checkpoint_dir=str(config.checkpoint_dir),
            merged_output_dir=str(config.merged_output_dir),
            environment=environment,
            merged_checkpoint=merged.to_dict(),
            warnings=warnings,
            errors=errors,
            dry_run=True,
        )

    env = os.environ.copy()
    if force_notebook_mode:
        env.setdefault("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    stdin_text = stdin_path.read_text(encoding="utf-8")

    with config.log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            input=stdin_text,
            text=True,
            cwd=config.work_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    returncode = process.returncode

    generated_generation_config = ensure_generation_config(
        config.merged_output_dir,
        base_model_id=config.base_model_id,
    )
    if generated_generation_config:
        warnings.append("generation_config.json was copied from the base model after merge")

    merged = validate_merged_checkpoint(config.merged_output_dir)
    if returncode != 0 and merged.ok:
        warnings.append(
            "Heretic exited non-zero after stdin ended, but the merged checkpoint is complete"
        )
    if not merged.ok:
        errors.append(
            "merged checkpoint is incomplete; optimization artifacts were left under the work/checkpoint dirs"
        )

    return KaggleHereticReport(
        ok=merged.ok,
        label=config.label,
        base_model_id=config.base_model_id,
        command=command,
        stdin_answers_path=str(stdin_path),
        config_path=str(config.config_path),
        log_path=str(config.log_path),
        work_dir=str(config.work_dir),
        checkpoint_dir=str(config.checkpoint_dir),
        merged_output_dir=str(config.merged_output_dir),
        environment=environment,
        merged_checkpoint=merged.to_dict(),
        returncode=returncode,
        warnings=warnings,
        errors=errors,
    )
