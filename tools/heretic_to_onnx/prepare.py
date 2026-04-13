from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest
from .repo import RepoHandle
from .runtime import run_command, resolve_hf_command
from .workdir import resolve_work_dir


@dataclass(slots=True)
class PrepareReport:
    ok: bool
    work_dir: str
    source_path: str
    base_path: str
    commands: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metadata_files(manifest: Manifest) -> list[str]:
    files = list(manifest.inherit_assets.from_source)
    for candidate in ("config.json",):
        if candidate not in files:
            files.append(candidate)
    return files


def _download_repo(repo_id: str, destination: Path, extra_args: list[str]) -> dict[str, Any]:
    args = [*resolve_hf_command(), "download", repo_id, "--local-dir", str(destination), *extra_args]
    result = run_command(args)
    return result.to_dict()


def _has_all_files(root: Path, required_files: list[str]) -> bool:
    return root.exists() and all((root / relative_path).exists() for relative_path in required_files)


def _can_reuse_base_snapshot(root: Path, required_files: list[str]) -> bool:
    if not root.exists():
        return False
    for relative_path in required_files:
        if (root / relative_path).exists():
            continue
        if relative_path == "preprocessor_config.json" and (root / "processor_config.json").exists():
            continue
        return False
    return True


def _has_full_source_weights(root: Path) -> bool:
    return (root / "config.json").exists() and (
        (root / "model.safetensors").exists() or (root / "model.safetensors.index.json").exists()
    )


def prepare_repos(
    manifest: Manifest,
    work_dir: str | Path | None = None,
    *,
    source_mode: str = "metadata",
) -> PrepareReport:
    layout = resolve_work_dir(manifest, work_dir).ensure()
    source_repo = RepoHandle(manifest.source_model_id, manifest.manifest_dir)
    base_repo = RepoHandle(manifest.base_model_id, manifest.manifest_dir)

    commands: list[dict[str, Any]] = []
    notes: list[str] = []

    if source_repo.is_local or source_mode == "skip":
        source_path = source_repo.local_path if source_repo.is_local else layout.source_snapshot
        if not source_repo.is_local and source_mode == "skip":
            notes.append("source repo download was skipped")
    else:
        source_path = layout.source_snapshot
        if source_mode == "full":
            if _has_full_source_weights(source_path):
                notes.append(f"reused existing full source snapshot at {source_path}")
            else:
                commands.append(_download_repo(manifest.source_model_id, source_path, []))
        elif source_mode == "metadata":
            metadata_files = _metadata_files(manifest)
            if _has_all_files(source_path, metadata_files):
                notes.append(f"reused existing metadata source snapshot at {source_path}")
            else:
                commands.append(_download_repo(manifest.source_model_id, source_path, metadata_files))
        else:
            raise ValueError(f"unsupported source_mode: {source_mode}")

    if base_repo.is_local:
        base_path = base_repo.local_path
    else:
        base_path = layout.base_snapshot
        extra_files = manifest.inherit_assets.from_base_if_missing or ["processor_config.json", "preprocessor_config.json"]
        download_files = list(dict.fromkeys(extra_files))
        if "preprocessor_config.json" in download_files and "processor_config.json" not in download_files:
            try:
                if not base_repo.exists("preprocessor_config.json") and base_repo.exists("processor_config.json"):
                    download_files.append("processor_config.json")
            except Exception:
                pass
        if _can_reuse_base_snapshot(base_path, extra_files):
            notes.append(f"reused existing base snapshot at {base_path}")
        else:
            commands.append(_download_repo(manifest.base_model_id, base_path, download_files))

    report = PrepareReport(
        ok=True,
        work_dir=str(layout.root),
        source_path=str(source_path),
        base_path=str(base_path),
        commands=commands,
        notes=notes,
    )
    report_path = layout.reports / "prepare-report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
