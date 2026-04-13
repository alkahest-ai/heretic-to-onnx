from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest
from .repo import RepoHandle


@dataclass(slots=True)
class PackageReport:
    ok: bool
    output_dir: str
    copied_assets: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_output_dir(manifest: Manifest) -> Path:
    return manifest.manifest_dir.parent / "build" / manifest.slug


def _expected_session_names(manifest: Manifest) -> list[str]:
    session_names: list[str] = []
    for relative_path in manifest.expected_onnx_files:
        path = Path(relative_path)
        if path.suffix != ".onnx":
            continue
        name = path.stem
        suffix = f"_{manifest.target_dtype}"
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        if name not in session_names:
            session_names.append(name)
    return session_names


def _external_data_mapping(manifest: Manifest) -> dict[str, int]:
    return {name: 1 for name in _expected_session_names(manifest)}


def _patch_config(config_path: Path, manifest: Manifest) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = [manifest.expected_architecture]
    transformers_js_config = config.setdefault("transformers.js_config", {})
    transformers_js_config["use_external_data_format"] = _external_data_mapping(manifest)
    kv_cache_dtype = transformers_js_config.setdefault("kv_cache_dtype", {})
    if manifest.target_dtype in {"q4f16", "fp16"}:
        kv_cache_dtype[manifest.target_dtype] = "float16"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthesize_preprocessor_config(repo: RepoHandle, destination: Path) -> bool:
    if not repo.exists("processor_config.json"):
        return False
    processor_config = repo.read_json("processor_config.json")
    image_processor = processor_config.get("image_processor")
    if not isinstance(image_processor, dict) or not image_processor:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(image_processor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def _synthesize_video_preprocessor_config(repo: RepoHandle, destination: Path) -> bool:
    if not repo.exists("processor_config.json"):
        return False
    processor_config = repo.read_json("processor_config.json")
    video_processor = processor_config.get("video_processor")
    if not isinstance(video_processor, dict) or not video_processor:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(video_processor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def _copy_onnx_artifacts(manifest: Manifest, onnx_source_dir: Path, destination: Path) -> tuple[int, list[str]]:
    copied = 0
    missing: list[str] = []
    for relative_path in manifest.expected_onnx_files:
        filename = Path(relative_path).name
        source_path = onnx_source_dir / filename
        dest_path = destination / filename
        if source_path.exists():
            shutil.copyfile(source_path, dest_path)
            copied += 1
            if filename.endswith(".onnx"):
                for sibling in sorted(onnx_source_dir.glob(f"{filename}_data*")):
                    shutil.copyfile(sibling, destination / sibling.name)
        else:
            missing.append(filename)
    return copied, missing


def package_repo(
    manifest: Manifest,
    output_dir: str | Path | None = None,
    *,
    force: bool = False,
    allow_missing_onnx: bool = True,
    onnx_source_dir: str | Path | None = None,
    source_spec: str | Path | None = None,
    base_spec: str | Path | None = None,
) -> PackageReport:
    source_repo = RepoHandle(str(source_spec or manifest.source_model_id), manifest.manifest_dir)
    base_repo = RepoHandle(str(base_spec or manifest.base_model_id), manifest.manifest_dir)
    destination = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(manifest)

    if destination.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {destination}")
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)
    onnx_dir = destination / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    report = PackageReport(ok=True, output_dir=str(destination))

    for asset in manifest.inherit_assets.from_source:
        dest_path = destination / asset
        source_repo.copy_file(asset, dest_path)
        report.copied_assets[asset] = source_repo.descriptor

    for asset in manifest.inherit_assets.from_base_if_missing:
        if source_repo.exists(asset):
            source_repo.copy_file(asset, destination / asset)
            report.copied_assets[asset] = source_repo.descriptor
        else:
            dest_path = destination / asset
            if base_repo.exists(asset):
                base_repo.copy_file(asset, dest_path)
                report.copied_assets[asset] = base_repo.descriptor
            elif asset == "preprocessor_config.json":
                synthesized = _synthesize_preprocessor_config(source_repo, dest_path) or _synthesize_preprocessor_config(
                    base_repo, dest_path
                )
                if synthesized:
                    report.copied_assets[asset] = "synthetic-from-processor_config.json"
                    report.notes.append("synthesized preprocessor_config.json from processor_config.json")
                else:
                    raise FileNotFoundError(f"unable to source or synthesize required asset: {asset}")
            elif asset == "video_preprocessor_config.json":
                synthesized = _synthesize_video_preprocessor_config(
                    source_repo, dest_path
                ) or _synthesize_video_preprocessor_config(base_repo, dest_path)
                if synthesized:
                    report.copied_assets[asset] = "synthetic-from-processor_config.json"
                    report.notes.append("synthesized video_preprocessor_config.json from processor_config.json")
                else:
                    raise FileNotFoundError(f"unable to source or synthesize required asset: {asset}")
            else:
                raise FileNotFoundError(f"unable to source required asset: {asset}")

    _patch_config(destination / "config.json", manifest)

    copied_onnx = 0
    missing_onnx: list[str] = []
    if onnx_source_dir:
        copied_onnx, missing_onnx = _copy_onnx_artifacts(
            manifest,
            Path(onnx_source_dir).expanduser().resolve(),
            onnx_dir,
        )

    if copied_onnx:
        report.notes.append(f"copied {copied_onnx} ONNX artifact(s) into {onnx_dir}")

    if missing_onnx:
        report.warnings.append(
            "some ONNX artifacts were not found in the provided source directory: "
            + ", ".join(missing_onnx)
        )

    if copied_onnx == len(manifest.expected_onnx_files):
        pass
    elif allow_missing_onnx:
        expected = "\n".join(manifest.expected_onnx_files)
        note_path = onnx_dir / "MISSING_ONNX_ARTIFACTS.txt"
        note_path.write_text(
            "The ONNX export and quantization stages have not produced the full artifact set yet.\n"
            "Expected artifact paths:\n"
            f"{expected}\n",
            encoding="utf-8",
        )
        report.warnings.append("onnx artifacts are not complete yet; wrote placeholder note instead")
    else:
        report.ok = False
        report.warnings.append("onnx artifacts were required but are not complete")

    package_report_path = destination / "package-report.json"
    report.notes.append(f"package report written to {package_report_path}")
    package_report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return report
