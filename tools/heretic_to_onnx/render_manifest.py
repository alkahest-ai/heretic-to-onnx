from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RenderManifestReport:
    ok: bool
    template_path: str
    output_path: str
    overrides: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_manifest(
    template_path: str | Path,
    output_path: str | Path,
    *,
    source_model_id: str | None = None,
    base_model_id: str | None = None,
    target_repo_id: str | None = None,
) -> RenderManifestReport:
    template = Path(template_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()

    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest template must be a mapping")

    overrides: dict[str, str] = {}
    if source_model_id:
        data["source_model_id"] = source_model_id
        overrides["source_model_id"] = source_model_id
    if base_model_id:
        data["base_model_id"] = base_model_id
        overrides["base_model_id"] = base_model_id
    if target_repo_id:
        data["target_repo_id"] = target_repo_id
        overrides["target_repo_id"] = target_repo_id

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    return RenderManifestReport(
        ok=True,
        template_path=str(template),
        output_path=str(output),
        overrides=overrides,
    )
