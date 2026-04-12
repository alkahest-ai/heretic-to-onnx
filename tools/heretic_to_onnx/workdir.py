from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Manifest


@dataclass(slots=True)
class WorkDirLayout:
    root: Path
    inputs: Path
    source_snapshot: Path
    base_snapshot: Path
    export_raw: Path
    export_quantized: Path
    package_dir: Path
    reports: Path

    def ensure(self) -> "WorkDirLayout":
        for path in (
            self.root,
            self.inputs,
            self.export_raw,
            self.export_quantized,
            self.package_dir,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def resolve_work_dir(manifest: Manifest, work_dir: str | Path | None = None) -> WorkDirLayout:
    root = (
        Path(work_dir).expanduser().resolve()
        if work_dir
        else (manifest.manifest_dir.parent / "build" / "work" / manifest.slug).resolve()
    )
    return WorkDirLayout(
        root=root,
        inputs=root / "inputs",
        source_snapshot=root / "inputs" / "source",
        base_snapshot=root / "inputs" / "base",
        export_raw=root / "export" / "raw",
        export_quantized=root / "export" / "quantized",
        package_dir=root / "package",
        reports=root / "reports",
    )

