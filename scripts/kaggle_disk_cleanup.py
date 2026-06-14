#!/usr/bin/env python3
"""Free Kaggle /kaggle/working disk before large Unsloth or HF jobs."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path


def disk_snapshot(path: Path) -> dict[str, float | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "free_gb": round(usage.free / 1024**3, 3),
        "used_gb": round(usage.used / 1024**3, 3),
        "total_gb": round(usage.total / 1024**3, 3),
    }


def _rm(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    return True


def cleanup_kaggle_working(
    *,
    root: Path | None = None,
    pip_cache: bool = True,
    pycache: bool = True,
    unsloth_trash: bool = True,
    strip_git_metadata: bool = True,
    torch_cache: bool = True,
    hf_transfer_tmp: bool = True,
) -> dict[str, object]:
    work_root = (root or Path("/kaggle/working")).expanduser().resolve()
    actions: list[str] = []
    before = disk_snapshot(work_root)

    if pip_cache:
        subprocess.run(
            [sys.executable, "-m", "pip", "cache", "purge"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        actions.append("pip_cache_purge")

    if torch_cache:
        for pattern in ("torchinductor", "triton", ".cache/torch"):
            for path in (Path("/tmp").glob(pattern), work_root.glob(f"**/{pattern}")):
                for candidate in path:
                    if _rm(candidate):
                        actions.append(f"removed:{candidate}")

    if hf_transfer_tmp:
        for path in Path("/tmp").glob("hf_transfer_*"):
            if _rm(path):
                actions.append(f"removed:{path}")

    if unsloth_trash:
        for path in work_root.glob("unsloth*"):
            if path.is_dir() and path.name not in {"unsloth", "unsloth_zoo"}:
                if _rm(path):
                    actions.append(f"removed:{path}")

    if strip_git_metadata:
        for git_dir in work_root.glob("**/.git"):
            if _rm(git_dir):
                actions.append(f"removed:{git_dir}")

    if pycache:
        for cache_dir in work_root.glob("**/__pycache__"):
            if _rm(cache_dir):
                actions.append(f"removed:{cache_dir}")

    gc.collect()
    after = disk_snapshot(work_root)
    return {
        "before": before,
        "after": after,
        "actions": actions,
        "freed_gb": round(after["free_gb"] - before["free_gb"], 3),  # type: ignore[index]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/kaggle/working")
    parser.add_argument("--no-strip-git", action="store_true")
    args = parser.parse_args(argv)
    report = cleanup_kaggle_working(
        root=Path(args.root),
        strip_git_metadata=not args.no_strip_git,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())