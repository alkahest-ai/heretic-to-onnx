from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CommandError(RuntimeError):
    """Raised when an external command fails."""


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_hf_command() -> list[str]:
    if shutil.which("hf"):
        return ["hf"]
    return ["uvx", "hf"]


def resolve_optimum_command() -> list[str]:
    if shutil.which("optimum-cli"):
        return ["optimum-cli"]
    return ["uvx", "--from", "optimum[onnx]", "optimum-cli"]


def run_command(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    check: bool = True,
) -> CommandResult:
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    result = CommandResult(
        args=list(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and completed.returncode != 0:
        raise CommandError(
            f"command failed with exit code {completed.returncode}: {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return result

