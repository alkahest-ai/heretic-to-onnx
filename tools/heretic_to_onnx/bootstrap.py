from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .runtime import resolve_hf_command, resolve_optimum_command


@dataclass(slots=True)
class BootstrapReport:
    ok: bool
    hf_command: list[str]
    optimum_command: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_report() -> BootstrapReport:
    return BootstrapReport(
        ok=True,
        hf_command=resolve_hf_command(),
        optimum_command=resolve_optimum_command(),
        notes=[
            "If hf or optimum-cli are not installed locally, the scaffold falls back to uvx.",
            "Gemma 4 export now generates a custom split-graph runner, but execute mode still needs torch, transformers, and onnx in the selected Python environment.",
            "Gemma 4 q4f16 quantization now generates a custom runner, but execute mode needs onnx, onnxruntime, and onnxconverter_common.",
        ],
    )
