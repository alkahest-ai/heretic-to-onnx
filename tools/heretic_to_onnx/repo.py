from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class RepoAccessError(RuntimeError):
    """Raised when a model repo cannot be accessed."""


def _resolve_local_spec(spec: str, manifest_dir: Path) -> Path | None:
    candidate = Path(spec).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    relative_to_manifest = (manifest_dir / spec).resolve()
    if relative_to_manifest.exists():
        return relative_to_manifest

    relative_to_cwd = (Path.cwd() / spec).resolve()
    if relative_to_cwd.exists():
        return relative_to_cwd

    return None


@dataclass(slots=True)
class RepoHandle:
    spec: str
    manifest_dir: Path
    local_path: Path | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.local_path = _resolve_local_spec(self.spec, self.manifest_dir)

    @property
    def is_local(self) -> bool:
        return self.local_path is not None

    @property
    def descriptor(self) -> str:
        if self.is_local:
            return f"local:{self.local_path}"
        return f"hf:{self.spec}"

    def exists(self, relative_path: str) -> bool:
        if self.is_local:
            return (self.local_path / relative_path).exists()
        return self._remote_exists(relative_path)

    def read_bytes(self, relative_path: str) -> bytes:
        if self.is_local:
            path = self.local_path / relative_path
            if not path.exists():
                raise RepoAccessError(f"missing file in {self.descriptor}: {relative_path}")
            return path.read_bytes()
        return self._remote_read(relative_path)

    def read_text(self, relative_path: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(relative_path).decode(encoding)

    def read_json(self, relative_path: str) -> dict[str, Any]:
        return json.loads(self.read_text(relative_path))

    def copy_file(self, relative_path: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.is_local:
            source = self.local_path / relative_path
            if not source.exists():
                raise RepoAccessError(f"missing file in {self.descriptor}: {relative_path}")
            shutil.copyfile(source, destination)
            return

        data = self._remote_read(relative_path)
        destination.write_bytes(data)

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "heretic-to-onnx/0.1"}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _resolve_url(self, relative_path: str) -> str:
        repo = quote(self.spec, safe="/-_.")
        file_path = quote(relative_path, safe="/-_.")
        return f"https://huggingface.co/{repo}/resolve/main/{file_path}"

    def _remote_exists(self, relative_path: str) -> bool:
        url = self._resolve_url(relative_path)
        request = Request(url, headers=self._headers(), method="HEAD")
        try:
            with urlopen(request):
                return True
        except HTTPError as error:
            if error.code == 404:
                return False
            if error.code == 405:
                try:
                    self._remote_read(relative_path)
                    return True
                except RepoAccessError:
                    return False
            raise RepoAccessError(f"failed to inspect {url}: HTTP {error.code}") from error
        except URLError as error:
            raise RepoAccessError(f"failed to access {url}: {error.reason}") from error

    def _remote_read(self, relative_path: str) -> bytes:
        url = self._resolve_url(relative_path)
        request = Request(url, headers=self._headers(), method="GET")
        try:
            with urlopen(request) as response:
                return response.read()
        except HTTPError as error:
            if error.code == 404:
                raise RepoAccessError(f"missing file in {self.descriptor}: {relative_path}") from error
            raise RepoAccessError(f"failed to download {url}: HTTP {error.code}") from error
        except URLError as error:
            raise RepoAccessError(f"failed to access {url}: {error.reason}") from error
