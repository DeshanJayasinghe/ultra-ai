from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ArtifactStoreError(RuntimeError):
    """Raised when a worker cannot read or write an AI artifact."""


class ArtifactReader(Protocol):
    def read_bytes(self, key: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class LocalArtifactStore:
    root: Path

    def read_bytes(self, key: str) -> bytes:
        path = self._path_for(key)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ArtifactStoreError(f"Could not read artifact: {key}.") from exc

    def write_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        path = self._path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        except OSError as exc:
            raise ArtifactStoreError(f"Could not write artifact: {key}.") from exc

    def _path_for(self, key: str) -> Path:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ArtifactStoreError(f"Invalid artifact key: {key}.")

        return self.root / key
