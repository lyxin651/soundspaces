"""Deterministic dataset path and atomic text primitives."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class StorageError(ValueError):
    """Raised for invalid dataset storage operations."""


V0_DEBUG_DATASET_RELATIVE_ROOT = Path(
    "datasets/active_audition_v0/aa_v0_replica_debug_001"
)


def json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class DatasetStorage:
    def __init__(self, root: str):
        self.root = Path(root)

    @property
    def success_path(self) -> Path:
        return self.root / "_SUCCESS"

    @classmethod
    def v0_debug(cls, repo_root: str) -> "DatasetStorage":
        """Resolve the frozen V0 dataset root; state is only `_SUCCESS`-based."""

        return cls(Path(repo_root) / V0_DEBUG_DATASET_RELATIVE_ROOT)

    def ensure_writable(self) -> None:
        if self.success_path.exists():
            raise StorageError("finalized dataset is immutable: {}".format(self.root))
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def manifest_path(self, name: str) -> Path:
        if not name.endswith(".jsonl"):
            raise StorageError("manifest must use .jsonl: {}".format(name))
        return self.root / "manifests" / name

    def atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, str(path))
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def atomic_write_jsonl(self, name: str, rows: Iterable[Mapping[str, Any]]) -> Path:
        path = self.manifest_path(name)
        text = "\n".join(json_line(row) for row in rows)
        if text:
            text += "\n"
        self.atomic_write_text(path, text)
        return path
