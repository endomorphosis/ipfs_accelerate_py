"""Dataset-backed artifact storage for autonomous agent objective scans."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetArtifact:
    """Summary of a persisted objective-scan dataset artifact."""

    dataset_id: str
    backend: str
    row_count: int
    jsonl_path: Path
    manifest_path: Path
    parquet_path: Path | None = None
    manager_result: dict[str, Any] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["jsonl_path"] = str(self.jsonl_path)
        payload["manifest_path"] = str(self.manifest_path)
        if self.parquet_path is not None:
            payload["parquet_path"] = str(self.parquet_path)
        return payload


class ObjectiveDatasetStore:
    """Persist large AST/evidence records with an optional ipfs_datasets backend.

    The primary contract is stable even when optional dataset dependencies are
    absent: every call writes JSONL plus a manifest.  When ``ipfs_datasets_py``
    and HuggingFace ``datasets`` are available, rows are also saved through
    ``DatasetManager`` and written as parquet.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def persist_records(
        self,
        *,
        dataset_id: str,
        records: Iterable[dict[str, Any]],
    ) -> DatasetArtifact:
        rows = [dict(record) for record in records]
        safe_id = _safe_dataset_id(dataset_id)
        self.root.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.root / f"{safe_id}.jsonl"
        manifest_path = self.root / f"{safe_id}.manifest.json"
        parquet_path = self.root / f"{safe_id}.parquet"

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        backend = "jsonl"
        manager_result: dict[str, Any] | None = None
        error = ""
        written_parquet: Path | None = None

        try:
            dataset_cls, dataset_backend = _import_dataset_cls()
            manager_cls = _import_dataset_manager_cls()
            dataset = dataset_cls.from_list(rows)
            if hasattr(dataset, "to_parquet"):
                dataset.to_parquet(str(parquet_path))
                written_parquet = parquet_path
            if manager_cls is not None:
                manager = manager_cls(use_accelerate=False)
                manager.save_dataset(dataset_id, dataset)
                managed = manager.get_dataset(dataset_id)
                if hasattr(managed, "save"):
                    result = managed.save(str(written_parquet or jsonl_path), format="parquet" if written_parquet else "jsonl")
                    if isinstance(result, dict):
                        manager_result = result
            backend = "ipfs_datasets_py" if manager_cls is not None else dataset_backend
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        artifact = DatasetArtifact(
            dataset_id=dataset_id,
            backend=backend,
            row_count=len(rows),
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            parquet_path=written_parquet,
            manager_result=manager_result,
            error=error,
        )
        manifest = artifact.to_dict()
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact


def _safe_dataset_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value.strip())
    safe = safe.strip("-._")
    return safe or "objective-dataset"


def _import_dataset_cls() -> tuple[Any, str]:
    try:
        from ipfs_datasets_py.ipfs_datasets import Dataset

        if Dataset is not object and hasattr(Dataset, "from_list"):
            return Dataset, "ipfs_datasets_py"
    except Exception:
        pass
    from datasets import Dataset

    return Dataset, "datasets"


def _import_dataset_manager_cls() -> Any:
    try:
        from ipfs_datasets_py.dataset_manager import DatasetManager

        return DatasetManager
    except Exception:
        return None
