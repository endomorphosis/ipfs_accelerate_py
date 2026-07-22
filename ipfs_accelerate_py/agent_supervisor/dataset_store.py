"""Dataset-backed artifact storage for autonomous agent objective scans."""

from __future__ import annotations

import json
import os
import tempfile
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
    scanned_record_count: int = 0
    parsed_record_count: int = 0
    reused_record_count: int = 0
    deleted_record_count: int = 0
    renamed_record_count: int = 0
    invalidated_record_count: int = 0
    scan_elapsed_seconds: float = 0.0
    parse_elapsed_seconds: float = 0.0
    saved_parse_seconds: float = 0.0
    deleted_paths: tuple[str, ...] = ()

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of current records served from the prior snapshot."""

        return self.reused_record_count / self.scanned_record_count if self.scanned_record_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["jsonl_path"] = str(self.jsonl_path)
        payload["manifest_path"] = str(self.manifest_path)
        if self.parquet_path is not None:
            payload["parquet_path"] = str(self.parquet_path)
        payload["deleted_paths"] = list(self.deleted_paths)
        payload["cache_hit_ratio"] = self.cache_hit_ratio
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
        scan_stats: dict[str, Any] | None = None,
    ) -> DatasetArtifact:
        rows = [dict(record) for record in records]
        safe_id = _safe_dataset_id(dataset_id)
        self.root.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.root / f"{safe_id}.jsonl"
        manifest_path = self.root / f"{safe_id}.manifest.json"
        parquet_path = self.root / f"{safe_id}.parquet"

        # A reader must see either the preceding complete snapshot or the new
        # complete snapshot.  In particular, a refill process being killed may
        # not leave an empty/truncated cache that forces unrelated files to be
        # reparsed on the next pass.
        _atomic_write_text(
            jsonl_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )

        backend = "jsonl"
        manager_result: dict[str, Any] | None = None
        error = ""
        written_parquet: Path | None = None

        try:
            dataset_cls, dataset_backend = _import_dataset_cls()
            manager_cls = _import_dataset_manager_cls()
            dataset = dataset_cls.from_list(rows)
            if hasattr(dataset, "to_parquet"):
                with tempfile.NamedTemporaryFile(
                    prefix=f".{parquet_path.name}.",
                    suffix=".tmp",
                    dir=self.root,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    dataset.to_parquet(str(temporary_path))
                    os.replace(temporary_path, parquet_path)
                    written_parquet = parquet_path
                finally:
                    temporary_path.unlink(missing_ok=True)
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

        # JSONL is the authoritative fallback snapshot.  Never leave a parquet
        # file from an older, larger snapshot discoverable when the optional
        # backend could not materialize the current rows (notably an empty
        # snapshot after all tracked candidates were deleted).
        if written_parquet is None:
            parquet_path.unlink(missing_ok=True)

        stats = dict(scan_stats or {})
        artifact = DatasetArtifact(
            dataset_id=dataset_id,
            backend=backend,
            row_count=len(rows),
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            parquet_path=written_parquet,
            manager_result=manager_result,
            error=error,
            scanned_record_count=_nonnegative_int(stats.get("scanned_record_count"), len(rows)),
            parsed_record_count=_nonnegative_int(stats.get("parsed_record_count")),
            reused_record_count=_nonnegative_int(stats.get("reused_record_count")),
            deleted_record_count=_nonnegative_int(stats.get("deleted_record_count")),
            renamed_record_count=_nonnegative_int(stats.get("renamed_record_count")),
            invalidated_record_count=_nonnegative_int(stats.get("invalidated_record_count")),
            scan_elapsed_seconds=_nonnegative_float(stats.get("scan_elapsed_seconds")),
            parse_elapsed_seconds=_nonnegative_float(stats.get("parse_elapsed_seconds")),
            saved_parse_seconds=_nonnegative_float(stats.get("saved_parse_seconds")),
            deleted_paths=_normalized_paths(stats.get("deleted_paths")),
        )
        manifest = artifact.to_dict()
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return artifact

    def load_records(self, dataset_id: str) -> list[dict[str, Any]]:
        """Load the last complete JSONL snapshot for ``dataset_id``.

        Invalid rows are ignored rather than making the cache authoritative for
        only a prefix of a damaged legacy file.  Atomic writes prevent this for
        new snapshots, while the defensive reader lets existing installations
        recover by reparsing just the records that cannot be reused.
        """

        path = self.root / f"{_safe_dataset_id(dataset_id)}.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def load_manifest(self, dataset_id: str) -> dict[str, Any]:
        """Return the last complete manifest, or an empty mapping."""

        path = self.root / f"{_safe_dataset_id(dataset_id)}.manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, float(default))


def _normalized_paths(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, (str, Path)) else value
    try:
        return tuple(sorted({str(path) for path in values if str(path)}))
    except TypeError:
        return ()


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
