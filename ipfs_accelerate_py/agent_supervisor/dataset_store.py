"""Dataset-backed artifact storage for autonomous agent objective scans."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping


SCAN_DETAILS_ARTIFACT_SCHEMA_VERSION = 1
AUDIT_SNAPSHOT_SCHEMA_VERSION = 1
EXHAUSTION_QUORUM_STORE_SCHEMA_VERSION = 1
PROOF_SCOPE_INDEX_STORE_SCHEMA_VERSION = 1


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


@dataclass(frozen=True)
class DatasetScanDetailsArtifact:
    """Content-addressed reference to complete per-path scan diagnostics."""

    artifact_id: str
    scan_id: str
    jsonl_path: Path
    manifest_path: Path
    detail_count: int
    sha256: str
    byte_count: int
    reason_counts: Mapping[str, int]
    created_at: str
    schema_version: int = SCAN_DETAILS_ARTIFACT_SCHEMA_VERSION

    @property
    def row_count(self) -> int:
        return self.detail_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "scan_id": self.scan_id,
            "jsonl_path": str(self.jsonl_path),
            "manifest_path": str(self.manifest_path),
            "detail_count": self.detail_count,
            "row_count": self.detail_count,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DatasetAuditSnapshotArtifact:
    """Immutable full-finding catalog used by fingerprint-independent audits."""

    artifact_id: str
    scope_id: str
    jsonl_path: Path
    manifest_path: Path
    row_count: int
    sha256: str
    byte_count: int
    created_at: str
    metadata: Mapping[str, Any]
    schema_version: int = AUDIT_SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "scope_id": self.scope_id,
            "jsonl_path": str(self.jsonl_path),
            "manifest_path": str(self.manifest_path),
            "row_count": self.row_count,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "created_at": self.created_at,
            "metadata": _json_compatible(self.metadata),
        }


@dataclass(frozen=True)
class DatasetProofScopeIndexArtifact:
    """Durable content-addressed proof-scope index snapshot."""

    artifact_id: str
    index_name: str
    index_id: str
    json_path: Path
    manifest_path: Path
    sha256: str
    byte_count: int
    scope_count: int
    obligation_count: int
    receipt_count: int
    active_obligation_count: int
    active_receipt_count: int
    created_at: str
    schema_version: int = PROOF_SCOPE_INDEX_STORE_SCHEMA_VERSION

    @property
    def row_count(self) -> int:
        """Compatibility count for generic dataset artifact consumers."""

        return self.scope_count + self.obligation_count + self.receipt_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "index_name": self.index_name,
            "index_id": self.index_id,
            "json_path": str(self.json_path),
            "manifest_path": str(self.manifest_path),
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "scope_count": self.scope_count,
            "obligation_count": self.obligation_count,
            "receipt_count": self.receipt_count,
            "active_obligation_count": self.active_obligation_count,
            "active_receipt_count": self.active_receipt_count,
            "row_count": self.row_count,
            "created_at": self.created_at,
        }


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

    def persist_scan_details(
        self,
        *,
        scan_id: str,
        details: Iterable[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetScanDetailsArtifact:
        """Persist an immutable, content-addressed JSONL diagnostic artifact."""

        normalized_scan_id = str(scan_id or "").strip()
        if not normalized_scan_id:
            raise ValueError("scan_id must not be empty")
        rows: list[dict[str, Any]] = []
        for index, detail in enumerate(details):
            if not isinstance(detail, Mapping):
                raise TypeError(f"scan detail at index {index} must be a mapping")
            projected = _json_compatible(detail)
            if not isinstance(projected, dict):  # pragma: no cover - mapping invariant
                raise TypeError(f"scan detail at index {index} must project to an object")
            rows.append(projected)

        detail_bytes = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ).encode("utf-8")
        content_sha256 = sha256(detail_bytes).hexdigest()
        safe_id = _safe_dataset_id(normalized_scan_id)
        details_root = self.root / "scan-details"
        basename = f"{safe_id}-{content_sha256}"
        jsonl_path = details_root / f"{basename}.jsonl"
        manifest_path = details_root / f"{basename}.manifest.json"
        latest_path = details_root / f"{safe_id}.latest.json"
        created_at = datetime.now(timezone.utc).isoformat()
        reason_counts: dict[str, int] = {}
        for row in rows:
            reason = str(row.get("reason_code") or row.get("reason") or "unspecified").strip()
            reason_counts[reason or "unspecified"] = reason_counts.get(reason or "unspecified", 0) + 1

        artifact = DatasetScanDetailsArtifact(
            artifact_id=f"sha256:{content_sha256}",
            scan_id=normalized_scan_id,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            detail_count=len(rows),
            sha256=content_sha256,
            byte_count=len(detail_bytes),
            reason_counts=reason_counts,
            created_at=created_at,
        )
        manifest = artifact.to_dict()
        manifest["metadata"] = _json_compatible(metadata or {})
        _atomic_write_bytes(jsonl_path, detail_bytes)
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(manifest_path, manifest_text)
        _atomic_write_text(latest_path, manifest_text)
        return artifact

    def load_scan_details(
        self,
        scan: str | Path | Mapping[str, Any] | DatasetScanDetailsArtifact,
    ) -> list[dict[str, Any]]:
        path = self._scan_details_jsonl_path(scan)
        if path is None:
            return []
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

    def load_scan_details_manifest(
        self,
        scan: str | Path | Mapping[str, Any] | DatasetScanDetailsArtifact,
    ) -> dict[str, Any]:
        if isinstance(scan, DatasetScanDetailsArtifact):
            path: Path | None = scan.manifest_path
        elif isinstance(scan, Mapping):
            raw_path = scan.get("manifest_path")
            path = Path(str(raw_path)) if raw_path else None
        elif isinstance(scan, Path):
            path = scan if scan.name.endswith(".json") else None
        else:
            path = self.root / "scan-details" / f"{_safe_dataset_id(scan)}.latest.json"
        if path is None:
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def persist_audit_snapshot(
        self,
        *,
        scope_id: str,
        findings: Iterable[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetAuditSnapshotArtifact:
        """Persist a sorted, content-addressed audit baseline and latest pointer."""

        normalized_scope = str(scope_id or "").strip()
        if not normalized_scope:
            raise ValueError("scope_id must not be empty")
        rows: list[dict[str, Any]] = []
        for index, finding in enumerate(findings):
            if not isinstance(finding, Mapping):
                raise TypeError(f"audit finding at index {index} must be a mapping")
            row = _json_compatible(finding)
            if not isinstance(row, dict):  # pragma: no cover - mapping invariant
                raise TypeError(f"audit finding at index {index} must project to an object")
            rows.append(row)
        rows.sort(
            key=lambda row: (
                str(row.get("audit_key") or row.get("semantic_key") or ""),
                str(row.get("content_revision") or row.get("fingerprint") or ""),
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            )
        )
        encoded = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ).encode("utf-8")
        digest = sha256(encoded).hexdigest()
        safe_scope = _safe_dataset_id(normalized_scope)
        directory = self.root / "audit-scans" / safe_scope
        jsonl_path = directory / f"{digest}.jsonl"
        manifest_path = directory / f"{digest}.manifest.json"
        latest_path = self.root / "audit-scans" / f"{safe_scope}.latest.json"
        created_at = datetime.now(timezone.utc).isoformat()
        artifact_metadata = dict(metadata or {})
        if manifest_path.exists():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                existing_manifest = {}
            if isinstance(existing_manifest, Mapping):
                created_at = str(existing_manifest.get("created_at") or created_at)
                existing_metadata = existing_manifest.get("metadata")
                if isinstance(existing_metadata, Mapping):
                    artifact_metadata = dict(existing_metadata)
        artifact = DatasetAuditSnapshotArtifact(
            artifact_id=f"sha256:{digest}",
            scope_id=normalized_scope,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            row_count=len(rows),
            sha256=digest,
            byte_count=len(encoded),
            created_at=created_at,
            metadata=artifact_metadata,
        )
        manifest_text = json.dumps(
            artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        if not jsonl_path.exists():
            _atomic_write_bytes(jsonl_path, encoded)
        elif jsonl_path.read_bytes() != encoded:
            raise ValueError(f"audit snapshot digest collision at {jsonl_path}")
        if not manifest_path.exists():
            _atomic_write_text(manifest_path, manifest_text)
        else:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        _atomic_write_text(latest_path, manifest_text)
        return artifact

    def load_audit_snapshot(
        self,
        scope: str | Path | Mapping[str, Any] | DatasetAuditSnapshotArtifact,
    ) -> list[dict[str, Any]]:
        """Load an audit catalog by stable scope, artifact, manifest, or JSONL path."""

        if isinstance(scope, DatasetAuditSnapshotArtifact):
            path: Path | None = scope.jsonl_path
        elif isinstance(scope, Mapping):
            raw_path = scope.get("jsonl_path") or scope.get("path")
            path = Path(str(raw_path)) if raw_path else None
        elif isinstance(scope, Path):
            path = scope
        else:
            latest = self.root / "audit-scans" / f"{_safe_dataset_id(scope)}.latest.json"
            try:
                manifest = json.loads(latest.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                manifest = {}
            raw_path = manifest.get("jsonl_path") if isinstance(manifest, Mapping) else None
            path = Path(str(raw_path)) if raw_path else None
        if path is None:
            return []
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

    def load_audit_snapshot_manifest(self, scope_id: str) -> dict[str, Any]:
        latest = self.root / "audit-scans" / f"{_safe_dataset_id(scope_id)}.latest.json"
        try:
            value = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def persist_proof_scope_index(
        self,
        index: Any,
        *,
        index_name: str = "proof-scope-index",
    ) -> DatasetProofScopeIndexArtifact:
        """Persist an immutable index and atomically advance its latest pointer.

        The import stays local so the general objective dataset store remains
        usable in reduced installations which do not import proof contracts.
        """

        from .proof_scope_index import ProofScopeIndex

        if isinstance(index, Mapping):
            index = ProofScopeIndex.from_dict(index)
        if not isinstance(index, ProofScopeIndex):
            raise TypeError("index must be a ProofScopeIndex or index mapping")
        normalized_name = str(index_name or "").strip()
        if not normalized_name:
            raise ValueError("index_name must not be empty")

        payload = index.canonical_dict()
        encoded = (
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            + "\n"
        ).encode("utf-8")
        digest = sha256(encoded).hexdigest()
        safe_name = _safe_dataset_id(normalized_name)
        directory = self.root / "proof-scope-indexes" / safe_name
        json_path = directory / f"{digest}.json"
        manifest_path = directory / f"{digest}.manifest.json"
        latest_path = self.root / "proof-scope-indexes" / f"{safe_name}.latest.json"
        created_at = datetime.now(timezone.utc).isoformat()

        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                existing = {}
            if isinstance(existing, Mapping):
                created_at = str(existing.get("created_at") or created_at)

        artifact = DatasetProofScopeIndexArtifact(
            artifact_id=f"sha256:{digest}",
            index_name=normalized_name,
            index_id=index.index_id,
            json_path=json_path,
            manifest_path=manifest_path,
            sha256=digest,
            byte_count=len(encoded),
            scope_count=len(index.scope_records),
            obligation_count=len(index.obligations),
            receipt_count=len(index.receipts),
            active_obligation_count=len(index.active_obligation_ids),
            active_receipt_count=len(index.active_receipt_ids),
            created_at=created_at,
        )
        manifest_text = json.dumps(
            artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        if not json_path.exists():
            _atomic_write_bytes(json_path, encoded)
        elif json_path.read_bytes() != encoded:
            raise ValueError(f"proof scope index digest collision at {json_path}")
        if not manifest_path.exists():
            _atomic_write_text(manifest_path, manifest_text)
        else:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        _atomic_write_text(latest_path, manifest_text)
        return artifact

    def load_proof_scope_index(
        self,
        index: str | Path | Mapping[str, Any] | DatasetProofScopeIndexArtifact = "proof-scope-index",
    ) -> Any | None:
        """Load a proof-scope index by name, artifact, manifest, or JSON path."""

        from .proof_scope_index import ProofScopeIndex

        if isinstance(index, DatasetProofScopeIndexArtifact):
            path: Path | None = index.json_path
        elif isinstance(index, Mapping):
            raw_path = index.get("json_path") or index.get("path")
            path = Path(str(raw_path)) if raw_path else None
        elif isinstance(index, Path):
            if index.name.endswith(".manifest.json"):
                try:
                    manifest = json.loads(index.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError):
                    manifest = {}
                raw_path = (
                    manifest.get("json_path")
                    if isinstance(manifest, Mapping)
                    else None
                )
                path = Path(str(raw_path)) if raw_path else None
            else:
                path = index
        else:
            manifest = self.load_proof_scope_index_manifest(index)
            raw_path = manifest.get("json_path")
            path = Path(str(raw_path)) if raw_path else None
        if path is None:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        return ProofScopeIndex.from_dict(payload)

    def load_proof_scope_index_manifest(
        self, index_name: str = "proof-scope-index"
    ) -> dict[str, Any]:
        """Return the latest proof-scope index manifest for ``index_name``."""

        path = (
            self.root
            / "proof-scope-indexes"
            / f"{_safe_dataset_id(index_name)}.latest.json"
        )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}

    # Short spellings for persistence adapters which already establish the
    # artifact kind in their call site.
    persist_scope_index = persist_proof_scope_index
    load_scope_index = load_proof_scope_index

    def persist_exhaustion_quorum(self, quorum: Any) -> Path:
        """Atomically persist a bounded quorum projection by repository.

        The repository-keyed latest file lets a changed binding explicitly
        invalidate prior members.  Receipt artifacts remain the durable source
        of truth; this projection is a restart-friendly accumulator.
        """

        payload = quorum.to_dict() if hasattr(quorum, "to_dict") else dict(quorum)
        binding = payload.get("binding")
        if not isinstance(binding, Mapping):
            raise ValueError("quorum projection is missing binding")
        repository_id = str(binding.get("repository_id") or "").strip()
        if not repository_id:
            raise ValueError("quorum binding is missing repository_id")
        directory = self.root / "exhaustion-quorum"
        key = sha256(repository_id.encode("utf-8")).hexdigest()
        path = directory / f"{key}.json"
        envelope = {
            "schema_version": EXHAUSTION_QUORUM_STORE_SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "quorum": _json_compatible(payload),
        }
        _atomic_write_text(
            path,
            json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return path

    def load_exhaustion_quorum(self, repository_id: str) -> dict[str, Any]:
        key = sha256(str(repository_id).encode("utf-8")).hexdigest()
        path = self.root / "exhaustion-quorum" / f"{key}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        if not isinstance(value, Mapping):
            return {}
        quorum = value.get("quorum", value)
        return dict(quorum) if isinstance(quorum, Mapping) else {}

    def record_exhaustion_receipt(
        self,
        receipt: Any,
        *,
        binding: Any,
        required_members: int,
    ) -> Any:
        """Merge one receipt into persisted quorum state without double voting."""

        from .scan_receipts import ExhaustionBinding, evaluate_exhaustion_quorum

        resolved = binding if isinstance(binding, ExhaustionBinding) else ExhaustionBinding.from_dict(binding)
        previous = self.load_exhaustion_quorum(resolved.repository_id)
        prior_members = previous.get("members", ()) if isinstance(previous, Mapping) else ()
        result = evaluate_exhaustion_quorum(
            [*prior_members, receipt],
            binding=resolved,
            required_members=required_members,
        )
        self.persist_exhaustion_quorum(result)
        return result

    def _scan_details_jsonl_path(
        self,
        scan: str | Path | Mapping[str, Any] | DatasetScanDetailsArtifact,
    ) -> Path | None:
        if isinstance(scan, DatasetScanDetailsArtifact):
            return scan.jsonl_path
        if isinstance(scan, Mapping):
            raw_path = scan.get("jsonl_path") or scan.get("path")
            return Path(str(raw_path)) if raw_path else None
        if isinstance(scan, Path):
            return scan
        manifest = self.load_scan_details_manifest(scan)
        raw_path = manifest.get("jsonl_path")
        return Path(str(raw_path)) if raw_path else None


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_compatible(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


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
