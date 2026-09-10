"""Bounded derived codebase and artifact coordination on the native Quack owner.

This service stores disposable acceleration evidence and source references. Git,
ipfs_kit_py and ipfs_datasets_py retain their respective semantic authority.
No client path opens a database or falls back to an embedded writer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .derived_artifacts import (
    ARTIFACT_KINDS, ARTIFACT_OPERATIONS, ARTIFACT_SCHEMA,
    MAX_PAGE_SIZE, DerivedArtifactRegistry,
)

SCHEMA = "ipfs_accelerate_py/agent-supervisor/derived-coordination@1"
MAX_REQUEST_BYTES = 262144
MAX_SOURCE_BYTES = 32768
MAX_FILES = 8


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _text(value: Any, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid derived {name}")
    return value


class DerivedCoordinationService:
    """Server-only cache, sharing the exclusive gateway handle and lock."""

    def __init__(
        self,
        connection: Any,
        *,
        transaction_lock: Any,
        owner_identity: Mapping[str, Any],
    ):
        self._connection = connection
        self._lock = transaction_lock
        self._identity = dict(owner_identity)
        self._ast = None
        # References are permitted even when the sealed datasets profile forbids
        # accelerator-local parsing. No AST writer is constructed at admission.
        connection.execute(
            "CREATE TABLE IF NOT EXISTS derived_coordination_references "
            "(reference_id VARCHAR PRIMARY KEY, repository_id VARCHAR NOT NULL, "
            "tree_id VARCHAR NOT NULL, body_json VARCHAR NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS derived_coordination_ingests "
            "(snapshot_id VARCHAR PRIMARY KEY, input_digest VARCHAR NOT NULL, result_json VARCHAR NOT NULL)"
        )
        self._artifacts = DerivedArtifactRegistry(connection)

    def _index(self):
        from .duckdb_ast_index import DuckDBASTIndex
        from .semantic_truth_authority import (
            assert_accelerator_semantic_writer_permitted,
        )

        assert_accelerator_semantic_writer_permitted(writer=DuckDBASTIndex.INTERFACE)
        if self._ast is None:
            self._ast = DuckDBASTIndex.from_owner_connection(
                self._connection,
                database_path=Path("owner-bound-derived-index"),
                transaction_lock=self._lock,
            )
        return self._ast

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if len(_canonical(payload).encode()) > MAX_REQUEST_BYTES:
            raise ValueError("derived request exceeds bound")
        repository_id = _text(payload.get("repository_id"), "repository_id")
        operation = payload.get("operation")
        allowed = {
            "capabilities": set(),
            **ARTIFACT_OPERATIONS,
            "record_reference": {"tree_id", "ast_cid", "content_hash", "state_root"},
            "list_references": {"tree_id"},
            "ingest_snapshot": {"tree_id", "files", "worktree_id"},
            "snapshot": {"snapshot_id"},
            "parse_cache": {"content_hash"},
        }
        if (
            operation not in allowed
            or set(payload) - {"operation", "repository_id"} - allowed[operation]
        ):
            raise ValueError("unknown derived operation or request field")
        if operation == "capabilities":
            result = {
                "operations": sorted(allowed), "artifact_kinds": list(ARTIFACT_KINDS),
                "artifact_schema": ARTIFACT_SCHEMA, "artifact_page_limit": MAX_PAGE_SIZE,
                "artifact_verified": False,
            }
        elif operation in ARTIFACT_OPERATIONS:
            result = self._artifacts.execute(payload)
        elif operation == "record_reference":
            record = {
                key: _text(payload.get(key), key, 512) for key in allowed[operation]
            }
            record["repository_id"] = repository_id
            record["reference_id"] = _identity(record)
            self._connection.execute(
                "INSERT OR IGNORE INTO derived_coordination_references VALUES (?, ?, ?, ?)",
                [
                    record["reference_id"],
                    repository_id,
                    record["tree_id"],
                    _canonical(record),
                ],
            )
            result = {"reference": record, "source_reference_verified": False}
        elif operation == "list_references":
            tree_id = _text(payload.get("tree_id"), "tree_id")
            rows = self._connection.execute(
                "SELECT body_json FROM derived_coordination_references WHERE repository_id = ? "
                "AND tree_id = ? ORDER BY reference_id LIMIT 257",
                [repository_id, tree_id],
            ).fetchall()
            result = {
                "references": [json.loads(row[0]) for row in rows[:256]],
                "has_more": len(rows) > 256,
                "source_reference_verified": False,
            }
        elif operation == "ingest_snapshot":
            result = self._ingest(payload)
        elif operation == "snapshot":
            index = self._index()
            snapshot_id = _text(payload.get("snapshot_id"), "snapshot_id")
            snapshot = index.get_snapshot(snapshot_id)
            if snapshot is not None and snapshot.repository_id != repository_id:
                raise ValueError("snapshot repository differs from the admitted scope")
            result = {"snapshot": None if snapshot is None else snapshot.to_dict()}
            if snapshot is not None:
                result["files"] = list(index.list_files(snapshot_id))
                result["symbols"] = [
                    symbol.to_dict() for symbol in index.list_symbols(snapshot_id)
                ]
                result["frontiers"] = [
                    row.to_dict() for row in index.list_frontiers(snapshot_id)
                ]
        else:
            # A digest alone cannot read another repository's cache facts.
            index = self._index()
            digest = _text(payload.get("content_hash"), "content_hash")
            owned = self._connection.execute(
                "SELECT 1 FROM source_files f JOIN source_snapshots s ON s.snapshot_id = f.snapshot_id "
                "WHERE s.repository_id = ? AND f.content_digest = ? LIMIT 1",
                [repository_id, digest],
            ).fetchone()
            result = {
                "cache_entry": index.get_parse_cache_entry(digest) if owned else None
            }
        envelope = {
            "schema": SCHEMA,
            "authority": "derived_evidence",
            "completion_authority": False,
            "owner_identity": self._identity,
            "result": result,
        }
        if len(_canonical(envelope).encode()) > MAX_REQUEST_BYTES:
            raise ValueError("derived result exceeds bound")
        return envelope

    def _ingest(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from .duckdb_ast_index import SourceFileSpec, SourceSnapshot

        files = payload.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
            raise ValueError("derived snapshot file count exceeds bound")
        size = 0
        for row in files:
            if not isinstance(row, dict) or set(row) - {
                "path",
                "content",
                "content_digest",
                "language",
                "blob_id",
                "ignored",
                "ast_record",
            }:
                raise ValueError("invalid derived file")
            content = row.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("derived source must be text")
            size += len((content or "").encode())
        if size > MAX_SOURCE_BYTES:
            raise ValueError("derived source bytes exceed bound")
        index = self._index()
        specs = [SourceFileSpec(**row) for row in files]
        # Exact tree/hash identities are immutable. Replayed ingestion reuses the
        # original result; a changed body cannot replace an existing snapshot.
        identity_input = {
            "repository_id": payload["repository_id"],
            "tree_id": _text(payload.get("tree_id"), "tree_id"),
            "files": sorted(
                [
                    {
                        **{
                            key: value for key, value in raw.items() if key != "content"
                        },
                        "content_digest": spec.content_digest,
                    }
                    for raw, spec in zip(files, specs)
                ],
                key=lambda row: row["path"],
            ),
        }
        digest = _identity(identity_input)
        snapshot = SourceSnapshot(
            snapshot_id="",
            repository_id=identity_input["repository_id"],
            tree_id=identity_input["tree_id"],
            overlay_digest="",
            created_at="",
            scanner_version=index.scanner_version,
        )
        prior = self._connection.execute(
            "SELECT input_digest, result_json FROM derived_coordination_ingests WHERE snapshot_id = ?",
            [snapshot.snapshot_id],
        ).fetchone()
        if prior is not None:
            if prior[0] != digest:
                raise ValueError(
                    "derived snapshot identity has conflicting source hashes"
                )
            return json.loads(prior[1])
        existing = index.list_files(snapshot.snapshot_id)
        if existing and sorted(
            (row["path"], row["content_digest"]) for row in existing
        ) != sorted((row.path, row.content_digest) for row in specs):
            raise ValueError("derived snapshot source hashes differ")
        result = index.ingest_snapshot(
            repository_id=identity_input["repository_id"],
            tree_id=identity_input["tree_id"],
            files=specs,
            worktree_id=str(payload.get("worktree_id") or ""),
        ).to_dict()
        self._connection.execute(
            "INSERT INTO derived_coordination_ingests VALUES (?, ?, ?)",
            [snapshot.snapshot_id, digest, _canonical(result)],
        )
        return result


class DerivedCoordinationClient:
    """Typed owner facade with explicit repository scope and no fallback.

    Long-running supervisors should supply a connection factory. Each operation
    then authenticates a fresh short-lived owner session and closes it after the
    response, so idle clients do not retain expired grants. A supplied connection
    remains caller-owned. Neither form retries a failed operation.
    """

    def __init__(self, connection: Any = None, *, repository_id: str,
                 connection_factory: Callable[[], Any] | None = None):
        if (connection is None) == (connection_factory is None):
            raise ValueError("provide either a derived connection or a connection factory")
        self._connection = connection
        self._connection_factory = connection_factory
        self.repository_id = _text(repository_id, "repository_id")

    def call(self, operation: str, **parameters: Any) -> Mapping[str, Any]:
        if "repository_id" in parameters:
            raise ValueError("derived repository scope cannot be overridden")
        payload = {"operation": operation, "repository_id": self.repository_id, **parameters}
        if self._connection_factory is None:
            return self._connection.derived_coordination(payload)
        connection = self._connection_factory()
        try:
            return connection.derived_coordination(payload)
        finally:
            connection.close()

    @classmethod
    def from_fleet_deployment(cls, deployment_path: Path, *, repository_id: str,
                             client_id: str, timeout_seconds: float = 5) -> "DerivedCoordinationClient":
        """Reconnect through current owner credentials on each explicit call.

        No credential is retained across owner restarts, and a failed request
        is never replayed. Native admission still binds every new session.
        """
        from ..runtime.derived_fleet_client import fleet_connection_factory

        return cls(repository_id=repository_id, connection_factory=fleet_connection_factory(
            deployment_path, repository_id=repository_id, client_id=client_id,
            timeout_seconds=timeout_seconds))
