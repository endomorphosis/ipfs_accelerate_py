"""Bounded derived codebase and artifact coordination on the native Quack owner.

This service stores disposable acceleration evidence and source references. Git,
ipfs_kit_py and ipfs_datasets_py retain their respective semantic authority.
No client path opens a database or falls back to an embedded writer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .derived_artifacts import (
    ARTIFACT_KINDS, ARTIFACT_OPERATIONS, ARTIFACT_SCHEMA,
    IDENTITY_FIELDS, MAX_PAGE_SIZE, DerivedArtifactRegistry,
)

SCHEMA = "ipfs_accelerate_py/agent-supervisor/derived-coordination@1"
MAX_REQUEST_BYTES = 262144
MAX_SOURCE_BYTES = 32768
MAX_FILES = 8
MAX_REFERENCE_PAGE_SIZE = 256
_REFERENCE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


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
            "list_references": {"tree_id", "after", "limit"},
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
                "reference_page_limit": MAX_REFERENCE_PAGE_SIZE,
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
            after = payload.get("after", "")
            limit = payload.get("limit", MAX_REFERENCE_PAGE_SIZE)
            if not isinstance(after, str) or (after and _REFERENCE_ID.fullmatch(after) is None):
                raise ValueError("invalid derived reference cursor")
            if type(limit) is not int or not 1 <= limit <= MAX_REFERENCE_PAGE_SIZE:
                raise ValueError("invalid derived reference page size")
            rows = self._connection.execute(
                "SELECT body_json FROM derived_coordination_references WHERE repository_id = ? "
                "AND tree_id = ? AND reference_id > ? ORDER BY reference_id LIMIT ?",
                [repository_id, tree_id, after, limit + 1],
            ).fetchall()
            page = [json.loads(row[0]) for row in rows[:limit]]
            more = len(rows) > limit
            result = {
                "references": page,
                "has_more": more,
                "next_cursor": page[-1]["reference_id"] if more else "",
                "source_reference_verified": False,
            }
            # Reference fields may each be up to 512 characters. Honor the
            # response byte bound as well as the requested row bound, leaving
            # omitted rows reachable through the last returned identity.
            while len(_canonical(self._response(result)).encode()) > MAX_REQUEST_BYTES:
                page.pop()
                if not page:
                    raise ValueError("derived reference exceeds response bound")
                result["has_more"] = True
                result["next_cursor"] = page[-1]["reference_id"]
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
        envelope = self._response(result)
        if len(_canonical(envelope).encode()) > MAX_REQUEST_BYTES:
            raise ValueError("derived result exceeds bound")
        return envelope

    def _response(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "authority": "derived_evidence",
            "completion_authority": False,
            "owner_identity": self._identity,
            "result": result,
        }

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


class DerivedDiscoveryLimitExceeded(RuntimeError):
    """The caller's page budget ended before discovery was exhausted."""

    def __init__(self, next_cursor: str):
        self.next_cursor = next_cursor
        super().__init__("derived discovery page budget exhausted; resume at next_cursor")


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

    def iter_references(
        self, *, tree_id: str, after: str = "", limit: int = MAX_REFERENCE_PAGE_SIZE,
        max_pages: int = 1024,
    ) -> Iterator[Mapping[str, Any]]:
        """Discover all source references through bounded, fresh owner sessions.

        Discovery is not a point-in-time snapshot: references inserted behind
        the cursor during a scan are visible on the next scan. An exhausted
        page budget or invalid response raises instead of implying completeness.
        Transport failures propagate without replaying the failed request.
        """
        yield from self._iter_discovery(
            "list_references", tree_id=tree_id, after=after, limit=limit,
            max_pages=max_pages,
        )

    def iter_artifacts(
        self, *, tree_id: str, artifact_kind: str, after: str = "",
        limit: int = MAX_PAGE_SIZE, max_pages: int = 1024,
    ) -> Iterator[Mapping[str, Any]]:
        """Discover artifact references; callers still verify their contents."""
        if artifact_kind not in ARTIFACT_KINDS:
            raise ValueError("unsupported derived artifact kind")
        yield from self._iter_discovery(
            "list_artifacts", tree_id=tree_id, after=after, limit=limit,
            max_pages=max_pages, artifact_kind=artifact_kind,
        )

    def _iter_discovery(
        self, operation: str, *, tree_id: str, after: str, limit: int,
        max_pages: int, artifact_kind: str = "",
    ) -> Iterator[Mapping[str, Any]]:
        _text(tree_id, "tree_id")
        page_cap = MAX_PAGE_SIZE if artifact_kind else MAX_REFERENCE_PAGE_SIZE
        if type(limit) is not int or not 1 <= limit <= page_cap:
            raise ValueError("invalid derived discovery page size")
        if type(max_pages) is not int or not 1 <= max_pages <= 1_000_000:
            raise ValueError("invalid derived discovery page budget")
        if not isinstance(after, str) or (after and _REFERENCE_ID.fullmatch(after) is None):
            raise ValueError("invalid derived discovery cursor")
        key = "artifact_key" if artifact_kind else "reference_id"
        collection = "artifacts" if artifact_kind else "references"
        verified = "artifact_verified" if artifact_kind else "source_reference_verified"
        for _page_number in range(max_pages):
            parameters = {"tree_id": tree_id, "after": after, "limit": limit}
            if artifact_kind:
                parameters["artifact_kind"] = artifact_kind
            envelope = self.call(operation, **parameters)
            if (
                not isinstance(envelope, Mapping) or envelope.get("schema") != SCHEMA
                or envelope.get("authority") != "derived_evidence"
                or envelope.get("completion_authority") is not False
            ):
                raise ValueError("invalid derived discovery envelope")
            result = envelope.get("result")
            if not isinstance(result, Mapping) or result.get(verified) is not False:
                raise ValueError("invalid derived discovery result")
            rows = result.get(collection)
            more, next_cursor = result.get("has_more"), result.get("next_cursor")
            if type(rows) is not list or len(rows) > limit or type(more) is not bool:
                raise ValueError("invalid derived discovery page")
            previous = after
            for row in rows:
                if (
                    not isinstance(row, Mapping)
                    or row.get("repository_id") != self.repository_id
                    or row.get("tree_id") != tree_id
                    or (artifact_kind and row.get("artifact_kind") != artifact_kind)
                    or not isinstance(row.get(key), str)
                    or _REFERENCE_ID.fullmatch(row[key]) is None
                    or row[key] <= previous
                ):
                    raise ValueError("derived discovery identity or order differs")
                unsigned = dict(row)
                reference_id = unsigned.pop("reference_id", None)
                if _identity(unsigned) != reference_id:
                    raise ValueError("derived discovery reference digest differs")
                if artifact_kind:
                    if (
                        set(row) != IDENTITY_FIELDS | {
                            "schema", "artifact_key", "artifact_cid", "reference_id",
                        }
                        or row.get("schema") != ARTIFACT_SCHEMA
                        or row["artifact_key"] != _identity({
                            "schema": ARTIFACT_SCHEMA,
                            **{field: row[field] for field in IDENTITY_FIELDS},
                        })
                    ):
                        raise ValueError("derived discovery artifact identity differs")
                elif set(row) != {
                    "repository_id", "tree_id", "ast_cid", "content_hash",
                    "state_root", "reference_id",
                }:
                    raise ValueError("derived discovery reference fields differ")
                previous = row[key]
            if (
                (more and (not rows or next_cursor != previous))
                or (not more and next_cursor != "")
            ):
                raise ValueError("derived discovery cursor did not advance")
            # Validate the entire page before exposing even its first row.
            yield from rows
            if not more:
                return
            after = next_cursor
        raise DerivedDiscoveryLimitExceeded(after)

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
