"""Immutable artifact references shared through the exclusive derived owner.

The registry coordinates cache discovery, not semantic verification. In
particular, registering a proof or certificate does not validate its claims.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

ARTIFACT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/derived-artifact-reference@1"
ARTIFACT_KINDS = (
    "vector_embeddings", "bm25_index", "knowledge_graph", "proof_cache", "certificate",
)
IDENTITY_FIELDS = frozenset({
    "repository_id", "tree_id", "artifact_kind", "input_digest",
    "producer_id", "producer_revision", "parameters_digest",
})
ARTIFACT_OPERATIONS = {
    "record_artifact": (IDENTITY_FIELDS - {"repository_id"}) | {"artifact_cid"},
    "lookup_artifact": IDENTITY_FIELDS - {"repository_id"},
    "list_artifacts": {"tree_id", "artifact_kind", "after", "limit"},
}
MAX_RECORD_BYTES = 2048
MAX_PAGE_SIZE = 64
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _text(value: Any, field: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > 256 or any(ord(c) < 32 for c in value)):
        raise ValueError(f"invalid derived artifact {field}")
    return value


def _kind(value: Any) -> str:
    if value not in ARTIFACT_KINDS:
        raise ValueError("unsupported derived artifact kind")
    return value


def _identity(payload: Mapping[str, Any]) -> dict[str, str]:
    identity = {field: _text(payload.get(field), field) for field in IDENTITY_FIELDS}
    _kind(identity["artifact_kind"])
    for field in ("input_digest", "parameters_digest"):
        if _DIGEST.fullmatch(identity[field]) is None:
            raise ValueError(f"derived artifact {field} must be a SHA-256 digest")
    return identity


class DerivedArtifactRegistry:
    """Owner-only storage; callers retain the gateway transaction lock."""

    def __init__(self, connection: Any):
        self._connection = connection
        connection.execute(
            "CREATE TABLE IF NOT EXISTS derived_coordination_artifacts "
            "(artifact_key VARCHAR PRIMARY KEY, repository_id VARCHAR NOT NULL, "
            "tree_id VARCHAR NOT NULL, artifact_kind VARCHAR NOT NULL, body_json VARCHAR NOT NULL)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS derived_artifact_scope "
            "ON derived_coordination_artifacts(repository_id, tree_id, artifact_kind, artifact_key)"
        )

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        operation = payload["operation"]
        repository_id = _text(payload.get("repository_id"), "repository_id")
        if operation == "list_artifacts":
            return self._list(payload, repository_id)
        identity = _identity(payload)
        key = _digest({"schema": ARTIFACT_SCHEMA, **identity})
        row = self._connection.execute(
            "SELECT body_json FROM derived_coordination_artifacts "
            "WHERE repository_id = ? AND artifact_key = ?",
            [repository_id, key],
        ).fetchone()
        prior = json.loads(row[0]) if row else None
        if operation == "lookup_artifact":
            return {"artifact": prior, "artifact_verified": False}
        record = {
            "schema": ARTIFACT_SCHEMA, **identity, "artifact_key": key,
            "artifact_cid": _text(payload.get("artifact_cid"), "artifact_cid"),
        }
        record["reference_id"] = _digest(record)
        body = _canonical(record)
        if len(body.encode()) > MAX_RECORD_BYTES:
            raise ValueError("derived artifact reference exceeds bound")
        if prior is not None:
            if prior != record:
                raise ValueError("derived artifact identity has a conflicting publication")
        else:
            self._connection.execute(
                "INSERT INTO derived_coordination_artifacts VALUES (?, ?, ?, ?, ?)",
                [key, repository_id, identity["tree_id"], identity["artifact_kind"], body],
            )
        return {"artifact": record, "artifact_verified": False}

    def _list(self, payload: Mapping[str, Any], repository_id: str) -> dict[str, Any]:
        tree = _text(payload.get("tree_id"), "tree_id")
        kind = _kind(payload.get("artifact_kind"))
        after = payload.get("after", "")
        if not isinstance(after, str) or (after and _DIGEST.fullmatch(after) is None):
            raise ValueError("invalid derived artifact cursor")
        limit = payload.get("limit", MAX_PAGE_SIZE)
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError("invalid derived artifact page size")
        rows = self._connection.execute(
            "SELECT artifact_key, body_json FROM derived_coordination_artifacts "
            "WHERE repository_id = ? AND tree_id = ? AND artifact_kind = ? "
            "AND artifact_key > ? ORDER BY artifact_key LIMIT ?",
            [repository_id, tree, kind, after, limit + 1],
        ).fetchall()
        more = len(rows) > limit
        page = rows[:limit]
        return {
            "artifacts": [json.loads(row[1]) for row in page],
            "has_more": more, "next_cursor": page[-1][0] if more else "",
            "artifact_verified": False,
        }
