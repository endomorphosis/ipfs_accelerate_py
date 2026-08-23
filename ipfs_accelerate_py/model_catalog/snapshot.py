"""Immutable snapshot helpers and revision-bound cursor pagination.

The schema module owns the wire representation of :class:`CatalogSnapshot`.
This module contains operations over that representation.  In particular,
cursors are deliberately tied to both a snapshot revision and a record kind so
that a caller can never accidentally continue a listing against changed data.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from .identity import canonical_json, canonical_json_bytes, content_cid
from .schema import (
    CatalogSnapshot,
    DeploymentDescriptor,
    ModelDescriptor,
    ProviderDescriptor,
    RouterBinding,
    SCHEMA_VERSION,
)

MAX_PAGE_SIZE = 1_000
CURSOR_VERSION = 1

_RECORD_TYPES = {
    "providers": ProviderDescriptor,
    "models": ModelDescriptor,
    "deployments": DeploymentDescriptor,
    "bindings": RouterBinding,
}


class PaginationError(ValueError):
    """Base class for invalid or unusable catalog pagination cursors."""


class InvalidCursorError(PaginationError):
    """The supplied cursor is malformed or does not describe this listing."""


class StaleCursorError(PaginationError):
    """The supplied cursor belongs to a different catalog snapshot."""


@dataclass(frozen=True)
class CatalogPage:
    """One immutable page from a particular snapshot."""

    items: Tuple[Any, ...]
    snapshot_revision: str
    record_type: str
    total: int
    next_cursor: Optional[str] = None

    def __post_init__(self) -> None:
        if self.record_type not in _RECORD_TYPES:
            raise ValueError("unknown catalog record type: %s" % self.record_type)
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items))
        expected = _RECORD_TYPES[self.record_type]
        if any(not isinstance(item, expected) for item in self.items):
            raise TypeError("page contains a record of the wrong type")
        if isinstance(self.total, bool) or not isinstance(self.total, int) or self.total < 0:
            raise ValueError("total must be a non-negative integer")
        if len(self.items) > self.total:
            raise ValueError("page cannot contain more records than total")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "snapshot_revision": self.snapshot_revision,
            "record_type": self.record_type,
            "total": self.total,
            "next_cursor": self.next_cursor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CatalogPage":
        if not isinstance(data, Mapping):
            raise ValueError("CatalogPage must be an object")
        allowed = {
            "items",
            "snapshot_revision",
            "record_type",
            "total",
            "next_cursor",
        }
        if set(data) != allowed:
            raise ValueError("CatalogPage has missing or unknown fields")
        record_type = data["record_type"]
        if record_type not in _RECORD_TYPES:
            raise ValueError("unknown catalog record type: %s" % record_type)
        items = data["items"]
        if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
            raise ValueError("CatalogPage items must be an array")
        parsed = tuple(_RECORD_TYPES[record_type].from_dict(item) for item in items)
        return cls(
            items=parsed,
            snapshot_revision=data["snapshot_revision"],
            record_type=record_type,
            total=data["total"],
            next_cursor=data["next_cursor"],
        )


def create_snapshot(
    records: Iterable[Any] = (),
    *,
    providers: Iterable[ProviderDescriptor] = (),
    models: Iterable[ModelDescriptor] = (),
    deployments: Iterable[DeploymentDescriptor] = (),
    bindings: Iterable[RouterBinding] = (),
    created_at: Optional[str] = None,
) -> CatalogSnapshot:
    """Create a snapshot from typed records in any input order.

    ``records`` is convenient for registries that hold a heterogeneous set;
    explicitly named collections can be supplied at the same time.
    """

    grouped = {
        "providers": list(providers),
        "models": list(models),
        "deployments": list(deployments),
        "bindings": list(bindings),
    }
    kind_by_type = {
        ProviderDescriptor: "providers",
        ModelDescriptor: "models",
        DeploymentDescriptor: "deployments",
        RouterBinding: "bindings",
    }
    for record in records:
        kind = kind_by_type.get(type(record))
        if kind is None:
            raise TypeError("unsupported catalog record type: %s" % type(record).__name__)
        grouped[kind].append(record)
    return CatalogSnapshot(
        providers=tuple(grouped["providers"]),
        models=tuple(grouped["models"]),
        deployments=tuple(grouped["deployments"]),
        bindings=tuple(grouped["bindings"]),
        created_at=created_at,
    )


def snapshot_records(
    snapshot: CatalogSnapshot, record_type: Optional[str] = None
) -> Tuple[Any, ...]:
    """Return deterministic immutable records from *snapshot*."""

    if not isinstance(snapshot, CatalogSnapshot):
        raise TypeError("snapshot must be a CatalogSnapshot")
    if record_type is None:
        return (
            tuple(snapshot.providers)
            + tuple(snapshot.models)
            + tuple(snapshot.deployments)
            + tuple(snapshot.bindings)
        )
    if record_type not in _RECORD_TYPES:
        raise ValueError("unknown catalog record type: %s" % record_type)
    return tuple(getattr(snapshot, record_type))


def iter_snapshot(snapshot: CatalogSnapshot, record_type: Optional[str] = None) -> Iterator[Any]:
    """Iterate over a stable tuple, never mutable registry storage."""

    return iter(snapshot_records(snapshot, record_type))


def _cursor_checksum(payload: Mapping[str, Any]) -> str:
    return content_cid({"catalog_cursor": payload})


def _encode_cursor(revision: str, record_type: str, offset: int, query_key: str) -> str:
    payload = {
        "v": CURSOR_VERSION,
        "revision": revision,
        "record_type": record_type,
        "offset": offset,
        "query": query_key,
    }
    envelope = {"payload": payload, "checksum": _cursor_checksum(payload)}
    return base64.urlsafe_b64encode(canonical_json_bytes(envelope)).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> Dict[str, Any]:
    if not isinstance(cursor, str) or not cursor or len(cursor) > 4096:
        raise InvalidCursorError("cursor must be a bounded non-empty string")
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(
            (cursor + padding).encode("ascii"), altchars=b"-_", validate=True
        )
        envelope = json.loads(decoded.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor is malformed") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "checksum"}:
        raise InvalidCursorError("cursor envelope is malformed")
    payload = envelope["payload"]
    if not isinstance(payload, dict) or set(payload) != {
        "v",
        "revision",
        "record_type",
        "offset",
        "query",
    }:
        raise InvalidCursorError("cursor payload is malformed")
    if envelope["checksum"] != _cursor_checksum(payload):
        raise InvalidCursorError("cursor checksum does not match")
    if payload["v"] != CURSOR_VERSION:
        raise InvalidCursorError("cursor version is not supported")
    if (
        isinstance(payload["offset"], bool)
        or not isinstance(payload["offset"], int)
        or payload["offset"] < 0
    ):
        raise InvalidCursorError("cursor offset is invalid")
    if not all(isinstance(payload[name], str) for name in ("revision", "record_type", "query")):
        raise InvalidCursorError("cursor fields are invalid")
    return payload


def paginate_snapshot(
    snapshot: CatalogSnapshot,
    record_type: str,
    *,
    limit: int = 100,
    cursor: Optional[str] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
    query: Any = None,
) -> CatalogPage:
    """Return a stable page and an opaque cursor for the next page.

    When a predicate is used, ``query`` must identify its immutable filter
    parameters.  This key is embedded in the cursor, preventing a cursor from
    being reused with a different filter intersection.
    """

    if not isinstance(snapshot, CatalogSnapshot):
        raise TypeError("snapshot must be a CatalogSnapshot")
    if record_type not in _RECORD_TYPES:
        raise ValueError("unknown catalog record type: %s" % record_type)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError("limit must be between 1 and %d" % MAX_PAGE_SIZE)
    if predicate is not None and query is None:
        raise ValueError("query is required when predicate is supplied")
    query_key = content_cid(
        {"schema_version": SCHEMA_VERSION, "record_type": record_type, "query": query}
    )
    records = snapshot_records(snapshot, record_type)
    if predicate is not None:
        records = tuple(item for item in records if predicate(item))

    offset = 0
    if cursor is not None:
        payload = _decode_cursor(cursor)
        if payload["revision"] != snapshot.revision:
            raise StaleCursorError("cursor belongs to a different snapshot revision")
        if payload["record_type"] != record_type:
            raise InvalidCursorError("cursor belongs to a different record type")
        if payload["query"] != query_key:
            raise InvalidCursorError("cursor belongs to a different query")
        offset = payload["offset"]
        if offset > len(records):
            raise InvalidCursorError("cursor offset exceeds the result set")

    items = tuple(records[offset : offset + limit])
    following = offset + len(items)
    next_cursor = (
        _encode_cursor(snapshot.revision, record_type, following, query_key)
        if following < len(records)
        else None
    )
    return CatalogPage(
        items=items,
        snapshot_revision=snapshot.revision,
        record_type=record_type,
        total=len(records),
        next_cursor=next_cursor,
    )


def snapshot_to_json(snapshot: CatalogSnapshot) -> str:
    """Serialize a snapshot using the catalog's canonical JSON rules."""

    if not isinstance(snapshot, CatalogSnapshot):
        raise TypeError("snapshot must be a CatalogSnapshot")
    return canonical_json(snapshot)


def snapshot_from_json(payload: str) -> CatalogSnapshot:
    """Deserialize canonical snapshot JSON and validate its revision."""

    if not isinstance(payload, str):
        raise TypeError("snapshot payload must be text")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("snapshot payload is not valid JSON") from exc
    return CatalogSnapshot.from_dict(data)


# Friendly names for callers that use serialize/deserialize terminology.
serialize_snapshot = snapshot_to_json
deserialize_snapshot = snapshot_from_json
build_snapshot = create_snapshot
paginate = paginate_snapshot
SnapshotPage = CatalogPage


__all__ = [
    "CURSOR_VERSION",
    "CatalogPage",
    "InvalidCursorError",
    "MAX_PAGE_SIZE",
    "PaginationError",
    "StaleCursorError",
    "SnapshotPage",
    "build_snapshot",
    "create_snapshot",
    "deserialize_snapshot",
    "iter_snapshot",
    "paginate_snapshot",
    "paginate",
    "serialize_snapshot",
    "snapshot_from_json",
    "snapshot_records",
    "snapshot_to_json",
]
