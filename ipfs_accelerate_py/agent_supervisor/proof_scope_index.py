"""Incremental reverse index for proof inputs and dependent evidence.

The proof contracts bind receipts to immutable inputs, but callers also need a
fast answer to the inverse question: "what evidence became stale when this
semantic input changed?"  This module builds that query surface without
granting authority to the index itself.  It retains invalidated records for
audit, while :attr:`ProofScopeIndex.active_receipt_ids` is the fail-closed view
used by proof consumers.

Incremental rebuilds take a *complete current snapshot*.  Unchanged parsed
scope blobs are reused from the preceding index; changed, deleted, and renamed
paths seed deterministic invalidation.  A cold rebuild of the same current
snapshot therefore has the same active evidence set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


PROOF_SCOPE_INDEX_SCHEMA_VERSION = 1
PROOF_SCOPE_INDEX_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-scope-index@1"
)
DEFAULT_MAX_INVALIDATION_REASON_CHAIN = 8


class ProofScopeIndexError(ValueError):
    """Raised when proof-scope index input is structurally unsafe."""


class ProofInputKind(str, Enum):
    """Semantic inputs which can make proof evidence stale."""

    FILE = "file"
    QUALIFIED_SYMBOL = "qualified_symbol"
    INTERFACE = "interface"
    ASSUMPTION = "assumption"
    TEMPLATE = "template"
    TOOLCHAIN = "toolchain"
    POLICY = "policy"


# Descriptive compatibility spellings used by early callers.
ScopeKind = ProofInputKind
ProofScopeDimension = ProofInputKind


def _canonical(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return [_canonical(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _identity(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _repo_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        return ""
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ProofScopeIndexError(f"repository path escapes its root: {value!r}")
    return path.as_posix()


def _strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        values = value.keys()
    else:
        try:
            values = iter(value)
        except TypeError:
            values = (value,)
    return tuple(
        sorted({str(item).strip() for item in values if str(item).strip()})
    )


def _record(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif hasattr(value, "to_dict"):
        projected = value.to_dict()
        if not isinstance(projected, Mapping):
            raise TypeError("to_dict() must return a mapping")
        result = dict(projected)
    elif is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
    else:
        raise TypeError(f"proof index records must be mappings, got {type(value)!r}")
    # Content-addressed contracts expose identities as properties rather than
    # constructor fields, so preserve those identities when available.
    for name in (
        "scope_id",
        "obligation_id",
        "receipt_id",
        "plan_id",
        "content_id",
    ):
        if name not in result and hasattr(value, name):
            item = getattr(value, name)
            if item not in (None, ""):
                result[name] = item
    return _canonical(result)


def _first(record: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            if isinstance(value, Enum):
                value = value.value
            return str(value).strip()
    return ""


def _many(record: Mapping[str, Any], *names: str) -> tuple[str, ...]:
    result: set[str] = set()
    for name in names:
        if name in record:
            result.update(_strings(record.get(name)))
    return tuple(sorted(result))


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True, order=True)
class ProofScopeKey:
    """Canonical lookup key for one semantic proof input."""

    kind: ProofInputKind
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProofInputKind(self.kind))
        value = _repo_path(self.value) if self.kind is ProofInputKind.FILE else str(
            self.value or ""
        ).strip()
        if not value:
            raise ProofScopeIndexError("proof scope key value must not be empty")
        object.__setattr__(self, "value", value)

    @property
    def key(self) -> str:
        return f"{self.kind.value}:{self.value}"

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "value": self.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofScopeKey":
        return cls(kind=ProofInputKind(str(payload["kind"])), value=str(payload["value"]))


@dataclass(frozen=True)
class IndexedScopeRecord:
    """Normalized parsed scope attached to one immutable source blob."""

    scope_id: str
    path: str
    blob_id: str
    keys: tuple[ProofScopeKey, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", str(self.scope_id or "").strip())
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "blob_id", str(self.blob_id or "").strip())
        object.__setattr__(self, "keys", tuple(sorted(set(self.keys))))
        object.__setattr__(self, "payload", _canonical(dict(self.payload)))
        if not self.scope_id or not self.path or not self.blob_id:
            raise ProofScopeIndexError(
                "indexed scopes require scope_id, repository path, and blob_id"
            )
        if not self.keys:
            raise ProofScopeIndexError("indexed scopes require at least one input key")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "path": self.path,
            "blob_id": self.blob_id,
            "keys": [key.to_dict() for key in self.keys],
            "payload": _canonical(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndexedScopeRecord":
        return cls(
            scope_id=str(payload["scope_id"]),
            path=str(payload["path"]),
            blob_id=str(payload["blob_id"]),
            keys=tuple(ProofScopeKey.from_dict(item) for item in payload.get("keys", ())),
            payload=payload.get("payload") or {},
        )


@dataclass(frozen=True)
class ProofScopeBlobRecord:
    """Content-addressed cache unit containing parsed proof scopes."""

    path: str
    blob_id: str
    scopes: tuple[IndexedScopeRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "blob_id", str(self.blob_id or "").strip())
        object.__setattr__(
            self, "scopes", tuple(sorted(self.scopes, key=lambda item: item.scope_id))
        )
        if not self.path or not self.blob_id:
            raise ProofScopeIndexError("scope blobs require path and blob_id")
        if any(scope.path != self.path or scope.blob_id != self.blob_id for scope in self.scopes):
            raise ProofScopeIndexError("scope blob children must share its path and blob_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "blob_id": self.blob_id,
            "scopes": [scope.to_dict() for scope in self.scopes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofScopeBlobRecord":
        return cls(
            path=str(payload["path"]),
            blob_id=str(payload["blob_id"]),
            scopes=tuple(
                IndexedScopeRecord.from_dict(item) for item in payload.get("scopes", ())
            ),
        )


@dataclass(frozen=True)
class IndexedObligation:
    obligation_id: str
    scope_ids: tuple[str, ...]
    scope_keys: tuple[ProofScopeKey, ...]
    dependency_ids: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", str(self.obligation_id or "").strip())
        object.__setattr__(self, "scope_ids", _strings(self.scope_ids))
        object.__setattr__(self, "scope_keys", tuple(sorted(set(self.scope_keys))))
        object.__setattr__(self, "dependency_ids", _strings(self.dependency_ids))
        object.__setattr__(self, "payload", _canonical(dict(self.payload)))
        if not self.obligation_id:
            raise ProofScopeIndexError("obligation identity must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "scope_ids": list(self.scope_ids),
            "scope_keys": [key.to_dict() for key in self.scope_keys],
            "dependency_ids": list(self.dependency_ids),
            "payload": _canonical(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndexedObligation":
        return cls(
            obligation_id=str(payload["obligation_id"]),
            scope_ids=tuple(payload.get("scope_ids", ())),
            scope_keys=tuple(
                ProofScopeKey.from_dict(item) for item in payload.get("scope_keys", ())
            ),
            dependency_ids=tuple(payload.get("dependency_ids", ())),
            payload=payload.get("payload") or {},
        )


@dataclass(frozen=True)
class IndexedReceipt:
    receipt_id: str
    obligation_id: str
    scope_ids: tuple[str, ...]
    scope_keys: tuple[ProofScopeKey, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", str(self.receipt_id or "").strip())
        object.__setattr__(self, "obligation_id", str(self.obligation_id or "").strip())
        object.__setattr__(self, "scope_ids", _strings(self.scope_ids))
        object.__setattr__(self, "scope_keys", tuple(sorted(set(self.scope_keys))))
        object.__setattr__(self, "payload", _canonical(dict(self.payload)))
        if not self.receipt_id or not self.obligation_id:
            raise ProofScopeIndexError("receipts require receipt_id and obligation_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "obligation_id": self.obligation_id,
            "scope_ids": list(self.scope_ids),
            "scope_keys": [key.to_dict() for key in self.scope_keys],
            "payload": _canonical(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndexedReceipt":
        return cls(
            receipt_id=str(payload["receipt_id"]),
            obligation_id=str(payload["obligation_id"]),
            scope_ids=tuple(payload.get("scope_ids", ())),
            scope_keys=tuple(
                ProofScopeKey.from_dict(item) for item in payload.get("scope_keys", ())
            ),
            payload=payload.get("payload") or {},
        )


@dataclass(frozen=True)
class InvalidationRecord:
    """Why one obligation or receipt is inactive.

    ``reason_chain`` is a bounded root-to-subject path.  It is explanatory
    provenance, not proof evidence.
    """

    subject_kind: str
    subject_id: str
    reason_code: str
    changed_input: ProofScopeKey | None = None
    reason_chain: tuple[str, ...] = ()
    chain_truncated: bool = False

    def __post_init__(self) -> None:
        kind = str(self.subject_kind or "").strip().lower()
        if kind not in {"obligation", "receipt"}:
            raise ProofScopeIndexError("invalidation subject must be obligation or receipt")
        object.__setattr__(self, "subject_kind", kind)
        object.__setattr__(self, "subject_id", str(self.subject_id or "").strip())
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        object.__setattr__(self, "reason_chain", _strings_preserving_order(self.reason_chain))
        if not self.subject_id or not self.reason_code:
            raise ProofScopeIndexError("invalidation requires subject_id and reason_code")

    @property
    def invalidation_id(self) -> str:
        return _identity(
            "proof-invalidation",
            {
                "subject_kind": self.subject_kind,
                "subject_id": self.subject_id,
                "reason_code": self.reason_code,
                "changed_input": (
                    self.changed_input.to_dict() if self.changed_input else None
                ),
                "reason_chain": list(self.reason_chain),
                "chain_truncated": self.chain_truncated,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "invalidation_id": self.invalidation_id,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "reason_code": self.reason_code,
            "changed_input": self.changed_input.to_dict() if self.changed_input else None,
            "reason_chain": list(self.reason_chain),
            "chain_truncated": self.chain_truncated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InvalidationRecord":
        changed = payload.get("changed_input")
        return cls(
            subject_kind=str(payload["subject_kind"]),
            subject_id=str(payload["subject_id"]),
            reason_code=str(payload["reason_code"]),
            changed_input=ProofScopeKey.from_dict(changed) if isinstance(changed, Mapping) else None,
            reason_chain=tuple(payload.get("reason_chain", ())),
            chain_truncated=bool(payload.get("chain_truncated", False)),
        )


def _strings_preserving_order(value: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


@dataclass(frozen=True)
class ScopeDependents:
    key: ProofScopeKey
    obligation_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.key.to_dict(),
            "obligation_ids": list(self.obligation_ids),
            "receipt_ids": list(self.receipt_ids),
        }


@dataclass(frozen=True)
class ProofScopeIndexStats:
    scanned_blob_count: int = 0
    parsed_blob_count: int = 0
    reused_blob_count: int = 0
    deleted_blob_count: int = 0
    renamed_blob_count: int = 0
    invalidated_obligation_count: int = 0
    invalidated_receipt_count: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        return (
            self.reused_blob_count / self.scanned_blob_count
            if self.scanned_blob_count
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cache_hit_ratio": self.cache_hit_ratio}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofScopeIndexStats":
        names = cls.__dataclass_fields__
        return cls(**{name: max(0, int(payload.get(name, 0))) for name in names})


@dataclass(frozen=True)
class ProofScopeIndex:
    """Immutable current index plus auditable stale evidence records."""

    blobs: tuple[ProofScopeBlobRecord, ...]
    obligations: tuple[IndexedObligation, ...]
    receipts: tuple[IndexedReceipt, ...]
    invalidations: tuple[InvalidationRecord, ...] = ()
    stats: ProofScopeIndexStats = field(default_factory=ProofScopeIndexStats)
    schema_version: int = PROOF_SCOPE_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != PROOF_SCOPE_INDEX_SCHEMA_VERSION:
            raise ProofScopeIndexError(
                f"unsupported proof scope index schema version {self.schema_version}"
            )
        object.__setattr__(self, "blobs", tuple(sorted(self.blobs, key=lambda item: item.path)))
        object.__setattr__(
            self, "obligations", tuple(sorted(self.obligations, key=lambda item: item.obligation_id))
        )
        object.__setattr__(
            self, "receipts", tuple(sorted(self.receipts, key=lambda item: item.receipt_id))
        )
        object.__setattr__(
            self,
            "invalidations",
            tuple(
                sorted(
                    _dedupe_invalidations(self.invalidations),
                    key=lambda item: (
                        item.subject_kind,
                        item.subject_id,
                        len(item.reason_chain),
                        item.reason_code,
                        item.invalidation_id,
                    ),
                )
            ),
        )
        _unique(self.blobs, "path", "scope blob path")
        scopes = [scope for blob in self.blobs for scope in blob.scopes]
        _unique(scopes, "scope_id", "scope")
        _unique(self.obligations, "obligation_id", "obligation")
        _unique(self.receipts, "receipt_id", "receipt")

    @property
    def index_id(self) -> str:
        return _identity(
            "proof-scope-index",
            {
                "schema": PROOF_SCOPE_INDEX_SCHEMA,
                "schema_version": self.schema_version,
                "blobs": [item.to_dict() for item in self.blobs],
                "obligations": [item.to_dict() for item in self.obligations],
                "receipts": [item.to_dict() for item in self.receipts],
                "invalidations": [item.to_dict() for item in self.invalidations],
            },
        )

    @property
    def scope_records(self) -> tuple[IndexedScopeRecord, ...]:
        return tuple(scope for blob in self.blobs for scope in blob.scopes)

    @property
    def invalidated_obligation_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.subject_id
                    for item in self.invalidations
                    if item.subject_kind == "obligation"
                }
            )
        )

    @property
    def invalidated_receipt_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.subject_id
                    for item in self.invalidations
                    if item.subject_kind == "receipt"
                }
            )
        )

    @property
    def active_obligation_ids(self) -> tuple[str, ...]:
        stale = set(self.invalidated_obligation_ids)
        return tuple(
            item.obligation_id for item in self.obligations if item.obligation_id not in stale
        )

    @property
    def active_receipt_ids(self) -> tuple[str, ...]:
        stale = set(self.invalidated_receipt_ids)
        return tuple(item.receipt_id for item in self.receipts if item.receipt_id not in stale)

    @property
    def active_evidence_ids(self) -> tuple[str, ...]:
        return self.active_receipt_ids

    def is_obligation_active(self, obligation_id: str) -> bool:
        return str(obligation_id) in set(self.active_obligation_ids)

    def is_receipt_active(self, receipt_id: str) -> bool:
        return str(receipt_id) in set(self.active_receipt_ids)

    def reasons_for(self, subject_id: str) -> tuple[InvalidationRecord, ...]:
        return tuple(item for item in self.invalidations if item.subject_id == subject_id)

    def dependents(
        self,
        kind: ProofInputKind | str | ProofScopeKey,
        value: str = "",
        *,
        active_only: bool = False,
    ) -> ScopeDependents:
        key = kind if isinstance(kind, ProofScopeKey) else ProofScopeKey(
            ProofInputKind(kind), value
        )
        obligation_ids = {
            item.obligation_id for item in self.obligations if key in item.scope_keys
        }
        # Receipt-only bindings such as a concrete toolchain still identify
        # the obligation whose evidence depends on that input.
        obligation_ids.update(
            item.obligation_id for item in self.receipts if key in item.scope_keys
        )
        reverse: dict[str, set[str]] = {}
        for obligation in self.obligations:
            for dependency_id in obligation.dependency_ids:
                reverse.setdefault(dependency_id, set()).add(
                    obligation.obligation_id
                )
        queue = sorted(obligation_ids)
        cursor = 0
        while cursor < len(queue):
            dependency_id = queue[cursor]
            cursor += 1
            for dependent_id in sorted(reverse.get(dependency_id, ())):
                if dependent_id not in obligation_ids:
                    obligation_ids.add(dependent_id)
                    queue.append(dependent_id)
        receipt_ids = {
            item.receipt_id
            for item in self.receipts
            if key in item.scope_keys or item.obligation_id in obligation_ids
        }
        if active_only:
            obligation_ids.intersection_update(self.active_obligation_ids)
            receipt_ids.intersection_update(self.active_receipt_ids)
        return ScopeDependents(
            key=key,
            obligation_ids=tuple(sorted(obligation_ids)),
            receipt_ids=tuple(sorted(receipt_ids)),
        )

    def obligations_for_scope(
        self, kind: ProofInputKind | str, value: str, *, active_only: bool = False
    ) -> tuple[str, ...]:
        return self.dependents(kind, value, active_only=active_only).obligation_ids

    def receipts_for_scope(
        self, kind: ProofInputKind | str, value: str, *, active_only: bool = False
    ) -> tuple[str, ...]:
        return self.dependents(kind, value, active_only=active_only).receipt_ids

    def lookup(
        self,
        kind: ProofInputKind | str | ProofScopeKey,
        value: str = "",
        *,
        active_only: bool = False,
    ) -> ScopeDependents:
        """Compatibility spelling for :meth:`dependents`."""

        return self.dependents(kind, value, active_only=active_only)

    def invalidate(
        self,
        changed_inputs: Iterable[Any],
        *,
        max_reason_chain: int = DEFAULT_MAX_INVALIDATION_REASON_CHAIN,
    ) -> "ProofScopeIndex":
        """Return an idempotently invalidated view for changed semantic inputs."""

        return invalidate_proof_scope_inputs(
            self,
            changed_inputs,
            max_reason_chain=max_reason_chain,
        )

    @property
    def scope_index(self) -> Mapping[str, ScopeDependents]:
        keys = {key for item in self.obligations for key in item.scope_keys}
        keys.update(key for item in self.receipts for key in item.scope_keys)
        return {key.key: self.dependents(key) for key in sorted(keys)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_SCOPE_INDEX_SCHEMA,
            "schema_version": self.schema_version,
            "index_id": self.index_id,
            "blobs": [item.to_dict() for item in self.blobs],
            "obligations": [item.to_dict() for item in self.obligations],
            "receipts": [item.to_dict() for item in self.receipts],
            "invalidations": [item.to_dict() for item in self.invalidations],
            "stats": self.stats.to_dict(),
        }

    def canonical_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.canonical_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofScopeIndex":
        schema = payload.get("schema")
        if schema not in (None, "", PROOF_SCOPE_INDEX_SCHEMA):
            raise ProofScopeIndexError(f"unsupported proof scope index schema {schema!r}")
        result = cls(
            schema_version=int(
                payload.get("schema_version", PROOF_SCOPE_INDEX_SCHEMA_VERSION)
            ),
            blobs=tuple(
                ProofScopeBlobRecord.from_dict(item) for item in payload.get("blobs", ())
            ),
            obligations=tuple(
                IndexedObligation.from_dict(item)
                for item in payload.get("obligations", ())
            ),
            receipts=tuple(
                IndexedReceipt.from_dict(item) for item in payload.get("receipts", ())
            ),
            invalidations=tuple(
                InvalidationRecord.from_dict(item)
                for item in payload.get("invalidations", ())
            ),
            stats=ProofScopeIndexStats.from_dict(payload.get("stats") or {}),
        )
        claimed = str(payload.get("index_id") or "")
        if claimed and claimed != result.index_id:
            raise ProofScopeIndexError("proof scope index identity does not match payload")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "ProofScopeIndex":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise ProofScopeIndexError("proof scope index JSON must contain an object")
        return cls.from_dict(payload)


def _unique(records: Iterable[Any], name: str, noun: str) -> None:
    seen: set[str] = set()
    for record in records:
        value = str(getattr(record, name))
        if value in seen:
            raise ProofScopeIndexError(f"duplicate {noun} identity {value!r}")
        seen.add(value)


def _dedupe_invalidations(
    records: Iterable[InvalidationRecord],
) -> tuple[InvalidationRecord, ...]:
    by_id = {item.invalidation_id: item for item in records}
    return tuple(by_id.values())


def _dimension_values(record: Mapping[str, Any]) -> dict[ProofInputKind, set[str]]:
    metadata = _metadata(record)
    result: dict[ProofInputKind, set[str]] = {kind: set() for kind in ProofInputKind}

    path = _first(record, "path", "new_path", "root_relative_path", "file")
    if path:
        result[ProofInputKind.FILE].add(_repo_path(path))

    aliases: dict[ProofInputKind, tuple[str, ...]] = {
        ProofInputKind.QUALIFIED_SYMBOL: (
            "qualified_symbol", "qualified_symbols", "symbol", "symbols",
        ),
        ProofInputKind.INTERFACE: (
            "interface", "interfaces", "interface_id", "interface_ids",
        ),
        ProofInputKind.ASSUMPTION: (
            "assumption", "assumptions", "assumption_id", "assumption_ids",
        ),
        ProofInputKind.TEMPLATE: ("template", "templates", "template_id", "template_ids"),
        ProofInputKind.TOOLCHAIN: (
            "toolchain", "toolchains", "toolchain_id", "toolchain_ids",
        ),
        ProofInputKind.POLICY: ("policy", "policies", "policy_id", "policy_ids"),
    }
    for kind, names in aliases.items():
        for source in (record, metadata):
            result[kind].update(_many(source, *names))

    scope_kind = _first(record, "kind", "scope_kind", "scope_type").lower()
    value = _first(record, "value")
    if scope_kind == "interface" and value:
        result[ProofInputKind.INTERFACE].add(value)
    elif scope_kind == "qualified_symbol" and value:
        result[ProofInputKind.QUALIFIED_SYMBOL].add(value)
    elif scope_kind in {kind.value for kind in ProofInputKind} and value:
        result[ProofInputKind(scope_kind)].add(value)

    # Bind template semantics as independently invalidatable assumptions while
    # retaining the human-friendly template id query.
    template_id = _first(record, "template_id")
    version = _first(record, "template_version")
    semantic_hash = _first(record, "template_semantic_hash")
    if template_id and (version or semantic_hash):
        result[ProofInputKind.TEMPLATE].add(
            f"{template_id}@{version or '?'}#{semantic_hash or '?'}"
        )
    return result


def _keys(record: Mapping[str, Any]) -> tuple[ProofScopeKey, ...]:
    values = _dimension_values(record)
    return tuple(
        sorted(
            ProofScopeKey(kind, value)
            for kind in ProofInputKind
            for value in values[kind]
            if value
        )
    )


def _scope_key(value: Any) -> ProofScopeKey:
    if isinstance(value, ProofScopeKey):
        return value
    if isinstance(value, Mapping):
        return ProofScopeKey.from_dict(value)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) != 2:
            raise ProofScopeIndexError("scope key sequences require kind and value")
        return ProofScopeKey(ProofInputKind(value[0]), str(value[1]))
    if isinstance(value, str):
        kind, separator, item = value.partition(":")
        if separator and kind in {candidate.value for candidate in ProofInputKind}:
            return ProofScopeKey(ProofInputKind(kind), item)
    raise ProofScopeIndexError(
        "changed inputs must be ProofScopeKey values, mappings, or kind/value pairs"
    )


def _blob_id(record: Mapping[str, Any], *, source: Any = None) -> str:
    existing = _first(
        record,
        "blob_id",
        "after_blob_id",
        "source_blob_id",
        "content_hash",
        "source_hash",
        "after_source_hash",
    )
    if existing:
        return existing
    material = source if source is not None else record
    if isinstance(material, bytes):
        encoded = material
    elif isinstance(material, str):
        encoded = material.encode("utf-8", errors="surrogatepass")
    else:
        encoded = _canonical_json(material).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalize_scope(
    value: Any, *, path: str = "", blob_id: str = ""
) -> IndexedScopeRecord:
    record = _record(value)
    resolved_path = _repo_path(
        path or _first(record, "path", "new_path", "root_relative_path", "file")
    )
    resolved_blob = blob_id or _blob_id(record)
    if resolved_path:
        record["path"] = resolved_path
    record["blob_id"] = resolved_blob
    keys = _keys(record)
    if not any(key.kind is ProofInputKind.FILE for key in keys):
        keys = tuple(sorted((*keys, ProofScopeKey(ProofInputKind.FILE, resolved_path))))
    scope_id = _first(record, "scope_id", "ast_scope_id", "content_id", "id")
    if not scope_id:
        scope_id = _identity(
            "proof-scope",
            {
                "blob_id": resolved_blob,
                "keys": [item.to_dict() for item in keys],
                "payload": record,
            },
        )
    return IndexedScopeRecord(
        scope_id=scope_id,
        path=resolved_path,
        blob_id=resolved_blob,
        keys=keys,
        payload=record,
    )


def _rebind_scope(
    scope: IndexedScopeRecord, *, path: str, blob_id: str
) -> IndexedScopeRecord:
    payload = dict(scope.payload)
    old_path = scope.path
    payload["path"] = path
    payload["blob_id"] = blob_id
    old_module = _module_name(old_path)
    new_module = _module_name(path)

    def requalify(value: str) -> str:
        if old_module and new_module and (
            value == old_module or value.startswith(old_module + ".")
        ):
            return new_module + value[len(old_module) :]
        return value

    # AST blob reuse is path-independent, but Python qualification is not.
    # Rebind only symbols whose prefix exactly matches the old path-derived
    # module; package-qualified values which cannot be derived safely remain
    # untouched and the old scope is still invalidated.
    for name in ("qualified_symbol", "owner_symbol"):
        value = payload.get(name)
        if isinstance(value, str):
            payload[name] = requalify(value)
    if _first(payload, "kind", "scope_kind", "scope_type").lower() in {
        "qualified_symbol",
        "interface",
    }:
        value = payload.get("value")
        if isinstance(value, str):
            payload["value"] = requalify(value)

    rebound_keys: set[ProofScopeKey] = set()
    for key in scope.keys:
        if key.kind is ProofInputKind.FILE:
            continue
        if key.kind in {
            ProofInputKind.QUALIFIED_SYMBOL,
            ProofInputKind.INTERFACE,
        }:
            rebound_keys.add(ProofScopeKey(key.kind, requalify(key.value)))
        else:
            rebound_keys.add(key)
    keys = tuple(
        sorted(
            {
                *rebound_keys,
                ProofScopeKey(ProofInputKind.FILE, path),
            }
        )
    )
    scope_id = scope.scope_id
    if old_path != path:
        scope_id = _identity(
            "proof-scope",
            {
                "blob_id": blob_id,
                "keys": [item.to_dict() for item in keys],
                "payload": payload,
            },
        )
    return IndexedScopeRecord(
        scope_id=scope_id, path=path, blob_id=blob_id, keys=keys, payload=payload
    )


def _module_name(path: str) -> str:
    normalized = _repo_path(path)
    for suffix in (".pyi", ".py"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    parts = list(PurePosixPath(normalized).parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _parse_blob(
    raw: Any,
    *,
    parser: Callable[[Any], Iterable[Any]] | None,
    cached_by_blob: Mapping[str, ProofScopeBlobRecord],
) -> tuple[ProofScopeBlobRecord, bool]:
    record = _record(raw)
    path = _repo_path(_first(record, "path", "new_path", "root_relative_path", "file"))
    source = record.get("source", record.get("content"))
    blob_id = _blob_id(record, source=source)
    if not path:
        raise ProofScopeIndexError("scope blob requires a repository path")

    explicit = record.get("scopes", record.get("scope_records", record.get("records")))
    if blob_id in cached_by_blob:
        cached = cached_by_blob[blob_id]
        if explicit is None:
            scopes = tuple(
                _rebind_scope(scope, path=path, blob_id=blob_id)
                for scope in cached.scopes
            )
        else:
            scopes = tuple(
                _normalize_scope(item, path=path, blob_id=blob_id)
                for item in _iter_records(explicit)
            )
        return ProofScopeBlobRecord(path=path, blob_id=blob_id, scopes=scopes), True

    if explicit is not None:
        parsed = _iter_records(explicit)
    elif parser is not None:
        parsed = parser(raw)
    else:
        # A file-only blob is still a valid proof input and can invalidate all
        # evidence which directly covers that file.
        parsed = ({"path": path, "blob_id": blob_id, "kind": "file", "value": path},)
    scopes = tuple(
        _normalize_scope(item, path=path, blob_id=blob_id)
        for item in _iter_records(parsed)
    )
    return ProofScopeBlobRecord(path=path, blob_id=blob_id, scopes=scopes), False


def _iter_records(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _coerce_blob_inputs(scope_blobs: Iterable[Any], scopes: Iterable[Any]) -> tuple[Any, ...]:
    blobs = list(scope_blobs)
    if not scopes:
        return tuple(blobs)
    grouped: dict[tuple[str, str], list[Any]] = {}
    for value in scopes:
        record = _record(value)
        path = _repo_path(_first(record, "path", "new_path", "root_relative_path", "file"))
        blob_id = _blob_id(record)
        grouped.setdefault((path, blob_id), []).append(value)
    blobs.extend(
        {"path": path, "blob_id": blob_id, "scopes": values}
        for (path, blob_id), values in sorted(grouped.items())
    )
    return tuple(blobs)


def _normalize_obligation(
    value: Any,
    *,
    scopes_by_id: Mapping[str, IndexedScopeRecord],
    extra_keys: Iterable[ProofScopeKey] = (),
    extra_dependencies: Iterable[str] = (),
) -> IndexedObligation:
    record = _record(value)
    obligation_id = _first(record, "obligation_id", "content_id", "id")
    if not obligation_id:
        obligation_id = _identity("proof-obligation", record)
    scope_ids = _many(record, "ast_scope_ids", "scope_ids", "proof_scope_ids")
    keys = set(_keys(record))
    for scope_id in scope_ids:
        scope = scopes_by_id.get(scope_id)
        if scope:
            keys.update(scope.keys)
    keys.update(extra_keys)
    dependencies = set(
        _many(record, "dependency_ids", "depends_on", "obligation_dependencies")
    )
    dependencies.update(extra_dependencies)
    # CodeProofObligation premise ids are proof-plan dependencies when they
    # name another indexed obligation; non-obligation premises remain
    # assumption keys and never create dangling graph edges.
    premise_ids = _many(record, "premise_ids")
    keys.update(ProofScopeKey(ProofInputKind.ASSUMPTION, item) for item in premise_ids)
    dependencies.update(premise_ids)
    dependencies.discard(obligation_id)
    return IndexedObligation(
        obligation_id=obligation_id,
        scope_ids=scope_ids,
        scope_keys=tuple(keys),
        dependency_ids=tuple(dependencies),
        payload=record,
    )


def _normalize_receipt(
    value: Any,
    *,
    scopes_by_id: Mapping[str, IndexedScopeRecord],
    obligations_by_id: Mapping[str, IndexedObligation],
) -> IndexedReceipt:
    record = _record(value)
    receipt_id = _first(
        record, "receipt_id", "proof_id", "content_id", "artifact_id", "id"
    )
    if not receipt_id:
        receipt_id = _identity("proof-receipt", record)
    obligation_id = _first(record, "obligation_id", "subject_id")
    if not obligation_id:
        raise ProofScopeIndexError(f"receipt {receipt_id!r} is missing obligation_id")
    scope_ids = _many(record, "ast_scope_ids", "scope_ids", "proof_scope_ids")
    keys = set(_keys(record))
    obligation = obligations_by_id.get(obligation_id)
    if obligation:
        keys.update(obligation.scope_keys)
    for scope_id in scope_ids:
        scope = scopes_by_id.get(scope_id)
        if scope:
            keys.update(scope.keys)
    for premise in _many(record, "premise_ids"):
        keys.add(ProofScopeKey(ProofInputKind.ASSUMPTION, premise))
    return IndexedReceipt(
        receipt_id=receipt_id,
        obligation_id=obligation_id,
        scope_ids=scope_ids,
        scope_keys=tuple(keys),
        payload=record,
    )


def _plan_links(
    plans: Iterable[Any],
) -> tuple[dict[str, set[str]], dict[str, set[ProofScopeKey]]]:
    dependencies: dict[str, set[str]] = {}
    keys: dict[str, set[ProofScopeKey]] = {}
    for value in plans:
        plan = _record(value)
        policy_id = _first(plan, "policy_id")
        plan_keys = set(_keys(plan))
        if policy_id:
            plan_keys.add(ProofScopeKey(ProofInputKind.POLICY, policy_id))
        steps = [_record(item) for item in _iter_records(plan.get("steps"))]
        step_obligation = {
            _first(step, "step_id", "node_id", "id"): _first(step, "obligation_id")
            for step in steps
        }
        for step in steps:
            obligation_id = _first(step, "obligation_id")
            if not obligation_id:
                continue
            keys.setdefault(obligation_id, set()).update(plan_keys)
            keys[obligation_id].update(_keys(step))
            for dependency_step in _many(step, "depends_on", "dependency_ids"):
                dependency_obligation = step_obligation.get(dependency_step, dependency_step)
                if dependency_obligation and dependency_obligation != obligation_id:
                    dependencies.setdefault(obligation_id, set()).add(
                        dependency_obligation
                    )
    return dependencies, keys


def _bounded_chain(
    chain: Sequence[str], item: str, maximum: int
) -> tuple[tuple[str, ...], bool]:
    candidate = tuple(chain) + (item,)
    if len(candidate) <= maximum:
        return candidate, False
    return candidate[:maximum], True


def build_proof_scope_index(
    *,
    scope_blobs: Iterable[Any] = (),
    scopes: Iterable[Any] = (),
    scope_records: Iterable[Any] = (),
    obligations: Iterable[Any] = (),
    proof_obligations: Iterable[Any] = (),
    receipts: Iterable[Any] = (),
    proof_receipts: Iterable[Any] = (),
    proof_plans: Iterable[Any] = (),
    plans: Iterable[Any] = (),
    previous: ProofScopeIndex | Mapping[str, Any] | None = None,
    parser: Callable[[Any], Iterable[Any]] | None = None,
    changed_inputs: Iterable[Any] = (),
    invalidated_scopes: Iterable[Any] = (),
    max_reason_chain: int = DEFAULT_MAX_INVALIDATION_REASON_CHAIN,
    exhaustive: bool = False,
) -> ProofScopeIndex:
    """Build or incrementally rebuild a deterministic proof scope index.

    All iterable inputs describe the complete current snapshot.  ``previous``
    only supplies the blob cache and historical evidence needed to explain
    invalidation; it never makes omitted current evidence active.
    """

    if isinstance(max_reason_chain, bool) or int(max_reason_chain) < 1:
        raise ValueError("max_reason_chain must be a positive integer")
    maximum = int(max_reason_chain)
    if previous is not None and not isinstance(previous, ProofScopeIndex):
        previous = ProofScopeIndex.from_dict(previous)
    prior = None if exhaustive else previous
    cached_by_blob = (
        {blob.blob_id: blob for blob in prior.blobs} if prior is not None else {}
    )

    raw_blobs = _coerce_blob_inputs(
        scope_blobs, (*tuple(scopes), *tuple(scope_records))
    )
    current_blobs: list[ProofScopeBlobRecord] = []
    reused = 0
    parsed = 0
    for raw in raw_blobs:
        blob, was_reused = _parse_blob(
            raw, parser=parser, cached_by_blob=cached_by_blob
        )
        current_blobs.append(blob)
        reused += int(was_reused)
        parsed += int(not was_reused)
    current_blobs.sort(key=lambda item: item.path)
    _unique(current_blobs, "path", "scope blob path")

    scopes_by_id = {
        scope.scope_id: scope for blob in current_blobs for scope in blob.scopes
    }
    if len(scopes_by_id) != sum(len(blob.scopes) for blob in current_blobs):
        raise ProofScopeIndexError("scope identities must be unique")

    plan_dependencies, plan_keys = _plan_links(
        (*tuple(proof_plans), *tuple(plans))
    )
    raw_obligations = (*tuple(obligations), *tuple(proof_obligations))
    normalized_obligations: list[IndexedObligation] = []
    for value in raw_obligations:
        record = _record(value)
        obligation_id = _first(record, "obligation_id", "content_id", "id")
        normalized_obligations.append(
            _normalize_obligation(
                value,
                scopes_by_id=scopes_by_id,
                extra_keys=plan_keys.get(obligation_id, ()),
                extra_dependencies=plan_dependencies.get(obligation_id, ()),
            )
        )
    obligations_by_id = {
        item.obligation_id: item for item in normalized_obligations
    }
    if len(obligations_by_id) != len(normalized_obligations):
        raise ProofScopeIndexError("obligation identities must be unique")

    normalized_receipts = [
        _normalize_receipt(
            value,
            scopes_by_id=scopes_by_id,
            obligations_by_id=obligations_by_id,
        )
        for value in (*tuple(receipts), *tuple(proof_receipts))
    ]
    receipts_by_id = {item.receipt_id: item for item in normalized_receipts}
    if len(receipts_by_id) != len(normalized_receipts):
        raise ProofScopeIndexError("receipt identities must be unique")

    invalidations: list[InvalidationRecord] = []
    obligation_reasons: dict[str, InvalidationRecord] = {}
    receipt_reasons: dict[str, InvalidationRecord] = {}

    if prior is not None:
        for historical in prior.invalidations:
            chain = historical.reason_chain[:maximum]
            retained = InvalidationRecord(
                subject_kind=historical.subject_kind,
                subject_id=historical.subject_id,
                reason_code=historical.reason_code,
                changed_input=historical.changed_input,
                reason_chain=chain,
                chain_truncated=bool(
                    historical.chain_truncated
                    or len(historical.reason_chain) > maximum
                ),
            )
            invalidations.append(retained)
            if retained.subject_kind == "obligation":
                obligation_reasons[retained.subject_id] = retained
            else:
                receipt_reasons[retained.subject_id] = retained

    def invalidate_obligation(
        obligation_id: str,
        reason_code: str,
        changed_input: ProofScopeKey | None,
        chain: Sequence[str],
        truncated: bool = False,
    ) -> bool:
        item_chain, clipped = _bounded_chain(chain, f"obligation:{obligation_id}", maximum)
        candidate = InvalidationRecord(
            subject_kind="obligation",
            subject_id=obligation_id,
            reason_code=reason_code,
            changed_input=changed_input,
            reason_chain=item_chain,
            chain_truncated=bool(truncated or clipped),
        )
        existing = obligation_reasons.get(obligation_id)
        if existing is None or (
            len(candidate.reason_chain), candidate.invalidation_id
        ) < (len(existing.reason_chain), existing.invalidation_id):
            obligation_reasons[obligation_id] = candidate
            return existing is None
        return False

    current_paths = {blob.path: blob for blob in current_blobs}
    deleted_count = 0
    renamed_count = 0
    if prior is not None:
        current_blob_paths: dict[str, set[str]] = {}
        for blob in current_blobs:
            current_blob_paths.setdefault(blob.blob_id, set()).add(blob.path)
        prior_obligations_by_scope: dict[str, set[str]] = {}
        for obligation in prior.obligations:
            for scope_id in obligation.scope_ids:
                prior_obligations_by_scope.setdefault(scope_id, set()).add(
                    obligation.obligation_id
                )
        for old_blob in prior.blobs:
            current_at_path = current_paths.get(old_blob.path)
            if current_at_path and current_at_path.blob_id == old_blob.blob_id:
                continue
            if old_blob.blob_id in current_blob_paths:
                reason = "scope_renamed"
                renamed_count += 1
            elif current_at_path is not None:
                reason = "scope_changed"
            else:
                reason = "scope_deleted"
                deleted_count += 1
            for scope in old_blob.scopes:
                file_key = ProofScopeKey(ProofInputKind.FILE, old_blob.path)
                root_chain = (f"input:{file_key.key}",)
                dependents = set(prior_obligations_by_scope.get(scope.scope_id, ()))
                dependents.update(
                    item.obligation_id
                    for item in prior.obligations
                    if set(item.scope_keys).intersection(scope.keys)
                )
                for obligation_id in sorted(dependents):
                    invalidate_obligation(
                        obligation_id, reason, file_key, root_chain
                    )

    known_scope_ids = set(scopes_by_id)
    for obligation in normalized_obligations:
        missing = sorted(set(obligation.scope_ids) - known_scope_ids)
        if missing:
            root = f"missing_scope:{missing[0]}"
            invalidate_obligation(
                obligation.obligation_id,
                "missing_scope",
                None,
                (root,),
            )

    known_obligation_ids = set(obligations_by_id)
    reverse_dependencies: dict[str, set[str]] = {}
    dependency_records: Iterable[IndexedObligation] = normalized_obligations
    if prior is not None:
        dependency_records = (*tuple(normalized_obligations), *prior.obligations)
    for obligation in dependency_records:
        for dependency in obligation.dependency_ids:
            reverse_dependencies.setdefault(dependency, set()).add(
                obligation.obligation_id
            )

    queue = sorted(obligation_reasons)
    cursor = 0
    while cursor < len(queue):
        dependency_id = queue[cursor]
        cursor += 1
        parent = obligation_reasons[dependency_id]
        for dependent_id in sorted(reverse_dependencies.get(dependency_id, ())):
            added = invalidate_obligation(
                dependent_id,
                "dependency_invalidated",
                parent.changed_input,
                parent.reason_chain,
                parent.chain_truncated,
            )
            if added:
                queue.append(dependent_id)

    for receipt in normalized_receipts:
        reason: InvalidationRecord | None = obligation_reasons.get(receipt.obligation_id)
        reason_code = "obligation_invalidated"
        chain: Sequence[str]
        changed_input: ProofScopeKey | None
        truncated = False
        if reason is not None:
            chain = reason.reason_chain
            changed_input = reason.changed_input
            truncated = reason.chain_truncated
        elif receipt.obligation_id not in known_obligation_ids:
            reason_code = "missing_obligation"
            chain = (f"missing_obligation:{receipt.obligation_id}",)
            changed_input = None
        else:
            missing = sorted(set(receipt.scope_ids) - known_scope_ids)
            if not missing:
                continue
            reason_code = "missing_scope"
            chain = (f"missing_scope:{missing[0]}",)
            changed_input = None
        item_chain, clipped = _bounded_chain(
            chain, f"receipt:{receipt.receipt_id}", maximum
        )
        receipt_reasons[receipt.receipt_id] = InvalidationRecord(
            subject_kind="receipt",
            subject_id=receipt.receipt_id,
            reason_code=reason_code,
            changed_input=changed_input,
            reason_chain=item_chain,
            chain_truncated=bool(truncated or clipped),
        )

    if prior is not None:
        for receipt in prior.receipts:
            if receipt.receipt_id in receipts_by_id:
                continue
            reason = obligation_reasons.get(receipt.obligation_id)
            if reason is None:
                continue
            item_chain, clipped = _bounded_chain(
                reason.reason_chain, f"receipt:{receipt.receipt_id}", maximum
            )
            receipt_reasons[receipt.receipt_id] = InvalidationRecord(
                subject_kind="receipt",
                subject_id=receipt.receipt_id,
                reason_code="obligation_invalidated",
                changed_input=reason.changed_input,
                reason_chain=item_chain,
                chain_truncated=bool(reason.chain_truncated or clipped),
            )

    invalidations.extend(obligation_reasons.values())
    invalidations.extend(receipt_reasons.values())
    result_stats = ProofScopeIndexStats(
        scanned_blob_count=len(current_blobs),
        parsed_blob_count=parsed,
        reused_blob_count=reused,
        deleted_blob_count=deleted_count,
        renamed_blob_count=renamed_count,
        invalidated_obligation_count=len(obligation_reasons),
        invalidated_receipt_count=len(receipt_reasons),
    )
    result = ProofScopeIndex(
        blobs=tuple(current_blobs),
        obligations=tuple(normalized_obligations),
        receipts=tuple(normalized_receipts),
        invalidations=tuple(invalidations),
        stats=result_stats,
    )
    explicit_changes = (*tuple(changed_inputs), *tuple(invalidated_scopes))
    if explicit_changes:
        result = invalidate_proof_scope_inputs(
            result,
            explicit_changes,
            max_reason_chain=maximum,
        )
    return result


def invalidate_proof_scope_inputs(
    index: ProofScopeIndex | Mapping[str, Any],
    changed_inputs: Iterable[Any],
    *,
    max_reason_chain: int = DEFAULT_MAX_INVALIDATION_REASON_CHAIN,
) -> ProofScopeIndex:
    """Invalidate all transitive dependents of explicit semantic changes.

    This covers non-file inputs (templates, toolchains, and policies) whose
    changes are discovered outside an AST scan.  Repeating the same call is
    idempotent because invalidation identities are content addressed.
    """

    if not isinstance(index, ProofScopeIndex):
        index = ProofScopeIndex.from_dict(index)
    if isinstance(max_reason_chain, bool) or int(max_reason_chain) < 1:
        raise ValueError("max_reason_chain must be a positive integer")
    maximum = int(max_reason_chain)
    keys = tuple(sorted({_scope_key(value) for value in changed_inputs}))
    if not keys:
        return index

    additions: list[InvalidationRecord] = []
    reasons: dict[str, InvalidationRecord] = {}
    for key in keys:
        direct = {
            item.obligation_id
            for item in index.obligations
            if key in item.scope_keys
        }
        direct.update(
            item.obligation_id for item in index.receipts if key in item.scope_keys
        )
        for obligation_id in sorted(direct):
            chain, clipped = _bounded_chain(
                (f"input:{key.key}",), f"obligation:{obligation_id}", maximum
            )
            reason = InvalidationRecord(
                subject_kind="obligation",
                subject_id=obligation_id,
                reason_code="input_changed",
                changed_input=key,
                reason_chain=chain,
                chain_truncated=clipped,
            )
            additions.append(reason)
            current = reasons.get(obligation_id)
            if current is None or (
                len(reason.reason_chain), reason.invalidation_id
            ) < (len(current.reason_chain), current.invalidation_id):
                reasons[obligation_id] = reason

    reverse: dict[str, set[str]] = {}
    for obligation in index.obligations:
        for dependency_id in obligation.dependency_ids:
            reverse.setdefault(dependency_id, set()).add(obligation.obligation_id)
    queue = sorted(reasons)
    cursor = 0
    while cursor < len(queue):
        dependency_id = queue[cursor]
        cursor += 1
        parent = reasons[dependency_id]
        for dependent_id in sorted(reverse.get(dependency_id, ())):
            if dependent_id in reasons:
                continue
            chain, clipped = _bounded_chain(
                parent.reason_chain, f"obligation:{dependent_id}", maximum
            )
            reason = InvalidationRecord(
                subject_kind="obligation",
                subject_id=dependent_id,
                reason_code="dependency_invalidated",
                changed_input=parent.changed_input,
                reason_chain=chain,
                chain_truncated=bool(parent.chain_truncated or clipped),
            )
            reasons[dependent_id] = reason
            additions.append(reason)
            queue.append(dependent_id)

    for receipt in index.receipts:
        if receipt.obligation_id not in reasons:
            continue
        parent = reasons[receipt.obligation_id]
        chain, clipped = _bounded_chain(
            parent.reason_chain, f"receipt:{receipt.receipt_id}", maximum
        )
        additions.append(
            InvalidationRecord(
                subject_kind="receipt",
                subject_id=receipt.receipt_id,
                reason_code="obligation_invalidated",
                changed_input=parent.changed_input,
                reason_chain=chain,
                chain_truncated=bool(parent.chain_truncated or clipped),
            )
        )

    all_invalidations = _dedupe_invalidations((*index.invalidations, *additions))
    stale_obligations = {
        item.subject_id
        for item in all_invalidations
        if item.subject_kind == "obligation"
    }
    stale_receipts = {
        item.subject_id
        for item in all_invalidations
        if item.subject_kind == "receipt"
    }
    return ProofScopeIndex(
        blobs=index.blobs,
        obligations=index.obligations,
        receipts=index.receipts,
        invalidations=all_invalidations,
        stats=ProofScopeIndexStats(
            scanned_blob_count=index.stats.scanned_blob_count,
            parsed_blob_count=index.stats.parsed_blob_count,
            reused_blob_count=index.stats.reused_blob_count,
            deleted_blob_count=index.stats.deleted_blob_count,
            renamed_blob_count=index.stats.renamed_blob_count,
            invalidated_obligation_count=len(stale_obligations),
            invalidated_receipt_count=len(stale_receipts),
        ),
    )


def rebuild_proof_scope_index(**kwargs: Any) -> ProofScopeIndex:
    """Compatibility wrapper for :func:`build_proof_scope_index`."""

    return build_proof_scope_index(**kwargs)


def update_proof_scope_index(
    previous: ProofScopeIndex | Mapping[str, Any], **kwargs: Any
) -> ProofScopeIndex:
    """Incrementally rebuild an index from a complete current snapshot."""

    return build_proof_scope_index(previous=previous, **kwargs)


ProofScopeDependencyIndex = ProofScopeIndex
ProofScopeIndexRecord = ScopeDependents
ScopeIndex = ProofScopeIndex


__all__ = [
    "DEFAULT_MAX_INVALIDATION_REASON_CHAIN",
    "IndexedObligation",
    "IndexedReceipt",
    "IndexedScopeRecord",
    "InvalidationRecord",
    "PROOF_SCOPE_INDEX_SCHEMA",
    "PROOF_SCOPE_INDEX_SCHEMA_VERSION",
    "ProofInputKind",
    "ProofScopeBlobRecord",
    "ProofScopeDependencyIndex",
    "ProofScopeDimension",
    "ProofScopeIndex",
    "ProofScopeIndexError",
    "ProofScopeIndexRecord",
    "ProofScopeIndexStats",
    "ProofScopeKey",
    "ScopeDependents",
    "ScopeIndex",
    "ScopeKind",
    "build_proof_scope_index",
    "invalidate_proof_scope_inputs",
    "rebuild_proof_scope_index",
    "update_proof_scope_index",
]
