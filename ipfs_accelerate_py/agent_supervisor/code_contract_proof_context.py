"""Minimal, dependency-complete contexts for code-contract proof work.

This module is intentionally a data compiler, not a retrieval or language
model component.  It accepts already-normalized symbolic facts, selects the
transitive closure for exactly one obligation, and emits canonical receipts.
Source text and full program graphs remain behind content-addressed expansion
handles.

The important fail-closed property is that limits never remove a required
item.  A required closure which exceeds a limit is returned intact with
``status=incomplete``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar

from .proof.formal_verification_contracts import (
    canonical_json,
    canonical_json_bytes,
    content_identity,
)


PROOF_CONTEXT_VERSION = 1
MINIMAL_PROOF_CONTEXT_EVIDENCE = "vfs/minimal-proof-context@1"
PROOF_CONTEXT_ITEM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-proof-context-item@1"
)
PROOF_CONTEXT_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-proof-context-request@1"
)
PROOF_CONTEXT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-proof-context@1"
)
PROOF_CONTEXT_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-proof-context-receipt@1"
)
PROOF_CONTEXT_DELTA_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-proof-context-delta@1"
)
EXPANSION_HANDLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-context-expansion-handle@1"
)
DEFAULT_MAX_CONTEXT_BYTES = 12_288
DEFAULT_MAX_CONTEXT_ITEMS = 256

_FORBIDDEN_EMBEDDED_KEYS = frozenset(
    {
        "ast",
        "body",
        "contents",
        "file_contents",
        "full_graph",
        "full_source",
        "graph",
        "source",
        "source_body",
        "source_code",
        "source_text",
    }
)


class CodeContractProofContextError(ValueError):
    """Raised for a malformed or unsafe context input."""


# Compatibility spelling for callers which use the shorter contract name.
ProofContextValidationError = CodeContractProofContextError


class ProofContextItemKind(str, Enum):
    """Kinds admitted to a symbolic proof context."""

    OBLIGATION = "obligation"
    CONTRACT = "contract"
    CALL = "call"
    DEFINITION = "definition"
    ASSUMPTION = "assumption"
    EFFECT = "effect"
    RULE = "rule"
    COUNTEREXAMPLE = "counterexample"
    EVIDENCE = "evidence"


ProofContextKind = ProofContextItemKind


class ProofContextStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALIDATED = "invalidated"


class ProofContextDecision(str, Enum):
    INCLUDED = "included"
    EXCLUDED = "excluded"
    MISSING = "missing"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodeContractProofContextError(f"{name} is required")
    return value.strip()


def _optional_text(value: Any, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise CodeContractProofContextError(f"{name} must be a string")
    return value.strip()


def _sorted_ids(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise CodeContractProofContextError(
            f"{name} values must be an iterable of identifiers"
        )
    result: set[str] = set()
    for value in values or ():
        result.add(_required_text(value, name))
    return tuple(sorted(result))


def _plain(value: Any, path: str = "value") -> Any:
    """Return immutable-input data as canonical plain data.

    Floats, bytes and source/graph body keys are rejected here rather than
    silently omitted: a receipt must describe precisely what was admitted.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        raise CodeContractProofContextError(f"{path} cannot contain floats")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise CodeContractProofContextError(f"{path} cannot contain bytes")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise CodeContractProofContextError(
                    f"{path} keys must be non-empty strings"
                )
            key = raw_key.strip()
            if key.casefold() in _FORBIDDEN_EMBEDDED_KEYS:
                raise CodeContractProofContextError(
                    f"{path}.{key} embeds source or a full graph; use an "
                    "expansion handle"
                )
            result[key] = _plain(raw_value, f"{path}.{key}")
        return result
    if isinstance(value, Sequence):
        return [_plain(item, f"{path}[]") for item in value]
    raise CodeContractProofContextError(
        f"{path} has unsupported value type {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    """Recursively freeze canonical data retained by a frozen dataclass."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return a detached plain copy suitable for serialization."""

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_thaw(item) for item in value]
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value or "").strip())
    except ValueError as exc:
        raise CodeContractProofContextError(
            f"unsupported {name}: {value!r}"
        ) from exc


@dataclass(frozen=True)
class ProofContextLimits:
    """Non-destructive admission limits for a compiled context."""

    max_bytes: int = DEFAULT_MAX_CONTEXT_BYTES
    max_items: int = DEFAULT_MAX_CONTEXT_ITEMS

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_items"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CodeContractProofContextError(
                    f"{name} must be a positive integer"
                )

    def to_dict(self) -> dict[str, int]:
        return {"max_bytes": self.max_bytes, "max_items": self.max_items}

    @classmethod
    def from_value(cls, value: Any) -> "ProofContextLimits":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise CodeContractProofContextError("limits must be a mapping")
        return cls(
            max_bytes=value.get("max_bytes", DEFAULT_MAX_CONTEXT_BYTES),
            max_items=value.get("max_items", DEFAULT_MAX_CONTEXT_ITEMS),
        )


@dataclass(frozen=True)
class ProofContextItem:
    """One normalized symbolic fact.

    ``payload`` should be a compact IR fragment.  ``expansion_locator`` may be
    a CID, record key, or other resolver-owned opaque locator; its referenced
    body is deliberately absent.
    """

    item_id: str
    kind: ProofContextItemKind
    payload: Mapping[str, Any]
    dependency_ids: tuple[str, ...] = ()
    expansion_locator: str = ""
    referenced_content_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = PROOF_CONTEXT_ITEM_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _required_text(self.item_id, "item_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProofContextItemKind, "item kind")
        )
        if not isinstance(self.payload, Mapping):
            raise CodeContractProofContextError("payload must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise CodeContractProofContextError("metadata must be a mapping")
        object.__setattr__(
            self,
            "payload",
            _freeze(_plain(self.payload, "payload")),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze(_plain(self.metadata, "metadata")),
        )
        object.__setattr__(
            self,
            "dependency_ids",
            _sorted_ids(self.dependency_ids, "dependency_id"),
        )
        object.__setattr__(
            self,
            "expansion_locator",
            _optional_text(self.expansion_locator, "expansion_locator"),
        )
        object.__setattr__(
            self,
            "referenced_content_id",
            _optional_text(self.referenced_content_id, "referenced_content_id"),
        )
        if self.item_id in self.dependency_ids:
            # Self references add no information and make "smallest" ambiguous.
            raise CodeContractProofContextError(
                f"item {self.item_id!r} cannot depend on itself"
            )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict(include_content_id=False))

    @property
    def byte_count(self) -> int:
        return len(canonical_json_bytes(self.to_dict()))

    def to_dict(self, *, include_content_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "item_id": self.item_id,
            "kind": self.kind.value,
            "payload": _thaw(self.payload),
            "dependency_ids": list(self.dependency_ids),
            "expansion_locator": self.expansion_locator,
            "referenced_content_id": self.referenced_content_id,
            "metadata": _thaw(self.metadata),
        }
        if include_content_id:
            result["content_id"] = self.content_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofContextItem":
        if not isinstance(value, Mapping):
            raise CodeContractProofContextError("item must be a mapping")
        return cls(
            item_id=value.get("item_id", ""),
            kind=value.get("kind", ""),
            payload=value.get("payload") or {},
            dependency_ids=tuple(value.get("dependency_ids") or ()),
            expansion_locator=value.get("expansion_locator", ""),
            referenced_content_id=value.get("referenced_content_id", ""),
            metadata=value.get("metadata") or {},
        )


# More specific public spellings used by a few callers.
ProofContextArtifact = ProofContextItem


@dataclass(frozen=True)
class ExpansionHandle:
    """Content-addressed way to expand an omitted or summarized fact."""

    item_id: str
    item_kind: str
    target_content_id: str
    locator: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        for name in ("item_id", "item_kind", "target_content_id"):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(self, "locator", _optional_text(self.locator, "locator"))
        object.__setattr__(self, "reason", _optional_text(self.reason, "reason"))

    @property
    def handle_id(self) -> str:
        return content_identity(self.to_dict(include_handle_id=False))

    def to_dict(self, *, include_handle_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": EXPANSION_HANDLE_SCHEMA,
            "item_id": self.item_id,
            "item_kind": self.item_kind,
            "target_content_id": self.target_content_id,
            "locator": self.locator,
            "reason": self.reason,
        }
        if include_handle_id:
            result["handle_id"] = self.handle_id
        return result


ProofContextExpansionHandle = ExpansionHandle


@dataclass(frozen=True)
class InclusionDecision:
    """Auditable deterministic decision for a candidate or missing item."""

    item_id: str
    decision: ProofContextDecision
    reason: str
    required_by: tuple[str, ...] = ()
    item_content_id: str = ""
    expansion_handle_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _required_text(self.item_id, "item_id"))
        object.__setattr__(
            self,
            "decision",
            _enum(self.decision, ProofContextDecision, "decision"),
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(
            self,
            "required_by",
            _sorted_ids(self.required_by, "required_by"),
        )
        for name in ("item_content_id", "expansion_handle_id"):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name)
            )

    @property
    def included(self) -> bool:
        return self.decision is ProofContextDecision.INCLUDED

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "decision": self.decision.value,
            "included": self.included,
            "reason": self.reason,
            "required_by": list(self.required_by),
            "item_content_id": self.item_content_id,
            "expansion_handle_id": self.expansion_handle_id,
        }


ProofContextInclusionDecision = InclusionDecision


@dataclass(frozen=True)
class ProgramGraphSliceReference:
    """Bounded reference to a VFS-013 slice, never the slice body."""

    slice_id: str
    query_id: str = ""
    forest_id: str = ""
    graph_id: str = ""
    complete: bool = False
    minimal: bool = False
    dependency_complete: bool = False
    truncated: bool = False
    node_count: int = 0
    edge_count: int = 0
    missing_node_ids: tuple[str, ...] = ()
    omitted_dependencies: tuple[str, ...] = ()
    truncation_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_id", _required_text(self.slice_id, "slice_id"))
        for name in ("query_id", "forest_id", "graph_id"):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name)
            )
        for name in ("complete", "minimal", "dependency_complete", "truncated"):
            if not isinstance(getattr(self, name), bool):
                raise CodeContractProofContextError(f"{name} must be a boolean")
        for name in ("node_count", "edge_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CodeContractProofContextError(
                    f"{name} must be a non-negative integer"
                )
        for name in (
            "missing_node_ids",
            "omitted_dependencies",
            "truncation_reasons",
        ):
            object.__setattr__(
                self, name, _sorted_ids(getattr(self, name), name)
            )

    @property
    def incomplete(self) -> bool:
        return bool(
            not self.complete
            or not self.dependency_complete
            or self.truncated
            or self.missing_node_ids
            or self.omitted_dependencies
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "query_id": self.query_id,
            "forest_id": self.forest_id,
            "graph_id": self.graph_id,
            "complete": self.complete,
            "minimal": self.minimal,
            "dependency_complete": self.dependency_complete,
            "truncated": self.truncated,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "missing_node_ids": list(self.missing_node_ids),
            "omitted_dependencies": list(self.omitted_dependencies),
            "truncation_reasons": list(self.truncation_reasons),
            "embeds_source_bodies": False,
            "embeds_full_graph": False,
        }

    @classmethod
    def from_value(cls, value: Any) -> "ProgramGraphSliceReference | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            raw = dict(value)
        elif hasattr(value, "to_dict"):
            raw = dict(value.to_dict())
        else:
            raise CodeContractProofContextError(
                "program_graph_slice must be a mapping or ProgramGraphSlice"
            )
        raw_slice_id = raw.get("slice_id")
        if raw_slice_id is None or raw_slice_id == "":
            # Hashing the input is safe; only the bounded reference is retained.
            slice_id = content_identity(raw)
        else:
            slice_id = _required_text(raw_slice_id, "slice_id")
        node_ids = raw.get("node_ids", ())
        edge_ids = raw.get("edge_ids", ())
        for name, values in (("node_ids", node_ids), ("edge_ids", edge_ids)):
            if values is None:
                continue
            if isinstance(values, (str, bytes, bytearray, memoryview)):
                raise CodeContractProofContextError(
                    f"{name} must be a collection"
                )
            if not isinstance(values, (Sequence, set, frozenset)):
                raise CodeContractProofContextError(
                    f"{name} must be a collection"
                )
        return cls(
            slice_id=slice_id,
            query_id=raw.get("query_id", ""),
            forest_id=raw.get("forest_id", ""),
            graph_id=raw.get("graph_id", ""),
            complete=raw.get("complete", False),
            minimal=raw.get("minimal", False),
            dependency_complete=raw.get("dependency_complete", False),
            truncated=raw.get("truncated", False),
            node_count=(
                raw["node_count"]
                if "node_count" in raw
                else len(node_ids or ())
            ),
            edge_count=(
                raw["edge_count"]
                if "edge_count" in raw
                else len(edge_ids or ())
            ),
            missing_node_ids=tuple(raw.get("missing_node_ids") or ()),
            omitted_dependencies=tuple(raw.get("omitted_dependencies") or ()),
            truncation_reasons=tuple(raw.get("truncation_reasons") or ()),
        )


@dataclass(frozen=True)
class ProofContextRequest:
    """Compilation request for exactly one proof obligation."""

    obligation_id: str
    items: tuple[ProofContextItem, ...]
    required_item_ids: tuple[str, ...] = ()
    limits: ProofContextLimits = field(default_factory=ProofContextLimits)
    program_graph_slice: Any = None
    policy_id: str = "policy:minimal-code-contract-proof-context@1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = PROOF_CONTEXT_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obligation_id", _required_text(self.obligation_id, "obligation_id")
        )
        items: list[ProofContextItem] = []
        for value in self.items or ():
            item = (
                value
                if isinstance(value, ProofContextItem)
                else ProofContextItem.from_dict(value)
            )
            items.append(item)
        ids = [item.item_id for item in items]
        if len(ids) != len(set(ids)):
            raise CodeContractProofContextError("item_id values must be unique")
        object.__setattr__(
            self, "items", tuple(sorted(items, key=lambda item: item.item_id))
        )
        object.__setattr__(
            self,
            "required_item_ids",
            _sorted_ids(self.required_item_ids, "required_item_id"),
        )
        object.__setattr__(self, "limits", ProofContextLimits.from_value(self.limits))
        object.__setattr__(
            self,
            "program_graph_slice",
            ProgramGraphSliceReference.from_value(self.program_graph_slice),
        )
        object.__setattr__(self, "policy_id", _required_text(self.policy_id, "policy_id"))
        if not isinstance(self.metadata, Mapping):
            raise CodeContractProofContextError("metadata must be a mapping")
        object.__setattr__(
            self, "metadata", _freeze(_plain(self.metadata, "metadata"))
        )

    @property
    def request_id(self) -> str:
        return content_identity(self.to_dict(include_request_id=False))

    def to_dict(self, *, include_request_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "version": PROOF_CONTEXT_VERSION,
            "obligation_id": self.obligation_id,
            "items": [item.to_dict() for item in self.items],
            "required_item_ids": list(self.required_item_ids),
            "limits": self.limits.to_dict(),
            "program_graph_slice": (
                self.program_graph_slice.to_dict()
                if self.program_graph_slice is not None
                else None
            ),
            "policy_id": self.policy_id,
            "metadata": _thaw(self.metadata),
        }
        if include_request_id:
            result["request_id"] = self.request_id
        return result


CodeContractProofContextRequest = ProofContextRequest


@dataclass(frozen=True)
class ProofContextMetrics:
    """Exact non-LLM measurements of the transmitted payload."""

    item_count: int
    byte_count: int
    item_bytes: int
    max_items: int
    max_bytes: int
    measurement: str = "canonical-json-utf8"
    llm_invocations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_count": self.item_count,
            "byte_count": self.byte_count,
            "item_bytes": self.item_bytes,
            "max_items": self.max_items,
            "max_bytes": self.max_bytes,
            "measurement": self.measurement,
            "llm_invocations": self.llm_invocations,
        }


@dataclass(frozen=True)
class ProofContextReceipt:
    """Durable binding between dependencies and one compiled context."""

    context_id: str
    request_id: str
    obligation_id: str
    dependency_fingerprint: str
    included_item_ids: tuple[str, ...]
    included_content_ids: tuple[str, ...]
    status: ProofContextStatus
    incomplete_reasons: tuple[str, ...]
    metrics: ProofContextMetrics
    decision_digest: str
    invalidated_receipt_ids: tuple[str, ...] = ()

    SCHEMA: ClassVar[str] = PROOF_CONTEXT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "context_id",
            "request_id",
            "obligation_id",
            "dependency_fingerprint",
            "decision_digest",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(
            self, "status", _enum(self.status, ProofContextStatus, "status")
        )
        for name in (
            "included_item_ids",
            "included_content_ids",
            "incomplete_reasons",
            "invalidated_receipt_ids",
        ):
            object.__setattr__(
                self, name, _sorted_ids(getattr(self, name), name)
            )
        if not isinstance(self.metrics, ProofContextMetrics):
            raise CodeContractProofContextError(
                "metrics must be ProofContextMetrics"
            )

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "version": PROOF_CONTEXT_VERSION,
            "evidence": MINIMAL_PROOF_CONTEXT_EVIDENCE,
            "context_id": self.context_id,
            "request_id": self.request_id,
            "obligation_id": self.obligation_id,
            "dependency_fingerprint": self.dependency_fingerprint,
            "included_item_ids": list(self.included_item_ids),
            "included_content_ids": list(self.included_content_ids),
            "status": self.status.value,
            "incomplete_reasons": list(self.incomplete_reasons),
            "metrics": self.metrics.to_dict(),
            "decision_digest": self.decision_digest,
            "invalidated_receipt_ids": list(self.invalidated_receipt_ids),
        }
        if include_receipt_id:
            result["receipt_id"] = self.receipt_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class CompiledProofContext:
    """One immutable selected context and its audit records."""

    obligation_id: str
    items: tuple[ProofContextItem, ...]
    decisions: tuple[InclusionDecision, ...]
    expansion_handles: tuple[ExpansionHandle, ...]
    status: ProofContextStatus
    incomplete_reasons: tuple[str, ...]
    dependency_fingerprint: str
    request_id: str
    metrics: ProofContextMetrics
    program_graph_slice: ProgramGraphSliceReference | None = None
    invalidated_receipt_ids: tuple[str, ...] = ()

    SCHEMA: ClassVar[str] = PROOF_CONTEXT_SCHEMA

    @property
    def context_id(self) -> str:
        return content_identity(self.to_dict(include_context_id=False))

    @property
    def included_item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)

    @property
    def complete(self) -> bool:
        return self.status is ProofContextStatus.COMPLETE

    @property
    def receipt(self) -> ProofContextReceipt:
        return ProofContextReceipt(
            context_id=self.context_id,
            request_id=self.request_id,
            obligation_id=self.obligation_id,
            dependency_fingerprint=self.dependency_fingerprint,
            included_item_ids=self.included_item_ids,
            included_content_ids=tuple(item.content_id for item in self.items),
            status=self.status,
            incomplete_reasons=self.incomplete_reasons,
            metrics=self.metrics,
            decision_digest=content_identity(
                [decision.to_dict() for decision in self.decisions]
            ),
            invalidated_receipt_ids=self.invalidated_receipt_ids,
        )

    def to_dict(self, *, include_context_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "version": PROOF_CONTEXT_VERSION,
            "evidence": MINIMAL_PROOF_CONTEXT_EVIDENCE,
            "obligation_id": self.obligation_id,
            "items": [item.to_dict() for item in self.items],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "expansion_handles": [
                handle.to_dict() for handle in self.expansion_handles
            ],
            "status": self.status.value,
            "complete": self.complete,
            "incomplete_reasons": list(self.incomplete_reasons),
            "dependency_fingerprint": self.dependency_fingerprint,
            "request_id": self.request_id,
            "metrics": self.metrics.to_dict(),
            "program_graph_slice": (
                self.program_graph_slice.to_dict()
                if self.program_graph_slice is not None
                else None
            ),
            "invalidated_receipt_ids": list(self.invalidated_receipt_ids),
            "required_inputs_truncated": False,
            "embeds_source_bodies": False,
            "embeds_full_graph": False,
        }
        if include_context_id:
            result["context_id"] = self.context_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


CodeContractProofContext = CompiledProofContext
ProofContext = CompiledProofContext


def _closure(
    roots: Iterable[str],
    item_by_id: Mapping[str, ProofContextItem],
) -> tuple[set[str], set[str]]:
    selected: set[str] = set()
    missing: set[str] = set()
    pending = list(sorted(set(roots), reverse=True))
    while pending:
        item_id = pending.pop()
        if item_id in selected or item_id in missing:
            continue
        item = item_by_id.get(item_id)
        if item is None:
            missing.add(item_id)
            continue
        selected.add(item_id)
        for dependency_id in reversed(item.dependency_ids):
            if dependency_id not in selected:
                pending.append(dependency_id)
    return selected, missing


def _handle_for(item: ProofContextItem, reason: str) -> ExpansionHandle:
    return ExpansionHandle(
        item_id=item.item_id,
        item_kind=item.kind.value,
        target_content_id=item.referenced_content_id or item.content_id,
        locator=item.expansion_locator,
        reason=reason,
    )


def _missing_handle(item_id: str) -> ExpansionHandle:
    return ExpansionHandle(
        item_id=item_id,
        item_kind="missing",
        target_content_id=content_identity(
            {"missing_item_id": item_id, "schema": PROOF_CONTEXT_ITEM_SCHEMA}
        ),
        reason="missing_required_dependency",
    )


def _transmitted_payload(
    obligation_id: str, items: Sequence[ProofContextItem]
) -> dict[str, Any]:
    return {
        "schema": PROOF_CONTEXT_SCHEMA,
        "version": PROOF_CONTEXT_VERSION,
        "obligation_id": obligation_id,
        "items": [item.to_dict() for item in items],
    }


def _incomplete_reasons(
    *,
    missing: Iterable[str],
    metrics: ProofContextMetrics,
    slice_reference: ProgramGraphSliceReference | None,
) -> tuple[str, ...]:
    reasons = {f"missing_required_item:{item_id}" for item_id in missing}
    if metrics.item_count > metrics.max_items:
        reasons.add("required_item_limit_exceeded")
    if metrics.byte_count > metrics.max_bytes:
        reasons.add("required_byte_limit_exceeded")
    if slice_reference is not None:
        if not slice_reference.dependency_complete:
            reasons.add("program_graph_slice_dependency_incomplete")
        if not slice_reference.complete:
            reasons.add("program_graph_slice_incomplete")
        if slice_reference.truncated:
            reasons.add("program_graph_slice_truncated")
        if slice_reference.missing_node_ids:
            reasons.add("program_graph_slice_missing_nodes")
        if slice_reference.omitted_dependencies:
            reasons.add("program_graph_slice_omitted_dependencies")
    return tuple(sorted(reasons))


def _compile(
    request: ProofContextRequest,
    *,
    invalidated_receipt_ids: Iterable[str] = (),
) -> CompiledProofContext:
    item_by_id = {item.item_id: item for item in request.items}
    obligation = item_by_id.get(request.obligation_id)
    roots = (request.obligation_id, *request.required_item_ids)
    selected_ids, missing = _closure(roots, item_by_id)
    if obligation is not None and obligation.kind is not ProofContextItemKind.OBLIGATION:
        raise CodeContractProofContextError(
            "obligation_id must identify an obligation item"
        )

    items = tuple(item_by_id[item_id] for item_id in sorted(selected_ids))
    reverse_required_by: dict[str, set[str]] = {
        item_id: set() for item_id in selected_ids | missing
    }
    for parent_id in selected_ids:
        for dependency_id in item_by_id[parent_id].dependency_ids:
            reverse_required_by.setdefault(dependency_id, set()).add(parent_id)

    handles_by_id: dict[str, ExpansionHandle] = {}
    decisions: list[InclusionDecision] = []
    explicit_roots = set(request.required_item_ids)
    for item in request.items:
        if item.item_id == request.obligation_id:
            reason = "selected_obligation"
        elif item.item_id in explicit_roots:
            reason = "explicitly_required"
        elif item.item_id in selected_ids:
            reason = "transitive_required_dependency"
        else:
            reason = "not_in_obligation_dependency_closure"
        handle = _handle_for(item, reason)
        handles_by_id[item.item_id] = handle
        decisions.append(
            InclusionDecision(
                item_id=item.item_id,
                decision=(
                    ProofContextDecision.INCLUDED
                    if item.item_id in selected_ids
                    else ProofContextDecision.EXCLUDED
                ),
                reason=reason,
                required_by=tuple(reverse_required_by.get(item.item_id, ())),
                item_content_id=item.content_id,
                expansion_handle_id=handle.handle_id,
            )
        )
    for item_id in sorted(missing):
        handle = _missing_handle(item_id)
        handles_by_id[item_id] = handle
        decisions.append(
            InclusionDecision(
                item_id=item_id,
                decision=ProofContextDecision.MISSING,
                reason="missing_required_dependency",
                required_by=tuple(reverse_required_by.get(item_id, ())),
                expansion_handle_id=handle.handle_id,
            )
        )

    transmitted = _transmitted_payload(request.obligation_id, items)
    metrics = ProofContextMetrics(
        item_count=len(items),
        byte_count=len(canonical_json_bytes(transmitted)),
        item_bytes=sum(item.byte_count for item in items),
        max_items=request.limits.max_items,
        max_bytes=request.limits.max_bytes,
    )
    reasons = _incomplete_reasons(
        missing=missing,
        metrics=metrics,
        slice_reference=request.program_graph_slice,
    )
    dependency_payload = {
        "obligation_id": request.obligation_id,
        "roots": sorted(set(roots)),
        "items": [
            {
                "item_id": item.item_id,
                "content_id": item.content_id,
                "dependency_ids": list(item.dependency_ids),
            }
            for item in items
        ],
        "missing": sorted(missing),
        "program_graph_slice": (
            request.program_graph_slice.to_dict()
            if request.program_graph_slice is not None
            else None
        ),
    }
    return CompiledProofContext(
        obligation_id=request.obligation_id,
        items=items,
        decisions=tuple(sorted(decisions, key=lambda item: item.item_id)),
        expansion_handles=tuple(
            handles_by_id[item_id] for item_id in sorted(handles_by_id)
        ),
        status=(
            ProofContextStatus.INCOMPLETE
            if reasons
            else ProofContextStatus.COMPLETE
        ),
        incomplete_reasons=reasons,
        dependency_fingerprint=content_identity(dependency_payload),
        request_id=request.request_id,
        metrics=metrics,
        program_graph_slice=request.program_graph_slice,
        invalidated_receipt_ids=tuple(invalidated_receipt_ids),
    )


@dataclass(frozen=True)
class ProofContextDelta:
    """A retry packet containing no items already present in its base."""

    base_receipt_id: str
    base_dependency_fingerprint: str
    items: tuple[ProofContextItem, ...]
    decisions: tuple[InclusionDecision, ...]
    expansion_handles: tuple[ExpansionHandle, ...]
    counterexample_item_ids: tuple[str, ...]
    requested_evidence_item_ids: tuple[str, ...]
    status: ProofContextStatus
    incomplete_reasons: tuple[str, ...]
    metrics: ProofContextMetrics

    SCHEMA: ClassVar[str] = PROOF_CONTEXT_DELTA_SCHEMA

    @property
    def transmitted_item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)

    @property
    def delta_id(self) -> str:
        return content_identity(self.to_dict(include_delta_id=False))

    @property
    def complete(self) -> bool:
        return self.status is ProofContextStatus.COMPLETE

    def to_dict(self, *, include_delta_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "version": PROOF_CONTEXT_VERSION,
            "evidence": MINIMAL_PROOF_CONTEXT_EVIDENCE,
            "base_receipt_id": self.base_receipt_id,
            "base_dependency_fingerprint": self.base_dependency_fingerprint,
            "items": [item.to_dict() for item in self.items],
            "transmitted_item_ids": list(self.transmitted_item_ids),
            "decisions": [item.to_dict() for item in self.decisions],
            "expansion_handles": [
                item.to_dict() for item in self.expansion_handles
            ],
            "counterexample_item_ids": list(self.counterexample_item_ids),
            "requested_evidence_item_ids": list(
                self.requested_evidence_item_ids
            ),
            "status": self.status.value,
            "complete": self.complete,
            "incomplete_reasons": list(self.incomplete_reasons),
            "metrics": self.metrics.to_dict(),
            "required_inputs_truncated": False,
        }
        if include_delta_id:
            result["delta_id"] = self.delta_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def _receipt_matches_compiled_base(
    receipt: ProofContextReceipt,
    current: CompiledProofContext,
) -> bool:
    """Validate fields which must survive adding retry-only candidates.

    A retry request may add excluded counterexample or evidence candidates, so
    its request, context, and decision identifiers legitimately differ from
    the request which produced ``receipt``.  The required closure itself must
    remain byte-for-byte equivalent.
    """

    expected = current.receipt
    return bool(
        receipt.obligation_id == expected.obligation_id
        and receipt.dependency_fingerprint
        == expected.dependency_fingerprint
        and receipt.included_item_ids == expected.included_item_ids
        and receipt.included_content_ids == expected.included_content_ids
        and receipt.status is expected.status
        and receipt.incomplete_reasons == expected.incomplete_reasons
        and receipt.metrics == expected.metrics
    )


def _invalid_delta(
    request: ProofContextRequest,
    base_receipt: ProofContextReceipt,
    *,
    counterexample_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    reason: str,
) -> ProofContextDelta:
    """Return a deterministic, empty retry after base invalidation."""

    metrics = ProofContextMetrics(
        item_count=0,
        byte_count=len(
            canonical_json_bytes(
                _transmitted_payload(request.obligation_id, ())
            )
        ),
        item_bytes=0,
        max_items=request.limits.max_items,
        max_bytes=request.limits.max_bytes,
    )
    return ProofContextDelta(
        base_receipt_id=base_receipt.receipt_id,
        base_dependency_fingerprint=base_receipt.dependency_fingerprint,
        items=(),
        decisions=(),
        expansion_handles=(),
        counterexample_item_ids=counterexample_ids,
        requested_evidence_item_ids=evidence_ids,
        status=ProofContextStatus.INVALIDATED,
        incomplete_reasons=(reason,),
        metrics=metrics,
    )


class CodeContractProofContextCompiler:
    """Deterministic compiler with exact-receipt reuse."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[str, ...]], CompiledProofContext] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def compile(
        self,
        request: ProofContextRequest,
        *,
        previous_receipt: ProofContextReceipt | None = None,
    ) -> CompiledProofContext:
        if not isinstance(request, ProofContextRequest):
            raise CodeContractProofContextError(
                "request must be ProofContextRequest"
            )
        invalidated: tuple[str, ...] = ()
        if previous_receipt is not None:
            if not isinstance(previous_receipt, ProofContextReceipt):
                raise CodeContractProofContextError(
                    "previous_receipt must be ProofContextReceipt"
                )
            if self._receipt_matches_request(previous_receipt, request):
                invalidated = previous_receipt.invalidated_receipt_ids
            else:
                invalidated = (previous_receipt.receipt_id,)
        key = (request.request_id, invalidated)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = _compile(request, invalidated_receipt_ids=invalidated)
        self._cache[key] = result
        return result

    @staticmethod
    def _receipt_matches_request(
        receipt: ProofContextReceipt,
        request: ProofContextRequest,
    ) -> bool:
        expected = _compile(
            request,
            invalidated_receipt_ids=receipt.invalidated_receipt_ids,
        ).receipt
        return receipt.receipt_id == expected.receipt_id

    def receipt_is_valid(
        self,
        receipt: ProofContextReceipt,
        request: ProofContextRequest,
    ) -> bool:
        if not isinstance(receipt, ProofContextReceipt):
            return False
        return self._receipt_matches_request(receipt, request)

    def compile_delta(
        self,
        request: ProofContextRequest,
        *,
        base_receipt: ProofContextReceipt,
        counterexample_item_ids: Iterable[str] = (),
        requested_evidence_item_ids: Iterable[str] = (),
    ) -> ProofContextDelta:
        if not isinstance(request, ProofContextRequest):
            raise CodeContractProofContextError(
                "request must be ProofContextRequest"
            )
        if not isinstance(base_receipt, ProofContextReceipt):
            raise CodeContractProofContextError(
                "base_receipt must be ProofContextReceipt"
            )
        counterexample_ids = _sorted_ids(
            counterexample_item_ids, "counterexample_item_id"
        )
        evidence_ids = _sorted_ids(
            requested_evidence_item_ids, "requested_evidence_item_id"
        )
        current = _compile(request)
        if (
            base_receipt.dependency_fingerprint
            != current.dependency_fingerprint
        ):
            return _invalid_delta(
                request,
                base_receipt,
                counterexample_ids=counterexample_ids,
                evidence_ids=evidence_ids,
                reason="base_dependencies_changed",
            )
        if not _receipt_matches_compiled_base(base_receipt, current):
            return _invalid_delta(
                request,
                base_receipt,
                counterexample_ids=counterexample_ids,
                evidence_ids=evidence_ids,
                reason="base_receipt_mismatch",
            )

        item_by_id = {item.item_id: item for item in request.items}
        for item_id in counterexample_ids:
            item = item_by_id.get(item_id)
            if item is not None and item.kind is not ProofContextItemKind.COUNTEREXAMPLE:
                raise CodeContractProofContextError(
                    f"counterexample item {item_id!r} has kind {item.kind.value!r}"
                )
        selected, missing = _closure(
            (*counterexample_ids, *evidence_ids), item_by_id
        )
        already_sent = set(base_receipt.included_item_ids)
        transmitted_ids = selected - already_sent
        items = tuple(item_by_id[item_id] for item_id in sorted(transmitted_ids))
        decisions: list[InclusionDecision] = []
        handles: list[ExpansionHandle] = []
        root_kinds: dict[str, str] = {
            item_id: "new_counterexample" for item_id in counterexample_ids
        }
        root_kinds.update(
            {item_id: "requested_evidence" for item_id in evidence_ids}
        )
        for item in items:
            reason = root_kinds.get(item.item_id, "new_required_dependency")
            handle = _handle_for(item, reason)
            handles.append(handle)
            decisions.append(
                InclusionDecision(
                    item_id=item.item_id,
                    decision=ProofContextDecision.INCLUDED,
                    reason=reason,
                    item_content_id=item.content_id,
                    expansion_handle_id=handle.handle_id,
                )
            )
        for item_id in sorted(missing):
            handle = _missing_handle(item_id)
            handles.append(handle)
            decisions.append(
                InclusionDecision(
                    item_id=item_id,
                    decision=ProofContextDecision.MISSING,
                    reason="missing_requested_delta_item",
                    expansion_handle_id=handle.handle_id,
                )
            )
        transmitted = _transmitted_payload(request.obligation_id, items)
        metrics = ProofContextMetrics(
            item_count=len(items),
            byte_count=len(canonical_json_bytes(transmitted)),
            item_bytes=sum(item.byte_count for item in items),
            max_items=request.limits.max_items,
            max_bytes=request.limits.max_bytes,
        )
        reasons = set(
            _incomplete_reasons(
                missing=missing,
                metrics=metrics,
                slice_reference=None,
            )
        )
        return ProofContextDelta(
            base_receipt_id=base_receipt.receipt_id,
            base_dependency_fingerprint=base_receipt.dependency_fingerprint,
            items=items,
            decisions=tuple(sorted(decisions, key=lambda item: item.item_id)),
            expansion_handles=tuple(
                sorted(handles, key=lambda item: item.item_id)
            ),
            counterexample_item_ids=counterexample_ids,
            requested_evidence_item_ids=evidence_ids,
            status=(
                ProofContextStatus.INCOMPLETE
                if reasons
                else ProofContextStatus.COMPLETE
            ),
            incomplete_reasons=tuple(sorted(reasons)),
            metrics=metrics,
        )


ProofContextCompiler = CodeContractProofContextCompiler

_DEFAULT_COMPILER = CodeContractProofContextCompiler()


def compile_code_contract_proof_context(
    request: ProofContextRequest,
    *,
    previous_receipt: ProofContextReceipt | None = None,
    compiler: CodeContractProofContextCompiler | None = None,
) -> CompiledProofContext:
    """Compile and cache the smallest closed context for ``request``."""

    return (compiler or _DEFAULT_COMPILER).compile(
        request, previous_receipt=previous_receipt
    )


compile_proof_context = compile_code_contract_proof_context


def compile_proof_context_delta(
    request: ProofContextRequest,
    *,
    base_receipt: ProofContextReceipt,
    counterexample_item_ids: Iterable[str] = (),
    requested_evidence_item_ids: Iterable[str] = (),
    compiler: CodeContractProofContextCompiler | None = None,
) -> ProofContextDelta:
    """Compile a retry which never retransmits base-context items."""

    return (compiler or _DEFAULT_COMPILER).compile_delta(
        request,
        base_receipt=base_receipt,
        counterexample_item_ids=counterexample_item_ids,
        requested_evidence_item_ids=requested_evidence_item_ids,
    )


__all__ = [
    "CodeContractProofContext",
    "CodeContractProofContextCompiler",
    "CodeContractProofContextError",
    "CodeContractProofContextRequest",
    "CompiledProofContext",
    "DEFAULT_MAX_CONTEXT_BYTES",
    "DEFAULT_MAX_CONTEXT_ITEMS",
    "ExpansionHandle",
    "InclusionDecision",
    "MINIMAL_PROOF_CONTEXT_EVIDENCE",
    "PROOF_CONTEXT_VERSION",
    "ProgramGraphSliceReference",
    "ProofContext",
    "ProofContextArtifact",
    "ProofContextCompiler",
    "ProofContextDecision",
    "ProofContextDelta",
    "ProofContextExpansionHandle",
    "ProofContextInclusionDecision",
    "ProofContextItem",
    "ProofContextItemKind",
    "ProofContextKind",
    "ProofContextLimits",
    "ProofContextMetrics",
    "ProofContextReceipt",
    "ProofContextRequest",
    "ProofContextStatus",
    "ProofContextValidationError",
    "compile_code_contract_proof_context",
    "compile_proof_context",
    "compile_proof_context_delta",
]
