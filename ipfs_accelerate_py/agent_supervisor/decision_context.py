"""Immutable, complete model context for one proof-directed decision.

This module contains the generation-3 decision-context contracts.  A decision
context is deliberately narrower than a retrieval receipt: only the exact
decision envelope and its authoritative mandatory dependency closure are
model-facing.  Approximate retrieval candidates and unrelated corpus entries
may affect bounded index metadata, but cannot enter the required core.

Every mandatory closure node has exactly one :class:`DecisionContextReference`
and one :class:`ContextCompletenessEntry`.  Small canonical node bodies are
inlined; larger bodies are represented by a bounded summary and an expansion
handle whose content identity and repository/root bindings were verified by
the compiler.  Completeness is therefore checkable without trusting prompt
prose or value-of-information selection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar

from .context_contracts import ContextReference, ContextTier
from .formal_verification_contracts import CanonicalContract, canonical_json_bytes


DECISION_CONTEXT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/decision-context@1"
)
DECISION_CONTEXT_REFERENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/decision-context-reference@1"
)
CONTEXT_COMPLETENESS_ENTRY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/context-completeness-entry@1"
)
CONTEXT_COMPLETENESS_WITNESS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/context-completeness-witness@1"
)
DECISION_CONTEXT_COMPILATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/decision-context-compilation@1"
)
DECISION_CONTEXT_CONTRACT_VERSION = 1
MAX_INLINE_SUMMARY_BYTES = 4_096
MAX_WITNESS_ENTRIES = 16_384

REQUIRED_CORE_FIELDS = (
    "decision",
    "roots",
    "intent_action_contract",
    "legal_constraints",
    "legal_unknowns",
    "security_constraints",
    "security_unknowns",
    "authorization_state",
    "program_scope",
    "effect_scope",
    "assumptions",
    "obligations",
    "proof_state",
    "monitor_state",
    "validation",
    "acceptance",
    "failure_behavior",
)


class DecisionContextError(ValueError):
    """A decision context or completeness claim is malformed."""


class DecisionContextBindingError(DecisionContextError):
    """A context crossed a decision, graph, receipt, root, or tree boundary."""


class DecisionContextOverflowError(DecisionContextError):
    """Complete mandatory context cannot be represented within its hard budget."""


class MissingDecisionContextExpansionError(DecisionContextError):
    """A required expansion body is absent or cannot be content-verified."""


class DecisionContextRepresentation(str, Enum):
    INLINE = "inline"
    EXPANSION = "expansion"


class DecisionContextOverflowBehavior(str, Enum):
    SPLIT = "split"
    REQUEST_EXPANSION = "request_expansion"
    FAIL_CLOSED = "fail_closed"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise DecisionContextError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise DecisionContextError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise DecisionContextError(f"{name} is required")
    if len(value.encode("utf-8")) > 65_536:
        raise DecisionContextError(f"{name} is oversized")
    return value


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise DecisionContextError(f"{name} must be a {qualifier} integer")
    return value


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise DecisionContextError("decision context exceeds its nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise DecisionContextError(
            "floating-point values are not canonical decision context"
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise DecisionContextError("decision context keys must be strings")
        return {
            key: _plain(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise DecisionContextError(
        f"unsupported decision-context value: {type(value).__name__}"
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionContextError(f"{name} must be an object")
    normalized = _plain(value)
    if not isinstance(normalized, dict):
        raise DecisionContextError(f"{name} must normalize to an object")

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType(
                {key: freeze(member) for key, member in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(member) for member in item)
        return item

    return freeze(normalized)


def _strings(
    value: Iterable[Any],
    name: str,
    *,
    unique: bool = True,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise DecisionContextError(f"{name} must be a sequence")
    normalized = tuple(_text(item, name) for item in value)
    if unique and len(normalized) != len(set(normalized)):
        raise DecisionContextError(f"{name} must not contain duplicates")
    return normalized


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json_bytes(_plain(value))).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _contract_payload(
    payload: Mapping[str, Any],
    *,
    schema: str,
    allowed: Iterable[str],
    noun: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise DecisionContextError(f"{noun} payload must be an object")
    if payload.get("schema") not in (None, schema):
        raise DecisionContextError(f"unsupported {noun} schema")
    if payload.get("contract_version") not in (
        None,
        DECISION_CONTEXT_CONTRACT_VERSION,
    ):
        raise DecisionContextError(f"unsupported {noun} contract version")
    if set(payload).difference(allowed):
        raise DecisionContextError(
            f"{noun} contains unsupported fields; rebuild its canonical payload"
        )


def _json_object(payload: str, noun: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DecisionContextError(f"{noun} JSON is malformed") from exc
    if not isinstance(value, Mapping):
        raise DecisionContextError(f"{noun} JSON must contain an object")
    return value


@dataclass(frozen=True)
class DecisionContextReference(CanonicalContract):
    """One required dependency represented inline or by a verified handle."""

    SCHEMA: ClassVar[str] = DECISION_CONTEXT_REFERENCE_SCHEMA

    reference_id: str
    node_id: str
    node_kind: str
    node_content_id: str
    representation: DecisionContextRepresentation
    summary: Mapping[str, Any]
    body: Mapping[str, Any] = field(default_factory=dict)
    expansion_handle: ContextReference | None = None

    def __post_init__(self) -> None:
        for name in ("reference_id", "node_id", "node_kind", "node_content_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        try:
            representation = (
                self.representation
                if isinstance(self.representation, DecisionContextRepresentation)
                else DecisionContextRepresentation(str(self.representation))
            )
        except ValueError as exc:
            raise DecisionContextError(
                "unsupported decision-context representation"
            ) from exc
        object.__setattr__(self, "representation", representation)
        summary = _mapping(self.summary, "reference summary")
        body = _mapping(self.body, "reference body")
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "body", body)
        if len(canonical_json_bytes(summary)) > MAX_INLINE_SUMMARY_BYTES:
            raise DecisionContextError("reference summary exceeds its byte bound")
        handle = self.expansion_handle
        if handle is not None and not isinstance(handle, ContextReference):
            if not isinstance(handle, Mapping):
                raise DecisionContextError("expansion handle is malformed")
            handle = ContextReference.from_dict(handle)
            object.__setattr__(self, "expansion_handle", handle)
        if representation is DecisionContextRepresentation.INLINE:
            if not body or handle is not None:
                raise DecisionContextError(
                    "inline references require a body and prohibit a handle"
                )
            # SemanticNode's ID is over a payload without content_id and
            # authoritative.  A compiler-provided body therefore also carries
            # its authoritative node ID explicitly; validate the direct claim
            # rather than manufacture a different semantic schema hash here.
            claimed = str(body.get("content_id") or "")
            if not claimed or claimed != self.node_content_id:
                raise DecisionContextBindingError(
                    "inline body does not match its node content identity"
                )
            from .semantic_dependency_graph import SemanticNode

            try:
                restored = SemanticNode.from_dict(body)
            except ValueError as exc:
                raise DecisionContextBindingError(
                    "inline body is not a canonical semantic node"
                ) from exc
            if (
                restored.node_id != self.node_id
                or restored.kind.value != self.node_kind
                or restored.content_id != self.node_content_id
            ):
                raise DecisionContextBindingError(
                    "inline body does not bind its declared semantic node"
                )
        else:
            if body or handle is None:
                raise DecisionContextError(
                    "expansion references require a handle and prohibit a body"
                )
            if handle.tier is not ContextTier.EXPANSION:
                raise DecisionContextError(
                    "decision-context expansion handle has the wrong tier"
                )
            if (
                handle.reference_id != self.reference_id
                or handle.metadata.get("mandatory_node_id") != self.node_id
                or handle.metadata.get("node_content_id")
                != self.node_content_id
            ):
                raise DecisionContextBindingError(
                    "expansion handle does not bind its mandatory node"
                )

    @property
    def resolvable_content_id(self) -> str:
        if self.expansion_handle is None:
            return self.node_content_id
        return self.expansion_handle.referenced_content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTEXT_CONTRACT_VERSION,
            "reference_id": self.reference_id,
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "node_content_id": self.node_content_id,
            "representation": self.representation,
            "summary": self.summary,
            "body": self.body,
            "expansion_handle": (
                self.expansion_handle.to_dict()
                if self.expansion_handle is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DecisionContextReference":
        _contract_payload(
            payload,
            schema=cls.SCHEMA,
            allowed={
                "schema",
                "contract_version",
                "content_id",
                "reference_id",
                "node_id",
                "node_kind",
                "node_content_id",
                "representation",
                "summary",
                "body",
                "expansion_handle",
            },
            noun="decision-context reference",
        )
        result = cls(
            reference_id=payload.get("reference_id", ""),
            node_id=payload.get("node_id", ""),
            node_kind=payload.get("node_kind", ""),
            node_content_id=payload.get("node_content_id", ""),
            representation=payload.get("representation", ""),
            summary=payload.get("summary") or {},
            body=payload.get("body") or {},
            expansion_handle=payload.get("expansion_handle"),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", result.content_id):
            raise DecisionContextBindingError(
                "decision-context reference identity mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str) -> "DecisionContextReference":
        return cls.from_dict(_json_object(payload, "decision-context reference"))


@dataclass(frozen=True)
class ContextCompletenessEntry(CanonicalContract):
    """Witness mapping from one closure path to its concrete representation."""

    SCHEMA: ClassVar[str] = CONTEXT_COMPLETENESS_ENTRY_SCHEMA

    node_id: str
    node_kind: str
    node_content_id: str
    path: tuple[str, ...]
    path_edge_ids: tuple[str, ...]
    reference_id: str
    reference_content_id: str
    representation: DecisionContextRepresentation

    def __post_init__(self) -> None:
        for name in (
            "node_id",
            "node_kind",
            "node_content_id",
            "reference_id",
            "reference_content_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        path = _strings(self.path, "dependency path")
        edges = _strings(self.path_edge_ids, "dependency path edges")
        if not path or path[-1] != self.node_id or len(edges) != len(path) - 1:
            raise DecisionContextError(
                "completeness entry has an invalid dependency path"
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "path_edge_ids", edges)
        try:
            representation = (
                self.representation
                if isinstance(self.representation, DecisionContextRepresentation)
                else DecisionContextRepresentation(str(self.representation))
            )
        except ValueError as exc:
            raise DecisionContextError(
                "completeness entry representation is invalid"
            ) from exc
        object.__setattr__(self, "representation", representation)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTEXT_CONTRACT_VERSION,
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "node_content_id": self.node_content_id,
            "path": self.path,
            "path_edge_ids": self.path_edge_ids,
            "reference_id": self.reference_id,
            "reference_content_id": self.reference_content_id,
            "representation": self.representation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextCompletenessEntry":
        _contract_payload(
            payload,
            schema=cls.SCHEMA,
            allowed={
                "schema",
                "contract_version",
                "content_id",
                "node_id",
                "node_kind",
                "node_content_id",
                "path",
                "path_edge_ids",
                "reference_id",
                "reference_content_id",
                "representation",
            },
            noun="completeness entry",
        )
        result = cls(
            node_id=payload.get("node_id", ""),
            node_kind=payload.get("node_kind", ""),
            node_content_id=payload.get("node_content_id", ""),
            path=tuple(payload.get("path") or ()),
            path_edge_ids=tuple(payload.get("path_edge_ids") or ()),
            reference_id=payload.get("reference_id", ""),
            reference_content_id=payload.get("reference_content_id", ""),
            representation=payload.get("representation", ""),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", result.content_id):
            raise DecisionContextBindingError(
                "completeness-entry identity mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str) -> "ContextCompletenessEntry":
        return cls.from_dict(_json_object(payload, "completeness entry"))


@dataclass(frozen=True)
class ContextCompletenessWitness(CanonicalContract):
    """Proof that every mandatory dependency and path has a representation."""

    SCHEMA: ClassVar[str] = CONTEXT_COMPLETENESS_WITNESS_SCHEMA

    decision_request_id: str
    semantic_graph_root_id: str
    semantic_graph_id: str
    retrieval_receipt_id: str
    closure_id: str
    mandatory_node_ids: tuple[str, ...]
    mandatory_edge_ids: tuple[str, ...]
    entries: tuple[ContextCompletenessEntry, ...]
    required_core_fields: tuple[str, ...] = REQUIRED_CORE_FIELDS
    inline_reference_ids: tuple[str, ...] = ()
    expansion_reference_ids: tuple[str, ...] = ()
    roots_digest: str = ""
    complete: bool = True
    truncated: bool = False

    def __post_init__(self) -> None:
        for name in (
            "decision_request_id",
            "semantic_graph_root_id",
            "semantic_graph_id",
            "retrieval_receipt_id",
            "closure_id",
            "roots_digest",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        nodes = tuple(sorted(_strings(self.mandatory_node_ids, "mandatory nodes")))
        edges = tuple(sorted(_strings(self.mandatory_edge_ids, "mandatory edges")))
        object.__setattr__(self, "mandatory_node_ids", nodes)
        object.__setattr__(self, "mandatory_edge_ids", edges)
        normalized_entries = tuple(
            item
            if isinstance(item, ContextCompletenessEntry)
            else ContextCompletenessEntry.from_dict(item)
            for item in self.entries
        )
        if len(normalized_entries) > MAX_WITNESS_ENTRIES:
            raise DecisionContextOverflowError(
                "completeness witness exceeds its entry bound"
            )
        normalized_entries = tuple(
            sorted(normalized_entries, key=lambda item: item.node_id)
        )
        object.__setattr__(self, "entries", normalized_entries)
        entry_nodes = tuple(item.node_id for item in normalized_entries)
        if entry_nodes != nodes:
            raise DecisionContextError(
                "completeness witness must map every mandatory node exactly once"
            )
        represented_edges = {
            edge_id
            for item in normalized_entries
            for edge_id in item.path_edge_ids
        }
        if not represented_edges.issubset(set(edges)):
            raise DecisionContextError(
                "completeness paths contain an edge outside mandatory closure"
            )
        path_roots = {item.path[0] for item in normalized_entries}
        if len(path_roots) != 1:
            raise DecisionContextError(
                "completeness paths must share one decision root"
            )
        fields = _strings(self.required_core_fields, "required core fields")
        if fields != REQUIRED_CORE_FIELDS:
            raise DecisionContextError(
                "completeness witness omits a required core field"
            )
        object.__setattr__(self, "required_core_fields", fields)
        inline = tuple(
            sorted(_strings(self.inline_reference_ids, "inline references"))
        )
        expansions = tuple(
            sorted(
                _strings(
                    self.expansion_reference_ids, "expansion references"
                )
            )
        )
        if set(inline).intersection(expansions):
            raise DecisionContextError(
                "a completeness reference cannot be inline and expandable"
            )
        entry_inline = {
            item.reference_id
            for item in normalized_entries
            if item.representation is DecisionContextRepresentation.INLINE
        }
        entry_expansions = {
            item.reference_id
            for item in normalized_entries
            if item.representation is DecisionContextRepresentation.EXPANSION
        }
        if set(inline) != entry_inline or set(expansions) != entry_expansions:
            raise DecisionContextError(
                "witness representation indexes do not match its entries"
            )
        object.__setattr__(self, "inline_reference_ids", inline)
        object.__setattr__(self, "expansion_reference_ids", expansions)
        if not self.complete or self.truncated:
            raise DecisionContextError(
                "a completeness witness cannot be partial or truncated"
            )

    @property
    def dependency_paths(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {item.node_id: item.path for item in self.entries}
        )

    @property
    def dependency_references(self) -> Mapping[str, str]:
        return MappingProxyType(
            {item.node_id: item.reference_id for item in self.entries}
        )

    @property
    def dependency_map(self) -> Mapping[str, Mapping[str, Any]]:
        """Canonical node-to-path/reference projection for audit consumers."""

        return MappingProxyType(
            {
                item.node_id: MappingProxyType(
                    {
                        "path": item.path,
                        "path_edge_ids": item.path_edge_ids,
                        "reference_id": item.reference_id,
                        "reference_content_id": item.reference_content_id,
                        "representation": item.representation.value,
                    }
                )
                for item in self.entries
            }
        )

    def entry(self, node_id: str) -> ContextCompletenessEntry:
        for item in self.entries:
            if item.node_id == node_id:
                return item
        raise KeyError(node_id)

    @property
    def content_id(self) -> str:
        """Return the dependency-completeness identity.

        Source graph and retrieval receipt identities are audit provenance, not
        semantic dependencies.  Excluding those two index identities preserves
        exact reuse when unrelated corpus/graph growth produces a new index
        receipt but the mandatory closure, roots, paths, and representations
        are byte-for-byte unchanged.
        """

        return _identity(
            "context-completeness-witness",
            {
                "schema": self.SCHEMA,
                "contract_version": DECISION_CONTEXT_CONTRACT_VERSION,
                "decision_request_id": self.decision_request_id,
                "semantic_graph_root_id": self.semantic_graph_root_id,
                "closure_id": self.closure_id,
                "mandatory_node_ids": self.mandatory_node_ids,
                "mandatory_edge_ids": self.mandatory_edge_ids,
                "entries": tuple(item.to_record() for item in self.entries),
                "required_core_fields": self.required_core_fields,
                "inline_reference_ids": self.inline_reference_ids,
                "expansion_reference_ids": self.expansion_reference_ids,
                "roots_digest": self.roots_digest,
                "complete": self.complete,
                "truncated": self.truncated,
            },
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTEXT_CONTRACT_VERSION,
            "decision_request_id": self.decision_request_id,
            "semantic_graph_root_id": self.semantic_graph_root_id,
            "semantic_graph_id": self.semantic_graph_id,
            "retrieval_receipt_id": self.retrieval_receipt_id,
            "closure_id": self.closure_id,
            "mandatory_node_ids": self.mandatory_node_ids,
            "mandatory_edge_ids": self.mandatory_edge_ids,
            "entries": tuple(item.to_record() for item in self.entries),
            "required_core_fields": self.required_core_fields,
            "inline_reference_ids": self.inline_reference_ids,
            "expansion_reference_ids": self.expansion_reference_ids,
            "roots_digest": self.roots_digest,
            "complete": self.complete,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ContextCompletenessWitness":
        _contract_payload(
            payload,
            schema=cls.SCHEMA,
            allowed={
                "schema",
                "contract_version",
                "content_id",
                "decision_request_id",
                "semantic_graph_root_id",
                "semantic_graph_id",
                "retrieval_receipt_id",
                "closure_id",
                "mandatory_node_ids",
                "mandatory_edge_ids",
                "entries",
                "required_core_fields",
                "inline_reference_ids",
                "expansion_reference_ids",
                "roots_digest",
                "complete",
                "truncated",
            },
            noun="context-completeness witness",
        )
        result = cls(
            decision_request_id=payload.get("decision_request_id", ""),
            semantic_graph_root_id=payload.get("semantic_graph_root_id", ""),
            semantic_graph_id=payload.get("semantic_graph_id", ""),
            retrieval_receipt_id=payload.get("retrieval_receipt_id", ""),
            closure_id=payload.get("closure_id", ""),
            mandatory_node_ids=tuple(payload.get("mandatory_node_ids") or ()),
            mandatory_edge_ids=tuple(payload.get("mandatory_edge_ids") or ()),
            entries=tuple(
                ContextCompletenessEntry.from_dict(item)
                for item in payload.get("entries") or ()
            ),
            required_core_fields=tuple(
                payload.get("required_core_fields") or ()
            ),
            inline_reference_ids=tuple(
                payload.get("inline_reference_ids") or ()
            ),
            expansion_reference_ids=tuple(
                payload.get("expansion_reference_ids") or ()
            ),
            roots_digest=payload.get("roots_digest", ""),
            complete=payload.get("complete", False),
            truncated=payload.get("truncated", True),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", result.content_id):
            raise DecisionContextBindingError(
                "context-completeness witness identity mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str) -> "ContextCompletenessWitness":
        return cls.from_dict(
            _json_object(payload, "context-completeness witness")
        )


@dataclass(frozen=True)
class DecisionContext(CanonicalContract):
    """One provider-sized segment of a complete immutable decision context."""

    SCHEMA: ClassVar[str] = DECISION_CONTEXT_SCHEMA

    required_core: Mapping[str, Any]
    references: tuple[DecisionContextReference, ...]
    completeness_witness_id: str
    witness_entries: tuple[ContextCompletenessEntry, ...]
    index_metadata: Mapping[str, Any]
    provider_input_tokens: int
    effective_input_limit: int
    segment_index: int = 0
    segment_count: int = 1
    expansion_request: str = ""

    def __post_init__(self) -> None:
        core = _mapping(self.required_core, "required core")
        missing = set(REQUIRED_CORE_FIELDS).difference(core)
        extra = set(core).difference(REQUIRED_CORE_FIELDS)
        if missing or extra:
            raise DecisionContextError(
                "decision context core must contain exactly the required fields"
            )
        object.__setattr__(self, "required_core", core)
        references = tuple(
            item
            if isinstance(item, DecisionContextReference)
            else DecisionContextReference.from_dict(item)
            for item in self.references
        )
        references = tuple(sorted(references, key=lambda item: item.node_id))
        if len({item.node_id for item in references}) != len(references):
            raise DecisionContextError(
                "decision context contains duplicate mandatory nodes"
            )
        object.__setattr__(self, "references", references)
        entries = tuple(
            item
            if isinstance(item, ContextCompletenessEntry)
            else ContextCompletenessEntry.from_dict(item)
            for item in self.witness_entries
        )
        entries = tuple(sorted(entries, key=lambda item: item.node_id))
        if {item.node_id for item in entries} != {
            item.node_id for item in references
        }:
            raise DecisionContextError(
                "segment witness entries must cover its references exactly"
            )
        object.__setattr__(self, "witness_entries", entries)
        object.__setattr__(
            self,
            "completeness_witness_id",
            _text(self.completeness_witness_id, "completeness_witness_id"),
        )
        object.__setattr__(
            self,
            "index_metadata",
            _mapping(self.index_metadata, "index metadata"),
        )
        tokens = _positive(
            self.provider_input_tokens,
            "provider_input_tokens",
            allow_zero=True,
        )
        limit = _positive(self.effective_input_limit, "effective_input_limit")
        if tokens > limit:
            raise DecisionContextOverflowError(
                "decision-context segment exceeds the provider input limit"
            )
        if (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or self.segment_index < 0
        ):
            raise DecisionContextError(
                "segment_index must be a non-negative integer"
            )
        count = _positive(self.segment_count, "segment_count")
        if self.segment_index >= count:
            raise DecisionContextError("segment_index exceeds segment_count")
        object.__setattr__(
            self,
            "expansion_request",
            _text(
                self.expansion_request,
                "expansion_request",
                required=False,
            ),
        )

    @property
    def decision_request_id(self) -> str:
        return str(self.required_core["decision"]["content_id"])

    @property
    def split(self) -> bool:
        return self.segment_count > 1

    @property
    def core(self) -> Mapping[str, Any]:
        return self.required_core

    def __getattr__(self, name: str) -> Any:
        # Expose the named required-core domains as read-only convenience
        # attributes without duplicating them in the canonical record.
        if name in REQUIRED_CORE_FIELDS:
            return self.required_core[name]
        raise AttributeError(name)

    def provider_payload(self) -> dict[str, Any]:
        """Return exactly the payload measured by the provider tokenizer."""

        return {
            "schema": self.SCHEMA,
            "contract_version": DECISION_CONTEXT_CONTRACT_VERSION,
            "required_core": _plain(self.required_core),
            "references": [item.to_record() for item in self.references],
            "completeness_witness_id": self.completeness_witness_id,
            "witness_entries": [
                item.to_record() for item in self.witness_entries
            ],
            "index_metadata": _plain(self.index_metadata),
            "segment": {
                "index": self.segment_index,
                "count": self.segment_count,
                "expansion_request": self.expansion_request,
            },
        }

    def _payload(self) -> dict[str, Any]:
        return {
            **self.provider_payload(),
            "provider_input_tokens": self.provider_input_tokens,
            "effective_input_limit": self.effective_input_limit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionContext":
        _contract_payload(
            payload,
            schema=cls.SCHEMA,
            allowed={
                "schema",
                "contract_version",
                "content_id",
                "required_core",
                "references",
                "completeness_witness_id",
                "witness_entries",
                "index_metadata",
                "provider_input_tokens",
                "effective_input_limit",
                "segment",
                "segment_index",
                "segment_count",
                "expansion_request",
            },
            noun="decision context",
        )
        segment = payload.get("segment") or {}
        if not isinstance(segment, Mapping):
            raise DecisionContextError("decision-context segment is malformed")
        result = cls(
            required_core=payload.get("required_core") or {},
            references=tuple(
                DecisionContextReference.from_dict(item)
                for item in payload.get("references") or ()
            ),
            completeness_witness_id=payload.get(
                "completeness_witness_id", ""
            ),
            witness_entries=tuple(
                ContextCompletenessEntry.from_dict(item)
                for item in payload.get("witness_entries") or ()
            ),
            index_metadata=payload.get("index_metadata") or {},
            provider_input_tokens=payload.get("provider_input_tokens", 0),
            effective_input_limit=payload.get("effective_input_limit", 0),
            segment_index=segment.get(
                "index", payload.get("segment_index", 0)
            ),
            segment_count=segment.get(
                "count", payload.get("segment_count", 1)
            ),
            expansion_request=segment.get(
                "expansion_request",
                payload.get("expansion_request", ""),
            ),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", result.content_id):
            raise DecisionContextBindingError(
                "decision-context identity mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str) -> "DecisionContext":
        return cls.from_dict(_json_object(payload, "decision context"))


@dataclass(frozen=True)
class DecisionContextCompilation(CanonicalContract):
    """Complete compiler result, including all deterministic split segments."""

    SCHEMA: ClassVar[str] = DECISION_CONTEXT_COMPILATION_SCHEMA

    contexts: tuple[DecisionContext, ...]
    witness: ContextCompletenessWitness
    complete_input_tokens: int
    provider_tokenizer: str
    overflow_behavior: DecisionContextOverflowBehavior
    required_nodes_participated_in_value_selection: bool = False
    verifier: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        contexts = tuple(self.contexts)
        if not contexts or not all(
            isinstance(item, DecisionContext) for item in contexts
        ):
            raise DecisionContextError(
                "compilation requires decision-context segments"
            )
        object.__setattr__(self, "contexts", contexts)
        if not isinstance(self.witness, ContextCompletenessWitness):
            raise DecisionContextError(
                "compilation requires a ContextCompletenessWitness"
            )
        for index, context in enumerate(contexts):
            if (
                context.segment_index != index
                or context.segment_count != len(contexts)
                or context.completeness_witness_id != self.witness.content_id
            ):
                raise DecisionContextBindingError(
                    "decision-context segments do not bind their complete witness"
                )
        covered = {
            reference.node_id
            for context in contexts
            for reference in context.references
        }
        if covered != set(self.witness.mandatory_node_ids):
            raise DecisionContextError(
                "compiled segments do not cover every mandatory dependency"
            )
        duplicate_count = sum(
            len(context.references) for context in contexts
        )
        if duplicate_count != len(covered):
            raise DecisionContextError(
                "a mandatory dependency appears in more than one split"
            )
        reference_by_node = {
            reference.node_id: reference
            for context in contexts
            for reference in context.references
        }
        for entry in self.witness.entries:
            reference = reference_by_node[entry.node_id]
            if (
                entry.node_kind != reference.node_kind
                or entry.node_content_id != reference.node_content_id
                or entry.reference_id != reference.reference_id
                or entry.reference_content_id
                != reference.resolvable_content_id
                or entry.representation is not reference.representation
            ):
                raise DecisionContextBindingError(
                    "completeness witness does not bind its dependency "
                    "representation"
                )
        total = _positive(
            self.complete_input_tokens,
            "complete_input_tokens",
            allow_zero=True,
        )
        if total != sum(item.provider_input_tokens for item in contexts):
            raise DecisionContextError(
                "complete input token accounting is not reproducible"
            )
        object.__setattr__(
            self,
            "provider_tokenizer",
            _text(self.provider_tokenizer, "provider_tokenizer"),
        )
        try:
            behavior = (
                self.overflow_behavior
                if isinstance(
                    self.overflow_behavior, DecisionContextOverflowBehavior
                )
                else DecisionContextOverflowBehavior(str(self.overflow_behavior))
            )
        except ValueError as exc:
            raise DecisionContextError("unsupported overflow behavior") from exc
        object.__setattr__(self, "overflow_behavior", behavior)
        if self.required_nodes_participated_in_value_selection:
            raise DecisionContextError(
                "required nodes must never participate in value selection"
            )

    @property
    def context(self) -> DecisionContext:
        return self.contexts[0]

    @property
    def split(self) -> bool:
        return len(self.contexts) > 1

    @property
    def expansion_requests(self) -> tuple[str, ...]:
        return tuple(
            item.expansion_request
            for item in self.contexts
            if item.expansion_request
        )

    @property
    def required_core(self) -> Mapping[str, Any]:
        return self.context.required_core

    @property
    def input_tokens(self) -> int:
        return self.complete_input_tokens

    @property
    def index_metadata(self) -> Mapping[str, Any]:
        return self.context.index_metadata

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DECISION_CONTEXT_CONTRACT_VERSION,
            "contexts": tuple(item.to_record() for item in self.contexts),
            "witness": self.witness.to_record(),
            "complete_input_tokens": self.complete_input_tokens,
            "provider_tokenizer": self.provider_tokenizer,
            "overflow_behavior": self.overflow_behavior,
            "required_nodes_participated_in_value_selection": (
                self.required_nodes_participated_in_value_selection
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DecisionContextCompilation":
        _contract_payload(
            payload,
            schema=cls.SCHEMA,
            allowed={
                "schema",
                "contract_version",
                "content_id",
                "contexts",
                "witness",
                "complete_input_tokens",
                "provider_tokenizer",
                "overflow_behavior",
                "required_nodes_participated_in_value_selection",
            },
            noun="decision-context compilation",
        )
        result = cls(
            contexts=tuple(
                DecisionContext.from_dict(item)
                for item in payload.get("contexts") or ()
            ),
            witness=ContextCompletenessWitness.from_dict(
                payload.get("witness") or {}
            ),
            complete_input_tokens=payload.get("complete_input_tokens", 0),
            provider_tokenizer=payload.get("provider_tokenizer", ""),
            overflow_behavior=payload.get("overflow_behavior", ""),
            required_nodes_participated_in_value_selection=payload.get(
                "required_nodes_participated_in_value_selection", False
            ),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", result.content_id):
            raise DecisionContextBindingError(
                "decision-context compilation identity mismatch"
            )
        return result

    @classmethod
    def from_json(cls, payload: str) -> "DecisionContextCompilation":
        return cls.from_dict(
            _json_object(payload, "decision-context compilation")
        )


def render_decision_context(context: DecisionContext) -> str:
    """Render exactly the canonical provider input measured by the compiler."""

    if not isinstance(context, DecisionContext):
        raise DecisionContextError("context must be a DecisionContext")
    return canonical_json_bytes(context.provider_payload()).decode("utf-8")


def compile_decision_context(*args: Any, **kwargs: Any) -> Any:
    """Lazy convenience export of the decision-context compiler wrapper."""

    from .context_compiler import compile_decision_context as compile_context

    return compile_context(*args, **kwargs)


def __getattr__(name: str) -> Any:
    # Avoid a module-import cycle while supporting the natural
    # ``decision_context.DecisionContextCompiler`` spelling.
    if name == "DecisionContextCompiler":
        from .context_compiler import DecisionContextCompiler

        return DecisionContextCompiler
    raise AttributeError(name)


# Compatibility aliases use the domain names likely used by callers while
# retaining one canonical wire contract.
CompiledDecisionContext = DecisionContext
DecisionContextResult = DecisionContextCompilation
CompletenessWitness = ContextCompletenessWitness
MandatoryDependencyWitnessEntry = ContextCompletenessEntry


__all__ = [
    "CONTEXT_COMPLETENESS_ENTRY_SCHEMA",
    "CONTEXT_COMPLETENESS_WITNESS_SCHEMA",
    "DECISION_CONTEXT_COMPILATION_SCHEMA",
    "DECISION_CONTEXT_CONTRACT_VERSION",
    "DECISION_CONTEXT_REFERENCE_SCHEMA",
    "DECISION_CONTEXT_SCHEMA",
    "MAX_INLINE_SUMMARY_BYTES",
    "REQUIRED_CORE_FIELDS",
    "CompiledDecisionContext",
    "CompletenessWitness",
    "ContextCompletenessEntry",
    "ContextCompletenessWitness",
    "DecisionContext",
    "DecisionContextBindingError",
    "DecisionContextCompilation",
    "DecisionContextCompiler",
    "DecisionContextError",
    "DecisionContextOverflowBehavior",
    "DecisionContextOverflowError",
    "DecisionContextReference",
    "DecisionContextRepresentation",
    "DecisionContextResult",
    "MandatoryDependencyWitnessEntry",
    "MissingDecisionContextExpansionError",
    "compile_decision_context",
    "render_decision_context",
]
