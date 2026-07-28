"""Trust-preserving normalization for verified shared-IR artifacts.

These adapters perform structural normalization only.  They do not decide
IntentIR conformance, LegalIR applicability, SecurityIR authorization, or the
truth of a formalized claim.  In particular, no normalized record can grant
execution authority.

Large source corpora and provider payloads stay behind their pinned artifact
references.  The normalized records contain only bounded declarations,
formal views, claims, assumptions, obligations, and compact provenance/source
references needed by later proof-dependency compilation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..context.decision_contracts import canonical_artifact_bytes
from .ir_registry import (
    IRDeclaredAuthority,
    IRFailure,
    IRFailureCode,
    IRFamily,
    IRLoadResult,
    IRLoadStatus,
    IRRegistryError,
    IRReviewState,
    IRTrustState,
    VerifiedIRArtifact,
    normalize_ir_family,
)


IR_ADAPTER_VERSION: Final[int] = 1
IR_ADAPTER_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ir-adapter-capability@1"
)
NORMALIZED_IR_NODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/normalized-ir-node@1"
)
NORMALIZED_IR_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/normalized-ir-artifact@1"
)
IR_ADAPTER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ir-adapter-result@1"
)

DEFAULT_MAX_NORMALIZED_NODES: Final[int] = 4096
DEFAULT_MAX_NORMALIZED_BYTES: Final[int] = 512 * 1024
DEFAULT_MAX_NODE_BYTES: Final[int] = 16 * 1024
DEFAULT_MAX_REFERENCES_PER_NODE: Final[int] = 64

_FORBIDDEN_CORPUS_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "bytes",
        "content",
        "contents",
        "corpus",
        "corpora",
        "dataset",
        "document",
        "documents",
        "embedding",
        "embeddings",
        "full_source",
        "graph",
        "model_output",
        "model_response",
        "payload",
        "prompt",
        "raw",
        "raw_output",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "text",
        "transcript",
    }
)
_IDENTIFIER_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "node_id",
    "declaration_id",
    "view_id",
    "claim_id",
    "assumption_id",
    "obligation_id",
    "result_id",
)
_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact_id",
        "capability_id",
        "cid",
        "cid_v1",
        "configuration_id",
        "digest",
        "evidence_id",
        "model_id",
        "producer_id",
        "record_id",
        "reference_id",
        "revision",
        "schema",
        "source_id",
        "span_id",
        "supervisor_digest",
        "uri",
    }
)


class IRAdapterError(IRRegistryError):
    """An adapter declaration or direct call is malformed."""


class IRNodeKind(str, Enum):
    DECLARATION = "declaration"
    FORMAL_VIEW = "formal_view"
    CLAIM = "claim"
    ASSUMPTION = "assumption"
    OBLIGATION = "obligation"
    RESULT_AUTHORITY = "result_authority"


class NormalizedResultAuthority(str, Enum):
    """What a normalized record may contribute to a later decision.

    There is intentionally no ``execution`` or ``authorization`` member.
    """

    AUTHORITATIVE_INPUT = "authoritative_input"
    VERIFIED_INPUT = "verified_input"
    DESCRIPTIVE_INPUT = "descriptive_input"
    CONSTRAINT_INPUT = "constraint_input"
    POLICY_INPUT = "policy_input"
    PROPOSAL_ONLY = "proposal_only"
    CONTEXT_ONLY = "context_only"
    UNTRUSTED = "untrusted"
    NONE = "none"

    @property
    def grants_execution(self) -> bool:
        return False


class IRAdapterStatus(str, Enum):
    NORMALIZED = "normalized"
    VERIFIED = "normalized"
    UNSUPPORTED = IRFailureCode.UNSUPPORTED.value
    UNAVAILABLE = IRFailureCode.UNAVAILABLE.value
    PARTIAL = IRFailureCode.PARTIAL.value
    STALE = IRFailureCode.STALE.value
    QUARANTINED = IRFailureCode.QUARANTINED.value
    AMBIGUOUS = IRFailureCode.AMBIGUOUS.value
    CONTRADICTION = IRFailureCode.CONTRADICTION.value
    BOUNDS = IRFailureCode.BOUNDS.value
    BOUNDS_EXCEEDED = IRFailureCode.BOUNDS.value

    @property
    def successful(self) -> bool:
        return self is IRAdapterStatus.NORMALIZED


@dataclass(frozen=True)
class IRAdapterBounds:
    max_nodes: int = DEFAULT_MAX_NORMALIZED_NODES
    max_normalized_bytes: int = DEFAULT_MAX_NORMALIZED_BYTES
    max_node_bytes: int = DEFAULT_MAX_NODE_BYTES
    max_references_per_node: int = DEFAULT_MAX_REFERENCES_PER_NODE

    def __post_init__(self) -> None:
        limits = {
            "max_nodes": (self.max_nodes, 100_000),
            "max_normalized_bytes": (self.max_normalized_bytes, 64 * 1024 * 1024),
            "max_node_bytes": (self.max_node_bytes, 1024 * 1024),
            "max_references_per_node": (self.max_references_per_node, 4096),
        }
        for name, (value, maximum) in limits.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise IRAdapterError(
                    f"{name} must be an integer from 1 through {maximum}"
                )


def _text(value: Any, name: str, *, required: bool = True, maximum: int = 8192) -> str:
    if not isinstance(value, str):
        raise IRAdapterError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise IRAdapterError(f"{name} must not be empty")
    if result != value:
        raise IRAdapterError(f"{name} must not have surrounding whitespace")
    if "\x00" in result:
        raise IRAdapterError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise IRAdapterError(f"{name} exceeds {maximum} UTF-8 bytes")
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _encoded(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:" + hashlib.sha256(_encoded(value)).hexdigest()


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 16:
        raise OverflowError("normalized node exceeds nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise TypeError("floating values are not canonical IR metadata")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("normalized metadata keys must be strings")
            if key.lower() in _FORBIDDEN_CORPUS_FIELDS:
                continue
            result[key] = _compact_value(value[key], depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_compact_value(item, depth=depth + 1) for item in value]
    raise TypeError(f"unsupported normalized metadata {type(value).__name__}")


def _compact_reference(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, str):
        return MappingProxyType({"reference_id": _text(value, "reference")})
    if not isinstance(value, Mapping):
        return None
    result = {
        key: _compact_value(item)
        for key, item in value.items()
        if key in _REFERENCE_FIELDS
    }
    if not result:
        return None
    return MappingProxyType(dict(sorted(result.items())))


def _item_id(item: Mapping[str, Any]) -> str:
    for name in _IDENTIFIER_FIELDS:
        value = item.get(name)
        if isinstance(value, str) and value:
            return _text(value, name, maximum=2048)
    raise KeyError("id")


def _item_kind(item: Mapping[str, Any], fallback: IRNodeKind) -> str:
    for name in ("kind", "type", "declaration_kind", "view_kind", "claim_kind"):
        value = item.get(name)
        if isinstance(value, str) and value:
            return _text(value, name, maximum=512)
    return fallback.value


def _grounded(item: Mapping[str, Any], kind: IRNodeKind) -> bool:
    if "grounded" in item:
        if not isinstance(item["grounded"], bool):
            raise TypeError("grounded must be a boolean")
        return item["grounded"]
    origin = item.get("origin")
    if origin == "inferred":
        return False
    # Formal views and assumptions are never promoted merely because a source
    # omitted its origin marker.
    return kind not in {IRNodeKind.FORMAL_VIEW, IRNodeKind.ASSUMPTION}


def _result_authority(
    family: IRFamily,
    kind: IRNodeKind,
    declared: IRDeclaredAuthority,
    grounded: bool,
) -> NormalizedResultAuthority:
    if declared is IRDeclaredAuthority.UNTRUSTED:
        return NormalizedResultAuthority.UNTRUSTED
    if declared is IRDeclaredAuthority.NONE:
        return NormalizedResultAuthority.NONE
    if declared is IRDeclaredAuthority.PROPOSAL:
        return NormalizedResultAuthority.PROPOSAL_ONLY
    if declared in {
        IRDeclaredAuthority.ADVISORY,
        IRDeclaredAuthority.CONTEXT_ONLY,
    }:
        return NormalizedResultAuthority.CONTEXT_ONLY
    if kind is IRNodeKind.FORMAL_VIEW:
        return NormalizedResultAuthority.PROPOSAL_ONLY
    if kind is IRNodeKind.ASSUMPTION or not grounded:
        return NormalizedResultAuthority.CONTEXT_ONLY
    if kind is IRNodeKind.RESULT_AUTHORITY:
        # A producer's result-authority declaration is itself an input.  The
        # later verifier, not this adapter, decides whether it is effective.
        return NormalizedResultAuthority.VERIFIED_INPUT
    if family is IRFamily.INTENT:
        return NormalizedResultAuthority.DESCRIPTIVE_INPUT
    if family is IRFamily.LEGAL:
        return NormalizedResultAuthority.CONSTRAINT_INPUT
    if family is IRFamily.SECURITY:
        return NormalizedResultAuthority.POLICY_INPUT
    if declared is IRDeclaredAuthority.AUTHORITATIVE:
        return NormalizedResultAuthority.AUTHORITATIVE_INPUT
    if declared is IRDeclaredAuthority.VERIFIED:
        return NormalizedResultAuthority.VERIFIED_INPUT
    return NormalizedResultAuthority.CONTEXT_ONLY


def _node_authority(
    item: Mapping[str, Any], artifact: VerifiedIRArtifact
) -> IRDeclaredAuthority:
    value = item.get("declared_authority", item.get("authority"))
    if value is None:
        return artifact.declared_authority
    if isinstance(value, Mapping):
        value = value.get(
            "class", value.get("authority", value.get("declared_authority"))
        )
    try:
        declared = (
            value
            if isinstance(value, IRDeclaredAuthority)
            else IRDeclaredAuthority(str(value))
        )
    except ValueError as exc:
        raise TypeError("node declared_authority is unsupported") from exc
    ranks = {
        IRDeclaredAuthority.NONE: 0,
        IRDeclaredAuthority.UNTRUSTED: 0,
        IRDeclaredAuthority.CONTEXT_ONLY: 1,
        IRDeclaredAuthority.ADVISORY: 1,
        IRDeclaredAuthority.PROPOSAL: 1,
        IRDeclaredAuthority.VERIFIED: 2,
        IRDeclaredAuthority.AUTHORITATIVE: 3,
    }
    if ranks[declared] > ranks[artifact.declared_authority]:
        raise TypeError("node authority exceeds its verified artifact")
    return declared


def _node_review(item: Mapping[str, Any], artifact: VerifiedIRArtifact) -> IRReviewState:
    value = item.get("review_state", item.get("review"))
    if isinstance(value, Mapping):
        value = value.get("status")
    if value is None:
        return artifact.review_state
    try:
        return value if isinstance(value, IRReviewState) else IRReviewState(str(value))
    except ValueError as exc:
        raise TypeError("node review state is unsupported") from exc


def _node_trust(item: Mapping[str, Any], artifact: VerifiedIRArtifact) -> IRTrustState:
    value = item.get("trust_state", item.get("trust"))
    if isinstance(value, Mapping):
        value = value.get("state")
    if value is None:
        return artifact.trust_state
    try:
        return value if isinstance(value, IRTrustState) else IRTrustState(str(value))
    except ValueError as exc:
        raise TypeError("node trust state is unsupported") from exc


@dataclass(frozen=True)
class NormalizedIRNode:
    node_id: str
    family: IRFamily
    node_kind: IRNodeKind
    declaration_kind: str
    attributes: Mapping[str, Any]
    source_references: tuple[Mapping[str, Any], ...]
    provenance_references: tuple[Mapping[str, Any], ...]
    grounded: bool
    review_state: IRReviewState
    trust_state: IRTrustState
    declared_authority: IRDeclaredAuthority
    result_authority: NormalizedResultAuthority

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id", maximum=2048))
        object.__setattr__(self, "family", normalize_ir_family(self.family))
        if not isinstance(self.node_kind, IRNodeKind):
            object.__setattr__(self, "node_kind", IRNodeKind(str(self.node_kind)))
        object.__setattr__(
            self,
            "declaration_kind",
            _text(self.declaration_kind, "declaration_kind", maximum=512),
        )
        if not isinstance(self.attributes, Mapping):
            raise IRAdapterError("attributes must be an object")
        object.__setattr__(self, "attributes", _freeze(dict(self.attributes)))
        object.__setattr__(
            self,
            "source_references",
            tuple(_freeze(dict(item)) for item in self.source_references),
        )
        object.__setattr__(
            self,
            "provenance_references",
            tuple(_freeze(dict(item)) for item in self.provenance_references),
        )
        if not isinstance(self.grounded, bool):
            raise IRAdapterError("grounded must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NORMALIZED_IR_NODE_SCHEMA,
            "adapter_version": IR_ADAPTER_VERSION,
            "node_id": self.node_id,
            "family": self.family.value,
            "node_kind": self.node_kind.value,
            "declaration_kind": self.declaration_kind,
            "attributes": _plain(self.attributes),
            "source_references": [_plain(item) for item in self.source_references],
            "provenance_references": [
                _plain(item) for item in self.provenance_references
            ],
            "grounded": self.grounded,
            "review_state": self.review_state.value,
            "trust_state": self.trust_state.value,
            "declared_authority": self.declared_authority.value,
            "result_authority": self.result_authority.value,
            "grants_execution_authority": False,
        }

    @property
    def content_id(self) -> str:
        return _content_id("normalized-ir-node", self.to_dict())


NormalizedDeclaration = NormalizedIRNode
NormalizedFormalView = NormalizedIRNode
NormalizedClaim = NormalizedIRNode
NormalizedAssumption = NormalizedIRNode
NormalizedObligation = NormalizedIRNode


@dataclass(frozen=True)
class NormalizedIRArtifact:
    source_artifact_id: str
    source_cid_v1: str
    source_supervisor_digest: str
    root_artifact_id: str
    root_cid_v1: str
    root_supervisor_digest: str
    family: IRFamily
    artifact_schema: str
    artifact_schema_version: str
    producer_id: str
    producer_configuration_id: str
    review_state: IRReviewState
    trust_state: IRTrustState
    declared_authority: IRDeclaredAuthority
    declarations: tuple[NormalizedIRNode, ...] = ()
    formal_views: tuple[NormalizedIRNode, ...] = ()
    claims: tuple[NormalizedIRNode, ...] = ()
    assumptions: tuple[NormalizedIRNode, ...] = ()
    obligations: tuple[NormalizedIRNode, ...] = ()
    result_authority: tuple[NormalizedIRNode, ...] = ()

    @property
    def nodes(self) -> tuple[NormalizedIRNode, ...]:
        return (
            self.declarations
            + self.formal_views
            + self.claims
            + self.assumptions
            + self.obligations
            + self.result_authority
        )

    @property
    def proof_obligations(self) -> tuple[NormalizedIRNode, ...]:
        return self.obligations

    @property
    def grants_execution_authority(self) -> bool:
        return False

    @property
    def source_corpus_copied(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NORMALIZED_IR_ARTIFACT_SCHEMA,
            "adapter_version": IR_ADAPTER_VERSION,
            "source_artifact_id": self.source_artifact_id,
            "source_cid_v1": self.source_cid_v1,
            "source_supervisor_digest": self.source_supervisor_digest,
            "root_artifact_id": self.root_artifact_id,
            "root_cid_v1": self.root_cid_v1,
            "root_supervisor_digest": self.root_supervisor_digest,
            "family": self.family.value,
            "artifact_schema": self.artifact_schema,
            "artifact_schema_version": self.artifact_schema_version,
            "producer_id": self.producer_id,
            "producer_configuration_id": self.producer_configuration_id,
            "review_state": self.review_state.value,
            "trust_state": self.trust_state.value,
            "declared_authority": self.declared_authority.value,
            "declarations": [item.to_dict() for item in self.declarations],
            "formal_views": [item.to_dict() for item in self.formal_views],
            "claims": [item.to_dict() for item in self.claims],
            "assumptions": [item.to_dict() for item in self.assumptions],
            "obligations": [item.to_dict() for item in self.obligations],
            "result_authority": [item.to_dict() for item in self.result_authority],
            "source_corpus_copied": False,
            "grants_execution_authority": False,
        }

    @property
    def content_id(self) -> str:
        return _content_id("normalized-ir-artifact", self.to_dict())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_artifact_bytes(self.to_dict())


@dataclass(frozen=True)
class IRAdapterResult:
    status: IRAdapterStatus
    artifact: NormalizedIRArtifact | None = None
    failure: IRFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, IRAdapterStatus):
            object.__setattr__(self, "status", IRAdapterStatus(str(self.status)))
        if self.status.successful:
            if self.artifact is None or self.failure is not None:
                raise IRAdapterError("normalized results require only an artifact")
        elif self.artifact is not None or self.failure is None:
            raise IRAdapterError("failed adapter results require only a failure")
        if self.failure is not None and self.failure.code.value != self.status.value:
            raise IRAdapterError("failure code must match adapter status")

    @property
    def successful(self) -> bool:
        return self.status.successful

    @property
    def usable(self) -> bool:
        return self.successful

    @property
    def accepted(self) -> bool:
        return self.successful

    @property
    def failure_code(self) -> IRFailureCode | None:
        return self.failure.code if self.failure is not None else None

    @property
    def fail_closed(self) -> bool:
        return bool(self.failure and self.failure.fail_closed)

    def require_artifact(self) -> NormalizedIRArtifact:
        if self.artifact is None:
            assert self.failure is not None
            raise IRAdapterError(
                f"IR normalization failed closed: {self.failure.code.value}: "
                f"{self.failure.reason}"
            )
        return self.artifact

    def __bool__(self) -> bool:
        raise TypeError("IRAdapterResult has no truth value; inspect status explicitly")


@dataclass(frozen=True)
class IRAdapterCapability:
    adapter_id: str
    family: IRFamily
    operations: tuple[str, ...] = (
        "normalize_declarations",
        "normalize_formal_views",
        "normalize_claims",
        "normalize_assumptions",
        "normalize_obligations",
        "normalize_result_authority",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", _text(self.adapter_id, "adapter_id"))
        object.__setattr__(self, "family", normalize_ir_family(self.family))
        if isinstance(self.operations, str) or not isinstance(self.operations, Sequence):
            raise IRAdapterError("operations must be a sequence")
        values = tuple(sorted({_text(item, "operation", maximum=256) for item in self.operations}))
        if not values:
            raise IRAdapterError("operations must not be empty")
        object.__setattr__(self, "operations", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IR_ADAPTER_CAPABILITY_SCHEMA,
            "adapter_version": IR_ADAPTER_VERSION,
            "adapter_id": self.adapter_id,
            "family": self.family.value,
            "operations": list(self.operations),
            "lazy": True,
            "grants_execution_authority": False,
        }


_SECTION_KINDS: Final[tuple[tuple[str, IRNodeKind], ...]] = (
    ("declarations", IRNodeKind.DECLARATION),
    ("formal_views", IRNodeKind.FORMAL_VIEW),
    ("claims", IRNodeKind.CLAIM),
    ("assumptions", IRNodeKind.ASSUMPTION),
    ("obligations", IRNodeKind.OBLIGATION),
    ("result_authority", IRNodeKind.RESULT_AUTHORITY),
)
_SECTION_ALIASES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "formal_views": ("views", "logical_views"),
        "obligations": ("proof_obligations",),
        "result_authority": ("result_authorities",),
    }
)


class BaseIRAdapter:
    """Deterministic normalizer shared by all five IR families."""

    family: IRFamily
    adapter_id: str

    def __init__(
        self,
        family: IRFamily | str,
        *,
        adapter_id: str | None = None,
        bounds: IRAdapterBounds | None = None,
    ) -> None:
        self.family = normalize_ir_family(family)
        self.adapter_id = adapter_id or f"supervisor-{self.family.value}-adapter@1"
        self.bounds = bounds or IRAdapterBounds()

    @property
    def capability(self) -> IRAdapterCapability:
        return IRAdapterCapability(adapter_id=self.adapter_id, family=self.family)

    def normalize(
        self, artifact: VerifiedIRArtifact, *, required: bool = True
    ) -> IRAdapterResult:
        if not isinstance(artifact, VerifiedIRArtifact):
            raise IRAdapterError("artifact must be a VerifiedIRArtifact")
        if not isinstance(required, bool):
            raise IRAdapterError("required must be a boolean")
        if artifact.family is not self.family:
            return self._failure(
                artifact,
                IRFailureCode.UNSUPPORTED,
                "adapter family does not match verified artifact family",
                required=required,
            )

        normalized_sections: dict[str, tuple[NormalizedIRNode, ...]] = {}
        seen: dict[str, NormalizedIRNode] = {}
        node_count = 0
        try:
            for section, node_kind in _SECTION_KINDS:
                source = artifact.payload.get(section)
                if source is None:
                    for alias in _SECTION_ALIASES.get(section, ()):
                        if alias in artifact.payload:
                            source = artifact.payload[alias]
                            break
                if source is None:
                    source = ()
                if isinstance(source, Mapping) and section == "result_authority":
                    source = (source,)
                if isinstance(source, str) and section == "result_authority":
                    source = (
                        {
                            "id": (
                                f"result-authority:"
                                f"{artifact.reference.artifact_id}"
                            ),
                            "kind": "result_authority",
                            "value": source,
                        },
                    )
                if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
                    return self._failure(
                        artifact,
                        IRFailureCode.PARTIAL,
                        f"{section} must be a bounded sequence",
                        required=required,
                    )
                nodes: list[NormalizedIRNode] = []
                for item in source:
                    node_count += 1
                    if node_count > self.bounds.max_nodes:
                        raise OverflowError("normalized node count exceeds bound")
                    if not isinstance(item, Mapping):
                        return self._failure(
                            artifact,
                            IRFailureCode.PARTIAL,
                            f"{section} contains a non-object record",
                            required=required,
                        )
                    node = self._normalize_node(artifact, item, node_kind)
                    previous = seen.get(node.node_id)
                    if previous is not None:
                        code = (
                            IRFailureCode.AMBIGUOUS
                            if previous == node
                            else IRFailureCode.CONTRADICTION
                        )
                        return self._failure(
                            artifact,
                            code,
                            f"duplicate normalized node id: {node.node_id}",
                            required=required,
                        )
                    seen[node.node_id] = node
                    nodes.append(node)
                normalized_sections[section] = tuple(
                    sorted(nodes, key=lambda item: item.node_id)
                )
        except KeyError:
            return self._failure(
                artifact,
                IRFailureCode.PARTIAL,
                "normalized records require explicit stable identifiers",
                required=required,
            )
        except (OverflowError, TypeError, ValueError, IRAdapterError) as exc:
            return self._failure(
                artifact,
                IRFailureCode.BOUNDS,
                f"normalized record is malformed or outside bounds: {type(exc).__name__}",
                required=required,
            )

        normalized = NormalizedIRArtifact(
            source_artifact_id=artifact.reference.artifact_id,
            source_cid_v1=artifact.reference.cid_v1,
            source_supervisor_digest=artifact.reference.supervisor_digest,
            root_artifact_id=artifact.root_reference.artifact_id,
            root_cid_v1=artifact.root_reference.cid_v1,
            root_supervisor_digest=artifact.root_reference.supervisor_digest,
            family=artifact.family,
            artifact_schema=artifact.reference.artifact_schema,
            artifact_schema_version=artifact.reference.artifact_schema_version,
            producer_id=artifact.reference.producer_id,
            producer_configuration_id=artifact.producer_configuration_id,
            review_state=artifact.review_state,
            trust_state=artifact.trust_state,
            declared_authority=artifact.declared_authority,
            **normalized_sections,
        )
        if len(normalized.canonical_bytes) > self.bounds.max_normalized_bytes:
            return self._failure(
                artifact,
                IRFailureCode.BOUNDS,
                "normalized artifact exceeds max_normalized_bytes",
                required=required,
            )
        return IRAdapterResult(
            status=IRAdapterStatus.NORMALIZED,
            artifact=normalized,
        )

    def _normalize_node(
        self,
        artifact: VerifiedIRArtifact,
        item: Mapping[str, Any],
        node_kind: IRNodeKind,
    ) -> NormalizedIRNode:
        node_id = _item_id(item)
        compact = _compact_value(item)
        for name in _IDENTIFIER_FIELDS:
            compact.pop(name, None)
        source_values: list[Any] = []
        for name in ("source_reference", "source_references", "sources"):
            value = item.get(name)
            if value is None:
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                source_values.extend(value)
            else:
                source_values.append(value)
            compact.pop(name, None)
        sources = tuple(
            reference
            for value in source_values
            for reference in (_compact_reference(value),)
            if reference is not None
        )
        provenance_values = item.get("provenance", artifact.provenance)
        if isinstance(provenance_values, Mapping):
            provenance_values = (provenance_values,)
        if isinstance(provenance_values, (str, bytes)) or not isinstance(
            provenance_values, Sequence
        ):
            raise TypeError("node provenance must be a sequence")
        provenance = tuple(
            reference
            for value in provenance_values
            for reference in (_compact_reference(value),)
            if reference is not None
        )
        compact.pop("provenance", None)
        if (
            len(sources) > self.bounds.max_references_per_node
            or len(provenance) > self.bounds.max_references_per_node
        ):
            raise OverflowError("node reference count exceeds bound")
        grounded = _grounded(item, node_kind)
        declared_authority = _node_authority(item, artifact)
        review_state = _node_review(item, artifact)
        trust_state = _node_trust(item, artifact)
        authority = _result_authority(
            artifact.family, node_kind, declared_authority, grounded
        )
        if not review_state.accepted:
            authority = NormalizedResultAuthority.CONTEXT_ONLY
        if not trust_state.accepted:
            authority = NormalizedResultAuthority.UNTRUSTED
        for metadata_name in (
            "authority",
            "declared_authority",
            "review",
            "review_state",
            "trust",
            "trust_state",
        ):
            compact.pop(metadata_name, None)
        node = NormalizedIRNode(
            node_id=node_id,
            family=artifact.family,
            node_kind=node_kind,
            declaration_kind=_item_kind(item, node_kind),
            attributes=compact,
            source_references=sources,
            provenance_references=provenance,
            grounded=grounded,
            review_state=review_state,
            trust_state=trust_state,
            declared_authority=declared_authority,
            result_authority=authority,
        )
        if len(_encoded(node.to_dict())) > self.bounds.max_node_bytes:
            raise OverflowError("normalized node exceeds max_node_bytes")
        return node

    def _failure(
        self,
        artifact: VerifiedIRArtifact,
        code: IRFailureCode,
        reason: str,
        *,
        required: bool,
    ) -> IRAdapterResult:
        return IRAdapterResult(
            status=IRAdapterStatus(code.value),
            failure=IRFailure(
                code=code,
                reason=reason,
                required=required,
                artifact_id=artifact.reference.artifact_id,
                provider_id=self.adapter_id,
            ),
        )


class IRCoreAdapter(BaseIRAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(IRFamily.IR_CORE, **kwargs)


class FormalizationIRAdapter(BaseIRAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(IRFamily.FORMALIZATION, **kwargs)


class IntentIRAdapter(BaseIRAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(IRFamily.INTENT, **kwargs)


class LegalIRAdapter(BaseIRAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(IRFamily.LEGAL, **kwargs)


class SecurityIRAdapter(BaseIRAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(IRFamily.SECURITY, **kwargs)


class IRAdapterRegistry:
    """Provider-free adapter selection with declaration-only discovery."""

    def __init__(
        self,
        *,
        bounds: IRAdapterBounds | None = None,
        include_shared: bool = True,
    ) -> None:
        selected_bounds = bounds or IRAdapterBounds()
        adapters: list[BaseIRAdapter] = [
            IntentIRAdapter(bounds=selected_bounds),
            LegalIRAdapter(bounds=selected_bounds),
            SecurityIRAdapter(bounds=selected_bounds),
        ]
        if include_shared:
            adapters.extend(
                (
                    IRCoreAdapter(bounds=selected_bounds),
                    FormalizationIRAdapter(bounds=selected_bounds),
                )
            )
        self._adapters = {
            adapter.family: adapter
            for adapter in sorted(adapters, key=lambda item: item.family.value)
        }

    def discover_capabilities(
        self, family: IRFamily | str | None = None
    ) -> tuple[IRAdapterCapability, ...]:
        normalized = normalize_ir_family(family) if family is not None else None
        return tuple(
            adapter.capability
            for item_family, adapter in self._adapters.items()
            if normalized is None or item_family is normalized
        )

    discover = discover_capabilities

    def normalize(
        self,
        artifact: VerifiedIRArtifact | IRLoadResult,
        *,
        required: bool | None = None,
    ) -> IRAdapterResult:
        if isinstance(artifact, IRLoadResult):
            effective_required = (
                artifact.request.required if required is None else required
            )
            if artifact.status is not IRLoadStatus.VERIFIED:
                assert artifact.failure is not None
                failure = (
                    artifact.failure
                    if artifact.failure.required == effective_required
                    else IRFailure(
                        code=artifact.failure.code,
                        reason=artifact.failure.reason,
                        required=effective_required,
                        artifact_id=artifact.failure.artifact_id,
                        provider_id=artifact.failure.provider_id,
                        details=artifact.failure.details,
                    )
                )
                return IRAdapterResult(
                    status=IRAdapterStatus(artifact.status.value),
                    failure=failure,
                )
            verified = artifact.require_artifact()
        elif isinstance(artifact, VerifiedIRArtifact):
            effective_required = True if required is None else required
            verified = artifact
        else:
            raise IRAdapterError(
                "artifact must be VerifiedIRArtifact or IRLoadResult"
            )
        adapter = self._adapters.get(verified.family)
        if adapter is None:
            return IRAdapterResult(
                status=IRAdapterStatus.UNSUPPORTED,
                failure=IRFailure(
                    code=IRFailureCode.UNSUPPORTED,
                    reason="no adapter supports the exact verified IR family",
                    required=effective_required,
                    artifact_id=verified.reference.artifact_id,
                ),
            )
        return adapter.normalize(verified, required=effective_required)


def normalize_ir_artifact(
    artifact: VerifiedIRArtifact | IRLoadResult,
    *,
    bounds: IRAdapterBounds | None = None,
    required: bool | None = None,
) -> IRAdapterResult:
    """Normalize through the default provider-free adapter registry."""

    return IRAdapterRegistry(bounds=bounds).normalize(
        artifact, required=required
    )


__all__ = [
    "DEFAULT_MAX_NODE_BYTES",
    "DEFAULT_MAX_NORMALIZED_BYTES",
    "DEFAULT_MAX_NORMALIZED_NODES",
    "BaseIRAdapter",
    "FormalizationIRAdapter",
    "IRAdapterBounds",
    "IRAdapterCapability",
    "IRAdapterError",
    "IRAdapterRegistry",
    "IRAdapterResult",
    "IRAdapterStatus",
    "IRCoreAdapter",
    "IRNodeKind",
    "IntentIRAdapter",
    "LegalIRAdapter",
    "NORMALIZED_IR_ARTIFACT_SCHEMA",
    "NORMALIZED_IR_NODE_SCHEMA",
    "NormalizedAssumption",
    "NormalizedClaim",
    "NormalizedDeclaration",
    "NormalizedFormalView",
    "NormalizedIRArtifact",
    "NormalizedIRNode",
    "NormalizedObligation",
    "NormalizedResultAuthority",
    "SecurityIRAdapter",
    "normalize_ir_artifact",
]
