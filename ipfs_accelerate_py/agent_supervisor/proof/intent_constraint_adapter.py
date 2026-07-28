"""Compile pinned IntentIR action contracts into supervisor constraints.

This module is a provider-free semantic boundary.  It consumes artifacts that
were verified by :mod:`ir_registry` and normalized by :mod:`ir_adapters`; it
does not retrieve documents, invoke a model, or parse prose.  Only explicit
reviewed fields are compiled.  Everything else is retained as an unsupported
statement and therefore makes conformance fail closed.

Intent describes required work.  It never authorizes that work.  Retrieval,
SkillCenter, GraphRAG, IntentIR, and formalization records are context or
constraint inputs only; a separate SecurityIR decision is required before
execution.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..context.decision_contracts import canonical_artifact_bytes
from ..planning.formal_planning_contracts import FormalWorkPlan
from .ir_adapters import (
    FormalizationIRAdapter,
    IRAdapterResult,
    IntentIRAdapter,
    IRNodeKind,
    NormalizedIRArtifact,
    NormalizedIRNode,
    NormalizedResultAuthority,
)
from .ir_registry import (
    IRDeclaredAuthority,
    IRFamily,
    IRReviewState,
    IRTrustState,
    VerifiedIRArtifact,
)


INTENT_CONSTRAINT_ADAPTER_VERSION: Final[int] = 1
INTENT_CONSTRAINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-constraint@1"
)
INTENT_SOURCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-source-binding@1"
)
INTENT_PROOF_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-proof-obligation@1"
)
INTENT_CONSTRAINT_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-constraint-set@1"
)
INTENT_COMPILATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-constraint-compilation@1"
)
INTENT_CONFORMANCE_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-conformance-request@1"
)
INTENT_CONFORMANCE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-conformance-result@1"
)

DEFAULT_MAX_INTENT_NODES: Final[int] = 4096
DEFAULT_MAX_INTENT_EDGES: Final[int] = 8192
DEFAULT_MAX_CANDIDATE_ACTIONS: Final[int] = 4096
DEFAULT_MAX_CANONICAL_BYTES: Final[int] = 2 * 1024 * 1024


class IntentConstraintError(ValueError):
    """A constraint artifact or conformance request is malformed."""


class IntentConstraintKind(str, Enum):
    GOAL = "goal"
    ACTION = "action"
    CONTROL_FLOW = "control_flow"
    PRECONDITION = "precondition"
    GUARD = "guard"
    INVARIANT = "invariant"
    EFFECT = "effect"
    POSTCONDITION = "postcondition"
    ASSUMPTION = "assumption"
    FAILURE = "failure"
    RETRY = "retry"
    VERIFICATION = "verification"


class IntentControlFlowKind(str, Enum):
    ORDER = "order"
    SEQUENCE = "sequence"
    PARALLEL = "parallel"
    JOIN = "join"


class IntentCompilationStatus(str, Enum):
    COMPILED = "compiled"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class IntentConformanceVerdict(str, Enum):
    CONFORMANT = "conformant"
    NONCONFORMANT = "nonconformant"
    INVALID = "invalid"


class IntentFindingCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    WRONG_IR_FAMILY = "wrong_ir_family"
    ROOT_CHANGED = "root_changed"
    GRAPH_TRUNCATED = "graph_truncated"
    UNKNOWN_CONSTRAINT_KIND = "unknown_constraint_kind"
    UNSUPPORTED_STATEMENT = "unsupported_statement"
    UNKNOWN_REFERENCE = "unknown_reference"
    CONTROL_FLOW_CYCLE = "control_flow_cycle"
    MISSING_REQUIRED_ACTION = "missing_required_action"
    MISSING_REQUIRED_GOAL = "missing_required_goal"
    ORDERING_VIOLATION = "ordering_violation"
    PARALLEL_JOIN_VIOLATION = "parallel_join_violation"
    UNSATISFIED_PRECONDITION = "unsatisfied_precondition"
    UNSATISFIED_GUARD = "unsatisfied_guard"
    UNSATISFIED_INVARIANT = "unsatisfied_invariant"
    UNSATISFIED_POSTCONDITION = "unsatisfied_postcondition"
    UNSATISFIED_ASSUMPTION = "unsatisfied_assumption"
    UNSATISFIED_FAILURE_CONTRACT = "unsatisfied_failure_contract"
    UNSATISFIED_RETRY_CONTRACT = "unsatisfied_retry_contract"
    MISSING_VERIFICATION = "missing_verification"
    UNBOUND_INFERRED_REQUIREMENT = "unbound_inferred_requirement"
    UNDECLARED_EFFECT = "undeclared_effect"
    MISSING_EFFECT = "missing_effect"
    CONTRADICTORY_EFFECT = "contradictory_effect"
    PROOF_OBLIGATION_UNDISCHARGED = "proof_obligation_undischarged"
    INTENT_USED_AS_AUTHORIZATION = "intent_used_as_authorization"
    RETRIEVAL_USED_AS_AUTHORIZATION = "retrieval_used_as_authorization"


_KIND_ALIASES: Final[Mapping[str, IntentConstraintKind]] = MappingProxyType(
    {
        "objective": IntentConstraintKind.GOAL,
        "desired_state": IntentConstraintKind.GOAL,
        "task": IntentConstraintKind.ACTION,
        "step": IntentConstraintKind.ACTION,
        "operation": IntentConstraintKind.ACTION,
        "ordering": IntentConstraintKind.CONTROL_FLOW,
        "order": IntentConstraintKind.CONTROL_FLOW,
        "sequence": IntentConstraintKind.CONTROL_FLOW,
        "parallel": IntentConstraintKind.CONTROL_FLOW,
        "join": IntentConstraintKind.CONTROL_FLOW,
        "condition": IntentConstraintKind.PRECONDITION,
        "requires": IntentConstraintKind.PRECONDITION,
        "requirement": IntentConstraintKind.PRECONDITION,
        "safety_invariant": IntentConstraintKind.INVARIANT,
        "result": IntentConstraintKind.EFFECT,
        "outcome": IntentConstraintKind.EFFECT,
        "ensures": IntentConstraintKind.POSTCONDITION,
        "failure_mode": IntentConstraintKind.FAILURE,
        "on_failure": IntentConstraintKind.FAILURE,
        "retry_policy": IntentConstraintKind.RETRY,
        "evidence": IntentConstraintKind.VERIFICATION,
        "proof": IntentConstraintKind.VERIFICATION,
        **{item.value: item for item in IntentConstraintKind},
    }
)
_FLOW_ALIASES: Final[Mapping[str, IntentControlFlowKind]] = MappingProxyType(
    {
        "before": IntentControlFlowKind.ORDER,
        "after": IntentControlFlowKind.ORDER,
        "dependency": IntentControlFlowKind.ORDER,
        "depends_on": IntentControlFlowKind.ORDER,
        **{item.value: item for item in IntentControlFlowKind},
    }
)
_REFERENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_id",
        "action_ids",
        "after",
        "before",
        "depends_on",
        "effect_id",
        "event_id",
        "failure_id",
        "fluent_id",
        "formalization_id",
        "formula_id",
        "from",
        "goal_id",
        "guard_id",
        "invariant_id",
        "join_action_id",
        "member_action_ids",
        "node_id",
        "obligation_id",
        "postcondition_id",
        "precondition_id",
        "requirement_id",
        "retry_id",
        "statement_id",
        "target",
        "to",
        "verification_id",
    }
)
_EXPRESSION_IGNORED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "declaration_id",
        "view_id",
        "claim_id",
        "assumption_id",
        "obligation_id",
        "result_id",
        "kind",
        "type",
        "declaration_kind",
        "view_kind",
        "claim_kind",
        "origin",
        "grounded",
        "required",
        "source_references",
        "sources",
        "provenance",
        "provenance_references",
    }
)
_AUTHORIZATION_CONTEXT_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "intent",
        "intentir",
        "intent_ir",
        "retrieval",
        "retrieved",
        "skillcenter",
        "skill_center",
        "graphrag",
        "graph_rag",
        "rag",
    }
)
_AUTHORIZATION_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "authoriz",
        "authorit",
        "grant",
        "permission",
        "permit",
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _encoded(value: Any) -> bytes:
    try:
        return canonical_artifact_bytes(_plain(value))
    except (TypeError, ValueError) as exc:
        raise IntentConstraintError("value is not canonical DAG-JSON") from exc


def _identity(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:" + hashlib.sha256(_encoded(value)).hexdigest()


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise IntentConstraintError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise IntentConstraintError(f"{name} is not canonical text")
    if required and not value:
        raise IntentConstraintError(f"{name} must not be empty")
    if len(value.encode("utf-8")) > 8192:
        raise IntentConstraintError(f"{name} exceeds its byte bound")
    return value


def _strings(
    value: Any, name: str, *, required: bool = False, maximum: int = 4096
) -> tuple[str, ...]:
    if value is None:
        value = ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise IntentConstraintError(f"{name} must be a sequence")
    if len(value) > maximum:
        raise IntentConstraintError(f"{name} exceeds its count bound")
    items = tuple(_text(item, name) for item in value)
    if required and not items:
        raise IntentConstraintError(f"{name} must not be empty")
    if len(items) != len(set(items)):
        raise IntentConstraintError(f"{name} contains duplicates")
    return tuple(sorted(items))


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise IntentConstraintError(f"{name} must be a boolean")
    return value


def _kind(value: str) -> IntentConstraintKind | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _KIND_ALIASES.get(normalized)


def _flow_kind(value: Any, fallback: str) -> IntentControlFlowKind:
    raw = str(value or fallback).strip().lower().replace("-", "_")
    return _FLOW_ALIASES.get(raw, IntentControlFlowKind.ORDER)


def _ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_text(value, "reference"),) if value else ()
    if isinstance(value, Sequence):
        return tuple(
            sorted({_text(item, "reference") for item in value if item != ""})
        )
    raise IntentConstraintError("reference field must be a string or sequence")


def _artifact_root(artifact: NormalizedIRArtifact) -> dict[str, str]:
    return {
        "artifact_id": artifact.root_artifact_id,
        "cid_v1": artifact.root_cid_v1,
        "supervisor_digest": artifact.root_supervisor_digest,
    }


def _root_key(root: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(root.get("artifact_id") or ""),
        str(root.get("cid_v1") or ""),
        str(root.get("supervisor_digest") or ""),
    )


def _canonical_root(value: Any, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise IntentConstraintError(f"{name} must be an object")
    expected = {"artifact_id", "cid_v1", "supervisor_digest"}
    if set(value) != expected:
        raise IntentConstraintError(
            f"{name} must contain exactly artifact_id, cid_v1, and supervisor_digest"
        )
    root = {
        key: _text(value[key], f"{name}.{key}")
        for key in sorted(expected)
    }
    return _freeze(root)


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    claimed = payload.get("schema")
    if claimed is not None and claimed != expected:
        raise IntentConstraintError(
            f"schema mismatch: expected {expected}, received {claimed}"
        )


def _claimed(payload: Mapping[str, Any], name: str, expected: str) -> None:
    value = payload.get(name)
    if value is not None and value != expected:
        raise IntentConstraintError(f"{name} identity mismatch")


@dataclass(frozen=True)
class IntentAdapterBounds:
    max_nodes: int = DEFAULT_MAX_INTENT_NODES
    max_edges: int = DEFAULT_MAX_INTENT_EDGES
    max_candidate_actions: int = DEFAULT_MAX_CANDIDATE_ACTIONS
    max_canonical_bytes: int = DEFAULT_MAX_CANONICAL_BYTES

    def __post_init__(self) -> None:
        limits = {
            "max_nodes": (self.max_nodes, 100_000),
            "max_edges": (self.max_edges, 200_000),
            "max_candidate_actions": (self.max_candidate_actions, 100_000),
            "max_canonical_bytes": (self.max_canonical_bytes, 64 * 1024 * 1024),
        }
        for name, (value, maximum) in limits.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise IntentConstraintError(
                    f"{name} must be an integer from 1 through {maximum}"
                )


@dataclass(frozen=True)
class IntentSourceBinding:
    node_id: str
    artifact_family: IRFamily
    artifact_id: str
    artifact_root: Mapping[str, str]
    source_references: tuple[Mapping[str, Any], ...] = ()
    provenance_references: tuple[Mapping[str, Any], ...] = ()
    grounded: bool = True
    review_state: IRReviewState = IRReviewState.UNREVIEWED
    trust_state: IRTrustState = IRTrustState.UNKNOWN
    declared_authority: IRDeclaredAuthority = IRDeclaredAuthority.NONE
    result_authority: NormalizedResultAuthority = NormalizedResultAuthority.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(self, "artifact_family", IRFamily(self.artifact_family))
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "artifact_root",
            _canonical_root(self.artifact_root, "artifact_root"),
        )
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
        object.__setattr__(self, "grounded", _bool(self.grounded, "grounded"))
        object.__setattr__(self, "review_state", IRReviewState(self.review_state))
        object.__setattr__(self, "trust_state", IRTrustState(self.trust_state))
        object.__setattr__(
            self, "declared_authority", IRDeclaredAuthority(self.declared_authority)
        )
        object.__setattr__(
            self, "result_authority", NormalizedResultAuthority(self.result_authority)
        )

    @property
    def context_only(self) -> bool:
        return (
            not self.grounded
            or not self.review_state.accepted
            or not self.trust_state.accepted
            or self.result_authority
            in {
                NormalizedResultAuthority.CONTEXT_ONLY,
                NormalizedResultAuthority.PROPOSAL_ONLY,
                NormalizedResultAuthority.UNTRUSTED,
                NormalizedResultAuthority.NONE,
            }
        )

    @property
    def grants_execution_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INTENT_SOURCE_BINDING_SCHEMA,
            "adapter_version": INTENT_CONSTRAINT_ADAPTER_VERSION,
            "node_id": self.node_id,
            "artifact_family": self.artifact_family.value,
            "artifact_id": self.artifact_id,
            "artifact_root": _plain(self.artifact_root),
            "source_references": [_plain(item) for item in self.source_references],
            "provenance_references": [
                _plain(item) for item in self.provenance_references
            ],
            "grounded": self.grounded,
            "inferred": not self.grounded,
            "review_state": self.review_state.value,
            "trust_state": self.trust_state.value,
            "declared_authority": self.declared_authority.value,
            "result_authority": self.result_authority.value,
            "context_only": self.context_only,
            "grants_execution_authority": False,
        }

    @property
    def binding_id(self) -> str:
        return _identity("intent-source-binding", self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentSourceBinding":
        _schema(payload, INTENT_SOURCE_BINDING_SCHEMA)
        result = cls(
            node_id=payload.get("node_id", ""),
            artifact_family=payload.get("artifact_family", ""),
            artifact_id=payload.get("artifact_id", ""),
            artifact_root=payload.get("artifact_root") or {},
            source_references=tuple(payload.get("source_references") or ()),
            provenance_references=tuple(
                payload.get("provenance_references") or ()
            ),
            grounded=payload.get("grounded", True),
            review_state=payload.get("review_state", IRReviewState.UNREVIEWED),
            trust_state=payload.get("trust_state", IRTrustState.UNKNOWN),
            declared_authority=payload.get(
                "declared_authority", IRDeclaredAuthority.NONE
            ),
            result_authority=payload.get(
                "result_authority", NormalizedResultAuthority.NONE
            ),
        )
        _claimed(payload, "binding_id", result.binding_id)
        return result


@dataclass(frozen=True)
class IntentConstraint:
    kind: IntentConstraintKind
    node_id: str
    action_ids: tuple[str, ...] = ()
    goal_ids: tuple[str, ...] = ()
    expression: Mapping[str, Any] = field(default_factory=dict)
    required: bool = True
    grounded: bool = True
    context_only: bool = False
    source_binding_ids: tuple[str, ...] = ()
    formalization_ids: tuple[str, ...] = ()
    constraint_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", IntentConstraintKind(self.kind))
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(self, "action_ids", _strings(self.action_ids, "action_ids"))
        object.__setattr__(self, "goal_ids", _strings(self.goal_ids, "goal_ids"))
        if not isinstance(self.expression, Mapping):
            raise IntentConstraintError("expression must be an object")
        object.__setattr__(self, "expression", _freeze(dict(self.expression)))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "grounded", _bool(self.grounded, "grounded"))
        object.__setattr__(
            self, "context_only", _bool(self.context_only, "context_only")
        )
        object.__setattr__(
            self,
            "source_binding_ids",
            _strings(self.source_binding_ids, "source_binding_ids", required=True),
        )
        object.__setattr__(
            self,
            "formalization_ids",
            _strings(self.formalization_ids, "formalization_ids"),
        )
        expected = _identity("intent-constraint", self._identity_payload())
        if self.constraint_id and self.constraint_id != expected:
            raise IntentConstraintError("constraint identity mismatch")
        object.__setattr__(self, "constraint_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTENT_CONSTRAINT_SCHEMA,
            "adapter_version": INTENT_CONSTRAINT_ADAPTER_VERSION,
            "kind": self.kind.value,
            "node_id": self.node_id,
            "action_ids": list(self.action_ids),
            "goal_ids": list(self.goal_ids),
            "expression": _plain(self.expression),
            "required": self.required,
            "grounded": self.grounded,
            "context_only": self.context_only,
            "source_binding_ids": list(self.source_binding_ids),
            "formalization_ids": list(self.formalization_ids),
            "grants_execution_authority": False,
        }

    @property
    def inferred(self) -> bool:
        return not self.grounded

    @property
    def grants_execution_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "constraint_id": self.constraint_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentConstraint":
        _schema(payload, INTENT_CONSTRAINT_SCHEMA)
        return cls(
            kind=payload.get("kind", ""),
            node_id=payload.get("node_id", ""),
            action_ids=tuple(payload.get("action_ids") or ()),
            goal_ids=tuple(payload.get("goal_ids") or ()),
            expression=payload.get("expression") or {},
            required=payload.get("required", True),
            grounded=payload.get("grounded", True),
            context_only=payload.get("context_only", False),
            source_binding_ids=tuple(payload.get("source_binding_ids") or ()),
            formalization_ids=tuple(payload.get("formalization_ids") or ()),
            constraint_id=payload.get("constraint_id", ""),
        )


@dataclass(frozen=True)
class IntentControlEdge:
    flow_kind: IntentControlFlowKind
    before_action_ids: tuple[str, ...]
    after_action_id: str
    source_constraint_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "flow_kind", IntentControlFlowKind(self.flow_kind))
        object.__setattr__(
            self,
            "before_action_ids",
            _strings(
                self.before_action_ids, "before_action_ids", required=True
            ),
        )
        object.__setattr__(
            self, "after_action_id", _text(self.after_action_id, "after_action_id")
        )
        object.__setattr__(
            self,
            "source_constraint_id",
            _text(self.source_constraint_id, "source_constraint_id"),
        )
        if self.after_action_id in self.before_action_ids:
            raise IntentConstraintError("control edge cannot be self-referential")

    @property
    def edge_id(self) -> str:
        return _identity("intent-control-edge", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_kind": self.flow_kind.value,
            "before_action_ids": list(self.before_action_ids),
            "after_action_id": self.after_action_id,
            "source_constraint_id": self.source_constraint_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentControlEdge":
        result = cls(
            flow_kind=payload.get("flow_kind", IntentControlFlowKind.ORDER),
            before_action_ids=tuple(payload.get("before_action_ids") or ()),
            after_action_id=payload.get("after_action_id", ""),
            source_constraint_id=payload.get("source_constraint_id", ""),
        )
        _claimed(payload, "edge_id", result.edge_id)
        return result


@dataclass(frozen=True)
class IntentProofObligation:
    obligation_kind: str
    subject_constraint_ids: tuple[str, ...]
    source_binding_ids: tuple[str, ...]
    required_evidence_ids: tuple[str, ...] = ()
    context_only: bool = False
    obligation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obligation_kind", _text(self.obligation_kind, "obligation_kind")
        )
        object.__setattr__(
            self,
            "subject_constraint_ids",
            _strings(
                self.subject_constraint_ids,
                "subject_constraint_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "source_binding_ids",
            _strings(self.source_binding_ids, "source_binding_ids", required=True),
        )
        object.__setattr__(
            self,
            "required_evidence_ids",
            _strings(self.required_evidence_ids, "required_evidence_ids"),
        )
        object.__setattr__(
            self, "context_only", _bool(self.context_only, "context_only")
        )
        expected = _identity("intent-proof-obligation", self._identity_payload())
        if self.obligation_id and self.obligation_id != expected:
            raise IntentConstraintError("proof obligation identity mismatch")
        object.__setattr__(self, "obligation_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTENT_PROOF_OBLIGATION_SCHEMA,
            "adapter_version": INTENT_CONSTRAINT_ADAPTER_VERSION,
            "obligation_kind": self.obligation_kind,
            "subject_constraint_ids": list(self.subject_constraint_ids),
            "source_binding_ids": list(self.source_binding_ids),
            "required_evidence_ids": list(self.required_evidence_ids),
            "context_only": self.context_only,
            "grants_execution_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "obligation_id": self.obligation_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentProofObligation":
        _schema(payload, INTENT_PROOF_OBLIGATION_SCHEMA)
        return cls(
            obligation_kind=payload.get("obligation_kind", ""),
            subject_constraint_ids=tuple(
                payload.get("subject_constraint_ids") or ()
            ),
            source_binding_ids=tuple(payload.get("source_binding_ids") or ()),
            required_evidence_ids=tuple(
                payload.get("required_evidence_ids") or ()
            ),
            context_only=payload.get("context_only", False),
            obligation_id=payload.get("obligation_id", ""),
        )


@dataclass(frozen=True)
class IntentFinding:
    code: IntentFindingCode
    message: str
    constraint_id: str = ""
    action_id: str = ""
    source_node_id: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", IntentFindingCode(self.code))
        object.__setattr__(self, "message", _text(self.message, "message"))
        for name in ("constraint_id", "action_id", "source_node_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if not isinstance(self.details, Mapping):
            raise IntentConstraintError("finding details must be an object")
        object.__setattr__(self, "details", _freeze(dict(self.details)))

    @property
    def finding_id(self) -> str:
        return _identity("intent-finding", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "constraint_id": self.constraint_id,
            "action_id": self.action_id,
            "source_node_id": self.source_node_id,
            "details": _plain(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentFinding":
        result = cls(
            code=payload.get("code", ""),
            message=payload.get("message", ""),
            constraint_id=payload.get("constraint_id", ""),
            action_id=payload.get("action_id", ""),
            source_node_id=payload.get("source_node_id", ""),
            details=payload.get("details") or {},
        )
        _claimed(payload, "finding_id", result.finding_id)
        return result


@dataclass(frozen=True)
class IntentConstraintSet:
    intent_artifact_id: str
    intent_root: Mapping[str, str]
    formalization_artifact_id: str
    formalization_root: Mapping[str, str]
    constraints: tuple[IntentConstraint, ...]
    control_edges: tuple[IntentControlEdge, ...]
    proof_obligations: tuple[IntentProofObligation, ...]
    source_bindings: tuple[IntentSourceBinding, ...]
    unsupported_node_ids: tuple[str, ...] = ()
    contradictory_effect_groups: tuple[tuple[str, ...], ...] = ()
    graph_truncated: bool = False

    def __post_init__(self) -> None:
        for name in ("intent_artifact_id", "formalization_artifact_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        for name in ("intent_root", "formalization_root"):
            object.__setattr__(
                self, name, _canonical_root(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "constraints",
            tuple(sorted(self.constraints, key=lambda item: item.constraint_id)),
        )
        object.__setattr__(
            self,
            "control_edges",
            tuple(sorted(self.control_edges, key=lambda item: item.edge_id)),
        )
        object.__setattr__(
            self,
            "proof_obligations",
            tuple(
                sorted(self.proof_obligations, key=lambda item: item.obligation_id)
            ),
        )
        object.__setattr__(
            self,
            "source_bindings",
            tuple(sorted(self.source_bindings, key=lambda item: item.binding_id)),
        )
        object.__setattr__(
            self,
            "unsupported_node_ids",
            _strings(self.unsupported_node_ids, "unsupported_node_ids"),
        )
        groups = tuple(
            sorted(
                {
                    _strings(group, "contradictory_effect_group", required=True)
                    for group in self.contradictory_effect_groups
                }
            )
        )
        object.__setattr__(self, "contradictory_effect_groups", groups)
        object.__setattr__(
            self, "graph_truncated", _bool(self.graph_truncated, "graph_truncated")
        )
        self._validate_references()

    def _validate_references(self) -> None:
        constraint_ids = {item.constraint_id for item in self.constraints}
        binding_ids = {item.binding_id for item in self.source_bindings}
        action_ids = {
            action_id
            for item in self.constraints
            if item.kind is IntentConstraintKind.ACTION
            for action_id in (item.action_ids or (item.node_id,))
        }
        for constraint in self.constraints:
            if not set(constraint.source_binding_ids) <= binding_ids:
                raise IntentConstraintError("constraint has an unknown source binding")
        for edge in self.control_edges:
            if edge.source_constraint_id not in constraint_ids:
                raise IntentConstraintError("control edge has an unknown constraint")
            if not set(edge.before_action_ids) <= action_ids:
                raise IntentConstraintError("control edge has an unknown source action")
            if edge.after_action_id not in action_ids:
                raise IntentConstraintError("control edge has an unknown target action")
        for obligation in self.proof_obligations:
            if not set(obligation.subject_constraint_ids) <= constraint_ids:
                raise IntentConstraintError(
                    "proof obligation has an unknown constraint"
                )
            if not set(obligation.source_binding_ids) <= binding_ids:
                raise IntentConstraintError("proof obligation has an unknown binding")

    @property
    def grants_execution_authority(self) -> bool:
        return False

    def constraints_of_kind(
        self, kind: IntentConstraintKind | str
    ) -> tuple[IntentConstraint, ...]:
        selected = IntentConstraintKind(kind)
        return tuple(item for item in self.constraints if item.kind is selected)

    @property
    def goals(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.GOAL)

    @property
    def actions(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.ACTION)

    @property
    def control_flow_constraints(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.CONTROL_FLOW)

    @property
    def preconditions(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.PRECONDITION)

    @property
    def guards(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.GUARD)

    @property
    def invariants(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.INVARIANT)

    @property
    def effects(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.EFFECT)

    @property
    def postconditions(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.POSTCONDITION)

    @property
    def assumptions(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.ASSUMPTION)

    @property
    def failures(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.FAILURE)

    @property
    def retries(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.RETRY)

    @property
    def verifications(self) -> tuple[IntentConstraint, ...]:
        return self.constraints_of_kind(IntentConstraintKind.VERIFICATION)

    @property
    def parallel_joins(self) -> tuple[IntentControlEdge, ...]:
        return tuple(
            item
            for item in self.control_edges
            if item.flow_kind is IntentControlFlowKind.JOIN
        )

    @property
    def constraint_set_id(self) -> str:
        return _identity("intent-constraint-set", self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTENT_CONSTRAINT_SET_SCHEMA,
            "adapter_version": INTENT_CONSTRAINT_ADAPTER_VERSION,
            "intent_artifact_id": self.intent_artifact_id,
            "intent_root": _plain(self.intent_root),
            "formalization_artifact_id": self.formalization_artifact_id,
            "formalization_root": _plain(self.formalization_root),
            "constraints": [item.to_dict() for item in self.constraints],
            "control_edges": [item.to_dict() for item in self.control_edges],
            "proof_obligations": [
                item.to_dict() for item in self.proof_obligations
            ],
            "source_bindings": [item.to_dict() for item in self.source_bindings],
            "unsupported_node_ids": list(self.unsupported_node_ids),
            "contradictory_effect_groups": [
                list(item) for item in self.contradictory_effect_groups
            ],
            "graph_truncated": self.graph_truncated,
            "grants_execution_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "constraint_set_id": self.constraint_set_id}

    @property
    def canonical_bytes(self) -> bytes:
        return _encoded(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentConstraintSet":
        _schema(payload, INTENT_CONSTRAINT_SET_SCHEMA)
        result = cls(
            intent_artifact_id=payload.get("intent_artifact_id", ""),
            intent_root=payload.get("intent_root") or {},
            formalization_artifact_id=payload.get(
                "formalization_artifact_id", ""
            ),
            formalization_root=payload.get("formalization_root") or {},
            constraints=tuple(
                IntentConstraint.from_dict(item)
                for item in payload.get("constraints") or ()
            ),
            control_edges=tuple(
                IntentControlEdge.from_dict(item)
                for item in payload.get("control_edges") or ()
            ),
            proof_obligations=tuple(
                IntentProofObligation.from_dict(item)
                for item in payload.get("proof_obligations") or ()
            ),
            source_bindings=tuple(
                IntentSourceBinding.from_dict(item)
                for item in payload.get("source_bindings") or ()
            ),
            unsupported_node_ids=tuple(
                payload.get("unsupported_node_ids") or ()
            ),
            contradictory_effect_groups=tuple(
                tuple(item)
                for item in payload.get("contradictory_effect_groups") or ()
            ),
            graph_truncated=payload.get("graph_truncated", False),
        )
        _claimed(payload, "constraint_set_id", result.constraint_set_id)
        return result


@dataclass(frozen=True)
class IntentConstraintCompilationResult:
    status: IntentCompilationStatus
    constraint_set: IntentConstraintSet | None = None
    findings: tuple[IntentFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", IntentCompilationStatus(self.status))
        object.__setattr__(
            self,
            "findings",
            tuple(
                sorted(
                    self.findings,
                    key=lambda item: (
                        item.code.value,
                        item.constraint_id,
                        item.source_node_id,
                        item.finding_id,
                    ),
                )
            ),
        )
        if self.status is IntentCompilationStatus.INVALID and self.constraint_set:
            raise IntentConstraintError("invalid compilation cannot carry constraints")
        if (
            self.status is not IntentCompilationStatus.INVALID
            and not self.constraint_set
        ):
            raise IntentConstraintError("non-invalid compilation requires constraints")

    @property
    def successful(self) -> bool:
        return self.status is IntentCompilationStatus.COMPILED

    @property
    def valid(self) -> bool:
        return self.successful

    @property
    def supported(self) -> bool:
        return self.status is not IntentCompilationStatus.UNSUPPORTED

    @property
    def compilation_id(self) -> str:
        return _identity("intent-compilation", self._identity_payload())

    def require_constraint_set(self) -> IntentConstraintSet:
        if self.constraint_set is None:
            raise IntentConstraintError("intent constraint compilation failed closed")
        return self.constraint_set

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTENT_COMPILATION_RESULT_SCHEMA,
            "adapter_version": INTENT_CONSTRAINT_ADAPTER_VERSION,
            "status": self.status.value,
            "constraint_set": (
                self.constraint_set.to_dict() if self.constraint_set else None
            ),
            "findings": [
                {**item.to_dict(), "finding_id": item.finding_id}
                for item in self.findings
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "compilation_id": self.compilation_id}

    @property
    def canonical_bytes(self) -> bytes:
        return _encoded(self.to_dict())

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "IntentConstraintCompilationResult":
        _schema(payload, INTENT_COMPILATION_RESULT_SCHEMA)
        result = cls(
            status=payload.get("status", IntentCompilationStatus.INVALID),
            constraint_set=(
                IntentConstraintSet.from_dict(payload["constraint_set"])
                if payload.get("constraint_set")
                else None
            ),
            findings=tuple(
                IntentFinding.from_dict(item)
                for item in payload.get("findings") or ()
            ),
        )
        _claimed(payload, "compilation_id", result.compilation_id)
        return result


def _normalized(
    value: NormalizedIRArtifact | IRAdapterResult | VerifiedIRArtifact,
    family: IRFamily,
) -> tuple[NormalizedIRArtifact, bool]:
    graph_truncated = False
    if isinstance(value, IRAdapterResult):
        artifact = value.require_artifact()
    elif isinstance(value, NormalizedIRArtifact):
        artifact = value
    elif isinstance(value, VerifiedIRArtifact):
        graph_truncated = bool(value.payload.get("truncated", False))
        adapter = (
            IntentIRAdapter()
            if family is IRFamily.INTENT
            else FormalizationIRAdapter()
        )
        artifact = adapter.normalize(value).require_artifact()
    else:
        raise IntentConstraintError(
            "artifact must be verified, normalized, or a successful adapter result"
        )
    if artifact.family is not family:
        raise IntentConstraintError(
            f"expected {family.value}, received {artifact.family.value}"
        )
    return artifact, graph_truncated


def _binding(
    artifact: NormalizedIRArtifact, node: NormalizedIRNode
) -> IntentSourceBinding:
    return IntentSourceBinding(
        node_id=node.node_id,
        artifact_family=artifact.family,
        artifact_id=artifact.source_artifact_id,
        artifact_root=_artifact_root(artifact),
        source_references=node.source_references,
        provenance_references=node.provenance_references,
        grounded=node.grounded,
        review_state=node.review_state,
        trust_state=node.trust_state,
        declared_authority=node.declared_authority,
        result_authority=node.result_authority,
    )


def _constraint_from_node(
    node: NormalizedIRNode,
    binding: IntentSourceBinding,
    kind: IntentConstraintKind,
) -> IntentConstraint:
    attributes = _plain(node.attributes)
    action_ids = set(_ids(attributes.get("action_id")))
    action_ids.update(_ids(attributes.get("action_ids")))
    if kind is IntentConstraintKind.ACTION and not action_ids:
        action_ids.add(node.node_id)
    goal_ids = set(_ids(attributes.get("goal_id")))
    goal_ids.update(_ids(attributes.get("goal_ids")))
    if kind is IntentConstraintKind.GOAL and not goal_ids:
        goal_ids.add(node.node_id)
    formalization_ids = set(_ids(attributes.get("formalization_id")))
    formalization_ids.update(_ids(attributes.get("formalization_ids")))
    formalization_ids.update(_ids(attributes.get("formula_id")))
    expression = {
        key: value
        for key, value in attributes.items()
        if key not in _EXPRESSION_IGNORED_KEYS
    }
    expression["declared_kind"] = node.declaration_kind
    required_value = attributes.get("required", True)
    if not isinstance(required_value, bool):
        raise IntentConstraintError("constraint required must be a boolean")
    return IntentConstraint(
        kind=kind,
        node_id=node.node_id,
        action_ids=tuple(action_ids),
        goal_ids=tuple(goal_ids),
        expression=expression,
        required=required_value,
        grounded=node.grounded,
        context_only=binding.context_only,
        source_binding_ids=(binding.binding_id,),
        formalization_ids=tuple(formalization_ids),
    )


def _control_edges(
    constraint: IntentConstraint,
) -> tuple[IntentControlEdge, ...]:
    expression = constraint.expression
    flow = _flow_kind(
        expression.get("flow_kind", expression.get("control_flow_kind")),
        str(expression.get("declared_kind") or "order"),
    )
    result: list[IntentControlEdge] = []
    sequence = _ids(expression.get("sequence"))
    if not sequence and flow is IntentControlFlowKind.SEQUENCE:
        sequence = _ids(expression.get("action_ids"))
    for before, after in zip(sequence, sequence[1:]):
        result.append(
            IntentControlEdge(flow, (before,), after, constraint.constraint_id)
        )
    before_ids = set(_ids(expression.get("before")))
    before_ids.update(_ids(expression.get("from")))
    before_ids.update(_ids(expression.get("source_action_id")))
    after_ids = set(_ids(expression.get("after")))
    after_ids.update(_ids(expression.get("to")))
    after_ids.update(_ids(expression.get("target_action_id")))
    if before_ids and after_ids:
        for after in sorted(after_ids):
            result.append(
                IntentControlEdge(
                    flow, tuple(before_ids), after, constraint.constraint_id
                )
            )
    members = set(_ids(expression.get("member_action_ids")))
    if flow in {IntentControlFlowKind.PARALLEL, IntentControlFlowKind.JOIN}:
        members.update(_ids(expression.get("action_ids")))
    join = str(
        expression.get("join_action_id")
        or expression.get("join")
        or expression.get("after")
        or ""
    )
    if members and join:
        result.append(
            IntentControlEdge(
                IntentControlFlowKind.JOIN,
                tuple(members),
                _text(join, "join_action_id"),
                constraint.constraint_id,
            )
        )
    unique = {item.edge_id: item for item in result}
    return tuple(unique[key] for key in sorted(unique))


def _action_dependency_edges(
    constraint: IntentConstraint,
) -> tuple[IntentControlEdge, ...]:
    dependencies = _ids(constraint.expression.get("depends_on"))
    if not dependencies or len(constraint.action_ids) != 1:
        return ()
    return (
        IntentControlEdge(
            IntentControlFlowKind.ORDER,
            dependencies,
            constraint.action_ids[0],
            constraint.constraint_id,
        ),
    )


def _effect_signature(constraint: IntentConstraint) -> tuple[str, str]:
    expression = constraint.expression
    target = str(
        expression.get("fluent_id")
        or expression.get("event_id")
        or expression.get("target")
        or ""
    )
    semantic = {
        key: _plain(expression.get(key))
        for key in ("operation", "fluent_id", "event_id", "target", "value")
        if key in expression
    }
    return (
        target,
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )


def _acyclic(edges: Sequence[IntentControlEdge]) -> bool:
    graph: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for edge in edges:
        nodes.add(edge.after_action_id)
        for before in edge.before_action_ids:
            graph[before].add(edge.after_action_id)
            nodes.add(before)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        if not all(visit(child) for child in graph[node]):
            return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(visit(node) for node in sorted(nodes))


class IntentConstraintAdapter:
    """Deterministic compiler and exact candidate-plan conformance checker."""

    def __init__(self, *, bounds: IntentAdapterBounds | None = None) -> None:
        self.bounds = bounds or IntentAdapterBounds()

    def compile(
        self,
        intent: NormalizedIRArtifact | IRAdapterResult | VerifiedIRArtifact,
        formalization: (
            NormalizedIRArtifact | IRAdapterResult | VerifiedIRArtifact
        ),
    ) -> IntentConstraintCompilationResult:
        try:
            intent_artifact, intent_truncated = _normalized(intent, IRFamily.INTENT)
            formal_artifact, formal_truncated = _normalized(
                formalization, IRFamily.FORMALIZATION
            )
            nodes = tuple(intent_artifact.nodes + formal_artifact.nodes)
            if len(nodes) > self.bounds.max_nodes:
                raise IntentConstraintError("IR node count exceeds compiler bound")
            return self._compile_normalized(
                intent_artifact,
                formal_artifact,
                nodes,
                graph_truncated=intent_truncated or formal_truncated,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            return IntentConstraintCompilationResult(
                status=IntentCompilationStatus.INVALID,
                findings=(
                    IntentFinding(
                        IntentFindingCode.INVALID_INPUT,
                        str(exc),
                    ),
                ),
            )

    def _compile_normalized(
        self,
        intent: NormalizedIRArtifact,
        formalization: NormalizedIRArtifact,
        nodes: tuple[NormalizedIRNode, ...],
        *,
        graph_truncated: bool,
    ) -> IntentConstraintCompilationResult:
        bindings: list[IntentSourceBinding] = []
        bindings_by_node: dict[str, IntentSourceBinding] = {}
        constraints: list[IntentConstraint] = []
        deferred_obligation_nodes: list[
            tuple[NormalizedIRNode, IntentSourceBinding]
        ] = []
        unsupported: list[str] = []
        findings: list[IntentFinding] = []
        node_artifacts: dict[str, NormalizedIRArtifact] = {}
        node_identities: dict[str, str] = {}
        duplicate_ids: set[str] = set()
        for artifact in (intent, formalization):
            for node in artifact.nodes:
                if node.node_id in node_artifacts:
                    duplicate_ids.add(node.node_id)
                else:
                    node_artifacts[node.node_id] = artifact
                    node_identities[node.node_id] = node.content_id
        if duplicate_ids:
            return IntentConstraintCompilationResult(
                status=IntentCompilationStatus.INVALID,
                findings=tuple(
                    IntentFinding(
                        IntentFindingCode.INVALID_INPUT,
                        "node identifier is ambiguous across pinned artifacts",
                        source_node_id=node_id,
                        details={
                            "same_content": node_identities.get(node_id)
                            == next(
                                node.content_id
                                for node in formalization.nodes
                                if node.node_id == node_id
                            )
                        },
                    )
                    for node_id in sorted(duplicate_ids)
                ),
            )
        for node in sorted(nodes, key=lambda item: (item.node_id, item.content_id)):
            artifact = node_artifacts[node.node_id]
            binding = _binding(artifact, node)
            bindings.append(binding)
            bindings_by_node[node.node_id] = binding
        for node in sorted(nodes, key=lambda item: (item.node_id, item.content_id)):
            artifact = node_artifacts[node.node_id]
            binding = bindings_by_node[node.node_id]
            if node.node_kind is IRNodeKind.RESULT_AUTHORITY:
                continue
            kind = _kind(node.declaration_kind)
            if node.node_kind is IRNodeKind.ASSUMPTION:
                kind = IntentConstraintKind.ASSUMPTION
            if node.node_kind is IRNodeKind.OBLIGATION and not (
                _ids(node.attributes.get("action_id"))
                or _ids(node.attributes.get("action_ids"))
            ):
                deferred_obligation_nodes.append((node, binding))
                continue
            if kind is None:
                if artifact.family is IRFamily.FORMALIZATION:
                    # A formal view may use a logic-specific kind such as
                    # ``tdfol`` or ``first_order``.  It is a source binding,
                    # not a model-invented IntentIR constraint.
                    continue
                unsupported.append(node.node_id)
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNKNOWN_CONSTRAINT_KIND,
                        "node uses an unreviewed constraint kind",
                        source_node_id=node.node_id,
                        details={"declaration_kind": node.declaration_kind},
                    )
                )
                continue
            constraint = _constraint_from_node(node, binding, kind)
            linked_bindings = {
                bindings_by_node[item].binding_id
                for item in constraint.formalization_ids
                if item in bindings_by_node
                and node_artifacts[item].family is IRFamily.FORMALIZATION
            }
            if linked_bindings:
                linked_records = tuple(
                    bindings_by_node[item]
                    for item in constraint.formalization_ids
                    if item in bindings_by_node
                    and node_artifacts[item].family is IRFamily.FORMALIZATION
                )
                constraint = replace(
                    constraint,
                    source_binding_ids=tuple(
                        set(constraint.source_binding_ids) | linked_bindings
                    ),
                    grounded=(
                        constraint.grounded
                        and all(item.grounded for item in linked_records)
                    ),
                    context_only=(
                        constraint.context_only
                        or any(item.context_only for item in linked_records)
                    ),
                    constraint_id="",
                )
            constraints.append(constraint)
            if (
                not binding.review_state.accepted
                or not binding.trust_state.accepted
                or (
                    linked_bindings
                    and any(
                        not item.review_state.accepted
                        or not item.trust_state.accepted
                        for item in linked_records
                    )
                )
            ):
                unsupported.append(node.node_id)
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNSUPPORTED_STATEMENT,
                        "constraint source is unreviewed or untrusted",
                        constraint_id=constraint.constraint_id,
                        source_node_id=node.node_id,
                    )
                )
            action_bound_kinds = {
                IntentConstraintKind.PRECONDITION,
                IntentConstraintKind.GUARD,
                IntentConstraintKind.INVARIANT,
                IntentConstraintKind.EFFECT,
                IntentConstraintKind.POSTCONDITION,
                IntentConstraintKind.FAILURE,
                IntentConstraintKind.RETRY,
                IntentConstraintKind.VERIFICATION,
            }
            if kind in action_bound_kinds and not constraint.action_ids:
                unsupported.append(node.node_id)
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNKNOWN_REFERENCE,
                        "action-scoped constraint lacks an exact action reference",
                        constraint_id=constraint.constraint_id,
                        source_node_id=node.node_id,
                    )
                )
            if (
                constraint.expression.get("supported") is False
                or constraint.expression.get("unsupported") is True
            ):
                unsupported.append(node.node_id)
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNSUPPORTED_STATEMENT,
                        "constraint explicitly marks its statement unsupported",
                        constraint_id=constraint.constraint_id,
                        source_node_id=node.node_id,
                    )
                )
            if kind is IntentConstraintKind.EFFECT:
                raw_target = (
                    constraint.expression.get("fluent_id")
                    or constraint.expression.get("event_id")
                    or constraint.expression.get("target")
                )
                operation = constraint.expression.get("operation")
                if (
                    not isinstance(raw_target, str)
                    or not raw_target
                    or raw_target != raw_target.strip()
                    or "\x00" in raw_target
                    or not isinstance(operation, str)
                    or not operation
                    or operation != operation.strip()
                    or "\x00" in operation
                ):
                    unsupported.append(node.node_id)
                    findings.append(
                        IntentFinding(
                            IntentFindingCode.UNSUPPORTED_STATEMENT,
                            "effect lacks an exact operation and target",
                            constraint_id=constraint.constraint_id,
                            source_node_id=node.node_id,
                        )
                    )

        control_edges: list[IntentControlEdge] = []
        for constraint in constraints:
            if constraint.kind is IntentConstraintKind.CONTROL_FLOW:
                declared_edges = _control_edges(constraint)
                control_edges.extend(declared_edges)
                if constraint.required and not declared_edges:
                    unsupported.append(constraint.node_id)
                    findings.append(
                        IntentFinding(
                            IntentFindingCode.UNSUPPORTED_STATEMENT,
                            "control-flow constraint lacks an exact action edge",
                            constraint_id=constraint.constraint_id,
                            source_node_id=constraint.node_id,
                        )
                    )
            if constraint.kind is IntentConstraintKind.ACTION:
                control_edges.extend(_action_dependency_edges(constraint))
        unique_edges = {item.edge_id: item for item in control_edges}
        control_edges = [unique_edges[key] for key in sorted(unique_edges)]
        if len(control_edges) > self.bounds.max_edges:
            raise IntentConstraintError(
                "control-flow edge count exceeds compiler bound"
            )

        action_ids = {
            action_id
            for constraint in constraints
            if constraint.kind is IntentConstraintKind.ACTION
            for action_id in constraint.action_ids
        }
        goal_ids = {
            goal_id
            for constraint in constraints
            if constraint.kind is IntentConstraintKind.GOAL
            for goal_id in constraint.goal_ids
        }
        intent_goal_ids = {
            goal_id
            for constraint in constraints
            if constraint.kind is IntentConstraintKind.GOAL
            and node_artifacts[constraint.node_id].family is IRFamily.INTENT
            for goal_id in constraint.goal_ids
        }
        intent_action_ids = {
            action_id
            for constraint in constraints
            if constraint.kind is IntentConstraintKind.ACTION
            and node_artifacts[constraint.node_id].family is IRFamily.INTENT
            for action_id in constraint.action_ids
        }
        if not intent_goal_ids:
            marker = f"{intent.source_artifact_id}#missing-goal"
            unsupported.append(marker)
            findings.append(
                IntentFinding(
                    IntentFindingCode.UNSUPPORTED_STATEMENT,
                    "IntentIR does not declare an exact goal",
                    source_node_id=marker,
                )
            )
        if not intent_action_ids:
            marker = f"{intent.source_artifact_id}#missing-action"
            unsupported.append(marker)
            findings.append(
                IntentFinding(
                    IntentFindingCode.UNSUPPORTED_STATEMENT,
                    "IntentIR does not declare an exact action",
                    source_node_id=marker,
                )
            )
        for constraint in constraints:
            if constraint.kind is not IntentConstraintKind.ACTION:
                unknown_actions = set(constraint.action_ids) - action_ids
                if unknown_actions:
                    unsupported.append(constraint.node_id)
                    findings.append(
                        IntentFinding(
                            IntentFindingCode.UNKNOWN_REFERENCE,
                            "constraint references an undeclared action",
                            constraint_id=constraint.constraint_id,
                            source_node_id=constraint.node_id,
                            details={"action_ids": sorted(unknown_actions)},
                        )
                    )
            if (
                constraint.kind is IntentConstraintKind.ACTION
                and set(constraint.goal_ids) - goal_ids
            ):
                unknown_goals = set(constraint.goal_ids) - goal_ids
                unsupported.append(constraint.node_id)
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNKNOWN_REFERENCE,
                        "action references an undeclared goal",
                        constraint_id=constraint.constraint_id,
                        source_node_id=constraint.node_id,
                        details={"goal_ids": sorted(unknown_goals)},
                    )
                )
        for edge in control_edges:
            unknown_actions = (
                set(edge.before_action_ids) | {edge.after_action_id}
            ) - action_ids
            if unknown_actions:
                source = next(
                    item
                    for item in constraints
                    if item.constraint_id == edge.source_constraint_id
                )
                unsupported.append(source.node_id)
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNKNOWN_REFERENCE,
                        "control flow references an undeclared action",
                        constraint_id=source.constraint_id,
                        source_node_id=source.node_id,
                        details={"action_ids": sorted(unknown_actions)},
                    )
                )

        if not _acyclic(control_edges):
            findings.append(
                IntentFinding(
                    IntentFindingCode.CONTROL_FLOW_CYCLE,
                    "declared action control flow contains a cycle",
                )
            )
            unsupported.extend(
                item.node_id
                for item in constraints
                if item.kind is IntentConstraintKind.CONTROL_FLOW
            )

        formal_ids = {
            node.node_id for node in formalization.nodes
        } | {
            value
            for node in formalization.nodes
            for key, raw in node.attributes.items()
            if key in _REFERENCE_KEYS
            for value in _ids(raw)
        }
        for constraint in constraints:
            missing = set(constraint.formalization_ids) - formal_ids
            if missing:
                unsupported.append(constraint.node_id)
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNSUPPORTED_STATEMENT,
                        "constraint references an unavailable formalization",
                        constraint_id=constraint.constraint_id,
                        source_node_id=constraint.node_id,
                        details={"formalization_ids": sorted(missing)},
                    )
                )

        effect_groups: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for constraint in constraints:
            if constraint.kind is IntentConstraintKind.EFFECT:
                target, semantic = _effect_signature(constraint)
                for action_id in constraint.action_ids:
                    effect_groups[f"{action_id}::{target}"][semantic].append(
                        constraint.constraint_id
                    )
        contradictory = tuple(
            tuple(
                sorted(
                    constraint_id
                    for ids in semantics.values()
                    for constraint_id in ids
                )
            )
            for target, semantics in sorted(effect_groups.items())
            if target.rsplit("::", 1)[-1] and len(semantics) > 1
        )
        for group in contradictory:
            findings.append(
                IntentFinding(
                    IntentFindingCode.CONTRADICTORY_EFFECT,
                    "declared effects conflict for the same action target",
                    details={"constraint_ids": list(group)},
                )
            )

        by_node = {item.node_id: item for item in constraints}
        by_id = {item.constraint_id: item for item in constraints}
        obligations: list[IntentProofObligation] = []
        for constraint in constraints:
            binding_ids = constraint.source_binding_ids
            evidence = set(_ids(constraint.expression.get("evidence_ids")))
            evidence.update(_ids(constraint.expression.get("verification_ids")))
            if constraint.inferred:
                obligations.append(
                    IntentProofObligation(
                        "bind_inferred_requirement",
                        (constraint.constraint_id,),
                        binding_ids,
                        tuple(evidence),
                        context_only=True,
                    )
                )
            if constraint.kind in {
                IntentConstraintKind.GUARD,
                IntentConstraintKind.INVARIANT,
                IntentConstraintKind.VERIFICATION,
            } and constraint.required:
                obligations.append(
                    IntentProofObligation(
                        f"establish_{constraint.kind.value}",
                        (constraint.constraint_id,),
                        binding_ids,
                        tuple(evidence),
                        context_only=constraint.context_only,
                    )
                )
        for node, binding in deferred_obligation_nodes:
            attributes = node.attributes
            subject_references: set[str] = set()
            for name in (
                "constraint_id",
                "constraint_ids",
                "subject_constraint_id",
                "subject_constraint_ids",
                "requirement_id",
                "requirement_ids",
                "subject_id",
                "subject_ids",
                "intent_node_id",
                "intent_node_ids",
            ):
                subject_references.update(_ids(attributes.get(name)))
            subjects = {
                value
                for reference in subject_references
                for value in (
                    (
                        reference
                        if reference in by_id
                        else by_node[reference].constraint_id
                        if reference in by_node
                        else ""
                    ),
                )
                if value
            }
            action_references = set(_ids(attributes.get("action_id")))
            action_references.update(_ids(attributes.get("action_ids")))
            if not subjects and action_references:
                subjects.update(
                    item.constraint_id
                    for item in constraints
                    if set(item.action_ids) & action_references
                )
            if not subjects:
                unsupported.append(node.node_id)
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNKNOWN_REFERENCE,
                        "proof obligation lacks an exact constraint subject",
                        source_node_id=node.node_id,
                    )
                )
                continue
            obligations.append(
                IntentProofObligation(
                    str(
                        attributes.get("obligation_kind")
                        or "declared_proof"
                    ),
                    tuple(subjects),
                    (binding.binding_id,),
                    _ids(attributes.get("evidence_ids")),
                    context_only=binding.context_only,
                )
            )
        unique_obligations = {
            item.obligation_id: item for item in obligations
        }

        constraint_set = IntentConstraintSet(
            intent_artifact_id=intent.source_artifact_id,
            intent_root=_artifact_root(intent),
            formalization_artifact_id=formalization.source_artifact_id,
            formalization_root=_artifact_root(formalization),
            constraints=tuple(constraints),
            control_edges=tuple(control_edges),
            proof_obligations=tuple(unique_obligations.values()),
            source_bindings=tuple(bindings),
            unsupported_node_ids=tuple(set(unsupported)),
            contradictory_effect_groups=contradictory,
            graph_truncated=graph_truncated,
        )
        if len(constraint_set.canonical_bytes) > self.bounds.max_canonical_bytes:
            raise IntentConstraintError("compiled constraint set exceeds byte bound")
        status = (
            IntentCompilationStatus.UNSUPPORTED
            if unsupported
            or contradictory
            or graph_truncated
            or any(
                item.code is IntentFindingCode.CONTROL_FLOW_CYCLE
                for item in findings
            )
            else IntentCompilationStatus.COMPILED
        )
        if graph_truncated:
            findings.append(
                IntentFinding(
                    IntentFindingCode.GRAPH_TRUNCATED,
                    "an input artifact reports a truncated semantic graph",
                )
            )
        return IntentConstraintCompilationResult(
            status=status,
            constraint_set=constraint_set,
            findings=tuple(findings),
        )

    def conform(
        self, request: "IntentConformanceRequest"
    ) -> "IntentConformanceResult":
        return evaluate_intent_conformance(request, bounds=self.bounds)


def _candidate_record(candidate: Mapping[str, Any] | FormalWorkPlan) -> dict[str, Any]:
    if isinstance(candidate, FormalWorkPlan):
        actions = []
        preconditions = {item.precondition_id: item for item in candidate.preconditions}
        effects = {item.effect_id: item for item in candidate.effects}
        for task in candidate.tasks:
            actions.append(
                {
                    "action_id": task.task_id,
                    "goal_ids": [task.goal_id],
                    "depends_on": list(task.depends_on),
                    "precondition_ids": list(task.precondition_ids),
                    "preconditions": [
                        preconditions[item].to_record()
                        for item in task.precondition_ids
                        if item in preconditions
                    ],
                    "effect_ids": list(task.effect_ids),
                    "effects": [
                        effects[item].to_record()
                        for item in task.effect_ids
                        if item in effects
                    ],
                    "verification_ids": list(task.evidence_requirement_ids),
                }
            )
        return {
            "plan_id": candidate.plan_id,
            "goal_ids": [item.goal_id for item in candidate.goals],
            "actions": actions,
            "source_ids": list(candidate.source_ids),
            "metadata": _plain(candidate.metadata),
        }
    if not isinstance(candidate, Mapping):
        raise IntentConstraintError(
            "candidate_plan must be a mapping or FormalWorkPlan"
        )
    return _plain(candidate)


@dataclass(frozen=True)
class IntentConformanceRequest:
    constraint_set: IntentConstraintSet
    candidate_plan: Mapping[str, Any] | FormalWorkPlan
    intent_root: Mapping[str, str] | None = None
    formalization_root: Mapping[str, str] | None = None
    inferred_requirement_bindings: Mapping[str, Any] = field(default_factory=dict)
    discharged_obligation_ids: tuple[str, ...] = ()
    supported_statement_ids: tuple[str, ...] = ()
    graph_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.constraint_set, IntentConstraintSet):
            raise IntentConstraintError(
                "constraint_set must be an IntentConstraintSet"
            )
        candidate = _candidate_record(self.candidate_plan)
        object.__setattr__(self, "candidate_plan", _freeze(candidate))
        for name, fallback in (
            ("intent_root", self.constraint_set.intent_root),
            ("formalization_root", self.constraint_set.formalization_root),
        ):
            object.__setattr__(
                self,
                name,
                _canonical_root(getattr(self, name) or fallback, name),
            )
        if not isinstance(self.inferred_requirement_bindings, Mapping):
            raise IntentConstraintError(
                "inferred_requirement_bindings must be an object"
            )
        object.__setattr__(
            self,
            "inferred_requirement_bindings",
            _freeze(dict(self.inferred_requirement_bindings)),
        )
        object.__setattr__(
            self,
            "discharged_obligation_ids",
            _strings(
                self.discharged_obligation_ids, "discharged_obligation_ids"
            ),
        )
        object.__setattr__(
            self,
            "supported_statement_ids",
            _strings(self.supported_statement_ids, "supported_statement_ids"),
        )
        object.__setattr__(
            self, "graph_complete", _bool(self.graph_complete, "graph_complete")
        )
        known_obligations = {
            item.obligation_id for item in self.constraint_set.proof_obligations
        }
        unknown_discharges = set(self.discharged_obligation_ids) - known_obligations
        if unknown_discharges:
            raise IntentConstraintError(
                "discharged_obligation_ids contains an unknown obligation"
            )
        inferred_constraints = {
            item.constraint_id: item
            for item in self.constraint_set.constraints
            if item.inferred
        }
        inferred_keys = set(inferred_constraints) | {
            item.node_id for item in inferred_constraints.values()
        }
        if set(self.inferred_requirement_bindings) - inferred_keys:
            raise IntentConstraintError(
                "inferred_requirement_bindings contains an unknown requirement"
            )
        if set(self.supported_statement_ids) - set(
            self.constraint_set.unsupported_node_ids
        ):
            raise IntentConstraintError(
                "supported_statement_ids contains an unknown statement"
            )
        if len(self.canonical_bytes) > DEFAULT_MAX_CANONICAL_BYTES:
            raise IntentConstraintError("conformance request exceeds byte bound")

    @property
    def grants_execution_authority(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTENT_CONFORMANCE_REQUEST_SCHEMA,
            "adapter_version": INTENT_CONSTRAINT_ADAPTER_VERSION,
            "constraint_set_id": self.constraint_set.constraint_set_id,
            "constraint_set": self.constraint_set.to_dict(),
            "candidate_plan": _plain(self.candidate_plan),
            "intent_root": _plain(self.intent_root),
            "formalization_root": _plain(self.formalization_root),
            "inferred_requirement_bindings": _plain(
                self.inferred_requirement_bindings
            ),
            "discharged_obligation_ids": list(self.discharged_obligation_ids),
            "supported_statement_ids": list(self.supported_statement_ids),
            "graph_complete": self.graph_complete,
            "grants_execution_authority": False,
        }

    @property
    def request_id(self) -> str:
        return _identity("intent-conformance-request", self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "request_id": self.request_id}

    @property
    def canonical_bytes(self) -> bytes:
        return _encoded(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentConformanceRequest":
        _schema(payload, INTENT_CONFORMANCE_REQUEST_SCHEMA)
        constraint_payload = payload.get("constraint_set")
        if not isinstance(constraint_payload, Mapping):
            raise IntentConstraintError(
                "canonical request requires its exact constraint_set"
            )
        result = cls(
            constraint_set=IntentConstraintSet.from_dict(constraint_payload),
            candidate_plan=payload.get("candidate_plan") or {},
            intent_root=payload.get("intent_root"),
            formalization_root=payload.get("formalization_root"),
            inferred_requirement_bindings=payload.get(
                "inferred_requirement_bindings"
            )
            or {},
            discharged_obligation_ids=tuple(
                payload.get("discharged_obligation_ids") or ()
            ),
            supported_statement_ids=tuple(
                payload.get("supported_statement_ids") or ()
            ),
            graph_complete=payload.get("graph_complete", True),
        )
        _claimed(payload, "request_id", result.request_id)
        _claimed(
            payload,
            "constraint_set_id",
            result.constraint_set.constraint_set_id,
        )
        return result


@dataclass(frozen=True)
class IntentConformanceResult:
    request_id: str
    constraint_set_id: str
    candidate_plan_id: str
    verdict: IntentConformanceVerdict
    findings: tuple[IntentFinding, ...] = ()
    checked_constraint_ids: tuple[str, ...] = ()
    checked_obligation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("request_id", "constraint_set_id", "candidate_plan_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "verdict", IntentConformanceVerdict(self.verdict))
        unique_findings = {item.finding_id: item for item in self.findings}
        object.__setattr__(
            self,
            "findings",
            tuple(
                unique_findings[key]
                for key in sorted(
                    unique_findings,
                    key=lambda key: (
                        unique_findings[key].code.value,
                        unique_findings[key].constraint_id,
                        key,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "checked_constraint_ids",
            _strings(self.checked_constraint_ids, "checked_constraint_ids"),
        )
        object.__setattr__(
            self,
            "checked_obligation_ids",
            _strings(self.checked_obligation_ids, "checked_obligation_ids"),
        )
        if self.verdict is IntentConformanceVerdict.CONFORMANT and self.findings:
            raise IntentConstraintError("conformant result cannot carry findings")
        if (
            self.verdict is not IntentConformanceVerdict.CONFORMANT
            and not self.findings
        ):
            raise IntentConstraintError("failed result requires findings")

    @property
    def conformant(self) -> bool:
        return self.verdict is IntentConformanceVerdict.CONFORMANT

    @property
    def authorizes_execution(self) -> bool:
        return False

    @property
    def result_id(self) -> str:
        return _identity("intent-conformance-result", self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTENT_CONFORMANCE_RESULT_SCHEMA,
            "adapter_version": INTENT_CONSTRAINT_ADAPTER_VERSION,
            "request_id": self.request_id,
            "constraint_set_id": self.constraint_set_id,
            "candidate_plan_id": self.candidate_plan_id,
            "verdict": self.verdict.value,
            "findings": [
                {**item.to_dict(), "finding_id": item.finding_id}
                for item in self.findings
            ],
            "checked_constraint_ids": list(self.checked_constraint_ids),
            "checked_obligation_ids": list(self.checked_obligation_ids),
            "authorizes_execution": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "result_id": self.result_id}

    @property
    def canonical_bytes(self) -> bytes:
        return _encoded(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentConformanceResult":
        _schema(payload, INTENT_CONFORMANCE_RESULT_SCHEMA)
        result = cls(
            request_id=payload.get("request_id", ""),
            constraint_set_id=payload.get("constraint_set_id", ""),
            candidate_plan_id=payload.get("candidate_plan_id", ""),
            verdict=payload.get("verdict", IntentConformanceVerdict.INVALID),
            findings=tuple(
                IntentFinding.from_dict(item)
                for item in payload.get("findings") or ()
            ),
            checked_constraint_ids=tuple(
                payload.get("checked_constraint_ids") or ()
            ),
            checked_obligation_ids=tuple(
                payload.get("checked_obligation_ids") or ()
            ),
        )
        _claimed(payload, "result_id", result.result_id)
        return result


def _candidate_actions(
    candidate: Mapping[str, Any], bounds: IntentAdapterBounds
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[IntentFinding]]:
    raw = candidate.get("actions", candidate.get("tasks", ()))
    findings: list[IntentFinding] = []
    if isinstance(raw, Mapping):
        raw = [
            (
                {**dict(value), "action_id": key}
                if isinstance(value, Mapping) and "action_id" not in value
                else value
            )
            for key, value in raw.items()
        ]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return {}, {}, [
            IntentFinding(
                IntentFindingCode.INVALID_INPUT,
                "candidate actions must be a bounded sequence or mapping",
            )
        ]
    if len(raw) > bounds.max_candidate_actions:
        return {}, {}, [
            IntentFinding(
                IntentFindingCode.INVALID_INPUT,
                "candidate action count exceeds conformance bound",
            )
        ]
    actions: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            findings.append(
                IntentFinding(
                    IntentFindingCode.INVALID_INPUT,
                    "candidate action must be an object",
                )
            )
            continue
        action = _plain(item)
        action_id = str(
            action.get("action_id") or action.get("task_id") or action.get("id") or ""
        )
        if not action_id:
            findings.append(
                IntentFinding(
                    IntentFindingCode.INVALID_INPUT,
                    "candidate action requires an exact action_id",
                )
            )
            continue
        if action_id in actions:
            findings.append(
                IntentFinding(
                    IntentFindingCode.INVALID_INPUT,
                    "candidate action identifiers must be unique",
                    action_id=action_id,
                )
            )
            continue
        actions[action_id] = action
    return actions, actions, findings


def _tokens(action: Mapping[str, Any], fields: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for field_name in fields:
        value = action.get(field_name)
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, Sequence):
            for item in value:
                if isinstance(item, str):
                    result.add(item)
                elif isinstance(item, Mapping):
                    for key in _REFERENCE_KEYS:
                        raw = item.get(key)
                        if isinstance(raw, str):
                            result.add(raw)
        elif isinstance(value, Mapping):
            result.update(str(key) for key, present in value.items() if present)
    return result


def _constraint_tokens(constraint: IntentConstraint) -> set[str]:
    result = {constraint.constraint_id, constraint.node_id}
    for key in _REFERENCE_KEYS:
        raw = constraint.expression.get(key)
        if isinstance(raw, str):
            result.add(raw)
        elif isinstance(raw, Sequence):
            result.update(item for item in raw if isinstance(item, str))
    return result


_SATISFACTION_FIELDS: Final[Mapping[IntentConstraintKind, tuple[str, ...]]] = (
    MappingProxyType(
        {
            IntentConstraintKind.PRECONDITION: (
                "precondition_ids",
                "preconditions",
                "satisfied_precondition_ids",
                "satisfied_constraint_ids",
            ),
            IntentConstraintKind.GUARD: (
                "guard_ids",
                "guards",
                "satisfied_guard_ids",
                "satisfied_constraint_ids",
            ),
            IntentConstraintKind.INVARIANT: (
                "invariant_ids",
                "invariants",
                "satisfied_invariant_ids",
                "satisfied_constraint_ids",
            ),
            IntentConstraintKind.POSTCONDITION: (
                "postcondition_ids",
                "postconditions",
                "satisfied_postcondition_ids",
                "satisfied_constraint_ids",
            ),
            IntentConstraintKind.ASSUMPTION: (
                "assumption_ids",
                "assumptions",
                "satisfied_assumption_ids",
                "satisfied_constraint_ids",
            ),
            IntentConstraintKind.FAILURE: (
                "failure_ids",
                "failure_contracts",
                "satisfied_constraint_ids",
            ),
            IntentConstraintKind.RETRY: (
                "retry_ids",
                "retry_contracts",
                "satisfied_constraint_ids",
            ),
            IntentConstraintKind.VERIFICATION: (
                "verification_ids",
                "verifications",
                "evidence_ids",
                "satisfied_constraint_ids",
            ),
        }
    )
)
_UNSATISFIED_CODES: Final[Mapping[IntentConstraintKind, IntentFindingCode]] = (
    MappingProxyType(
        {
            IntentConstraintKind.PRECONDITION: (
                IntentFindingCode.UNSATISFIED_PRECONDITION
            ),
            IntentConstraintKind.GUARD: IntentFindingCode.UNSATISFIED_GUARD,
            IntentConstraintKind.INVARIANT: IntentFindingCode.UNSATISFIED_INVARIANT,
            IntentConstraintKind.POSTCONDITION: (
                IntentFindingCode.UNSATISFIED_POSTCONDITION
            ),
            IntentConstraintKind.ASSUMPTION: (
                IntentFindingCode.UNSATISFIED_ASSUMPTION
            ),
            IntentConstraintKind.FAILURE: (
                IntentFindingCode.UNSATISFIED_FAILURE_CONTRACT
            ),
            IntentConstraintKind.RETRY: (
                IntentFindingCode.UNSATISFIED_RETRY_CONTRACT
            ),
            IntentConstraintKind.VERIFICATION: (
                IntentFindingCode.MISSING_VERIFICATION
            ),
        }
    )
)


def _effect_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _plain(value.get(key))
        for key in ("operation", "fluent_id", "event_id", "target", "value")
        if key in value
    }


def _effect_key(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _effect_projection(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _authorization_findings(
    candidate: Mapping[str, Any],
    constraints: IntentConstraintSet,
) -> list[IntentFinding]:
    findings: list[IntentFinding] = []
    intent_ids = {
        constraints.intent_artifact_id,
        constraints.intent_root["artifact_id"],
        constraints.intent_root["cid_v1"],
        constraints.intent_root["supervisor_digest"],
        constraints.constraint_set_id,
        *(item.node_id for item in constraints.source_bindings),
        *(item.binding_id for item in constraints.source_bindings),
        *(item.constraint_id for item in constraints.constraints),
    }

    def walk(value: Any, path: str = "$") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower().replace("-", "_")
                if any(
                    marker in lowered
                    for marker in _AUTHORIZATION_FIELD_MARKERS
                ):
                    values = (
                        [item]
                        if isinstance(item, (str, bool))
                        else list(item)
                        if isinstance(item, Sequence)
                        else list(item.values())
                        if isinstance(item, Mapping)
                        else []
                    )
                    normalized = {
                        str(entry).strip().lower().replace("-", "_")
                        for entry in values
                    }
                    asserted = (
                        item is True
                        or (
                            isinstance(item, str)
                            and item.strip().lower()
                            not in {"", "false", "none", "no", "denied"}
                        )
                        or (
                            isinstance(item, (Mapping, Sequence))
                            and not isinstance(item, (str, bytes))
                            and bool(item)
                        )
                    )
                    if asserted:
                        normalized.add(lowered)
                    if any(str(entry) in intent_ids for entry in values):
                        findings.append(
                            IntentFinding(
                                IntentFindingCode.INTENT_USED_AS_AUTHORIZATION,
                                "candidate attempts to use intent as authorization",
                                details={"path": f"{path}.{key}"},
                            )
                        )
                    if any(
                        marker in entry
                        for entry in normalized
                        for marker in _AUTHORIZATION_CONTEXT_MARKERS
                    ):
                        code = (
                            IntentFindingCode.INTENT_USED_AS_AUTHORIZATION
                            if any(
                                marker in entry
                                for entry in normalized
                                for marker in ("intent", "intentir", "intent_ir")
                            )
                            else IntentFindingCode.RETRIEVAL_USED_AS_AUTHORIZATION
                        )
                        findings.append(
                            IntentFinding(
                                code,
                                "candidate uses a context-only premise as "
                                "authorization",
                                details={"path": f"{path}.{key}"},
                            )
                        )
                walk(item, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(candidate)
    return findings


def _inferred_binding_present(
    bindings: Mapping[str, Any], constraint: IntentConstraint
) -> bool:
    marker = object()
    value = bindings.get(constraint.constraint_id, marker)
    if value is marker:
        value = bindings.get(constraint.node_id, marker)
    if value is marker or value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, bytes):
        return bool(value)
    return True


def evaluate_intent_conformance(
    request: IntentConformanceRequest,
    *,
    bounds: IntentAdapterBounds | None = None,
) -> IntentConformanceResult:
    """Check one exact candidate plan against one exact compiled intent root."""

    if not isinstance(request, IntentConformanceRequest):
        raise IntentConstraintError("request must be an IntentConformanceRequest")
    selected_bounds = bounds or IntentAdapterBounds()
    constraint_set = request.constraint_set
    candidate = _plain(request.candidate_plan)
    candidate_id = _identity("candidate-plan", candidate)
    findings: list[IntentFinding] = []

    if len(request.canonical_bytes) > selected_bounds.max_canonical_bytes:
        findings.append(
            IntentFinding(
                IntentFindingCode.INVALID_INPUT,
                "conformance request exceeds the selected byte bound",
            )
        )
    if _root_key(request.intent_root or {}) != _root_key(
        constraint_set.intent_root
    ) or _root_key(request.formalization_root or {}) != _root_key(
        constraint_set.formalization_root
    ):
        findings.append(
            IntentFinding(
                IntentFindingCode.ROOT_CHANGED,
                "conformance request roots differ from compiled intent roots",
            )
        )
    candidate_intent_root = candidate.get("intent_root")
    candidate_formal_root = candidate.get("formalization_root")
    if (
        isinstance(candidate_intent_root, Mapping)
        and _root_key(candidate_intent_root) != _root_key(constraint_set.intent_root)
    ) or (
        isinstance(candidate_formal_root, Mapping)
        and _root_key(candidate_formal_root)
        != _root_key(constraint_set.formalization_root)
    ):
        findings.append(
            IntentFinding(
                IntentFindingCode.ROOT_CHANGED,
                "candidate plan binds a changed intent or formalization root",
            )
        )
    if (
        constraint_set.graph_truncated
        or not request.graph_complete
        or candidate.get("graph_truncated") is True
        or candidate.get("graph_complete") is False
    ):
        findings.append(
            IntentFinding(
                IntentFindingCode.GRAPH_TRUNCATED,
                "conformance requires a complete, untruncated semantic graph",
            )
        )
    candidate_constraint_set_id = candidate.get("constraint_set_id")
    if (
        candidate_constraint_set_id is not None
        and candidate_constraint_set_id != constraint_set.constraint_set_id
    ):
        findings.append(
            IntentFinding(
                IntentFindingCode.ROOT_CHANGED,
                "candidate plan binds a changed intent constraint set",
            )
        )
    candidate_root_fields = {
        "artifact_id": candidate.get("intent_root_id"),
        "cid_v1": candidate.get("intent_root_cid_v1"),
        "supervisor_digest": candidate.get("intent_root_supervisor_digest"),
    }
    for name, value in candidate_root_fields.items():
        if value is not None and value != constraint_set.intent_root[name]:
            findings.append(
                IntentFinding(
                    IntentFindingCode.ROOT_CHANGED,
                    "candidate plan binds a changed intent root component",
                    details={"field": name},
                )
            )
    for node_id in constraint_set.unsupported_node_ids:
        findings.append(
            IntentFinding(
                IntentFindingCode.UNSUPPORTED_STATEMENT,
                "compiled intent contains an unsupported statement",
                source_node_id=node_id,
            )
        )
    for group in constraint_set.contradictory_effect_groups:
        findings.append(
            IntentFinding(
                IntentFindingCode.CONTRADICTORY_EFFECT,
                "compiled intent declares contradictory effects",
                details={"constraint_ids": list(group)},
            )
        )

    actions, _, action_findings = _candidate_actions(candidate, selected_bounds)
    findings.extend(action_findings)
    candidate_goals = set(_ids(candidate.get("goal_ids")))
    raw_goals = candidate.get("goals")
    if isinstance(raw_goals, Sequence) and not isinstance(raw_goals, str):
        for goal in raw_goals:
            if isinstance(goal, str):
                candidate_goals.add(goal)
            elif isinstance(goal, Mapping):
                value = goal.get("goal_id", goal.get("id"))
                if isinstance(value, str):
                    candidate_goals.add(value)

    constraints_by_action: dict[str, list[IntentConstraint]] = defaultdict(list)
    for constraint in constraint_set.constraints:
        for action_id in constraint.action_ids:
            constraints_by_action[action_id].append(constraint)
        if constraint.inferred and constraint.required:
            bindings = request.inferred_requirement_bindings
            bound = _inferred_binding_present(bindings, constraint)
            if not bound:
                findings.append(
                    IntentFinding(
                        IntentFindingCode.UNBOUND_INFERRED_REQUIREMENT,
                        "inferred requirement lacks an explicit reviewed binding",
                        constraint_id=constraint.constraint_id,
                        source_node_id=constraint.node_id,
                    )
                )
        if not constraint.required:
            continue
        if constraint.kind is IntentConstraintKind.ACTION:
            for action_id in constraint.action_ids:
                if action_id not in actions:
                    findings.append(
                        IntentFinding(
                            IntentFindingCode.MISSING_REQUIRED_ACTION,
                            "candidate omits a required action",
                            constraint_id=constraint.constraint_id,
                            action_id=action_id,
                            source_node_id=constraint.node_id,
                        )
                    )
        elif constraint.kind is IntentConstraintKind.GOAL:
            for goal_id in constraint.goal_ids:
                if goal_id not in candidate_goals:
                    findings.append(
                        IntentFinding(
                            IntentFindingCode.MISSING_REQUIRED_GOAL,
                            "candidate omits a required goal",
                            constraint_id=constraint.constraint_id,
                            source_node_id=constraint.node_id,
                            details={"goal_id": goal_id},
                        )
                    )
        elif constraint.kind in _SATISFACTION_FIELDS:
            applicable_actions = (
                constraint.action_ids
                if constraint.action_ids
                else tuple(sorted(actions))
            )
            expected = _constraint_tokens(constraint)
            for action_id in applicable_actions:
                action = actions.get(action_id)
                if action is None:
                    continue
                actual = _tokens(action, _SATISFACTION_FIELDS[constraint.kind])
                if expected.isdisjoint(actual):
                    findings.append(
                        IntentFinding(
                            _UNSATISFIED_CODES[constraint.kind],
                            "candidate does not establish required "
                            f"{constraint.kind.value}",
                            constraint_id=constraint.constraint_id,
                            action_id=action_id,
                            source_node_id=constraint.node_id,
                        )
                    )

    for edge in constraint_set.control_edges:
        after = actions.get(edge.after_action_id)
        if after is None:
            continue
        dependencies = set(_ids(after.get("depends_on")))
        missing = set(edge.before_action_ids) - dependencies
        if missing:
            code = (
                IntentFindingCode.PARALLEL_JOIN_VIOLATION
                if edge.flow_kind is IntentControlFlowKind.JOIN
                else IntentFindingCode.ORDERING_VIOLATION
            )
            findings.append(
                IntentFinding(
                    code,
                    "candidate action dependencies do not satisfy declared "
                    "control flow",
                    constraint_id=edge.source_constraint_id,
                    action_id=edge.after_action_id,
                    details={"missing_dependencies": sorted(missing)},
                )
            )

    declared_effects: dict[str, dict[str, IntentConstraint]] = defaultdict(dict)
    for constraint in constraint_set.constraints:
        if constraint.kind is not IntentConstraintKind.EFFECT:
            continue
        projection = _effect_projection(constraint.expression)
        for action_id in constraint.action_ids:
            declared_effects[action_id][_effect_key(projection)] = constraint
    for action_id, action in actions.items():
        raw_effects = action.get("effects", ())
        if isinstance(raw_effects, Mapping):
            raw_effects = (raw_effects,)
        if isinstance(raw_effects, str) or not isinstance(raw_effects, Sequence):
            findings.append(
                IntentFinding(
                    IntentFindingCode.INVALID_INPUT,
                    "candidate effects must be a sequence",
                    action_id=action_id,
                )
            )
            continue
        actual_effects: dict[str, Mapping[str, Any]] = {}
        actual_targets: dict[str, set[str]] = defaultdict(set)
        for raw_effect in raw_effects:
            if not isinstance(raw_effect, Mapping):
                findings.append(
                    IntentFinding(
                        IntentFindingCode.INVALID_INPUT,
                        "candidate effect must be an object",
                        action_id=action_id,
                    )
                )
                continue
            key = _effect_key(raw_effect)
            actual_effects[key] = raw_effect
            projection = _effect_projection(raw_effect)
            target = str(
                projection.get("fluent_id")
                or projection.get("event_id")
                or projection.get("target")
                or ""
            )
            if target:
                actual_targets[target].add(key)
        for keys in actual_targets.values():
            if len(keys) > 1:
                findings.append(
                    IntentFinding(
                        IntentFindingCode.CONTRADICTORY_EFFECT,
                        "candidate declares contradictory effects for one target",
                        action_id=action_id,
                        details={"effects": sorted(keys)},
                    )
                )
        declared = declared_effects.get(action_id, {})
        for key, constraint in declared.items():
            if constraint.required and key not in actual_effects:
                findings.append(
                    IntentFinding(
                        IntentFindingCode.MISSING_EFFECT,
                        "candidate omits a declared required effect",
                        constraint_id=constraint.constraint_id,
                        action_id=action_id,
                        source_node_id=constraint.node_id,
                    )
                )
        for key in set(actual_effects) - set(declared):
            findings.append(
                IntentFinding(
                    IntentFindingCode.UNDECLARED_EFFECT,
                    "candidate introduces an effect absent from IntentIR",
                    action_id=action_id,
                    details={"effect": _effect_projection(actual_effects[key])},
                )
            )

    discharged = set(request.discharged_obligation_ids)
    for obligation in constraint_set.proof_obligations:
        inferred_binding = obligation.obligation_kind == "bind_inferred_requirement"
        if inferred_binding:
            subjects = set(obligation.subject_constraint_ids)
            bound = bool(
                any(
                    _inferred_binding_present(
                        request.inferred_requirement_bindings, constraint
                    )
                    for constraint in constraint_set.constraints
                    if constraint.constraint_id in subjects
                )
            )
            if bound:
                continue
        if obligation.obligation_id not in discharged:
            findings.append(
                IntentFinding(
                    IntentFindingCode.PROOF_OBLIGATION_UNDISCHARGED,
                    "required intent proof obligation was not discharged",
                    constraint_id=obligation.subject_constraint_ids[0],
                    details={"obligation_id": obligation.obligation_id},
                )
            )

    findings.extend(_authorization_findings(candidate, constraint_set))
    unique = {item.finding_id: item for item in findings}
    ordered = tuple(unique[key] for key in sorted(unique))
    invalid_codes = {
        IntentFindingCode.INVALID_INPUT,
        IntentFindingCode.ROOT_CHANGED,
        IntentFindingCode.GRAPH_TRUNCATED,
    }
    verdict = (
        IntentConformanceVerdict.CONFORMANT
        if not ordered
        else IntentConformanceVerdict.INVALID
        if any(item.code in invalid_codes for item in ordered)
        else IntentConformanceVerdict.NONCONFORMANT
    )
    return IntentConformanceResult(
        request_id=request.request_id,
        constraint_set_id=constraint_set.constraint_set_id,
        candidate_plan_id=candidate_id,
        verdict=verdict,
        findings=ordered,
        checked_constraint_ids=tuple(
            item.constraint_id for item in constraint_set.constraints
        ),
        checked_obligation_ids=tuple(
            item.obligation_id for item in constraint_set.proof_obligations
        ),
    )


def compile_intent_constraints(
    intent: NormalizedIRArtifact | IRAdapterResult | VerifiedIRArtifact,
    formalization: NormalizedIRArtifact | IRAdapterResult | VerifiedIRArtifact,
    *,
    bounds: IntentAdapterBounds | None = None,
) -> IntentConstraintCompilationResult:
    return IntentConstraintAdapter(bounds=bounds).compile(intent, formalization)


def create_intent_conformance_request(
    compilation: IntentConstraintCompilationResult | IntentConstraintSet,
    candidate_plan: Mapping[str, Any] | FormalWorkPlan,
    **kwargs: Any,
) -> IntentConformanceRequest:
    constraint_set = (
        compilation.require_constraint_set()
        if isinstance(compilation, IntentConstraintCompilationResult)
        else compilation
    )
    return IntentConformanceRequest(
        constraint_set=constraint_set,
        candidate_plan=candidate_plan,
        **kwargs,
    )


check_intent_conformance = evaluate_intent_conformance
conform_intent_plan = evaluate_intent_conformance
evaluate_intent_plan_conformance = evaluate_intent_conformance
compile_intent_action_contracts = compile_intent_constraints
IntentConstraintCompilation = IntentConstraintCompilationResult
IntentConstraintCompilationStatus = IntentCompilationStatus
IntentConstraintStatus = IntentCompilationStatus
IntentConstraintResult = IntentConstraintCompilationResult
IntentConformanceCheckRequest = IntentConformanceRequest
IntentConformanceCheckResult = IntentConformanceResult
IntentConformanceStatus = IntentConformanceVerdict
IntentPlanConformanceRequest = IntentConformanceRequest
IntentPlanConformanceResult = IntentConformanceResult


__all__ = [
    "DEFAULT_MAX_CANDIDATE_ACTIONS",
    "DEFAULT_MAX_CANONICAL_BYTES",
    "DEFAULT_MAX_INTENT_EDGES",
    "DEFAULT_MAX_INTENT_NODES",
    "INTENT_COMPILATION_RESULT_SCHEMA",
    "INTENT_CONFORMANCE_REQUEST_SCHEMA",
    "INTENT_CONFORMANCE_RESULT_SCHEMA",
    "INTENT_CONSTRAINT_ADAPTER_VERSION",
    "INTENT_CONSTRAINT_SCHEMA",
    "INTENT_CONSTRAINT_SET_SCHEMA",
    "INTENT_PROOF_OBLIGATION_SCHEMA",
    "INTENT_SOURCE_BINDING_SCHEMA",
    "IntentAdapterBounds",
    "IntentCompilationStatus",
    "IntentConformanceCheckRequest",
    "IntentConformanceCheckResult",
    "IntentConformanceRequest",
    "IntentConformanceResult",
    "IntentConformanceStatus",
    "IntentConformanceVerdict",
    "IntentConstraint",
    "IntentConstraintAdapter",
    "IntentConstraintCompilation",
    "IntentConstraintCompilationResult",
    "IntentConstraintCompilationStatus",
    "IntentConstraintError",
    "IntentConstraintKind",
    "IntentConstraintResult",
    "IntentConstraintSet",
    "IntentConstraintStatus",
    "IntentControlEdge",
    "IntentControlFlowKind",
    "IntentFinding",
    "IntentFindingCode",
    "IntentPlanConformanceRequest",
    "IntentPlanConformanceResult",
    "IntentProofObligation",
    "IntentSourceBinding",
    "check_intent_conformance",
    "compile_intent_action_contracts",
    "compile_intent_constraints",
    "conform_intent_plan",
    "create_intent_conformance_request",
    "evaluate_intent_conformance",
    "evaluate_intent_plan_conformance",
]
