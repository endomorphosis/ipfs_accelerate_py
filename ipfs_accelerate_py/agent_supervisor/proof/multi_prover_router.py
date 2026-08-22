"""Property-specific, fail-closed multi-prover portfolio routing.

This module is deliberately an orchestration and trust-boundary module.  It
does not import or execute optional theorem provers itself.  Callers provide a
bounded :class:`PortfolioRunner`; the router selects the reviewed portfolio,
gates it against capability/conformance evidence when supplied, retains an
attempt for every selected lane, and derives the only authoritative verdict.

Solver and Hammer successes are candidates.  They can never promote
themselves to a proof.  A configured model-checking authority (for example
TLC for a state-machine property) or an independent Lean/Coq/Isabelle
reconstruction must accept the obligation before ``proved`` is returned.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ..autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    DeterministicRepairCapabilities,
    NetworkMode,
    SolverReadiness,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
    _canonical_value,
    content_identity,
)
from .mcp_contract_obligations import (
    MCP_GRAPH_OBLIGATION_SCHEMA,
    McpGraphContractObligation,
    McpObligationBackend,
    McpObligationDisposition,
    McpObligationFragment,
)
from .prover_conformance import (
    ConformanceGateDecision,
    ConformanceReport,
    ProverQuarantineRegistry,
    gate_prover_path,
)
from .prover_matrix_registry import ProverMatrixEntry, ProverMatrixSnapshot

MULTI_PROVER_ROUTER_VERSION = 1
PROPERTY_OBLIGATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/property-obligation@1"
)
PORTFOLIO_PLAN_SCHEMA = "ipfs_accelerate_py/agent-supervisor/prover-portfolio-plan@1"
PORTFOLIO_ATTEMPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prover-portfolio-attempt@1"
)
PORTFOLIO_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prover-portfolio-result@1"
)
AUTHORITY_LATTICE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/authority-lattice@1"
HAMMER_TRACE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/hammer-trace@1"
COUNTEREXAMPLE_TRACE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/solver-counterexample-trace@1"
)
CHECKER_TRACE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/independent-checker-trace@1"
)
AUTHORITATIVE_DISPOSITION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/authoritative-disposition@1"
)
DEFAULT_PORTFOLIO_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_PARALLEL_PROVERS = 8
DEFAULT_MAX_EVIDENCE_BYTES = 256 * 1024


class PropertyKind(str, Enum):
    """Semantic property families understood by the supervisor."""

    FINITE_CONSTRAINT = "finite_constraint"
    STATE_MACHINE = "state_machine"
    AUTHORIZATION = "authorization"
    PROTOCOL = "protocol"
    HYPERPROPERTY = "hyperproperty"
    RUNTIME_TRACE = "runtime_trace"
    KERNEL_CHECK = "kernel_check"
    TYPED_PLANNING = "typed_planning"
    TEMPORAL_DEONTIC = "temporal_deontic"
    FIRST_ORDER_THEOREM = "first_order_theorem"


# Compatibility-friendly semantic names.
PropertyType = PropertyKind
ObligationProperty = PropertyKind


class ProverRole(str, Enum):
    MODEL_ASSISTANT = "model_assistant"
    DOMAIN_REASONER = "domain_reasoner"
    ORCHESTRATOR = "orchestrator"
    CANDIDATE = "candidate"
    MODEL_CHECKER = "model_checker"
    KERNEL = "kernel"

    @property
    def authoritative(self) -> bool:
        return self in (ProverRole.MODEL_CHECKER, ProverRole.KERNEL)


class AttemptOutcome(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    COUNTEREXAMPLE = "counterexample"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    ERROR = "error"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class PortfolioVerdict(str, Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


RouteVerdict = PortfolioVerdict


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{name} is unsupported") from exc


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ContractValidationError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise ContractValidationError(f"{name} must not be empty")
    return value


def _strings(values: Iterable[Any] | None, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ContractValidationError(f"{name} must be a sequence")
    return tuple(
        sorted({_text(value, name) for value in values})
    )


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ContractValidationError(f"{name} must be an object with string keys")
    result = _canonical_value(dict(value))
    if not isinstance(result, dict):  # pragma: no cover
        raise ContractValidationError(f"{name} must be an object")
    return result


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ContractValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


def _claimed_identity(
    payload: Mapping[str, Any], actual: str, noun: str
) -> None:
    claimed = payload.get("content_id")
    if claimed and claimed != actual:
        raise ContractValidationError(f"{noun} content identity does not match")


def _strict_json_size(value: Mapping[str, Any], limit: int) -> dict[str, Any]:
    result = _mapping(value, "evidence")
    encoded = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > limit:
        raise ContractValidationError(
            f"evidence exceeds maximum of {limit} bytes"
        )
    return result


@dataclass(frozen=True)
class PropertyObligation(CanonicalContract):
    """A semantic obligation independent of any prover input language."""

    SCHEMA = PROPERTY_OBLIGATION_SCHEMA

    obligation_id: str
    property_kind: PropertyKind
    statement: str
    premise_ids: tuple[str, ...] = ()
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(
            self,
            "property_kind",
            _enum(self.property_kind, PropertyKind, "property_kind"),
        )
        object.__setattr__(self, "statement", _text(self.statement, "statement"))
        object.__setattr__(self, "premise_ids", _strings(self.premise_ids, "premise_ids"))
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def _payload(self) -> dict[str, Any]:
        return {
            "router_version": MULTI_PROVER_ROUTER_VERSION,
            "obligation_id": self.obligation_id,
            "property_kind": self.property_kind,
            "statement": self.statement,
            "premise_ids": self.premise_ids,
            "required_assurance": self.required_assurance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_code_obligation(
        cls,
        obligation: CodeProofObligation,
        *,
        property_kind: PropertyKind | str | None = None,
    ) -> "PropertyObligation":
        if not isinstance(obligation, CodeProofObligation):
            raise ContractValidationError("obligation must be a CodeProofObligation")
        kind = property_kind or classify_property_kind(
            obligation.invariant_class, obligation.metadata
        )
        return cls(
            obligation_id=obligation.obligation_id,
            property_kind=kind,
            statement=obligation.statement,
            premise_ids=obligation.premise_ids,
            required_assurance=obligation.required_assurance,
            metadata={
                **dict(obligation.metadata),
                "repository_id": obligation.repository_id,
                "repository_tree_id": obligation.repository_tree_id,
                "ast_scope_ids": list(obligation.ast_scope_ids),
                "source_contract": CodeProofObligation.SCHEMA,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropertyObligation":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("property obligation must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            obligation_id=payload.get("obligation_id", ""),
            property_kind=payload.get("property_kind", ""),
            statement=payload.get("statement", ""),
            premise_ids=tuple(payload.get("premise_ids") or ()),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.SOLVER_CHECKED
            ),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "obligation")
        return result


_PROPERTY_ALIASES: Mapping[str, PropertyKind] = {
    "finite": PropertyKind.FINITE_CONSTRAINT,
    "finite_constraint_satisfiability": PropertyKind.FINITE_CONSTRAINT,
    "smt": PropertyKind.FINITE_CONSTRAINT,
    "bounded_state_machine": PropertyKind.STATE_MACHINE,
    "tla": PropertyKind.STATE_MACHINE,
    "secpal": PropertyKind.AUTHORIZATION,
    "authorization_policy": PropertyKind.AUTHORIZATION,
    "protocol_reachability": PropertyKind.PROTOCOL,
    "protocol_trace_property": PropertyKind.PROTOCOL,
    "hyperltl": PropertyKind.HYPERPROPERTY,
    "noninterference": PropertyKind.HYPERPROPERTY,
    "mtl": PropertyKind.RUNTIME_TRACE,
    "trace": PropertyKind.RUNTIME_TRACE,
    "lean_kernel_check": PropertyKind.KERNEL_CHECK,
    "coq_kernel_check": PropertyKind.KERNEL_CHECK,
    "isabelle_kernel_check": PropertyKind.KERNEL_CHECK,
    "dcec": PropertyKind.TEMPORAL_DEONTIC,
    "tdfol": PropertyKind.TEMPORAL_DEONTIC,
    "planning": PropertyKind.TYPED_PLANNING,
    "fol": PropertyKind.FIRST_ORDER_THEOREM,
}


def classify_property_kind(
    invariant_class: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> PropertyKind:
    """Resolve an explicit property label without guessing from theorem text."""

    metadata = metadata or {}
    supplied = metadata.get("property_kind") or invariant_class
    value = _text(supplied, "property_kind")
    try:
        return PropertyKind(value)
    except ValueError:
        normalized = value.casefold().replace("-", "_").replace(" ", "_")
        if normalized in _PROPERTY_ALIASES:
            return _PROPERTY_ALIASES[normalized]
    raise ContractValidationError(f"unsupported property kind: {value}")


def to_canonical_property_kind(value: PropertyKind | str) -> str:
    """Project a supervisor property kind onto the datasets property vocabulary."""

    from .canonical_logic_adapter import map_property_kind_to_canonical

    try:
        return map_property_kind_to_canonical(value)
    except Exception as exc:
        raise ContractValidationError(
            f"cannot project property kind to canonical id: {value}"
        ) from exc


@dataclass(frozen=True)
class ProverLane:
    prover_id: str
    role: ProverRole
    stage: int = 0
    authority_capability: str = ""
    translation_path_id: str = ""
    requires_candidate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "prover_id", _text(self.prover_id, "prover_id"))
        object.__setattr__(self, "role", _enum(self.role, ProverRole, "role"))
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or self.stage < 0:
            raise ContractValidationError("stage must be a non-negative integer")
        object.__setattr__(
            self,
            "authority_capability",
            _text(self.authority_capability, "authority_capability", required=False),
        )
        object.__setattr__(
            self,
            "translation_path_id",
            _text(self.translation_path_id, "translation_path_id", required=False),
        )
        if not isinstance(self.requires_candidate, bool):
            raise ContractValidationError("requires_candidate must be boolean")
        if self.authority_capability and not self.role.authoritative:
            raise ContractValidationError(
                "only model-checker and kernel lanes may declare authority"
            )
        if self.prover_id.casefold().startswith("leanstral") and (
            self.role is not ProverRole.MODEL_ASSISTANT
            or self.authority_capability
        ):
            raise ContractValidationError(
                "Leanstral lanes must be model-assistant candidates"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prover_id": self.prover_id,
            "role": self.role.value,
            "stage": self.stage,
            "authority_capability": self.authority_capability,
            "translation_path_id": self.translation_path_id,
            "requires_candidate": self.requires_candidate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProverLane":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("prover lane must be an object")
        return cls(
            prover_id=payload.get("prover_id", ""),
            role=payload.get("role", ""),
            stage=payload.get("stage", 0),
            authority_capability=payload.get("authority_capability", ""),
            translation_path_id=payload.get("translation_path_id", ""),
            requires_candidate=payload.get("requires_candidate", False),
        )


@dataclass(frozen=True)
class PropertyPolicy:
    """Reviewed routing and fail-closed rules for one property family."""

    property_kind: PropertyKind
    lanes: tuple[ProverLane, ...]
    policy_id: str = ""
    timeout_seconds: float = DEFAULT_PORTFOLIO_TIMEOUT_SECONDS
    max_parallel: int = DEFAULT_MAX_PARALLEL_PROVERS
    require_capability_evidence: bool = False
    fail_on_disagreement: bool = True
    blocking_outcomes: tuple[AttemptOutcome, ...] = (
        AttemptOutcome.MALFORMED,
        AttemptOutcome.ERROR,
    )

    def __post_init__(self) -> None:
        kind = _enum(self.property_kind, PropertyKind, "property_kind")
        object.__setattr__(self, "property_kind", kind)
        lanes = tuple(self.lanes)
        if not lanes or any(not isinstance(item, ProverLane) for item in lanes):
            raise ContractValidationError("policy lanes must contain ProverLane values")
        ids = [item.prover_id for item in lanes]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("a policy cannot route a prover twice")
        if any(
            lane.role is ProverRole.MODEL_ASSISTANT and lane.requires_candidate
            for lane in lanes
        ):
            raise ContractValidationError(
                "model assistants produce candidates and cannot require one"
            )
        object.__setattr__(self, "lanes", lanes)
        policy_id = self.policy_id or f"property-portfolio:{kind.value}@1"
        object.__setattr__(self, "policy_id", _text(policy_id, "policy_id"))
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ContractValidationError("timeout_seconds must be positive")
        if (
            isinstance(self.max_parallel, bool)
            or not isinstance(self.max_parallel, int)
            or not 1 <= self.max_parallel <= 64
        ):
            raise ContractValidationError("max_parallel must be between 1 and 64")
        if not isinstance(self.require_capability_evidence, bool):
            raise ContractValidationError("require_capability_evidence must be boolean")
        if not isinstance(self.fail_on_disagreement, bool):
            raise ContractValidationError("fail_on_disagreement must be boolean")
        object.__setattr__(
            self,
            "blocking_outcomes",
            tuple(
                sorted(
                    {
                        _enum(item, AttemptOutcome, "blocking_outcomes")
                        for item in self.blocking_outcomes
                    },
                    key=lambda item: item.value,
                )
            ),
        )


def _authority(
    prover_id: str,
    capability: str,
    *,
    role: ProverRole = ProverRole.MODEL_CHECKER,
    stage: int = 0,
    requires_candidate: bool = False,
) -> ProverLane:
    return ProverLane(
        prover_id,
        role,
        stage,
        capability,
        requires_candidate=requires_candidate,
    )


_KERNEL_LANES = (
    _authority(
        "lean", "lean_kernel_check", role=ProverRole.KERNEL,
        stage=3, requires_candidate=True,
    ),
    _authority(
        "coq", "coq_kernel_check", role=ProverRole.KERNEL,
        stage=3, requires_candidate=True,
    ),
    _authority(
        "isabelle", "isabelle_kernel_check", role=ProverRole.KERNEL,
        stage=3, requires_candidate=True,
    ),
)


DEFAULT_PROPERTY_POLICIES: Mapping[PropertyKind, PropertyPolicy] = {
    PropertyKind.FINITE_CONSTRAINT: PropertyPolicy(
        PropertyKind.FINITE_CONSTRAINT,
        (
            _authority("z3", "finite_constraint_satisfiability"),
            _authority("cvc5", "finite_constraint_satisfiability"),
        ),
    ),
    PropertyKind.STATE_MACHINE: PropertyPolicy(
        PropertyKind.STATE_MACHINE,
        (
            _authority("tla_tlc", "bounded_state_machine"),
            _authority("apalache", "bounded_state_machine"),
        ),
    ),
    PropertyKind.AUTHORIZATION: PropertyPolicy(
        PropertyKind.AUTHORIZATION,
        (_authority("datalog_secpal", "authorization_policy"),),
    ),
    PropertyKind.PROTOCOL: PropertyPolicy(
        PropertyKind.PROTOCOL,
        (
            _authority("tamarin", "protocol_trace_property"),
            _authority("proverif", "protocol_reachability"),
        ),
    ),
    PropertyKind.HYPERPROPERTY: PropertyPolicy(
        PropertyKind.HYPERPROPERTY,
        (
            _authority(
                "hyperltl_autohyper_mchyper", "hyperproperty_model_check"
            ),
        ),
    ),
    PropertyKind.RUNTIME_TRACE: PropertyPolicy(
        PropertyKind.RUNTIME_TRACE,
        (_authority("runtime_mtl", "runtime_trace_monitoring"),),
    ),
    PropertyKind.KERNEL_CHECK: PropertyPolicy(
        PropertyKind.KERNEL_CHECK,
        tuple(
            ProverLane(
                lane.prover_id, lane.role, 0, lane.authority_capability,
                requires_candidate=False,
            )
            for lane in _KERNEL_LANES
        ),
    ),
    PropertyKind.TYPED_PLANNING: PropertyPolicy(
        PropertyKind.TYPED_PLANNING,
        (
            ProverLane("dcec", ProverRole.DOMAIN_REASONER, 0),
            ProverLane("tdfol", ProverRole.DOMAIN_REASONER, 0),
            ProverLane("hammer", ProverRole.ORCHESTRATOR, 1),
            ProverLane("vampire", ProverRole.CANDIDATE, 2),
            ProverLane("e", ProverRole.CANDIDATE, 2),
            ProverLane("z3", ProverRole.CANDIDATE, 2),
            *_KERNEL_LANES,
        ),
    ),
    PropertyKind.TEMPORAL_DEONTIC: PropertyPolicy(
        PropertyKind.TEMPORAL_DEONTIC,
        (
            ProverLane("dcec", ProverRole.DOMAIN_REASONER, 0),
            ProverLane("tdfol", ProverRole.DOMAIN_REASONER, 0),
            ProverLane("hammer", ProverRole.ORCHESTRATOR, 1),
            ProverLane("vampire", ProverRole.CANDIDATE, 2),
            ProverLane("e", ProverRole.CANDIDATE, 2),
            *_KERNEL_LANES,
        ),
    ),
    PropertyKind.FIRST_ORDER_THEOREM: PropertyPolicy(
        PropertyKind.FIRST_ORDER_THEOREM,
        (
            ProverLane("hammer", ProverRole.ORCHESTRATOR, 0),
            ProverLane("vampire", ProverRole.CANDIDATE, 1),
            ProverLane("e", ProverRole.CANDIDATE, 1),
            ProverLane("z3", ProverRole.CANDIDATE, 1),
            *tuple(
                ProverLane(
                    lane.prover_id, lane.role, 2, lane.authority_capability,
                    requires_candidate=True,
                )
                for lane in _KERNEL_LANES
            ),
        ),
    ),
}


@dataclass(frozen=True)
class PortfolioPlan(CanonicalContract):
    SCHEMA = PORTFOLIO_PLAN_SCHEMA

    obligation: PropertyObligation
    policy_id: str
    lanes: tuple[ProverLane, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.obligation, PropertyObligation):
            raise ContractValidationError("obligation must be a PropertyObligation")
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        if not self.lanes or any(
            not isinstance(lane, ProverLane) for lane in self.lanes
        ):
            raise ContractValidationError("portfolio plan must contain lanes")

    @property
    def plan_id(self) -> str:
        return self.content_id

    @property
    def prover_ids(self) -> tuple[str, ...]:
        return tuple(lane.prover_id for lane in self.lanes)

    def _payload(self) -> dict[str, Any]:
        return {
            "router_version": MULTI_PROVER_ROUTER_VERSION,
            "obligation": self.obligation,
            "policy_id": self.policy_id,
            "lanes": tuple(lane.to_dict() for lane in self.lanes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioPlan":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("portfolio plan must be an object")
        _schema(payload, cls.SCHEMA)
        obligation = payload.get("obligation")
        if not isinstance(obligation, Mapping):
            raise ContractValidationError("portfolio plan obligation must be an object")
        lanes = payload.get("lanes") or ()
        if isinstance(lanes, (str, bytes, bytearray)) or not isinstance(
            lanes, Sequence
        ):
            raise ContractValidationError("portfolio plan lanes must be a sequence")
        result = cls(
            obligation=PropertyObligation.from_dict(obligation),
            policy_id=payload.get("policy_id", ""),
            lanes=tuple(ProverLane.from_dict(item) for item in lanes),
        )
        _claimed_identity(payload, result.content_id, "portfolio plan")
        return result


@dataclass(frozen=True)
class AttemptRequest:
    plan_id: str
    obligation: PropertyObligation
    lane: ProverLane
    prior_attempts: tuple[Mapping[str, Any], ...]
    timeout_seconds: float

    @property
    def prover_id(self) -> str:
        return self.lane.prover_id


@dataclass(frozen=True)
class ProverOutput:
    """Strict normalized output returned by a portfolio runner."""

    outcome: AttemptOutcome
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    conclusive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", _enum(self.outcome, AttemptOutcome, "outcome"))
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))
        object.__setattr__(self, "evidence", _mapping(self.evidence, "evidence"))
        if not isinstance(self.conclusive, bool):
            raise ContractValidationError("conclusive must be boolean")
        if self.conclusive and self.outcome is not AttemptOutcome.COUNTEREXAMPLE:
            raise ContractValidationError(
                "only counterexample output may be conclusive"
            )
        if self.conclusive and not self.evidence:
            raise ContractValidationError(
                "a conclusive counterexample requires bounded evidence"
            )

    @classmethod
    def from_value(
        cls, value: Any, *, maximum_evidence_bytes: int
    ) -> "ProverOutput":
        if isinstance(value, cls):
            evidence = _strict_json_size(value.evidence, maximum_evidence_bytes)
            return cls(value.outcome, value.detail, evidence, value.conclusive)
        if not isinstance(value, Mapping):
            raise ContractValidationError("prover output must be an object")
        allowed = {"outcome", "status", "detail", "evidence", "conclusive"}
        unknown = set(value) - allowed
        if unknown:
            raise ContractValidationError(
                f"prover output has unsupported fields: {sorted(unknown)}"
            )
        raw_outcome = value.get("outcome", value.get("status"))
        if raw_outcome is None:
            raise ContractValidationError("prover output requires outcome")
        evidence = _strict_json_size(
            value.get("evidence") or {}, maximum_evidence_bytes
        )
        return cls(
            raw_outcome,
            value.get("detail", ""),
            evidence,
            value.get("conclusive", False),
        )


class PortfolioRunner(Protocol):
    def __call__(
        self, request: AttemptRequest, cancellation: threading.Event
    ) -> ProverOutput | Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class PortfolioAttempt(CanonicalContract):
    SCHEMA = PORTFOLIO_ATTEMPT_SCHEMA

    prover_id: str
    role: ProverRole
    stage: int
    reported_outcome: AttemptOutcome
    effective_outcome: AttemptOutcome
    authoritative: bool
    conclusive: bool
    detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    capability_receipt_id: str = ""
    conformance_gate_id: str = ""
    cancellation_requested: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "prover_id", _text(self.prover_id, "prover_id"))
        object.__setattr__(self, "role", _enum(self.role, ProverRole, "role"))
        object.__setattr__(
            self, "reported_outcome",
            _enum(self.reported_outcome, AttemptOutcome, "reported_outcome"),
        )
        object.__setattr__(
            self, "effective_outcome",
            _enum(self.effective_outcome, AttemptOutcome, "effective_outcome"),
        )
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))
        object.__setattr__(self, "evidence", _mapping(self.evidence, "evidence"))
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or self.stage < 0:
            raise ContractValidationError("stage must be a non-negative integer")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ContractValidationError("duration_ms must be non-negative")
        for name in ("authoritative", "conclusive", "cancellation_requested"):
            if not isinstance(getattr(self, name), bool):
                raise ContractValidationError(f"{name} must be boolean")
        if self.authoritative and not self.role.authoritative:
            raise ContractValidationError("candidate roles cannot be authoritative")
        if self.conclusive and self.effective_outcome is not AttemptOutcome.COUNTEREXAMPLE:
            raise ContractValidationError("conclusive attempt must be a counterexample")

    @property
    def attempt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "router_version": MULTI_PROVER_ROUTER_VERSION,
            "prover_id": self.prover_id,
            "role": self.role,
            "stage": self.stage,
            "reported_outcome": self.reported_outcome,
            "effective_outcome": self.effective_outcome,
            "authoritative": self.authoritative,
            "conclusive": self.conclusive,
            "detail": self.detail,
            "evidence": self.evidence,
            "duration_ms": self.duration_ms,
            "capability_receipt_id": self.capability_receipt_id,
            "conformance_gate_id": self.conformance_gate_id,
            "cancellation_requested": self.cancellation_requested,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioAttempt":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("portfolio attempt must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            prover_id=payload.get("prover_id", ""),
            role=payload.get("role", ""),
            stage=payload.get("stage", 0),
            reported_outcome=payload.get("reported_outcome", ""),
            effective_outcome=payload.get("effective_outcome", ""),
            authoritative=payload.get("authoritative", False),
            conclusive=payload.get("conclusive", False),
            detail=payload.get("detail", ""),
            evidence=payload.get("evidence") or {},
            duration_ms=payload.get("duration_ms", 0),
            capability_receipt_id=payload.get("capability_receipt_id", ""),
            conformance_gate_id=payload.get("conformance_gate_id", ""),
            cancellation_requested=payload.get("cancellation_requested", False),
        )
        _claimed_identity(payload, result.content_id, "portfolio attempt")
        return result


@dataclass(frozen=True)
class PortfolioResult(CanonicalContract):
    SCHEMA = PORTFOLIO_RESULT_SCHEMA

    plan: PortfolioPlan
    verdict: PortfolioVerdict
    assurance: AssuranceLevel
    attempts: tuple[PortfolioAttempt, ...]
    reason: str
    authority_attempt_ids: tuple[str, ...] = ()
    counterexample_attempt_id: str = ""
    disagreement: bool = False
    fail_closed: bool = True
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PortfolioPlan):
            raise ContractValidationError("plan must be a PortfolioPlan")
        object.__setattr__(self, "verdict", _enum(self.verdict, PortfolioVerdict, "verdict"))
        object.__setattr__(self, "assurance", _enum(self.assurance, AssuranceLevel, "assurance"))
        if any(not isinstance(item, PortfolioAttempt) for item in self.attempts):
            raise ContractValidationError(
                "attempts must contain PortfolioAttempt values"
            )
        if len(self.attempts) != len(self.plan.lanes):
            raise ContractValidationError("result must retain every planned attempt")
        if tuple(item.prover_id for item in self.attempts) != self.plan.prover_ids:
            raise ContractValidationError("attempt order must match plan order")
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(
            self, "authority_attempt_ids",
            _strings(self.authority_attempt_ids, "authority_attempt_ids"),
        )
        if self.counterexample_attempt_id:
            object.__setattr__(
                self, "counterexample_attempt_id",
                _text(self.counterexample_attempt_id, "counterexample_attempt_id"),
            )
        authoritative_attempt_ids = {
            item.attempt_id
            for item in self.attempts
            if item.authoritative
            and item.effective_outcome is AttemptOutcome.VERIFIED
        }
        if any(
            attempt_id not in authoritative_attempt_ids
            for attempt_id in self.authority_attempt_ids
        ):
            raise ContractValidationError(
                "authority_attempt_ids must reference verified authority attempts"
            )
        counterexample_attempt_ids = {
            item.attempt_id
            for item in self.attempts
            if item.conclusive
            and item.effective_outcome is AttemptOutcome.COUNTEREXAMPLE
        }
        if (
            self.counterexample_attempt_id
            and self.counterexample_attempt_id not in counterexample_attempt_ids
        ):
            raise ContractValidationError(
                "counterexample_attempt_id must reference conclusive evidence"
            )
        expected_disagreement = bool(
            authoritative_attempt_ids and counterexample_attempt_ids
        )
        if self.disagreement != expected_disagreement:
            raise ContractValidationError(
                "disagreement must be derived from retained authority attempts"
            )
        if self.verdict is PortfolioVerdict.PROVED and not self.authority_attempt_ids:
            raise ContractValidationError("proved result requires an authority attempt")
        if self.verdict is PortfolioVerdict.PROVED and not self.assurance.satisfies(
            AssuranceLevel.SOLVER_CHECKED
        ):
            raise ContractValidationError("proved result has insufficient assurance")
        if self.verdict is PortfolioVerdict.PROVED and not self.assurance.satisfies(
            self.plan.obligation.required_assurance
        ):
            raise ContractValidationError(
                "proved result does not meet the obligation's required assurance"
            )
        if (
            self.verdict is PortfolioVerdict.DISPROVED
            and not self.counterexample_attempt_id
        ):
            raise ContractValidationError(
                "disproved result requires a conclusive counterexample"
            )
        if (
            self.verdict is not PortfolioVerdict.PROVED
            and self.assurance is not AssuranceLevel.UNVERIFIED
        ):
            raise ContractValidationError("non-proved result must be unverified")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ContractValidationError("duration_ms must be non-negative")
        for name in ("disagreement", "fail_closed"):
            if not isinstance(getattr(self, name), bool):
                raise ContractValidationError(f"{name} must be boolean")
        if not self.fail_closed:
            raise ContractValidationError("portfolio results must fail closed")

    @property
    def result_id(self) -> str:
        return self.content_id

    @property
    def proved(self) -> bool:
        return self.verdict is PortfolioVerdict.PROVED

    def _payload(self) -> dict[str, Any]:
        return {
            "router_version": MULTI_PROVER_ROUTER_VERSION,
            "plan": self.plan,
            "verdict": self.verdict,
            "assurance": self.assurance,
            "attempts": self.attempts,
            "reason": self.reason,
            "authority_attempt_ids": self.authority_attempt_ids,
            "counterexample_attempt_id": self.counterexample_attempt_id,
            "disagreement": self.disagreement,
            "fail_closed": self.fail_closed,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioResult":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("portfolio result must be an object")
        _schema(payload, cls.SCHEMA)
        plan = payload.get("plan")
        attempts = payload.get("attempts") or ()
        if not isinstance(plan, Mapping):
            raise ContractValidationError("portfolio result plan must be an object")
        if isinstance(attempts, (str, bytes, bytearray)) or not isinstance(
            attempts, Sequence
        ):
            raise ContractValidationError("portfolio result attempts must be a sequence")
        result = cls(
            plan=PortfolioPlan.from_dict(plan),
            verdict=payload.get("verdict", ""),
            assurance=payload.get("assurance", AssuranceLevel.UNVERIFIED),
            attempts=tuple(PortfolioAttempt.from_dict(item) for item in attempts),
            reason=payload.get("reason", ""),
            authority_attempt_ids=tuple(payload.get("authority_attempt_ids") or ()),
            counterexample_attempt_id=payload.get("counterexample_attempt_id", ""),
            disagreement=payload.get("disagreement", False),
            fail_closed=payload.get("fail_closed", True),
            duration_ms=payload.get("duration_ms", 0),
        )
        _claimed_identity(payload, result.content_id, "portfolio result")
        return result


@dataclass(frozen=True)
class _LaneGate:
    runnable: bool
    authoritative: bool
    outcome: AttemptOutcome | None
    detail: str
    receipt_id: str = ""
    conformance_gate: ConformanceGateDecision | None = None


class MultiProverRouter:
    """Plan and execute reviewed multi-prover portfolios."""

    def __init__(
        self,
        policies: Mapping[PropertyKind | str, PropertyPolicy] | None = None,
        *,
        matrix: ProverMatrixSnapshot | None = None,
        conformance_reports: Mapping[str, ConformanceReport] | None = None,
        quarantine_registry: ProverQuarantineRegistry | None = None,
        maximum_evidence_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        source = policies or DEFAULT_PROPERTY_POLICIES
        normalized: dict[PropertyKind, PropertyPolicy] = {}
        for key, policy in source.items():
            kind = _enum(key, PropertyKind, "policy key")
            if not isinstance(policy, PropertyPolicy) or policy.property_kind is not kind:
                raise ContractValidationError("policy key and property_kind must agree")
            normalized[kind] = policy
        missing = set(PropertyKind) - set(normalized)
        if policies is None and missing:  # pragma: no cover - constant invariant
            raise RuntimeError(f"default policies missing {sorted(item.value for item in missing)}")
        self._policies = normalized
        if matrix is not None and not isinstance(matrix, ProverMatrixSnapshot):
            raise ContractValidationError("matrix must be a ProverMatrixSnapshot")
        self._matrix = matrix
        self._matrix_entries = (
            {entry.prover_id: entry for entry in matrix.entries} if matrix else {}
        )
        self._conformance_reports = dict(conformance_reports or {})
        if any(
            not isinstance(key, str) or not isinstance(value, ConformanceReport)
            for key, value in self._conformance_reports.items()
        ):
            raise ContractValidationError(
                "conformance_reports must map path ids to ConformanceReport values"
            )
        self._quarantine = quarantine_registry or ProverQuarantineRegistry()
        if (
            isinstance(maximum_evidence_bytes, bool)
            or not isinstance(maximum_evidence_bytes, int)
            or maximum_evidence_bytes < 1
        ):
            raise ContractValidationError("maximum_evidence_bytes must be positive")
        self._maximum_evidence_bytes = maximum_evidence_bytes
        self._monotonic = monotonic or time.monotonic

    @property
    def policies(self) -> Mapping[PropertyKind, PropertyPolicy]:
        return dict(self._policies)

    def policy_for(self, property_kind: PropertyKind | str) -> PropertyPolicy:
        kind = _enum(property_kind, PropertyKind, "property_kind")
        try:
            return self._policies[kind]
        except KeyError as exc:
            raise ContractValidationError(
                f"no portfolio policy for {kind.value}"
            ) from exc

    def _obligation(
        self,
        obligation: PropertyObligation | CodeProofObligation | Mapping[str, Any],
        property_kind: PropertyKind | str | None,
    ) -> PropertyObligation:
        if isinstance(obligation, PropertyObligation):
            if property_kind is not None and obligation.property_kind is not _enum(
                property_kind, PropertyKind, "property_kind"
            ):
                raise ContractValidationError(
                    "explicit property_kind conflicts with obligation"
                )
            return obligation
        if isinstance(obligation, CodeProofObligation):
            return PropertyObligation.from_code_obligation(
                obligation, property_kind=property_kind
            )
        if isinstance(obligation, Mapping):
            value = PropertyObligation.from_dict(obligation)
            return self._obligation(value, property_kind)
        raise ContractValidationError("unsupported obligation contract")

    def plan(
        self,
        obligation: PropertyObligation | CodeProofObligation | Mapping[str, Any],
        *,
        property_kind: PropertyKind | str | None = None,
    ) -> PortfolioPlan:
        normalized = self._obligation(obligation, property_kind)
        policy = self.policy_for(normalized.property_kind)
        lanes = policy.lanes
        if (
            normalized.required_assurance.satisfies(AssuranceLevel.KERNEL_VERIFIED)
            and not any(lane.role is ProverRole.KERNEL for lane in lanes)
        ):
            stage = max(lane.stage for lane in lanes) + 1
            lanes = (
                *lanes,
                *tuple(
                    ProverLane(
                        lane.prover_id,
                        lane.role,
                        stage,
                        lane.authority_capability,
                        requires_candidate=True,
                    )
                    for lane in _KERNEL_LANES
                ),
            )
        return PortfolioPlan(normalized, policy.policy_id, tuple(lanes))

    # "route" is the natural read-only API; execution is always explicit.
    route = plan
    route_obligation = plan

    def _lane_gate(self, lane: ProverLane, policy: PropertyPolicy) -> _LaneGate:
        entry: ProverMatrixEntry | None = self._matrix_entries.get(lane.prover_id)
        if self._matrix is not None or policy.require_capability_evidence:
            if entry is None:
                return _LaneGate(
                    False, False, AttemptOutcome.UNAVAILABLE,
                    "prover is absent from the executable capability matrix",
                )
            if not entry.discovered or not entry.smoke_tested:
                return _LaneGate(
                    False, False, AttemptOutcome.UNAVAILABLE,
                    f"prover capability is not smoke-tested: {entry.reason}",
                    entry.receipt.receipt_id if entry.receipt else "",
                )
            if not entry.translation_conformant:
                return _LaneGate(
                    False, False, AttemptOutcome.UNSUPPORTED,
                    "prover translation is not conformant",
                    entry.receipt.receipt_id if entry.receipt else "",
                )
            if lane.role is ProverRole.KERNEL and not entry.reconstruction_capable:
                return _LaneGate(
                    False, False, AttemptOutcome.UNSUPPORTED,
                    "kernel is not reconstruction-capable",
                    entry.receipt.receipt_id if entry.receipt else "",
                )

        path_id = lane.translation_path_id or lane.prover_id
        report = self._conformance_reports.get(path_id)
        rule = self._quarantine.rule(path_id)
        gate: ConformanceGateDecision | None = None
        if report is not None or rule is not None:
            gate = gate_prover_path(
                path_id,
                report,
                authoritative_for=(lane.authority_capability,)
                if lane.authority_capability else (),
                registry=self._quarantine,
            )
            if not gate.promotion_allowed:
                return _LaneGate(
                    False, False, AttemptOutcome.UNSUPPORTED,
                    "translation path is quarantined or not conformant",
                    entry.receipt.receipt_id if entry and entry.receipt else "",
                    gate,
                )

        authoritative = lane.role.authoritative
        # Model-produced proof sketches are useful inputs to later stages but
        # are never an authority, even if a capability matrix is malformed or
        # overclaims the provider.
        if lane.role is ProverRole.MODEL_ASSISTANT:
            authoritative = False
        if authoritative and entry is not None:
            authoritative = (
                lane.authority_capability in entry.authoritative_for
                and (
                    lane.role is not ProverRole.KERNEL
                    or entry.reconstruction_capable
                )
            )
        return _LaneGate(
            True,
            authoritative,
            None,
            "",
            entry.receipt.receipt_id if entry and entry.receipt else "",
            gate,
        )

    def _run_one(
        self,
        runner: PortfolioRunner,
        request: AttemptRequest,
        cancellation: threading.Event,
    ) -> tuple[ProverOutput, int]:
        started = self._monotonic()
        try:
            raw = runner(request, cancellation)
            output = ProverOutput.from_value(
                raw, maximum_evidence_bytes=self._maximum_evidence_bytes
            )
        except ContractValidationError as exc:
            output = ProverOutput(
                AttemptOutcome.MALFORMED,
                f"malformed prover output: {exc}",
                {"exception_type": type(exc).__name__},
            )
        except TimeoutError as exc:
            output = ProverOutput(
                AttemptOutcome.TIMEOUT,
                str(exc) or "prover attempt timed out",
            )
        except BaseException as exc:
            output = ProverOutput(
                AttemptOutcome.ERROR,
                f"prover runner {type(exc).__name__}: {exc}",
            )
        return output, max(0, round((self._monotonic() - started) * 1000))

    @staticmethod
    def _prior(attempts: Sequence[PortfolioAttempt]) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            {
                "attempt_id": item.attempt_id,
                "prover_id": item.prover_id,
                "role": item.role.value,
                "outcome": item.effective_outcome.value,
                "authoritative": item.authoritative,
                "evidence": dict(item.evidence),
            }
            for item in attempts
        )

    def _attempt_from_output(
        self,
        lane: ProverLane,
        gate: _LaneGate,
        output: ProverOutput,
        duration_ms: int,
        *,
        cancellation_requested: bool = False,
    ) -> PortfolioAttempt:
        reported = output.outcome
        effective = reported
        # A positive candidate requires reconstruction.  A concrete,
        # independently validated countermodel is asymmetric: once the runner
        # marks it conclusive it is sufficient to reject the universal claim,
        # even when produced by an ATP/SMT candidate lane.
        conclusive = output.conclusive
        if cancellation_requested and reported is not AttemptOutcome.COUNTEREXAMPLE:
            effective = AttemptOutcome.CANCELLED
            conclusive = False
        elif reported is AttemptOutcome.VERIFIED and not gate.authoritative:
            effective = AttemptOutcome.CANDIDATE
        elif reported is AttemptOutcome.COUNTEREXAMPLE and not output.conclusive:
            effective = AttemptOutcome.UNKNOWN
            conclusive = False
        return PortfolioAttempt(
            prover_id=lane.prover_id,
            role=lane.role,
            stage=lane.stage,
            reported_outcome=reported,
            effective_outcome=effective,
            authoritative=gate.authoritative,
            conclusive=conclusive,
            detail=output.detail,
            evidence=output.evidence,
            duration_ms=duration_ms,
            capability_receipt_id=gate.receipt_id,
            conformance_gate_id=(
                gate.conformance_gate.content_id if gate.conformance_gate else ""
            ),
            cancellation_requested=cancellation_requested,
        )

    def execute(
        self,
        obligation: PropertyObligation | CodeProofObligation | Mapping[str, Any],
        runner: PortfolioRunner,
        *,
        property_kind: PropertyKind | str | None = None,
    ) -> PortfolioResult:
        """Execute a bounded staged portfolio and derive a fail-closed verdict."""

        if not callable(runner):
            raise ContractValidationError("runner must be callable")
        plan = self.plan(obligation, property_kind=property_kind)
        policy = self.policy_for(plan.obligation.property_kind)
        started = self._monotonic()
        deadline = started + policy.timeout_seconds
        gates = {lane.prover_id: self._lane_gate(lane, policy) for lane in plan.lanes}
        records: dict[str, PortfolioAttempt] = {}
        global_cancel = threading.Event()
        cancellation: dict[str, threading.Event] = {
            lane.prover_id: threading.Event() for lane in plan.lanes
        }
        executor = ThreadPoolExecutor(
            max_workers=min(policy.max_parallel, len(plan.lanes)),
            thread_name_prefix="multi-prover",
        )
        stop_counterexample = ""
        try:
            for stage in sorted({lane.stage for lane in plan.lanes}):
                stage_lanes = [lane for lane in plan.lanes if lane.stage == stage]
                if global_cancel.is_set():
                    break
                previous = list(records.values())
                has_candidate = any(
                    item.effective_outcome
                    in (AttemptOutcome.CANDIDATE, AttemptOutcome.VERIFIED)
                    for item in previous
                )
                futures: dict[Future[tuple[ProverOutput, int]], ProverLane] = {}
                for lane in stage_lanes:
                    gate = gates[lane.prover_id]
                    if not gate.runnable:
                        outcome = gate.outcome or AttemptOutcome.UNSUPPORTED
                        records[lane.prover_id] = self._attempt_from_output(
                            lane, gate, ProverOutput(outcome, gate.detail), 0
                        )
                        continue
                    if lane.requires_candidate and not has_candidate:
                        records[lane.prover_id] = self._attempt_from_output(
                            lane,
                            gate,
                            ProverOutput(
                                AttemptOutcome.BLOCKED,
                                "reconstruction requires a successful solver candidate",
                            ),
                            0,
                        )
                        continue
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        records[lane.prover_id] = self._attempt_from_output(
                            lane, gate,
                            ProverOutput(AttemptOutcome.TIMEOUT, "portfolio deadline expired"),
                            0,
                        )
                        continue
                    request = AttemptRequest(
                        plan.plan_id,
                        plan.obligation,
                        lane,
                        self._prior(previous),
                        remaining,
                    )
                    future = executor.submit(
                        self._run_one,
                        runner,
                        request,
                        cancellation[lane.prover_id],
                    )
                    futures[future] = lane

                pending = set(futures)
                while pending and not global_cancel.is_set():
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        break
                    done, pending = wait(
                        pending, timeout=remaining, return_when=FIRST_COMPLETED
                    )
                    if not done:
                        break
                    found_counterexample = False
                    for future in done:
                        lane = futures[future]
                        output, duration_ms = future.result()
                        attempt = self._attempt_from_output(
                            lane, gates[lane.prover_id], output, duration_ms
                        )
                        records[lane.prover_id] = attempt
                        if attempt.conclusive:
                            stop_counterexample = attempt.attempt_id
                            found_counterexample = True
                    if found_counterexample:
                        global_cancel.set()
                        for other in pending:
                            other_lane = futures[other]
                            cancellation[other_lane.prover_id].set()
                            other.cancel()

                for future in pending:
                    lane = futures[future]
                    cancellation[lane.prover_id].set()
                    cancelled = future.cancel()
                    if cancelled or not future.done():
                        outcome = (
                            AttemptOutcome.CANCELLED
                            if global_cancel.is_set()
                            else AttemptOutcome.TIMEOUT
                        )
                        detail = (
                            "cancelled after conclusive counterexample"
                            if global_cancel.is_set()
                            else "portfolio deadline expired"
                        )
                        records[lane.prover_id] = self._attempt_from_output(
                            lane,
                            gates[lane.prover_id],
                            ProverOutput(outcome, detail),
                            0,
                            cancellation_requested=True,
                        )
                    else:
                        output, duration_ms = future.result()
                        records[lane.prover_id] = self._attempt_from_output(
                            lane,
                            gates[lane.prover_id],
                            output,
                            duration_ms,
                            cancellation_requested=global_cancel.is_set(),
                        )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # Every selected lane gets a durable terminal record, including stages
        # which were never started because a counterexample stopped the plan.
        for lane in plan.lanes:
            if lane.prover_id not in records:
                records[lane.prover_id] = self._attempt_from_output(
                    lane,
                    gates[lane.prover_id],
                    ProverOutput(
                        AttemptOutcome.CANCELLED,
                        "cancelled after conclusive counterexample",
                    ),
                    0,
                    cancellation_requested=True,
                )
        attempts = tuple(records[lane.prover_id] for lane in plan.lanes)
        return self._derive_result(
            plan,
            policy,
            attempts,
            stop_counterexample,
            max(0, round((self._monotonic() - started) * 1000)),
        )

    def _derive_result(
        self,
        plan: PortfolioPlan,
        policy: PropertyPolicy,
        attempts: tuple[PortfolioAttempt, ...],
        stopped_counterexample_id: str,
        duration_ms: int,
    ) -> PortfolioResult:
        positives = tuple(
            item
            for item in attempts
            if item.authoritative
            and item.effective_outcome is AttemptOutcome.VERIFIED
        )
        counterexamples = tuple(
            item
            for item in attempts
            if item.conclusive
            and item.effective_outcome is AttemptOutcome.COUNTEREXAMPLE
        )
        disagreement = bool(positives and counterexamples)
        blockers = tuple(
            item for item in attempts if item.effective_outcome in policy.blocking_outcomes
        )
        authority_ids = tuple(item.attempt_id for item in positives)
        counterexample_id = (
            counterexamples[0].attempt_id
            if counterexamples
            else stopped_counterexample_id
        )

        if disagreement and policy.fail_on_disagreement:
            verdict = PortfolioVerdict.INCONCLUSIVE
            assurance = AssuranceLevel.UNVERIFIED
            reason = "authoritative provers disagree; policy failed closed"
        elif counterexamples:
            verdict = PortfolioVerdict.DISPROVED
            assurance = AssuranceLevel.UNVERIFIED
            reason = "a prover produced a validated conclusive counterexample"
        elif blockers:
            verdict = (
                PortfolioVerdict.ERROR
                if any(
                    item.effective_outcome
                    in (AttemptOutcome.MALFORMED, AttemptOutcome.ERROR)
                    for item in blockers
                )
                else PortfolioVerdict.INCONCLUSIVE
            )
            assurance = AssuranceLevel.UNVERIFIED
            reason = "property policy failed closed on prover attempt outcomes"
        elif positives:
            available_assurance = (
                AssuranceLevel.KERNEL_VERIFIED
                if any(item.role is ProverRole.KERNEL for item in positives)
                else AssuranceLevel.SOLVER_CHECKED
            )
            if available_assurance.satisfies(plan.obligation.required_assurance):
                verdict = PortfolioVerdict.PROVED
                assurance = available_assurance
                reason = "configured verification authority accepted the obligation"
            else:
                verdict = PortfolioVerdict.INCONCLUSIVE
                assurance = AssuranceLevel.UNVERIFIED
                authority_ids = ()
                reason = (
                    "an authority accepted the obligation but did not meet "
                    "its required assurance"
                )
        elif (
            any(
                item.effective_outcome
                in (AttemptOutcome.UNSUPPORTED, AttemptOutcome.UNAVAILABLE)
                for item in attempts
            )
            and all(
                item.effective_outcome
                in (
                    AttemptOutcome.UNSUPPORTED,
                    AttemptOutcome.UNAVAILABLE,
                    AttemptOutcome.BLOCKED,
                    AttemptOutcome.CANCELLED,
                )
                for item in attempts
            )
        ):
            verdict = PortfolioVerdict.UNSUPPORTED
            assurance = AssuranceLevel.UNVERIFIED
            reason = "no selected prover supports the obligation with required evidence"
        else:
            verdict = PortfolioVerdict.INCONCLUSIVE
            assurance = AssuranceLevel.UNVERIFIED
            if any(
                item.effective_outcome is AttemptOutcome.CANDIDATE
                for item in attempts
            ):
                reason = (
                    "solver candidates were retained but no configured "
                    "reconstruction or model-checking authority accepted them"
                )
            else:
                reason = "portfolio produced no conclusive authoritative result"
        return PortfolioResult(
            plan=plan,
            verdict=verdict,
            assurance=assurance,
            attempts=attempts,
            reason=reason,
            authority_attempt_ids=authority_ids,
            counterexample_attempt_id=counterexample_id,
            disagreement=disagreement,
            fail_closed=True,
            duration_ms=duration_ms,
        )


PropertySpecificMultiProverRouter = MultiProverRouter


def route_obligation(
    obligation: PropertyObligation | CodeProofObligation | Mapping[str, Any],
    *,
    property_kind: PropertyKind | str | None = None,
    router: MultiProverRouter | None = None,
) -> PortfolioPlan:
    """Convenience read-only route selection entry point."""

    return (router or MultiProverRouter()).plan(
        obligation, property_kind=property_kind
    )


def execute_portfolio(
    obligation: PropertyObligation | CodeProofObligation | Mapping[str, Any],
    runner: PortfolioRunner,
    *,
    property_kind: PropertyKind | str | None = None,
    router: MultiProverRouter | None = None,
) -> PortfolioResult:
    """Convenience bounded execution entry point."""

    return (router or MultiProverRouter()).execute(
        obligation, runner, property_kind=property_kind
    )


# ---------------------------------------------------------------------------
# DCR-032 deterministic local-prover routing boundary
# ---------------------------------------------------------------------------

DCR032_PROVER_ROUTE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/dcr-032-prover-route@1"
DCR032_INTEGRATION_PENDING_REASON = "dcr031_obligation_integration_pending"


class DeterministicProverDisposition(str, Enum):
    """Closed outcomes for the DCR-032 router; none proves an obligation."""

    ROUTED = "routed"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    DEFER_CAPABILITY = "defer_capability"


@dataclass(frozen=True)
class DeterministicProverBackend:
    """Exact local module/toolchain binding for one permitted backend."""

    backend_id: str
    module_capability_id: str
    toolchain_id: str
    supported_fragments: tuple[str, ...]
    provider_kind: str = "local_offline"

    def __post_init__(self) -> None:
        for name in ("backend_id", "module_capability_id", "toolchain_id", "provider_kind"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        fragments = tuple(
            sorted({_text(item, "supported_fragments") for item in self.supported_fragments})
        )
        if not fragments:
            raise ContractValidationError("supported_fragments must not be empty")
        object.__setattr__(self, "supported_fragments", fragments)


@dataclass(frozen=True)
class DeterministicProverResources:
    """Bounded, replayable resource identity; the router executes nothing."""

    seed: int
    max_steps: int
    max_memory_bytes: int

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("seed", 0, 2**63 - 1),
            ("max_steps", 1, 10_000_000),
            ("max_memory_bytes", 1, 2**40),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ContractValidationError(f"{name} is outside deterministic bounds")

    def to_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "max_steps": self.max_steps,
            "max_memory_bytes": self.max_memory_bytes,
        }


@dataclass(frozen=True)
class DeterministicProverRoute(CanonicalContract):
    """A canonical, zero-execution router result.

    The route has no proof authority.  Only an exact, current DCR-031 compiled
    obligation can make it ``routed``; mappings and fixture-shaped data defer.
    """

    SCHEMA = DCR032_PROVER_ROUTE_SCHEMA

    obligation_id: str
    obligation_cid: str
    backend_id: str
    disposition: DeterministicProverDisposition
    reason_codes: tuple[str, ...]
    resources: DeterministicProverResources
    capability_receipt_ids: tuple[str, ...] = ()
    evidence_receipt_ids: tuple[str, ...] = ()
    integration_pending: bool = True
    proof_authority_call_count: int = 0
    model_call_count: int = 0
    external_execution_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id", required=False)
        )
        object.__setattr__(
            self, "obligation_cid", _text(self.obligation_cid, "obligation_cid", required=False)
        )
        object.__setattr__(
            self, "backend_id", _text(self.backend_id, "backend_id", required=False)
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DeterministicProverDisposition, "disposition"),
        )
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes, "reason_codes"))
        if not isinstance(self.resources, DeterministicProverResources):
            raise ContractValidationError("resources must be DeterministicProverResources")
        for name in (
            "integration_pending",
            "proof_authority_call_count",
            "model_call_count",
            "external_execution_count",
        ):
            value = getattr(self, name)
            if name == "integration_pending":
                if type(value) is not bool:
                    raise ContractValidationError("integration_pending must be boolean")
            elif type(value) is not int or value != 0:
                raise ContractValidationError(f"{name} must be exactly zero")
        object.__setattr__(
            self,
            "capability_receipt_ids",
            _strings(self.capability_receipt_ids, "capability_receipt_ids"),
        )
        object.__setattr__(
            self,
            "evidence_receipt_ids",
            _strings(self.evidence_receipt_ids, "evidence_receipt_ids"),
        )

    @property
    def proof_authorized(self) -> bool:
        return False

    @property
    def execution_permitted(self) -> bool:
        """Whether a reviewed local runner may be selected, never trusted."""

        return self.disposition is DeterministicProverDisposition.ROUTED

    def _payload(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "obligation_cid": self.obligation_cid,
            "backend_id": self.backend_id,
            "disposition": self.disposition,
            "reason_codes": self.reason_codes,
            "resources": self.resources.to_dict(),
            "capability_receipt_ids": self.capability_receipt_ids,
            "evidence_receipt_ids": self.evidence_receipt_ids,
            "integration_pending": self.integration_pending,
            "proof_authority_call_count": self.proof_authority_call_count,
            "model_call_count": self.model_call_count,
            "external_execution_count": self.external_execution_count,
        }


def _find_module(
    inventory: DeterministicRepairCapabilities, capability_id: str
) -> CapabilityReceipt | None:
    return next((item for item in inventory.modules if item.capability_id == capability_id), None)


def _find_toolchain(
    inventory: DeterministicRepairCapabilities, tool_id: str
) -> SolverReadiness | None:
    return next((item for item in inventory.toolchains if item.tool_id == tool_id), None)


def _has_evidence(
    receipts: Sequence[CapabilityEvidenceReceipt],
    *,
    evidence_id: str,
    evidence_kind: str,
    subject_id: str,
    subject_digest: str,
    subject_version: str,
) -> bool:
    return any(
        item.verifies(
            evidence_id=evidence_id,
            evidence_kind=evidence_kind,
            subject_id=subject_id,
            subject_digest=subject_digest,
            subject_version=subject_version,
        )
        for item in receipts
    )


def _module_reasons(
    module: CapabilityReceipt | None,
    receipts: Sequence[CapabilityEvidenceReceipt],
) -> list[str]:
    if module is None:
        return ["module_receipt_missing"]
    reasons: list[str] = []
    if (
        not module.available
        or module.network_mode is not NetworkMode.OFFLINE
        or not module.origin
        or not module.content_digest.startswith("module:sha256:")
        or module.distribution_version != module.expected_version
        or module.reason_codes
    ):
        reasons.append("module_receipt_unavailable_or_unqualified")
    if not module.initialized or not _has_evidence(
        receipts,
        evidence_id=module.capability_id,
        evidence_kind="initialization",
        subject_id=module.capability_id,
        subject_digest=module.content_digest,
        subject_version=module.distribution_version,
    ):
        reasons.append("module_initialization_receipt_missing")
    if not module.reconstructed or not _has_evidence(
        receipts,
        evidence_id=module.capability_id,
        evidence_kind="reconstruction",
        subject_id=module.capability_id,
        subject_digest=module.content_digest,
        subject_version=module.distribution_version,
    ):
        reasons.append("module_reconstruction_receipt_missing")
    if not module.self_test_passed or not _has_evidence(
        receipts,
        evidence_id=module.capability_id,
        evidence_kind="self_test",
        subject_id=module.capability_id,
        subject_digest=module.content_digest,
        subject_version=module.distribution_version,
    ):
        reasons.append("module_self_test_receipt_missing")
    return reasons


def _toolchain_reasons(
    toolchain: SolverReadiness | None,
    receipts: Sequence[CapabilityEvidenceReceipt],
) -> list[str]:
    if toolchain is None:
        return ["toolchain_receipt_missing"]
    reasons: list[str] = []
    if (
        not toolchain.available
        or toolchain.network_mode is not NetworkMode.OFFLINE
        or not toolchain.path
        or not toolchain.executable_digest.startswith("executable:sha256:")
        or toolchain.version != toolchain.expected_version
        or toolchain.reason_codes
    ):
        reasons.append("toolchain_receipt_unavailable_or_unqualified")
    if not toolchain.reconstructed or not _has_evidence(
        receipts,
        evidence_id=toolchain.reconstruction_id,
        evidence_kind="reconstruction",
        subject_id=toolchain.tool_id,
        subject_digest=toolchain.executable_digest,
        subject_version=toolchain.version,
    ):
        reasons.append("toolchain_reconstruction_receipt_missing")
    if not toolchain.self_test_passed or not _has_evidence(
        receipts,
        evidence_id=toolchain.self_test_id,
        evidence_kind="self_test",
        subject_id=toolchain.tool_id,
        subject_digest=toolchain.executable_digest,
        subject_version=toolchain.version,
    ):
        reasons.append("toolchain_self_test_receipt_missing")
    return reasons


def _valid_dcr031_obligation(
    obligation: object,
) -> tuple[McpGraphContractObligation | None, list[str]]:
    """Accept only the published typed DCR-031 object and exact roots."""

    if not isinstance(obligation, McpGraphContractObligation):
        return None, [DCR032_INTEGRATION_PENDING_REASON]
    payload = obligation.to_dict()
    roots = (
        obligation.graph_cid,
        obligation.candidate_cid,
        *obligation.input_cids,
    )
    if (
        payload.get("schema") != MCP_GRAPH_OBLIGATION_SCHEMA
        or payload.get("proof_status") != "not_proved"
        or payload.get("completion_authoritative") is not False
        or payload.get("mutation_authorized") is not False
        or obligation.disposition is not McpObligationDisposition.OPEN
        or obligation.backend is not McpObligationBackend.LOGIC_IR_CANDIDATE
        or obligation.fragment is McpObligationFragment.UNSUPPORTED
        or not obligation.input_cids
        or tuple(sorted(set(obligation.input_cids))) != obligation.input_cids
        or obligation.graph_cid not in obligation.input_cids
        or any(not isinstance(item, str) or not item for item in roots)
    ):
        return None, ["invalid_current_dcr031_obligation"]
    return obligation, []


def route_dcr032_local_prover(
    obligation: McpGraphContractObligation | Mapping[str, Any],
    *,
    backend: DeterministicProverBackend,
    capabilities: DeterministicRepairCapabilities,
    capability_evidence: Sequence[CapabilityEvidenceReceipt] = (),
    resources: DeterministicProverResources,
    reported_outcome: str = "",
    proof_reconstruction_receipt_id: str = "",
) -> DeterministicProverRoute:
    """Deterministically validate a local route without importing or running it.

    ``reported_outcome`` is only an untrusted diagnostic input.  In particular,
    ``sat`` without a bound reconstruction receipt is deferred and no result
    from this function carries proof authority.
    """

    normalized, input_reasons = _valid_dcr031_obligation(obligation)
    if normalized is None:
        return DeterministicProverRoute(
            obligation_id="",
            obligation_cid="",
            backend_id=backend.backend_id,
            disposition=DeterministicProverDisposition.DEFER_CAPABILITY,
            reason_codes=tuple(input_reasons),
            resources=resources,
        )

    reasons: list[str] = []
    if backend.provider_kind != "local_offline":
        reasons.append("remote_or_model_provider_forbidden")
    if normalized.fragment.value not in backend.supported_fragments:
        reasons.append("backend_does_not_support_logic_fragment")
    if capabilities.network_mode is not NetworkMode.OFFLINE:
        reasons.append("capability_inventory_not_offline")

    module = _find_module(capabilities, backend.module_capability_id)
    toolchain = _find_toolchain(capabilities, backend.toolchain_id)
    reasons.extend(_module_reasons(module, capability_evidence))
    reasons.extend(_toolchain_reasons(toolchain, capability_evidence))
    outcome = str(reported_outcome or "").strip().lower()
    if outcome in {"unknown", "error"}:
        reasons.append(f"reported_{outcome}_is_not_proof")
    elif outcome == "sat" and (
        not proof_reconstruction_receipt_id
        or toolchain is None
        or proof_reconstruction_receipt_id != toolchain.reconstruction_id
    ):
        reasons.append("sat_without_required_reconstruction")
    elif outcome not in {"", "sat", "unsat", "unknown", "error"}:
        reasons.append("unsupported_reported_outcome")

    unsupported = any(
        reason.startswith(("unsupported_", "open_logic", "remote_or_model", "backend_"))
        for reason in reasons
    )
    unavailable = any(
        reason.startswith(("module_", "toolchain_", "capability_inventory"))
        for reason in reasons
    )
    defer = any(
        reason.startswith(("reported_", "sat_without_")) for reason in reasons
    )
    if unsupported:
        disposition = DeterministicProverDisposition.UNSUPPORTED
    elif unavailable:
        disposition = DeterministicProverDisposition.UNAVAILABLE
    elif defer:
        disposition = DeterministicProverDisposition.DEFER_CAPABILITY
    else:
        disposition = DeterministicProverDisposition.ROUTED
    receipt_ids = tuple(
        item.receipt_id for item in (module, toolchain) if item is not None
    )
    evidence_ids = tuple(item.receipt_id for item in capability_evidence)
    return DeterministicProverRoute(
        obligation_id=normalized.obligation_id,
        obligation_cid=content_identity(normalized.to_dict()),
        backend_id=backend.backend_id,
        disposition=disposition,
        reason_codes=tuple(reasons),
        resources=resources,
        capability_receipt_ids=receipt_ids,
        evidence_receipt_ids=evidence_ids,
        integration_pending=False,
    )


__all__ = [
    "DEFAULT_MAX_EVIDENCE_BYTES",
    "DEFAULT_MAX_PARALLEL_PROVERS",
    "DEFAULT_PORTFOLIO_TIMEOUT_SECONDS",
    "DEFAULT_PROPERTY_POLICIES",
    "MULTI_PROVER_ROUTER_VERSION",
    "PORTFOLIO_ATTEMPT_SCHEMA",
    "PORTFOLIO_PLAN_SCHEMA",
    "PORTFOLIO_RESULT_SCHEMA",
    "PROPERTY_OBLIGATION_SCHEMA",
    "AttemptOutcome",
    "AttemptRequest",
    "DCR032_INTEGRATION_PENDING_REASON",
    "DCR032_PROVER_ROUTE_SCHEMA",
    "DeterministicProverBackend",
    "DeterministicProverDisposition",
    "DeterministicProverResources",
    "DeterministicProverRoute",
    "MultiProverRouter",
    "ObligationProperty",
    "PortfolioAttempt",
    "PortfolioPlan",
    "PortfolioResult",
    "PortfolioRunner",
    "PortfolioVerdict",
    "PropertyKind",
    "PropertyObligation",
    "PropertyPolicy",
    "PropertySpecificMultiProverRouter",
    "PropertyType",
    "ProverLane",
    "ProverOutput",
    "ProverRole",
    "RouteVerdict",
    "classify_property_kind",
    "execute_portfolio",
    "route_obligation",
    "route_dcr032_local_prover",
    "to_canonical_property_kind",
]

class AuthorityClass(str, Enum):
    """Closed authority lattice.  Candidates never author a proof."""

    CANDIDATE = "candidate"
    INDEPENDENT_CHECKER = "independent_checker"
    KERNEL = "kernel"

    @property
    def can_author_proof(self) -> bool:
        return self in (AuthorityClass.INDEPENDENT_CHECKER, AuthorityClass.KERNEL)


def authority_class_for_role(role: ProverRole | str) -> AuthorityClass:
    normalized = _enum(role, ProverRole, "role")
    if normalized is ProverRole.KERNEL:
        return AuthorityClass.KERNEL
    if normalized is ProverRole.MODEL_CHECKER:
        return AuthorityClass.INDEPENDENT_CHECKER
    return AuthorityClass.CANDIDATE


def obligation_scope_ids(obligation: PropertyObligation) -> tuple[str, ...]:
    """Deterministic scope identity retained with every solver counterexample."""

    metadata = obligation.metadata
    raw = (
        metadata.get("ast_scope_ids")
        or metadata.get("scope_ids")
        or metadata.get("changed_scope_set")
        or obligation.premise_ids
    )
    if isinstance(raw, str):
        raw = (raw,)
    if raw is None:
        raw = ()
    return _strings(raw, "ast_scope_ids")


def obligation_finite_bounds(obligation: PropertyObligation) -> dict[str, Any]:
    """Finite solver bounds bound to the obligation, possibly empty."""

    metadata = obligation.metadata
    raw = metadata.get("finite_bounds")
    if raw is None:
        raw = metadata.get("bounds")
    if raw is None:
        return {}
    return _mapping(raw, "finite_bounds")


def project_counterexample_evidence(
    obligation: PropertyObligation,
    evidence: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Retain scope/bounds and fail closed on obligation disagreement."""

    projected = _mapping(evidence, "evidence")
    supplied_scope = projected.get("ast_scope_ids")
    if supplied_scope is None:
        supplied_scope = projected.get("scope_ids")
    expected_scope = obligation_scope_ids(obligation)
    if supplied_scope in (None, "", ()):
        scope = expected_scope
    else:
        scope = _strings(supplied_scope, "ast_scope_ids")
        extra = set(scope) - set(expected_scope)
        if expected_scope and extra:
            return projected, "counterexample_scope_escapes_obligation"
    supplied_bounds = projected.get("finite_bounds")
    if supplied_bounds is None:
        supplied_bounds = projected.get("bounds")
    expected_bounds = obligation_finite_bounds(obligation)
    if supplied_bounds is None:
        bounds = expected_bounds
    else:
        bounds = _mapping(supplied_bounds, "finite_bounds")
        if expected_bounds and bounds != expected_bounds:
            return projected, "counterexample_bounds_disagree_with_obligation"
    projected["ast_scope_ids"] = list(scope)
    projected["finite_bounds"] = dict(bounds)
    return projected, ""


def _binding_version_failure(
    obligation: PropertyObligation, evidence: Mapping[str, Any]
) -> str:
    """Reject stale environment or version certificates on a claimed success."""

    metadata = obligation.metadata
    expected_lock = str(
        metadata.get("environment_lock_id") or metadata.get("toolchain_id") or ""
    ).strip()
    claimed_lock = str(
        evidence.get("environment_lock_id") or evidence.get("toolchain_id") or ""
    ).strip()
    if expected_lock and claimed_lock and expected_lock != claimed_lock:
        return "stale_environment_lock"
    expected_version = str(
        metadata.get("kernel_version") or metadata.get("itp_version") or ""
    ).strip()
    claimed_version = str(
        evidence.get("kernel_version") or evidence.get("itp_version") or ""
    ).strip()
    if expected_version and claimed_version and expected_version != claimed_version:
        return "stale_kernel_version"
    expected_statement = str(metadata.get("statement_digest") or "").strip()
    claimed_statement = str(evidence.get("statement_digest") or "").strip()
    if expected_statement and claimed_statement and expected_statement != claimed_statement:
        return "stale_statement_receipt"
    return ""


@dataclass(frozen=True)
class HammerTrace(CanonicalContract):
    """Non-authoritative datasets-hammer / ATP / SMT candidate trace."""

    SCHEMA = HAMMER_TRACE_SCHEMA

    request_id: str
    obligation_id: str
    prover_id: str
    role: ProverRole
    outcome: AttemptOutcome
    environment_lock_id: str = ""
    solver_versions: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    authoritative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "prover_id", _text(self.prover_id, "prover_id"))
        object.__setattr__(self, "role", _enum(self.role, ProverRole, "role"))
        object.__setattr__(self, "outcome", _enum(self.outcome, AttemptOutcome, "outcome"))
        object.__setattr__(
            self,
            "environment_lock_id",
            _text(self.environment_lock_id, "environment_lock_id", required=False),
        )
        object.__setattr__(
            self, "solver_versions", _mapping(self.solver_versions, "solver_versions")
        )
        object.__setattr__(
            self, "candidate_id", _text(self.candidate_id, "candidate_id", required=False)
        )
        object.__setattr__(self, "evidence", _mapping(self.evidence, "evidence"))
        if not isinstance(self.authoritative, bool):
            raise ContractValidationError("authoritative must be boolean")
        if self.authoritative or self.role.authoritative:
            raise ContractValidationError("hammer traces cannot carry proof authority")

    def _payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "obligation_id": self.obligation_id,
            "prover_id": self.prover_id,
            "role": self.role,
            "outcome": self.outcome,
            "environment_lock_id": self.environment_lock_id,
            "solver_versions": self.solver_versions,
            "candidate_id": self.candidate_id,
            "evidence": self.evidence,
            "authoritative": False,
            "authority_class": AuthorityClass.CANDIDATE.value,
        }


@dataclass(frozen=True)
class CounterexampleTrace(CanonicalContract):
    """Solver or checker counterexample that retains obligation scope/bounds."""

    SCHEMA = COUNTEREXAMPLE_TRACE_SCHEMA

    attempt_id: str
    obligation_id: str
    prover_id: str
    role: ProverRole
    ast_scope_ids: tuple[str, ...]
    finite_bounds: Mapping[str, Any]
    conclusive: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)
    environment_lock_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, "attempt_id"))
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "prover_id", _text(self.prover_id, "prover_id"))
        object.__setattr__(self, "role", _enum(self.role, ProverRole, "role"))
        object.__setattr__(self, "ast_scope_ids", _strings(self.ast_scope_ids, "ast_scope_ids"))
        object.__setattr__(self, "finite_bounds", _mapping(self.finite_bounds, "finite_bounds"))
        if not isinstance(self.conclusive, bool):
            raise ContractValidationError("conclusive must be boolean")
        object.__setattr__(self, "evidence", _mapping(self.evidence, "evidence"))
        object.__setattr__(
            self,
            "environment_lock_id",
            _text(self.environment_lock_id, "environment_lock_id", required=False),
        )
        if "ast_scope_ids" not in self.evidence or "finite_bounds" not in self.evidence:
            raise ContractValidationError("counterexample evidence must retain scope and bounds")

    def _payload(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "obligation_id": self.obligation_id,
            "prover_id": self.prover_id,
            "role": self.role,
            "ast_scope_ids": self.ast_scope_ids,
            "finite_bounds": self.finite_bounds,
            "conclusive": self.conclusive,
            "evidence": self.evidence,
            "environment_lock_id": self.environment_lock_id,
        }


@dataclass(frozen=True)
class CheckerTrace(CanonicalContract):
    """Independent kernel or reviewed model-checker acceptance trace."""

    SCHEMA = CHECKER_TRACE_SCHEMA

    attempt_id: str
    obligation_id: str
    prover_id: str
    role: ProverRole
    outcome: AttemptOutcome
    accepted: bool
    environment_lock_id: str = ""
    kernel_version: str = ""
    receipt_id: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, "attempt_id"))
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "prover_id", _text(self.prover_id, "prover_id"))
        object.__setattr__(self, "role", _enum(self.role, ProverRole, "role"))
        object.__setattr__(self, "outcome", _enum(self.outcome, AttemptOutcome, "outcome"))
        if not isinstance(self.accepted, bool):
            raise ContractValidationError("accepted must be boolean")
        object.__setattr__(
            self,
            "environment_lock_id",
            _text(self.environment_lock_id, "environment_lock_id", required=False),
        )
        object.__setattr__(
            self, "kernel_version", _text(self.kernel_version, "kernel_version", required=False)
        )
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, "receipt_id", required=False)
        )
        object.__setattr__(self, "evidence", _mapping(self.evidence, "evidence"))
        if self.accepted and not self.role.authoritative:
            raise ContractValidationError("only independent checkers may accept a proof")
        if self.accepted != (self.outcome is AttemptOutcome.VERIFIED and self.role.authoritative):
            raise ContractValidationError("checker acceptance and outcome disagree")

    def _payload(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "obligation_id": self.obligation_id,
            "prover_id": self.prover_id,
            "role": self.role,
            "outcome": self.outcome,
            "accepted": self.accepted,
            "environment_lock_id": self.environment_lock_id,
            "kernel_version": self.kernel_version,
            "receipt_id": self.receipt_id,
            "evidence": self.evidence,
            "authority_class": authority_class_for_role(self.role).value,
        }


@dataclass(frozen=True)
class AuthoritativeDisposition(CanonicalContract):
    """Fail-closed proof disposition derived from the authority lattice."""

    SCHEMA = AUTHORITATIVE_DISPOSITION_SCHEMA

    result_id: str
    obligation_id: str
    verdict: PortfolioVerdict
    assurance: AssuranceLevel
    reason: str
    authority_attempt_ids: tuple[str, ...] = ()
    counterexample_attempt_id: str = ""
    hammer_trace_ids: tuple[str, ...] = ()
    checker_trace_ids: tuple[str, ...] = ()
    counterexample_trace_ids: tuple[str, ...] = ()
    fail_closed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", _text(self.result_id, "result_id"))
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "verdict", _enum(self.verdict, PortfolioVerdict, "verdict"))
        object.__setattr__(self, "assurance", _enum(self.assurance, AssuranceLevel, "assurance"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(
            self,
            "authority_attempt_ids",
            _strings(self.authority_attempt_ids, "authority_attempt_ids"),
        )
        object.__setattr__(
            self,
            "counterexample_attempt_id",
            _text(self.counterexample_attempt_id, "counterexample_attempt_id", required=False),
        )
        object.__setattr__(
            self, "hammer_trace_ids", _strings(self.hammer_trace_ids, "hammer_trace_ids")
        )
        object.__setattr__(
            self, "checker_trace_ids", _strings(self.checker_trace_ids, "checker_trace_ids")
        )
        object.__setattr__(
            self,
            "counterexample_trace_ids",
            _strings(self.counterexample_trace_ids, "counterexample_trace_ids"),
        )
        if not isinstance(self.fail_closed, bool) or not self.fail_closed:
            raise ContractValidationError("authoritative dispositions must fail closed")
        if self.verdict is PortfolioVerdict.PROVED and not self.authority_attempt_ids:
            raise ContractValidationError("proved disposition requires independent checker authority")
        if self.verdict is PortfolioVerdict.PROVED and not self.assurance.satisfies(
            AssuranceLevel.SOLVER_CHECKED
        ):
            raise ContractValidationError("proved disposition has insufficient assurance")

    def _payload(self) -> dict[str, Any]:
        return {
            "authority_lattice": AUTHORITY_LATTICE_SCHEMA,
            "result_id": self.result_id,
            "obligation_id": self.obligation_id,
            "verdict": self.verdict,
            "assurance": self.assurance,
            "reason": self.reason,
            "authority_attempt_ids": self.authority_attempt_ids,
            "counterexample_attempt_id": self.counterexample_attempt_id,
            "hammer_trace_ids": self.hammer_trace_ids,
            "checker_trace_ids": self.checker_trace_ids,
            "counterexample_trace_ids": self.counterexample_trace_ids,
            "fail_closed": True,
        }


def project_portfolio_traces(
    result: PortfolioResult,
) -> tuple[tuple[HammerTrace, ...], tuple[CounterexampleTrace, ...], tuple[CheckerTrace, ...]]:
    """Project hammer, counterexample, and independent-checker traces from a result."""

    if not isinstance(result, PortfolioResult):
        raise ContractValidationError("result must be a PortfolioResult")
    obligation = result.plan.obligation
    hammer: list[HammerTrace] = []
    counterexamples: list[CounterexampleTrace] = []
    checkers: list[CheckerTrace] = []
    lock_id = str(obligation.metadata.get("environment_lock_id") or "")
    solver_versions = obligation.metadata.get("solver_versions") or {}
    if not isinstance(solver_versions, Mapping):
        solver_versions = {}
    for attempt in result.attempts:
        authority = authority_class_for_role(attempt.role)
        evidence = dict(attempt.evidence)
        if authority is AuthorityClass.CANDIDATE:
            hammer.append(
                HammerTrace(
                    request_id=result.plan.plan_id,
                    obligation_id=obligation.obligation_id,
                    prover_id=attempt.prover_id,
                    role=attempt.role,
                    outcome=attempt.effective_outcome,
                    environment_lock_id=str(evidence.get("environment_lock_id") or lock_id),
                    solver_versions=dict(solver_versions),
                    candidate_id=str(evidence.get("candidate_id") or ""),
                    evidence=evidence,
                )
            )
        if attempt.effective_outcome is AttemptOutcome.COUNTEREXAMPLE or attempt.conclusive:
            projected, failure = project_counterexample_evidence(obligation, evidence)
            if failure:
                continue
            counterexamples.append(
                CounterexampleTrace(
                    attempt_id=attempt.attempt_id,
                    obligation_id=obligation.obligation_id,
                    prover_id=attempt.prover_id,
                    role=attempt.role,
                    ast_scope_ids=tuple(projected.get("ast_scope_ids") or ()),
                    finite_bounds=projected.get("finite_bounds") or {},
                    conclusive=attempt.conclusive,
                    evidence=projected,
                    environment_lock_id=str(projected.get("environment_lock_id") or lock_id),
                )
            )
        if authority.can_author_proof:
            checkers.append(
                CheckerTrace(
                    attempt_id=attempt.attempt_id,
                    obligation_id=obligation.obligation_id,
                    prover_id=attempt.prover_id,
                    role=attempt.role,
                    outcome=attempt.effective_outcome,
                    accepted=(
                        attempt.authoritative
                        and attempt.effective_outcome is AttemptOutcome.VERIFIED
                    ),
                    environment_lock_id=str(evidence.get("environment_lock_id") or lock_id),
                    kernel_version=str(
                        evidence.get("kernel_version") or evidence.get("itp_version") or ""
                    ),
                    receipt_id=str(
                        evidence.get("kernel_receipt_id") or attempt.capability_receipt_id
                    ),
                    evidence=evidence,
                )
            )
    return tuple(hammer), tuple(counterexamples), tuple(checkers)


def derive_authoritative_disposition(result: PortfolioResult) -> AuthoritativeDisposition:
    """Bind hammer/counterexample/checker traces to one fail-closed disposition."""

    hammer, counterexamples, checkers = project_portfolio_traces(result)
    return AuthoritativeDisposition(
        result_id=result.result_id,
        obligation_id=result.plan.obligation.obligation_id,
        verdict=result.verdict,
        assurance=result.assurance,
        reason=result.reason,
        authority_attempt_ids=result.authority_attempt_ids,
        counterexample_attempt_id=result.counterexample_attempt_id,
        hammer_trace_ids=tuple(item.content_id for item in hammer),
        checker_trace_ids=tuple(item.content_id for item in checkers),
        counterexample_trace_ids=tuple(item.content_id for item in counterexamples),
    )

