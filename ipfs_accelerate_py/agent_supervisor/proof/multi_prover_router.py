"""Property-specific, fail-closed multi-prover portfolio routing.

This module is deliberately an orchestration and trust-boundary module.  It
does not import or execute optional theorem provers itself.  Callers provide a
bounded :class:`PortfolioRunner`; the router selects the reviewed portfolio,
gates it against capability/conformance evidence when supplied, retains an
attempt for every selected lane, and derives the only authoritative verdict.

Interface: ``ProverPortfolio@1`` (with ``SolverReadiness@1`` capability
evidence).  :class:`ProverCapabilityAdmission` admits only DCEC, TDFOL, SMT,
theorem, and structural backends where declared fragments and self-tests
match.  General LLM and remote nondeterministic providers are not
representable.  Required missing, unsupported, or error backends fail closed.

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

from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
    _canonical_value,
)
from .prover_conformance import (
    ConformanceGateDecision,
    ConformanceReport,
    ProverQuarantineRegistry,
    gate_prover_path,
)
from .prover_matrix_registry import ProverMatrixEntry, ProverMatrixSnapshot
from .solver_readiness import (
    SolverBackendFamily,
    SolverBackendReadiness,
    SolverReadinessReport,
    SolverReadinessStatus,
)


MULTI_PROVER_ROUTER_VERSION = 1
PROVER_PORTFOLIO_INTERFACE = "ProverPortfolio@1"
SOLVER_READINESS_INTERFACE = "SolverReadiness@1"
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
PROVER_CAPABILITY_ADMISSION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prover-capability-admission@1"
)
DEFAULT_PORTFOLIO_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_PARALLEL_PROVERS = 8
DEFAULT_MAX_EVIDENCE_BYTES = 256 * 1024

# Closed deterministic backend families admitted by DCR-032.  General LLM and
# remote nondeterministic providers are intentionally absent from this set.
class DeterministicBackendKind(str, Enum):
    """Capability-qualified deterministic prover families only."""

    DCEC = "dcec"
    TDFOL = "tdfol"
    SMT = "smt"
    THEOREM = "theorem"
    STRUCTURAL = "structural"


class CapabilityAdmissionStatus(str, Enum):
    """Terminal admission decision for one backend/lane."""

    ADMITTED = "admitted"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    UNREPRESENTABLE = "unrepresentable"


# Tokens that can never name a deterministic portfolio backend.  Matching any
# of these means the provider is not representable in ProverCapabilityAdmission.
_UNREPRESENTABLE_BACKEND_TOKENS: frozenset[str] = frozenset(
    {
        "llm",
        "llm_inference",
        "llm-proof-draft",
        "leanstral",
        "openai",
        "anthropic",
        "litellm",
        "langchain",
        "llama_cpp",
        "transformers",
        "groq",
        "cohere",
        "vertexai",
        "gpt",
        "grok",
        "codex",
        "chat",
        "model",
        "model_assistant",
        "remote",
        "remote_service",
        "remote_prover",
        "nondeterministic",
        "cloud_llm",
        "hosted_model",
    }
)

_PROVER_ID_TO_BACKEND_KIND: Mapping[str, DeterministicBackendKind] = {
    "dcec": DeterministicBackendKind.DCEC,
    "cec": DeterministicBackendKind.DCEC,
    "tdfol": DeterministicBackendKind.TDFOL,
    "z3": DeterministicBackendKind.SMT,
    "cvc5": DeterministicBackendKind.SMT,
    "smt": DeterministicBackendKind.SMT,
    "hammer": DeterministicBackendKind.THEOREM,
    "vampire": DeterministicBackendKind.THEOREM,
    "e": DeterministicBackendKind.THEOREM,
    "lean": DeterministicBackendKind.THEOREM,
    "coq": DeterministicBackendKind.THEOREM,
    "isabelle": DeterministicBackendKind.THEOREM,
    "tla_tlc": DeterministicBackendKind.STRUCTURAL,
    "apalache": DeterministicBackendKind.STRUCTURAL,
    "datalog_secpal": DeterministicBackendKind.STRUCTURAL,
    "tamarin": DeterministicBackendKind.STRUCTURAL,
    "proverif": DeterministicBackendKind.STRUCTURAL,
    "hyperltl_autohyper_mchyper": DeterministicBackendKind.STRUCTURAL,
    "runtime_mtl": DeterministicBackendKind.STRUCTURAL,
}

_SOLVER_FAMILY_TO_BACKEND_KIND: Mapping[SolverBackendFamily, DeterministicBackendKind] = {
    SolverBackendFamily.DCEC: DeterministicBackendKind.DCEC,
    SolverBackendFamily.CEC: DeterministicBackendKind.DCEC,
    SolverBackendFamily.TDFOL: DeterministicBackendKind.TDFOL,
    SolverBackendFamily.Z3: DeterministicBackendKind.SMT,
    SolverBackendFamily.HAMMER: DeterministicBackendKind.THEOREM,
}

_BACKEND_KIND_TO_SOLVER_FAMILIES: Mapping[
    DeterministicBackendKind, tuple[SolverBackendFamily, ...]
] = {
    DeterministicBackendKind.DCEC: (
        SolverBackendFamily.DCEC,
        SolverBackendFamily.CEC,
    ),
    DeterministicBackendKind.TDFOL: (SolverBackendFamily.TDFOL,),
    DeterministicBackendKind.SMT: (SolverBackendFamily.Z3,),
    DeterministicBackendKind.THEOREM: (SolverBackendFamily.HAMMER,),
    DeterministicBackendKind.STRUCTURAL: (),
}

_LOGIC_FRAGMENT_TO_BACKEND_KINDS: Mapping[str, frozenset[DeterministicBackendKind]] = {
    "dcec": frozenset({DeterministicBackendKind.DCEC, DeterministicBackendKind.THEOREM}),
    "cec": frozenset({DeterministicBackendKind.DCEC, DeterministicBackendKind.THEOREM}),
    "tdfol": frozenset({DeterministicBackendKind.TDFOL, DeterministicBackendKind.THEOREM}),
    "smt": frozenset({DeterministicBackendKind.SMT, DeterministicBackendKind.THEOREM}),
    "fol": frozenset({DeterministicBackendKind.THEOREM, DeterministicBackendKind.SMT}),
    "theorem": frozenset({DeterministicBackendKind.THEOREM}),
    "graph": frozenset({DeterministicBackendKind.STRUCTURAL}),
    "schema": frozenset({DeterministicBackendKind.STRUCTURAL}),
    "relation": frozenset({DeterministicBackendKind.STRUCTURAL}),
    "deontic": frozenset(
        {
            DeterministicBackendKind.DCEC,
            DeterministicBackendKind.TDFOL,
            DeterministicBackendKind.THEOREM,
        }
    ),
    "temporal": frozenset(
        {
            DeterministicBackendKind.DCEC,
            DeterministicBackendKind.TDFOL,
            DeterministicBackendKind.STRUCTURAL,
            DeterministicBackendKind.THEOREM,
        }
    ),
    "structural": frozenset({DeterministicBackendKind.STRUCTURAL}),
}


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
        object.__setattr__(self, "property_kind", _enum(self.property_kind, PropertyKind, "property_kind"))
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


def _normalize_backend_token(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_").strip(".:/")


def backend_is_representable(prover_id: str) -> bool:
    """Return whether *prover_id* can name a deterministic portfolio backend.

    General LLM and remote nondeterministic providers are never representable.
    """

    try:
        text = _text(prover_id, "prover_id")
    except ContractValidationError:
        return False
    tokens = {
        _normalize_backend_token(part)
        for part in text.replace(":", "/").replace(".", "/").split("/")
        if part
    }
    tokens.add(_normalize_backend_token(text))
    if tokens & _UNREPRESENTABLE_BACKEND_TOKENS:
        return False
    if any(
        token.startswith(prefix) or token.endswith(f"_{prefix}") or f"_{prefix}_" in token
        for token in tokens
        for prefix in ("llm", "leanstral", "remote", "gpt", "openai", "anthropic")
    ):
        return False
    return True


def classify_deterministic_backend(
    prover_id: str,
) -> DeterministicBackendKind:
    """Map a prover id onto the closed deterministic backend vocabulary.

    Raises :class:`ContractValidationError` when the id is unrepresentable or
    outside the reviewed deterministic families.
    """

    text = _text(prover_id, "prover_id")
    if not backend_is_representable(text):
        raise ContractValidationError(
            f"backend {text!r} is not representable as a deterministic prover "
            "(general LLM and remote nondeterministic providers are excluded)"
        )
    key = _normalize_backend_token(text)
    kind = _PROVER_ID_TO_BACKEND_KIND.get(key)
    if kind is None:
        # Allow dotted/provider-qualified ids such as "datasets:z3".
        for suffix, mapped in _PROVER_ID_TO_BACKEND_KIND.items():
            if key == suffix or key.endswith(f"_{suffix}") or key.endswith(f":{suffix}"):
                kind = mapped
                break
    if kind is None:
        raise ContractValidationError(
            f"backend {text!r} is outside the deterministic portfolio vocabulary"
        )
    return kind


@dataclass(frozen=True)
class CapabilityAdmissionDecision:
    """Fail-closed admission verdict for one backend or lane."""

    prover_id: str
    status: CapabilityAdmissionStatus
    backend_kind: DeterministicBackendKind | None
    detail: str
    capability_receipt_id: str = ""
    readiness_identity: str = ""
    self_test_passed: bool = False
    fragment_matched: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "prover_id", _text(self.prover_id, "prover_id"))
        object.__setattr__(
            self, "status", _enum(self.status, CapabilityAdmissionStatus, "status")
        )
        if self.backend_kind is not None:
            object.__setattr__(
                self,
                "backend_kind",
                _enum(self.backend_kind, DeterministicBackendKind, "backend_kind"),
            )
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))
        object.__setattr__(
            self,
            "capability_receipt_id",
            _text(self.capability_receipt_id, "capability_receipt_id", required=False),
        )
        object.__setattr__(
            self,
            "readiness_identity",
            _text(self.readiness_identity, "readiness_identity", required=False),
        )
        for name in ("self_test_passed", "fragment_matched"):
            if not isinstance(getattr(self, name), bool):
                raise ContractValidationError(f"{name} must be boolean")
        if (
            self.status is CapabilityAdmissionStatus.ADMITTED
            and self.backend_kind is None
        ):
            raise ContractValidationError("admitted decision requires a backend kind")
        if (
            self.status is CapabilityAdmissionStatus.UNREPRESENTABLE
            and self.backend_kind is not None
        ):
            raise ContractValidationError(
                "unrepresentable backends cannot carry a deterministic kind"
            )

    @property
    def admitted(self) -> bool:
        return self.status is CapabilityAdmissionStatus.ADMITTED

    @property
    def attempt_outcome(self) -> AttemptOutcome:
        if self.status is CapabilityAdmissionStatus.ADMITTED:
            raise ContractValidationError("admitted backends have no gate outcome")
        if self.status is CapabilityAdmissionStatus.MISSING:
            return AttemptOutcome.UNAVAILABLE
        if self.status is CapabilityAdmissionStatus.ERROR:
            return AttemptOutcome.ERROR
        if self.status is CapabilityAdmissionStatus.UNREPRESENTABLE:
            return AttemptOutcome.UNSUPPORTED
        return AttemptOutcome.UNSUPPORTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVER_CAPABILITY_ADMISSION_SCHEMA,
            "prover_id": self.prover_id,
            "status": self.status.value,
            "backend_kind": (
                self.backend_kind.value if self.backend_kind is not None else None
            ),
            "detail": self.detail,
            "capability_receipt_id": self.capability_receipt_id,
            "readiness_identity": self.readiness_identity,
            "self_test_passed": self.self_test_passed,
            "fragment_matched": self.fragment_matched,
            "admitted": self.admitted,
        }


@dataclass(frozen=True)
class RequiredBackendAdmission:
    """Aggregate fail-closed result for one required backend identity."""

    backend: str
    decision: CapabilityAdmissionDecision

    @property
    def admitted(self) -> bool:
        return self.decision.admitted


class ProverCapabilityAdmission:
    """Admit only capability-qualified deterministic provers into a portfolio.

    Interface: ``ProverPortfolio@1`` with readiness evidence from
    ``SolverReadiness@1``.

    Selection is restricted to DCEC, TDFOL, SMT, theorem, and structural
    backends.  General LLM and remote nondeterministic providers are not
    representable: classification raises and admission returns
    ``UNREPRESENTABLE`` rather than a runnable lane.  Required backends that
    are missing, unsupported, or in error fail closed.
    """

    INTERFACE = PROVER_PORTFOLIO_INTERFACE
    READINESS_INTERFACE = SOLVER_READINESS_INTERFACE

    def __init__(
        self,
        *,
        matrix: ProverMatrixSnapshot | None = None,
        readiness: SolverReadinessReport | None = None,
        require_self_test: bool = True,
        require_capability_evidence: bool = True,
        allowed_kinds: Iterable[DeterministicBackendKind | str] | None = None,
        required_backends: Iterable[str] | None = None,
        logic_fragment: str = "",
    ) -> None:
        if matrix is not None and not isinstance(matrix, ProverMatrixSnapshot):
            raise ContractValidationError("matrix must be a ProverMatrixSnapshot")
        if readiness is not None and not isinstance(readiness, SolverReadinessReport):
            raise ContractValidationError(
                "readiness must be a SolverReadinessReport (SolverReadiness@1)"
            )
        if not isinstance(require_self_test, bool):
            raise ContractValidationError("require_self_test must be boolean")
        if not isinstance(require_capability_evidence, bool):
            raise ContractValidationError(
                "require_capability_evidence must be boolean"
            )
        kinds: set[DeterministicBackendKind] = set()
        if allowed_kinds is None:
            kinds = set(DeterministicBackendKind)
        else:
            for item in allowed_kinds:
                kinds.add(_enum(item, DeterministicBackendKind, "allowed_kinds"))
        if not kinds:
            raise ContractValidationError("allowed_kinds must not be empty")
        self._matrix = matrix
        self._matrix_entries = (
            {entry.prover_id: entry for entry in matrix.entries} if matrix else {}
        )
        self._readiness = readiness
        self._require_self_test = require_self_test
        self._require_capability_evidence = require_capability_evidence
        self._allowed_kinds = frozenset(kinds)
        self._required_backends = _strings(required_backends, "required_backends")
        self._logic_fragment = _text(
            logic_fragment, "logic_fragment", required=False
        ).casefold()

    @property
    def matrix(self) -> ProverMatrixSnapshot | None:
        return self._matrix

    @property
    def readiness(self) -> SolverReadinessReport | None:
        return self._readiness

    @property
    def required_backends(self) -> tuple[str, ...]:
        return self._required_backends

    @property
    def allowed_kinds(self) -> frozenset[DeterministicBackendKind]:
        return self._allowed_kinds

    @property
    def logic_fragment(self) -> str:
        return self._logic_fragment

    def classify(self, prover_id: str) -> DeterministicBackendKind:
        """Classify *prover_id* or raise when it is unrepresentable."""

        return classify_deterministic_backend(prover_id)

    def is_representable(self, prover_id: str) -> bool:
        return backend_is_representable(prover_id)

    def _fragment_allows(self, kind: DeterministicBackendKind) -> bool:
        if not self._logic_fragment:
            return True
        allowed = _LOGIC_FRAGMENT_TO_BACKEND_KINDS.get(self._logic_fragment)
        if allowed is None:
            # Unknown fragment never widens the portfolio.
            return False
        return kind in allowed

    def _readiness_for(
        self, prover_id: str, kind: DeterministicBackendKind
    ) -> SolverBackendReadiness | None:
        if self._readiness is None:
            return None
        # Prefer an exact provider_id match when present.
        for item in self._readiness.backends:
            if item.provider_id == prover_id:
                return item
        # Exact family-name match (for example prover_id "z3" -> family Z3).
        try:
            family = SolverBackendFamily(prover_id)
        except ValueError:
            family = None
        if family is not None:
            try:
                return self._readiness.backend(family)
            except KeyError:
                return None
        # Kind→family fallback only when the readiness row names this prover.
        for family in _BACKEND_KIND_TO_SOLVER_FAMILIES.get(kind, ()):
            try:
                item = self._readiness.backend(family)
            except KeyError:
                continue
            if item.provider_id == prover_id or item.family.value == prover_id:
                return item
        return None

    def admit_backend(
        self,
        prover_id: str,
        *,
        role: ProverRole | str | None = None,
        authority_capability: str = "",
        required: bool = False,
    ) -> CapabilityAdmissionDecision:
        """Admit one backend id under the deterministic portfolio policy."""

        text = _text(prover_id, "prover_id")
        if not backend_is_representable(text):
            return CapabilityAdmissionDecision(
                text,
                CapabilityAdmissionStatus.UNREPRESENTABLE,
                None,
                (
                    "general LLM and remote nondeterministic providers are not "
                    "representable as deterministic portfolio backends"
                ),
            )
        try:
            kind = classify_deterministic_backend(text)
        except ContractValidationError as exc:
            status = (
                CapabilityAdmissionStatus.UNREPRESENTABLE
                if "not representable" in str(exc)
                else CapabilityAdmissionStatus.UNSUPPORTED
            )
            return CapabilityAdmissionDecision(
                text,
                status,
                None,
                str(exc),
            )
        if kind not in self._allowed_kinds:
            return CapabilityAdmissionDecision(
                text,
                CapabilityAdmissionStatus.UNSUPPORTED,
                kind,
                f"backend kind {kind.value} is not allowed by admission policy",
                fragment_matched=self._fragment_allows(kind),
            )
        fragment_ok = self._fragment_allows(kind)
        if not fragment_ok:
            return CapabilityAdmissionDecision(
                text,
                CapabilityAdmissionStatus.UNSUPPORTED,
                kind,
                (
                    f"backend kind {kind.value} does not match declared logic "
                    f"fragment {self._logic_fragment!r}"
                ),
                fragment_matched=False,
            )

        normalized_role = (
            None if role is None else _enum(role, ProverRole, "role")
        )
        if normalized_role is ProverRole.MODEL_ASSISTANT:
            return CapabilityAdmissionDecision(
                text,
                CapabilityAdmissionStatus.UNREPRESENTABLE,
                None,
                "model-assistant roles are not deterministic portfolio backends",
                fragment_matched=True,
            )

        readiness = self._readiness_for(text, kind)
        readiness_identity = readiness.readiness_identity if readiness else ""
        readiness_ok = False
        if readiness is not None:
            reason_code = str(getattr(readiness, "reason_code", "") or "").casefold()
            if reason_code in {
                "error",
                "probe_error",
                "backend_error",
                "readiness_error",
            }:
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.ERROR,
                    kind,
                    readiness.reason or "solver readiness reports backend error",
                    readiness_identity=readiness_identity,
                    self_test_passed=readiness.self_test_passed,
                    fragment_matched=True,
                )
            if readiness.status is SolverReadinessStatus.UNSUPPORTED:
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.MISSING
                    if required
                    else CapabilityAdmissionStatus.UNSUPPORTED,
                    kind,
                    readiness.reason or "solver readiness reports unsupported",
                    readiness_identity=readiness_identity,
                    self_test_passed=readiness.self_test_passed,
                    fragment_matched=True,
                )
            if not readiness.supported:
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.UNSUPPORTED,
                    kind,
                    readiness.reason or "solver readiness is not supported",
                    readiness_identity=readiness_identity,
                    self_test_passed=readiness.self_test_passed,
                    fragment_matched=True,
                )
            if self._require_self_test and not readiness.self_test_passed:
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.UNSUPPORTED,
                    kind,
                    "backend self-test has not passed",
                    readiness_identity=readiness_identity,
                    self_test_passed=False,
                    fragment_matched=True,
                )
            readiness_ok = True

        entry = self._matrix_entries.get(text)
        if self._matrix is not None:
            if entry is None:
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.MISSING,
                    kind,
                    "prover is absent from the executable capability matrix",
                    readiness_identity=readiness_identity,
                    self_test_passed=bool(readiness and readiness.self_test_passed),
                    fragment_matched=True,
                )
            receipt_id = entry.receipt.receipt_id if entry.receipt else ""
            if not entry.discovered:
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.MISSING,
                    kind,
                    entry.reason or "prover was not discovered",
                    capability_receipt_id=receipt_id,
                    readiness_identity=readiness_identity,
                    fragment_matched=True,
                )
            if not entry.smoke_tested:
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.UNSUPPORTED,
                    kind,
                    entry.reason or "prover capability is not smoke-tested",
                    capability_receipt_id=receipt_id,
                    readiness_identity=readiness_identity,
                    fragment_matched=True,
                )
            if not entry.translation_conformant:
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.UNSUPPORTED,
                    kind,
                    "prover translation is not conformant",
                    capability_receipt_id=receipt_id,
                    readiness_identity=readiness_identity,
                    self_test_passed=True,
                    fragment_matched=True,
                )
            if (
                normalized_role is ProverRole.KERNEL
                and not entry.reconstruction_capable
            ):
                return CapabilityAdmissionDecision(
                    text,
                    CapabilityAdmissionStatus.UNSUPPORTED,
                    kind,
                    "kernel is not reconstruction-capable",
                    capability_receipt_id=receipt_id,
                    readiness_identity=readiness_identity,
                    self_test_passed=True,
                    fragment_matched=True,
                )
            return CapabilityAdmissionDecision(
                text,
                CapabilityAdmissionStatus.ADMITTED,
                kind,
                "capability-qualified deterministic backend admitted",
                capability_receipt_id=receipt_id,
                readiness_identity=readiness_identity,
                self_test_passed=True,
                fragment_matched=True,
            )

        if readiness_ok:
            return CapabilityAdmissionDecision(
                text,
                CapabilityAdmissionStatus.ADMITTED,
                kind,
                "solver-readiness-qualified deterministic backend admitted",
                readiness_identity=readiness_identity,
                self_test_passed=bool(readiness and readiness.self_test_passed),
                fragment_matched=True,
            )

        if self._require_capability_evidence:
            return CapabilityAdmissionDecision(
                text,
                CapabilityAdmissionStatus.MISSING if required else CapabilityAdmissionStatus.UNSUPPORTED,
                kind,
                "capability evidence (matrix or SolverReadiness@1) is required",
                readiness_identity=readiness_identity,
                fragment_matched=True,
            )

        # No matrix/readiness evidence required: still only admit closed
        # deterministic kinds after representability and fragment checks.
        return CapabilityAdmissionDecision(
            text,
            CapabilityAdmissionStatus.ADMITTED,
            kind,
            "deterministic backend admitted without external capability matrix",
            readiness_identity=readiness_identity,
            self_test_passed=bool(readiness and readiness.self_test_passed),
            fragment_matched=True,
        )

    def admit_lane(self, lane: ProverLane, *, required: bool = False) -> CapabilityAdmissionDecision:
        if not isinstance(lane, ProverLane):
            raise ContractValidationError("lane must be a ProverLane")
        return self.admit_backend(
            lane.prover_id,
            role=lane.role,
            authority_capability=lane.authority_capability,
            required=required,
        )

    def admit_required_backends(
        self,
        required_backends: Iterable[str] | None = None,
    ) -> tuple[RequiredBackendAdmission, ...]:
        """Evaluate every required backend and retain fail-closed decisions."""

        backends = (
            _strings(required_backends, "required_backends")
            if required_backends is not None
            else self._required_backends
        )
        return tuple(
            RequiredBackendAdmission(
                backend=backend,
                decision=self.admit_backend(backend, required=True),
            )
            for backend in backends
        )

    def required_backends_fail_closed(
        self,
        required_backends: Iterable[str] | None = None,
    ) -> CapabilityAdmissionDecision | None:
        """Return the first required-backend failure, else ``None``."""

        for item in self.admit_required_backends(required_backends):
            if not item.admitted:
                return item.decision
        return None

    def filter_lanes(
        self,
        lanes: Sequence[ProverLane],
        *,
        required_backends: Iterable[str] | None = None,
    ) -> tuple[tuple[ProverLane, CapabilityAdmissionDecision], ...]:
        """Admit each planned lane; retain every decision for durable records."""

        required = set(
            _strings(required_backends, "required_backends")
            if required_backends is not None
            else self._required_backends
        )
        return tuple(
            (
                lane,
                self.admit_lane(lane, required=lane.prover_id in required),
            )
            for lane in lanes
        )

    def with_obligation_context(
        self,
        obligation: "PropertyObligation | Mapping[str, Any]",
    ) -> "ProverCapabilityAdmission":
        """Derive fragment/required-backend context from an obligation."""

        if isinstance(obligation, PropertyObligation):
            metadata = dict(obligation.metadata)
        elif isinstance(obligation, Mapping):
            metadata = dict(obligation.get("metadata") or {})
            if not isinstance(metadata, dict):
                metadata = {}
        else:
            raise ContractValidationError("obligation must provide metadata")
        fragment = str(
            metadata.get("logic_fragment")
            or metadata.get("fragment")
            or self._logic_fragment
            or ""
        )
        if "required_backends" in metadata:
            required = metadata.get("required_backends")
        elif "required_backend" in metadata:
            required = metadata.get("required_backend")
        else:
            required = None
        if required is None:
            required_backends = self._required_backends
        elif isinstance(required, str):
            required_backends = _strings((required,), "required_backends")
        else:
            required_backends = _strings(required, "required_backends")
        return ProverCapabilityAdmission(
            matrix=self._matrix,
            readiness=self._readiness,
            require_self_test=self._require_self_test,
            require_capability_evidence=self._require_capability_evidence,
            allowed_kinds=self._allowed_kinds,
            required_backends=required_backends,
            logic_fragment=fragment,
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
        if self.verdict is not PortfolioVerdict.PROVED and self.assurance is not AssuranceLevel.UNVERIFIED:
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
    """Plan and execute reviewed multi-prover portfolios.

    When a :class:`ProverCapabilityAdmission` is supplied, only capability-
    qualified deterministic backends run.  Required missing, unsupported, or
    error backends fail closed; general LLM and remote nondeterministic
    providers are never admitted.
    """

    INTERFACE = PROVER_PORTFOLIO_INTERFACE

    def __init__(
        self,
        policies: Mapping[PropertyKind | str, PropertyPolicy] | None = None,
        *,
        matrix: ProverMatrixSnapshot | None = None,
        conformance_reports: Mapping[str, ConformanceReport] | None = None,
        quarantine_registry: ProverQuarantineRegistry | None = None,
        capability_admission: ProverCapabilityAdmission | None = None,
        readiness: SolverReadinessReport | None = None,
        require_deterministic_admission: bool = False,
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
        if capability_admission is not None and not isinstance(
            capability_admission, ProverCapabilityAdmission
        ):
            raise ContractValidationError(
                "capability_admission must be a ProverCapabilityAdmission"
            )
        if readiness is not None and not isinstance(readiness, SolverReadinessReport):
            raise ContractValidationError(
                "readiness must be a SolverReadinessReport (SolverReadiness@1)"
            )
        if not isinstance(require_deterministic_admission, bool):
            raise ContractValidationError(
                "require_deterministic_admission must be boolean"
            )
        if capability_admission is not None:
            self._capability_admission = capability_admission
        elif require_deterministic_admission or readiness is not None:
            self._capability_admission = ProverCapabilityAdmission(
                matrix=matrix,
                readiness=readiness,
                require_capability_evidence=matrix is not None or readiness is not None,
                require_self_test=True,
            )
        else:
            self._capability_admission = None
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

    @property
    def capability_admission(self) -> ProverCapabilityAdmission | None:
        return self._capability_admission

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

    def _admission_for(
        self, obligation: PropertyObligation
    ) -> ProverCapabilityAdmission | None:
        if self._capability_admission is None:
            return None
        return self._capability_admission.with_obligation_context(obligation)

    def _lane_gate(
        self,
        lane: ProverLane,
        policy: PropertyPolicy,
        *,
        admission: ProverCapabilityAdmission | None = None,
        required: bool = False,
    ) -> _LaneGate:
        active_admission = admission if admission is not None else self._capability_admission
        if active_admission is not None:
            decision = active_admission.admit_lane(lane, required=required)
            if not decision.admitted:
                return _LaneGate(
                    False,
                    False,
                    decision.attempt_outcome,
                    decision.detail,
                    decision.capability_receipt_id,
                )

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
        receipt_id = ""
        if entry is not None and entry.receipt is not None:
            receipt_id = entry.receipt.receipt_id
        elif active_admission is not None:
            decision = active_admission.admit_lane(lane, required=required)
            receipt_id = decision.capability_receipt_id
        return _LaneGate(
            True,
            authoritative,
            None,
            "",
            receipt_id,
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
        admission = self._admission_for(plan.obligation)
        started = self._monotonic()
        deadline = started + policy.timeout_seconds

        # Required backends fail closed before any portfolio runner is invoked.
        if admission is not None:
            required_failure = admission.required_backends_fail_closed()
            if required_failure is not None:
                blocked = tuple(
                    self._attempt_from_output(
                        lane,
                        _LaneGate(
                            False,
                            False,
                            required_failure.attempt_outcome,
                            (
                                f"required backend {required_failure.prover_id!r} "
                                f"failed closed: {required_failure.detail}"
                            ),
                            required_failure.capability_receipt_id,
                        ),
                        ProverOutput(
                            required_failure.attempt_outcome,
                            (
                                f"required backend {required_failure.prover_id!r} "
                                f"failed closed: {required_failure.detail}"
                            ),
                        ),
                        0,
                    )
                    for lane in plan.lanes
                )
                return self._derive_result(
                    plan,
                    policy,
                    blocked,
                    "",
                    max(0, round((self._monotonic() - started) * 1000)),
                )

        required_ids = set(admission.required_backends) if admission is not None else set()
        gates = {
            lane.prover_id: self._lane_gate(
                lane,
                policy,
                admission=admission,
                required=lane.prover_id in required_ids,
            )
            for lane in plan.lanes
        }
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
    "PROVER_CAPABILITY_ADMISSION_SCHEMA",
    "PROVER_PORTFOLIO_INTERFACE",
    "SOLVER_READINESS_INTERFACE",
    "AttemptOutcome",
    "AttemptRequest",
    "CapabilityAdmissionDecision",
    "CapabilityAdmissionStatus",
    "DeterministicBackendKind",
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
    "ProverCapabilityAdmission",
    "ProverLane",
    "ProverOutput",
    "ProverRole",
    "RequiredBackendAdmission",
    "RouteVerdict",
    "backend_is_representable",
    "classify_deterministic_backend",
    "classify_property_kind",
    "execute_portfolio",
    "route_obligation",
    "to_canonical_property_kind",
]
