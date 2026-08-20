"""FACP-052 — bounded reactive supervisor controller (RPS).

Synthesize or mechanically validate control policies for provider routing,
retry/fallback, leases, human gates, proof escalation, compensation, and
safe shutdown. Hard temporal properties are never weakened; unrealizable
specifications produce explanatory cores; unknown irreversible outcomes are
never blindly retried; provider fallback cannot change authority or evidence
class.

This module does **not**:

* synthesize arbitrary repository source
* waive or weaken hard properties to force realizability
* admit unbounded retry or unbounded parallelism
* treat LLM / heuristic stages as assurance authority
* execute irreversible external effects (monitor/policy only)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Optional

from ..proof.formal_assurance_orchestrator import (
    PROHIBITED_ASSURANCE_STAGES,
    EscalationStage,
    escalation_ladder,
    next_stronger_stage,
)
from ..proof.formal_verification_contracts import canonical_json
from .formal_transition_monitor import (
    DEFAULT_MAX_FENCE_GEN,
    DEFAULT_MAX_RETRIES,
    REQUIRED_INVARIANTS as TEP_REQUIRED_INVARIANTS,
    REVERSIBILITY_CLASSES,
    TYPESTATES,
)


# ---------------------------------------------------------------------------
# FACP evidence envelope
# ---------------------------------------------------------------------------

TASK_ID: Final[str] = "FACP-052"
GOAL_ID: Final[str] = "FACP-G720"
BUNDLE: Final[str] = "facp/synthesis/controller"
SCHEMA: Final[str] = "facp/supervisor-controller@1"
REACTIVE_EVIDENCE: Final[str] = "facp/reactive-controller@1"
CORE_SCHEMA: Final[str] = "facp/unrealizable-core@1"
SPEC_SCHEMA: Final[str] = "facp/controller-spec@1"
POLICY_SCHEMA: Final[str] = "facp/controller-policy@1"
GUARD_SCHEMA: Final[str] = "facp/controller-guard@1"
MONITOR_SCHEMA: Final[str] = "facp/controller-monitor@1"
RESULT_SCHEMA: Final[str] = "facp/controller-result@1"
INTERFACE: Final[str] = "FormalAssuranceController@1"
ANALYZER_VERSION: Final[str] = "formal-assurance-controller/v1"
TOOLCHAIN_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.runtime.formal_assurance_controller/"
    + ANALYZER_VERSION
)

NORMATIVE_PINS: Final[tuple[str, ...]] = (
    "norm:eak-typestate",
    "norm:terminal-safety-statement",
    "norm:promotion-predicates",
    "norm:non-implications",
    "norm:safety-floors",
)

EVIDENCE_SUBSET: Final[tuple[str, ...]] = (
    "hard_safety",
    "liveness_under_healthy_assumptions",
    "soft_cost_objectives",
    "state_machine",
    "guards",
    "runtime_monitor",
    "unrealizable_core",
)

DEFAULT_MAX_PARALLEL: Final[int] = 2
DEFAULT_MAX_HUMAN_GATES: Final[int] = 2
DEFAULT_MAX_COMPENSATION_DEPTH: Final[int] = 2
DEFAULT_HORIZON: Final[int] = 32

# Closed effect vocabulary for the controller grammar (not TEP action names).
CONTROL_EFFECTS: Final[frozenset[str]] = frozenset(
    {
        "route_provider",
        "retry",
        "acquire_lease",
        "release_lease",
        "human_gate",
        "escalate_proof",
        "compensate",
        "shutdown",
        "abort",
        "mark_unavailable",
        "observe_unknown",
        "observe",
        "seal_receipt",
        "fallback_provider",
    }
)

EFFECTFUL_CONTROL: Final[frozenset[str]] = frozenset(
    {
        "route_provider",
        "retry",
        "fallback_provider",
        "compensate",
        "observe",
        "seal_receipt",
        "acquire_lease",
    }
)

TERMINAL_MODES: Final[frozenset[str]] = frozenset(
    {
        "terminal_success",
        "terminal_rejected",
        "terminal_unavailable",
        "terminal_compensated",
        "terminal_shutdown",
        "terminal_aborted",
    }
)

CONTROLLER_MODES: Final[frozenset[str]] = frozenset(
    {
        "idle",
        "admitting",
        "leased",
        "routing",
        "awaiting_observation",
        "unknown_pending",
        "failed",
        "human_gated",
        "escalating",
        "compensating",
        "shutting_down",
        *TERMINAL_MODES,
    }
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ControllerMode(str, Enum):
    SYNTHESIZE = "synthesize"
    VALIDATE = "validate"


class ControlEffect(str, Enum):
    ROUTE_PROVIDER = "route_provider"
    RETRY = "retry"
    ACQUIRE_LEASE = "acquire_lease"
    RELEASE_LEASE = "release_lease"
    HUMAN_GATE = "human_gate"
    ESCALATE_PROOF = "escalate_proof"
    COMPENSATE = "compensate"
    SHUTDOWN = "shutdown"
    ABORT = "abort"
    MARK_UNAVAILABLE = "mark_unavailable"
    OBSERVE_UNKNOWN = "observe_unknown"
    OBSERVE = "observe"
    SEAL_RECEIPT = "seal_receipt"
    FALLBACK_PROVIDER = "fallback_provider"


class HardPropertyId(str, Enum):
    NO_WEAKEN_HARD_SAFETY = "NoWeakenHardSafety"
    BOUNDED_RETRY = "BoundedRetry"
    BOUNDED_PARALLELISM = "BoundedParallelism"
    FALLBACK_PRESERVES_AUTHORITY = "FallbackPreservesAuthority"
    FALLBACK_PRESERVES_EVIDENCE_CLASS = "FallbackPreservesEvidenceClass"
    NO_BLIND_UNKNOWN_IRREVERSIBLE_RETRY = "NoBlindUnknownIrreversibleRetry"
    NO_SUCCESS_WITHOUT_OBSERVATION = "NoSuccessWithoutObservation"
    LEASE_BEFORE_EFFECT = "LeaseBeforeEffect"
    HUMAN_GATE_BEFORE_IRREVERSIBLE = "HumanGateBeforeIrreversible"
    PROOF_ESCALATION_MONOTONE = "ProofEscalationMonotone"
    SAFE_SHUTDOWN = "SafeShutdown"
    TYPESTATE_CLOSED = "TypestateClosed"
    COMPENSATION_EXPLICIT = "CompensationExplicit"


class SoftObjectiveId(str, Enum):
    MINIMIZE_COST = "MinimizeCost"
    PREFER_CHEAPEST_SOUND_STAGE = "PreferCheapestSoundStage"
    PREFER_HEALTHY_PROVIDER = "PreferHealthyProvider"


class ControllerVerdict(str, Enum):
    REALIZED = "realized"
    UNREALIZABLE = "unrealizable"
    INVALID_SPEC = "invalid_spec"
    REJECTED = "rejected"


class AuthorityClass(str, Enum):
    NONE = "none"
    PROPOSAL_ONLY = "proposal_only"
    KERNEL_ADMITTED = "kernel_admitted"
    LIVE_OBSERVED = "live_observed"


class EvidenceClass(str, Enum):
    FIXTURE = "fixture"
    SIMULATED = "simulated"
    HERMETIC = "hermetic"
    LIVE = "live"
    UNKNOWN = "unknown"


class ReversibilityClass(str, Enum):
    REVERSIBLE = "reversible"
    COMPENSATABLE = "compensatable"
    IRREVERSIBLE = "irreversible"


class ControllerErrorCode(str, Enum):
    INVALID_SPEC = "invalid_spec"
    UNKNOWN_EFFECT = "unknown_effect"
    UNKNOWN_MODE = "unknown_mode"
    UNKNOWN_PROPERTY = "unknown_property"
    UNKNOWN_FIELD = "unknown_field"
    BOUND_EXCEEDED = "bound_exceeded"
    HARD_PROPERTY_VIOLATION = "hard_property_violation"
    GUARD_FAILED = "guard_failed"
    UNREALIZABLE = "unrealizable"
    WEAKENED_HARD_PROPERTY = "weakened_hard_property"
    PROHIBITED_STAGE = "prohibited_stage"
    SHUTDOWN_LATCHED = "shutdown_latched"
    FALLBACK_PROMOTION = "fallback_promotion"
    PRESTATE_MISMATCH = "prestate_mismatch"


# Authority / evidence lattices (lower index = weaker). Fallback must not raise.
_AUTHORITY_RANK: Final[Mapping[AuthorityClass, int]] = MappingProxyType(
    {
        AuthorityClass.NONE: 0,
        AuthorityClass.PROPOSAL_ONLY: 1,
        AuthorityClass.KERNEL_ADMITTED: 2,
        AuthorityClass.LIVE_OBSERVED: 3,
    }
)
_EVIDENCE_RANK: Final[Mapping[EvidenceClass, int]] = MappingProxyType(
    {
        EvidenceClass.FIXTURE: 0,
        EvidenceClass.SIMULATED: 1,
        EvidenceClass.UNKNOWN: 2,
        EvidenceClass.HERMETIC: 3,
        EvidenceClass.LIVE: 4,
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ControllerError(ValueError):
    """Fail-closed rejection for malformed or unsafe controller inputs."""

    def __init__(
        self,
        code: ControllerErrorCode | str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(code, ControllerErrorCode):
            self.code = code
        else:
            try:
                self.code = ControllerErrorCode(code)
            except ValueError:
                self.code = ControllerErrorCode.INVALID_SPEC
        self.details = dict(details or {})
        super().__init__(f"{self.code.value}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": str(self),
            "details": dict(self.details),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ControllerError(
            ControllerErrorCode.INVALID_SPEC, f"{name} must be a string"
        )
    value = value.strip()
    if required and not value:
        raise ControllerError(
            ControllerErrorCode.INVALID_SPEC, f"{name} must not be empty"
        )
    return value


def _strings(values: Iterable[Any] | None, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ControllerError(
            ControllerErrorCode.INVALID_SPEC, f"{name} must be a sequence"
        )
    seen: list[str] = []
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.append(text)
    return tuple(seen)


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControllerError(
            ControllerErrorCode.INVALID_SPEC,
            f"{name} must be a non-negative integer",
        )
    if value < 0 or (value == 0 and not allow_zero):
        if allow_zero and value == 0:
            return 0
        raise ControllerError(
            ControllerErrorCode.INVALID_SPEC,
            f"{name} must be a positive integer",
        )
    return value


def _nonneg_int(value: Any, name: str) -> int:
    return _positive_int(value, name, allow_zero=True)


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ControllerError(
            ControllerErrorCode.INVALID_SPEC,
            f"{name} is unsupported: {value!r}",
        ) from exc


def _content_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:sha256:{digest}"


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ControllerError(
            ControllerErrorCode.UNKNOWN_FIELD,
            f"{name} contains unknown fields: {unknown}",
            details={"unknown": unknown},
        )


def authority_rank(value: AuthorityClass | str) -> int:
    cls = _enum(value, AuthorityClass, "authority_class")
    return _AUTHORITY_RANK[cls]


def evidence_rank(value: EvidenceClass | str) -> int:
    cls = _enum(value, EvidenceClass, "evidence_class")
    return _EVIDENCE_RANK[cls]


def _reject_prohibited_stage(stage_name: str) -> None:
    normalized = stage_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in PROHIBITED_ASSURANCE_STAGES:
        raise ControllerError(
            ControllerErrorCode.PROHIBITED_STAGE,
            f"prohibited assurance stage {stage_name!r}; LLM/heuristic stages "
            "are never admitted into controller proof escalation",
        )


# ---------------------------------------------------------------------------
# Bounds / properties / guards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerBounds:
    """Finite checked bounds for synthesis, validation, and monitoring."""

    max_retries: int = DEFAULT_MAX_RETRIES
    max_fence_gen: int = DEFAULT_MAX_FENCE_GEN
    max_parallel: int = DEFAULT_MAX_PARALLEL
    max_human_gates: int = DEFAULT_MAX_HUMAN_GATES
    max_compensation_depth: int = DEFAULT_MAX_COMPENSATION_DEPTH
    max_escalation_rank: int = field(
        default_factory=lambda: EscalationStage.HUMAN.rank
    )
    horizon: int = DEFAULT_HORIZON

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_retries", _nonneg_int(self.max_retries, "max_retries")
        )
        object.__setattr__(
            self, "max_fence_gen", _positive_int(self.max_fence_gen, "max_fence_gen")
        )
        object.__setattr__(
            self, "max_parallel", _positive_int(self.max_parallel, "max_parallel")
        )
        object.__setattr__(
            self,
            "max_human_gates",
            _nonneg_int(self.max_human_gates, "max_human_gates"),
        )
        object.__setattr__(
            self,
            "max_compensation_depth",
            _nonneg_int(self.max_compensation_depth, "max_compensation_depth"),
        )
        object.__setattr__(
            self,
            "max_escalation_rank",
            _nonneg_int(self.max_escalation_rank, "max_escalation_rank"),
        )
        object.__setattr__(
            self, "horizon", _positive_int(self.horizon, "horizon")
        )
        human_rank = EscalationStage.HUMAN.rank
        if self.max_escalation_rank > human_rank:
            raise ControllerError(
                ControllerErrorCode.BOUND_EXCEEDED,
                "max_escalation_rank exceeds proof ladder",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "max_fence_gen": self.max_fence_gen,
            "max_parallel": self.max_parallel,
            "max_human_gates": self.max_human_gates,
            "max_compensation_depth": self.max_compensation_depth,
            "max_escalation_rank": self.max_escalation_rank,
            "horizon": self.horizon,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ControllerBounds":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "bounds must be an object"
            )
        allowed = frozenset(cls().to_dict())
        _reject_unknown_keys(payload, allowed, "bounds")
        kwargs = {key: payload[key] for key in allowed if key in payload}
        return cls(**kwargs)


@dataclass(frozen=True)
class HardProperty:
    """Non-waivable hard safety / bound property."""

    property_id: HardPropertyId
    formula: str
    bound: int | None = None
    waivable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_id",
            _enum(self.property_id, HardPropertyId, "property_id"),
        )
        object.__setattr__(self, "formula", _text(self.formula, "formula"))
        if self.waivable:
            raise ControllerError(
                ControllerErrorCode.WEAKENED_HARD_PROPERTY,
                f"hard property {self.property_id.value} cannot be waivable",
            )
        if self.bound is not None:
            object.__setattr__(self, "bound", _nonneg_int(self.bound, "bound"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id.value,
            "formula": self.formula,
            "bound": self.bound,
            "waivable": False,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HardProperty":
        if not isinstance(payload, Mapping):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "hard property must be an object"
            )
        allowed = frozenset({"property_id", "formula", "bound", "waivable"})
        _reject_unknown_keys(payload, allowed, "hard_property")
        return cls(
            property_id=payload.get("property_id"),  # type: ignore[arg-type]
            formula=str(payload.get("formula") or ""),
            bound=payload.get("bound"),
            waivable=bool(payload.get("waivable", False)),
        )


@dataclass(frozen=True)
class SoftObjective:
    """Advisory cost preference; never overrides hard properties."""

    objective_id: SoftObjectiveId
    weight: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_id",
            _enum(self.objective_id, SoftObjectiveId, "objective_id"),
        )
        object.__setattr__(self, "weight", _positive_int(self.weight, "weight"))

    def to_dict(self) -> dict[str, Any]:
        return {"objective_id": self.objective_id.value, "weight": self.weight}


def default_hard_properties(bounds: ControllerBounds | None = None) -> tuple[HardProperty, ...]:
    """Baseline hard properties for checked bounds (never waivable)."""

    b = bounds or ControllerBounds()
    return (
        HardProperty(
            HardPropertyId.NO_WEAKEN_HARD_SAFETY,
            "hard properties remain non-waivable and complete vs baseline",
        ),
        HardProperty(
            HardPropertyId.BOUNDED_RETRY,
            f"retries <= {b.max_retries}",
            bound=b.max_retries,
        ),
        HardProperty(
            HardPropertyId.BOUNDED_PARALLELISM,
            f"parallel_instances <= {b.max_parallel}",
            bound=b.max_parallel,
        ),
        HardProperty(
            HardPropertyId.FALLBACK_PRESERVES_AUTHORITY,
            "fallback authority_class rank does not increase",
        ),
        HardProperty(
            HardPropertyId.FALLBACK_PRESERVES_EVIDENCE_CLASS,
            "fallback evidence_class rank does not increase",
        ),
        HardProperty(
            HardPropertyId.NO_BLIND_UNKNOWN_IRREVERSIBLE_RETRY,
            "retry forbidden when irreversible and (unknown_pending or effect applied)",
        ),
        HardProperty(
            HardPropertyId.NO_SUCCESS_WITHOUT_OBSERVATION,
            "terminal_success requires observed evidence",
        ),
        HardProperty(
            HardPropertyId.LEASE_BEFORE_EFFECT,
            "route/fallback/observe effects require lease_held",
        ),
        HardProperty(
            HardPropertyId.HUMAN_GATE_BEFORE_IRREVERSIBLE,
            "irreversible effects require confirmation when demanded",
        ),
        HardProperty(
            HardPropertyId.PROOF_ESCALATION_MONOTONE,
            "escalate_proof only to a strictly stronger admitted stage",
        ),
        HardProperty(
            HardPropertyId.SAFE_SHUTDOWN,
            "after shutdown latch, no further effectful control actions",
        ),
        HardProperty(
            HardPropertyId.TYPESTATE_CLOSED,
            "control modes stay inside the closed controller grammar",
        ),
        HardProperty(
            HardPropertyId.COMPENSATION_EXPLICIT,
            "compensatable failures use compensate; never silent success",
        ),
    )


def default_soft_objectives() -> tuple[SoftObjective, ...]:
    return (
        SoftObjective(SoftObjectiveId.MINIMIZE_COST, weight=1),
        SoftObjective(SoftObjectiveId.PREFER_CHEAPEST_SOUND_STAGE, weight=2),
        SoftObjective(SoftObjectiveId.PREFER_HEALTHY_PROVIDER, weight=3),
    )


@dataclass(frozen=True)
class ControllerGuard:
    """Predicate over observable controller state."""

    guard_id: str
    require_lease: bool = False
    require_confirmation: bool = False
    forbid_unknown_pending: bool = False
    forbid_shutdown: bool = True
    max_retries: int | None = None
    max_parallel: int | None = None
    allowed_reversibility: tuple[str, ...] = ()
    min_authority: str = AuthorityClass.NONE.value
    max_authority: str = AuthorityClass.LIVE_OBSERVED.value
    min_evidence: str = EvidenceClass.FIXTURE.value
    max_evidence: str = EvidenceClass.LIVE.value
    schema: str = GUARD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "guard_id", _text(self.guard_id, "guard_id"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != GUARD_SCHEMA:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                f"unsupported guard schema: {self.schema}",
            )
        if self.max_retries is not None:
            object.__setattr__(
                self, "max_retries", _nonneg_int(self.max_retries, "max_retries")
            )
        if self.max_parallel is not None:
            object.__setattr__(
                self, "max_parallel", _positive_int(self.max_parallel, "max_parallel")
            )
        revs = tuple(
            _text(item, "allowed_reversibility")
            for item in (self.allowed_reversibility or ())
        )
        for item in revs:
            if item not in REVERSIBILITY_CLASSES:
                raise ControllerError(
                    ControllerErrorCode.INVALID_SPEC,
                    f"unknown reversibility class: {item}",
                )
        object.__setattr__(self, "allowed_reversibility", revs)
        object.__setattr__(
            self,
            "min_authority",
            _enum(self.min_authority, AuthorityClass, "min_authority").value,
        )
        object.__setattr__(
            self,
            "max_authority",
            _enum(self.max_authority, AuthorityClass, "max_authority").value,
        )
        object.__setattr__(
            self,
            "min_evidence",
            _enum(self.min_evidence, EvidenceClass, "min_evidence").value,
        )
        object.__setattr__(
            self,
            "max_evidence",
            _enum(self.max_evidence, EvidenceClass, "max_evidence").value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "guard_id": self.guard_id,
            "require_lease": self.require_lease,
            "require_confirmation": self.require_confirmation,
            "forbid_unknown_pending": self.forbid_unknown_pending,
            "forbid_shutdown": self.forbid_shutdown,
            "max_retries": self.max_retries,
            "max_parallel": self.max_parallel,
            "allowed_reversibility": list(self.allowed_reversibility),
            "min_authority": self.min_authority,
            "max_authority": self.max_authority,
            "min_evidence": self.min_evidence,
            "max_evidence": self.max_evidence,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControllerGuard":
        if not isinstance(payload, Mapping):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "guard must be an object"
            )
        allowed = frozenset(
            {
                "schema",
                "guard_id",
                "require_lease",
                "require_confirmation",
                "forbid_unknown_pending",
                "forbid_shutdown",
                "max_retries",
                "max_parallel",
                "allowed_reversibility",
                "min_authority",
                "max_authority",
                "min_evidence",
                "max_evidence",
            }
        )
        _reject_unknown_keys(payload, allowed, "guard")
        return cls(
            guard_id=str(payload.get("guard_id") or ""),
            require_lease=bool(payload.get("require_lease", False)),
            require_confirmation=bool(payload.get("require_confirmation", False)),
            forbid_unknown_pending=bool(payload.get("forbid_unknown_pending", False)),
            forbid_shutdown=bool(payload.get("forbid_shutdown", True)),
            max_retries=payload.get("max_retries"),
            max_parallel=payload.get("max_parallel"),
            allowed_reversibility=tuple(payload.get("allowed_reversibility") or ()),
            min_authority=str(payload.get("min_authority") or AuthorityClass.NONE.value),
            max_authority=str(
                payload.get("max_authority") or AuthorityClass.LIVE_OBSERVED.value
            ),
            min_evidence=str(payload.get("min_evidence") or EvidenceClass.FIXTURE.value),
            max_evidence=str(payload.get("max_evidence") or EvidenceClass.LIVE.value),
            schema=str(payload.get("schema") or GUARD_SCHEMA),
        )


@dataclass(frozen=True)
class ControllerTransition:
    """One grammar edge: prior mode --effect/guards--> next mode."""

    transition_id: str
    prior_mode: str
    effect: ControlEffect
    next_mode: str
    guards: tuple[ControllerGuard, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "transition_id", _text(self.transition_id, "transition_id")
        )
        prior = _text(self.prior_mode, "prior_mode")
        nxt = _text(self.next_mode, "next_mode")
        if prior not in CONTROLLER_MODES:
            raise ControllerError(
                ControllerErrorCode.UNKNOWN_MODE, f"unknown prior_mode: {prior}"
            )
        if nxt not in CONTROLLER_MODES:
            raise ControllerError(
                ControllerErrorCode.UNKNOWN_MODE, f"unknown next_mode: {nxt}"
            )
        object.__setattr__(self, "prior_mode", prior)
        object.__setattr__(self, "next_mode", nxt)
        object.__setattr__(
            self, "effect", _enum(self.effect, ControlEffect, "effect")
        )
        guards = tuple(self.guards or ())
        for guard in guards:
            if not isinstance(guard, ControllerGuard):
                raise ControllerError(
                    ControllerErrorCode.INVALID_SPEC, "guards must be ControllerGuard"
                )
        object.__setattr__(self, "guards", guards)
        object.__setattr__(
            self, "notes", _text(self.notes, "notes", required=False)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "prior_mode": self.prior_mode,
            "effect": self.effect.value,
            "next_mode": self.next_mode,
            "guards": [guard.to_dict() for guard in self.guards],
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControllerTransition":
        if not isinstance(payload, Mapping):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "transition must be an object"
            )
        allowed = frozenset(
            {
                "transition_id",
                "prior_mode",
                "effect",
                "next_mode",
                "guards",
                "notes",
            }
        )
        _reject_unknown_keys(payload, allowed, "transition")
        guards_raw = payload.get("guards") or ()
        if isinstance(guards_raw, (str, bytes, bytearray)) or not isinstance(
            guards_raw, Sequence
        ):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "guards must be a sequence"
            )
        guards = tuple(
            item if isinstance(item, ControllerGuard) else ControllerGuard.from_mapping(item)
            for item in guards_raw
        )
        return cls(
            transition_id=str(payload.get("transition_id") or ""),
            prior_mode=str(payload.get("prior_mode") or ""),
            effect=payload.get("effect"),  # type: ignore[arg-type]
            next_mode=str(payload.get("next_mode") or ""),
            guards=guards,
            notes=str(payload.get("notes") or ""),
        )


# ---------------------------------------------------------------------------
# Spec / policy / core / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerSpec:
    """Requirements for synthesis or validation."""

    spec_id: str
    assumptions: tuple[str, ...] = ("assumption:deps-healthy",)
    hard_properties: tuple[HardProperty, ...] = ()
    soft_objectives: tuple[SoftObjective, ...] = ()
    required_effects: tuple[str, ...] = ()
    forbidden_effects: tuple[str, ...] = ()
    bounds: ControllerBounds = field(default_factory=ControllerBounds)
    require_human_gate_for_irreversible: bool = True
    allow_unbounded_retry: bool = False
    require_fallback_authority_promotion: bool = False
    require_fallback_evidence_promotion: bool = False
    required_contract_ids: tuple[str, ...] = ()
    schema: str = SPEC_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_id", _text(self.spec_id, "spec_id"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != SPEC_SCHEMA:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                f"unsupported spec schema: {self.schema}",
            )
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, "assumptions")
        )
        hard = tuple(self.hard_properties or default_hard_properties(self.bounds))
        for prop in hard:
            if not isinstance(prop, HardProperty):
                raise ControllerError(
                    ControllerErrorCode.INVALID_SPEC,
                    "hard_properties must contain HardProperty values",
                )
            if prop.waivable:
                raise ControllerError(
                    ControllerErrorCode.WEAKENED_HARD_PROPERTY,
                    f"hard property {prop.property_id.value} is waivable",
                )
        object.__setattr__(self, "hard_properties", hard)
        soft = tuple(self.soft_objectives or default_soft_objectives())
        for obj in soft:
            if not isinstance(obj, SoftObjective):
                raise ControllerError(
                    ControllerErrorCode.INVALID_SPEC,
                    "soft_objectives must contain SoftObjective values",
                )
        object.__setattr__(self, "soft_objectives", soft)
        if not isinstance(self.bounds, ControllerBounds):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "bounds must be ControllerBounds"
            )
        required = _strings(self.required_effects, "required_effects")
        forbidden = _strings(self.forbidden_effects, "forbidden_effects")
        for effect in (*required, *forbidden):
            if effect not in CONTROL_EFFECTS:
                raise ControllerError(
                    ControllerErrorCode.UNKNOWN_EFFECT,
                    f"unknown control effect: {effect}",
                )
        object.__setattr__(self, "required_effects", required)
        object.__setattr__(self, "forbidden_effects", forbidden)
        object.__setattr__(
            self,
            "required_contract_ids",
            _strings(self.required_contract_ids, "required_contract_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "spec_id": self.spec_id,
            "assumptions": list(self.assumptions),
            "hard_properties": [prop.to_dict() for prop in self.hard_properties],
            "soft_objectives": [obj.to_dict() for obj in self.soft_objectives],
            "required_effects": list(self.required_effects),
            "forbidden_effects": list(self.forbidden_effects),
            "bounds": self.bounds.to_dict(),
            "require_human_gate_for_irreversible": self.require_human_gate_for_irreversible,
            "allow_unbounded_retry": self.allow_unbounded_retry,
            "require_fallback_authority_promotion": self.require_fallback_authority_promotion,
            "require_fallback_evidence_promotion": self.require_fallback_evidence_promotion,
            "required_contract_ids": list(self.required_contract_ids),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControllerSpec":
        if not isinstance(payload, Mapping):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "spec must be an object"
            )
        allowed = frozenset(
            {
                "schema",
                "spec_id",
                "assumptions",
                "hard_properties",
                "soft_objectives",
                "required_effects",
                "forbidden_effects",
                "bounds",
                "require_human_gate_for_irreversible",
                "allow_unbounded_retry",
                "require_fallback_authority_promotion",
                "require_fallback_evidence_promotion",
                "required_contract_ids",
            }
        )
        _reject_unknown_keys(payload, allowed, "spec")
        bounds = ControllerBounds.from_mapping(payload.get("bounds"))
        hard_raw = payload.get("hard_properties")
        if hard_raw is None:
            hard: tuple[HardProperty, ...] = default_hard_properties(bounds)
        else:
            if not isinstance(hard_raw, Sequence) or isinstance(
                hard_raw, (str, bytes, bytearray)
            ):
                raise ControllerError(
                    ControllerErrorCode.INVALID_SPEC,
                    "hard_properties must be a sequence",
                )
            hard = tuple(
                item if isinstance(item, HardProperty) else HardProperty.from_mapping(item)
                for item in hard_raw
            )
        soft_raw = payload.get("soft_objectives")
        if soft_raw is None:
            soft = default_soft_objectives()
        else:
            if not isinstance(soft_raw, Sequence) or isinstance(
                soft_raw, (str, bytes, bytearray)
            ):
                raise ControllerError(
                    ControllerErrorCode.INVALID_SPEC,
                    "soft_objectives must be a sequence",
                )
            soft_list: list[SoftObjective] = []
            for item in soft_raw:
                if isinstance(item, SoftObjective):
                    soft_list.append(item)
                else:
                    soft_list.append(
                        SoftObjective(
                            objective_id=item.get("objective_id"),  # type: ignore[arg-type]
                            weight=int(item.get("weight") or 1),
                        )
                    )
            soft = tuple(soft_list)
        return cls(
            spec_id=str(payload.get("spec_id") or ""),
            assumptions=tuple(payload.get("assumptions") or ("assumption:deps-healthy",)),
            hard_properties=hard,
            soft_objectives=soft,
            required_effects=tuple(payload.get("required_effects") or ()),
            forbidden_effects=tuple(payload.get("forbidden_effects") or ()),
            bounds=bounds,
            require_human_gate_for_irreversible=bool(
                payload.get("require_human_gate_for_irreversible", True)
            ),
            allow_unbounded_retry=bool(payload.get("allow_unbounded_retry", False)),
            require_fallback_authority_promotion=bool(
                payload.get("require_fallback_authority_promotion", False)
            ),
            require_fallback_evidence_promotion=bool(
                payload.get("require_fallback_evidence_promotion", False)
            ),
            required_contract_ids=tuple(payload.get("required_contract_ids") or ()),
            schema=str(payload.get("schema") or SPEC_SCHEMA),
        )


@dataclass(frozen=True)
class ControllerPolicy:
    """Synthesized or validated bounded control policy artifact."""

    policy_id: str
    initial_mode: str
    transitions: tuple[ControllerTransition, ...]
    bounds: ControllerBounds
    hard_properties: tuple[HardProperty, ...]
    soft_objectives: tuple[SoftObjective, ...] = ()
    discharged_properties: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    schema: str = POLICY_SCHEMA
    evidence: tuple[str, ...] = (SCHEMA, REACTIVE_EVIDENCE)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != POLICY_SCHEMA:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                f"unsupported policy schema: {self.schema}",
            )
        initial = _text(self.initial_mode, "initial_mode")
        if initial not in CONTROLLER_MODES:
            raise ControllerError(
                ControllerErrorCode.UNKNOWN_MODE, f"unknown initial_mode: {initial}"
            )
        object.__setattr__(self, "initial_mode", initial)
        transitions = tuple(self.transitions or ())
        if not transitions:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                "policy must declare at least one transition",
            )
        for transition in transitions:
            if not isinstance(transition, ControllerTransition):
                raise ControllerError(
                    ControllerErrorCode.INVALID_SPEC,
                    "transitions must be ControllerTransition values",
                )
        object.__setattr__(self, "transitions", transitions)
        if not isinstance(self.bounds, ControllerBounds):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "bounds must be ControllerBounds"
            )
        hard = tuple(self.hard_properties or ())
        if not hard:
            raise ControllerError(
                ControllerErrorCode.WEAKENED_HARD_PROPERTY,
                "policy must retain baseline hard properties",
            )
        for prop in hard:
            if prop.waivable:
                raise ControllerError(
                    ControllerErrorCode.WEAKENED_HARD_PROPERTY,
                    f"policy hard property {prop.property_id.value} is waivable",
                )
        object.__setattr__(self, "hard_properties", hard)
        object.__setattr__(
            self,
            "soft_objectives",
            tuple(self.soft_objectives or ()),
        )
        object.__setattr__(
            self,
            "discharged_properties",
            _strings(self.discharged_properties, "discharged_properties"),
        )
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, "assumptions")
        )
        object.__setattr__(self, "evidence", _strings(self.evidence, "evidence"))
        # Content-address when caller supplies empty / provisional id.
        if not self.policy_id or self.policy_id == "provisional":
            object.__setattr__(self, "policy_id", self._derive_id())
        else:
            object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))

    def _derive_id(self) -> str:
        body = self.identity_payload()
        body.pop("policy_id", None)
        return _content_id("controller-policy", body)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "initial_mode": self.initial_mode,
            "transitions": [item.to_dict() for item in self.transitions],
            "bounds": self.bounds.to_dict(),
            "hard_properties": [prop.to_dict() for prop in self.hard_properties],
            "soft_objectives": [obj.to_dict() for obj in self.soft_objectives],
            "discharged_properties": list(self.discharged_properties),
            "assumptions": list(self.assumptions),
            "evidence": list(self.evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    def effects(self) -> frozenset[str]:
        return frozenset(transition.effect.value for transition in self.transitions)

    def modes(self) -> frozenset[str]:
        modes = {self.initial_mode}
        for transition in self.transitions:
            modes.add(transition.prior_mode)
            modes.add(transition.next_mode)
        return frozenset(modes)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControllerPolicy":
        if not isinstance(payload, Mapping):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "policy must be an object"
            )
        allowed = frozenset(
            {
                "schema",
                "policy_id",
                "initial_mode",
                "transitions",
                "bounds",
                "hard_properties",
                "soft_objectives",
                "discharged_properties",
                "assumptions",
                "evidence",
            }
        )
        _reject_unknown_keys(payload, allowed, "policy")
        transitions_raw = payload.get("transitions") or ()
        if not isinstance(transitions_raw, Sequence) or isinstance(
            transitions_raw, (str, bytes, bytearray)
        ):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "transitions must be a sequence"
            )
        transitions = tuple(
            item
            if isinstance(item, ControllerTransition)
            else ControllerTransition.from_mapping(item)
            for item in transitions_raw
        )
        hard_raw = payload.get("hard_properties") or ()
        hard = tuple(
            item if isinstance(item, HardProperty) else HardProperty.from_mapping(item)
            for item in hard_raw
        )
        soft_raw = payload.get("soft_objectives") or ()
        soft = tuple(
            item
            if isinstance(item, SoftObjective)
            else SoftObjective(
                objective_id=item.get("objective_id"),  # type: ignore[arg-type]
                weight=int(item.get("weight") or 1),
            )
            for item in soft_raw
        )
        return cls(
            policy_id=str(payload.get("policy_id") or "provisional"),
            initial_mode=str(payload.get("initial_mode") or ""),
            transitions=transitions,
            bounds=ControllerBounds.from_mapping(payload.get("bounds")),
            hard_properties=hard,
            soft_objectives=soft,
            discharged_properties=tuple(payload.get("discharged_properties") or ()),
            assumptions=tuple(payload.get("assumptions") or ()),
            schema=str(payload.get("schema") or POLICY_SCHEMA),
            evidence=tuple(payload.get("evidence") or (SCHEMA, REACTIVE_EVIDENCE)),
        )


@dataclass(frozen=True)
class UnrealizableCore:
    """Minimal explanatory core for unrealizable controller requirements."""

    core_id: str
    conflicting_requirements: tuple[str, ...]
    explanation: str
    conflicting_properties: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    schema: str = CORE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != CORE_SCHEMA:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                f"unsupported unrealizable-core schema: {self.schema}",
            )
        conflicts = _strings(
            self.conflicting_requirements, "conflicting_requirements"
        )
        if not conflicts:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                "unrealizable core must name conflicting requirements",
            )
        object.__setattr__(self, "conflicting_requirements", conflicts)
        object.__setattr__(
            self,
            "conflicting_properties",
            _strings(self.conflicting_properties, "conflicting_properties"),
        )
        object.__setattr__(self, "path", _strings(self.path, "path"))
        object.__setattr__(
            self, "explanation", _text(self.explanation, "explanation")
        )
        if not self.core_id or self.core_id == "provisional":
            object.__setattr__(self, "core_id", self._derive_id())
        else:
            object.__setattr__(self, "core_id", _text(self.core_id, "core_id"))

    def _derive_id(self) -> str:
        body = self.to_dict()
        body.pop("core_id", None)
        return _content_id("unrealizable-core", body)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "core_id": self.core_id,
            "conflicting_requirements": list(self.conflicting_requirements),
            "conflicting_properties": list(self.conflicting_properties),
            "path": list(self.path),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class ControllerMonitorVerdict:
    """One runtime monitor step over a policy."""

    accepted: bool
    code: str
    prior_mode: str
    next_mode: str
    effect: str
    message: str = ""
    invariant: str | None = None
    hard_properties: Mapping[str, bool] = field(default_factory=dict)
    schema: str = MONITOR_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "code": self.code,
            "prior_mode": self.prior_mode,
            "next_mode": self.next_mode,
            "effect": self.effect,
            "message": self.message,
            "invariant": self.invariant,
            "hard_properties": dict(self.hard_properties),
        }


@dataclass(frozen=True)
class ControllerResult:
    """Synthesis / validation outcome with evidence envelope."""

    verdict: ControllerVerdict
    assumptions: tuple[str, ...]
    toolchain: str
    bounds: ControllerBounds
    hard_property_results: Mapping[str, bool]
    policy: ControllerPolicy | None = None
    unrealizable_core: UnrealizableCore | None = None
    reason: str = ""
    soft_scores: Mapping[str, int] = field(default_factory=dict)
    normative_pins: tuple[str, ...] = NORMATIVE_PINS
    evidence: tuple[str, ...] = (SCHEMA, REACTIVE_EVIDENCE)
    schema: str = RESULT_SCHEMA
    result_id: str = "provisional"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ControllerVerdict, "verdict")
        )
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, "assumptions")
        )
        object.__setattr__(self, "toolchain", _text(self.toolchain, "toolchain"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != RESULT_SCHEMA:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                f"unsupported result schema: {self.schema}",
            )
        object.__setattr__(
            self,
            "normative_pins",
            _strings(self.normative_pins, "normative_pins"),
        )
        evidence = list(_strings(self.evidence, "evidence"))
        if self.unrealizable_core is not None and CORE_SCHEMA not in evidence:
            evidence.append(CORE_SCHEMA)
        object.__setattr__(self, "evidence", tuple(evidence))
        if not self.result_id or self.result_id == "provisional":
            object.__setattr__(self, "result_id", self._derive_id())
        else:
            object.__setattr__(self, "result_id", _text(self.result_id, "result_id"))

    def _derive_id(self) -> str:
        body = self.to_dict()
        body.pop("result_id", None)
        return _content_id("controller-result", body)

    @property
    def realized(self) -> bool:
        return self.verdict is ControllerVerdict.REALIZED

    @property
    def unrealizable(self) -> bool:
        return self.verdict is ControllerVerdict.UNREALIZABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "result_id": self.result_id,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "bundle": BUNDLE,
            "interface": INTERFACE,
            "verdict": self.verdict.value,
            "assumptions": list(self.assumptions),
            "toolchain": self.toolchain,
            "bounds": self.bounds.to_dict(),
            "hard_property_results": dict(self.hard_property_results),
            "soft_scores": dict(self.soft_scores),
            "policy": None if self.policy is None else self.policy.to_dict(),
            "unrealizable_core": (
                None
                if self.unrealizable_core is None
                else self.unrealizable_core.to_dict()
            ),
            "reason": self.reason,
            "normative_pins": list(self.normative_pins),
            "evidence": list(self.evidence),
            "analyzer_version": ANALYZER_VERSION,
        }


# ---------------------------------------------------------------------------
# Observation / runtime state for step monitor
# ---------------------------------------------------------------------------


@dataclass
class ControllerObservation:
    """Observable fields consulted by guards and hard-property checks."""

    mode: str = "idle"
    typestate: str = "Proposed"
    reversibility: str = ReversibilityClass.REVERSIBLE.value
    lease_held: bool = False
    confirmation_present: bool = False
    confirmation_spent: bool = False
    unknown_pending: bool = False
    observed: bool = False
    effect_count: int = 0
    retry_count: int = 0
    parallel_count: int = 0
    fence_gen: int = 1
    human_gate_count: int = 0
    compensation_depth: int = 0
    shutdown_latched: bool = False
    authority_class: str = AuthorityClass.NONE.value
    evidence_class: str = EvidenceClass.HERMETIC.value
    fallback_authority_class: str | None = None
    fallback_evidence_class: str | None = None
    proof_stage: str = EscalationStage.SCHEMA.value
    proposed_proof_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "typestate": self.typestate,
            "reversibility": self.reversibility,
            "lease_held": self.lease_held,
            "confirmation_present": self.confirmation_present,
            "confirmation_spent": self.confirmation_spent,
            "unknown_pending": self.unknown_pending,
            "observed": self.observed,
            "effect_count": self.effect_count,
            "retry_count": self.retry_count,
            "parallel_count": self.parallel_count,
            "fence_gen": self.fence_gen,
            "human_gate_count": self.human_gate_count,
            "compensation_depth": self.compensation_depth,
            "shutdown_latched": self.shutdown_latched,
            "authority_class": self.authority_class,
            "evidence_class": self.evidence_class,
            "fallback_authority_class": self.fallback_authority_class,
            "fallback_evidence_class": self.fallback_evidence_class,
            "proof_stage": self.proof_stage,
            "proposed_proof_stage": self.proposed_proof_stage,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ControllerObservation":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC, "observation must be an object"
            )
        allowed = frozenset(cls().to_dict())
        _reject_unknown_keys(payload, allowed, "observation")
        kwargs: dict[str, Any] = {}
        for key in allowed:
            if key in payload:
                kwargs[key] = payload[key]
        obs = cls(**kwargs)
        if obs.typestate and obs.typestate not in TYPESTATES:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                f"unknown typestate: {obs.typestate}",
            )
        if obs.reversibility not in REVERSIBILITY_CLASSES:
            raise ControllerError(
                ControllerErrorCode.INVALID_SPEC,
                f"unknown reversibility: {obs.reversibility}",
            )
        _enum(obs.authority_class, AuthorityClass, "authority_class")
        _enum(obs.evidence_class, EvidenceClass, "evidence_class")
        return obs


# ---------------------------------------------------------------------------
# Grammar template (deterministic synthesis)
# ---------------------------------------------------------------------------


def _guard(
    guard_id: str,
    *,
    require_lease: bool = False,
    require_confirmation: bool = False,
    forbid_unknown_pending: bool = False,
    max_retries: int | None = None,
    max_parallel: int | None = None,
    allowed_reversibility: Sequence[str] = (),
) -> ControllerGuard:
    return ControllerGuard(
        guard_id=guard_id,
        require_lease=require_lease,
        require_confirmation=require_confirmation,
        forbid_unknown_pending=forbid_unknown_pending,
        max_retries=max_retries,
        max_parallel=max_parallel,
        allowed_reversibility=tuple(allowed_reversibility),
    )


def build_default_grammar(
    bounds: ControllerBounds,
    *,
    require_human_gate_for_irreversible: bool = True,
) -> tuple[ControllerTransition, ...]:
    """Fixed controller grammar instantiated at the given bounds."""

    lease_guard = _guard("guard:lease", require_lease=True)
    retry_guard = _guard(
        "guard:retry",
        require_lease=True,
        forbid_unknown_pending=True,
        max_retries=bounds.max_retries,
        allowed_reversibility=(
            ReversibilityClass.REVERSIBLE.value,
            ReversibilityClass.COMPENSATABLE.value,
        ),
    )
    parallel_guard = _guard(
        "guard:parallel",
        require_lease=True,
        max_parallel=bounds.max_parallel,
    )
    irreversible_guards: list[ControllerGuard] = [
        _guard(
            "guard:irreversible-route",
            require_lease=True,
            require_confirmation=require_human_gate_for_irreversible,
            allowed_reversibility=(ReversibilityClass.IRREVERSIBLE.value,),
        )
    ]
    transitions = [
        ControllerTransition(
            "t:idle-admit",
            "idle",
            ControlEffect.ACQUIRE_LEASE,
            "leased",
            notes="acquire lease before effectful routing",
        ),
        ControllerTransition(
            "t:leased-route",
            "leased",
            ControlEffect.ROUTE_PROVIDER,
            "routing",
            guards=(parallel_guard,),
        ),
        ControllerTransition(
            "t:routing-observe",
            "routing",
            ControlEffect.OBSERVE,
            "awaiting_observation",
            guards=(lease_guard,),
        ),
        ControllerTransition(
            "t:await-seal",
            "awaiting_observation",
            ControlEffect.SEAL_RECEIPT,
            "terminal_success",
            guards=(lease_guard,),
            notes="seal only after observation",
        ),
        ControllerTransition(
            "t:routing-unknown",
            "routing",
            ControlEffect.OBSERVE_UNKNOWN,
            "unknown_pending",
            guards=(lease_guard,),
        ),
        ControllerTransition(
            "t:unknown-compensate",
            "unknown_pending",
            ControlEffect.COMPENSATE,
            "compensating",
            guards=(
                _guard(
                    "guard:compensate",
                    allowed_reversibility=(
                        ReversibilityClass.COMPENSATABLE.value,
                    ),
                ),
            ),
        ),
        ControllerTransition(
            "t:compensating-done",
            "compensating",
            ControlEffect.SEAL_RECEIPT,
            "terminal_compensated",
        ),
        ControllerTransition(
            "t:routing-fail",
            "routing",
            ControlEffect.ABORT,
            "failed",
        ),
        ControllerTransition(
            "t:failed-retry",
            "failed",
            ControlEffect.RETRY,
            "leased",
            guards=(retry_guard,),
        ),
        ControllerTransition(
            "t:failed-fallback",
            "failed",
            ControlEffect.FALLBACK_PROVIDER,
            "routing",
            guards=(lease_guard, parallel_guard),
            notes="fallback preserves authority/evidence class",
        ),
        ControllerTransition(
            "t:failed-escalate",
            "failed",
            ControlEffect.ESCALATE_PROOF,
            "escalating",
        ),
        ControllerTransition(
            "t:escalating-route",
            "escalating",
            ControlEffect.ROUTE_PROVIDER,
            "routing",
            guards=(lease_guard,),
        ),
        ControllerTransition(
            "t:leased-human",
            "leased",
            ControlEffect.HUMAN_GATE,
            "human_gated",
        ),
        ControllerTransition(
            "t:human-route",
            "human_gated",
            ControlEffect.ROUTE_PROVIDER,
            "routing",
            guards=(lease_guard, *irreversible_guards),
        ),
        ControllerTransition(
            "t:idle-unavailable",
            "idle",
            ControlEffect.MARK_UNAVAILABLE,
            "terminal_unavailable",
        ),
        ControllerTransition(
            "t:any-reject",
            "admitting",
            ControlEffect.ABORT,
            "terminal_rejected",
        ),
        ControllerTransition(
            "t:idle-shutdown",
            "idle",
            ControlEffect.SHUTDOWN,
            "shutting_down",
        ),
        ControllerTransition(
            "t:shutdown-done",
            "shutting_down",
            ControlEffect.SHUTDOWN,
            "terminal_shutdown",
        ),
        ControllerTransition(
            "t:failed-abort",
            "failed",
            ControlEffect.ABORT,
            "terminal_aborted",
        ),
        ControllerTransition(
            "t:release-idle",
            "leased",
            ControlEffect.RELEASE_LEASE,
            "idle",
        ),
    ]
    return tuple(transitions)


# ---------------------------------------------------------------------------
# Guard / hard-property evaluation
# ---------------------------------------------------------------------------


def evaluate_guard(
    guard: ControllerGuard,
    observation: ControllerObservation,
    *,
    bounds: ControllerBounds,
) -> tuple[bool, str]:
    if guard.forbid_shutdown and observation.shutdown_latched:
        return False, "shutdown latched"
    if guard.require_lease and not observation.lease_held:
        return False, "lease required"
    if guard.require_confirmation and not (
        observation.confirmation_present and not observation.confirmation_spent
    ):
        return False, "confirmation required"
    if guard.forbid_unknown_pending and observation.unknown_pending:
        return False, "unknown_pending forbidden"
    max_retries = (
        bounds.max_retries if guard.max_retries is None else guard.max_retries
    )
    if observation.retry_count > max_retries:
        return False, "retry bound exceeded"
    max_parallel = (
        bounds.max_parallel if guard.max_parallel is None else guard.max_parallel
    )
    if observation.parallel_count > max_parallel:
        return False, "parallelism bound exceeded"
    if guard.allowed_reversibility and (
        observation.reversibility not in guard.allowed_reversibility
    ):
        return False, "reversibility not allowed"
    auth = authority_rank(observation.authority_class)
    if auth < authority_rank(guard.min_authority) or auth > authority_rank(
        guard.max_authority
    ):
        return False, "authority class out of guard range"
    evid = evidence_rank(observation.evidence_class)
    if evid < evidence_rank(guard.min_evidence) or evid > evidence_rank(
        guard.max_evidence
    ):
        return False, "evidence class out of guard range"
    return True, "ok"


def _fallback_preserves(
    observation: ControllerObservation,
) -> tuple[bool, bool]:
    """Return (authority_ok, evidence_ok) for a proposed fallback."""

    auth_ok = True
    evid_ok = True
    if observation.fallback_authority_class is not None:
        auth_ok = authority_rank(observation.fallback_authority_class) <= authority_rank(
            observation.authority_class
        )
    if observation.fallback_evidence_class is not None:
        evid_ok = evidence_rank(observation.fallback_evidence_class) <= evidence_rank(
            observation.evidence_class
        )
    return auth_ok, evid_ok


def check_step_hard_properties(
    *,
    effect: ControlEffect,
    observation: ControllerObservation,
    bounds: ControllerBounds,
    require_human_gate_for_irreversible: bool = True,
) -> dict[str, bool]:
    """Evaluate hard properties against one proposed control step."""

    results: dict[str, bool] = {prop.value: True for prop in HardPropertyId}

    if observation.retry_count > bounds.max_retries:
        results[HardPropertyId.BOUNDED_RETRY.value] = False
    if observation.parallel_count > bounds.max_parallel:
        results[HardPropertyId.BOUNDED_PARALLELISM.value] = False

    if effect is ControlEffect.RETRY:
        irreversible = (
            observation.reversibility == ReversibilityClass.IRREVERSIBLE.value
        )
        if irreversible and (
            observation.unknown_pending or observation.effect_count > 0
        ):
            results[HardPropertyId.NO_BLIND_UNKNOWN_IRREVERSIBLE_RETRY.value] = False
        if observation.retry_count >= bounds.max_retries:
            results[HardPropertyId.BOUNDED_RETRY.value] = False

    if effect is ControlEffect.FALLBACK_PROVIDER:
        auth_ok, evid_ok = _fallback_preserves(observation)
        results[HardPropertyId.FALLBACK_PRESERVES_AUTHORITY.value] = auth_ok
        results[HardPropertyId.FALLBACK_PRESERVES_EVIDENCE_CLASS.value] = evid_ok

    if effect in {
        ControlEffect.ROUTE_PROVIDER,
        ControlEffect.FALLBACK_PROVIDER,
        ControlEffect.OBSERVE,
        ControlEffect.SEAL_RECEIPT,
    }:
        if not observation.lease_held and effect is not ControlEffect.SEAL_RECEIPT:
            # seal may occur after lease release only when already observed;
            # lease-before-effect still applies to route/fallback/observe.
            if effect in {
                ControlEffect.ROUTE_PROVIDER,
                ControlEffect.FALLBACK_PROVIDER,
                ControlEffect.OBSERVE,
            }:
                results[HardPropertyId.LEASE_BEFORE_EFFECT.value] = False

    if effect is ControlEffect.SEAL_RECEIPT and not observation.observed:
        # Compensating seal is allowed without prior success observation when
        # compensation path is explicit; still forbid success-without-observation.
        if observation.mode not in {"compensating", "unknown_pending"}:
            results[HardPropertyId.NO_SUCCESS_WITHOUT_OBSERVATION.value] = False

    if (
        require_human_gate_for_irreversible
        and observation.reversibility == ReversibilityClass.IRREVERSIBLE.value
        and effect
        in {
            ControlEffect.ROUTE_PROVIDER,
            ControlEffect.FALLBACK_PROVIDER,
        }
        and not (observation.confirmation_present and not observation.confirmation_spent)
    ):
        results[HardPropertyId.HUMAN_GATE_BEFORE_IRREVERSIBLE.value] = False

    if effect is ControlEffect.ESCALATE_PROOF:
        current = observation.proof_stage
        proposed = observation.proposed_proof_stage or ""
        try:
            _reject_prohibited_stage(proposed or current)
            current_stage = _enum(current, EscalationStage, "proof_stage")
            if not proposed:
                results[HardPropertyId.PROOF_ESCALATION_MONOTONE.value] = False
            else:
                _reject_prohibited_stage(proposed)
                next_stage = _enum(proposed, EscalationStage, "proposed_proof_stage")
                stronger = next_stronger_stage(current_stage)
                # Allow any strictly stronger admitted stage within the ladder.
                if next_stage.rank <= current_stage.rank:
                    results[HardPropertyId.PROOF_ESCALATION_MONOTONE.value] = False
                if next_stage.rank > bounds.max_escalation_rank:
                    results[HardPropertyId.PROOF_ESCALATION_MONOTONE.value] = False
                if stronger is None and next_stage is not EscalationStage.HUMAN:
                    results[HardPropertyId.PROOF_ESCALATION_MONOTONE.value] = False
        except ControllerError:
            results[HardPropertyId.PROOF_ESCALATION_MONOTONE.value] = False

    if observation.shutdown_latched and effect.value in EFFECTFUL_CONTROL:
        results[HardPropertyId.SAFE_SHUTDOWN.value] = False

    if observation.mode not in CONTROLLER_MODES:
        results[HardPropertyId.TYPESTATE_CLOSED.value] = False

    if (
        effect is ControlEffect.SEAL_RECEIPT
        and observation.mode == "failed"
        and observation.reversibility == ReversibilityClass.COMPENSATABLE.value
        and not observation.observed
    ):
        results[HardPropertyId.COMPENSATION_EXPLICIT.value] = False

    return results


def check_hard_properties(
    policy: ControllerPolicy,
    *,
    horizon: int | None = None,
    require_human_gate_for_irreversible: bool = True,
) -> dict[str, bool]:
    """Mechanically discharge hard properties over the policy grammar."""

    bounds = policy.bounds
    limit = horizon if horizon is not None else bounds.horizon
    results: dict[str, bool] = {
        prop.property_id.value: True for prop in policy.hard_properties
    }
    baseline_ids = {prop.property_id for prop in default_hard_properties(bounds)}
    policy_ids = {prop.property_id for prop in policy.hard_properties}
    if not baseline_ids.issubset(policy_ids):
        results[HardPropertyId.NO_WEAKEN_HARD_SAFETY.value] = False
    if any(prop.waivable for prop in policy.hard_properties):
        results[HardPropertyId.NO_WEAKEN_HARD_SAFETY.value] = False

    if bounds.max_retries < 0:
        results[HardPropertyId.BOUNDED_RETRY.value] = False
    if bounds.max_parallel < 1:
        results[HardPropertyId.BOUNDED_PARALLELISM.value] = False

    # Structural grammar checks within a finite exploration budget.
    explored = 0
    for transition in policy.transitions:
        explored += 1
        if explored > limit:
            break
        if transition.prior_mode not in CONTROLLER_MODES:
            results[HardPropertyId.TYPESTATE_CLOSED.value] = False
        if transition.next_mode not in CONTROLLER_MODES:
            results[HardPropertyId.TYPESTATE_CLOSED.value] = False
        if transition.effect.value not in CONTROL_EFFECTS:
            results[HardPropertyId.TYPESTATE_CLOSED.value] = False

        # Retry edges must carry a retry bound guard.
        if transition.effect is ControlEffect.RETRY:
            retry_guards = [
                guard
                for guard in transition.guards
                if guard.max_retries is not None or guard.forbid_unknown_pending
            ]
            if not retry_guards:
                results[HardPropertyId.BOUNDED_RETRY.value] = False
                results[HardPropertyId.NO_BLIND_UNKNOWN_IRREVERSIBLE_RETRY.value] = False
            else:
                for guard in retry_guards:
                    if guard.max_retries is not None and guard.max_retries > bounds.max_retries:
                        results[HardPropertyId.BOUNDED_RETRY.value] = False

        if transition.effect in {
            ControlEffect.ROUTE_PROVIDER,
            ControlEffect.FALLBACK_PROVIDER,
        }:
            parallel_ok = any(
                guard.max_parallel is not None or guard.require_lease
                for guard in transition.guards
            )
            if not parallel_ok:
                results[HardPropertyId.BOUNDED_PARALLELISM.value] = False
                results[HardPropertyId.LEASE_BEFORE_EFFECT.value] = False

        if transition.effect is ControlEffect.FALLBACK_PROVIDER:
            # Fallback edges are admitted only when authority/evidence
            # preservation is part of the hard set (checked at step time).
            if HardPropertyId.FALLBACK_PRESERVES_AUTHORITY not in policy_ids:
                results[HardPropertyId.FALLBACK_PRESERVES_AUTHORITY.value] = False
            if HardPropertyId.FALLBACK_PRESERVES_EVIDENCE_CLASS not in policy_ids:
                results[HardPropertyId.FALLBACK_PRESERVES_EVIDENCE_CLASS.value] = False

        if transition.effect is ControlEffect.SEAL_RECEIPT:
            if transition.next_mode == "terminal_success":
                # Success seal must come from an observation-aware mode.
                if transition.prior_mode not in {
                    "awaiting_observation",
                    "routing",
                }:
                    results[HardPropertyId.NO_SUCCESS_WITHOUT_OBSERVATION.value] = False

        if transition.effect is ControlEffect.SHUTDOWN:
            if transition.next_mode not in {"shutting_down", "terminal_shutdown"}:
                results[HardPropertyId.SAFE_SHUTDOWN.value] = False

        if (
            require_human_gate_for_irreversible
            and transition.effect is ControlEffect.ROUTE_PROVIDER
            and any(
                ReversibilityClass.IRREVERSIBLE.value in guard.allowed_reversibility
                for guard in transition.guards
            )
        ):
            if not any(guard.require_confirmation for guard in transition.guards):
                results[HardPropertyId.HUMAN_GATE_BEFORE_IRREVERSIBLE.value] = False

    # Liveness under healthy assumptions: some path to a terminal mode exists.
    reachable = {policy.initial_mode}
    changed = True
    steps = 0
    while changed and steps < limit:
        changed = False
        steps += 1
        for transition in policy.transitions:
            if transition.prior_mode in reachable and transition.next_mode not in reachable:
                reachable.add(transition.next_mode)
                changed = True
    if reachable.isdisjoint(TERMINAL_MODES):
        # Soft liveness failure does not flip hard safety, but SafeShutdown /
        # TypestateClosed still require a shutdown terminal when shutdown exists.
        if ControlEffect.SHUTDOWN.value in policy.effects():
            if "terminal_shutdown" not in reachable:
                results[HardPropertyId.SAFE_SHUTDOWN.value] = False

    # Align with TEP required invariant names for cross-module evidence.
    for tep_name in TEP_REQUIRED_INVARIANTS:
        if tep_name == "NoBlindUnknownRetry":
            if not results.get(
                HardPropertyId.NO_BLIND_UNKNOWN_IRREVERSIBLE_RETRY.value, True
            ):
                results[HardPropertyId.NO_BLIND_UNKNOWN_IRREVERSIBLE_RETRY.value] = False
        if tep_name == "NoSuccessWithoutObservation":
            if not results.get(HardPropertyId.NO_SUCCESS_WITHOUT_OBSERVATION.value, True):
                results[HardPropertyId.NO_SUCCESS_WITHOUT_OBSERVATION.value] = False

    return results


def _soft_scores(policy: ControllerPolicy) -> dict[str, int]:
    scores: dict[str, int] = {}
    effect_count = len(policy.transitions)
    for objective in policy.soft_objectives or default_soft_objectives():
        if objective.objective_id is SoftObjectiveId.MINIMIZE_COST:
            scores[objective.objective_id.value] = max(0, 100 - effect_count)
        elif objective.objective_id is SoftObjectiveId.PREFER_CHEAPEST_SOUND_STAGE:
            scores[objective.objective_id.value] = EscalationStage.SCHEMA.default_cost
        else:
            scores[objective.objective_id.value] = objective.weight
    return scores


# ---------------------------------------------------------------------------
# Unrealizability analysis
# ---------------------------------------------------------------------------


def explain_unrealizable(spec: ControllerSpec) -> UnrealizableCore:
    """Return a minimal explanatory core for contradictory requirements."""

    conflicts: list[str] = []
    properties: list[str] = []
    path: list[str] = []

    if spec.allow_unbounded_retry:
        conflicts.append("spec.allow_unbounded_retry")
        conflicts.append("hard.BoundedRetry")
        properties.append(HardPropertyId.BOUNDED_RETRY.value)
        path.append("unbounded retry conflicts with BoundedRetry")

    if spec.require_fallback_authority_promotion:
        conflicts.append("spec.require_fallback_authority_promotion")
        conflicts.append("hard.FallbackPreservesAuthority")
        properties.append(HardPropertyId.FALLBACK_PRESERVES_AUTHORITY.value)
        path.append("fallback authority promotion conflicts with FallbackPreservesAuthority")

    if spec.require_fallback_evidence_promotion:
        conflicts.append("spec.require_fallback_evidence_promotion")
        conflicts.append("hard.FallbackPreservesEvidenceClass")
        properties.append(HardPropertyId.FALLBACK_PRESERVES_EVIDENCE_CLASS.value)
        path.append(
            "fallback evidence promotion conflicts with FallbackPreservesEvidenceClass"
        )

    overlap = set(spec.required_effects) & set(spec.forbidden_effects)
    if overlap:
        for effect in sorted(overlap):
            conflicts.append(f"required_effect:{effect}")
            conflicts.append(f"forbidden_effect:{effect}")
            path.append(f"effect {effect} is both required and forbidden")

    # Missing hard properties relative to baseline is itself unrealizable vs
    # NoWeakenHardSafety (caller asked to weaken).
    baseline = {prop.property_id for prop in default_hard_properties(spec.bounds)}
    present = {prop.property_id for prop in spec.hard_properties}
    missing = baseline - present
    if missing:
        conflicts.append("spec.hard_properties")
        conflicts.append("hard.NoWeakenHardSafety")
        properties.append(HardPropertyId.NO_WEAKEN_HARD_SAFETY.value)
        path.append(
            "spec omits baseline hard properties: "
            + ", ".join(sorted(item.value for item in missing))
        )

    if any(prop.waivable for prop in spec.hard_properties):
        conflicts.append("spec.waivable_hard_property")
        conflicts.append("hard.NoWeakenHardSafety")
        properties.append(HardPropertyId.NO_WEAKEN_HARD_SAFETY.value)
        path.append("waivable hard property is forbidden")

    if not conflicts:
        conflicts.append("spec.unknown")
        path.append("no realizable controller grammar satisfies the specification")

    explanation = (
        "Controller specification is unrealizable under hard properties: "
        + "; ".join(path)
    )
    return UnrealizableCore(
        core_id="provisional",
        conflicting_requirements=tuple(conflicts),
        conflicting_properties=tuple(properties),
        path=tuple(path),
        explanation=explanation,
    )


def _spec_is_unrealizable(spec: ControllerSpec) -> bool:
    if spec.allow_unbounded_retry:
        return True
    if spec.require_fallback_authority_promotion:
        return True
    if spec.require_fallback_evidence_promotion:
        return True
    if set(spec.required_effects) & set(spec.forbidden_effects):
        return True
    baseline = {prop.property_id for prop in default_hard_properties(spec.bounds)}
    present = {prop.property_id for prop in spec.hard_properties}
    if not baseline.issubset(present):
        return True
    if any(prop.waivable for prop in spec.hard_properties):
        return True
    return False


# ---------------------------------------------------------------------------
# Synthesis / validation
# ---------------------------------------------------------------------------


def _coerce_spec(spec: ControllerSpec | Mapping[str, Any]) -> ControllerSpec:
    if isinstance(spec, ControllerSpec):
        return spec
    return ControllerSpec.from_mapping(spec)


def _coerce_policy(policy: ControllerPolicy | Mapping[str, Any]) -> ControllerPolicy:
    if isinstance(policy, ControllerPolicy):
        return policy
    return ControllerPolicy.from_mapping(policy)


def synthesize_controller(
    spec: ControllerSpec | Mapping[str, Any],
    *,
    bounds: ControllerBounds | None = None,
    toolchain: str = TOOLCHAIN_ID,
) -> ControllerResult:
    """Instantiate the fixed controller grammar and discharge hard properties."""

    try:
        parsed = _coerce_spec(spec)
    except ControllerError as exc:
        return ControllerResult(
            verdict=ControllerVerdict.INVALID_SPEC,
            assumptions=(),
            toolchain=toolchain,
            bounds=bounds or ControllerBounds(),
            hard_property_results={},
            reason=str(exc),
        )

    effective_bounds = bounds or parsed.bounds
    if bounds is not None and bounds != parsed.bounds:
        # Caller override wins for checked bounds.
        parsed = ControllerSpec(
            spec_id=parsed.spec_id,
            assumptions=parsed.assumptions,
            hard_properties=parsed.hard_properties,
            soft_objectives=parsed.soft_objectives,
            required_effects=parsed.required_effects,
            forbidden_effects=parsed.forbidden_effects,
            bounds=effective_bounds,
            require_human_gate_for_irreversible=parsed.require_human_gate_for_irreversible,
            allow_unbounded_retry=parsed.allow_unbounded_retry,
            require_fallback_authority_promotion=parsed.require_fallback_authority_promotion,
            require_fallback_evidence_promotion=parsed.require_fallback_evidence_promotion,
            required_contract_ids=parsed.required_contract_ids,
            schema=parsed.schema,
        )

    if _spec_is_unrealizable(parsed):
        core = explain_unrealizable(parsed)
        return ControllerResult(
            verdict=ControllerVerdict.UNREALIZABLE,
            assumptions=parsed.assumptions,
            toolchain=toolchain,
            bounds=effective_bounds,
            hard_property_results={
                prop.property_id.value: False
                for prop in parsed.hard_properties
                if prop.property_id.value in core.conflicting_properties
            }
            or {HardPropertyId.NO_WEAKEN_HARD_SAFETY.value: False},
            unrealizable_core=core,
            reason=core.explanation,
            evidence=(SCHEMA, REACTIVE_EVIDENCE, CORE_SCHEMA),
        )

    transitions = [
        transition
        for transition in build_default_grammar(
            effective_bounds,
            require_human_gate_for_irreversible=parsed.require_human_gate_for_irreversible,
        )
        if transition.effect.value not in parsed.forbidden_effects
    ]
    # Ensure required effects remain present after filtering.
    present_effects = {transition.effect.value for transition in transitions}
    missing_required = [
        effect for effect in parsed.required_effects if effect not in present_effects
    ]
    if missing_required:
        core = UnrealizableCore(
            core_id="provisional",
            conflicting_requirements=tuple(
                f"required_effect:{effect}" for effect in missing_required
            )
            + ("grammar:default",),
            conflicting_properties=(HardPropertyId.TYPESTATE_CLOSED.value,),
            path=tuple(
                f"required effect {effect} is absent from bounded grammar"
                for effect in missing_required
            ),
            explanation=(
                "Required effects cannot be realized by the bounded controller grammar: "
                + ", ".join(missing_required)
            ),
        )
        return ControllerResult(
            verdict=ControllerVerdict.UNREALIZABLE,
            assumptions=parsed.assumptions,
            toolchain=toolchain,
            bounds=effective_bounds,
            hard_property_results={HardPropertyId.TYPESTATE_CLOSED.value: False},
            unrealizable_core=core,
            reason=core.explanation,
            evidence=(SCHEMA, REACTIVE_EVIDENCE, CORE_SCHEMA),
        )

    hard = parsed.hard_properties or default_hard_properties(effective_bounds)
    policy = ControllerPolicy(
        policy_id="provisional",
        initial_mode="idle",
        transitions=tuple(transitions),
        bounds=effective_bounds,
        hard_properties=hard,
        soft_objectives=parsed.soft_objectives,
        assumptions=parsed.assumptions,
    )
    property_results = check_hard_properties(
        policy,
        require_human_gate_for_irreversible=parsed.require_human_gate_for_irreversible,
    )
    if not all(property_results.values()):
        failed = [name for name, ok in property_results.items() if not ok]
        core = UnrealizableCore(
            core_id="provisional",
            conflicting_requirements=tuple(f"hard.{name}" for name in failed),
            conflicting_properties=tuple(failed),
            path=tuple(f"hard property failed: {name}" for name in failed),
            explanation=(
                "Synthesized grammar failed hard-property discharge: "
                + ", ".join(failed)
            ),
        )
        return ControllerResult(
            verdict=ControllerVerdict.UNREALIZABLE,
            assumptions=parsed.assumptions,
            toolchain=toolchain,
            bounds=effective_bounds,
            hard_property_results=property_results,
            unrealizable_core=core,
            reason=core.explanation,
            evidence=(SCHEMA, REACTIVE_EVIDENCE, CORE_SCHEMA),
        )

    discharged = tuple(
        name for name, ok in property_results.items() if ok
    )
    policy = ControllerPolicy(
        policy_id="provisional",
        initial_mode=policy.initial_mode,
        transitions=policy.transitions,
        bounds=policy.bounds,
        hard_properties=policy.hard_properties,
        soft_objectives=policy.soft_objectives,
        discharged_properties=discharged,
        assumptions=policy.assumptions,
    )
    return ControllerResult(
        verdict=ControllerVerdict.REALIZED,
        assumptions=parsed.assumptions,
        toolchain=toolchain,
        bounds=effective_bounds,
        hard_property_results=property_results,
        policy=policy,
        reason="bounded controller grammar realizes the specification",
        soft_scores=_soft_scores(policy),
    )


def validate_controller(
    spec: ControllerSpec | Mapping[str, Any],
    policy: ControllerPolicy | Mapping[str, Any],
    *,
    bounds: ControllerBounds | None = None,
    toolchain: str = TOOLCHAIN_ID,
) -> ControllerResult:
    """Validate a caller-supplied policy against hard properties and the spec."""

    try:
        parsed_spec = _coerce_spec(spec)
        parsed_policy = _coerce_policy(policy)
    except ControllerError as exc:
        return ControllerResult(
            verdict=ControllerVerdict.INVALID_SPEC,
            assumptions=(),
            toolchain=toolchain,
            bounds=bounds or ControllerBounds(),
            hard_property_results={},
            reason=str(exc),
        )

    effective_bounds = bounds or parsed_spec.bounds
    if _spec_is_unrealizable(parsed_spec):
        core = explain_unrealizable(parsed_spec)
        return ControllerResult(
            verdict=ControllerVerdict.UNREALIZABLE,
            assumptions=parsed_spec.assumptions,
            toolchain=toolchain,
            bounds=effective_bounds,
            hard_property_results={
                name: False for name in core.conflicting_properties
            }
            or {HardPropertyId.NO_WEAKEN_HARD_SAFETY.value: False},
            unrealizable_core=core,
            reason=core.explanation,
            evidence=(SCHEMA, REACTIVE_EVIDENCE, CORE_SCHEMA),
        )

    # Reject policies that drop baseline hard properties.
    baseline = {prop.property_id for prop in default_hard_properties(effective_bounds)}
    present = {prop.property_id for prop in parsed_policy.hard_properties}
    if not baseline.issubset(present):
        return ControllerResult(
            verdict=ControllerVerdict.REJECTED,
            assumptions=parsed_spec.assumptions,
            toolchain=toolchain,
            bounds=effective_bounds,
            hard_property_results={HardPropertyId.NO_WEAKEN_HARD_SAFETY.value: False},
            policy=parsed_policy,
            reason="policy weakens or omits baseline hard properties",
        )

    for effect in parsed_policy.effects():
        if effect in parsed_spec.forbidden_effects:
            return ControllerResult(
                verdict=ControllerVerdict.REJECTED,
                assumptions=parsed_spec.assumptions,
                toolchain=toolchain,
                bounds=effective_bounds,
                hard_property_results={},
                policy=parsed_policy,
                reason=f"policy includes forbidden effect {effect}",
            )
        if effect not in CONTROL_EFFECTS:
            return ControllerResult(
                verdict=ControllerVerdict.REJECTED,
                assumptions=parsed_spec.assumptions,
                toolchain=toolchain,
                bounds=effective_bounds,
                hard_property_results={HardPropertyId.TYPESTATE_CLOSED.value: False},
                policy=parsed_policy,
                reason=f"policy includes unknown effect {effect}",
            )

    missing_required = [
        effect
        for effect in parsed_spec.required_effects
        if effect not in parsed_policy.effects()
    ]
    if missing_required:
        return ControllerResult(
            verdict=ControllerVerdict.REJECTED,
            assumptions=parsed_spec.assumptions,
            toolchain=toolchain,
            bounds=effective_bounds,
            hard_property_results={},
            policy=parsed_policy,
            reason="policy missing required effects: " + ", ".join(missing_required),
        )

    # Soft objectives never override: if policy tries cheaper provider via
    # authority/evidence promotion edges without preservation props, reject.
    property_results = check_hard_properties(
        parsed_policy,
        require_human_gate_for_irreversible=parsed_spec.require_human_gate_for_irreversible,
    )
    if not all(property_results.values()):
        failed = [name for name, ok in property_results.items() if not ok]
        return ControllerResult(
            verdict=ControllerVerdict.REJECTED,
            assumptions=parsed_spec.assumptions,
            toolchain=toolchain,
            bounds=effective_bounds,
            hard_property_results=property_results,
            policy=parsed_policy,
            reason="hard property violations: " + ", ".join(failed),
        )

    discharged = tuple(name for name, ok in property_results.items() if ok)
    accepted = ControllerPolicy(
        policy_id=parsed_policy.policy_id,
        initial_mode=parsed_policy.initial_mode,
        transitions=parsed_policy.transitions,
        bounds=parsed_policy.bounds,
        hard_properties=parsed_policy.hard_properties,
        soft_objectives=parsed_policy.soft_objectives,
        discharged_properties=discharged,
        assumptions=parsed_policy.assumptions or parsed_spec.assumptions,
        evidence=parsed_policy.evidence,
    )
    return ControllerResult(
        verdict=ControllerVerdict.REALIZED,
        assumptions=parsed_spec.assumptions,
        toolchain=toolchain,
        bounds=effective_bounds,
        hard_property_results=property_results,
        policy=accepted,
        reason="policy validates against hard properties and specification",
        soft_scores=_soft_scores(accepted),
    )


def synthesize_or_validate(
    *,
    mode: ControllerMode | str,
    spec: ControllerSpec | Mapping[str, Any],
    policy: ControllerPolicy | Mapping[str, Any] | None = None,
    bounds: ControllerBounds | None = None,
) -> ControllerResult:
    """Unified entry for synthesize / validate modes."""

    selected = _enum(mode, ControllerMode, "mode")
    if selected is ControllerMode.SYNTHESIZE:
        return synthesize_controller(spec, bounds=bounds)
    if policy is None:
        return ControllerResult(
            verdict=ControllerVerdict.INVALID_SPEC,
            assumptions=(),
            toolchain=TOOLCHAIN_ID,
            bounds=bounds or ControllerBounds(),
            hard_property_results={},
            reason="validate mode requires a policy",
        )
    return validate_controller(spec, policy, bounds=bounds)


# ---------------------------------------------------------------------------
# Runtime monitor facade
# ---------------------------------------------------------------------------


class FormalAssuranceController:
    """Bounded supervisor controller synthesizer, validator, and monitor."""

    def __init__(
        self,
        *,
        bounds: ControllerBounds | None = None,
        toolchain: str = TOOLCHAIN_ID,
    ) -> None:
        self._bounds = bounds or ControllerBounds()
        self._toolchain = toolchain

    @property
    def bounds(self) -> ControllerBounds:
        return self._bounds

    def synthesize(
        self, spec: ControllerSpec | Mapping[str, Any]
    ) -> ControllerResult:
        return synthesize_controller(
            spec, bounds=self._bounds, toolchain=self._toolchain
        )

    def validate(
        self,
        spec: ControllerSpec | Mapping[str, Any],
        policy: ControllerPolicy | Mapping[str, Any],
    ) -> ControllerResult:
        return validate_controller(
            spec, policy, bounds=self._bounds, toolchain=self._toolchain
        )

    def explain_unrealizable(
        self, spec: ControllerSpec | Mapping[str, Any]
    ) -> UnrealizableCore:
        return explain_unrealizable(_coerce_spec(spec))

    def check_hard_properties(
        self,
        policy: ControllerPolicy | Mapping[str, Any],
        *,
        horizon: int | None = None,
        require_human_gate_for_irreversible: bool = True,
    ) -> dict[str, bool]:
        return check_hard_properties(
            _coerce_policy(policy),
            horizon=horizon,
            require_human_gate_for_irreversible=require_human_gate_for_irreversible,
        )

    def step(
        self,
        policy: ControllerPolicy | Mapping[str, Any],
        *,
        mode: str,
        observation: Mapping[str, Any] | ControllerObservation | None = None,
        proposed_effect: ControlEffect | str,
    ) -> ControllerMonitorVerdict:
        """Interpret one proposed control effect against the policy grammar."""

        parsed_policy = _coerce_policy(policy)
        effect = _enum(proposed_effect, ControlEffect, "proposed_effect")
        if isinstance(observation, ControllerObservation):
            obs = observation
        else:
            obs = ControllerObservation.from_mapping(observation)
        prior = _text(mode, "mode")
        if prior not in CONTROLLER_MODES:
            return ControllerMonitorVerdict(
                accepted=False,
                code=ControllerErrorCode.UNKNOWN_MODE.value,
                prior_mode=prior,
                next_mode=prior,
                effect=effect.value,
                message=f"unknown mode {prior}",
                invariant=HardPropertyId.TYPESTATE_CLOSED.value,
            )

        obs.mode = prior
        hard = check_step_hard_properties(
            effect=effect,
            observation=obs,
            bounds=parsed_policy.bounds,
        )
        # Soft objectives never override hard failures.
        failed_hard = [name for name, ok in hard.items() if not ok]
        if failed_hard:
            code = ControllerErrorCode.HARD_PROPERTY_VIOLATION.value
            if HardPropertyId.NO_BLIND_UNKNOWN_IRREVERSIBLE_RETRY.value in failed_hard:
                code = ControllerErrorCode.HARD_PROPERTY_VIOLATION.value
            if (
                HardPropertyId.FALLBACK_PRESERVES_AUTHORITY.value in failed_hard
                or HardPropertyId.FALLBACK_PRESERVES_EVIDENCE_CLASS.value in failed_hard
            ):
                code = ControllerErrorCode.FALLBACK_PROMOTION.value
            if HardPropertyId.SAFE_SHUTDOWN.value in failed_hard:
                code = ControllerErrorCode.SHUTDOWN_LATCHED.value
            if HardPropertyId.BOUNDED_RETRY.value in failed_hard or (
                HardPropertyId.BOUNDED_PARALLELISM.value in failed_hard
            ):
                code = ControllerErrorCode.BOUND_EXCEEDED.value
            return ControllerMonitorVerdict(
                accepted=False,
                code=code,
                prior_mode=prior,
                next_mode=prior,
                effect=effect.value,
                message="hard property violation: " + ", ".join(failed_hard),
                invariant=failed_hard[0],
                hard_properties=hard,
            )

        matches = [
            transition
            for transition in parsed_policy.transitions
            if transition.prior_mode == prior and transition.effect is effect
        ]
        if not matches:
            return ControllerMonitorVerdict(
                accepted=False,
                code=ControllerErrorCode.PRESTATE_MISMATCH.value,
                prior_mode=prior,
                next_mode=prior,
                effect=effect.value,
                message="no matching policy transition",
                hard_properties=hard,
            )

        # Pick the first guard-satisfying transition (grammar is deterministic).
        for transition in matches:
            guard_failed: str | None = None
            for guard in transition.guards:
                ok, reason = evaluate_guard(
                    guard, obs, bounds=parsed_policy.bounds
                )
                if not ok:
                    guard_failed = reason
                    break
            if guard_failed is not None:
                continue
            return ControllerMonitorVerdict(
                accepted=True,
                code="accepted",
                prior_mode=prior,
                next_mode=transition.next_mode,
                effect=effect.value,
                message="transition accepted",
                hard_properties=hard,
            )

        return ControllerMonitorVerdict(
            accepted=False,
            code=ControllerErrorCode.GUARD_FAILED.value,
            prior_mode=prior,
            next_mode=prior,
            effect=effect.value,
            message="guards failed for all matching transitions",
            hard_properties=hard,
        )


def default_controller(bounds: ControllerBounds | None = None) -> FormalAssuranceController:
    return FormalAssuranceController(bounds=bounds)


def default_realizable_spec(
    *,
    spec_id: str = "spec:facp052-default",
    bounds: ControllerBounds | None = None,
) -> ControllerSpec:
    b = bounds or ControllerBounds()
    return ControllerSpec(
        spec_id=spec_id,
        assumptions=(
            "assumption:deps-healthy",
            "assumption:lease-available",
            "assumption:confirmation-available",
        ),
        hard_properties=default_hard_properties(b),
        soft_objectives=default_soft_objectives(),
        required_effects=(
            "route_provider",
            "retry",
            "acquire_lease",
            "human_gate",
            "escalate_proof",
            "compensate",
            "shutdown",
            "fallback_provider",
        ),
        bounds=b,
        require_human_gate_for_irreversible=True,
    )


__all__ = [
    "ANALYZER_VERSION",
    "AuthorityClass",
    "BUNDLE",
    "CONTROL_EFFECTS",
    "CONTROLLER_MODES",
    "CORE_SCHEMA",
    "ControlEffect",
    "ControllerBounds",
    "ControllerError",
    "ControllerErrorCode",
    "ControllerGuard",
    "ControllerMode",
    "ControllerMonitorVerdict",
    "ControllerObservation",
    "ControllerPolicy",
    "ControllerResult",
    "ControllerSpec",
    "ControllerTransition",
    "ControllerVerdict",
    "DEFAULT_HORIZON",
    "DEFAULT_MAX_PARALLEL",
    "EVIDENCE_SUBSET",
    "EvidenceClass",
    "FormalAssuranceController",
    "GOAL_ID",
    "GUARD_SCHEMA",
    "HardProperty",
    "HardPropertyId",
    "INTERFACE",
    "MONITOR_SCHEMA",
    "NORMATIVE_PINS",
    "POLICY_SCHEMA",
    "REACTIVE_EVIDENCE",
    "RESULT_SCHEMA",
    "ReversibilityClass",
    "SCHEMA",
    "SPEC_SCHEMA",
    "SoftObjective",
    "SoftObjectiveId",
    "TASK_ID",
    "TOOLCHAIN_ID",
    "UnrealizableCore",
    "authority_rank",
    "build_default_grammar",
    "check_hard_properties",
    "check_step_hard_properties",
    "default_controller",
    "default_hard_properties",
    "default_realizable_spec",
    "default_soft_objectives",
    "evidence_rank",
    "explain_unrealizable",
    "evaluate_guard",
    "synthesize_controller",
    "synthesize_or_validate",
    "validate_controller",
    "escalation_ladder",
]
