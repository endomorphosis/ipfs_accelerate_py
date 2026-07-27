"""Fail-closed execution of incident-bound rescue proposals.

``rescue_planner`` deliberately stops at a proposal.  This module is the
trust-preserving bridge from that proposal to the existing permit and control
transaction boundaries.  It has no shell, subprocess, file-write, or provider
surface.  A caller must supply:

* a source of current authoritative runtime state,
* five independent domain authorizers (IntentIR, LegalIR, SecurityIR, proof,
  and control),
* an execution-permit boundary which issues and consumes one-use permits, and
* a control-transaction adapter which is the only component allowed to apply
  an effect.

Every action is rebound, simulated, authorized, permitted, and checked again
immediately before dispatch.  The returned content-addressed receipts are
intentionally more useful than exceptions: denials, drift, partial effects,
budget exhaustion, and quarantine all have exact recovery guidance.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Final,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from ..control_contracts import (
    EffectClaim,
    Operation,
    OperationRequest,
    OperationResult,
    OperationStatus,
)
from ..prompt_workflow import (
    ProgrammaticRecoveryExhaustionReceipt,
    RecordStatus,
    RescueAction,
    RescueOperation,
    RescuePlan,
    SupervisorIncident,
)
from ..rescue_planner import (
    DEFAULT_RESCUE_OPERATION_CATALOG,
    RescueOperationSpec,
    RescuePlannerValidationError,
)


RESCUE_ORCHESTRATION_REQUIREMENT_ID: Final[str] = (
    "ASI-157:bounded-rescue-one-exact-action-per-permit"
)
RESCUE_ROOT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-root-binding@1"
)
RESCUE_RUNTIME_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-runtime-snapshot@1"
)
RESCUE_ACTION_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-action-binding@1"
)
RESCUE_AUTHORIZATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-authorization-receipt@1"
)
RESCUE_SIMULATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-simulation-receipt@1"
)
RESCUE_PERMIT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-permit-use-receipt@1"
)
RESCUE_HEALTH_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-health-receipt@1"
)
RESCUE_ACTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-action-execution-receipt@1"
)
RESCUE_RUN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rescue-run-receipt@1"
)

ABSOLUTE_MAX_RESCUE_ACTIONS: Final[int] = 32
ABSOLUTE_MAX_RESCUE_TIME_MS: Final[int] = 10 * 60 * 1_000
ABSOLUTE_MAX_PERMIT_TTL_MS: Final[int] = 60_000
ABSOLUTE_MAX_MODEL_TOKENS: Final[int] = 1_000_000
ABSOLUTE_MAX_MODEL_COST_MICROUNITS: Final[int] = 1_000_000_000


class RescueOrchestrationError(RuntimeError):
    """Invalid orchestration configuration or an untyped dependency result."""


class RescueAuthorizationDomain(str, Enum):
    """Independent authorities which must all admit an exact action."""

    INTENT = "intent"
    LEGAL = "legal"
    SECURITY = "security"
    PROOF = "proof"
    CONTROL = "control"


REQUIRED_AUTHORIZATION_DOMAINS: Final[Tuple[RescueAuthorizationDomain, ...]] = (
    RescueAuthorizationDomain.INTENT,
    RescueAuthorizationDomain.LEGAL,
    RescueAuthorizationDomain.SECURITY,
    RescueAuthorizationDomain.PROOF,
    RescueAuthorizationDomain.CONTROL,
)


class RescueAuthorizationVerdict(str, Enum):
    PERMIT = "permit"
    DENY = "deny"
    UNKNOWN = "unknown"


class RescueHealthState(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    QUARANTINED = "quarantined"


class RescueStopReason(str, Enum):
    """Closed terminal vocabulary for action and run receipts."""

    ACTION_APPLIED = "action_applied"
    HEALTH_RESTORED = "health_restored"
    ALREADY_HEALTHY = "already_healthy"
    ROOT_DRIFT = "root_drift"
    INCIDENT_DRIFT = "incident_drift"
    EXHAUSTION_DRIFT = "exhaustion_drift"
    LEASE_LOST = "lease_lost"
    FENCE_LOST = "fence_lost"
    IDEMPOTENCY_REPLAY = "idempotency_replay"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    COOLDOWN_ACTIVE = "cooldown_active"
    ACTION_BUDGET = "action_budget"
    MODEL_BUDGET = "model_budget"
    TIME_BUDGET = "time_budget"
    SCHEMA_DENIED = "schema_denied"
    SIMULATION_DENIED = "simulation_denied"
    AUTHORIZATION_DENIED = "authorization_denied"
    PERMIT_DENIED = "permit_denied"
    UNEXPECTED_EFFECT = "unexpected_effect"
    PARTIAL_EFFECT = "partial_effect"
    HEALTH_TEST_FAILED = "health_test_failed"
    QUARANTINED = "quarantined"
    CONTROL_DENIED = "control_denied"
    CONTROL_FAILED = "control_failed"


class RescueReceiptDisposition(str, Enum):
    APPLIED = "applied"
    RECOVERED = "recovered"
    STOPPED = "stopped"
    DENIED = "denied"
    PARTIAL = "partial"
    QUARANTINED = "quarantined"


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise RescueOrchestrationError("floating point values are not canonical")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise RescueOrchestrationError("mapping keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_canonical(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _canonical(converter())
    converter = getattr(value, "to_record", None)
    if callable(converter):
        return _canonical(converter())
    raise RescueOrchestrationError(
        "unsupported canonical value: " + type(value).__name__
    )


def _freeze(value: Any) -> Any:
    plain = _canonical(value)
    if isinstance(plain, dict):
        return MappingProxyType(
            {key: _freeze(item) for key, item in plain.items()}
        )
    if isinstance(plain, list):
        return tuple(_freeze(item) for item in plain)
    return plain


def _content_id(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise RescueOrchestrationError(name + " must be non-empty canonical text")
    if "\x00" in value:
        raise RescueOrchestrationError(name + " must not contain NUL")
    return value


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RescueOrchestrationError(name + " must be a non-negative integer")
    return value


def _positive(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if result == 0:
        raise RescueOrchestrationError(name + " must be positive")
    return result


def _strings(values: Sequence[str], name: str) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RescueOrchestrationError(name + " must be a sequence")
    result = tuple(_text(item, name) for item in values)
    if len(result) != len(set(result)):
        raise RescueOrchestrationError(name + " must not contain duplicates")
    return result


@dataclass(frozen=True)
class RescueRootBinding:
    """All authority-changing roots rebound for one rescue plan."""

    incident_cid: str
    exhaustion_receipt_cid: str
    request_root: str
    program_root: str
    repository_root_cid: str
    tree_id: str
    run_cid: str
    intent_ir_root: str
    legal_ir_root: str
    security_ir_root: str
    policy_root: str
    catalog_root: str

    def __post_init__(self) -> None:
        for name in (
            "incident_cid",
            "exhaustion_receipt_cid",
            "request_root",
            "program_root",
            "repository_root_cid",
            "tree_id",
            "run_cid",
            "intent_ir_root",
            "legal_ir_root",
            "security_ir_root",
            "policy_root",
            "catalog_root",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_ROOT_BINDING_SCHEMA,
            "incident_cid": self.incident_cid,
            "exhaustion_receipt_cid": self.exhaustion_receipt_cid,
            "request_root": self.request_root,
            "program_root": self.program_root,
            "repository_root_cid": self.repository_root_cid,
            "tree_id": self.tree_id,
            "run_cid": self.run_cid,
            "intent_ir_root": self.intent_ir_root,
            "legal_ir_root": self.legal_ir_root,
            "security_ir_root": self.security_ir_root,
            "policy_root": self.policy_root,
            "catalog_root": self.catalog_root,
        }

    @property
    def content_id(self) -> str:
        return _content_id(self.to_dict())


@dataclass(frozen=True)
class RescueRuntimeSnapshot:
    """Authoritative state sampled immediately before a trust transition."""

    roots: RescueRootBinding
    lease_id: str
    fencing_epoch: int
    cooldown_until_ms: int = 0
    quarantined: bool = False
    revision: int = 0
    observed_at_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.roots, RescueRootBinding):
            raise RescueOrchestrationError("roots must be RescueRootBinding")
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        for name in (
            "fencing_epoch",
            "cooldown_until_ms",
            "revision",
            "observed_at_ms",
        ):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        if not isinstance(self.quarantined, bool):
            raise RescueOrchestrationError("quarantined must be boolean")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_RUNTIME_SNAPSHOT_SCHEMA,
            "roots": self.roots.to_dict(),
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "cooldown_until_ms": self.cooldown_until_ms,
            "quarantined": self.quarantined,
            "revision": self.revision,
            "observed_at_ms": self.observed_at_ms,
        }

    @property
    def content_id(self) -> str:
        return _content_id(self.to_dict())


@dataclass(frozen=True)
class RescueExecutionBudget:
    """Non-renewable bounds for one plan execution."""

    max_actions: int = 4
    max_model_actions: int = 4
    max_elapsed_ms: int = 120_000
    max_model_tokens: int = 8_192
    max_model_cost_microunits: int = 50_000
    permit_ttl_ms: int = 30_000

    def __post_init__(self) -> None:
        for name in (
            "max_actions",
            "max_model_actions",
            "max_elapsed_ms",
            "max_model_tokens",
            "max_model_cost_microunits",
            "permit_ttl_ms",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.max_actions > ABSOLUTE_MAX_RESCUE_ACTIONS:
            raise RescueOrchestrationError("max_actions exceeds absolute bound")
        if self.max_model_actions > ABSOLUTE_MAX_RESCUE_ACTIONS:
            raise RescueOrchestrationError(
                "max_model_actions exceeds absolute bound"
            )
        if self.max_elapsed_ms > ABSOLUTE_MAX_RESCUE_TIME_MS:
            raise RescueOrchestrationError(
                "max_elapsed_ms exceeds absolute bound"
            )
        if self.permit_ttl_ms > ABSOLUTE_MAX_PERMIT_TTL_MS:
            raise RescueOrchestrationError(
                "permit_ttl_ms exceeds short-lived absolute bound"
            )
        if self.max_model_tokens > ABSOLUTE_MAX_MODEL_TOKENS:
            raise RescueOrchestrationError(
                "max_model_tokens exceeds absolute bound"
            )
        if (
            self.max_model_cost_microunits
            > ABSOLUTE_MAX_MODEL_COST_MICROUNITS
        ):
            raise RescueOrchestrationError(
                "max_model_cost_microunits exceeds absolute bound"
            )


@dataclass(frozen=True)
class RescueExecutionRequest:
    """Current caller-owned facts for one proposed plan."""

    plan: RescuePlan
    incident: SupervisorIncident
    exhaustion_receipt: ProgrammaticRecoveryExhaustionReceipt
    roots: RescueRootBinding
    lease_id: str
    fencing_epoch: int
    idempotency_scope: str
    rescue_plan_root: str = ""
    budget: RescueExecutionBudget = field(
        default_factory=RescueExecutionBudget
    )
    model_tokens: int = 0
    model_cost_microunits: int = 0
    start_action_index: int = 0
    caller: str = "agent-supervisor:rescue-orchestrator"
    control_request: Optional[OperationRequest] = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, RescuePlan):
            raise RescueOrchestrationError("plan must be RescuePlan")
        if not isinstance(self.incident, SupervisorIncident):
            raise RescueOrchestrationError("incident must be SupervisorIncident")
        if not isinstance(
            self.exhaustion_receipt, ProgrammaticRecoveryExhaustionReceipt
        ):
            raise RescueOrchestrationError(
                "exhaustion_receipt has the wrong type"
            )
        if not isinstance(self.roots, RescueRootBinding):
            raise RescueOrchestrationError("roots must be RescueRootBinding")
        if not isinstance(self.budget, RescueExecutionBudget):
            raise RescueOrchestrationError(
                "budget must be RescueExecutionBudget"
            )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self,
            "idempotency_scope",
            _text(self.idempotency_scope, "idempotency_scope"),
        )
        object.__setattr__(
            self,
            "rescue_plan_root",
            _text(
                self.rescue_plan_root or self.plan.rescue_plan_cid,
                "rescue_plan_root",
            ),
        )
        object.__setattr__(self, "caller", _text(self.caller, "caller"))
        for name in (
            "fencing_epoch",
            "model_tokens",
            "model_cost_microunits",
            "start_action_index",
        ):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        if self.control_request is not None:
            if not isinstance(self.control_request, OperationRequest):
                raise RescueOrchestrationError(
                    "control_request must be OperationRequest"
                )
            if self.control_request.operation is not Operation.RESCUE:
                raise RescueOrchestrationError(
                    "control_request must use the shared rescue operation"
                )


@dataclass(frozen=True)
class RescueSimulatedEffect:
    """One exact effect from the mandatory side-effect-free simulation."""

    effect_id: str
    effect: str
    target_id: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("effect_id", "effect", "target_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.details, Mapping):
            raise RescueOrchestrationError("effect details must be a mapping")
        object.__setattr__(self, "details", _freeze(self.details))

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "effect_id": self.effect_id,
            "effect": self.effect,
            "target_id": self.target_id,
            "details": _canonical(self.details),
        }


@dataclass(frozen=True)
class RescueSimulationReceipt:
    action_content_id: str
    root_binding_id: str
    effects: Tuple[RescueSimulatedEffect, ...]
    simulator_id: str
    simulated_at_ms: int

    def __post_init__(self) -> None:
        for name in ("action_content_id", "root_binding_id", "simulator_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not self.effects or not all(
            isinstance(item, RescueSimulatedEffect) for item in self.effects
        ):
            raise RescueOrchestrationError(
                "simulation must return typed non-empty effects"
            )
        ids = tuple(item.effect_id for item in self.effects)
        if len(ids) != len(set(ids)):
            raise RescueOrchestrationError(
                "simulation effect IDs must be unique"
            )
        object.__setattr__(
            self, "simulated_at_ms", _nonnegative(
                self.simulated_at_ms, "simulated_at_ms"
            )
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_SIMULATION_RECEIPT_SCHEMA,
            "action_content_id": self.action_content_id,
            "root_binding_id": self.root_binding_id,
            "effects": [item.to_dict() for item in self.effects],
            "simulator_id": self.simulator_id,
            "simulated_at_ms": self.simulated_at_ms,
        }

    @property
    def receipt_id(self) -> str:
        return _content_id(self.to_dict())


@dataclass(frozen=True)
class RescueActionBinding:
    """Exact immutable input to every authorizer and the permit boundary."""

    plan_cid: str
    action_index: int
    action: RescueAction
    roots: RescueRootBinding
    simulation: RescueSimulationReceipt
    lease_id: str
    fencing_epoch: int
    idempotency_key: str
    caller: str

    def __post_init__(self) -> None:
        for name in ("plan_cid", "lease_id", "idempotency_key", "caller"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("action_index", "fencing_epoch"):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        if not isinstance(self.action, RescueAction):
            raise RescueOrchestrationError("action must be RescueAction")
        if not isinstance(self.roots, RescueRootBinding):
            raise RescueOrchestrationError("roots must be RescueRootBinding")
        if not isinstance(self.simulation, RescueSimulationReceipt):
            raise RescueOrchestrationError(
                "simulation must be RescueSimulationReceipt"
            )
        if self.simulation.action_content_id != self.action.content_id:
            raise RescueOrchestrationError(
                "simulation belongs to a different action"
            )
        if self.simulation.root_binding_id != self.roots.content_id:
            raise RescueOrchestrationError(
                "simulation belongs to different roots"
            )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_ACTION_BINDING_SCHEMA,
            "plan_cid": self.plan_cid,
            "action_index": self.action_index,
            "action": self.action.to_dict(),
            "roots": self.roots.to_dict(),
            "simulation": self.simulation.to_dict(),
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "idempotency_key": self.idempotency_key,
            "caller": self.caller,
            "completion_authority": False,
        }

    @property
    def binding_id(self) -> str:
        return _content_id(self.to_dict())

    @property
    def content_id(self) -> str:
        return self.binding_id


@dataclass(frozen=True)
class RescueAuthorizationReceipt:
    """Independent verdict over the same exact action binding."""

    domain: RescueAuthorizationDomain
    verdict: RescueAuthorizationVerdict
    binding_id: str
    root_binding_id: str
    authority_id: str
    reason_code: str = ""
    evaluated_at_ms: int = 0
    expires_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "domain", RescueAuthorizationDomain(self.domain)
        )
        object.__setattr__(
            self, "verdict", RescueAuthorizationVerdict(self.verdict)
        )
        for name in ("binding_id", "root_binding_id", "authority_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        lowered = self.authority_id.lower()
        if "model" in lowered or "provider" in lowered or "planner" in lowered:
            raise RescueOrchestrationError(
                "a model/planner/provider cannot authorize rescue execution"
            )
        if self.verdict is not RescueAuthorizationVerdict.PERMIT:
            object.__setattr__(
                self, "reason_code", _text(self.reason_code, "reason_code")
            )
        elif self.reason_code:
            object.__setattr__(
                self, "reason_code", _text(self.reason_code, "reason_code")
            )
        for name in ("evaluated_at_ms", "expires_at_ms"):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        if self.expires_at_ms <= self.evaluated_at_ms:
            raise RescueOrchestrationError(
                "authorization receipt must have a positive lifetime"
            )

    @property
    def admitted(self) -> bool:
        return self.verdict is RescueAuthorizationVerdict.PERMIT

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_AUTHORIZATION_RECEIPT_SCHEMA,
            "domain": self.domain.value,
            "verdict": self.verdict.value,
            "binding_id": self.binding_id,
            "root_binding_id": self.root_binding_id,
            "authority_id": self.authority_id,
            "reason_code": self.reason_code,
            "evaluated_at_ms": self.evaluated_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @property
    def receipt_id(self) -> str:
        return _content_id(self.to_dict())


@dataclass(frozen=True)
class RescuePermitUseReceipt:
    """Projection of an issuer-backed, consumed one-use execution permit."""

    permit_id: str
    binding_id: str
    root_binding_id: str
    incident_cid: str
    lease_id: str
    fencing_epoch: int
    idempotency_key: str
    issued_at_ms: int
    expires_at_ms: int
    consumed_at_ms: int
    use_sequence: int = 1
    remaining_uses: int = 0
    grants_completion_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "permit_id",
            "binding_id",
            "root_binding_id",
            "incident_cid",
            "lease_id",
            "idempotency_key",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "fencing_epoch",
            "issued_at_ms",
            "expires_at_ms",
            "consumed_at_ms",
            "use_sequence",
            "remaining_uses",
        ):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        if self.use_sequence != 1 or self.remaining_uses != 0:
            raise RescueOrchestrationError(
                "a rescue permit must authorize exactly one use"
            )
        if (
            self.expires_at_ms <= self.issued_at_ms
            or self.expires_at_ms - self.issued_at_ms
            > ABSOLUTE_MAX_PERMIT_TTL_MS
        ):
            raise RescueOrchestrationError(
                "rescue permit is not short-lived"
            )
        if not (
            self.issued_at_ms <= self.consumed_at_ms < self.expires_at_ms
        ):
            raise RescueOrchestrationError(
                "permit was consumed outside its validity window"
            )
        if self.grants_completion_authority:
            raise RescueOrchestrationError(
                "rescue permits cannot grant completion authority"
            )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_PERMIT_RECEIPT_SCHEMA,
            "permit_id": self.permit_id,
            "binding_id": self.binding_id,
            "root_binding_id": self.root_binding_id,
            "incident_cid": self.incident_cid,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "idempotency_key": self.idempotency_key,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "consumed_at_ms": self.consumed_at_ms,
            "use_sequence": self.use_sequence,
            "remaining_uses": self.remaining_uses,
            "grants_completion_authority": False,
        }

    @property
    def receipt_id(self) -> str:
        return _content_id(self.to_dict())


@dataclass(frozen=True)
class RescueEffectObservation:
    """Effects observed from exactly one shared control transaction."""

    effects: Tuple[RescueSimulatedEffect, ...]
    transaction_receipt_id: str
    complete: bool
    quarantined: bool = False
    control_result: Optional[OperationResult] = None

    def __post_init__(self) -> None:
        if not all(isinstance(item, RescueSimulatedEffect) for item in self.effects):
            raise RescueOrchestrationError(
                "observed effects must be typed simulated-effect records"
            )
        object.__setattr__(
            self,
            "transaction_receipt_id",
            _text(self.transaction_receipt_id, "transaction_receipt_id"),
        )
        if not isinstance(self.complete, bool) or not isinstance(
            self.quarantined, bool
        ):
            raise RescueOrchestrationError(
                "effect completion flags must be boolean"
            )
        if self.control_result is not None and not isinstance(
            self.control_result, OperationResult
        ):
            raise RescueOrchestrationError(
                "control_result must be OperationResult"
            )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "effects": [item.to_dict() for item in self.effects],
            "transaction_receipt_id": self.transaction_receipt_id,
            "complete": self.complete,
            "quarantined": self.quarantined,
            "control_result_id": (
                ""
                if self.control_result is None
                else self.control_result.content_id
            ),
        }


@dataclass(frozen=True)
class RescueHealthReceipt:
    state: RescueHealthState
    incident_cid: str
    root_binding_id: str
    health_test_id: str
    evidence_ids: Tuple[str, ...] = ()
    checked_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", RescueHealthState(self.state))
        for name in ("incident_cid", "root_binding_id", "health_test_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "evidence_ids", _strings(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self, "checked_at_ms", _nonnegative(
                self.checked_at_ms, "checked_at_ms"
            )
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_HEALTH_RECEIPT_SCHEMA,
            "state": self.state.value,
            "incident_cid": self.incident_cid,
            "root_binding_id": self.root_binding_id,
            "health_test_id": self.health_test_id,
            "evidence_ids": list(self.evidence_ids),
            "checked_at_ms": self.checked_at_ms,
        }

    @property
    def receipt_id(self) -> str:
        return _content_id(self.to_dict())


@dataclass(frozen=True)
class RescueActionExecutionReceipt:
    """Exact success, denial, partial, or quarantine receipt for one action."""

    plan_cid: str
    action_index: int
    action_content_id: str
    binding_id: str
    root_binding_id: str
    disposition: RescueReceiptDisposition
    stop_reason: RescueStopReason
    authorization_receipts: Tuple[RescueAuthorizationReceipt, ...] = ()
    simulation_receipt_id: str = ""
    permit_use_receipt: Optional[RescuePermitUseReceipt] = None
    observed_effects: Tuple[RescueSimulatedEffect, ...] = ()
    transaction_receipt_id: str = ""
    health_receipt: Optional[RescueHealthReceipt] = None
    recovery_steps: Tuple[str, ...] = ()
    started_at_ms: int = 0
    finished_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "plan_cid",
            "action_content_id",
            "binding_id",
            "root_binding_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "action_index", _nonnegative(self.action_index, "action_index")
        )
        object.__setattr__(
            self, "disposition", RescueReceiptDisposition(self.disposition)
        )
        object.__setattr__(self, "stop_reason", RescueStopReason(self.stop_reason))
        if not all(
            isinstance(item, RescueAuthorizationReceipt)
            for item in self.authorization_receipts
        ):
            raise RescueOrchestrationError(
                "authorization_receipts are malformed"
            )
        if self.simulation_receipt_id:
            object.__setattr__(
                self,
                "simulation_receipt_id",
                _text(self.simulation_receipt_id, "simulation_receipt_id"),
            )
        if self.permit_use_receipt is not None and not isinstance(
            self.permit_use_receipt, RescuePermitUseReceipt
        ):
            raise RescueOrchestrationError(
                "permit_use_receipt has the wrong type"
            )
        if not all(
            isinstance(item, RescueSimulatedEffect)
            for item in self.observed_effects
        ):
            raise RescueOrchestrationError("observed_effects are malformed")
        if self.transaction_receipt_id:
            object.__setattr__(
                self,
                "transaction_receipt_id",
                _text(
                    self.transaction_receipt_id, "transaction_receipt_id"
                ),
            )
        if self.health_receipt is not None and not isinstance(
            self.health_receipt, RescueHealthReceipt
        ):
            raise RescueOrchestrationError("health_receipt has the wrong type")
        object.__setattr__(
            self,
            "recovery_steps",
            _strings(self.recovery_steps, "recovery_steps"),
        )
        for name in ("started_at_ms", "finished_at_ms"):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        if self.finished_at_ms < self.started_at_ms:
            raise RescueOrchestrationError(
                "action receipt finishes before it starts"
            )

    @property
    def partial(self) -> bool:
        return self.disposition is RescueReceiptDisposition.PARTIAL

    @property
    def quarantined(self) -> bool:
        return self.disposition is RescueReceiptDisposition.QUARANTINED

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_ACTION_RECEIPT_SCHEMA,
            "requirement_id": RESCUE_ORCHESTRATION_REQUIREMENT_ID,
            "plan_cid": self.plan_cid,
            "action_index": self.action_index,
            "action_content_id": self.action_content_id,
            "binding_id": self.binding_id,
            "root_binding_id": self.root_binding_id,
            "disposition": self.disposition.value,
            "stop_reason": self.stop_reason.value,
            "authorization_receipts": [
                item.to_dict() for item in self.authorization_receipts
            ],
            "simulation_receipt_id": self.simulation_receipt_id,
            "permit_use_receipt": (
                None
                if self.permit_use_receipt is None
                else self.permit_use_receipt.to_dict()
            ),
            "observed_effects": [
                item.to_dict() for item in self.observed_effects
            ],
            "transaction_receipt_id": self.transaction_receipt_id,
            "health_receipt": (
                None
                if self.health_receipt is None
                else self.health_receipt.to_dict()
            ),
            "recovery_steps": list(self.recovery_steps),
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "completion_authority": False,
        }

    @property
    def receipt_id(self) -> str:
        return _content_id(self.to_dict())


@dataclass(frozen=True)
class RescueRunReceipt:
    plan_cid: str
    incident_cid: str
    root_binding_id: str
    disposition: RescueReceiptDisposition
    stop_reason: RescueStopReason
    action_receipts: Tuple[RescueActionExecutionReceipt, ...]
    next_action_index: int
    started_at_ms: int
    finished_at_ms: int
    recovery_steps: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("plan_cid", "incident_cid", "root_binding_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "disposition", RescueReceiptDisposition(self.disposition)
        )
        object.__setattr__(self, "stop_reason", RescueStopReason(self.stop_reason))
        if not all(
            isinstance(item, RescueActionExecutionReceipt)
            for item in self.action_receipts
        ):
            raise RescueOrchestrationError("action_receipts are malformed")
        object.__setattr__(
            self,
            "next_action_index",
            _nonnegative(self.next_action_index, "next_action_index"),
        )
        for name in ("started_at_ms", "finished_at_ms"):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "recovery_steps",
            _strings(self.recovery_steps, "recovery_steps"),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema": RESCUE_RUN_RECEIPT_SCHEMA,
            "requirement_id": RESCUE_ORCHESTRATION_REQUIREMENT_ID,
            "plan_cid": self.plan_cid,
            "incident_cid": self.incident_cid,
            "root_binding_id": self.root_binding_id,
            "disposition": self.disposition.value,
            "stop_reason": self.stop_reason.value,
            "action_receipts": [
                item.to_dict() for item in self.action_receipts
            ],
            "next_action_index": self.next_action_index,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "recovery_steps": list(self.recovery_steps),
            "completion_authority": False,
        }

    @property
    def receipt_id(self) -> str:
        return _content_id(self.to_dict())

    @property
    def recovered(self) -> bool:
        return self.stop_reason in {
            RescueStopReason.HEALTH_RESTORED,
            RescueStopReason.ALREADY_HEALTHY,
        }


class RescueStateProvider(Protocol):
    def snapshot(self) -> RescueRuntimeSnapshot:
        ...


class RescueEffectSimulator(Protocol):
    def simulate(
        self, action: RescueAction, roots: RescueRootBinding, now_ms: int
    ) -> RescueSimulationReceipt:
        ...


class RescueDomainAuthorizer(Protocol):
    def authorize(
        self, binding: RescueActionBinding, now_ms: int
    ) -> RescueAuthorizationReceipt:
        ...


class RescueExecutionPermitBoundary(Protocol):
    """Adapter to the shared execution-permit issuer and use verifier."""

    def issue_and_consume(
        self,
        binding: RescueActionBinding,
        authorizations: Sequence[RescueAuthorizationReceipt],
        snapshot: RescueRuntimeSnapshot,
        issued_at_ms: int,
        expires_at_ms: int,
    ) -> RescuePermitUseReceipt:
        ...


class RescueControlTransaction(Protocol):
    """The sole authorized effect adapter; implementations use control plane."""

    def execute(
        self,
        binding: RescueActionBinding,
        permit: RescuePermitUseReceipt,
        control_request: Optional[OperationRequest],
    ) -> RescueEffectObservation:
        ...


class RescueHealthTester(Protocol):
    def test(
        self, binding: RescueActionBinding, now_ms: int
    ) -> RescueHealthReceipt:
        ...


def _invoke(
    dependency: Any, method: str, *args: Any
) -> Any:
    target = getattr(dependency, method, None)
    if callable(target):
        return target(*args)
    if callable(dependency):
        return dependency(*args)
    raise RescueOrchestrationError(
        type(dependency).__name__ + " does not implement " + method
    )


def _operation_spec(
    action: RescueAction,
    catalog: Mapping[RescueOperation, RescueOperationSpec],
) -> RescueOperationSpec:
    try:
        spec = catalog[action.operation]
    except (KeyError, TypeError) as exc:
        raise RescuePlannerValidationError(
            "action operation is outside the closed rescue catalog",
            reason_code="unknown_operation",
        ) from exc
    if set(action.parameters) != set(spec.parameters):
        required = {
            name for name, parameter in spec.parameters.items()
            if parameter.required
        }
        if not required.issubset(action.parameters) or not set(
            action.parameters
        ).issubset(spec.parameters):
            raise RescuePlannerValidationError(
                "action parameters differ from the closed catalog schema",
                reason_code="invalid_parameters",
            )
    for name, value in action.parameters.items():
        spec.parameters[name].validate(value, name)
    if spec.target_prefixes and not action.target_id.startswith(
        spec.target_prefixes
    ):
        raise RescuePlannerValidationError(
            "action target type is outside the operation schema",
            reason_code="invalid_target_type",
        )
    if (
        tuple(action.expected_effects) != spec.expected_effects
        or action.success_test != spec.success_test
        or action.stop_condition != spec.stop_condition
        or action.rollback_operation not in spec.rollback_operations
    ):
        raise RescuePlannerValidationError(
            "action semantics differ from the closed rescue catalog",
            reason_code="invalid_expected_effects",
        )
    return spec


def _effect_signature(
    effects: Sequence[RescueSimulatedEffect],
) -> Tuple[Tuple[str, str, str, str], ...]:
    return tuple(
        sorted(
            (
                item.effect_id,
                item.effect,
                item.target_id,
                _content_id(item.details),
            )
            for item in effects
        )
    )


def _control_effects(
    result: OperationResult,
    binding: RescueActionBinding,
) -> RescueEffectObservation:
    """Strictly adapt an existing control transaction result."""

    if result.status is not OperationStatus.SUCCEEDED:
        raise RescueOrchestrationError(
            "control transaction did not succeed: " + result.status.value
        )
    claims = tuple(result.effects)
    expected = {
        item.effect_id: item for item in binding.simulation.effects
    }
    observed = []
    for claim in claims:
        if not isinstance(claim, EffectClaim):
            raise RescueOrchestrationError(
                "control transaction returned an untyped effect claim"
            )
        simulated = expected.get(claim.effect_id)
        if simulated is None or not claim.applied:
            continue
        observed.append(simulated)
    return RescueEffectObservation(
        effects=tuple(observed),
        transaction_receipt_id=result.audit_receipt_id,
        complete=set(item.effect_id for item in observed) == set(expected),
        control_result=result,
    )


class RescueOrchestrator:
    """Validate and execute a rescue plan with one fresh permit per action."""

    def __init__(
        self,
        *,
        state_provider: RescueStateProvider,
        simulator: RescueEffectSimulator,
        authorizers: Mapping[
            RescueAuthorizationDomain, RescueDomainAuthorizer
        ],
        permit_boundary: RescueExecutionPermitBoundary,
        control_transaction: RescueControlTransaction,
        health_tester: RescueHealthTester,
        operation_catalog: Mapping[
            RescueOperation, RescueOperationSpec
        ] = DEFAULT_RESCUE_OPERATION_CATALOG,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self._state_provider = state_provider
        self._simulator = simulator
        normalized = {
            RescueAuthorizationDomain(key): value
            for key, value in authorizers.items()
        }
        if set(normalized) != set(REQUIRED_AUTHORIZATION_DOMAINS):
            raise RescueOrchestrationError(
                "exactly one IntentIR, LegalIR, SecurityIR, proof, and "
                "control authorizer is required"
            )
        if len({id(item) for item in normalized.values()}) != len(normalized):
            raise RescueOrchestrationError(
                "authorization domains require independent authorizers"
            )
        self._authorizers = MappingProxyType(normalized)
        self._permit_boundary = permit_boundary
        self._control_transaction = control_transaction
        self._health_tester = health_tester
        self._catalog = MappingProxyType(dict(operation_catalog))
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.RLock()
        self._consumed: dict[str, Tuple[str, str, str]] = {}

    def _now(self) -> int:
        return _nonnegative(self._clock_ms(), "clock_ms result")

    def _snapshot(self) -> RescueRuntimeSnapshot:
        value = _invoke(self._state_provider, "snapshot")
        if not isinstance(value, RescueRuntimeSnapshot):
            raise RescueOrchestrationError(
                "state provider returned an invalid snapshot"
            )
        return value

    @staticmethod
    def _root_stop(
        expected: RescueRootBinding, current: RescueRootBinding
    ) -> RescueStopReason:
        if current.incident_cid != expected.incident_cid:
            return RescueStopReason.INCIDENT_DRIFT
        if (
            current.exhaustion_receipt_cid
            != expected.exhaustion_receipt_cid
        ):
            return RescueStopReason.EXHAUSTION_DRIFT
        return RescueStopReason.ROOT_DRIFT

    def _guard_snapshot(
        self,
        request: RescueExecutionRequest,
        snapshot: RescueRuntimeSnapshot,
        now_ms: int,
        *,
        check_cooldown: bool = True,
    ) -> RescueStopReason | None:
        if snapshot.quarantined:
            return RescueStopReason.QUARANTINED
        if snapshot.roots != request.roots:
            return self._root_stop(request.roots, snapshot.roots)
        if snapshot.lease_id != request.lease_id:
            return RescueStopReason.LEASE_LOST
        if snapshot.fencing_epoch != request.fencing_epoch:
            return RescueStopReason.FENCE_LOST
        if check_cooldown and now_ms < snapshot.cooldown_until_ms:
            return RescueStopReason.COOLDOWN_ACTIVE
        return None

    @staticmethod
    def _validate_request_bindings(request: RescueExecutionRequest) -> None:
        plan = request.plan
        incident = request.incident
        exhaustion = request.exhaustion_receipt
        roots = request.roots
        exact = (
            (plan.incident_cid, incident.incident_cid, "incident"),
            (plan.incident_cid, roots.incident_cid, "incident root"),
            (
                plan.exhaustion_receipt_cid,
                exhaustion.receipt_cid,
                "exhaustion receipt",
            ),
            (
                plan.exhaustion_receipt_cid,
                roots.exhaustion_receipt_cid,
                "exhaustion root",
            ),
            (
                plan.repository_root_cid,
                incident.repository_root_cid,
                "repository root",
            ),
            (
                plan.repository_root_cid,
                exhaustion.repository_root_cid,
                "exhaustion repository root",
            ),
            (
                plan.repository_root_cid,
                roots.repository_root_cid,
                "current repository root",
            ),
            (plan.run_cid, incident.run_cid, "run root"),
            (plan.run_cid, exhaustion.run_cid, "exhaustion run root"),
            (plan.run_cid, roots.run_cid, "current run root"),
            (plan.policy_root, incident.policy_root, "policy root"),
            (
                plan.policy_root,
                exhaustion.policy_root,
                "exhaustion policy root",
            ),
            (plan.policy_root, roots.policy_root, "current policy root"),
        )
        for observed, expected, noun in exact:
            if observed != expected:
                if noun in {"incident", "incident root"}:
                    reason_code = "incident_drift"
                elif noun in {
                    "exhaustion receipt",
                    "exhaustion root",
                }:
                    reason_code = "exhaustion_drift"
                else:
                    reason_code = "stale_roots"
                raise RescuePlannerValidationError(
                    noun + " does not match the current incident authority",
                    reason_code=reason_code,
                )
        if incident.status is RecordStatus.COMPLETED:
            raise RescuePlannerValidationError(
                "terminal incidents cannot execute rescue plans",
                reason_code="self_authorization",
            )
        if plan.status is not RecordStatus.PROPOSED:
            raise RescuePlannerValidationError(
                "only proposal-tier rescue plans may be evaluated",
                reason_code="self_authorization",
            )
        if exhaustion.status is not RecordStatus.QUARANTINED:
            raise RescuePlannerValidationError(
                "rescue requires a terminal programmatic exhaustion receipt",
                reason_code="exhaustion_drift",
            )
        if exhaustion.circuit_open:
            raise RescuePlannerValidationError(
                "the current incident circuit is quarantined",
                reason_code="quarantine_required",
            )
        if request.model_tokens > request.budget.max_model_tokens or (
            request.model_cost_microunits
            > request.budget.max_model_cost_microunits
        ):
            raise RescuePlannerValidationError(
                "model budget exceeds execution authority",
                reason_code="model_budget",
            )
        maximum_actions = min(
            request.budget.max_actions,
            request.budget.max_model_actions,
            exhaustion.budget.max_rescue_actions,
            plan.max_actions,
        )
        if len(plan.actions) > maximum_actions:
            raise RescuePlannerValidationError(
                "plan action sequence exceeds current non-renewable budget",
                reason_code="excess_actions",
            )
        if request.start_action_index >= len(plan.actions):
            raise RescuePlannerValidationError(
                "start_action_index is outside the plan",
                reason_code="excess_actions",
            )
        if request.control_request is not None:
            control = request.control_request
            parameters = control.parameters
            if control.tree_id != roots.tree_id:
                raise RescuePlannerValidationError(
                    "control request roots differ from rescue roots",
                    reason_code="stale_roots",
                )
            bindings = {
                "incident_cid": roots.incident_cid,
                "rescue_plan_cid": plan.rescue_plan_cid,
                "rescue_plan_root": request.rescue_plan_root,
                "rescue_plan_incident_cid": roots.incident_cid,
                "rescue_plan_tree_id": roots.tree_id,
            }
            for name, expected in bindings.items():
                if parameters.get(name) != expected:
                    raise RescuePlannerValidationError(
                        "control request " + name + " is stale or unbound",
                        reason_code="stale_roots",
                    )
            supplied_exhaustion = parameters.get("exhaustion_receipt_cid")
            if supplied_exhaustion not in (
                None,
                roots.exhaustion_receipt_cid,
            ):
                raise RescuePlannerValidationError(
                    "control request exhaustion receipt is stale",
                    reason_code="stale_roots",
                )
            selected = parameters.get("action_index")
            if selected != request.start_action_index:
                raise RescuePlannerValidationError(
                    "control request selects a different action",
                    reason_code="changed_arguments",
                )
            if (
                control.dry_run
                or control.caller != request.caller
                or control.lease_id != request.lease_id
                or control.fencing_epoch != request.fencing_epoch
                or control.idempotency_key != request.idempotency_scope
            ):
                raise RescuePlannerValidationError(
                    "control request mutation guards differ from rescue request",
                    reason_code="changed_arguments",
                )

    @staticmethod
    def _recovery_steps(reason: RescueStopReason) -> Tuple[str, ...]:
        if reason in {
            RescueStopReason.PARTIAL_EFFECT,
            RescueStopReason.UNEXPECTED_EFFECT,
            RescueStopReason.QUARANTINED,
        }:
            return (
                "quarantine_exact_incident_scope",
                "inspect_control_transaction_receipt",
                "refresh_incident_and_roots",
                "require_operator_or_programmatic_recovery",
            )
        if reason in {
            RescueStopReason.ROOT_DRIFT,
            RescueStopReason.INCIDENT_DRIFT,
            RescueStopReason.EXHAUSTION_DRIFT,
            RescueStopReason.LEASE_LOST,
            RescueStopReason.FENCE_LOST,
        }:
            return (
                "discard_stale_permit",
                "refresh_incident_exhaustion_and_roots",
                "replan_if_still_exhausted",
            )
        if reason in {
            RescueStopReason.AUTHORIZATION_DENIED,
            RescueStopReason.PERMIT_DENIED,
            RescueStopReason.CONTROL_DENIED,
        }:
            return (
                "preserve_denial_receipts",
                "do_not_retry_without_authority_change",
                "operator_review",
            )
        if reason is RescueStopReason.IDEMPOTENCY_REPLAY:
            return (
                "load_exact_prior_action_receipt",
                "do_not_repeat_effect",
            )
        return ("refresh_health_and_incident_evidence",)

    def _empty_binding(
        self,
        request: RescueExecutionRequest,
        action_index: int,
        now_ms: int,
    ) -> RescueActionBinding:
        action = request.plan.actions[action_index]
        effect = RescueSimulatedEffect(
            effect_id="denied:" + action.content_id,
            effect=action.expected_effects[0],
            target_id=action.target_id,
        )
        simulation = RescueSimulationReceipt(
            action_content_id=action.content_id,
            root_binding_id=request.roots.content_id,
            effects=(effect,),
            simulator_id="rescue-orchestrator:denied-before-simulation",
            simulated_at_ms=now_ms,
        )
        return RescueActionBinding(
            plan_cid=request.plan.rescue_plan_cid,
            action_index=action_index,
            action=action,
            roots=request.roots,
            simulation=simulation,
            lease_id=request.lease_id,
            fencing_epoch=request.fencing_epoch,
            idempotency_key=self._action_idempotency(request, action_index),
            caller=request.caller,
        )

    @staticmethod
    def _action_idempotency(
        request: RescueExecutionRequest, action_index: int
    ) -> str:
        if request.control_request is not None:
            return request.control_request.idempotency_key
        return request.idempotency_scope + ":" + str(action_index)

    def _receipt(
        self,
        binding: RescueActionBinding,
        *,
        disposition: RescueReceiptDisposition,
        reason: RescueStopReason,
        started_at_ms: int,
        authorizations: Sequence[RescueAuthorizationReceipt] = (),
        permit: Optional[RescuePermitUseReceipt] = None,
        observation: Optional[RescueEffectObservation] = None,
        health: Optional[RescueHealthReceipt] = None,
    ) -> RescueActionExecutionReceipt:
        return RescueActionExecutionReceipt(
            plan_cid=binding.plan_cid,
            action_index=binding.action_index,
            action_content_id=binding.action.content_id,
            binding_id=binding.binding_id,
            root_binding_id=binding.roots.content_id,
            disposition=disposition,
            stop_reason=reason,
            authorization_receipts=tuple(authorizations),
            simulation_receipt_id=binding.simulation.receipt_id,
            permit_use_receipt=permit,
            observed_effects=(
                () if observation is None else observation.effects
            ),
            transaction_receipt_id=(
                "" if observation is None else observation.transaction_receipt_id
            ),
            health_receipt=health,
            recovery_steps=self._recovery_steps(reason),
            started_at_ms=started_at_ms,
            finished_at_ms=self._now(),
        )

    def _execute_action(
        self,
        request: RescueExecutionRequest,
        action_index: int,
        run_started_at_ms: int,
    ) -> RescueActionExecutionReceipt:
        started = self._now()
        if started - run_started_at_ms >= request.budget.max_elapsed_ms:
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.STOPPED,
                reason=RescueStopReason.TIME_BUDGET,
                started_at_ms=started,
            )
        action = request.plan.actions[action_index]
        _operation_spec(action, self._catalog)
        if action.target_id not in request.incident.target_ids:
            raise RescuePlannerValidationError(
                "action target is outside the exact incident",
                reason_code="unknown_target",
            )
        idempotency_key = self._action_idempotency(request, action_index)

        try:
            snapshot = self._snapshot()
        except Exception:
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=RescueStopReason.ROOT_DRIFT,
                started_at_ms=started,
            )
        guard = self._guard_snapshot(request, snapshot, started)
        if guard is not None:
            binding = self._empty_binding(request, action_index, started)
            disposition = (
                RescueReceiptDisposition.QUARANTINED
                if guard is RescueStopReason.QUARANTINED
                else RescueReceiptDisposition.DENIED
            )
            return self._receipt(
                binding,
                disposition=disposition,
                reason=guard,
                started_at_ms=started,
            )
        prior_action = self._consumed.get(idempotency_key)
        if prior_action is not None:
            binding = self._empty_binding(request, action_index, started)
            exact_action = prior_action == (
                request.plan.rescue_plan_cid,
                action.content_id,
                request.roots.content_id,
            )
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=(
                    RescueStopReason.IDEMPOTENCY_REPLAY
                    if exact_action
                    else RescueStopReason.IDEMPOTENCY_CONFLICT
                ),
                started_at_ms=started,
            )

        try:
            health = _invoke(self._health_tester, "test", None, started)
        except Exception:
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=RescueStopReason.HEALTH_TEST_FAILED,
                started_at_ms=started,
            )
        if not isinstance(health, RescueHealthReceipt):
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=RescueStopReason.HEALTH_TEST_FAILED,
                started_at_ms=started,
            )
        if (
            health.incident_cid != request.roots.incident_cid
            or health.root_binding_id != request.roots.content_id
        ):
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.ROOT_DRIFT,
                started_at_ms=started,
                health=health,
            )
        if health.state is RescueHealthState.HEALTHY:
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.RECOVERED,
                reason=RescueStopReason.ALREADY_HEALTHY,
                started_at_ms=started,
                health=health,
            )
        if health.state is RescueHealthState.QUARANTINED:
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=RescueStopReason.QUARANTINED,
                started_at_ms=started,
                health=health,
            )

        try:
            simulation = _invoke(
                self._simulator,
                "simulate",
                action,
                request.roots,
                self._now(),
            )
        except Exception:
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.SIMULATION_DENIED,
                started_at_ms=started,
            )
        if not isinstance(simulation, RescueSimulationReceipt):
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.SIMULATION_DENIED,
                started_at_ms=started,
            )
        if (
            simulation.action_content_id != action.content_id
            or simulation.root_binding_id != request.roots.content_id
            or tuple(item.effect for item in simulation.effects)
            != tuple(action.expected_effects)
            or any(item.target_id != action.target_id for item in simulation.effects)
        ):
            binding = self._empty_binding(request, action_index, started)
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.SIMULATION_DENIED,
                started_at_ms=started,
            )
        binding = RescueActionBinding(
            plan_cid=request.plan.rescue_plan_cid,
            action_index=action_index,
            action=action,
            roots=request.roots,
            simulation=simulation,
            lease_id=request.lease_id,
            fencing_epoch=request.fencing_epoch,
            idempotency_key=idempotency_key,
            caller=request.caller,
        )

        prior = self._consumed.get(binding.idempotency_key)
        if prior is not None:
            reason = (
                RescueStopReason.IDEMPOTENCY_REPLAY
                if prior
                == (
                    binding.plan_cid,
                    binding.action.content_id,
                    binding.roots.content_id,
                )
                else RescueStopReason.IDEMPOTENCY_CONFLICT
            )
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=reason,
                started_at_ms=started,
            )

        authorizations = []
        authority_ids = set()
        now = self._now()
        for domain in REQUIRED_AUTHORIZATION_DOMAINS:
            try:
                receipt = _invoke(
                    self._authorizers[domain], "authorize", binding, now
                )
            except Exception:
                return self._receipt(
                    binding,
                    disposition=RescueReceiptDisposition.DENIED,
                    reason=RescueStopReason.AUTHORIZATION_DENIED,
                    started_at_ms=started,
                    authorizations=authorizations,
                )
            if not isinstance(receipt, RescueAuthorizationReceipt):
                return self._receipt(
                    binding,
                    disposition=RescueReceiptDisposition.DENIED,
                    reason=RescueStopReason.AUTHORIZATION_DENIED,
                    started_at_ms=started,
                    authorizations=authorizations,
                )
            if (
                receipt.domain is not domain
                or receipt.binding_id != binding.binding_id
                or receipt.root_binding_id != request.roots.content_id
                or receipt.evaluated_at_ms > now
                or receipt.expires_at_ms <= now
            ):
                return self._receipt(
                    binding,
                    disposition=RescueReceiptDisposition.DENIED,
                    reason=RescueStopReason.AUTHORIZATION_DENIED,
                    started_at_ms=started,
                    authorizations=tuple(authorizations) + (receipt,),
                )
            if receipt.authority_id in authority_ids:
                raise RescueOrchestrationError(
                    "authorization receipts are not independently produced"
                )
            authority_ids.add(receipt.authority_id)
            authorizations.append(receipt)
            if not receipt.admitted:
                return self._receipt(
                    binding,
                    disposition=RescueReceiptDisposition.DENIED,
                    reason=RescueStopReason.AUTHORIZATION_DENIED,
                    started_at_ms=started,
                    authorizations=authorizations,
                )

        # This is the immediate pre-effect recheck.  Nothing authority-changing
        # may occur between this snapshot, permit consumption, and dispatch.
        now = self._now()
        if now - run_started_at_ms >= request.budget.max_elapsed_ms:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.STOPPED,
                reason=RescueStopReason.TIME_BUDGET,
                started_at_ms=started,
                authorizations=authorizations,
            )
        try:
            pre_effect = self._snapshot()
        except Exception:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.ROOT_DRIFT,
                started_at_ms=started,
                authorizations=authorizations,
            )
        guard = self._guard_snapshot(request, pre_effect, now)
        if guard is not None:
            return self._receipt(
                binding,
                disposition=(
                    RescueReceiptDisposition.QUARANTINED
                    if guard is RescueStopReason.QUARANTINED
                    else RescueReceiptDisposition.DENIED
                ),
                reason=guard,
                started_at_ms=started,
                authorizations=authorizations,
            )
        if pre_effect.revision != snapshot.revision:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.ROOT_DRIFT,
                started_at_ms=started,
                authorizations=authorizations,
            )
        try:
            permit = _invoke(
                self._permit_boundary,
                "issue_and_consume",
                binding,
                tuple(authorizations),
                pre_effect,
                now,
                min(
                    now + request.budget.permit_ttl_ms,
                    run_started_at_ms + request.budget.max_elapsed_ms,
                ),
            )
        except Exception:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.PERMIT_DENIED,
                started_at_ms=started,
                authorizations=authorizations,
            )
        if not isinstance(permit, RescuePermitUseReceipt):
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.PERMIT_DENIED,
                started_at_ms=started,
                authorizations=authorizations,
            )
        exact_permit = (
            permit.binding_id == binding.binding_id
            and permit.root_binding_id == request.roots.content_id
            and permit.incident_cid == request.roots.incident_cid
            and permit.lease_id == request.lease_id
            and permit.fencing_epoch == request.fencing_epoch
            and permit.idempotency_key == binding.idempotency_key
        )
        if not exact_permit:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.PERMIT_DENIED,
                started_at_ms=started,
                authorizations=authorizations,
            )
        # Mark consumed before handing authority to the transaction.  A crash
        # after this point is a recoverable partial operation, never a retry.
        self._consumed[binding.idempotency_key] = (
            binding.plan_cid,
            binding.action.content_id,
            binding.roots.content_id,
        )
        control_request = request.control_request
        control_failed = False
        control_denied = False
        try:
            observation = _invoke(
                self._control_transaction,
                "execute",
                binding,
                permit,
                control_request,
            )
            if isinstance(observation, OperationResult):
                if observation.status is OperationStatus.SUCCEEDED:
                    observation = _control_effects(observation, binding)
                else:
                    expected_by_id = {
                        item.effect_id: item
                        for item in binding.simulation.effects
                    }
                    observed = tuple(
                        expected_by_id[item.effect_id]
                        for item in observation.effects
                        if item.applied
                        and item.effect_id in expected_by_id
                    )
                    control_denied = not observed
                    observation = RescueEffectObservation(
                        effects=observed,
                        transaction_receipt_id=(
                            observation.audit_receipt_id
                            or "control-denial:" + observation.request_id
                        ),
                        complete=False,
                        control_result=observation,
                    )
            if not isinstance(observation, RescueEffectObservation):
                raise RescueOrchestrationError(
                    "control transaction returned an invalid observation"
                )
        except Exception:
            # The permit has been consumed, so an unknown transaction outcome
            # can never be retried.  Preserve it as a partial effect and force
            # recovery/quarantine review.
            control_failed = True
            observation = RescueEffectObservation(
                effects=(),
                transaction_receipt_id=(
                    "unknown-control-outcome:" + permit.receipt_id
                ),
                complete=False,
            )

        # A health test is mandatory after every transaction attempt, including
        # partial, unexpected, quarantined, and unknown outcomes.
        try:
            post_snapshot = self._snapshot()
            post_guard = self._guard_snapshot(
                request,
                post_snapshot,
                self._now(),
                check_cooldown=False,
            )
        except Exception:
            post_guard = RescueStopReason.ROOT_DRIFT
        try:
            post_health = _invoke(
                self._health_tester, "test", binding, self._now()
            )
            if not isinstance(post_health, RescueHealthReceipt):
                raise RescueOrchestrationError(
                    "post-effect health tester returned an invalid receipt"
                )
        except Exception:
            post_health = RescueHealthReceipt(
                state=RescueHealthState.UNKNOWN,
                incident_cid=request.roots.incident_cid,
                root_binding_id=request.roots.content_id,
                health_test_id=(
                    "health-test-unavailable:" + observation.transaction_receipt_id
                ),
                checked_at_ms=self._now(),
            )

        expected_signature = _effect_signature(simulation.effects)
        observed_signature = _effect_signature(observation.effects)
        if control_denied:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.DENIED,
                reason=RescueStopReason.CONTROL_DENIED,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )
        if control_failed:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.PARTIAL,
                reason=RescueStopReason.PARTIAL_EFFECT,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )
        if observation.quarantined:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=RescueStopReason.QUARANTINED,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )
        if not observation.complete or not observation.effects:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.PARTIAL,
                reason=RescueStopReason.PARTIAL_EFFECT,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )
        if observed_signature != expected_signature:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=RescueStopReason.UNEXPECTED_EFFECT,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )

        # Revision may change because the expected transaction just committed;
        # every semantic root, incident, lease and fence still must be exact.
        if post_guard is not None:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=post_guard,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )
        if (
            post_health.incident_cid != request.roots.incident_cid
            or post_health.root_binding_id != request.roots.content_id
        ):
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=RescueStopReason.ROOT_DRIFT,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )
        if post_health.state is RescueHealthState.HEALTHY:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.RECOVERED,
                reason=RescueStopReason.HEALTH_RESTORED,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )
        if post_health.state in {
            RescueHealthState.UNKNOWN,
            RescueHealthState.QUARANTINED,
        }:
            return self._receipt(
                binding,
                disposition=RescueReceiptDisposition.QUARANTINED,
                reason=RescueStopReason.HEALTH_TEST_FAILED,
                started_at_ms=started,
                authorizations=authorizations,
                permit=permit,
                observation=observation,
                health=post_health,
            )
        return self._receipt(
            binding,
            disposition=RescueReceiptDisposition.APPLIED,
            reason=RescueStopReason.ACTION_APPLIED,
            started_at_ms=started,
            authorizations=authorizations,
            permit=permit,
            observation=observation,
            health=post_health,
        )

    def execute(self, request: RescueExecutionRequest) -> RescueRunReceipt:
        """Execute bounded actions until a mandatory stop condition is met."""

        if not isinstance(request, RescueExecutionRequest):
            raise RescueOrchestrationError(
                "request must be RescueExecutionRequest"
            )
        with self._lock:
            started = self._now()
            try:
                self._validate_request_bindings(request)
            except RescuePlannerValidationError as exc:
                action_index = min(
                    request.start_action_index, len(request.plan.actions) - 1
                )
                binding = self._empty_binding(request, action_index, started)
                reason = {
                    "model_budget": RescueStopReason.MODEL_BUDGET,
                    "excess_actions": RescueStopReason.ACTION_BUDGET,
                    "incident_drift": RescueStopReason.INCIDENT_DRIFT,
                    "exhaustion_drift": RescueStopReason.EXHAUSTION_DRIFT,
                    "stale_roots": RescueStopReason.ROOT_DRIFT,
                    "quarantine_required": RescueStopReason.QUARANTINED,
                }.get(exc.reason_code, RescueStopReason.SCHEMA_DENIED)
                receipt_disposition = (
                    RescueReceiptDisposition.QUARANTINED
                    if reason is RescueStopReason.QUARANTINED
                    else RescueReceiptDisposition.DENIED
                )
                action_receipt = self._receipt(
                    binding,
                    disposition=receipt_disposition,
                    reason=reason,
                    started_at_ms=started,
                )
                return RescueRunReceipt(
                    plan_cid=request.plan.rescue_plan_cid,
                    incident_cid=request.roots.incident_cid,
                    root_binding_id=request.roots.content_id,
                    disposition=action_receipt.disposition,
                    stop_reason=action_receipt.stop_reason,
                    action_receipts=(action_receipt,),
                    next_action_index=action_index,
                    started_at_ms=started,
                    finished_at_ms=self._now(),
                    recovery_steps=action_receipt.recovery_steps,
                )

            receipts = []
            stop = RescueStopReason.ACTION_BUDGET
            disposition = RescueReceiptDisposition.STOPPED
            limit = min(
                len(request.plan.actions),
                request.start_action_index + request.budget.max_actions,
            )
            # A shared ``Operation.RESCUE`` transaction selects one action
            # index.  Later actions require new requests, idempotency keys,
            # control authorizations, and transactions.
            if request.control_request is not None:
                limit = min(limit, request.start_action_index + 1)
            next_index = request.start_action_index
            for index in range(request.start_action_index, limit):
                try:
                    receipt = self._execute_action(request, index, started)
                except RescuePlannerValidationError:
                    binding = self._empty_binding(request, index, self._now())
                    receipt = self._receipt(
                        binding,
                        disposition=RescueReceiptDisposition.DENIED,
                        reason=RescueStopReason.SCHEMA_DENIED,
                        started_at_ms=started,
                    )
                receipts.append(receipt)
                next_index = index + 1
                stop = receipt.stop_reason
                disposition = receipt.disposition
                if receipt.stop_reason is not RescueStopReason.ACTION_APPLIED:
                    break
                if self._now() - started >= request.budget.max_elapsed_ms:
                    stop = RescueStopReason.TIME_BUDGET
                    disposition = RescueReceiptDisposition.STOPPED
                    break
            else:
                if next_index < len(request.plan.actions):
                    stop = RescueStopReason.ACTION_BUDGET
                    disposition = RescueReceiptDisposition.STOPPED
                elif receipts:
                    stop = receipts[-1].stop_reason
                    disposition = receipts[-1].disposition
            recovery = (
                receipts[-1].recovery_steps
                if receipts
                else self._recovery_steps(stop)
            )
            return RescueRunReceipt(
                plan_cid=request.plan.rescue_plan_cid,
                incident_cid=request.roots.incident_cid,
                root_binding_id=request.roots.content_id,
                disposition=disposition,
                stop_reason=stop,
                action_receipts=tuple(receipts),
                next_action_index=next_index,
                started_at_ms=started,
                finished_at_ms=self._now(),
                recovery_steps=recovery,
            )

    orchestrate = execute
    execute_plan = execute


def execute_rescue_plan(
    request: RescueExecutionRequest,
    *,
    state_provider: RescueStateProvider,
    simulator: RescueEffectSimulator,
    authorizers: Mapping[RescueAuthorizationDomain, RescueDomainAuthorizer],
    permit_boundary: RescueExecutionPermitBoundary,
    control_transaction: RescueControlTransaction,
    health_tester: RescueHealthTester,
    operation_catalog: Mapping[
        RescueOperation, RescueOperationSpec
    ] = DEFAULT_RESCUE_OPERATION_CATALOG,
    clock_ms: Optional[Callable[[], int]] = None,
) -> RescueRunReceipt:
    """Functional entry point with caller-owned state and dependencies."""

    return RescueOrchestrator(
        state_provider=state_provider,
        simulator=simulator,
        authorizers=authorizers,
        permit_boundary=permit_boundary,
        control_transaction=control_transaction,
        health_tester=health_tester,
        operation_catalog=operation_catalog,
        clock_ms=clock_ms,
    ).execute(request)


# Concise compatibility names for control adapters and downstream ASI-158.
BoundedRescueOrchestrator = RescueOrchestrator
RescueExecutionReceipt = RescueRunReceipt
RescuePartialReceipt = RescueActionExecutionReceipt
RescueQuarantineReceipt = RescueActionExecutionReceipt


__all__ = [
    "ABSOLUTE_MAX_PERMIT_TTL_MS",
    "ABSOLUTE_MAX_RESCUE_ACTIONS",
    "BoundedRescueOrchestrator",
    "REQUIRED_AUTHORIZATION_DOMAINS",
    "RESCUE_ORCHESTRATION_REQUIREMENT_ID",
    "RescueActionBinding",
    "RescueActionExecutionReceipt",
    "RescueAuthorizationDomain",
    "RescueAuthorizationReceipt",
    "RescueAuthorizationVerdict",
    "RescueControlTransaction",
    "RescueDomainAuthorizer",
    "RescueEffectObservation",
    "RescueEffectSimulator",
    "RescueExecutionBudget",
    "RescueExecutionPermitBoundary",
    "RescueExecutionReceipt",
    "RescueExecutionRequest",
    "RescueHealthReceipt",
    "RescueHealthState",
    "RescueHealthTester",
    "RescueOrchestrationError",
    "RescueOrchestrator",
    "RescuePartialReceipt",
    "RescuePermitUseReceipt",
    "RescueQuarantineReceipt",
    "RescueReceiptDisposition",
    "RescueRootBinding",
    "RescueRunReceipt",
    "RescueRuntimeSnapshot",
    "RescueSimulatedEffect",
    "RescueSimulationReceipt",
    "RescueStateProvider",
    "RescueStopReason",
    "execute_rescue_plan",
]
