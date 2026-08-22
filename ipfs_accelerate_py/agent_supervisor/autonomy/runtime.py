"""Provider-free composition shell for the autonomous meta-controller.

``AutonomousMetaController`` names one unresolved decision, delegates all
action selection to :class:`~.cognitive_scheduler.CognitiveScheduler`, and
reserves objective budget before exposing the selected meta-action.  It does
not execute an action, call a model, write durable state, admit a repository
effect, or authorize task completion.

``AutonomyRuntime@1`` is the event-driven loop around that shell.  It consumes
closed wake kinds, runs at most the nearest safe segment, acknowledges
cursors only after a cycle, and stays idle on unchanged complete or healthily
exhausted boards.  It never refills a budget, never calls a model, and never
rewrites a checkpoint whose durable identity is unchanged.

The effect boundary remains
``agent_supervisor.context.decision_runtime.DecisionRuntime``.  A downstream
adapter must translate an ``ACTION_ADMITTED`` result into that authority's
typed input and obtain its permit before doing anything observable.  The
result below deliberately carries ``authorizes_effect == False`` so it cannot
be mistaken for a ``DecisionRuntime`` permit.

Persistence is also outside this module.  :meth:`snapshot_json` merely emits
a content-bound restart value for an existing state/CAS authority to store.
The injected budget controller owns budget transitions and their durable
representation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Protocol

from ..proof.formal_verification_contracts import canonical_json, content_identity
from .cognitive_budget import CognitiveCost, ObjectiveCognitiveBudgetLedger
from .cognitive_scheduler import CognitiveScheduler, CognitiveSchedulingContext
from .contracts import (
    AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
    MAX_CANONICAL_RECORD_BYTES,
    MAX_IDENTIFIER_BYTES,
    BudgetExhaustion,
    BudgetLedger,
    BudgetPurpose,
    BudgetReservation,
    BudgetReservationStatus,
    DecisionQuestion,
    MetaAction,
    MetaDecision,
    MetaDecisionDisposition,
    ResolutionCandidate,
)
from .decision_graph import DecisionGraphController, question_is_admissibly_terminal
from .metrics import AUTONOMY_METRICS_INTERFACE, AutonomyMetrics
from .receding_horizon import (
    PlanSuffixInvalidationReceipt,
    RecedingHorizonController,
    RecedingHorizonEvidence,
)

AUTONOMOUS_META_CONTROLLER_INTERFACE: Final[str] = "AutonomousMetaController@1"
AUTONOMOUS_META_CONTROLLER_SNAPSHOT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/runtime-snapshot@1"
)
AUTONOMY_RUNTIME_INTERFACE: Final[str] = "AutonomyRuntime@1"
AUTONOMY_RUNTIME_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/runtime-loop-snapshot@1"
)
AUTONOMY_CYCLE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/cycle-receipt@1"
)
AUTONOMY_WAKE_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/wake-event@1"
)
MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES = 4 * MAX_CANONICAL_RECORD_BYTES
MAX_ACKNOWLEDGED_CURSORS: Final[int] = 256
DEFAULT_SAFETY_INTERVAL_MS: Final[int] = 300_000
_MAX_SEQUENCE: Final[int] = (1 << 63) - 1

_RUNTIME_WAKE_KIND_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "task_board": "task",
        "task": "task",
        "objective": "objective",
        "repository": "repository",
        "child_process": "provider",
        "lease": "lease",
        "validation": "validation",
        "proof": "proof",
        "provider_capacity": "provider",
        "provider": "provider",
        "policy": "objective",
        "human": "human",
        "budget": "budget",
        "counterexample": "counterexample",
        "freshness": "freshness",
        "observation_window": "window",
        "window": "window",
        "cancellation": "cancellation",
    }
)


class AutonomousMetaControllerError(ValueError):
    """Raised when a composition invariant or restart binding is invalid."""


def _bounded_identifier(value: Any, name: str, *, required: bool = True) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        if required:
            raise AutonomousMetaControllerError(f"{name} must be a bounded identifier")
        return ""
    if (
        len(text.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in text)
        or "\x00" in text
    ):
        raise AutonomousMetaControllerError(f"{name} must be a compact bounded identifier")
    return text


def _bounded_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutonomousMetaControllerError(f"{name} must be a non-negative integer")
    if value < 0 or value > _MAX_SEQUENCE:
        raise AutonomousMetaControllerError(f"{name} is out of bounds")
    return value


def _bounded_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AutonomousMetaControllerError(f"{name} must be a boolean")
    return value


def _parse_snapshot_mapping(
    snapshot: Mapping[str, Any] | str | bytes,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    if isinstance(snapshot, (bytes, str)):
        encoded = snapshot if isinstance(snapshot, bytes) else snapshot.encode("utf-8")
        if len(encoded) > max_bytes:
            raise AutonomousMetaControllerError("runtime snapshot exceeds its bounded size")
        duplicates: set[str] = set()

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    duplicates.add(key)
                value[key] = item
            return value

        try:
            decoded = encoded.decode("utf-8")
            raw = json.loads(decoded, object_pairs_hook=pairs_hook)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AutonomousMetaControllerError("runtime snapshot is malformed") from exc
        if duplicates:
            raise AutonomousMetaControllerError("runtime snapshot contains duplicate fields")
    elif isinstance(snapshot, Mapping):
        raw = dict(snapshot)
        try:
            encoded = canonical_json(raw).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise AutonomousMetaControllerError("runtime snapshot is malformed") from exc
        if len(encoded) > max_bytes:
            raise AutonomousMetaControllerError("runtime snapshot exceeds its bounded size")
    else:
        raise AutonomousMetaControllerError("unsupported runtime snapshot")
    if not isinstance(raw, Mapping):
        raise AutonomousMetaControllerError("runtime snapshot must contain an object")
    return dict(raw)


class AutonomyWakeKind(str, Enum):  # noqa: UP042 - Python 3.8 support
    """Closed vocabulary of meaningful autonomy wakes."""

    REPOSITORY = "repository"
    OBJECTIVE = "objective"
    TASK = "task"
    VALIDATION = "validation"
    PROOF = "proof"
    PROVIDER = "provider"
    LEASE = "lease"
    HUMAN = "human"
    BUDGET = "budget"
    COUNTEREXAMPLE = "counterexample"
    FRESHNESS = "freshness"
    WINDOW = "window"
    CANCELLATION = "cancellation"


AUTONOMY_WAKE_KINDS: Final[tuple[str, ...]] = tuple(item.value for item in AutonomyWakeKind)


def coerce_autonomy_wake_kind(value: Any) -> AutonomyWakeKind:
    """Map a runtime/coordinator kind onto the closed autonomy vocabulary."""

    if isinstance(value, AutonomyWakeKind):
        return value
    raw = str(getattr(value, "value", value) or "").strip().lower()
    mapped = _RUNTIME_WAKE_KIND_ALIASES.get(raw, raw)
    try:
        return AutonomyWakeKind(mapped)
    except ValueError as exc:
        raise AutonomousMetaControllerError("unknown autonomy wake kind") from exc


class AutonomyRuntimeStatus(str, Enum):  # noqa: UP042 - Python 3.8 support
    """Closed outcome of one event-driven cycle."""

    IDLE = "idle"
    PROGRESSING = "progressing"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"
    QUARANTINED = "quarantined"


class BudgetAdmissionStatus(str, Enum):  # noqa: UP042 - Python 3.8 support
    """Closed outcome vocabulary for an injected budget authority."""

    RESERVED = "reserved"
    EXHAUSTED = "exhausted"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BudgetAdmission:
    """Typed result returned by :class:`BudgetController`.

    The budget implementation may update its own immutable ledger internally,
    but must return the resulting ledger so this shell can bind the decision
    to the exact reservation.  Exhaustion and unavailability are ordinary,
    fail-closed results rather than signals to expand a budget.
    """

    status: BudgetAdmissionStatus
    ledger: BudgetLedger
    reservation: BudgetReservation | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, BudgetAdmissionStatus):
            raise AutonomousMetaControllerError("invalid budget admission status")
        if not isinstance(self.ledger, BudgetLedger):
            raise AutonomousMetaControllerError("budget admission requires a ledger")
        if len(self.reason_codes) > 64 or any(
            not item or len(item.encode("utf-8")) > 512 for item in self.reason_codes
        ):
            raise AutonomousMetaControllerError("budget reason codes are unbounded")
        if self.status is BudgetAdmissionStatus.RESERVED:
            if not isinstance(self.reservation, BudgetReservation):
                raise AutonomousMetaControllerError("a reserved admission requires a reservation")
            if self.reservation.status is not BudgetReservationStatus.RESERVED:
                raise AutonomousMetaControllerError("an admitted reservation must remain reserved")
            if self.reservation.budget_id != self.ledger.budget.budget_id:
                raise AutonomousMetaControllerError("reservation and ledger use different budgets")
            if self.reservation.reservation_id not in {
                item.reservation_id for item in self.ledger.reservations
            }:
                raise AutonomousMetaControllerError(
                    "resulting ledger does not contain the reservation"
                )
        elif self.reservation is not None:
            raise AutonomousMetaControllerError(
                "only a reserved admission may expose a reservation"
            )


class BudgetController(Protocol):
    """Narrow injected interface; the cognitive-budget module owns its state."""

    @property
    def ledger(self) -> BudgetLedger:
        """Return the current immutable objective ledger."""

    def reserve_for_candidate(
        self,
        *,
        question: DecisionQuestion,
        candidate: ResolutionCandidate,
        idempotency_key: str,
    ) -> BudgetAdmission:
        """Reserve the candidate's declared maximum cost, or fail closed."""

    def snapshot(self) -> Mapping[str, Any]:
        """Return a bounded restart snapshot owned by the budget module."""


_MODEL_ACTIONS = frozenset(
    {
        MetaAction.CALL_LOCAL_SMALL_MODEL,
        MetaAction.CALL_REMOTE_STANDARD_MODEL,
        MetaAction.CALL_REMOTE_STRONG_MODEL,
    }
)
_VALIDATION_ACTIONS = frozenset(
    {
        MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        MetaAction.RUN_SCHEMA_VALIDATION,
        MetaAction.RUN_TYPE_CHECK,
        MetaAction.RUN_SELECTED_TEST,
        MetaAction.RUN_FULL_VALIDATION,
    }
)


def _purpose_for_action(action: MetaAction) -> BudgetPurpose:
    if action in _MODEL_ACTIONS:
        return BudgetPurpose.MODEL
    if action is MetaAction.RUN_SMT_OR_PROVER:
        return BudgetPurpose.PROOF
    if action in _VALIDATION_ACTIONS:
        return BudgetPurpose.VALIDATION
    if action is MetaAction.REQUEST_HUMAN_DECISION:
        return BudgetPurpose.HUMAN
    if action is MetaAction.GENERATE_BOUNDED_REPAIR:
        return BudgetPurpose.REPAIR
    if action is MetaAction.EXPAND_CONTEXT_REFERENCE:
        return BudgetPurpose.CONTEXT
    if action is MetaAction.REPLAN_AFFECTED_SUFFIX:
        return BudgetPurpose.PLANNING
    return BudgetPurpose.ANALYSIS


def _cost_for_candidate(candidate: ResolutionCandidate) -> CognitiveCost:
    """Translate the action's declared maxima into objective-ledger units."""

    action = candidate.resolution_action
    kind = action.action
    return CognitiveCost(
        total_model_calls=1 if kind in _MODEL_ACTIONS else 0,
        strong_model_calls=1 if kind is MetaAction.CALL_REMOTE_STRONG_MODEL else 0,
        input_tokens=action.token_cost,
        output_tokens=action.token_cost if kind in _MODEL_ACTIONS else 0,
        provider_spend_micros=action.provider_cost_micros,
        proof_time_ms=action.latency_cost_ms if kind is MetaAction.RUN_SMT_OR_PROVER else 0,
        validation_time_ms=action.latency_cost_ms if kind in _VALIDATION_ACTIONS else 0,
        human_questions=1 if kind is MetaAction.REQUEST_HUMAN_DECISION else 0,
        repair_rounds=1 if kind is MetaAction.GENERATE_BOUNDED_REPAIR else 0,
        plan_branches=1 if kind is MetaAction.REPLAN_AFFECTED_SUFFIX else 0,
        context_expansions=1 if kind is MetaAction.EXPAND_CONTEXT_REFERENCE else 0,
        wall_time_ms=action.latency_cost_ms,
    )


class ObjectiveBudgetControllerAdapter:
    """Thin composition adapter over the canonical objective budget ledger.

    It contains no independent accounting state.  Every reservation and
    restart projection is delegated to :class:`ObjectiveCognitiveBudgetLedger`.
    """

    def __init__(self, ledger: ObjectiveCognitiveBudgetLedger) -> None:
        if not isinstance(ledger, ObjectiveCognitiveBudgetLedger):
            raise AutonomousMetaControllerError(
                "objective budget adapter requires ObjectiveCognitiveBudgetLedger"
            )
        self._objective_ledger = ledger

    @property
    def ledger(self) -> BudgetLedger:
        return self._objective_ledger.snapshot()

    def reserve_for_candidate(
        self,
        *,
        question: DecisionQuestion,
        candidate: ResolutionCandidate,
        idempotency_key: str,
    ) -> BudgetAdmission:
        outcome = self._objective_ledger.reserve(
            idempotency_key=idempotency_key,
            question_id=question.question_id,
            action_id=candidate.resolution_action.action_id,
            purpose=_purpose_for_action(candidate.resolution_action.action),
            requested=_cost_for_candidate(candidate),
        )
        current = self._objective_ledger.snapshot()
        if isinstance(outcome, BudgetExhaustion):
            return BudgetAdmission(
                status=BudgetAdmissionStatus.EXHAUSTED,
                ledger=current,
                reason_codes=(
                    "budget_exhausted",
                    outcome.reason.value,
                    outcome.dimension.value,
                    outcome.exhaustion_id,
                ),
            )
        if outcome.status is not BudgetReservationStatus.RESERVED:
            # Reusing the same idempotency key after reconciliation, release,
            # or cancellation must never resurrect spent authority.  A new
            # graph/candidate/policy identity is required before retrying.
            return BudgetAdmission(
                status=BudgetAdmissionStatus.UNAVAILABLE,
                ledger=current,
                reason_codes=("reservation_not_active", outcome.status.value),
            )
        return BudgetAdmission(
            status=BudgetAdmissionStatus.RESERVED,
            ledger=current,
            reservation=outcome,
        )

    def snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType({"ledger": self.ledger.to_record()})

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> ObjectiveBudgetControllerAdapter:
        if not isinstance(snapshot, Mapping) or set(snapshot) != {"ledger"}:
            raise AutonomousMetaControllerError("objective budget adapter snapshot is malformed")
        ledger = BudgetLedger.from_dict(snapshot["ledger"])
        return cls(ObjectiveCognitiveBudgetLedger.from_snapshot(ledger))


class MetaControllerStepStatus(str, Enum):  # noqa: UP042 - Python 3.8 support
    """Closed, non-authoritative result of one receding-horizon step."""

    IDLE = "idle"
    ACTION_ADMITTED = "action_admitted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class MetaControllerStep:
    """Prepared meta-action plus its budget evidence, never an effect permit."""

    status: MetaControllerStepStatus
    graph_id: str
    ledger_id: str
    decision: MetaDecision
    question: DecisionQuestion | None = None
    candidate: ResolutionCandidate | None = None
    reservation: BudgetReservation | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, MetaControllerStepStatus):
            raise AutonomousMetaControllerError("invalid step status")
        if not self.graph_id or not self.ledger_id:
            raise AutonomousMetaControllerError("step must bind graph and ledger")
        if not isinstance(self.decision, MetaDecision):
            raise AutonomousMetaControllerError("step requires a meta-decision")
        admitted = self.status is MetaControllerStepStatus.ACTION_ADMITTED
        if admitted:
            if not all((self.question, self.candidate, self.reservation)):
                raise AutonomousMetaControllerError(
                    "an admitted action requires question, candidate, and reservation"
                )
            assert self.question is not None
            assert self.candidate is not None
            assert self.reservation is not None
            if self.decision.disposition is not MetaDecisionDisposition.SELECTED:
                raise AutonomousMetaControllerError(
                    "an admitted action requires a selected scheduler decision"
                )
            if self.decision.reservation_id != self.reservation.reservation_id:
                raise AutonomousMetaControllerError(
                    "decision is not bound to its budget reservation"
                )
            if self.candidate.question_id != self.question.question_id:
                raise AutonomousMetaControllerError("candidate is bound to a different question")
        elif self.reservation is not None or self.decision.reservation_id:
            raise AutonomousMetaControllerError(
                "a non-admitted step cannot expose reserved authority"
            )

    @property
    def admitted(self) -> bool:
        """Whether a meta-action has budget admission (not effect admission)."""

        return self.status is MetaControllerStepStatus.ACTION_ADMITTED

    @property
    def authorizes_effect(self) -> bool:
        """Always false; only ``DecisionRuntime`` may issue an effect permit."""

        return False

    @property
    def requires_decision_runtime(self) -> bool:
        """Whether a downstream adapter must seek ``DecisionRuntime`` admission."""

        return self.admitted


class DecisionRuntimeAdapter(Protocol):
    """Explicit downstream boundary to the existing effect authority.

    Implementations live with ``context.decision_runtime.DecisionRuntime`` and
    must build its typed request from current authoritative evidence.  The
    autonomous meta-controller intentionally never calls this protocol.
    """

    def admit_with_decision_runtime(self, step: MetaControllerStep) -> object:
        """Return a DecisionRuntime decision/permit for an admitted step."""


@dataclass(frozen=True)
class AutonomyWakeEvent:
    """One content-bound wake.  Cursors, not timestamps, are the identity."""

    kind: AutonomyWakeKind
    cursor_id: str
    sequence: int = 0
    subject_id: str = ""
    evidence_id: str = ""
    cancelled: bool = False
    stale: bool = False
    safety_timer: bool = False
    reason: str = "notification"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", coerce_autonomy_wake_kind(self.kind))
        object.__setattr__(
            self, "cursor_id", _bounded_identifier(self.cursor_id, "cursor_id")
        )
        object.__setattr__(self, "sequence", _bounded_int(self.sequence, "sequence"))
        object.__setattr__(
            self,
            "subject_id",
            _bounded_identifier(self.subject_id, "subject_id", required=False),
        )
        object.__setattr__(
            self,
            "evidence_id",
            _bounded_identifier(self.evidence_id, "evidence_id", required=False),
        )
        object.__setattr__(self, "cancelled", _bounded_bool(self.cancelled, "cancelled"))
        object.__setattr__(self, "stale", _bounded_bool(self.stale, "stale"))
        object.__setattr__(
            self, "safety_timer", _bounded_bool(self.safety_timer, "safety_timer")
        )
        reason = str(self.reason or "notification").strip() or "notification"
        if len(reason.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
            raise AutonomousMetaControllerError("wake reason is unbounded")
        object.__setattr__(self, "reason", reason)
        if self.kind is AutonomyWakeKind.WINDOW:
            object.__setattr__(self, "safety_timer", True)
            if self.reason == "notification":
                object.__setattr__(self, "reason", "safety_timer")
        elif self.safety_timer:
            raise AutonomousMetaControllerError(
                "only a window wake may be marked as the safety timer"
            )
        if self.kind is AutonomyWakeKind.CANCELLATION:
            object.__setattr__(self, "cancelled", True)

    @property
    def event_id(self) -> str:
        return content_identity(
            {
                "schema": AUTONOMY_WAKE_EVENT_SCHEMA,
                "kind": self.kind.value,
                "cursor_id": self.cursor_id,
                "sequence": self.sequence,
                "subject_id": self.subject_id,
                "evidence_id": self.evidence_id,
                "cancelled": self.cancelled,
                "stale": self.stale,
                "safety_timer": self.safety_timer,
            }
        )

    def to_record(self) -> Mapping[str, Any]:
        payload = {
            "schema": AUTONOMY_WAKE_EVENT_SCHEMA,
            "kind": self.kind.value,
            "cursor_id": self.cursor_id,
            "sequence": self.sequence,
            "subject_id": self.subject_id,
            "evidence_id": self.evidence_id,
            "cancelled": self.cancelled,
            "stale": self.stale,
            "safety_timer": self.safety_timer,
            "reason": self.reason,
        }
        payload["event_id"] = self.event_id
        return MappingProxyType(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | AutonomyWakeEvent) -> AutonomyWakeEvent:
        if isinstance(payload, AutonomyWakeEvent):
            return payload
        if not isinstance(payload, Mapping):
            raise AutonomousMetaControllerError("wake event must be an object")
        return cls(
            kind=payload.get("kind", ""),
            cursor_id=payload.get("cursor_id", ""),
            sequence=int(payload.get("sequence", 0) or 0),
            subject_id=str(payload.get("subject_id") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
            cancelled=bool(payload.get("cancelled", False)),
            stale=bool(payload.get("stale", False)),
            safety_timer=bool(payload.get("safety_timer", False)),
            reason=str(payload.get("reason") or "notification"),
        )

    @classmethod
    def from_runtime_wake(cls, event: Any) -> AutonomyWakeEvent:
        """Adapt a coordinator wake without absorbing that coordinator."""

        if isinstance(event, AutonomyWakeEvent):
            return event
        kinds = getattr(event, "kinds", None)
        if kinds:
            kind = kinds[0]
        else:
            kind = getattr(event, "kind", "")
        cursor_ids = tuple(getattr(event, "cursor_ids", ()) or ())
        cursor_id = cursor_ids[0] if cursor_ids else str(getattr(event, "cursor_id", "") or "")
        semantic = getattr(event, "semantic_cursors", None) or {}
        if not cursor_id and isinstance(semantic, Mapping) and semantic:
            cursor_id = str(next(iter(semantic.values())))
        return cls(
            kind=kind,
            cursor_id=cursor_id,
            sequence=_bounded_int(getattr(event, "sequence", 0) or 0, "sequence"),
            subject_id=str(getattr(event, "subject_id", "") or ""),
            evidence_id=str(getattr(event, "evidence_id", "") or ""),
            cancelled=bool(getattr(event, "cancelled", False)),
            stale=bool(getattr(event, "stale", False)),
            safety_timer=bool(getattr(event, "safety_timer", False)),
            reason=str(getattr(event, "reason", "") or "notification"),
        )


class AutonomyCheckpointSink(Protocol):
    """Optional durable sink.  Implementations must skip unchanged payloads."""

    def persist(self, snapshot: Mapping[str, Any]) -> bool:
        """Return True only when a durable write actually occurred."""


class InMemoryAutonomyCheckpointSink:
    """Test/helper sink that refuses to rewrite an identical snapshot."""

    def __init__(self) -> None:
        self._encoded = ""
        self.write_count = 0
        self.last_snapshot: Mapping[str, Any] | None = None

    def persist(self, snapshot: Mapping[str, Any]) -> bool:
        encoded = canonical_json(dict(snapshot))
        if encoded == self._encoded:
            return False
        self._encoded = encoded
        self.write_count += 1
        self.last_snapshot = MappingProxyType(dict(snapshot))
        return True


@dataclass(frozen=True)
class AutonomyCycleResult:
    """One cycle's compact receipt.  Never an effect or completion permit."""

    status: AutonomyRuntimeStatus
    cursor_id: str
    acknowledged: bool
    wrote_state: bool
    model_called: bool
    scanned: bool
    refilled: bool
    safety_timer: bool
    reason_codes: tuple[str, ...] = ()
    step: MetaControllerStep | None = None
    suffix_receipt: PlanSuffixInvalidationReceipt | None = None
    nearest_safe_segment_ids: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    receipt_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, AutonomyRuntimeStatus):
            raise AutonomousMetaControllerError("invalid cycle status")
        object.__setattr__(
            self, "cursor_id", _bounded_identifier(self.cursor_id, "cursor_id")
        )
        for name in (
            "acknowledged",
            "wrote_state",
            "model_called",
            "scanned",
            "refilled",
            "safety_timer",
        ):
            object.__setattr__(self, name, _bounded_bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_bounded_identifier(item, "reason_codes") for item in self.reason_codes),
        )
        if self.step is not None and not isinstance(self.step, MetaControllerStep):
            raise AutonomousMetaControllerError("cycle step must be a MetaControllerStep")
        if self.suffix_receipt is not None and not isinstance(
            self.suffix_receipt, PlanSuffixInvalidationReceipt
        ):
            raise AutonomousMetaControllerError("suffix receipt is not a plan-suffix adapter")
        object.__setattr__(
            self,
            "nearest_safe_segment_ids",
            tuple(
                _bounded_identifier(item, "nearest_safe_segment_ids")
                for item in self.nearest_safe_segment_ids
            ),
        )
        if self.model_called:
            raise AutonomousMetaControllerError(
                "the autonomy runtime is provider-free and cannot call a model"
            )
        if self.refilled:
            raise AutonomousMetaControllerError(
                "the autonomy runtime cannot refill an objective budget"
            )
        metrics = dict(self.metrics) if self.metrics else {}
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        body = {
            "schema": AUTONOMY_CYCLE_RECEIPT_SCHEMA,
            "status": self.status.value,
            "cursor_id": self.cursor_id,
            "acknowledged": self.acknowledged,
            "wrote_state": self.wrote_state,
            "model_called": False,
            "scanned": self.scanned,
            "refilled": False,
            "safety_timer": self.safety_timer,
            "reason_codes": list(self.reason_codes),
            "nearest_safe_segment_ids": list(self.nearest_safe_segment_ids),
            "step_status": None if self.step is None else self.step.status.value,
            "suffix_disposition": (
                None if self.suffix_receipt is None else self.suffix_receipt.disposition.value
            ),
        }
        claimed = self.receipt_id
        identity = content_identity(body)
        if claimed and claimed != identity:
            raise AutonomousMetaControllerError("cycle receipt identity mismatch")
        object.__setattr__(self, "receipt_id", identity)

    @property
    def authorizes_effect(self) -> bool:
        return False

    @property
    def authorizes_completion(self) -> bool:
        return False

    def to_record(self) -> Mapping[str, Any]:
        payload = {
            "schema": AUTONOMY_CYCLE_RECEIPT_SCHEMA,
            "receipt_id": self.receipt_id,
            "status": self.status.value,
            "cursor_id": self.cursor_id,
            "acknowledged": self.acknowledged,
            "wrote_state": self.wrote_state,
            "model_called": self.model_called,
            "scanned": self.scanned,
            "refilled": self.refilled,
            "safety_timer": self.safety_timer,
            "reason_codes": list(self.reason_codes),
            "nearest_safe_segment_ids": list(self.nearest_safe_segment_ids),
        }
        return MappingProxyType(payload)


def _budget_reason(reason_codes: Sequence[str]) -> bool:
    return any(
        "budget_exhausted" in item
        or item.startswith("protected_validation_")
        or item.startswith("protected_proof_")
        for item in reason_codes
    )


def _unavailable_reason(reason_codes: Sequence[str]) -> bool:
    return not reason_codes or any(
        "unavailable" in item or item == "no_resolution_candidate" for item in reason_codes
    )


class AutonomousMetaController:
    """Pure coordinator above graph, scheduler, and objective budget layers.

    One call observes at most one question and reserves at most one action.
    It has no provider client, tool runner, filesystem mutator, planner,
    context compiler, persistence handle, or ``DecisionRuntime`` permit issuer.
    """

    interface: Final[str] = AUTONOMOUS_META_CONTROLLER_INTERFACE

    def __init__(
        self,
        *,
        decision_graph: DecisionGraphController,
        budget_controller: BudgetController | ObjectiveCognitiveBudgetLedger,
        scheduler: CognitiveScheduler | None = None,
    ) -> None:
        if not isinstance(decision_graph, DecisionGraphController):
            raise AutonomousMetaControllerError("decision_graph must be a DecisionGraphController")
        if isinstance(budget_controller, ObjectiveCognitiveBudgetLedger):
            budget_controller = ObjectiveBudgetControllerAdapter(budget_controller)
        if not isinstance(budget_controller.ledger, BudgetLedger):
            raise AutonomousMetaControllerError(
                "budget_controller must expose an immutable BudgetLedger"
            )
        self._decision_graph = decision_graph
        self._budget_controller = budget_controller
        self._scheduler = scheduler or CognitiveScheduler()

    @property
    def decision_graph(self) -> DecisionGraphController:
        return self._decision_graph

    @property
    def budget_controller(self) -> BudgetController:
        return self._budget_controller

    @property
    def scheduler(self) -> CognitiveScheduler:
        return self._scheduler

    def has_work(self) -> bool:
        """Whether at least one mandatory question is still eligible."""

        return self._next_question() is not None

    def _next_question(self) -> DecisionQuestion | None:
        graph = self._decision_graph.graph
        by_id = {item.question_id: item for item in graph.questions}

        # Dependencies of a mandatory question become part of its mandatory
        # closure even if their source contract was initially marked optional.
        required_ids = {item.question_id for item in graph.questions if item.mandatory}
        pending = list(required_ids)
        while pending:
            question_id = pending.pop()
            for dependency_id in by_id[question_id].dependency_question_ids:
                if dependency_id not in required_ids:
                    required_ids.add(dependency_id)
                    pending.append(dependency_id)

        terminal_ids = {
            item.question_id for item in graph.questions if question_is_admissibly_terminal(item)
        }
        eligible = [
            by_id[question_id]
            for question_id in required_ids
            if question_id not in terminal_ids
            and set(by_id[question_id].dependency_question_ids).issubset(terminal_ids)
        ]
        if not eligible:
            return None

        # Expiring decisions first, then higher unresolved risk, then content
        # identity.  All comparisons are closed integers/identifiers.
        return min(
            eligible,
            key=lambda item: (
                item.decision_deadline_ms if item.decision_deadline_ms else (1 << 63) - 1,
                -item.risk_if_left_unresolved.rank,
                item.question_id,
            ),
        )

    def _idle_step(
        self,
        *,
        context: CognitiveSchedulingContext,
        reason_code: str,
    ) -> MetaControllerStep:
        decision = self._scheduler.select(
            question=None,
            candidates=(),
            budget_ledger=self._budget_controller.ledger,
            context=context,
        )
        return MetaControllerStep(
            status=MetaControllerStepStatus.IDLE,
            graph_id=self._decision_graph.graph.graph_id,
            ledger_id=self._budget_controller.ledger.ledger_id,
            decision=decision,
            reason_codes=(reason_code,),
        )

    def step(
        self,
        *,
        candidates: Sequence[ResolutionCandidate],
        context: CognitiveSchedulingContext,
        meaningful_change: bool = True,
    ) -> MetaControllerStep:
        """Prepare one cheapest admissible action without executing it.

        ``meaningful_change=False`` is the event runtime's explicit idle path:
        it performs no graph or budget transition and therefore creates no
        unchanged-state writes.  Candidate order never affects selection.
        """

        if not meaningful_change:
            return self._idle_step(context=context, reason_code="unchanged_state")

        question = self._next_question()
        if question is None:
            return self._idle_step(context=context, reason_code="no_unresolved_mandatory_question")

        applicable = tuple(item for item in candidates if item.question_id == question.question_id)
        decision = self._scheduler.select(
            question=question,
            candidates=applicable,
            budget_ledger=self._budget_controller.ledger,
            context=context,
        )
        if decision.disposition is not MetaDecisionDisposition.SELECTED:
            if _budget_reason(decision.reason_codes):
                status = MetaControllerStepStatus.BUDGET_EXHAUSTED
            elif decision.disposition is MetaDecisionDisposition.QUARANTINE:
                status = MetaControllerStepStatus.QUARANTINED
            elif _unavailable_reason(decision.reason_codes):
                status = MetaControllerStepStatus.UNAVAILABLE
            else:
                status = MetaControllerStepStatus.BLOCKED
            return MetaControllerStep(
                status=status,
                graph_id=self._decision_graph.graph.graph_id,
                ledger_id=self._budget_controller.ledger.ledger_id,
                decision=decision,
                question=question,
                reason_codes=decision.reason_codes,
            )

        selected = {item.candidate_id: item for item in applicable}.get(
            decision.selected_candidate_id
        )
        if selected is None:
            raise AutonomousMetaControllerError(
                "scheduler selected a candidate outside the named question"
            )
        idempotency_key = content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/budget-admission-key@1",
                "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
                "graph_id": self._decision_graph.graph.graph_id,
                "question_id": question.question_id,
                "candidate_id": selected.candidate_id,
                "policy_id": decision.policy_id,
            }
        )
        admission = self._budget_controller.reserve_for_candidate(
            question=question,
            candidate=selected,
            idempotency_key=idempotency_key,
        )
        if not isinstance(admission, BudgetAdmission):
            raise AutonomousMetaControllerError(
                "budget controller returned an invalid admission result"
            )
        if admission.status is BudgetAdmissionStatus.EXHAUSTED:
            return MetaControllerStep(
                status=MetaControllerStepStatus.BUDGET_EXHAUSTED,
                graph_id=self._decision_graph.graph.graph_id,
                ledger_id=admission.ledger.ledger_id,
                decision=decision,
                question=question,
                candidate=selected,
                reason_codes=admission.reason_codes or ("budget_exhausted",),
            )
        if admission.status is BudgetAdmissionStatus.UNAVAILABLE:
            return MetaControllerStep(
                status=MetaControllerStepStatus.UNAVAILABLE,
                graph_id=self._decision_graph.graph.graph_id,
                ledger_id=admission.ledger.ledger_id,
                decision=decision,
                question=question,
                candidate=selected,
                reason_codes=admission.reason_codes or ("budget_unavailable",),
            )

        reservation = admission.reservation
        assert reservation is not None
        if reservation.question_id != question.question_id:
            raise AutonomousMetaControllerError("reservation is bound to a different question")
        if reservation.action_id != selected.resolution_action.action_id:
            raise AutonomousMetaControllerError("reservation is bound to a different action")
        admitted_decision = replace(decision, reservation_id=reservation.reservation_id)
        return MetaControllerStep(
            status=MetaControllerStepStatus.ACTION_ADMITTED,
            graph_id=self._decision_graph.graph.graph_id,
            ledger_id=admission.ledger.ledger_id,
            decision=admitted_decision,
            question=question,
            candidate=selected,
            reservation=reservation,
            reason_codes=("scheduler_selected", "objective_budget_reserved"),
        )

    def snapshot(self) -> Mapping[str, Any]:
        """Return a content-bound restart value without storing it."""

        budget_snapshot = self._budget_controller.snapshot()
        if not isinstance(budget_snapshot, Mapping):
            raise AutonomousMetaControllerError("budget controller snapshot must be a mapping")
        payload: dict[str, Any] = {
            "schema": AUTONOMOUS_META_CONTROLLER_SNAPSHOT_SCHEMA,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "graph_id": self._decision_graph.graph.graph_id,
            "ledger_id": self._budget_controller.ledger.ledger_id,
            "decision_graph": self._decision_graph.graph.to_record(),
            "cognitive_budget": dict(budget_snapshot),
        }
        payload["snapshot_id"] = content_identity(payload)
        if len(canonical_json(payload).encode("utf-8")) > MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES:
            raise AutonomousMetaControllerError("runtime snapshot exceeds its bounded size")
        return MappingProxyType(payload)

    def snapshot_json(self) -> str:
        return canonical_json(dict(self.snapshot()))

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any] | str | bytes,
        *,
        budget_loader: Callable[[Mapping[str, Any]], BudgetController],
        scheduler: CognitiveScheduler | None = None,
    ) -> AutonomousMetaController:
        """Rebuild from a checked snapshot using the budget owner's loader."""

        raw = _parse_snapshot_mapping(
            snapshot, max_bytes=MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES
        )
        expected_fields = {
            "schema",
            "program_id",
            "graph_id",
            "ledger_id",
            "decision_graph",
            "cognitive_budget",
            "snapshot_id",
        }
        if set(raw) != expected_fields:
            raise AutonomousMetaControllerError(
                "runtime snapshot contains missing or unknown fields"
            )
        if raw["schema"] != AUTONOMOUS_META_CONTROLLER_SNAPSHOT_SCHEMA:
            raise AutonomousMetaControllerError("runtime snapshot schema mismatch")
        if raw["program_id"] != AUTONOMOUS_META_CONTROLLER_PROGRAM_ID:
            raise AutonomousMetaControllerError("runtime program identity mismatch")
        claimed_id = raw.pop("snapshot_id")
        if claimed_id != content_identity(raw):
            raise AutonomousMetaControllerError("runtime snapshot identity mismatch")
        if not isinstance(raw["decision_graph"], Mapping) or not isinstance(
            raw["cognitive_budget"], Mapping
        ):
            raise AutonomousMetaControllerError("runtime snapshot bodies are malformed")
        graph_controller = DecisionGraphController.from_snapshot(raw["decision_graph"])
        budget_controller = budget_loader(raw["cognitive_budget"])
        if raw["graph_id"] != graph_controller.graph.graph_id:
            raise AutonomousMetaControllerError("runtime graph binding mismatch")
        if raw["ledger_id"] != budget_controller.ledger.ledger_id:
            raise AutonomousMetaControllerError("runtime ledger binding mismatch")
        return cls(
            decision_graph=graph_controller,
            budget_controller=budget_controller,
            scheduler=scheduler,
        )


def _status_from_step(step: MetaControllerStep) -> AutonomyRuntimeStatus:
    mapping = {
        MetaControllerStepStatus.IDLE: AutonomyRuntimeStatus.IDLE,
        MetaControllerStepStatus.ACTION_ADMITTED: AutonomyRuntimeStatus.PROGRESSING,
        MetaControllerStepStatus.BUDGET_EXHAUSTED: AutonomyRuntimeStatus.EXHAUSTED,
        MetaControllerStepStatus.UNAVAILABLE: AutonomyRuntimeStatus.UNAVAILABLE,
        MetaControllerStepStatus.BLOCKED: AutonomyRuntimeStatus.BLOCKED,
        MetaControllerStepStatus.QUARANTINED: AutonomyRuntimeStatus.QUARANTINED,
    }
    try:
        return mapping[step.status]
    except KeyError as exc:
        raise AutonomousMetaControllerError("unmapped meta-controller step status") from exc


class AutonomyRuntime:
    """Event-driven loop over the provider-free meta-controller.

    Lower services retain their own state.  This class consumes already
    characterized wake events, selects at most one nearest-safe action, and
    emits compact receipts.  A two-phase acknowledgement advances a cursor
    only after the cycle; a crash before acknowledgement therefore replays.
    """

    interface: Final[str] = AUTONOMY_RUNTIME_INTERFACE

    def __init__(
        self,
        *,
        controller: AutonomousMetaController,
        metrics: AutonomyMetrics | None = None,
        horizon: RecedingHorizonController | None = None,
        checkpoint_sink: AutonomyCheckpointSink | None = None,
        safety_interval_ms: int = DEFAULT_SAFETY_INTERVAL_MS,
        now_ms: int = 0,
    ) -> None:
        if not isinstance(controller, AutonomousMetaController):
            raise AutonomousMetaControllerError(
                "controller must be an AutonomousMetaController"
            )
        if horizon is not None and not isinstance(horizon, RecedingHorizonController):
            raise AutonomousMetaControllerError(
                "horizon must be a RecedingHorizonController or None"
            )
        interval = _bounded_int(safety_interval_ms, "safety_interval_ms")
        if interval <= 0:
            raise AutonomousMetaControllerError("safety_interval_ms must be positive")
        self._controller = controller
        self._metrics = metrics if metrics is not None else AutonomyMetrics()
        if not isinstance(self._metrics, AutonomyMetrics):
            raise AutonomousMetaControllerError("metrics must be AutonomyMetrics")
        self._horizon = horizon
        self._checkpoint_sink = checkpoint_sink
        self._safety_interval_ms = interval
        self._next_safety_deadline_ms = _bounded_int(now_ms, "now_ms") + interval
        self._acknowledged_cursor_ids: list[str] = []
        self._acked_index: set[str] = set()
        self._ephemeral_cursor_ids: set[str] = set()
        self._pending_cursor_id = ""
        self._healthy_idle = self._board_is_idle()
        self._healthy_exhausted = False
        self._last_durable_identity = self._durable_identity()

    @property
    def controller(self) -> AutonomousMetaController:
        return self._controller

    @property
    def metrics(self) -> AutonomyMetrics:
        return self._metrics

    @property
    def horizon(self) -> RecedingHorizonController | None:
        return self._horizon

    @property
    def safety_interval_ms(self) -> int:
        return self._safety_interval_ms

    @property
    def healthy_idle(self) -> bool:
        return self._healthy_idle

    @property
    def healthy_exhausted(self) -> bool:
        return self._healthy_exhausted

    @property
    def acknowledged_cursor_ids(self) -> tuple[str, ...]:
        return tuple(self._acknowledged_cursor_ids)

    def _board_is_idle(self) -> bool:
        horizon_idle = True if self._horizon is None else self._horizon.idle
        return (not self._controller.has_work()) and horizon_idle

    def _remember_cursor(self, cursor_id: str) -> None:
        if cursor_id in self._acked_index:
            return
        self._acknowledged_cursor_ids.append(cursor_id)
        self._acked_index.add(cursor_id)
        overflow = len(self._acknowledged_cursor_ids) - MAX_ACKNOWLEDGED_CURSORS
        if overflow > 0:
            dropped = self._acknowledged_cursor_ids[:overflow]
            del self._acknowledged_cursor_ids[:overflow]
            for item in dropped:
                self._acked_index.discard(item)

    def acknowledge(self, event: AutonomyWakeEvent | Mapping[str, Any]) -> None:
        bound = AutonomyWakeEvent.from_dict(event)
        if bound.safety_timer or bound.kind is AutonomyWakeKind.WINDOW:
            self._ephemeral_cursor_ids.add(bound.cursor_id)
            if len(self._ephemeral_cursor_ids) > MAX_ACKNOWLEDGED_CURSORS:
                self._ephemeral_cursor_ids.clear()
                self._ephemeral_cursor_ids.add(bound.cursor_id)
        else:
            self._remember_cursor(bound.cursor_id)
        if self._pending_cursor_id == bound.cursor_id:
            self._pending_cursor_id = ""

    def safety_timer_event(self, *, now_ms: int) -> AutonomyWakeEvent | None:
        """Emit a window wake only when the bounded safety interval elapsed."""

        current = _bounded_int(now_ms, "now_ms")
        if current < self._next_safety_deadline_ms:
            return None
        self._next_safety_deadline_ms = current + self._safety_interval_ms
        return AutonomyWakeEvent(
            kind=AutonomyWakeKind.WINDOW,
            cursor_id=f"window:{current}",
            sequence=current,
            safety_timer=True,
            reason="safety_timer",
        )

    def _persist_if_changed(self) -> bool:
        durable = self._durable_identity()
        if durable == self._last_durable_identity:
            return False
        snapshot = self.snapshot()
        wrote = True
        if self._checkpoint_sink is not None:
            wrote = bool(self._checkpoint_sink.persist(snapshot))
        if wrote:
            self._metrics.record_write()
            self._last_durable_identity = self._durable_identity()
        else:
            self._last_durable_identity = durable
        return wrote

    def _result(
        self,
        *,
        status: AutonomyRuntimeStatus,
        event: AutonomyWakeEvent,
        reason_codes: tuple[str, ...],
        scanned: bool,
        wrote_state: bool,
        acknowledged: bool,
        step: MetaControllerStep | None = None,
        suffix_receipt: PlanSuffixInvalidationReceipt | None = None,
    ) -> AutonomyCycleResult:
        segment_ids: tuple[str, ...] = ()
        if suffix_receipt is not None:
            segment_ids = suffix_receipt.nearest_safe_segment_ids
        elif self._horizon is not None and scanned:
            segment_ids = self._horizon.select_nearest_safe_segment().step_ids
        return AutonomyCycleResult(
            status=status,
            cursor_id=event.cursor_id,
            acknowledged=acknowledged,
            wrote_state=wrote_state,
            model_called=False,
            scanned=scanned,
            refilled=False,
            safety_timer=event.safety_timer,
            reason_codes=reason_codes,
            step=step,
            suffix_receipt=suffix_receipt,
            nearest_safe_segment_ids=segment_ids,
            metrics=self._metrics.snapshot(),
        )

    def handle_wake(
        self,
        event: AutonomyWakeEvent | Mapping[str, Any] | Any,
        *,
        candidates: Sequence[ResolutionCandidate] = (),
        context: CognitiveSchedulingContext,
        horizon_evidence: RecedingHorizonEvidence | Mapping[str, Any] | None = None,
        auto_acknowledge: bool = True,
        cancelled: bool = False,
        now_ms: int | None = None,
        deadline_milliseconds: int | None = None,
    ) -> AutonomyCycleResult:
        """Process one wake without executing an effect or calling a model."""

        bound = (
            event
            if isinstance(event, AutonomyWakeEvent)
            else AutonomyWakeEvent.from_runtime_wake(event)
            if not isinstance(event, Mapping)
            else AutonomyWakeEvent.from_dict(event)
        )
        if cancelled:
            bound = replace(bound, cancelled=True)
        if now_ms is not None:
            current = _bounded_int(now_ms, "now_ms")
            if current >= self._next_safety_deadline_ms:
                self._next_safety_deadline_ms = current + self._safety_interval_ms
        self._metrics.record_wake(bound.kind, safety_timer=bound.safety_timer)
        if bound.cursor_id in self._acked_index or bound.cursor_id in self._ephemeral_cursor_ids:
            self._metrics.record_idle(
                status="idle",
                reason_codes=("duplicate_cursor",),
            )
            return self._result(
                status=AutonomyRuntimeStatus.IDLE,
                event=bound,
                reason_codes=("duplicate_cursor",),
                scanned=False,
                wrote_state=False,
                acknowledged=True,
            )

        self._pending_cursor_id = bound.cursor_id
        if bound.cancelled or bound.kind is AutonomyWakeKind.CANCELLATION:
            self._metrics.record_status("cancelled", reason_codes=("cancelled",))
            self._healthy_idle = False
            wrote = self._persist_if_changed()
            acknowledged = False
            if auto_acknowledge:
                self.acknowledge(bound)
                acknowledged = True
            return self._result(
                status=AutonomyRuntimeStatus.CANCELLED,
                event=bound,
                reason_codes=("cancelled",),
                scanned=False,
                wrote_state=wrote,
                acknowledged=acknowledged,
            )
        if bound.stale:
            self._metrics.record_status(
                "blocked", reason_codes=("stale_evidence",)
            )
            wrote = self._persist_if_changed()
            acknowledged = False
            if auto_acknowledge:
                self.acknowledge(bound)
                acknowledged = True
            return self._result(
                status=AutonomyRuntimeStatus.BLOCKED,
                event=bound,
                reason_codes=("stale_evidence",),
                scanned=False,
                wrote_state=wrote,
                acknowledged=acknowledged,
            )

        cached_idle = self._healthy_idle or self._healthy_exhausted
        if bound.safety_timer and cached_idle:
            reason = (
                "healthy_exhaustion"
                if self._healthy_exhausted
                else "unchanged_complete_board"
            )
            self._metrics.record_idle(status="idle", reason_codes=(reason,))
            acknowledged = False
            if auto_acknowledge:
                self.acknowledge(bound)
                acknowledged = True
            return self._result(
                status=(
                    AutonomyRuntimeStatus.EXHAUSTED
                    if self._healthy_exhausted
                    else AutonomyRuntimeStatus.IDLE
                ),
                event=bound,
                reason_codes=(reason,),
                scanned=False,
                wrote_state=False,
                acknowledged=acknowledged,
            )

        suffix_receipt: PlanSuffixInvalidationReceipt | None = None
        if self._horizon is not None and horizon_evidence is not None:
            suffix_receipt = self._horizon.observe(
                horizon_evidence,
                now_milliseconds=now_ms,
                deadline_milliseconds=deadline_milliseconds,
                cancelled=bound.cancelled,
            )

        self._metrics.record_scan()
        idle_now = self._board_is_idle()
        self._healthy_idle = idle_now and not self._healthy_exhausted
        if idle_now and not self._healthy_exhausted:
            self._metrics.record_idle(
                status="idle",
                reason_codes=("no_unresolved_mandatory_question",),
            )
            wrote = self._persist_if_changed()
            acknowledged = False
            if auto_acknowledge:
                self.acknowledge(bound)
                acknowledged = True
            return self._result(
                status=AutonomyRuntimeStatus.IDLE,
                event=bound,
                reason_codes=("no_unresolved_mandatory_question",),
                scanned=True,
                wrote_state=wrote,
                acknowledged=acknowledged,
                suffix_receipt=suffix_receipt,
            )

        step = self._controller.step(
            candidates=candidates,
            context=context,
            meaningful_change=True,
        )
        status = _status_from_step(step)
        if step.admitted:
            action = (
                None
                if step.candidate is None
                else step.candidate.resolution_action.action
            )
            self._metrics.record_model_action(action)
        self._metrics.record_status(status.value, reason_codes=step.reason_codes)
        self._healthy_exhausted = status is AutonomyRuntimeStatus.EXHAUSTED
        self._healthy_idle = status is AutonomyRuntimeStatus.IDLE
        wrote = self._persist_if_changed()
        acknowledged = False
        if auto_acknowledge:
            self.acknowledge(bound)
            acknowledged = True
        return self._result(
            status=status,
            event=bound,
            reason_codes=step.reason_codes,
            scanned=True,
            wrote_state=wrote,
            acknowledged=acknowledged,
            step=step,
            suffix_receipt=suffix_receipt,
        )

    ingest = handle_wake
    run_cycle = handle_wake

    def _durable_body(self) -> dict[str, Any]:
        horizon_payload: Mapping[str, Any] | None
        if self._horizon is None:
            horizon_payload = None
        else:
            horizon_payload = dict(self._horizon.snapshot())
        return {
            "schema": AUTONOMY_RUNTIME_SNAPSHOT_SCHEMA,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "interface": AUTONOMY_RUNTIME_INTERFACE,
            "controller": dict(self._controller.snapshot()),
            "metrics": dict(self._metrics.durable_snapshot()),
            "acknowledged_cursor_ids": list(self._acknowledged_cursor_ids),
            "healthy_idle": self._healthy_idle,
            "healthy_exhausted": self._healthy_exhausted,
            "safety_interval_ms": self._safety_interval_ms,
            "horizon": horizon_payload,
            "metrics_interface": AUTONOMY_METRICS_INTERFACE,
        }

    def _material_body(self) -> dict[str, Any]:
        payload = self._durable_body()
        payload.pop("acknowledged_cursor_ids", None)
        return payload

    def _durable_identity(self) -> str:
        return content_identity(self._material_body())

    def snapshot(self) -> Mapping[str, Any]:
        payload = self._durable_body()
        payload["snapshot_id"] = content_identity(payload)
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES:
            raise AutonomousMetaControllerError("runtime snapshot exceeds its bounded size")
        return MappingProxyType(payload)

    def snapshot_json(self) -> str:
        return canonical_json(dict(self.snapshot()))

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any] | str | bytes,
        *,
        budget_loader: Callable[[Mapping[str, Any]], BudgetController],
        scheduler: CognitiveScheduler | None = None,
        metrics: AutonomyMetrics | None = None,
        horizon: RecedingHorizonController | None = None,
        checkpoint_sink: AutonomyCheckpointSink | None = None,
        now_ms: int = 0,
    ) -> AutonomyRuntime:
        raw = _parse_snapshot_mapping(
            snapshot, max_bytes=MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES
        )
        expected = {
            "schema",
            "program_id",
            "interface",
            "controller",
            "metrics",
            "acknowledged_cursor_ids",
            "healthy_idle",
            "healthy_exhausted",
            "safety_interval_ms",
            "horizon",
            "metrics_interface",
            "snapshot_id",
        }
        if set(raw) != expected:
            raise AutonomousMetaControllerError(
                "runtime snapshot contains missing or unknown fields"
            )
        if raw["schema"] != AUTONOMY_RUNTIME_SNAPSHOT_SCHEMA:
            raise AutonomousMetaControllerError("runtime snapshot schema mismatch")
        if raw["program_id"] != AUTONOMOUS_META_CONTROLLER_PROGRAM_ID:
            raise AutonomousMetaControllerError("runtime program identity mismatch")
        if raw["interface"] != AUTONOMY_RUNTIME_INTERFACE:
            raise AutonomousMetaControllerError("runtime interface mismatch")
        if raw["metrics_interface"] != AUTONOMY_METRICS_INTERFACE:
            raise AutonomousMetaControllerError("metrics interface mismatch")
        claimed_id = raw.pop("snapshot_id")
        if claimed_id != content_identity(raw):
            raise AutonomousMetaControllerError("runtime snapshot identity mismatch")
        if not isinstance(raw["controller"], Mapping) or not isinstance(
            raw["metrics"], Mapping
        ):
            raise AutonomousMetaControllerError("runtime snapshot bodies are malformed")
        controller = AutonomousMetaController.from_snapshot(
            raw["controller"],
            budget_loader=budget_loader,
            scheduler=scheduler,
        )
        restored_metrics = metrics or AutonomyMetrics.from_snapshot(raw["metrics"])
        restored_horizon = horizon
        if restored_horizon is None and raw["horizon"] is not None:
            if not isinstance(raw["horizon"], Mapping):
                raise AutonomousMetaControllerError("horizon snapshot is malformed")
            restored_horizon = RecedingHorizonController.from_snapshot(raw["horizon"])
        runtime = cls(
            controller=controller,
            metrics=restored_metrics,
            horizon=restored_horizon,
            checkpoint_sink=checkpoint_sink,
            safety_interval_ms=raw["safety_interval_ms"],
            now_ms=now_ms,
        )
        cursors = raw["acknowledged_cursor_ids"]
        if isinstance(cursors, str) or not isinstance(cursors, (list, tuple)):
            raise AutonomousMetaControllerError("acknowledged cursors must be a sequence")
        if len(cursors) > MAX_ACKNOWLEDGED_CURSORS:
            raise AutonomousMetaControllerError("acknowledged cursors exceed the bound")
        for cursor_id in cursors:
            runtime._remember_cursor(_bounded_identifier(cursor_id, "cursor_id"))
        runtime._healthy_idle = _bounded_bool(raw["healthy_idle"], "healthy_idle")
        runtime._healthy_exhausted = _bounded_bool(
            raw["healthy_exhausted"], "healthy_exhausted"
        )
        runtime._last_durable_identity = runtime._durable_identity()
        return runtime


__all__ = [
    "AUTONOMOUS_META_CONTROLLER_INTERFACE",
    "AUTONOMOUS_META_CONTROLLER_SNAPSHOT_SCHEMA",
    "AUTONOMY_CYCLE_RECEIPT_SCHEMA",
    "AUTONOMY_RUNTIME_INTERFACE",
    "AUTONOMY_RUNTIME_SNAPSHOT_SCHEMA",
    "AUTONOMY_WAKE_EVENT_SCHEMA",
    "AUTONOMY_WAKE_KINDS",
    "DEFAULT_SAFETY_INTERVAL_MS",
    "MAX_ACKNOWLEDGED_CURSORS",
    "MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES",
    "AutonomyCheckpointSink",
    "AutonomyCycleResult",
    "AutonomyRuntime",
    "AutonomyRuntimeStatus",
    "AutonomyWakeEvent",
    "AutonomyWakeKind",
    "AutonomousMetaController",
    "AutonomousMetaControllerError",
    "BudgetAdmission",
    "BudgetAdmissionStatus",
    "BudgetController",
    "DecisionRuntimeAdapter",
    "InMemoryAutonomyCheckpointSink",
    "MetaControllerStep",
    "MetaControllerStepStatus",
    "ObjectiveBudgetControllerAdapter",
    "coerce_autonomy_wake_kind",
]
