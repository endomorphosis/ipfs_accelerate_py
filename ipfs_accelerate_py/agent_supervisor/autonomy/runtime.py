"""Provider-free composition shell for the autonomous meta-controller.

``AutonomousMetaController`` names one unresolved decision, delegates all
action selection to :class:`~.cognitive_scheduler.CognitiveScheduler`, and
reserves objective budget before exposing the selected meta-action.  It does
not execute an action, call a model, write durable state, admit a repository
effect, or authorize task completion.

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
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from ..proof.formal_verification_contracts import canonical_json, content_identity
from .cognitive_budget import CognitiveCost, ObjectiveCognitiveBudgetLedger
from .cognitive_scheduler import CognitiveScheduler, CognitiveSchedulingContext
from .contracts import (
    AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
    MAX_CANONICAL_RECORD_BYTES,
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

AUTONOMOUS_META_CONTROLLER_SNAPSHOT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/runtime-snapshot@1"
)
MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES = 4 * MAX_CANONICAL_RECORD_BYTES


class AutonomousMetaControllerError(ValueError):
    """Raised when a composition invariant or restart binding is invalid."""


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

        if isinstance(snapshot, (bytes, str)):
            encoded = snapshot if isinstance(snapshot, bytes) else snapshot.encode("utf-8")
            if len(encoded) > MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES:
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
            if len(encoded) > MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES:
                raise AutonomousMetaControllerError("runtime snapshot exceeds its bounded size")
        else:
            raise AutonomousMetaControllerError("unsupported runtime snapshot")
        if not isinstance(raw, Mapping):
            raise AutonomousMetaControllerError("runtime snapshot must contain an object")
        raw = dict(raw)
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


__all__ = [
    "AUTONOMOUS_META_CONTROLLER_SNAPSHOT_SCHEMA",
    "MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES",
    "AutonomousMetaController",
    "AutonomousMetaControllerError",
    "BudgetAdmission",
    "BudgetAdmissionStatus",
    "BudgetController",
    "DecisionRuntimeAdapter",
    "MetaControllerStep",
    "MetaControllerStepStatus",
    "ObjectiveBudgetControllerAdapter",
]
