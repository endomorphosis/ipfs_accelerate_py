"""Production event adapters for residual evaluation and event-driven reassessment.

Adapters normalize scheduler, validation, review, merge, Doctor, retry,
drift, low-water, and open-goal signals into a single observation surface.
They never authorize append or completion themselves.

Event-driven reassessment (DOEP-051) extends this same adapter.  Given
already-delivered authoritative events, a live plan epoch, and an explicit
plan surface, it applies idempotent in-memory consumption, fails closed on
stale plan epochs, and routes to incremental plan-impact analysis.  It does
not own an event log, planner, PlanDelta, or refill CAS append.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..analysis.dynamic_impact_frontier import (
    INCREMENTAL_PLAN_IMPACT_SCHEMA,
    AuthoritativeImpactEvent,
    FrontierObservation,
    ImpactFrontierEntry,
    IncrementalPlanImpactAnalysis,
    OrdinaryRefillDisposition,
    PlanImpactNode,
    analyze_incremental_plan_impact,
)
from ..analysis.change_propagation_contracts import ImpactClosureReceipt
from ..runtime.database_event_log import IDEMPOTENT_EVENT_CONSUMPTION_BINDING
from ..task_sources.plan_revision_store import (
    STALE_PLAN_EPOCH_BINDING,
    assert_plan_epoch_current,
)
from .refill_controller import RefillObservation


PRODUCTION_EVENT_ADAPTER_MANIFEST: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-refill-event-adapter@1"
)
EVENT_DRIVEN_REASSESSMENT_BINDING: Final[str] = "EventDrivenReassessment@1"
EVENT_DRIVEN_REASSESSMENT_INTERFACE: Final[str] = EVENT_DRIVEN_REASSESSMENT_BINDING
EVENT_DRIVEN_REASSESSMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/event-driven-reassessment@1"
)
EVENT_DRIVEN_REASSESSMENT_CONSUMES: Final[tuple[str, ...]] = (
    IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
    STALE_PLAN_EPOCH_BINDING,
    INCREMENTAL_PLAN_IMPACT_SCHEMA,
)
PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE: Final[str] = (
    "ProductionRefillEventAdapter@1"
)

_EVENT_KINDS: Final[tuple[str, ...]] = (
    "scheduler_low_water",
    "scheduler_drained_open_goal",
    "validation_rejected",
    "review_rejected",
    "merge_rejected",
    "doctor_finding",
    "retry_exhausted",
    "actionable_drift",
    "rollout_threshold_missed",
    "stale_evidence",
    "branch_only_completion",
)

_SEED_LIST_KEYS: Final[tuple[str, ...]] = (
    "seed_node_ids",
    "affected_task_ids",
    "node_ids",
)
_SEED_SCALAR_KEYS: Final[tuple[str, ...]] = (
    "seed_node_id",
    "node_id",
    "task_id",
)


class EventDrivenReassessmentError(ValueError):
    """Malformed reassessment input or authority fence."""


class EventDrivenReassessmentDisposition(str, Enum):
    """Closed outcomes for one event-driven reassessment pass."""

    NO_REASSESSMENT = "no_reassessment"
    REASSESSED = "reassessed"
    REPLAYED = "replayed"
    BLOCKED = "blocked"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, AuthoritativeImpactEvent):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    raise EventDrivenReassessmentError(f"{name} must be a mapping")


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = event.get("payload")
    return nested if isinstance(nested, Mapping) else {}


def _compact(value: Any) -> str:
    return str(value or "").strip()


def _event_id(event: Mapping[str, Any]) -> str:
    payload = _payload(event)
    event_id = _compact(event.get("event_id")) or _compact(payload.get("event_id"))
    if not event_id:
        raise EventDrivenReassessmentError("event_id is required")
    if any(character.isspace() for character in event_id):
        raise EventDrivenReassessmentError("event_id must be a compact identifier")
    return event_id


def _event_plan_epoch(event: Mapping[str, Any], default: int | None) -> int:
    payload = _payload(event)
    raw = event.get("plan_epoch", payload.get("plan_epoch", default))
    if raw is None or raw == "":
        raise EventDrivenReassessmentError("plan_epoch is required")
    try:
        epoch = int(raw)
    except (TypeError, ValueError) as exc:
        raise EventDrivenReassessmentError("plan_epoch must be an integer") from exc
    if epoch < 1:
        raise EventDrivenReassessmentError("plan_epoch must be >= 1")
    return epoch


def _event_plan_root(event: Mapping[str, Any], fallback: str) -> str:
    payload = _payload(event)
    return (
        _compact(event.get("plan_root"))
        or _compact(event.get("plan_root_cid"))
        or _compact(payload.get("plan_root"))
        or _compact(payload.get("plan_root_cid"))
        or fallback
    )


def _extend_ids(collected: list[str], value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, (str, bytes, bytearray)):
        text = _compact(value)
        if text:
            collected.append(text)
        return
    if isinstance(value, Sequence):
        for item in value:
            text = _compact(item)
            if text:
                collected.append(text)
        return
    text = _compact(value)
    if text:
        collected.append(text)


def _event_delta_id(event: Mapping[str, Any]) -> str:
    payload = _payload(event)
    return _compact(event.get("delta_id")) or _compact(payload.get("delta_id"))


def _event_roots(event: Mapping[str, Any]) -> Any:
    payload = _payload(event)
    if "roots" in event:
        return event.get("roots")
    return payload.get("roots")


def _event_evidence_refs(event: Mapping[str, Any]) -> tuple[str, ...]:
    payload = _payload(event)
    collected: list[str] = []
    _extend_ids(collected, event.get("evidence_refs"))
    _extend_ids(collected, payload.get("evidence_refs"))
    event_id = _compact(event.get("event_id")) or _compact(payload.get("event_id"))
    if event_id:
        collected.append(event_id)
    return _unique_ids(collected)


def _seed_node_ids(event: Mapping[str, Any]) -> tuple[str, ...]:
    collected: list[str] = []
    for source in (event, _payload(event)):
        for key in _SEED_LIST_KEYS:
            _extend_ids(collected, source.get(key))
        for key in _SEED_SCALAR_KEYS:
            _extend_ids(collected, source.get(key))
    unique: list[str] = []
    seen: set[str] = set()
    for item in collected:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def _unique_ids(values: Sequence[str]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _compact(item)
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return tuple(unique)


def _observation_dict(observation: RefillObservation) -> dict[str, Any]:
    return {
        "plan_root_cid": observation.plan_root_cid,
        "revision": observation.revision,
        "ready_tasks": observation.ready_tasks,
        "active_tasks": observation.active_tasks,
        "open_goals": observation.open_goals,
        "validation_rejected": observation.validation_rejected,
        "review_rejected": observation.review_rejected,
        "merge_rejected": observation.merge_rejected,
        "stale_evidence": observation.stale_evidence,
        "branch_only_completion": observation.branch_only_completion,
        "actionable_drift": observation.actionable_drift,
        "retry_exhausted_with_refinement": observation.retry_exhausted_with_refinement,
        "rollout_threshold_missed": observation.rollout_threshold_missed,
    }


def _disposition_for(
    analysis: IncrementalPlanImpactAnalysis | None,
    *,
    applied_event_ids: tuple[str, ...],
    replayed_event_ids: tuple[str, ...],
) -> EventDrivenReassessmentDisposition:
    if not applied_event_ids:
        if replayed_event_ids:
            return EventDrivenReassessmentDisposition.REPLAYED
        return EventDrivenReassessmentDisposition.NO_REASSESSMENT
    if analysis is None:
        return EventDrivenReassessmentDisposition.NO_REASSESSMENT
    disposition = analysis.refill_decision.disposition
    if disposition is OrdinaryRefillDisposition.REFILL_AFFECTED_SUFFIX:
        return EventDrivenReassessmentDisposition.REASSESSED
    if disposition is OrdinaryRefillDisposition.NO_REFILL:
        return EventDrivenReassessmentDisposition.NO_REASSESSMENT
    return EventDrivenReassessmentDisposition.BLOCKED


@dataclass(frozen=True)
class EventDrivenReassessment:
    """Advisory event-driven reassessment receipt; never completion authority."""

    disposition: EventDrivenReassessmentDisposition
    observation: RefillObservation
    live_plan_epoch: int
    applied_event_ids: tuple[str, ...] = ()
    replayed_event_ids: tuple[str, ...] = ()
    consumed_event_ids: tuple[str, ...] = ()
    seed_node_ids: tuple[str, ...] = ()
    impact: IncrementalPlanImpactAnalysis | None = None
    schema: str = EVENT_DRIVEN_REASSESSMENT_SCHEMA
    binding: str = EVENT_DRIVEN_REASSESSMENT_BINDING
    carrier: str = PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, EventDrivenReassessmentDisposition):
            object.__setattr__(
                self,
                "disposition",
                EventDrivenReassessmentDisposition(str(self.disposition)),
            )
        if not isinstance(self.observation, RefillObservation):
            raise EventDrivenReassessmentError("observation must be RefillObservation")
        epoch = int(self.live_plan_epoch)
        if epoch < 1:
            raise EventDrivenReassessmentError("live_plan_epoch must be >= 1")
        object.__setattr__(self, "live_plan_epoch", epoch)
        object.__setattr__(
            self, "applied_event_ids", _unique_ids(self.applied_event_ids)
        )
        object.__setattr__(
            self, "replayed_event_ids", _unique_ids(self.replayed_event_ids)
        )
        object.__setattr__(
            self, "consumed_event_ids", _unique_ids(self.consumed_event_ids)
        )
        object.__setattr__(self, "seed_node_ids", _unique_ids(self.seed_node_ids))
        if self.impact is not None and not isinstance(
            self.impact, IncrementalPlanImpactAnalysis
        ):
            raise EventDrivenReassessmentError(
                "impact must be IncrementalPlanImpactAnalysis when provided"
            )
        object.__setattr__(
            self,
            "schema",
            self.schema or EVENT_DRIVEN_REASSESSMENT_SCHEMA,
        )
        if self.schema != EVENT_DRIVEN_REASSESSMENT_SCHEMA:
            raise EventDrivenReassessmentError(
                f"unsupported event-driven reassessment schema: {self.schema}"
            )
        object.__setattr__(
            self,
            "binding",
            self.binding or EVENT_DRIVEN_REASSESSMENT_BINDING,
        )
        object.__setattr__(
            self,
            "carrier",
            self.carrier or PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE,
        )

    @property
    def model_free(self) -> bool:
        if self.impact is None:
            return True
        return bool(self.impact.refill_decision.model_free)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "binding": self.binding,
            "interface": EVENT_DRIVEN_REASSESSMENT_INTERFACE,
            "carrier": self.carrier,
            "consumes": list(EVENT_DRIVEN_REASSESSMENT_CONSUMES),
            "disposition": self.disposition.value,
            "live_plan_epoch": self.live_plan_epoch,
            "applied_event_ids": list(self.applied_event_ids),
            "replayed_event_ids": list(self.replayed_event_ids),
            "consumed_event_ids": list(self.consumed_event_ids),
            "seed_node_ids": list(self.seed_node_ids),
            "observation": _observation_dict(self.observation),
            "model_free": True,
            "authorizes_append": False,
            "authorizes_completion": False,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
            "no_competing_subsystem_created": True,
        }
        if self.impact is not None:
            payload["impact"] = self.impact.to_dict()
            payload["refill_decision"] = self.impact.refill_decision.to_dict()
            payload["delta_seeds"] = [item.to_dict() for item in self.impact.delta_seeds]
            payload["preserved_receipt_refs"] = list(
                self.impact.cone.preserved_receipt_refs
            )
            payload["unaffected_preserved_ids"] = list(
                self.impact.cone.unaffected_preserved_ids
            )
            payload["affected_ids"] = list(self.impact.cone.affected_ids)
        else:
            payload["impact"] = None
            payload["refill_decision"] = None
            payload["delta_seeds"] = []
            payload["preserved_receipt_refs"] = []
            payload["unaffected_preserved_ids"] = []
            payload["affected_ids"] = []
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EventDrivenReassessment":
        if not isinstance(payload, Mapping):
            raise EventDrivenReassessmentError(
                "event-driven reassessment payload must be a mapping"
            )
        schema = payload.get("schema", EVENT_DRIVEN_REASSESSMENT_SCHEMA)
        if schema != EVENT_DRIVEN_REASSESSMENT_SCHEMA:
            raise EventDrivenReassessmentError(
                f"unsupported event-driven reassessment schema: {schema}"
            )
        observation_payload = payload.get("observation") or {}
        if not isinstance(observation_payload, Mapping):
            raise EventDrivenReassessmentError("observation must be a mapping")
        impact_payload = payload.get("impact")
        return cls(
            disposition=EventDrivenReassessmentDisposition(
                str(payload.get("disposition") or "no_reassessment")
            ),
            observation=RefillObservation(
                plan_root_cid=str(observation_payload.get("plan_root_cid") or ""),
                revision=int(observation_payload.get("revision") or 0),
                ready_tasks=int(observation_payload.get("ready_tasks") or 0),
                active_tasks=int(observation_payload.get("active_tasks") or 0),
                open_goals=int(observation_payload.get("open_goals") or 0),
                validation_rejected=bool(
                    observation_payload.get("validation_rejected") or False
                ),
                review_rejected=bool(observation_payload.get("review_rejected") or False),
                merge_rejected=bool(observation_payload.get("merge_rejected") or False),
                stale_evidence=bool(observation_payload.get("stale_evidence") or False),
                branch_only_completion=bool(
                    observation_payload.get("branch_only_completion") or False
                ),
                actionable_drift=bool(
                    observation_payload.get("actionable_drift") or False
                ),
                retry_exhausted_with_refinement=bool(
                    observation_payload.get("retry_exhausted_with_refinement") or False
                ),
                rollout_threshold_missed=bool(
                    observation_payload.get("rollout_threshold_missed") or False
                ),
            ),
            live_plan_epoch=int(payload.get("live_plan_epoch") or 0),
            applied_event_ids=tuple(payload.get("applied_event_ids") or ()),
            replayed_event_ids=tuple(payload.get("replayed_event_ids") or ()),
            consumed_event_ids=tuple(payload.get("consumed_event_ids") or ()),
            seed_node_ids=tuple(payload.get("seed_node_ids") or ()),
            impact=(
                IncrementalPlanImpactAnalysis.from_dict(impact_payload)
                if isinstance(impact_payload, Mapping)
                else None
            ),
            schema=str(schema),
            binding=str(payload.get("binding") or EVENT_DRIVEN_REASSESSMENT_BINDING),
            carrier=str(
                payload.get("carrier") or PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE
            ),
        )


@dataclass(frozen=True)
class ProductionRefillEventAdapter:
    """Compose production events into a :class:`RefillObservation`.

    Event-driven reassessment is a binding of this adapter, not a second
    event bus, planner, or refill owner.
    """

    schema: str = PRODUCTION_EVENT_ADAPTER_MANIFEST
    EVENT_DRIVEN_REASSESSMENT_BINDING: ClassVar[str] = EVENT_DRIVEN_REASSESSMENT_BINDING
    INTERFACE: ClassVar[str] = PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE

    def supported_event_kinds(self) -> tuple[str, ...]:
        return _EVENT_KINDS

    def to_observation(
        self,
        *,
        plan_root_cid: str,
        revision: int,
        events: Sequence[Mapping[str, Any]] = (),
        ready_tasks: int = 0,
        active_tasks: int = 0,
        open_goals: int = 0,
    ) -> RefillObservation:
        kinds = {str(item.get("kind") or "") for item in events}
        unknown = sorted(k for k in kinds if k and k not in _EVENT_KINDS)
        if unknown:
            raise ValueError(f"unsupported production refill events: {unknown}")
        return RefillObservation(
            plan_root_cid=plan_root_cid,
            revision=revision,
            ready_tasks=ready_tasks,
            active_tasks=active_tasks,
            open_goals=open_goals,
            validation_rejected="validation_rejected" in kinds,
            review_rejected="review_rejected" in kinds,
            merge_rejected="merge_rejected" in kinds,
            stale_evidence="stale_evidence" in kinds,
            branch_only_completion="branch_only_completion" in kinds,
            actionable_drift="actionable_drift" in kinds or "doctor_finding" in kinds,
            retry_exhausted_with_refinement="retry_exhausted" in kinds,
            rollout_threshold_missed="rollout_threshold_missed" in kinds,
        )

    def reassess(
        self,
        events: Sequence[Mapping[str, Any] | AuthoritativeImpactEvent] = (),
        nodes: Sequence[PlanImpactNode | Mapping[str, Any]] = (),
        *,
        live_plan_epoch: int,
        plan_root_cid: str,
        revision: int,
        ready_tasks: int = 0,
        active_tasks: int = 0,
        open_goals: int = 0,
        previously_consumed_event_ids: Sequence[str] = (),
        worker_assertion: bool = False,
        observations: Sequence[
            FrontierObservation | Mapping[str, Any] | ImpactFrontierEntry
        ] = (),
        impact_closure: ImpactClosureReceipt | None = None,
    ) -> EventDrivenReassessment:
        return reassess_events(
            events,
            nodes,
            live_plan_epoch=live_plan_epoch,
            plan_root_cid=plan_root_cid,
            revision=revision,
            ready_tasks=ready_tasks,
            active_tasks=active_tasks,
            open_goals=open_goals,
            previously_consumed_event_ids=previously_consumed_event_ids,
            worker_assertion=worker_assertion,
            observations=observations,
            impact_closure=impact_closure,
            adapter=self,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.INTERFACE,
            "event_kinds": list(_EVENT_KINDS),
            "authorizes_append": False,
            "authorizes_completion": False,
            "binding": EVENT_DRIVEN_REASSESSMENT_BINDING,
            "reassessment_schema": EVENT_DRIVEN_REASSESSMENT_SCHEMA,
            "consumes": list(EVENT_DRIVEN_REASSESSMENT_CONSUMES),
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
        }


def reassess_events(
    events: Sequence[Mapping[str, Any] | AuthoritativeImpactEvent] = (),
    nodes: Sequence[PlanImpactNode | Mapping[str, Any]] = (),
    *,
    live_plan_epoch: int,
    plan_root_cid: str,
    revision: int,
    ready_tasks: int = 0,
    active_tasks: int = 0,
    open_goals: int = 0,
    previously_consumed_event_ids: Sequence[str] = (),
    worker_assertion: bool = False,
    observations: Sequence[
        FrontierObservation | Mapping[str, Any] | ImpactFrontierEntry
    ] = (),
    impact_closure: ImpactClosureReceipt | None = None,
    adapter: ProductionRefillEventAdapter | None = None,
) -> EventDrivenReassessment:
    """Reassess the current plan from authoritative events.

    Physical delivery remains the caller's responsibility.  This function
    performs exactly-once *logical* application keyed by ``event_id`` against
    ``previously_consumed_event_ids``.  It never writes DuckDB/DuckLake,
    never appends refill work, and never completes a task.  ``worker_assertion``
    is diagnostic only and cannot skip the live plan-epoch fence.
    """

    del worker_assertion
    carrier = adapter or ProductionRefillEventAdapter()
    plan_root = _compact(plan_root_cid)
    if not plan_root:
        raise EventDrivenReassessmentError("plan_root_cid is required")
    live_epoch = int(live_plan_epoch)
    if live_epoch < 1:
        raise EventDrivenReassessmentError("live_plan_epoch must be >= 1")

    if isinstance(events, (Mapping, AuthoritativeImpactEvent)):
        incoming: tuple[Any, ...] = (events,)
    else:
        if isinstance(events, (str, bytes, bytearray)) or not isinstance(
            events, Sequence
        ):
            raise EventDrivenReassessmentError("events must be a sequence")
        incoming = tuple(events)

    previously_consumed = set(_unique_ids(previously_consumed_event_ids))
    applied: list[Mapping[str, Any]] = []
    applied_ids: list[str] = []
    replayed_ids: list[str] = []
    seen_in_batch: set[str] = set()

    for raw in incoming:
        event = _mapping(raw, "event")
        event_id = _event_id(event)
        event_epoch = _event_plan_epoch(event, live_epoch)
        assert_plan_epoch_current(event_epoch, live_epoch)
        event_root = _event_plan_root(event, plan_root)
        if event_root != plan_root:
            raise EventDrivenReassessmentError(
                "event plan_root does not match plan_root_cid"
            )
        if event_id in previously_consumed or event_id in seen_in_batch:
            replayed_ids.append(event_id)
            continue
        seen_in_batch.add(event_id)
        applied.append(event)
        applied_ids.append(event_id)

    observation = carrier.to_observation(
        plan_root_cid=plan_root,
        revision=revision,
        events=applied,
        ready_tasks=ready_tasks,
        active_tasks=active_tasks,
        open_goals=open_goals,
    )

    seed_ids = _unique_ids(
        [seed for event in applied for seed in _seed_node_ids(event)]
    )
    analysis: IncrementalPlanImpactAnalysis | None = None
    if applied_ids and seed_ids:
        evidence_refs = _unique_ids(
            [ref for event in applied for ref in _event_evidence_refs(event)]
        )
        roots_payload = next(
            (roots for event in applied if (roots := _event_roots(event)) is not None),
            None,
        )
        delta_id = next(
            (
                item
                for item in (_event_delta_id(event) for event in applied)
                if item
            ),
            "",
        )
        impact_event: dict[str, Any] = {
            "event_id": applied_ids[0],
            "plan_epoch": live_epoch,
            "plan_root": plan_root,
            "seed_node_ids": list(seed_ids),
            "evidence_refs": list(evidence_refs),
            "delta_id": delta_id,
        }
        if roots_payload is not None:
            impact_event["roots"] = (
                roots_payload.to_dict()
                if hasattr(roots_payload, "to_dict")
                else roots_payload
            )
        analysis = analyze_incremental_plan_impact(
            impact_event,
            nodes,
            observations=observations,
            impact_closure=impact_closure,
        )
    elif applied_ids and not seed_ids:
        raise EventDrivenReassessmentError(
            "applied events require non-empty seed_node_ids"
        )

    consumed = _unique_ids(
        tuple(previously_consumed_event_ids) + tuple(applied_ids) + tuple(replayed_ids)
    )
    return EventDrivenReassessment(
        disposition=_disposition_for(
            analysis,
            applied_event_ids=tuple(applied_ids),
            replayed_event_ids=_unique_ids(replayed_ids),
        ),
        observation=observation,
        live_plan_epoch=live_epoch,
        applied_event_ids=tuple(applied_ids),
        replayed_event_ids=_unique_ids(replayed_ids),
        consumed_event_ids=consumed,
        seed_node_ids=seed_ids,
        impact=analysis,
        carrier=carrier.INTERFACE,
    )


__all__ = [
    "EVENT_DRIVEN_REASSESSMENT_BINDING",
    "EVENT_DRIVEN_REASSESSMENT_CONSUMES",
    "EVENT_DRIVEN_REASSESSMENT_INTERFACE",
    "EVENT_DRIVEN_REASSESSMENT_SCHEMA",
    "EventDrivenReassessment",
    "EventDrivenReassessmentDisposition",
    "EventDrivenReassessmentError",
    "PRODUCTION_EVENT_ADAPTER_MANIFEST",
    "PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE",
    "ProductionRefillEventAdapter",
    "reassess_events",
]
