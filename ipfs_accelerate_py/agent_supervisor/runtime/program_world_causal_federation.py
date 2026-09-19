"""SAWM-033 causal-event federation adapter.

Publishes program-world events through an outbox and computes idempotent
affected-supervisor wake plans. Never scans every supervisor and never
completes a board task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class CausalFederationError(ValueError):
    """Closed causal-federation contract violation."""


@dataclass(frozen=True, slots=True)
class ProgramWorldEventProjection:
    event_id: str
    kind: str
    topic: str
    payload_cid: str


@dataclass(frozen=True, slots=True)
class AffectedSupervisorWakePlan:
    supervisor_ids: tuple[str, ...]
    idempotent: bool = True
    full_scan: bool = False


@dataclass
class ProgramWorldCausalFederationAdapter:
    _outbox: dict[str, ProgramWorldEventProjection] = field(default_factory=dict)
    _subscriptions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def subscribe(self, topic: str, supervisor_ids: tuple[str, ...]) -> None:
        self._subscriptions[topic] = tuple(dict.fromkeys(supervisor_ids))

    def publish_program_world_event(self, event: Mapping[str, Any]) -> ProgramWorldEventProjection:
        event_id = str(event.get("event_id") or "")
        if not event_id:
            raise CausalFederationError("event_id is required")
        projection = ProgramWorldEventProjection(
            event_id=event_id,
            kind=str(event.get("kind") or "update"),
            topic=str(event.get("topic") or "program-world"),
            payload_cid=str(event.get("payload_cid") or event_id),
        )
        self._outbox[event_id] = projection
        return projection

    def compute_affected_supervisor_wakes(
        self, event_id: str
    ) -> AffectedSupervisorWakePlan:
        projection = self._outbox.get(event_id)
        if projection is None:
            raise CausalFederationError("unknown event")
        supervisors = self._subscriptions.get(projection.topic, ())
        return AffectedSupervisorWakePlan(supervisor_ids=supervisors)


def publish_program_world_event(
    event: Mapping[str, Any],
    *,
    adapter: ProgramWorldCausalFederationAdapter | None = None,
) -> ProgramWorldEventProjection:
    return (adapter or ProgramWorldCausalFederationAdapter()).publish_program_world_event(event)


def compute_affected_supervisor_wakes(
    event_id: str,
    *,
    adapter: ProgramWorldCausalFederationAdapter,
) -> AffectedSupervisorWakePlan:
    return adapter.compute_affected_supervisor_wakes(event_id)
