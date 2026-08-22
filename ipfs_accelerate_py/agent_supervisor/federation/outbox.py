"""Content-addressed event/outbox construction primitives."""

# Python 3.8 compatibility requires ``str, Enum`` rather than ``StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from .contracts import (
    ClosedContract,
    FederationContractError,
    _identifier,
    _integer,
    _strings,
    _timestamp,
    utc_now,
)
from .events import DomainEvent, EventClass, EventEffectClass

_SCHEMA_PREFIX = "ipfs_accelerate_py/agent-supervisor/causal-federation"


class OutboxState(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    PROJECTED = "projected"
    RETRY = "retry"
    DEAD_LETTERED = "dead_lettered"


@dataclass(frozen=True)
class EventDraft:
    event_type: EventClass
    stream_id: str
    causal_parent_ids: tuple[str, ...]
    correlation_id: str
    causation_id: str
    tenant_id: str
    federation_id: str
    supervisor_id: str = ""
    task_id: str = ""
    repository_id: str = ""
    tree_id: str = ""
    goal_id: str = ""
    subgoal_id: str = ""
    symbol_id: str = ""
    contract_id: str = ""
    proof_obligation_id: str = ""
    resource_class: str = ""
    payload_ref: str = ""
    changed_fact_refs: tuple[str, ...] = ()
    effect_class: EventEffectClass = EventEffectClass.AUTHORITATIVE_STATE
    expires_at: str = ""
    deduplication_key: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventClass):
            raise FederationContractError("event draft type is not closed")
        for name in (
            "stream_id",
            "correlation_id",
            "causation_id",
            "tenant_id",
            "federation_id",
            "payload_ref",
            "deduplication_key",
        ):
            _identifier(getattr(self, name), name)
        _strings(self.causal_parent_ids, "causal_parent_ids", maximum=256)
        _strings(self.changed_fact_refs, "changed_fact_refs", maximum=10_000, required=True)
        if not isinstance(self.effect_class, EventEffectClass):
            raise FederationContractError("event draft effect class is not closed")
        if self.expires_at:
            _timestamp(self.expires_at, "expires_at")


@dataclass(frozen=True)
class OutboxRecord(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/outbox-record@1"

    outbox_id: str
    event_id: str
    event_cid: str
    tenant_id: str
    federation_id: str
    global_sequence: int
    state: OutboxState
    attempt_count: int
    next_attempt_at: str
    created_at: str
    projected_at: str

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType({"state": OutboxState})

    def __post_init__(self) -> None:
        for name in (
            "outbox_id",
            "event_id",
            "event_cid",
            "tenant_id",
            "federation_id",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.global_sequence, "global_sequence", minimum=1)
        if not isinstance(self.state, OutboxState):
            raise FederationContractError("outbox state is not closed")
        _integer(self.attempt_count, "attempt_count", maximum=1_000)
        _timestamp(self.next_attempt_at, "next_attempt_at")
        _timestamp(self.created_at, "created_at")
        if self.projected_at:
            _timestamp(self.projected_at, "projected_at")


def materialize_event(
    draft: EventDraft,
    *,
    stream_sequence: int,
    global_sequence: int,
    recorded_at: str | None = None,
) -> tuple[DomainEvent, OutboxRecord]:
    """Create deterministic event and outbox identities for one transaction."""

    stream_sequence = _integer(stream_sequence, "stream_sequence", minimum=1)
    global_sequence = _integer(global_sequence, "global_sequence", minimum=1)
    observed_at = recorded_at or utc_now()
    _timestamp(observed_at, "recorded_at")
    immutable = {
        "event_type": draft.event_type.value,
        "stream_id": draft.stream_id,
        "stream_sequence": stream_sequence,
        "global_sequence": global_sequence,
        "causal_parent_ids": list(draft.causal_parent_ids),
        "correlation_id": draft.correlation_id,
        "causation_id": draft.causation_id,
        "tenant_id": draft.tenant_id,
        "federation_id": draft.federation_id,
        "supervisor_id": draft.supervisor_id,
        "task_id": draft.task_id,
        "repository_id": draft.repository_id,
        "tree_id": draft.tree_id,
        "goal_id": draft.goal_id,
        "subgoal_id": draft.subgoal_id,
        "symbol_id": draft.symbol_id,
        "contract_id": draft.contract_id,
        "proof_obligation_id": draft.proof_obligation_id,
        "resource_class": draft.resource_class,
        "payload_ref": draft.payload_ref,
        "changed_fact_refs": list(draft.changed_fact_refs),
        "effect_class": draft.effect_class.value,
        "recorded_at": observed_at,
        "expires_at": draft.expires_at,
        "deduplication_key": draft.deduplication_key,
    }
    event_cid = content_identity(immutable)
    event_id = f"event:{event_cid.split(':', 1)[-1]}"
    event = DomainEvent(
        event_id=event_id,
        event_cid=event_cid,
        event_type=draft.event_type,
        stream_id=draft.stream_id,
        stream_sequence=stream_sequence,
        global_sequence=global_sequence,
        causal_parent_ids=draft.causal_parent_ids,
        correlation_id=draft.correlation_id,
        causation_id=draft.causation_id,
        tenant_id=draft.tenant_id,
        federation_id=draft.federation_id,
        supervisor_id=draft.supervisor_id,
        task_id=draft.task_id,
        repository_id=draft.repository_id,
        tree_id=draft.tree_id,
        goal_id=draft.goal_id,
        subgoal_id=draft.subgoal_id,
        symbol_id=draft.symbol_id,
        contract_id=draft.contract_id,
        proof_obligation_id=draft.proof_obligation_id,
        resource_class=draft.resource_class,
        payload_ref=draft.payload_ref,
        changed_fact_refs=draft.changed_fact_refs,
        effect_class=draft.effect_class,
        recorded_at=observed_at,
        expires_at=draft.expires_at,
        deduplication_key=draft.deduplication_key,
    )
    outbox_id = f"outbox:{event_cid.split(':', 1)[-1]}"
    outbox = OutboxRecord(
        outbox_id=outbox_id,
        event_id=event.event_id,
        event_cid=event.event_cid,
        tenant_id=event.tenant_id,
        federation_id=event.federation_id,
        global_sequence=event.global_sequence,
        state=OutboxState.PENDING,
        attempt_count=0,
        next_attempt_at=observed_at,
        created_at=observed_at,
        projected_at="",
    )
    return event, outbox


__all__ = [
    "EventDraft",
    "OutboxRecord",
    "OutboxState",
    "materialize_event",
]
