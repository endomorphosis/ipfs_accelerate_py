"""Closed event, subscription, cursor, and delivery contracts for CASF.

Network delivery is at-least-once.  These contracts never describe network
delivery as exactly once; authoritative effects rely on idempotency, CAS,
leases, and fencing in the state-owner transaction boundary.
"""

# Python 3.8 compatibility requires ``str, Enum`` rather than ``StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar

from .contracts import (
    ClosedContract,
    FederationBoundsError,
    FederationContractError,
    _identifier,
    _integer,
    _strings,
    _text,
    _timestamp,
)

_SCHEMA_PREFIX = "ipfs_accelerate_py/agent-supervisor/causal-federation"
MAX_EVENT_PARENTS = 256
MAX_CHANGED_FACTS = 10_000
MAX_SUBSCRIPTION_SELECTORS = 1_024
MAX_EVENT_BATCH = 4_096
MAX_EVENT_PAYLOAD_REF_BYTES = 1_024


class EventClass(str, Enum):
    REPOSITORY_CHANGED = "REPOSITORY_CHANGED"
    TREE_CHANGED = "TREE_CHANGED"
    SYMBOL_CHANGED = "SYMBOL_CHANGED"
    CONTRACT_CHANGED = "CONTRACT_CHANGED"
    CAUSAL_GRAPH_CHANGED = "CAUSAL_GRAPH_CHANGED"
    SEMANTIC_ROOT_CHANGED = "SEMANTIC_ROOT_CHANGED"
    CAPSULE_CHANGED = "CAPSULE_CHANGED"
    PROOF_OBLIGATION_CHANGED = "PROOF_OBLIGATION_CHANGED"
    PROOF_COMPLETED = "PROOF_COMPLETED"
    TEST_COMPLETED = "TEST_COMPLETED"
    TASK_CREATED = "TASK_CREATED"
    TASK_READY = "TASK_READY"
    TASK_BLOCKED = "TASK_BLOCKED"
    TASK_CLAIMED = "TASK_CLAIMED"
    TASK_RELEASED = "TASK_RELEASED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    GOAL_CHANGED = "GOAL_CHANGED"
    SUBGOAL_CHANGED = "SUBGOAL_CHANGED"
    PLAN_REVISED = "PLAN_REVISED"
    POLICY_CHANGED = "POLICY_CHANGED"
    CAPABILITY_CHANGED = "CAPABILITY_CHANGED"
    PROVIDER_CAPACITY_CHANGED = "PROVIDER_CAPACITY_CHANGED"
    LEASE_EXPIRING = "LEASE_EXPIRING"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    MERGE_READY = "MERGE_READY"
    MERGE_COMPLETED = "MERGE_COMPLETED"
    MERGE_FAILED = "MERGE_FAILED"
    COUNTEREXAMPLE_FOUND = "COUNTEREXAMPLE_FOUND"
    HUMAN_RESPONSE = "HUMAN_RESPONSE"
    SUPERVISOR_HEALTH_CHANGED = "SUPERVISOR_HEALTH_CHANGED"
    RESOURCE_PRESSURE = "RESOURCE_PRESSURE"
    FEDERATION_REBALANCE_REQUESTED = "FEDERATION_REBALANCE_REQUESTED"
    DUCKLAKE_PROJECTION_CHANGED = "DUCKLAKE_PROJECTION_CHANGED"


class EventEffectClass(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    REVERSIBLE_STATE = "reversible_state"
    AUTHORITATIVE_STATE = "authoritative_state"
    LEASE_OR_FENCE = "lease_or_fence"
    EXTERNAL_REVERSIBLE = "external_reversible"
    EXTERNAL_IRREVERSIBLE = "external_irreversible"
    SECURITY_OR_LEGAL = "security_or_legal"
    PAYMENT = "payment"
    PROOF_LINEAGE = "proof_lineage"


class SelectorKind(str, Enum):
    EVENT_CLASS = "event_class"
    REPOSITORY = "repository"
    TREE = "tree"
    GOAL = "goal"
    SUBGOAL = "subgoal"
    TASK = "task"
    SYMBOL = "symbol"
    CONTRACT = "contract"
    PROOF_OBLIGATION = "proof_obligation"
    SUPERVISOR = "supervisor"
    RESOURCE_CLASS = "resource_class"
    CAUSAL_ANCESTOR = "causal_ancestor"
    CAUSAL_DESCENDANT = "causal_descendant"


class SubscriptionState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    QUARANTINED = "quarantined"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class DeliveryState(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    RETRY = "retry"
    DEAD_LETTERED = "dead_lettered"
    EXPIRED = "expired"


@dataclass(frozen=True)
class DomainEvent(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/domain-event@1"

    event_id: str
    event_cid: str
    event_type: EventClass
    stream_id: str
    stream_sequence: int
    global_sequence: int
    causal_parent_ids: tuple[str, ...]
    correlation_id: str
    causation_id: str
    tenant_id: str
    federation_id: str
    supervisor_id: str
    task_id: str
    repository_id: str
    tree_id: str
    goal_id: str
    subgoal_id: str
    symbol_id: str
    contract_id: str
    proof_obligation_id: str
    resource_class: str
    payload_ref: str
    changed_fact_refs: tuple[str, ...]
    effect_class: EventEffectClass
    recorded_at: str
    expires_at: str
    deduplication_key: str

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "event_type": EventClass,
            "causal_parent_ids": tuple,
            "changed_fact_refs": tuple,
            "effect_class": EventEffectClass,
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "event_cid",
            "stream_id",
            "correlation_id",
            "causation_id",
            "tenant_id",
            "federation_id",
            "deduplication_key",
        ):
            _identifier(getattr(self, name), name)
        if not isinstance(self.event_type, EventClass):
            raise FederationContractError("event_type is not in the closed catalog")
        _integer(self.stream_sequence, "stream_sequence", minimum=1)
        _integer(self.global_sequence, "global_sequence", minimum=1)
        _strings(
            self.causal_parent_ids,
            "causal_parent_ids",
            maximum=MAX_EVENT_PARENTS,
        )
        for name in (
            "supervisor_id",
            "task_id",
            "repository_id",
            "tree_id",
            "goal_id",
            "subgoal_id",
            "symbol_id",
            "contract_id",
            "proof_obligation_id",
            "resource_class",
        ):
            _identifier(getattr(self, name), name, required=False)
        _text(self.payload_ref, "payload_ref", maximum=MAX_EVENT_PAYLOAD_REF_BYTES)
        _strings(
            self.changed_fact_refs,
            "changed_fact_refs",
            maximum=MAX_CHANGED_FACTS,
            required=True,
        )
        if not isinstance(self.effect_class, EventEffectClass):
            raise FederationContractError("effect_class is not closed")
        _timestamp(self.recorded_at, "recorded_at")
        if self.expires_at:
            _timestamp(self.expires_at, "expires_at")

    @property
    def coalescing_forbidden(self) -> bool:
        return self.effect_class in {
            EventEffectClass.LEASE_OR_FENCE,
            EventEffectClass.EXTERNAL_IRREVERSIBLE,
            EventEffectClass.SECURITY_OR_LEGAL,
            EventEffectClass.PAYMENT,
            EventEffectClass.PROOF_LINEAGE,
        }


@dataclass(frozen=True)
class EventSelector(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/event-selector@1"

    kind: SelectorKind
    value: str

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType({"kind": SelectorKind})

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SelectorKind):
            raise FederationContractError("selector kind is not closed")
        _identifier(self.value, "selector value")
        if self.kind is SelectorKind.EVENT_CLASS:
            EventClass(self.value)


def _selectors(value: Any) -> tuple[EventSelector, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FederationContractError("selectors must be an array")
    if len(value) > MAX_SUBSCRIPTION_SELECTORS:
        raise FederationBoundsError("subscription selector bound exceeded")
    result = tuple(
        item if isinstance(item, EventSelector) else EventSelector.from_dict(item) for item in value
    )
    if len({(item.kind, item.value) for item in result}) != len(result):
        raise FederationContractError("subscription contains duplicate selectors")
    return result


@dataclass(frozen=True)
class EventSubscription(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/event-subscription@1"

    subscription_id: str
    tenant_id: str
    federation_id: str
    consumer_id: str
    revision: int
    event_classes: tuple[EventClass, ...]
    selectors: tuple[EventSelector, ...]
    maximum_batch: int
    maximum_pending: int
    retry_budget: int
    expires_at: str
    state: SubscriptionState

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "event_classes": lambda value: tuple(EventClass(item) for item in value),
            "selectors": _selectors,
            "state": SubscriptionState,
        }
    )

    def __post_init__(self) -> None:
        for name in ("subscription_id", "tenant_id", "federation_id", "consumer_id"):
            _identifier(getattr(self, name), name)
        _integer(self.revision, "revision", minimum=1)
        if not self.event_classes or len(self.event_classes) > len(EventClass):
            raise FederationBoundsError("event_classes must be nonempty and bounded")
        if any(not isinstance(item, EventClass) for item in self.event_classes):
            raise FederationContractError("event_classes contains an unknown class")
        if len(set(self.event_classes)) != len(self.event_classes):
            raise FederationContractError("event_classes contains duplicates")
        _selectors(self.selectors)
        _integer(self.maximum_batch, "maximum_batch", minimum=1, maximum=MAX_EVENT_BATCH)
        _integer(
            self.maximum_pending,
            "maximum_pending",
            minimum=self.maximum_batch,
            maximum=1_000_000,
        )
        _integer(self.retry_budget, "retry_budget", maximum=1_000)
        _timestamp(self.expires_at, "expires_at")
        if not isinstance(self.state, SubscriptionState):
            raise FederationContractError("subscription state is not closed")


@dataclass(frozen=True)
class ConsumerCursor(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/consumer-cursor@1"

    consumer_id: str
    subscription_id: str
    subscription_revision: int
    global_sequence: int
    store_generation: int
    revision: int
    updated_at: str

    def __post_init__(self) -> None:
        _identifier(self.consumer_id, "consumer_id")
        _identifier(self.subscription_id, "subscription_id")
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        _integer(self.global_sequence, "global_sequence")
        _integer(self.store_generation, "store_generation", minimum=1)
        _integer(self.revision, "revision", minimum=1)
        _timestamp(self.updated_at, "updated_at")


@dataclass(frozen=True)
class EventWaitRequest(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/event-wait-request@1"

    consumer_id: str
    after_cursor: int
    subscription_id: str
    subscription_revision: int
    deadline: str
    maximum_events: int

    def __post_init__(self) -> None:
        _identifier(self.consumer_id, "consumer_id")
        _integer(self.after_cursor, "after_cursor")
        _identifier(self.subscription_id, "subscription_id")
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        _timestamp(self.deadline, "deadline")
        _integer(self.maximum_events, "maximum_events", minimum=1, maximum=MAX_EVENT_BATCH)


def _events(value: Any) -> tuple[DomainEvent, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FederationContractError("events must be an array")
    if len(value) > MAX_EVENT_BATCH:
        raise FederationBoundsError("event batch exceeds bound")
    return tuple(
        item if isinstance(item, DomainEvent) else DomainEvent.from_dict(item) for item in value
    )


@dataclass(frozen=True)
class EventBatch(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/event-batch@1"

    consumer_id: str
    subscription_id: str
    subscription_revision: int
    after_cursor: int
    next_cursor: int
    store_generation: int
    events: tuple[DomainEvent, ...]
    timed_out: bool
    cancelled: bool
    server_shutdown: bool

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType({"events": _events})

    def __post_init__(self) -> None:
        _identifier(self.consumer_id, "consumer_id")
        _identifier(self.subscription_id, "subscription_id")
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        _integer(self.after_cursor, "after_cursor")
        _integer(self.next_cursor, "next_cursor", minimum=self.after_cursor)
        _integer(self.store_generation, "store_generation", minimum=1)
        _events(self.events)
        for name in ("timed_out", "cancelled", "server_shutdown"):
            if type(getattr(self, name)) is not bool:
                raise FederationContractError(f"{name} must be boolean")
        terminal_flags = int(self.timed_out) + int(self.cancelled) + int(self.server_shutdown)
        if terminal_flags > 1:
            raise FederationContractError("event batch has conflicting terminal flags")
        if self.events and terminal_flags:
            raise FederationContractError("event-bearing batch cannot be timeout/cancel/shutdown")
        if self.events and self.next_cursor != self.events[-1].global_sequence:
            raise FederationContractError("next_cursor must equal the last delivered event")


@dataclass(frozen=True)
class DeliveryAttempt(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/delivery-attempt@1"

    attempt_id: str
    event_id: str
    subscription_id: str
    consumer_id: str
    attempt_number: int
    state: DeliveryState
    error_code: str
    recorded_at: str

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType({"state": DeliveryState})

    def __post_init__(self) -> None:
        for name in ("attempt_id", "event_id", "subscription_id", "consumer_id"):
            _identifier(getattr(self, name), name)
        _integer(self.attempt_number, "attempt_number", minimum=1, maximum=1_000)
        if not isinstance(self.state, DeliveryState):
            raise FederationContractError("delivery state is not closed")
        _identifier(self.error_code, "error_code", required=False)
        _timestamp(self.recorded_at, "recorded_at")


@dataclass(frozen=True)
class DeadLetter(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/dead-letter@1"

    dead_letter_id: str
    event_id: str
    subscription_id: str
    consumer_id: str
    retry_count: int
    error_code: str
    evidence_ref: str
    quarantined: bool
    created_at: str
    expires_at: str

    def __post_init__(self) -> None:
        for name in (
            "dead_letter_id",
            "event_id",
            "subscription_id",
            "consumer_id",
            "error_code",
            "evidence_ref",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.retry_count, "retry_count", minimum=1, maximum=1_000)
        if type(self.quarantined) is not bool:
            raise FederationContractError("quarantined must be boolean")
        _timestamp(self.created_at, "created_at")
        if self.expires_at:
            _timestamp(self.expires_at, "expires_at")


@dataclass(frozen=True)
class EventAcknowledgement(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/event-acknowledgement@1"

    acknowledgement_id: str
    event_id: str
    consumer_id: str
    subscription_id: str
    subscription_revision: int
    global_sequence: int
    processed_effect_ref: str
    recorded_at: str

    def __post_init__(self) -> None:
        for name in (
            "acknowledgement_id",
            "event_id",
            "consumer_id",
            "subscription_id",
            "processed_effect_ref",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        _integer(self.global_sequence, "global_sequence", minimum=1)
        _timestamp(self.recorded_at, "recorded_at")


NON_COALESCIBLE_EFFECT_CLASSES = frozenset(
    {
        EventEffectClass.LEASE_OR_FENCE,
        EventEffectClass.EXTERNAL_IRREVERSIBLE,
        EventEffectClass.SECURITY_OR_LEGAL,
        EventEffectClass.PAYMENT,
        EventEffectClass.PROOF_LINEAGE,
    }
)


__all__ = [
    "ConsumerCursor",
    "DeadLetter",
    "DeliveryAttempt",
    "DeliveryState",
    "DomainEvent",
    "EventAcknowledgement",
    "EventBatch",
    "EventClass",
    "EventEffectClass",
    "EventSelector",
    "EventSubscription",
    "EventWaitRequest",
    "MAX_EVENT_BATCH",
    "NON_COALESCIBLE_EFFECT_CLASSES",
    "SelectorKind",
    "SubscriptionState",
]
