"""Event-driven supervisor wake and transactional cursor advancement.

Supervisors wake only from validated event batches classified by the causal
frontier.  Idle timeouts perform no board scan, model call, context rebuild,
or write.  Cursor advancement is atomic with batch processing: a crash before
commit leaves the previous cursor so the batch replays.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_frontier import (
    CompiledFrontier,
    FrontierSubject,
    IndependenceAdmission,
    compile_frontier,
)
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    CausalEdge,
    CausalNode,
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    FrontierDisposition,
    _identifier,
    _integer,
    _strings,
    _timestamp,
    utc_now,
)
from .events import ConsumerCursor, DomainEvent, EventBatch, EventWaitRequest
from .registry import _template
from .retrieval_projection import RetrievalProjectionStore, retrieval_establishes_authority

_UTC = timezone.utc  # noqa: UP017 - package supports Python 3.8.

MAX_WAKE_SLICE_NODES = 4_096
WAKE_RECEIPT_KIND = "wake"
QUALIFIED_WAIT_INTERFACES = frozenset(
    {
        "StateOwnerEventWait@1",
        "TypedStateOwnerEventWait@1",
    }
)


class SchedulerError(CausalGraphError):
    """Base typed supervisor-wake and cursor-advancement failure."""


class SchedulerAuthorityError(FederationAuthorityError, SchedulerError):
    """An attempt to wake, scan, or advance without admitted event-driven capability."""


class SchedulerCrash(SchedulerError):
    """Injected crash before the cursor commit; the previous cursor remains."""


def qualified_event_wait_capability() -> dict[str, object]:
    """Hermetic fixture for a server-owned, event-driven-qualified wait path."""

    return {
        "available": True,
        "interface": "TypedStateOwnerEventWait@1",
        "client_interface": "QuackStateClientEventWait@1",
        "transport": "typed_state_owner_bounded_long_wait",
        "server_owned": True,
        "blocking_condition": True,
        "adaptive_polling": False,
        "event_driven_qualified": True,
        "idle_repeated_database_scans": False,
    }


def require_event_driven_capability(capability: Mapping[str, object] | None) -> None:
    """Fail closed unless the wait path is server-owned and event-driven qualified."""

    if not isinstance(capability, Mapping):
        raise SchedulerAuthorityError("event wait capability is missing")
    if capability.get("available") is not True:
        raise SchedulerAuthorityError("typed event wait is unavailable")
    if capability.get("event_driven_qualified") is not True:
        raise SchedulerAuthorityError("event-driven wait is not qualified")
    if capability.get("adaptive_polling") is not False:
        raise SchedulerAuthorityError("adaptive polling cannot claim event-driven operation")
    if capability.get("server_owned") is not True:
        raise SchedulerAuthorityError("event wait is not server-owned")
    interface = str(capability.get("interface") or "")
    if interface not in QUALIFIED_WAIT_INTERFACES:
        raise SchedulerAuthorityError("event wait interface is not admitted")
    if capability.get("idle_repeated_database_scans") is True:
        raise SchedulerAuthorityError("idle wait must not repeatedly scan the database")


def refuse_ducklake_wake_authority(receipt: Mapping[str, Any] | None) -> None:
    if not receipt:
        return
    if receipt.get("authoritative") is True or receipt.get("schedules") is True:
        raise SchedulerAuthorityError("DuckLake cannot schedule supervisor wake")


def _parse_timestamp(value: str, name: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SchedulerError(f"{name} must include a timezone")
    return parsed.astimezone(_UTC)


def _reachable(seeds: Sequence[str], adjacency: Mapping[str, Sequence[str]]) -> set[str]:
    seen: set[str] = set()
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                queue.append(nxt)
    return seen


@dataclass(frozen=True)
class WakeGraph:
    """Exact current-revision graph used to classify one event batch."""

    SCHEMA: ClassVar[str] = "ipfs_accelerate_py/agent-supervisor/causal-federation/wake-graph@1"

    nodes: tuple[CausalNode, ...]
    edges: tuple[CausalEdge, ...]
    subjects: tuple[FrontierSubject, ...]
    independence: tuple[IndependenceAdmission, ...] = ()
    admitted_projection_edge_ids: tuple[str, ...] = ()
    graph_revision: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or not all(
            isinstance(item, CausalNode) for item in self.nodes
        ):
            raise FederationContractError("wake graph nodes must be CausalNode records")
        if not isinstance(self.edges, tuple) or not all(
            isinstance(item, CausalEdge) for item in self.edges
        ):
            raise FederationContractError("wake graph edges must be CausalEdge records")
        if not self.subjects:
            raise FederationContractError("wake graph requires at least one subject")
        if not all(isinstance(item, FrontierSubject) for item in self.subjects):
            raise FederationContractError("wake graph subjects must be FrontierSubject records")
        _integer(self.graph_revision, "graph_revision", minimum=1)
        object.__setattr__(
            self,
            "admitted_projection_edge_ids",
            tuple(
                _identifier(item, "admitted_projection_edge_ids")
                for item in self.admitted_projection_edge_ids
            ),
        )


@dataclass(frozen=True)
class SupervisorWakeSlice:
    """Bounded causal/context slice for one processed event batch.

    The slice contains only changed facts and affected wake subjects.  It never
    enumerates the complete task board or an unrelated graph population.
    """

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/supervisor-wake-slice@1"
    )

    event_ids: tuple[str, ...]
    changed_fact_refs: tuple[str, ...]
    node_ids: tuple[str, ...]
    must_wake: tuple[str, ...]
    may_wake: tuple[str, ...]
    do_not_wake: tuple[str, ...]
    reused_receipt_refs: tuple[str, ...]
    frontier_ref: str

    def __post_init__(self) -> None:
        _strings(self.event_ids, "event_ids", maximum=4_096, required=True)
        _strings(self.changed_fact_refs, "changed_fact_refs", maximum=10_000, required=True)
        _strings(self.node_ids, "node_ids", maximum=MAX_WAKE_SLICE_NODES, required=False)
        if len(self.node_ids) > MAX_WAKE_SLICE_NODES:
            raise SchedulerError("wake slice exceeds bound")
        _strings(self.must_wake, "must_wake", maximum=1_024, required=False)
        _strings(self.may_wake, "may_wake", maximum=1_024, required=False)
        _strings(self.do_not_wake, "do_not_wake", maximum=1_024, required=False)
        overlap = set(self.must_wake) & set(self.may_wake)
        overlap |= set(self.must_wake) & set(self.do_not_wake)
        overlap |= set(self.may_wake) & set(self.do_not_wake)
        if overlap:
            raise FederationContractError("wake slice dispositions overlap")
        _strings(self.reused_receipt_refs, "reused_receipt_refs", maximum=10_000, required=False)
        _identifier(self.frontier_ref, "frontier_ref")

    @property
    def woke_supervisor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.must_wake) | set(self.may_wake)))

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "event_ids": list(self.event_ids),
                "changed_fact_refs": list(self.changed_fact_refs),
                "node_ids": list(self.node_ids),
                "must_wake": list(self.must_wake),
                "may_wake": list(self.may_wake),
                "do_not_wake": list(self.do_not_wake),
                "reused_receipt_refs": list(self.reused_receipt_refs),
                "frontier_ref": self.frontier_ref,
            }
        )


@dataclass(frozen=True)
class SupervisorWakeReceipt:
    """Evidence that one event batch was classified, sliced, and cursor-advanced."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/supervisor-wake-receipt@1"
    )

    consumer_id: str
    subscription_id: str
    subscription_revision: int
    after_cursor: int
    next_cursor: int
    store_generation: int
    cursor_revision: int
    woke_supervisor_ids: tuple[str, ...]
    asleep_supervisor_ids: tuple[str, ...]
    slice_ref: str
    reused_receipt_refs: tuple[str, ...]
    idle: bool
    idle_board_scans: int
    idle_model_calls: int
    idle_writes: int
    idle_context_rebuilds: int
    event_driven_qualified: bool
    recorded_at: str

    def __post_init__(self) -> None:
        _identifier(self.consumer_id, "consumer_id")
        _identifier(self.subscription_id, "subscription_id")
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        _integer(self.after_cursor, "after_cursor")
        _integer(self.next_cursor, "next_cursor", minimum=self.after_cursor)
        _integer(self.store_generation, "store_generation", minimum=1)
        _integer(self.cursor_revision, "cursor_revision", minimum=1)
        _strings(self.woke_supervisor_ids, "woke_supervisor_ids", maximum=1_024, required=False)
        _strings(self.asleep_supervisor_ids, "asleep_supervisor_ids", maximum=1_024, required=False)
        _identifier(self.slice_ref, "slice_ref", required=False)
        _strings(self.reused_receipt_refs, "reused_receipt_refs", maximum=10_000, required=False)
        if type(self.idle) is not bool or type(self.event_driven_qualified) is not bool:
            raise FederationContractError("wake receipt flags must be boolean")
        for name in (
            "idle_board_scans",
            "idle_model_calls",
            "idle_writes",
            "idle_context_rebuilds",
        ):
            _integer(getattr(self, name), name)
        if self.idle and (
            self.idle_board_scans
            or self.idle_model_calls
            or self.idle_writes
            or self.idle_context_rebuilds
            or self.woke_supervisor_ids
            or self.next_cursor != self.after_cursor
        ):
            raise SchedulerAuthorityError("idle wake must not scan, write, or advance")
        if self.event_driven_qualified is not True:
            raise SchedulerAuthorityError("wake receipt cannot claim an unqualified wait")
        _timestamp(self.recorded_at, "recorded_at")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "consumer_id": self.consumer_id,
                "subscription_id": self.subscription_id,
                "after_cursor": self.after_cursor,
                "next_cursor": self.next_cursor,
                "cursor_revision": self.cursor_revision,
                "woke_supervisor_ids": list(self.woke_supervisor_ids),
                "slice_ref": self.slice_ref,
                "idle": self.idle,
            }
        )


@dataclass
class InMemoryCursorLedger:
    """Durable-cursor stand-in for hermetic tests.  Advances only on commit."""

    cursor: ConsumerCursor
    crash_before_advance: bool = False
    advance_count: int = 0

    def load(self) -> ConsumerCursor:
        return self.cursor

    def advance(
        self,
        *,
        next_cursor: int,
        store_generation: int,
        last_event_id: str,
        recorded_at: str,
    ) -> ConsumerCursor:
        if self.crash_before_advance:
            raise SchedulerCrash("simulated crash before cursor commit")
        current = self.cursor
        if int(next_cursor) < current.global_sequence:
            raise SchedulerError("cursor cannot rewind")
        if int(next_cursor) == current.global_sequence:
            return current
        _identifier(last_event_id, "last_event_id", required=False)
        self.cursor = replace(
            current,
            global_sequence=int(next_cursor),
            store_generation=int(store_generation),
            revision=current.revision + 1,
            updated_at=recorded_at,
        )
        self.advance_count += 1
        return self.cursor


def eligible_supervisor_ids(compiled: CompiledFrontier) -> frozenset[str]:
    if not isinstance(compiled, CompiledFrontier):
        raise FederationContractError("compiled frontier is required")
    return frozenset(compiled.must_wake) | frozenset(compiled.may_wake)


def _changed_facts(events: Sequence[DomainEvent]) -> tuple[str, ...]:
    facts: list[str] = []
    for event in events:
        facts.extend(event.changed_fact_refs)
    return tuple(dict.fromkeys(facts))


def _validate_batch(batch: EventBatch, cursor: ConsumerCursor) -> None:
    if not isinstance(batch, EventBatch):
        raise SchedulerError("wake input must be an EventBatch")
    if batch.consumer_id != cursor.consumer_id:
        raise SchedulerAuthorityError("event batch consumer differs from the durable cursor")
    if batch.subscription_id != cursor.subscription_id:
        raise SchedulerAuthorityError("event batch subscription differs from the durable cursor")
    if batch.subscription_revision != cursor.subscription_revision:
        raise SchedulerAuthorityError("event batch subscription revision is stale")
    if batch.after_cursor != cursor.global_sequence:
        raise SchedulerAuthorityError("event batch cursor does not match the durable cursor")
    if batch.store_generation < cursor.store_generation:
        raise SchedulerAuthorityError("event batch store generation is stale")
    if not batch.events:
        if batch.next_cursor != batch.after_cursor:
            raise SchedulerError("idle batch cannot advance the cursor")
        return
    sequences = [event.global_sequence for event in batch.events]
    if sequences != sorted(sequences):
        raise SchedulerError("event batch is not ordered by global sequence")
    if len(set(sequences)) != len(sequences):
        raise SchedulerError("event batch contains duplicate sequences")
    if sequences[0] <= batch.after_cursor:
        raise SchedulerAuthorityError("event batch includes already-acknowledged sequences")
    if batch.next_cursor != sequences[-1]:
        raise SchedulerError("next_cursor must equal the last delivered event")


def build_minimal_slice(
    *,
    events: Sequence[DomainEvent],
    compiled: CompiledFrontier,
    graph: WakeGraph,
    known_receipts: Mapping[str, str] | None = None,
) -> SupervisorWakeSlice:
    """Load only changed facts and affected wake subjects."""

    if not events:
        raise SchedulerError("minimal slice requires at least one event")
    if retrieval_establishes_authority() is not False:
        raise SchedulerAuthorityError("retrieval cannot mint wake authority")
    changed = _changed_facts(events)
    node_ids = {item.record_id for item in graph.nodes}
    subject_refs = {item.subject_ref: item.record_id for item in graph.nodes}
    seeds: list[str] = []
    for fact in changed:
        if fact in node_ids:
            seeds.append(fact)
        elif fact in subject_refs:
            seeds.append(subject_refs[fact])
    exact_adj: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.nomination_only:
            continue
        exact_adj[edge.source_node_id].append(edge.target_node_id)
    included = _reachable(seeds, exact_adj)
    woken_nodes = {
        entry.node_id
        for entry in compiled.entries
        if entry.disposition is not FrontierDisposition.DO_NOT_WAKE
    }
    included |= woken_nodes
    if len(included) > MAX_WAKE_SLICE_NODES:
        raise SchedulerError("wake slice exceeds bound")
    ordered_nodes = tuple(item.record_id for item in graph.nodes if item.record_id in included)
    if len(ordered_nodes) != len(included):
        extras = tuple(sorted(included - set(ordered_nodes)))
        ordered_nodes = ordered_nodes + extras
    known = known_receipts or {}
    reused = tuple(
        dict.fromkeys(
            known[subject.supervisor_id]
            for subject in graph.subjects
            if subject.supervisor_id in compiled.do_not_wake and subject.supervisor_id in known
        )
    )
    return SupervisorWakeSlice(
        event_ids=tuple(event.event_id for event in events),
        changed_fact_refs=changed,
        node_ids=ordered_nodes,
        must_wake=compiled.must_wake,
        may_wake=compiled.may_wake,
        do_not_wake=compiled.do_not_wake,
        reused_receipt_refs=reused,
        frontier_ref="frontier:" + compiled.cid,
    )


class SupervisorEventLoop:
    """Atomic event-batch processing and cursor advancement for one consumer."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/supervisor-event-loop@1"
    )

    def __init__(
        self,
        *,
        binding: FederationBinding,
        cursor_ledger: InMemoryCursorLedger,
        wait_capability: Mapping[str, object],
        wait: Callable[[EventWaitRequest], EventBatch] | None = None,
        apply_work: Callable[[SupervisorWakeSlice], None] | None = None,
        now: Callable[[], str] = utc_now,
        lease_expires_at: str = "",
        known_receipts: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(binding, FederationBinding):
            raise FederationContractError("binding must be a FederationBinding")
        if not isinstance(cursor_ledger, InMemoryCursorLedger):
            raise SchedulerError("cursor ledger is required")
        require_event_driven_capability(wait_capability)
        self._binding = binding
        self._ledger = cursor_ledger
        self._capability = dict(wait_capability)
        self._wait = wait
        self._apply_work = apply_work
        self._now = now
        self._lease_expires_at = str(lease_expires_at or "")
        self._known_receipts = dict(known_receipts or {})
        self.idle_board_scans = 0
        self.idle_model_calls = 0
        self.idle_writes = 0
        self.idle_context_rebuilds = 0
        self.wait_calls = 0
        self.processed_batches = 0

    @property
    def cursor(self) -> ConsumerCursor:
        return self._ledger.load()

    def _assert_lease(self, observed: str) -> None:
        if not self._lease_expires_at:
            return
        if _parse_timestamp(observed, "recorded_at") >= _parse_timestamp(
            self._lease_expires_at, "lease_expires_at"
        ):
            raise SchedulerAuthorityError("supervisor lease expired")

    def process_batch(
        self,
        batch: EventBatch,
        *,
        graph: WakeGraph | None = None,
        wait_capability: Mapping[str, object] | None = None,
        ducklake_receipt: Mapping[str, Any] | None = None,
        force_wake: Sequence[str] = (),
    ) -> SupervisorWakeReceipt:
        require_event_driven_capability(wait_capability or self._capability)
        refuse_ducklake_wake_authority(ducklake_receipt)
        if force_wake:
            raise SchedulerAuthorityError("supervisor wake cannot be forced past the frontier")
        observed = self._now()
        self._assert_lease(observed)
        cursor = self._ledger.load()
        _validate_batch(batch, cursor)
        if not batch.events:
            return SupervisorWakeReceipt(
                consumer_id=cursor.consumer_id,
                subscription_id=cursor.subscription_id,
                subscription_revision=cursor.subscription_revision,
                after_cursor=cursor.global_sequence,
                next_cursor=cursor.global_sequence,
                store_generation=batch.store_generation,
                cursor_revision=cursor.revision,
                woke_supervisor_ids=(),
                asleep_supervisor_ids=(),
                slice_ref="",
                reused_receipt_refs=(),
                idle=True,
                idle_board_scans=self.idle_board_scans,
                idle_model_calls=self.idle_model_calls,
                idle_writes=self.idle_writes,
                idle_context_rebuilds=self.idle_context_rebuilds,
                event_driven_qualified=True,
                recorded_at=observed,
            )
        if graph is None:
            raise SchedulerError("event-bearing batch requires a wake graph")
        changed = _changed_facts(batch.events)
        compiled = compile_frontier(
            event_id=batch.events[-1].event_id,
            binding=self._binding,
            graph_revision=graph.graph_revision,
            nodes=graph.nodes,
            edges=graph.edges,
            changed_fact_refs=changed,
            subjects=graph.subjects,
            independence=graph.independence,
            admitted_projection_edge_ids=graph.admitted_projection_edge_ids,
        )
        slice_ = build_minimal_slice(
            events=batch.events,
            compiled=compiled,
            graph=graph,
            known_receipts=self._known_receipts,
        )
        if self._apply_work is not None:
            self._apply_work(slice_)
        advanced = self._ledger.advance(
            next_cursor=batch.next_cursor,
            store_generation=batch.store_generation,
            last_event_id=batch.events[-1].event_id,
            recorded_at=observed,
        )
        self.processed_batches += 1
        return SupervisorWakeReceipt(
            consumer_id=advanced.consumer_id,
            subscription_id=advanced.subscription_id,
            subscription_revision=advanced.subscription_revision,
            after_cursor=cursor.global_sequence,
            next_cursor=advanced.global_sequence,
            store_generation=advanced.store_generation,
            cursor_revision=advanced.revision,
            woke_supervisor_ids=slice_.woke_supervisor_ids,
            asleep_supervisor_ids=slice_.do_not_wake,
            slice_ref="slice:" + slice_.cid,
            reused_receipt_refs=slice_.reused_receipt_refs,
            idle=False,
            idle_board_scans=self.idle_board_scans,
            idle_model_calls=self.idle_model_calls,
            idle_writes=self.idle_writes,
            idle_context_rebuilds=self.idle_context_rebuilds,
            event_driven_qualified=True,
            recorded_at=observed,
        )

    def wait_and_process(
        self,
        request: EventWaitRequest,
        *,
        graph: WakeGraph | None = None,
        wait_capability: Mapping[str, object] | None = None,
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> SupervisorWakeReceipt:
        if self._wait is None:
            raise SchedulerError("typed event wait boundary is unbound")
        if not isinstance(request, EventWaitRequest):
            raise SchedulerError("wait request must be EventWaitRequest")
        cursor = self._ledger.load()
        if request.consumer_id != cursor.consumer_id:
            raise SchedulerAuthorityError("wait request consumer differs from the durable cursor")
        if request.after_cursor != cursor.global_sequence:
            raise SchedulerAuthorityError("wait request cursor does not match the durable cursor")
        require_event_driven_capability(wait_capability or self._capability)
        batch = self._wait(request)
        self.wait_calls += 1
        return self.process_batch(
            batch,
            graph=graph,
            wait_capability=wait_capability,
            ducklake_receipt=ducklake_receipt,
        )


def _scheduler_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_causal_slice",
            """
            INSERT INTO causal_slices (
                causal_slice_id, tenant_id, federation_id, graph_revision,
                root_event_id, root_fact_ref, node_population_ref,
                edge_population_ref, content_ref, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "causal_slice_id",
                "tenant_id",
                "federation_id",
                "graph_revision",
                "root_event_id",
                "root_fact_ref",
                "node_population_ref",
                "edge_population_ref",
                "content_ref",
                "created_at",
            ),
        ),
        _template(
            "casf_select_causal_slice",
            """
            SELECT causal_slice_id, graph_revision, root_event_id,
                   root_fact_ref, node_population_ref, content_ref
            FROM causal_slices
            WHERE causal_slice_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("causal_slice_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_supervisor_receipt",
            """
            INSERT INTO supervisor_receipts (
                supervisor_receipt_id, tenant_id, federation_id, supervisor_id,
                receipt_kind, assignment_revision, fencing_epoch, content_ref,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "supervisor_receipt_id",
                "tenant_id",
                "federation_id",
                "supervisor_id",
                "receipt_kind",
                "assignment_revision",
                "fencing_epoch",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_supervisor_receipt",
            """
            SELECT supervisor_receipt_id, supervisor_id, receipt_kind,
                   content_ref, recorded_at
            FROM supervisor_receipts
            WHERE supervisor_receipt_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("supervisor_receipt_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class FederationSchedulerStore(RetrievalProjectionStore):
    """Persist wake slices and receipts through the sealed state owner."""

    INTERFACE = "FederationSchedulerStore@1"

    def __init__(
        self,
        client: QuackStateClient,
        *,
        event_notifier: Callable[[int], None] | None = None,
        outbox_notifier: Callable[[int], None] | None = None,
        test_failure_hook: Callable[[str], None] | None = None,
        require_quack_authority: bool = False,
    ) -> None:
        if isinstance(client, (str, bytes, Path)):
            raise SchedulerError("scheduler store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise SchedulerError("scheduler store requires an already-attached typed state client")
        registered = set(client.list_templates())
        missing = [
            template.name for template in _scheduler_templates() if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise SchedulerError("scheduler templates are absent from the sealed catalog")
        else:
            for template in _scheduler_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_wake_slice(
        self,
        slice_: SupervisorWakeSlice,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        if not isinstance(slice_, SupervisorWakeSlice):
            raise FederationContractError("wake slice is required")
        slice_id = "slice:" + slice_.cid
        return self._commit_fact(
            operation="federation.scheduler.slice.record",
            fact_id=slice_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((slice_id, event_id, *slice_.changed_fact_refs[:8]))
            ),
            payload_ref=slice_.cid,
            prepare_fact=lambda: self._prepare_slice(
                slice_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_slice(
                slice_,
                slice_id=slice_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                graph_revision=revision,
                recorded_at=recorded_at,
            ),
        )

    def record_wake_receipt(
        self,
        receipt: SupervisorWakeReceipt,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        supervisor_id: str,
        fencing_epoch: int = 1,
        assignment_revision: int = 1,
    ) -> CausalGraphCommit:
        if not isinstance(receipt, SupervisorWakeReceipt):
            raise FederationContractError("wake receipt is required")
        if receipt.event_driven_qualified is not True:
            raise SchedulerAuthorityError("unqualified wait cannot persist a wake receipt")
        receipt_id = "supervisor-receipt:" + receipt.cid
        return self._commit_fact(
            operation="federation.scheduler.wake.record",
            fact_id=receipt_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(receipt_id, receipt.slice_ref or receipt.consumer_id),
            payload_ref=receipt.cid,
            prepare_fact=lambda: self._prepare_receipt(
                receipt_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_receipt(
                receipt,
                receipt_id=receipt_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                supervisor_id=supervisor_id,
                fencing_epoch=fencing_epoch,
                assignment_revision=assignment_revision,
                recorded_at=recorded_at,
            ),
        )

    def load_slice(
        self,
        *,
        slice_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_causal_slice",
            {
                "causal_slice_id": _identifier(slice_id, "slice_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise SchedulerError("wake slice is absent")
        return dict(rows[0])

    def load_receipt(
        self,
        *,
        receipt_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_supervisor_receipt",
            {
                "supervisor_receipt_id": _identifier(receipt_id, "receipt_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise SchedulerError("wake receipt is absent")
        return dict(rows[0])

    def _prepare_slice(self, slice_id: str, *, tenant_id: str, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_causal_slice",
            {
                "causal_slice_id": slice_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise SchedulerError("wake slice identity is already bound")

    def _prepare_receipt(self, receipt_id: str, *, tenant_id: str, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_supervisor_receipt",
            {
                "supervisor_receipt_id": receipt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise SchedulerError("wake receipt identity is already bound")

    def _insert_slice(
        self,
        slice_: SupervisorWakeSlice,
        *,
        slice_id: str,
        federation_id: str,
        tenant_id: str,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_causal_slice",
            {
                "causal_slice_id": slice_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "graph_revision": graph_revision,
                "root_event_id": slice_.event_ids[0],
                "root_fact_ref": slice_.changed_fact_refs[0],
                "node_population_ref": "nodes:" + content_identity(list(slice_.node_ids)),
                "edge_population_ref": slice_.frontier_ref,
                "content_ref": slice_.cid,
                "created_at": recorded_at,
            },
        )

    def _insert_receipt(
        self,
        receipt: SupervisorWakeReceipt,
        *,
        receipt_id: str,
        federation_id: str,
        tenant_id: str,
        supervisor_id: str,
        fencing_epoch: int,
        assignment_revision: int,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_supervisor_receipt",
            {
                "supervisor_receipt_id": receipt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": _identifier(supervisor_id, "supervisor_id"),
                "receipt_kind": WAKE_RECEIPT_KIND,
                "assignment_revision": _integer(
                    assignment_revision, "assignment_revision", minimum=1
                ),
                "fencing_epoch": _integer(fencing_epoch, "fencing_epoch", minimum=1),
                "content_ref": receipt.cid,
                "recorded_at": recorded_at,
            },
        )


__all__ = (
    "FederationSchedulerStore",
    "InMemoryCursorLedger",
    "MAX_WAKE_SLICE_NODES",
    "SchedulerAuthorityError",
    "SchedulerCrash",
    "SchedulerError",
    "SupervisorEventLoop",
    "SupervisorWakeReceipt",
    "SupervisorWakeSlice",
    "WakeGraph",
    "build_minimal_slice",
    "eligible_supervisor_ids",
    "qualified_event_wait_capability",
    "refuse_ducklake_wake_authority",
    "require_event_driven_capability",
)
