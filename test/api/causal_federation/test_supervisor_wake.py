"""Hermetic tests for CASF event-driven supervisor wake and cursor advancement."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.causal_frontier import (
    FrontierSubject,
    IndependenceAdmission,
    compile_frontier,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    ConsumerCursor,
    EventBatch,
    EventWaitRequest,
)
from ipfs_accelerate_py.agent_supervisor.federation.scheduler import (
    FederationSchedulerStore,
    InMemoryCursorLedger,
    SchedulerAuthorityError,
    SchedulerCrash,
    SchedulerError,
    SupervisorEventLoop,
    WakeGraph,
    build_minimal_slice,
    qualified_event_wait_capability,
    refuse_ducklake_wake_authority,
    require_event_driven_capability,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    SupervisorTrack,
    casf_select_tracks_for_frontier,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    open_embedded_client,
)
from test.api.causal_federation.test_causal_frontier import _edge, _node, _subject
from test.api.causal_federation.test_contracts import sample_binding
from test.api.causal_federation.test_event_wait import sample_event
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request

NOW = "2030-01-01T00:00:00Z"
_UTC = timezone.utc  # noqa: UP017


def _cursor(*, sequence: int = 0, revision: int = 1) -> ConsumerCursor:
    return ConsumerCursor(
        consumer_id="consumer:test",
        subscription_id="subscription:test",
        subscription_revision=1,
        global_sequence=sequence,
        store_generation=1,
        revision=revision,
        updated_at=NOW,
    )


def _event(sequence: int, *, changed: tuple[str, ...] = ("node:changed",)):
    return replace(sample_event(sequence), changed_fact_refs=changed)


def _batch(
    events: tuple = (),
    *,
    after_cursor: int = 0,
    timed_out: bool = False,
    cancelled: bool = False,
    server_shutdown: bool = False,
) -> EventBatch:
    next_cursor = events[-1].global_sequence if events else after_cursor
    return EventBatch(
        consumer_id="consumer:test",
        subscription_id="subscription:test",
        subscription_revision=1,
        after_cursor=after_cursor,
        next_cursor=next_cursor,
        store_generation=1,
        events=events,
        timed_out=timed_out,
        cancelled=cancelled,
        server_shutdown=server_shutdown,
    )


def _graph(
    *,
    independence: tuple[IndependenceAdmission, ...] = (),
    extra_nodes: tuple[contracts.CausalNode, ...] = (),
    extra_subjects: tuple[FrontierSubject, ...] = (),
) -> WakeGraph:
    changed = _node("node:changed", "symbol:changed")
    child = _node("node:child", "symbol:child")
    independent = _node("node:independent", "symbol:independent")
    return WakeGraph(
        nodes=(changed, child, independent, *extra_nodes),
        edges=(_edge("node:changed", "node:child"),),
        subjects=(
            _subject("supervisor:changed", "node:changed"),
            _subject("supervisor:child", "node:child"),
            _subject("supervisor:idle", "node:independent"),
            *extra_subjects,
        ),
        independence=independence,
        graph_revision=1,
    )


def _independence() -> IndependenceAdmission:
    return IndependenceAdmission(
        subject=_subject("supervisor:idle", "node:independent"),
        evidence_refs=("evidence:independence",),
        authoritative=True,
    )


def _loop(**overrides: object) -> SupervisorEventLoop:
    values: dict[str, object] = {
        "binding": sample_binding(),
        "cursor_ledger": InMemoryCursorLedger(_cursor()),
        "wait_capability": qualified_event_wait_capability(),
        "now": lambda: NOW,
    }
    values.update(overrides)
    return SupervisorEventLoop(**values)  # type: ignore[arg-type]


def test_missing_capability_fails_closed() -> None:
    with pytest.raises(SchedulerAuthorityError, match="capability is missing"):
        require_event_driven_capability(None)
    with pytest.raises(SchedulerAuthorityError, match="not qualified"):
        SupervisorEventLoop(
            binding=sample_binding(),
            cursor_ledger=InMemoryCursorLedger(_cursor()),
            wait_capability={"available": True, "server_owned": True, "adaptive_polling": False},
        )


def test_adaptive_polling_cannot_claim_event_driven() -> None:
    capability = qualified_event_wait_capability()
    capability["adaptive_polling"] = True
    with pytest.raises(SchedulerAuthorityError, match="adaptive polling"):
        require_event_driven_capability(capability)


def test_idle_timeout_does_not_scan_or_write() -> None:
    loop = _loop()
    receipt = loop.process_batch(_batch(timed_out=True))
    assert receipt.idle is True
    assert receipt.woke_supervisor_ids == ()
    assert receipt.next_cursor == 0
    assert receipt.idle_board_scans == 0
    assert receipt.idle_model_calls == 0
    assert receipt.idle_writes == 0
    assert receipt.idle_context_rebuilds == 0
    assert loop.cursor.global_sequence == 0
    assert loop._ledger.advance_count == 0


def test_exact_descendants_wake_and_independent_sleep() -> None:
    loop = _loop(known_receipts={"supervisor:idle": "receipt:idle"})
    receipt = loop.process_batch(
        _batch((_event(1),)),
        graph=_graph(independence=(_independence(),)),
    )
    assert receipt.idle is False
    assert receipt.woke_supervisor_ids == ("supervisor:changed", "supervisor:child")
    assert receipt.asleep_supervisor_ids == ("supervisor:idle",)
    assert receipt.reused_receipt_refs == ("receipt:idle",)
    assert receipt.next_cursor == 1
    assert receipt.cursor_revision == 2
    assert loop.cursor.global_sequence == 1


def test_minimal_slice_excludes_unrelated_and_independent_nodes() -> None:
    unrelated = _node("node:unrelated", "symbol:unrelated")
    graph = _graph(
        independence=(_independence(),),
        extra_nodes=(unrelated,),
    )
    compiled_loop = _loop()
    receipt = compiled_loop.process_batch(_batch((_event(1),)), graph=graph)
    assert "supervisor:idle" in receipt.asleep_supervisor_ids
    slice_ = build_minimal_slice(
        events=(_event(1),),
        compiled=compile_frontier(
            event_id="event:change",
            binding=sample_binding(),
            graph_revision=1,
            nodes=graph.nodes,
            edges=graph.edges,
            changed_fact_refs=("node:changed",),
            subjects=graph.subjects,
            independence=graph.independence,
        ),
        graph=graph,
        known_receipts={"supervisor:idle": "receipt:idle"},
    )
    assert "node:changed" in slice_.node_ids
    assert "node:child" in slice_.node_ids
    assert "node:independent" not in slice_.node_ids
    assert "node:unrelated" not in slice_.node_ids
    assert slice_.reused_receipt_refs == ("receipt:idle",)


def test_slice_bound_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.federation import scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod, "MAX_WAKE_SLICE_NODES", 1)
    graph = _graph(independence=(_independence(),))
    compiled = compile_frontier(
        event_id="event:change",
        binding=sample_binding(),
        graph_revision=1,
        nodes=graph.nodes,
        edges=graph.edges,
        changed_fact_refs=("node:changed",),
        subjects=graph.subjects,
        independence=graph.independence,
    )
    with pytest.raises(SchedulerError, match="exceeds bound"):
        build_minimal_slice(events=(_event(1),), compiled=compiled, graph=graph)


def test_cursor_advances_only_after_successful_processing() -> None:
    def boom(_slice: object) -> None:
        raise RuntimeError("work failed")

    loop = _loop(apply_work=boom)
    with pytest.raises(RuntimeError, match="work failed"):
        loop.process_batch(
            _batch((_event(1),)),
            graph=_graph(independence=(_independence(),)),
        )
    assert loop.cursor.global_sequence == 0
    assert loop._ledger.advance_count == 0


def test_crash_before_cursor_commit_replays_the_batch() -> None:
    ledger = InMemoryCursorLedger(_cursor(), crash_before_advance=True)
    loop = _loop(cursor_ledger=ledger)
    batch = _batch((_event(1),))
    graph = _graph(independence=(_independence(),))
    with pytest.raises(SchedulerCrash, match="before cursor commit"):
        loop.process_batch(batch, graph=graph)
    assert loop.cursor.global_sequence == 0
    ledger.crash_before_advance = False
    receipt = loop.process_batch(batch, graph=graph)
    assert receipt.next_cursor == 1
    assert loop.cursor.global_sequence == 1
    with pytest.raises(SchedulerAuthorityError, match="does not match the durable cursor"):
        loop.process_batch(batch, graph=graph)


def test_stale_batch_cursor_fails_closed() -> None:
    loop = _loop()
    loop.process_batch(
        _batch((_event(1),)),
        graph=_graph(independence=(_independence(),)),
    )
    with pytest.raises(SchedulerAuthorityError, match="does not match the durable cursor"):
        loop.process_batch(
            _batch((_event(2),), after_cursor=0),
            graph=_graph(independence=(_independence(),)),
        )


def test_lease_expired_fails_closed_without_cursor_advance() -> None:
    loop = _loop(lease_expires_at="2000-01-01T00:00:00Z")
    with pytest.raises(SchedulerAuthorityError, match="lease expired"):
        loop.process_batch(
            _batch((_event(1),)),
            graph=_graph(independence=(_independence(),)),
        )
    assert loop.cursor.global_sequence == 0


def test_capability_change_fails_closed_without_cursor_advance() -> None:
    loop = _loop()
    degraded = qualified_event_wait_capability()
    degraded["event_driven_qualified"] = False
    with pytest.raises(SchedulerAuthorityError, match="not qualified"):
        loop.process_batch(
            _batch((_event(1),)),
            graph=_graph(independence=(_independence(),)),
            wait_capability=degraded,
        )
    assert loop.cursor.global_sequence == 0


def test_do_not_wake_cannot_be_forced() -> None:
    loop = _loop()
    with pytest.raises(SchedulerAuthorityError, match="cannot be forced"):
        loop.process_batch(
            _batch((_event(1),)),
            graph=_graph(independence=(_independence(),)),
            force_wake=("supervisor:idle",),
        )
    assert loop.cursor.global_sequence == 0


def test_ducklake_cannot_schedule_wake() -> None:
    with pytest.raises(SchedulerAuthorityError, match="DuckLake cannot schedule"):
        refuse_ducklake_wake_authority({"authoritative": True})
    loop = _loop()
    with pytest.raises(SchedulerAuthorityError, match="DuckLake cannot schedule"):
        loop.process_batch(
            _batch((_event(1),)),
            graph=_graph(independence=(_independence(),)),
            ducklake_receipt={"schedules": True},
        )
    assert loop.cursor.global_sequence == 0


def test_wait_and_process_uses_event_batch_only() -> None:
    observed: list[int] = []

    def wait(request: EventWaitRequest) -> EventBatch:
        observed.append(request.after_cursor)
        return _batch((_event(1),), after_cursor=request.after_cursor)

    loop = _loop(wait=wait)
    deadline = datetime.now(_UTC) + timedelta(seconds=1)
    request = EventWaitRequest(
        consumer_id="consumer:test",
        after_cursor=0,
        subscription_id="subscription:test",
        subscription_revision=1,
        deadline=deadline.isoformat().replace("+00:00", "Z"),
        maximum_events=8,
    )
    receipt = loop.wait_and_process(
        request,
        graph=_graph(independence=(_independence(),)),
    )
    assert observed == [0]
    assert receipt.next_cursor == 1
    assert loop.wait_calls == 1


def test_runner_wakes_only_eligible_tracks(tmp_path: Path) -> None:
    def track(name: str) -> SupervisorTrack:
        return SupervisorTrack(
            name=name,
            script_path=tmp_path / f"{name}.py",
            log_path=tmp_path / f"{name}.log",
            supervisor_pid_path=tmp_path / f"{name}.pid",
            daemon_pid_path=tmp_path / f"{name}.daemon.pid",
        )

    tracks = (
        track("supervisor:changed"),
        track("supervisor:child"),
        track("supervisor:idle"),
        track("supervisor:other"),
    )
    selected = casf_select_tracks_for_frontier(
        tracks,
        must_wake=("supervisor:changed", "supervisor:child"),
        may_wake=(),
        do_not_wake=("supervisor:idle",),
        wait_capability=qualified_event_wait_capability(),
    )
    assert tuple(item.name for item in selected) == (
        "supervisor:changed",
        "supervisor:child",
    )


def test_runner_fails_closed_without_qualified_wait(tmp_path: Path) -> None:
    track = SupervisorTrack(
        name="supervisor:changed",
        script_path=tmp_path / "changed.py",
        log_path=tmp_path / "changed.log",
        supervisor_pid_path=tmp_path / "changed.pid",
        daemon_pid_path=tmp_path / "changed.daemon.pid",
    )
    with pytest.raises(SchedulerAuthorityError, match="not qualified"):
        casf_select_tracks_for_frontier(
            (track,),
            must_wake=("supervisor:changed",),
            may_wake=(),
            do_not_wake=(),
            wait_capability={"available": True, "adaptive_polling": True},
        )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(SchedulerError, match="database path"):
        FederationSchedulerStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for scheduler persistence")
def test_store_records_wake_slice_and_receipt(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:scheduler")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:scheduler",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = FederationSchedulerStore(client)
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
        causal_graph_revision=1,
    )
    identity, _receipt = _create(
        store,
        request=sample_request(binding=binding, maximum_supervisors=2, maximum_subagents=2),
        policy=sample_policy(
            binding,
            maximum_supervisors=2,
            maximum_subagents=2,
            maximum_concurrent_subagents=2,
        ),
    )
    loop = _loop(
        binding=binding,
        known_receipts={"supervisor:idle": "receipt:idle"},
    )
    graph = _graph(independence=(_independence(),))
    wake = loop.process_batch(_batch((_event(1),)), graph=graph)
    compiled = compile_frontier(
        event_id="event:change",
        binding=binding,
        graph_revision=1,
        nodes=graph.nodes,
        edges=graph.edges,
        changed_fact_refs=("node:changed",),
        subjects=graph.subjects,
        independence=graph.independence,
    )
    slice_ = build_minimal_slice(
        events=(_event(1),),
        compiled=compiled,
        graph=graph,
        known_receipts={"supervisor:idle": "receipt:idle"},
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    revision = store.record_wake_slice(
        slice_,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:slice",
        event_id="event:1",
    ).graph_revision
    store.record_wake_receipt(
        wake,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:wake",
        supervisor_id="supervisor:changed",
    )
    loaded_slice = store.load_slice(
        slice_id="slice:" + slice_.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_receipt = store.load_receipt(
        receipt_id="supervisor-receipt:" + wake.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded_slice["content_ref"] == slice_.cid
    assert loaded_receipt["receipt_kind"] == "wake"
    assert loaded_receipt["content_ref"] == wake.cid
