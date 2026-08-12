"""ASE3-020 transactional run truth and crash-safe saga hardening."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.launch_guard import (
    CompleteLaunchPlanGuard,
    EffectBoundarySnapshot,
    LaunchPlanGuard,
    StaleLaunchPlanError,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry_backend import (
    AuthoritativeRunRevisionStore,
    DuckDBRunRegistryBackend,
    DurableRunHead,
    EffectRecoveryError,
    MonitorReadyEffectReservation,
    ProcessBirthObservation,
    RunRevisionCAS,
    RunRevisionConflictError,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.runtime_factory import (
    CompleteLaunchPlan,
    PromptToRunSaga,
    RequiredArgumentCoverageReceipt,
    RuntimeConstructionError,
    StandardSupervisorRuntimeFactory,
    reject_fixture_launch_plan,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import RunRegistry
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    LaunchPlan,
)


REPO = Path(__file__).resolve().parents[2]
BACKEND = (
    REPO
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "entrypoints"
    / "run_registry_backend.py"
)


def _head(run_id: str = "run-1", rev: int = 1, state: str = "ready") -> DurableRunHead:
    return DurableRunHead(
        run_id=run_id,
        run_revision=rev,
        handle_cid=f"bafyhandle{run_id}{rev}".ljust(59, "a")[:59],
        state=state,
        health="healthy" if state == "running" else "idle",
        process_cid="bafyprocess" + "b" * 48 if state == "running" else "",
        process_birth_identity="birth:" + run_id if state == "running" else "",
        event_cursor=f"cursor-{rev}",
        updated_at_ms=1_000 + rev,
    )


def test_run_registry_backend_uses_policy_helper_not_raw_connect() -> None:
    source = BACKEND.read_text(encoding="utf-8")
    assert "connect_duckdb_with_policy" in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "connect" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "duckdb":
                    raise AssertionError(
                        f"raw duckdb.connect remains at line {node.lineno}"
                    )


def test_authoritative_revision_cas_one_winner(tmp_path: Path) -> None:
    store: AuthoritativeRunRevisionStore = DuckDBRunRegistryBackend(
        tmp_path / "runs.duckdb"
    )
    head = store.create(_head())
    next_head = _head(rev=2)
    next_head = DurableRunHead(
        run_id=head.run_id,
        run_revision=2,
        handle_cid="bafynext" + "c" * 51,
        state="ready",
        health="idle",
        event_cursor="cursor-2",
        updated_at_ms=2_000,
    )
    winner = store.compare_and_swap(
        RunRevisionCAS(head.run_id, 1, head.handle_cid), next_head
    )
    assert winner.run_revision == 2
    with pytest.raises(RunRevisionConflictError):
        store.compare_and_swap(
            RunRevisionCAS(head.run_id, 1, head.handle_cid), next_head
        )


def test_history_vector_is_hash_linked_and_rejects_drift(tmp_path: Path) -> None:
    store = DuckDBRunRegistryBackend(tmp_path / "hist.duckdb")
    store.create(_head("run-h"))
    v1 = store.append_history(run_id="run-h", entry={"kind": "created", "n": 1})
    assert v1.length == 1
    v2 = store.append_history(run_id="run-h", entry={"kind": "advanced", "n": 2})
    assert v2.length == 2
    assert v2.entries[1]["predecessor_cid"] == v1.tip_cid
    again = store.history_vector("run-h")
    assert again.tip_cid == v2.tip_cid
    assert again.content_id == v2.content_id


def test_cursor_vectors_are_monotonic(tmp_path: Path) -> None:
    store = DuckDBRunRegistryBackend(tmp_path / "cur.duckdb")
    store.create(_head("run-c"))
    c1 = store.advance_cursor(
        run_id="run-c", cursor_kind="lifecycle", payload={"tick": 1}
    )
    c2 = store.advance_cursor(
        run_id="run-c", cursor_kind="lifecycle", payload={"tick": 2}
    )
    assert c2.sequence == c1.sequence + 1
    assert c2.predecessor_cid == c1.cursor_cid
    m1 = store.advance_cursor(
        run_id="run-c", cursor_kind="monitor", payload={"hb": True}
    )
    assert m1.cursor_kind == "monitor"
    with pytest.raises(Exception):
        store.advance_cursor(
            run_id="run-c", cursor_kind="not-a-kind", payload={}
        )


def test_effect_unknown_prohibits_replay(tmp_path: Path) -> None:
    store = DuckDBRunRegistryBackend(tmp_path / "fx.duckdb")
    store.create(_head("run-u"))
    store.record_intent(run_id="run-u", effect_key="e1", intent_cid="intent-1")
    unknown = store.record_unknown(
        run_id="run-u", effect_key="e1", reason_cid="reason-ambiguous"
    )
    assert unknown.replay_prohibited is True
    assert store.continuation_for(run_id="run-u", effect_key="e1") == (
        "unknown_outcome_no_replay"
    )
    with pytest.raises(EffectRecoveryError, match="UNKNOWN|unknown|prohibit"):
        store.record_effect(run_id="run-u", effect_key="e1", effect_cid="effect-x")


def test_monitor_ready_reservation_and_process_birth(tmp_path: Path) -> None:
    store = DuckDBRunRegistryBackend(tmp_path / "mr.duckdb")
    store.create(_head("run-m"))
    reservation = store.reserve_monitor_ready_effect(
        run_id="run-m",
        effect_key="start",
        intent_cid="intent-start",
        fence_token="fence-1",
    )
    assert isinstance(reservation, MonitorReadyEffectReservation)
    assert reservation.monitor_ready is True
    birth = ProcessBirthObservation(
        run_id="run-m",
        process_cid="bafyproc" + "d" * 51,
        process_birth_identity="birth:run-m",
        lease_id="lease-1",
        fencing_generation=1,
        observed_at_ms=5_000,
        healthy=True,
    )
    assert birth.content_id.startswith("b")


def test_complete_launch_plan_guard_rejects_stale_fields() -> None:
    guard: CompleteLaunchPlanGuard = LaunchPlanGuard()
    planned = EffectBoundarySnapshot(
        run_id="run-1",
        run_revision=1,
        target_tree_cid="tree-a",
        scope_cid="scope",
        authority_cid="auth",
        policy_cid="policy",
        provider_id="grok",
        task_source_cid="tasks",
        lease_id="lease",
        fencing_generation=1,
        plan_cid="plan",
        effect_kind="start",
    )
    current = EffectBoundarySnapshot(
        run_id="run-1",
        run_revision=1,
        target_tree_cid="tree-b",
        scope_cid="scope",
        authority_cid="auth",
        policy_cid="policy",
        provider_id="grok",
        task_source_cid="tasks",
        lease_id="lease",
        fencing_generation=1,
        plan_cid="plan",
        effect_kind="start",
    )
    with pytest.raises(StaleLaunchPlanError):
        guard.revalidate(planned, current)


def test_production_factory_rejects_fixture_and_missing_handlers(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "reg")
    with pytest.raises(RuntimeConstructionError):
        StandardSupervisorRuntimeFactory(registry=registry, handlers={}, production=True)

    def _ok(*_a, **_k):
        return {
            "receipt_cid": "bafyreceipt" + "e" * 48,
            "effect_applied": True,
        }

    handlers = {name: _ok for name in (
        "resolve", "preview", "authorize", "materialize", "start", "adopt",
        "observe", "steer", "validate", "stop",
    )}
    factory = StandardSupervisorRuntimeFactory(
        registry=registry, handlers=handlers, production=True
    )
    assert all(factory.handler_manifest().values())

    # RequiredArgumentCoverageReceipt fails closed on missing args.
    with pytest.raises(RuntimeConstructionError):
        RequiredArgumentCoverageReceipt(
            parser_identity="daemon",
            covered_arguments=("--plan-root",),
            signed_defaults=(),
            missing_arguments=("--worktree",),
        )
    coverage = RequiredArgumentCoverageReceipt(
        parser_identity="daemon",
        covered_arguments=("--plan-root", "--worktree"),
        signed_defaults=("--timeout",),
    )
    assert coverage.content_id

    saga = PromptToRunSaga(
        run_id="run-1",
        planning_attempt_id="plan-attempt-1",
        program_revision_cid="prog-1",
        launch_plan_cid="plan-1",
    )
    assert "PLAN_ADMITTED" in saga.phases
    with pytest.raises(RuntimeConstructionError):
        PromptToRunSaga(
            run_id="run-1",
            planning_attempt_id="x",
            program_revision_cid="y",
            launch_plan_cid="fixture-plan",
        )


def test_reject_fixture_launch_plan_mapping() -> None:
    # Constructing a real LaunchPlan is heavy; reject non-instances.
    with pytest.raises(RuntimeConstructionError):
        reject_fixture_launch_plan({"fixture": True})  # type: ignore[arg-type]
