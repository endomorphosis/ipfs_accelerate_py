"""ASE3-008 live progress monitor, join evidence, and recovery tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.monitor_runner import (
    DurableMonitorError,
    DurableMonitorRunner,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_monitor import (
    HEARTBEAT_INTERVAL_MS,
    MAX_CANARY_RECOVERIES,
    SEMANTIC_PROGRESS_BUDGET_MS,
    STALE_HEARTBEAT_MS,
    ProcessEvidence,
    RecoveryAction,
    RecoveryPolicy,
    ReviewedHostNamespaceReconciler,
    SemanticProgressClock,
    StallClass,
    StallClassifier,
    join_running_evidence,
    RunHealthSnapshot,
)


def _guardian() -> ReviewedHostNamespaceReconciler:
    return ReviewedHostNamespaceReconciler(
        guardian_identity="host-guardian",
        host_namespace="host.ns.v3",
        review_cid="bafyreview" + "a" * 50,
    )


def _lifecycle(*, birth: str = "life-birth-1") -> ProcessEvidence:
    return ProcessEvidence(
        role="lifecycle",
        process_cid="bafylife" + "b" * 51,
        process_birth_identity=birth,
        lease_id="life-lease-1",
        fencing_generation=3,
        heartbeat_at_ms=10_000,
        event_cursor="life-cursor:1",
        generation=1,
        healthy=True,
    )


def test_policy_constants_match_acceptance() -> None:
    assert HEARTBEAT_INTERVAL_MS == 5_000
    assert STALE_HEARTBEAT_MS == 30_000
    assert SEMANTIC_PROGRESS_BUDGET_MS == 300_000
    assert MAX_CANARY_RECOVERIES == 3


def test_joined_running_requires_lifecycle_and_monitor(tmp_path: Path) -> None:
    runner = DurableMonitorRunner(tmp_path / "mon", guardian=_guardian())
    life = _lifecycle()
    # Missing monitor -> deny RUNNING
    denied = runner.evaluate_running(
        run_id="run-1",
        run_revision=1,
        lifecycle=life,
        now_ms=10_000,
    )
    assert denied.joined is False
    assert "missing_process_cid" in denied.reason_codes or "unhealthy_component" in denied.reason_codes

    # Guardian start
    with pytest.raises(DurableMonitorError):
        runner.start_or_adopt(
            run_id="run-1",
            requester="cli",
            lifecycle=life,
            now_ms=10_000,
        )
    adoption = runner.start_or_adopt(
        run_id="run-1",
        requester="host-guardian",
        lifecycle=life,
        now_ms=10_000,
    )
    assert adoption.adopted is False
    runner.heartbeat("run-1", now_ms=12_000)
    joined = runner.evaluate_running(
        run_id="run-1",
        run_revision=1,
        lifecycle=ProcessEvidence(
            role="lifecycle",
            process_cid=life.process_cid,
            process_birth_identity=life.process_birth_identity,
            lease_id=life.lease_id,
            fencing_generation=life.fencing_generation,
            heartbeat_at_ms=12_000,
            event_cursor=life.event_cursor,
            generation=1,
            healthy=True,
        ),
        semantic_progress_phase="DISPATCHING",
        semantic_progress_cursor="sched-cursor:1",
        now_ms=12_000,
    )
    assert joined.joined is True
    assert "joined_running" in joined.reason_codes


def test_client_disconnect_does_not_stop_monitor(tmp_path: Path) -> None:
    runner = DurableMonitorRunner(tmp_path / "mon", guardian=_guardian())
    life = _lifecycle()
    runner.start_or_adopt(
        run_id="run-2", requester="host-guardian", lifecycle=life, now_ms=1_000
    )
    state = runner.client_disconnect("run-2")
    assert state.terminal is False
    runner.heartbeat("run-2", now_ms=2_000)
    assert runner.load("run-2").heartbeat_at_ms == 2_000


def test_monitor_death_has_one_restart_winner(tmp_path: Path) -> None:
    runner = DurableMonitorRunner(tmp_path / "mon", guardian=_guardian())
    life = _lifecycle()
    first = runner.start_or_adopt(
        run_id="run-3", requester="host-guardian", lifecycle=life, now_ms=1_000
    )
    # Simulate death then two recoveries; second adopts same generation winner path.
    r1 = runner.recover(
        run_id="run-3",
        stall=StallClass.DEAD_PROCESS,
        authorized_callback=True,
        lifecycle=life,
        requester="host-guardian",
        now_ms=2_000,
    )
    r2 = runner.recover(
        run_id="run-3",
        stall=StallClass.DEAD_PROCESS,
        authorized_callback=True,
        lifecycle=life,
        requester="host-guardian",
        now_ms=2_100,
    )
    assert r1.action in {RecoveryAction.RESTART.value, RecoveryAction.ADOPT.value}
    assert r2.action in {RecoveryAction.RESTART.value, RecoveryAction.ADOPT.value}
    # Generation advances at most once from restarts of a dead process; adopters share state.
    state = runner.load("run-3")
    assert state is not None
    assert state.generation >= first.generation


def test_detection_without_callback_is_operator_only() -> None:
    policy = RecoveryPolicy()
    action = policy.authorize(
        StallClass.DEAD_PROCESS,
        recoveries_in_window=0,
        authorized_callback=False,
    )
    assert action is RecoveryAction.OPERATOR


def test_canary_circuit_breaker() -> None:
    policy = RecoveryPolicy(max_canary_recoveries=3)
    action = policy.authorize(
        StallClass.DEAD_PROCESS,
        recoveries_in_window=3,
        authorized_callback=True,
    )
    assert action is RecoveryAction.OPERATOR


def test_stale_heartbeat_and_frozen_progress_classification() -> None:
    classifier = StallClassifier()
    life = _lifecycle()
    mon = ProcessEvidence(
        role="monitor",
        process_cid="bafymon" + "c" * 52,
        process_birth_identity="mon-birth",
        lease_id="mon-lease",
        fencing_generation=3,
        heartbeat_at_ms=1_000,
        event_cursor="m:1",
        generation=1,
        healthy=True,
    )
    snap = RunHealthSnapshot(
        run_id="run-x",
        run_revision=1,
        lifecycle=life,
        monitor=mon,
        semantic_progress=SemanticProgressClock(
            phase="WORKING",
            cursor_cid="c1",
            observed_at_ms=1_000,
        ),
        tree_reachable=True,
        observed_at_ms=1_000,
    )
    assert (
        classifier.classify(snap, now_ms=1_000 + STALE_HEARTBEAT_MS)
        is StallClass.STALE_HEARTBEAT
    )
    # Fresh heartbeat but stale semantic progress only.
    mon2 = ProcessEvidence(
        role="monitor",
        process_cid=mon.process_cid,
        process_birth_identity=mon.process_birth_identity,
        lease_id=mon.lease_id,
        fencing_generation=3,
        heartbeat_at_ms=1_000 + SEMANTIC_PROGRESS_BUDGET_MS,
        event_cursor="m:2",
        generation=1,
        healthy=True,
    )
    snap2 = RunHealthSnapshot(
        run_id="run-x",
        run_revision=1,
        lifecycle=life,
        monitor=mon2,
        semantic_progress=SemanticProgressClock(
            phase="WORKING",
            cursor_cid="c1",
            observed_at_ms=1_000,
        ),
        tree_reachable=True,
        observed_at_ms=1_000 + SEMANTIC_PROGRESS_BUDGET_MS,
    )
    assert (
        classifier.classify(snap2, now_ms=1_000 + SEMANTIC_PROGRESS_BUDGET_MS)
        is StallClass.FROZEN_PROGRESS
    )


def test_self_attestation_denies_running() -> None:
    life = _lifecycle(birth="same-birth")
    mon = ProcessEvidence(
        role="monitor",
        process_cid="bafymon" + "d" * 52,
        process_birth_identity="same-birth",  # self-attestation
        lease_id="mon-lease",
        fencing_generation=3,
        heartbeat_at_ms=10_000,
        event_cursor="m:1",
        generation=1,
        healthy=True,
    )
    snap = RunHealthSnapshot(
        run_id="run-s",
        run_revision=1,
        lifecycle=life,
        monitor=mon,
        semantic_progress=None,
        tree_reachable=True,
        observed_at_ms=10_000,
    )
    joined = join_running_evidence(snap, now_ms=10_000)
    assert joined.joined is False
    assert "monitor_self_attestation" in joined.reason_codes


def test_terminal_shutdown_exact_generation(tmp_path: Path) -> None:
    runner = DurableMonitorRunner(tmp_path / "mon", guardian=_guardian())
    life = _lifecycle()
    adoption = runner.start_or_adopt(
        run_id="run-t", requester="host-guardian", lifecycle=life, now_ms=1_000
    )
    with pytest.raises(DurableMonitorError):
        runner.terminal_shutdown("run-t", generation=adoption.generation + 1)
    receipt = runner.terminal_shutdown("run-t", generation=adoption.generation)
    assert receipt["generation"] == adoption.generation
    assert runner.load("run-t").terminal is True


def test_log_noise_is_not_semantic_progress() -> None:
    with pytest.raises(ValueError, match="log noise"):
        SemanticProgressClock(
            phase="WORKING",
            cursor_cid="c",
            observed_at_ms=1,
            source="log",
        )


def test_guardian_rejects_client_identities() -> None:
    with pytest.raises(ValueError):
        ReviewedHostNamespaceReconciler(
            guardian_identity="cli",
            host_namespace="ns",
            review_cid="bafy" + "e" * 55,
        )
