"""Focused proof tests for protected-path control-plane recovery."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA,
    DatabasePortalBridgeError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_CONTEXT,
    ATTEMPT_PHASE_FAILED,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    PortalTask,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    bind_database_portal_execution_from_args,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database control recovery tests",
)

PROTECTED_PATH = "docs/architecture/plan.md"


def _population(*, manual: bool = False) -> dict[str, object]:
    task: dict[str, object] = {
        "task_cid": "task:cid:protected-recovery",
        "task_id": "PCAR-001",
        "goal_cid": "goal:cid:root",
        "status": "ready",
        "priority": "P0",
        "ordinal": 1,
        "title": "Protected recovery",
    }
    if manual:
        task["completion"] = "manual"
    return {
        "repository_tree_id": "tree:protected-recovery",
        "objectives": [
            {
                "objective_id": "objective:protected-recovery",
                "objective_alias": "PCAR-O001",
                "title": "Protected recovery",
                "goal_cid": "goal:cid:root",
                "goal_alias": "PCAR-G000",
                "status": "open",
            }
        ],
        "tasks": [task],
    }


def _open_daemon(
    tmp_path: Path,
    *,
    provider_fn: Callable[[DatabaseTaskAttempt], Mapping[str, object]] | None = None,
    protected_path_recovery_fn: Callable[
        [DatabaseTaskAttempt], Mapping[str, object]
    ]
    | None = None,
    max_task_attempts: int = 3,
    clock_ms: Callable[[], int] | None = None,
) -> DatabaseImplementationDaemon:
    def provider(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        return {
            "status": "succeeded",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    def effect(
        attempt: DatabaseTaskAttempt,
        provider_result: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {
            "status": "applied",
            "task_cid": attempt.task_cid,
            "provider_result": dict(provider_result),
        }

    def validation(
        attempt: DatabaseTaskAttempt,
        effect_result: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {
            "outcome": "passed",
            "evidence_digest": "sha256:" + "f" * 64,
            "argv": ["focused-protected-recovery", attempt.task_cid],
            "effect_result": dict(effect_result),
        }

    return DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:protected-recovery",
        authority_mode="embedded",
        task_source_kind="duckdb",
        max_task_attempts=max_task_attempts,
        provider_fn=provider_fn or provider,
        effect_fn=effect,
        validation_fn=validation,
        protected_path_recovery_fn=protected_path_recovery_fn,
        require_real_execution=True,
        clock_ms=clock_ms,
    )


def _recovery_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
    tmp_path: Path,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA,
        "disposition": "retry",
        "reason": "ephemeral_workspace_protected_deletions_recovered",
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "portal_attempt": 1,
        "binding_id": "sha256:" + "1" * 64,
        "workspace_path": str((tmp_path / "worktrees" / "disposed").absolute()),
        "incident_digest": "sha256:" + "2" * 64,
        "active_snapshot_digest": "sha256:" + "3" * 64,
        "clearance_id": "sha256:" + "4" * 64,
        "clearance_receipt_digest": "sha256:" + "5" * 64,
        "protected_paths": [PROTECTED_PATH],
        "mutated_paths": [PROTECTED_PATH],
        "class_codes": ["workspace_protected_deletion"],
        "shared_path_digests": {PROTECTED_PATH: "sha256:" + "6" * 64},
        "event_stream_id": "stream:protected-recovery",
        "mutation_event_id": "sha256:" + "7" * 64,
        "clearance_event_id": "sha256:" + "8" * 64,
        "events_digest": "sha256:" + "9" * 64,
        "backoff_seconds": 0,
        "attempt_consumed": True,
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def _blocked_protected_attempt(
    daemon: DatabaseImplementationDaemon,
    *,
    manual: bool = False,
) -> DatabaseTaskAttempt:
    daemon.materialize_population(_population(manual=manual))
    if manual:
        task = daemon.task_source.get("task:cid:protected-recovery")
        assert task is not None
        claim = daemon.coordinator.claim_ready_task(
            owner_session_id=daemon.owner_session_id,
            lease_ms=daemon.lease_ms,
            now_ms=daemon._now_ms(),
        )
        assert claim is not None
        daemon._protect_new_claim(claim)
        daemon._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="in_progress",
            receipt={"operation": "fixture_manual_claim"},
        )
        attempt = daemon._insert_attempt_from_claim(
            claim,
            task_alias=task.task_alias,
        )
        attempt = daemon.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        attempt = daemon.commit_phase(
            attempt,
            ATTEMPT_PHASE_FAILED,
            body={
                "reason": "implementation_protected_path_mutated",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
                "deferred": False,
                "attempt_consumed": "unknown",
                "provider_dispatched": "unknown",
                "typed_deferral_slot_consumed": "unknown",
                "backoff_seconds": 0,
            },
        )
        daemon.reconcile_terminal_portal_failures()
        return attempt

    result = daemon.run_once()
    attempt = daemon.get_attempt(str(result["attempt_id"]))
    assert attempt is not None
    assert attempt.status == "failed"
    return attempt


def test_run_once_retries_proved_disposal_then_requires_fresh_completion(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    successor_claim_receipts: list[dict[str, object]] = []
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def provider(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        provider_calls.append(attempt.attempt_id)
        if len(provider_calls) == 1:
            raise DatabasePortalBridgeError(
                "implementation_protected_path_mutated"
            )
        task = holder["daemon"].task_source.get(attempt.task_cid)
        assert task is not None
        successor_claim_receipts.append(
            dict(task.body.get("completion_receipt") or {})
        )
        return {"status": "succeeded", "accepted": True}

    def recover(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        return _recovery_receipt(holder["daemon"], attempt, tmp_path)

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        protected_path_recovery_fn=recover,
        max_task_attempts=3,
        clock_ms=lambda: now["ms"],
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population())
        failed = daemon.run_once()
        source_attempt = daemon.get_attempt(str(failed["attempt_id"]))
        assert source_attempt is not None
        blocked = daemon.task_source.get(source_attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"

        now["ms"] = 100_000
        repaired = daemon.run_once()

        completed = daemon.task_source.get(source_attempt.task_cid)
        assert completed is not None and completed.status == "completed"
        recovery = repaired["protected_path_recovery_reconciliations"]
        assert len(recovery) == 1
        assert recovery[0]["changed"] is True
        assert recovery[0]["control_previous_status"] == "blocked"
        assert recovery[0]["protected_path_recovery_budget"][
            "failed_attempt_count"
        ] == 1
        assert provider_calls == [source_attempt.attempt_id, repaired["attempt_id"]]
        assert repaired["attempt_id"] != source_attempt.attempt_id
        claim_receipt = successor_claim_receipts[0]
        assert claim_receipt["protected_path_recovery_source_attempt_id"] == (
            source_attempt.attempt_id
        )
        assert claim_receipt["protected_path_recovery_seed"][
            "attempt_consumed"
        ] is True
        assert daemon.run_once()["unchanged"] is True
    finally:
        daemon.close()


@pytest.mark.parametrize("failure_kind", ["malformed", "genuine_mutation"])
def test_automatic_recovery_fails_closed_on_unproved_incident(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def provider(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeError("implementation_protected_path_mutated")

    def recover(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        if failure_kind == "genuine_mutation":
            raise DatabasePortalBridgeError(
                "protected-path incident is not a pure workspace disposal"
            )
        receipt = _recovery_receipt(holder["daemon"], attempt, tmp_path)
        receipt["mutated_paths"] = ["../escape"]
        return receipt

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        protected_path_recovery_fn=recover,
    )
    holder["daemon"] = daemon
    try:
        attempt = _blocked_protected_attempt(daemon)
        result = daemon.run_once()

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert result["write_count"] == 0
        assert result["protected_path_recovery_reconciliations"] == [
            {
                "task_cid": attempt.task_cid,
                "attempt_id": attempt.attempt_id,
                "status": "blocked",
                "changed": False,
                "reason": "protected_path_recovery_not_admitted",
                "error_type": (
                    "DatabasePortalBridgeError"
                    if failure_kind == "genuine_mutation"
                    else "DatabaseImplementationAuthorityError"
                ),
            }
        ]
    finally:
        daemon.close()


def test_manual_task_and_exhausted_budget_remain_blocked(tmp_path: Path) -> None:
    calls: list[str] = []
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def recover(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        calls.append(attempt.attempt_id)
        return _recovery_receipt(holder["daemon"], attempt, tmp_path)

    manual = _open_daemon(
        tmp_path / "manual",
        protected_path_recovery_fn=recover,
    )
    holder["daemon"] = manual
    try:
        attempt = _blocked_protected_attempt(manual, manual=True)
        outcome = manual.reconcile_blocked_protected_path_recoveries()
        task = manual.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert calls == []
        assert outcome[0]["reason"] == "manual_or_review_only_task"
    finally:
        manual.close()

    exhausted_calls: list[str] = []
    exhausted_holder: dict[str, DatabaseImplementationDaemon] = {}

    def exhausted_recover(
        attempt: DatabaseTaskAttempt,
    ) -> Mapping[str, object]:
        exhausted_calls.append(attempt.attempt_id)
        return _recovery_receipt(
            exhausted_holder["daemon"],
            attempt,
            tmp_path,
        )

    exhausted = _open_daemon(
        tmp_path / "exhausted",
        provider_fn=lambda _attempt: (_ for _ in ()).throw(
            DatabasePortalBridgeError("implementation_protected_path_mutated")
        ),
        protected_path_recovery_fn=exhausted_recover,
        max_task_attempts=1,
    )
    exhausted_holder["daemon"] = exhausted
    try:
        attempt = _blocked_protected_attempt(exhausted)
        outcomes = exhausted.reconcile_blocked_protected_path_recoveries()
        assert exhausted_calls == []
        assert outcomes[0]["reason"] == (
            "protected_path_recovery_budget_exhausted"
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="exhausted max_task_attempts",
        ):
            exhausted.recover_blocked_portal_protected_path_retry(
                attempt,
                recovery_evidence=_recovery_receipt(
                    exhausted,
                    attempt,
                    tmp_path,
                ),
            )
        task = exhausted.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "blocked"
    finally:
        exhausted.close()


def test_exact_recovery_is_idempotent_and_rejects_superseded_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        provider_fn=lambda _attempt: (_ for _ in ()).throw(
            DatabasePortalBridgeError("implementation_protected_path_mutated")
        ),
    )
    try:
        source = _blocked_protected_attempt(daemon)
        receipt = _recovery_receipt(daemon, source, tmp_path)

        latest_failed_attempts = daemon._latest_failed_attempts
        monkeypatch.setattr(daemon, "_latest_failed_attempts", lambda: [])
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="superseded attempt",
        ):
            daemon.recover_blocked_portal_protected_path_retry(
                source,
                recovery_evidence=receipt,
            )
        assert daemon.task_source.get(source.task_cid).status == "blocked"

        monkeypatch.setattr(
            daemon,
            "_latest_failed_attempts",
            latest_failed_attempts,
        )
        first = daemon.recover_blocked_portal_protected_path_retry(
            source,
            recovery_evidence=receipt,
        )
        repeated = daemon.recover_blocked_portal_protected_path_retry(
            source,
            recovery_evidence=receipt,
        )
        assert first["changed"] is True
        assert repeated["changed"] is False
        assert daemon.task_source.get(source.task_cid).status == "retrying"
    finally:
        daemon.close()


def test_restart_sweep_accepts_only_exact_protected_recovery_projection(
    tmp_path: Path,
) -> None:
    def terminal_provider(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeError("implementation_protected_path_mutated")

    first = _open_daemon(tmp_path, provider_fn=terminal_provider)
    try:
        source = _blocked_protected_attempt(first)
        source_attempt_id = source.attempt_id
    finally:
        first.close()

    holder: dict[str, DatabaseImplementationDaemon] = {}

    def recover(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        return _recovery_receipt(holder["daemon"], attempt, tmp_path)

    restarted = _open_daemon(
        tmp_path,
        provider_fn=terminal_provider,
        protected_path_recovery_fn=recover,
    )
    holder["daemon"] = restarted
    try:
        result = restarted.run_once()
        source = restarted.get_attempt(source_attempt_id)
        assert source is not None
        task = restarted.task_source.get(source.task_cid)
        assert task is not None and task.status == "retrying"
        assert result["protected_path_recovery_reconciliations"][0][
            "changed"
        ] is True

        # The immutable failed phase remains terminal, but its exact typed
        # recovery projection prevents restart reconciliation from reblocking.
        assert restarted.reconcile_terminal_portal_failures() == []
        assert restarted.reconcile_blocked_protected_path_recoveries() == []
        assert restarted.task_source.get(source.task_cid).status == "retrying"
    finally:
        restarted.close()


def test_runner_binds_bridge_protected_path_recovery_callback(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeDaemon:
        task_source = object()

        def bind_execution_callbacks(self, **callbacks: object) -> None:
            captured.update(callbacks)

    parsed = argparse.Namespace(
        implement=True,
        state_dir=tmp_path / "state",
        state_prefix="lane",
        worktree_submodule_path=None,
        implementation_protected_path=None,
        task_prefix="## PCAR-",
        max_task_attempts=3,
    )
    bridge = bind_database_portal_execution_from_args(
        FakeDaemon(),
        parsed,
        repo_root=tmp_path,
        portal_daemon_class=PortalImplementationDaemon,
    )

    callback = captured.get("protected_path_recovery_fn")
    assert bridge is not None
    assert callable(callback)
    assert getattr(callback, "__self__", None) is bridge
    assert getattr(callback, "__name__", "") == "recover_protected_path_retry"


def test_crash_fence_reaches_locked_auto_clear_and_keeps_genuine_mutation(
    tmp_path: Path,
) -> None:
    protected = tmp_path / PROTECTED_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("authoritative\n", encoding="utf-8")
    worktrees = tmp_path / "worktrees"
    worktrees.mkdir()

    def portal(state_name: str) -> PortalImplementationDaemon:
        return PortalImplementationDaemon(
            todo_path=tmp_path / "tasks.todo.md",
            state_path=tmp_path / state_name / "task-state.json",
            strategy_path=tmp_path / state_name / "strategy.json",
            events_path=tmp_path / state_name / "events.jsonl",
            repo_root=tmp_path,
            worktree_root=worktrees,
            implement=True,
            implementation_command="must-not-run",
            implementation_protected_paths=(PROTECTED_PATH,),
        )

    task = PortalTask(
        task_id="PCAR-001",
        title="Protected recovery",
        status="ready",
        completion="manual",
        priority="P0",
        track="architecture",
    )

    benign = portal("benign-state")
    benign_workspace = worktrees / "benign"
    benign_workspace_protected = benign_workspace / PROTECTED_PATH
    benign_workspace_protected.parent.mkdir(parents=True)
    benign_workspace_protected.write_text("authoritative\n", encoding="utf-8")
    benign_before = benign._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=benign_workspace,
    )
    benign_workspace_protected.unlink()
    benign_violation = benign._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=benign_workspace,
        before=benign_before,
    )
    assert benign_violation["reason"] == "implementation_protected_path_mutated"
    cleared = benign._reconcile_implementation_protected_path_fence()
    assert cleared["cleared"] is True
    assert cleared["class_codes"] == ["workspace_protected_deletion"]

    genuine = portal("genuine-state")
    genuine_workspace = worktrees / "genuine"
    genuine_workspace_protected = genuine_workspace / PROTECTED_PATH
    genuine_workspace_protected.parent.mkdir(parents=True)
    genuine_workspace_protected.write_text("authoritative\n", encoding="utf-8")
    genuine_before = genuine._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=genuine_workspace,
    )
    protected.write_text("genuine shared mutation\n", encoding="utf-8")
    genuine_violation = genuine._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=genuine_workspace,
        before=genuine_before,
    )
    assert genuine_violation["reason"] == "implementation_protected_path_mutated"
    blocked = genuine._reconcile_implementation_protected_path_fence()
    assert blocked["blocked"] is True
    assert blocked["reason"] == "implementation_protected_path_incident_latched"
    assert genuine._implementation_protected_incident_path().exists()
