"""Focused proof tests for leftover checkout-lock control-plane recovery."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_lock_metadata,
    checkout_mutation_lock_path,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_EXTERNAL_PROTECTED_CHECKOUT_RECOVERY_SCHEMA,
    DATABASE_PORTAL_INFLIGHT_PROCESS_RECOVERY_SCHEMA,
    DatabasePortalBridgeError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_CONTEXT,
    ATTEMPT_PHASE_FAILED,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor,
    TodoSupervisorConfig,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database control recovery tests",
)


def _population(*, manual: bool = False) -> dict[str, object]:
    task: dict[str, object] = {
        "task_cid": "task:cid:external-checkout-recovery",
        "task_id": "PCAR-003",
        "goal_cid": "goal:cid:root",
        "status": "ready",
        "priority": "P0",
        "ordinal": 1,
        "title": "External checkout recovery",
    }
    if manual:
        task["completion"] = "manual"
    return {
        "repository_tree_id": "tree:external-checkout-recovery",
        "objectives": [
            {
                "objective_id": "objective:external-checkout-recovery",
                "objective_alias": "PCAR-O003",
                "title": "External checkout recovery",
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
    external_protected_checkout_recovery_fn: Callable[
        [DatabaseTaskAttempt], Mapping[str, object]
    ]
    | None = None,
    inflight_process_recovery_fn: Callable[
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
            "argv": ["focused-external-checkout-recovery", attempt.task_cid],
            "effect_result": dict(effect_result),
        }

    return DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:external-checkout-recovery",
        authority_mode="embedded",
        task_source_kind="duckdb",
        max_task_attempts=max_task_attempts,
        provider_fn=provider_fn or provider,
        effect_fn=effect,
        validation_fn=validation,
        external_protected_checkout_recovery_fn=(
            external_protected_checkout_recovery_fn
        ),
        inflight_process_recovery_fn=inflight_process_recovery_fn,
        require_real_execution=True,
        clock_ms=clock_ms,
    )


def _recovery_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
    tmp_path: Path,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_EXTERNAL_PROTECTED_CHECKOUT_RECOVERY_SCHEMA,
        "disposition": "retry",
        "reason": "external_protected_checkout_lock_absent",
        "source_reason": "external_protected_checkout_recovery_required",
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "lock_path": str((tmp_path / "implementation-main-merge.lock").absolute()),
        "lock_present": False,
        "backoff_seconds": 0,
        "attempt_consumed": False,
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def test_run_once_retries_absent_checkout_lock_then_requires_fresh_completion(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def provider(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        provider_calls.append(attempt.attempt_id)
        if len(provider_calls) == 1:
            raise DatabasePortalBridgeError(
                "external_protected_checkout_recovery_required"
            )
        return {"status": "succeeded", "accepted": True}

    def recover(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        return _recovery_receipt(holder["daemon"], attempt, tmp_path)

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        external_protected_checkout_recovery_fn=recover,
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
        recovery = repaired["external_protected_checkout_recovery_reconciliations"]
        assert len(recovery) == 1
        assert recovery[0]["changed"] is True
        assert recovery[0]["control_previous_status"] == "blocked"
        assert provider_calls == [
            source_attempt.attempt_id,
            repaired["attempt_id"],
        ]
        assert repaired["attempt_id"] != source_attempt.attempt_id
        assert daemon.reconcile_terminal_portal_failures() == []
        assert daemon.reconcile_blocked_external_protected_checkout_recoveries() == []
        assert daemon.task_source.get(source_attempt.task_cid).status == "completed"
    finally:
        daemon.close()


def test_automatic_recovery_fails_closed_while_lock_present(
    tmp_path: Path,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def provider(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeError(
            "external_protected_checkout_recovery_required"
        )

    def recover(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeError(
            "external checkout recovery requires the checkout mutation "
            "lock to be absent"
        )

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        external_protected_checkout_recovery_fn=recover,
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population())
        failed = daemon.run_once()
        source_attempt = daemon.get_attempt(str(failed["attempt_id"]))
        assert source_attempt is not None
        blocked = daemon.task_source.get(source_attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"

        outcomes = daemon.reconcile_blocked_external_protected_checkout_recoveries()
        assert len(outcomes) == 1
        assert outcomes[0]["changed"] is False
        assert outcomes[0]["reason"] == (
            "external_protected_checkout_recovery_not_admitted"
        )
        assert daemon.task_source.get(source_attempt.task_cid).status == "blocked"
    finally:
        daemon.close()


def test_automatic_recovery_rejects_manual_tasks(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def recover(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        calls.append(attempt.attempt_id)
        return _recovery_receipt(holder["daemon"], attempt, tmp_path)

    daemon = _open_daemon(
        tmp_path,
        external_protected_checkout_recovery_fn=recover,
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(manual=True))
        task = daemon.task_source.get("task:cid:external-checkout-recovery")
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
                "reason": "external_protected_checkout_recovery_required",
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
        outcomes = daemon.reconcile_blocked_external_protected_checkout_recoveries()
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert calls == []
        assert outcomes[0]["reason"] == "manual_or_review_only_task"
    finally:
        daemon.close()


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _seed_protected_board(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    todo_path = repo / "docs" / "generated.todo.md"
    todo_path.parent.mkdir()
    todo_path.write_text("# Generated board\n", encoding="utf-8")
    _git(repo, "add", "docs/generated.todo.md")
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "seed generated board",
    )
    return repo, todo_path


def test_managed_daemon_fences_supervisor_protected_recovery_journal(
    tmp_path: Path,
) -> None:
    repo, todo_path = _seed_protected_board(tmp_path)
    state_dir = tmp_path / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "supervisor_task_state.json",
            strategy_path=state_dir / "supervisor_strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            implementation_protected_paths=("docs/generated.todo.md",),
        )
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "daemon_task_state.json",
        strategy_path=state_dir / "daemon_strategy.json",
        events_path=state_dir / "daemon_events.jsonl",
        repo_root=repo,
        task_header_prefix="## AUTO-",
        implementation_protected_paths=("docs/generated.todo.md",),
    )
    observed: dict[str, object] = {}

    def inspect_supervisor_journal() -> dict[str, object]:
        lock_path = supervisor._repo_merge_lock_path()
        journal_before = lock_path.read_bytes()
        observed["daemon_context_empty_before"] = (
            daemon._current_checkout_mutation_lease() is None
        )
        observed.update(daemon._adopt_protected_checkout_recovery())
        observed["journal_unchanged"] = (
            lock_path.read_bytes() == journal_before
        )
        observed["daemon_context_empty_after"] = (
            daemon._current_checkout_mutation_lease() is None
        )
        return {"inspected": True}

    result = supervisor._run_generated_board_producer(
        producer="cross-component-test",
        commit_outputs=True,
        operation="generated_dirty_repair",
        callback=inspect_supervisor_journal,
    )

    assert result == {"inspected": True}
    assert observed["required"] is True
    assert observed["blocked"] is True
    assert observed["reason"] == (
        "external_protected_checkout_recovery_required"
    )
    assert observed["protected_recovery_owner"] == (
        "implementation_supervisor"
    )
    assert observed["journal_unchanged"] is True
    assert observed["daemon_context_empty_before"] is True
    assert observed["daemon_context_empty_after"] is True
    assert not checkout_mutation_lock_path(repo).exists()


def test_supervisor_does_not_maintenance_block_on_daemon_protected_recovery(
    tmp_path: Path,
) -> None:
    repo, todo_path = _seed_protected_board(tmp_path)
    state_dir = tmp_path / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "supervisor_task_state.json",
            strategy_path=state_dir / "supervisor_strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            implementation_protected_paths=("docs/generated.todo.md",),
        )
    )
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="AUTO-001",
        attempt=1,
        extra={"operation": "mark_tasks_completed"},
    )
    metadata.update(
        {
            "protected_recovery_required": True,
            "protected_recovery_owner": "implementation_daemon",
            "protected_paths": ["docs/generated.todo.md"],
        }
    )
    lock_path = supervisor._repo_merge_lock_path()
    lock_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    adoption = supervisor._adopt_supervisor_protected_recovery()
    recovered = supervisor._recover_retained_generated_checkout_lease()

    assert adoption["required"] is True
    assert adoption["blocked"] is False
    assert adoption["pending"] is True
    assert adoption["reason"] == "daemon_protected_checkout_recovery_pending"
    assert recovered["retained_lease"] is False
    assert recovered["reason"] == "daemon_protected_checkout_recovery_pending"
    assert recovered["blocked"] is False
    assert lock_path.exists()

    metadata["protected_recovery_owner"] = "foreign_owner"
    lock_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    foreign = supervisor._adopt_supervisor_protected_recovery()
    assert foreign["blocked"] is True
    assert foreign["reason"] == "external_protected_checkout_recovery_required"
    lock_path.unlink()


def _inflight_recovery_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_INFLIGHT_PROCESS_RECOVERY_SCHEMA,
        "disposition": "retry",
        "reason": "inflight_process_absent",
        "source_reason": "inflight_process",
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "live_runner_present": False,
        "backoff_seconds": 0,
        "attempt_consumed": False,
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def test_run_once_retries_absent_inflight_process_then_completes(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def provider(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        provider_calls.append(attempt.attempt_id)
        if len(provider_calls) == 1:
            raise DatabasePortalBridgeError("inflight_process")
        return {"status": "succeeded", "accepted": True}

    def recover(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        return _inflight_recovery_receipt(holder["daemon"], attempt)

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        inflight_process_recovery_fn=recover,
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
        recovery = repaired["inflight_process_recovery_reconciliations"]
        assert len(recovery) == 1
        assert recovery[0]["changed"] is True
        assert recovery[0]["control_previous_status"] == "blocked"
        assert provider_calls == [
            source_attempt.attempt_id,
            repaired["attempt_id"],
        ]
        assert repaired["attempt_id"] != source_attempt.attempt_id
        assert daemon.reconcile_blocked_inflight_process_recoveries() == []
    finally:
        daemon.close()


def test_automatic_inflight_recovery_fails_closed_while_runner_live(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeError("inflight_process")

    def recover(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeError(
            "inflight-process recovery requires the implementation "
            "runner to be absent"
        )

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        inflight_process_recovery_fn=recover,
    )
    try:
        daemon.materialize_population(_population())
        failed = daemon.run_once()
        source_attempt = daemon.get_attempt(str(failed["attempt_id"]))
        assert source_attempt is not None
        blocked = daemon.task_source.get(source_attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"
        outcomes = daemon.reconcile_blocked_inflight_process_recoveries()
        assert len(outcomes) == 1
        assert outcomes[0]["changed"] is False
        assert outcomes[0]["reason"] == "inflight_process_recovery_not_admitted"
        assert daemon.task_source.get(source_attempt.task_cid).status == "blocked"
    finally:
        daemon.close()
