"""Focused proof tests for leftover validation-retry seed-conflict recovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON,
    DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_SCHEMA,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database control recovery tests",
)


def _population() -> dict[str, object]:
    return {
        "repository_tree_id": "tree:validation-retry-seed-conflict",
        "objectives": [
            {
                "objective_id": "objective:validation-retry-seed-conflict",
                "objective_alias": "PCAR-O011",
                "title": "Validation retry seed-conflict recovery",
                "goal_cid": "goal:cid:root",
                "goal_alias": "PCAR-G000",
                "status": "open",
            }
        ],
        "tasks": [
            {
                "task_cid": "task:cid:validation-retry-seed-conflict",
                "task_id": "PCAR-011",
                "goal_cid": "goal:cid:root",
                "status": "ready",
                "priority": "P0",
                "ordinal": 1,
                "title": "Validation retry seed-conflict recovery",
            }
        ],
    }


def _open_daemon(
    tmp_path: Path,
    *,
    provider_fn: Callable[[DatabaseTaskAttempt], Mapping[str, object]] | None = None,
    validation_retry_seed_conflict_recovery_fn: Callable[
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
            "argv": ["focused-validation-retry-seed-conflict", attempt.task_cid],
            "effect_result": dict(effect_result),
        }

    return DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:validation-retry-seed-conflict",
        authority_mode="embedded",
        task_source_kind="duckdb",
        max_task_attempts=max_task_attempts,
        provider_fn=provider_fn or provider,
        effect_fn=effect,
        validation_fn=validation,
        validation_retry_seed_conflict_recovery_fn=(
            validation_retry_seed_conflict_recovery_fn
        ),
        require_real_execution=True,
        clock_ms=clock_ms,
    )


def _seed_conflict_recovery_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_SCHEMA,
        "disposition": "retry",
        "reason": "validation_retry_seed_state_progressed",
        "source_reason": DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON,
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "seed_id": "sha256:" + "a" * 64,
        "seed_commit": "a" * 40,
        "seed_rescue_branch": "rescue/pcar-011-attempt-1-failed-validation",
        "observed_commit": "b" * 40,
        "observed_branch": "implementation/pcar-011-attempt-2-progressed",
        "identity_bound": True,
        "backoff_seconds": 0,
        "attempt_consumed": False,
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def test_run_once_retries_progressed_validation_retry_seed_then_completes(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def provider(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        provider_calls.append(attempt.attempt_id)
        if len(provider_calls) == 1:
            raise DatabasePortalBridgeError(
                DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON
            )
        return {"status": "succeeded", "accepted": True}

    def recover(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        return _seed_conflict_recovery_receipt(holder["daemon"], attempt)

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        validation_retry_seed_conflict_recovery_fn=recover,
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
        recovery = repaired[
            "validation_retry_seed_conflict_recovery_reconciliations"
        ]
        assert len(recovery) == 1
        assert recovery[0]["changed"] is True
        assert recovery[0]["control_previous_status"] == "blocked"
        assert provider_calls == [
            source_attempt.attempt_id,
            repaired["attempt_id"],
        ]
        assert repaired["attempt_id"] != source_attempt.attempt_id
        assert (
            daemon.reconcile_blocked_validation_retry_seed_conflict_recoveries()
            == []
        )
    finally:
        daemon.close()


def test_automatic_seed_conflict_recovery_fails_closed_for_foreign_state(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeError(
            DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON
        )

    def recover(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeError(
            "validation-retry seed-conflict recovery requires "
            "identity-bound progressed Portal state"
        )

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        validation_retry_seed_conflict_recovery_fn=recover,
    )
    try:
        daemon.materialize_population(_population())
        failed = daemon.run_once()
        source_attempt = daemon.get_attempt(str(failed["attempt_id"]))
        assert source_attempt is not None
        blocked = daemon.task_source.get(source_attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"
        outcomes = daemon.reconcile_blocked_validation_retry_seed_conflict_recoveries()
        assert len(outcomes) == 1
        assert outcomes[0]["changed"] is False
        assert outcomes[0]["reason"] == (
            "validation_retry_seed_conflict_recovery_not_admitted"
        )
        assert daemon.task_source.get(source_attempt.task_cid).status == "blocked"
    finally:
        daemon.close()


def test_leftover_wait_deferrals_do_not_exhaust_typed_budget(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []

    def provider(_attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        provider_calls.append(_attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "worktree_lifecycle_claim_exists",
            backoff_seconds=30,
        )

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        max_task_attempts=3,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population())
        first = daemon.run_once()
        task_cid = str(first["claimed_task_cid"])
        assert first["implementation_result"]["retry_budget_exhausted"] is False
        assert daemon.task_source.get(task_cid).status == "retrying"

        for offset in (40_000, 80_000, 120_000):
            now["ms"] = offset
            result = daemon.run_once()
            implementation = result.get("implementation_result")
            if isinstance(implementation, Mapping):
                assert implementation.get("retry_budget_exhausted") is not True
        assert daemon.task_source.get(task_cid).status == "retrying"
        assert len(provider_calls) >= 1
    finally:
        daemon.close()


def test_run_once_rearms_leftover_wait_budget_exhaustion(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}

    def provider(attempt: DatabaseTaskAttempt) -> Mapping[str, object]:
        raise DatabasePortalBridgeDeferred(
            "worktree_lifecycle_claim_exists",
            backoff_seconds=30,
        )

    daemon = _open_daemon(
        tmp_path,
        provider_fn=provider,
        max_task_attempts=3,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population())
        failed = daemon.run_once()
        source = daemon.get_attempt(str(failed["attempt_id"]))
        assert source is not None
        task = daemon.task_source.get(source.task_cid)
        assert task is not None and task.status == "retrying"
        cas = daemon.task_source.compare_and_set_status
        cas(
            source.task_cid,
            expected_revision=int(task.revision),
            status="blocked",
            receipt={
                "operation": "database_portal_typed_deferral_budget_exhausted",
                "reason": "typed_portal_deferral_budget_exhausted",
                "attempt_id": source.attempt_id,
                "retry_budget": {
                    "matching_attempts": [
                        {"reason": "worktree_lifecycle_claim_exists"},
                        {
                            "reason": (
                                "external_protected_checkout_recovery_required"
                            )
                        },
                    ]
                },
            },
        )
        blocked = daemon.task_source.get(source.task_cid)
        assert blocked is not None and blocked.status == "blocked"

        now["ms"] = 100_000
        repaired = daemon.run_once()
        recovery = repaired[
            "leftover_wait_deferral_budget_recovery_reconciliations"
        ]
        assert recovery
        assert recovery[0]["changed"] is True
        rearmed = daemon.task_source.get(source.task_cid)
        assert rearmed is not None
        assert rearmed.status in {"retrying", "in_progress", "completed"}
    finally:
        daemon.close()
