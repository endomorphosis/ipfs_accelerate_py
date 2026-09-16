"""Exact terminal failure receipts; no providers or live campaign state."""
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalBridgeError, DatabasePortalBridgeDeferred


def opened(tmp_path, provider):
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb", authority_mode="embedded",
        owner_session_id="session:failure-test", provider_fn=provider,
    )
    daemon.materialize_population({
        "repository_tree_id": "tree:failure-test",
        "objectives": [{"objective_id": "objective:failure-test", "title": "Failure handling",
                        "goal_cid": "goal:failure-test", "status": "open"}],
        "tasks": [{"task_cid": "task:failure:" + str(index), "task_id": "FAIL-00" + str(index),
                   "goal_cid": "goal:failure-test", "title": "Fixture task", "ordinal": index,
                   "status": "ready", "priority": "P0"} for index in (1, 2)],
    })
    return daemon


def error(setup=True):
    return DatabasePortalBridgeError(
        "validation_project_dependency_preflight_failed" if setup else "provider_execution_failed",
        result={"implementation": {"failure_kind": "lifecycle_setup" if setup else "implementation",
                                    "attempt_consumed": not setup, "provider_dispatched": not setup,
                                    "deferred": setup, "backoff_seconds": 300 if setup else 0}},
    )


@pytest.mark.parametrize("setup", (False, True))
def test_failure_refreshes_revision_blocks_only_claimed_task_and_releases_lease(tmp_path, setup):
    def fail(_attempt):
        raise error(setup)
    daemon = opened(tmp_path, fail)
    try:
        attempt = daemon.claim_next()
        assert attempt.revision == 1
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert result["status"] == ("blocked" if setup else "failed")
        assert "fail_error" not in result
        terminal = daemon.get_attempt(attempt.attempt_id)
        assert terminal.revision == 3  # CLAIMED -> CONTEXT -> terminal, no stale CAS.
        task = daemon.task_source.get(attempt.task_cid)
        assert task.status == "blocked"
        receipt = task.body["completion_receipt"]
        assert receipt["attempt_consumed"] is (not setup)
        assert receipt["control_result_revision"] == task.revision
        assert receipt["attempt_id"] == attempt.attempt_id
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        assert daemon.coordinator.get_task_attempt(attempt.attempt_id).status.value == "released"
        next_attempt = daemon.claim_next()
        assert next_attempt.task_alias == "FAIL-002"
        assert daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key="provider:" + attempt.attempt_id) is None
    finally:
        daemon.close()


def test_genuine_inflight_deferred_retains_same_live_claim(tmp_path):
    def deferred(_attempt):
        raise DatabasePortalBridgeDeferred("inflight_process")
    daemon = opened(tmp_path, deferred)
    try:
        attempt = daemon.claim_next()
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert result["status"] == "running"
        assert result["deferred"] is True
        assert daemon.task_source.get(attempt.task_cid).status == "in_progress"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        assert daemon.get_attempt(attempt.attempt_id).committed_phase == "context"
    finally:
        daemon.close()


def test_setup_failure_accepts_forbidden_dispatch_evidence_without_inventing_observation(tmp_path):
    failed = error()
    detail = failed.result["implementation"]
    del detail["provider_dispatched"]
    detail["provider_call_allowed"] = False
    def fail(_attempt):
        raise failed
    daemon = opened(tmp_path, fail)
    try:
        attempt = daemon.claim_next()
        assert daemon._resume_attempt_without_process_crash(attempt)["status"] == "blocked"
        receipt = daemon.task_source.get(attempt.task_cid).body["completion_receipt"]
        assert receipt["attempt_consumed"] is False
        assert receipt["provider_call_allowed"] is False
        assert receipt["provider_dispatched"] is None
    finally:
        daemon.close()


def test_setup_failure_rejects_conflicting_durable_provider_phase(tmp_path):
    daemon = opened(tmp_path, None)
    def advanced(attempt):
        daemon.commit_phase(attempt, "provider")
        raise error()
    daemon._provider_fn = advanced
    try:
        attempt = daemon.claim_next()
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert "durable provider progress" in result["fail_error"]
        assert daemon.task_source.get(attempt.task_cid).status == "in_progress"
        assert daemon.get_attempt(attempt.attempt_id).committed_phase == "provider"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
    finally:
        daemon.close()


def test_failure_never_overwrites_operator_change(tmp_path):
    daemon = opened(tmp_path, None)
    def changed(_attempt):
        task = daemon.task_source.get(_attempt.task_cid)
        daemon.task_source.compare_and_set_status(task.task_cid, expected_revision=task.revision,
                                                 status="cancelled", receipt={"operation": "operator_cancel"})
        raise error()
    daemon._provider_fn = changed
    try:
        attempt = daemon.claim_next()
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert "owner row changed" in result["fail_error"]
        assert daemon.task_source.get(attempt.task_cid).status == "cancelled"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
    finally:
        daemon.close()


@pytest.mark.parametrize("after_release", (False, True))
def test_restart_finishes_only_exact_failure_intent_without_provider_replay(tmp_path, monkeypatch, after_release):
    def fail(_attempt):
        raise error()
    daemon = opened(tmp_path, fail)
    original_release = daemon.coordinator.release
    def crash(*args, **kwargs):
        if after_release:
            original_release(*args, **kwargs)
        raise RuntimeError("injected failure boundary")
    try:
        attempt = daemon.claim_next()
        monkeypatch.setattr(daemon.coordinator, "release", crash)
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert "injected failure boundary" in result["fail_error"]
        assert daemon.get_attempt(attempt.attempt_id).status == "running"
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        monkeypatch.setattr(daemon.coordinator, "release", original_release)
        repaired = daemon.reconcile_portal_failure_attempts()
        assert repaired[0]["status"] == "blocked"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        assert daemon.reconcile_portal_failure_attempts() == []
        assert daemon.claim_next().task_alias == "FAIL-002"
    finally:
        daemon.close()
