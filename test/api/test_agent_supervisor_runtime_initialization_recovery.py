"""Negative admission cases for the pending native initialization adapter."""

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.runtime_initialization_recovery import (
    PORTAL_INITIALIZATION_FAILURE,
    RuntimeInitializationRecoveryRejected,
    admit_closed_initialization_failure,
)


def context():
    identity = {
        "task_cid": "task:1",
        "attempt_id": "attempt:1",
        "claim_id": "claim:1",
        "lease_id": "lease:1",
        "owner_session_id": "owner:1",
        "attempt_number": 638,
        "fencing_token": 638,
        "fence_epoch": 638,
    }
    attempt = dict(identity, revision=3, status="failed", committed_phase="failed")
    receipt = dict(
        identity,
        operation="database_portal_terminal_failure",
        execution_revision=3,
        execution_phase="failed",
        retryable=False,
        reason=PORTAL_INITIALIZATION_FAILURE,
    )
    return {
        "attempt": attempt,
        "latest_attempt": dict(attempt),
        "task": {
            "task_cid": "task:1",
            "revision": 1877,
            "status": "blocked",
            "body": {"completion_receipt": receipt},
        },
        "failed_phase": {
            "phase": "failed",
            "body": {
                "portal_terminal_failure": True,
                "portal_retryable_failure": False,
                "reason": PORTAL_INITIALIZATION_FAILURE,
            },
        },
        "callback": dict(
            identity,
            state="closed_failure",
            execution_revision=3,
            active=False,
            unknown_outcome=False,
        ),
        "source": {
            "qualified": True,
            "constructor_probe_passed": True,
            "retention_guard_passed": True,
            "head": "a" * 40,
            "tree": "b" * 40,
            "qualification_receipt_cid": "receipt:qualified-source",
        },
    }


def test_admission_only_proposes_retry_and_preserves_failure():
    observed = context()
    before = copy.deepcopy(observed)
    result = admit_closed_initialization_failure(**observed)
    assert result["completion_authority"] is False
    assert result["task_revision"] == 1877
    assert observed == before
    assert result == admit_closed_initialization_failure(**observed)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("latest_attempt", "attempt_id", "attempt:new"),
        ("latest_attempt", "revision", 4),
        ("task", "status", "completed"),
        ("task", "revision", None),
        ("attempt", "committed_phase", "provider"),
        ("attempt", "fence_epoch", True),
        ("callback", "state", "started_outcome_unknown"),
        ("callback", "active", True),
        ("callback", "unknown_outcome", True),
        ("callback", "execution_revision", 2),
        ("callback", "lease_id", "lease:foreign"),
        ("source", "qualified", False),
        ("source", "constructor_probe_passed", False),
        ("source", "retention_guard_passed", False),
        ("source", "qualification_receipt_cid", ""),
        ("source", "head", "main"),
    ],
)
def test_rejects_unknown_foreign_or_unqualified_observation(section, key, value):
    observed = context()
    observed[section][key] = value
    with pytest.raises(RuntimeInitializationRecoveryRejected):
        admit_closed_initialization_failure(**observed)


@pytest.mark.parametrize(
    "key,value",
    [
        ("reason", "unrelated error mentioning isolate_merge_queue_to_task_projection"),
        ("operation", "database_portal_typed_deferral_budget_exhausted"),
        ("claim_id", "claim:new"),
        ("task_cid", "task:foreign"),
        ("execution_revision", 4),
        ("retryable", True),
    ],
)
def test_exception_substring_or_foreign_terminal_receipt_is_not_authority(key, value):
    observed = context()
    observed["task"]["body"]["completion_receipt"][key] = value
    with pytest.raises(RuntimeInitializationRecoveryRejected):
        admit_closed_initialization_failure(**observed)


def test_admission_accepts_real_native_terminal_receipt(tmp_path):
    from dataclasses import asdict

    from test.api.test_agent_supervisor_database_implementation_daemon import (
        _open_daemon,
        _population,
    )

    daemon = _open_daemon(tmp_path, session="session:initialization-recovery")
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        attempt = daemon.commit_phase(attempt, "context")
        attempt = daemon.commit_phase(
            attempt,
            "failed",
            body={
                "reason": PORTAL_INITIALIZATION_FAILURE,
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        daemon.run_once()
        task = daemon.task_source.get(attempt.task_cid)
        assert task.status == "blocked"
        terminal = dict(task.body["completion_receipt"])
        # The native receipt binds task identity through its canonical row.
        assert "task_cid" not in terminal
        # Source and closed callback observations here are hermetic fixtures;
        # only the task/execution/phase/terminal receipt come from the daemon.
        observed = context()
        observed["attempt"] = asdict(attempt)
        observed["latest_attempt"] = asdict(daemon._latest_failed_attempts()[0])
        observed["task"] = task.to_dict()
        observed["failed_phase"] = next(
            p
            for p in reversed(daemon.phase_history(attempt.attempt_id))
            if p["phase"] == "failed"
        )
        for key in (
            "task_cid",
            "attempt_id",
            "claim_id",
            "lease_id",
            "owner_session_id",
            "attempt_number",
            "fencing_token",
            "fence_epoch",
        ):
            observed["callback"][key] = getattr(attempt, key)
        observed["callback"]["execution_revision"] = attempt.revision
        proposal = admit_closed_initialization_failure(**observed)
        assert proposal["task_revision"] == task.revision
        assert (
            daemon.task_source.get(attempt.task_cid).body["completion_receipt"]
            == terminal
        )
    finally:
        daemon.close()


def test_native_callback_exception_keeps_unknown_intent(tmp_path):
    """An exception/failed phase alone cannot assert closed callback evidence."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DatabasePortalBridgeError,
    )
    from test.api.test_agent_supervisor_database_implementation_daemon import (
        _open_daemon,
        _population,
    )

    def fail(_attempt):
        raise DatabasePortalBridgeError(PORTAL_INITIALIZATION_FAILURE)

    daemon = _open_daemon(
        tmp_path,
        session="session:unknown-initialization",
        provider_fn=fail,
    )
    try:
        daemon.materialize_population(_population(1))
        claimed = daemon.claim_next()
        daemon._resume_attempt_without_process_crash(claimed)
        attempt = daemon._latest_failed_attempts()[0]
        task = daemon.task_source.get(attempt.task_cid)
        assert task.status == "blocked"
        intent = daemon.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert intent["callback_state"] == "started_outcome_unknown"
        assert intent["provider_effect_state"] == "unknown_may_have_started"
        assert intent["attempt_id"] == attempt.attempt_id
        terminal = task.body["completion_receipt"]
        assert terminal["reason"] == PORTAL_INITIALIZATION_FAILURE
        assert terminal["retryable"] is False
        # Both durable facts coexist. Do not turn the exception into a
        # no-effects claim or overwrite this intent to admit a retry.
        assert (
            daemon.provider_invocation_recorded(
                attempt.attempt_id,
                idempotency_key=f"provider:{attempt.attempt_id}",
            )
            == intent
        )
    finally:
        daemon.close()
