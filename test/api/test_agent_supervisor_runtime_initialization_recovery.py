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
        ("execution_revision", 4),
        ("retryable", True),
    ],
)
def test_exception_substring_or_foreign_terminal_receipt_is_not_authority(key, value):
    observed = context()
    observed["task"]["body"]["completion_receipt"][key] = value
    with pytest.raises(RuntimeInitializationRecoveryRejected):
        admit_closed_initialization_failure(**observed)
