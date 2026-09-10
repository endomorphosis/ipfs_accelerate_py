"""Scoped remediation denials must retain their terminal failure class."""

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)


REASONS = (
    "scoped_test_secret_remediation_proposal_unchanged",
    "scoped_test_secret_remediation_proposal_rejected",
    "scoped_test_secret_remediation_test_semantics_not_preserved",
)


def _failed_result(reason):
    # The outer failed result has no reason. Portal recorded a dispatched
    # provider and an explicit rejection inside the validation result.
    return {
        "implementation_result": {
            "returncode": 78,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "validation_result": {
                "attempted": True,
                "passed": False,
                "error": "proposal_validation_failed",
                "reason": reason,
            },
        },
    }


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("outer_reason", [None, "portal_provider_failed"])
def test_scoped_remediation_denial_is_not_a_provider_exit(reason, outer_reason):
    result = _failed_result(reason)
    if outer_reason:
        result["implementation_result"]["reason"] = outer_reason
    before = deepcopy(result)
    assert DatabasePortalExecutionBridge._terminal_failure(result) == reason
    assert result == before  # Historical failure evidence is never rewritten.


@pytest.mark.parametrize(
    "change",
    [
        {"passed": True},
        {"attempted": False},
        {"attempted": 1},
        {"error": "unrecognized"},
        {"reason": "capacity_backoff"},
        {"reason": "resource_claim_deferred"},
        {"reason": {"untrusted": "value"}},
    ],
)
def test_unrecognized_validation_cannot_inject_a_retry_class(change):
    result = _failed_result(REASONS[-1])
    result["implementation_result"]["validation_result"].update(change)
    assert DatabasePortalExecutionBridge._terminal_failure(result) == "portal_provider_failed"


def test_successful_implementation_does_not_become_a_terminal_failure():
    result = _failed_result(REASONS[-1])
    result["implementation_result"]["returncode"] = 0
    assert DatabasePortalExecutionBridge._terminal_failure(result) == ""


def test_plain_provider_failure_keeps_existing_classification():
    assert DatabasePortalExecutionBridge._terminal_failure(
        {"implementation_result": {"returncode": 1}}
    ) == "portal_provider_failed"
