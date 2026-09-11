"""Exact receipt and epoch checks for the shared ordinary-finalizer classifier."""
from copy import deepcopy
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.ordinary_finalizer_replay import (
    RETRY_SCHEMA, ordinary_finalized_disposition,
)


def inputs():
    attempt = dict(attempt_id="attempt:1", claim_id="claim:1", task_cid="task:1",
                   attempt_number=1, owner_session_id="session:1", fencing_token=1,
                   fence_epoch=1, lease_id="lease:1")
    control = dict(task_cid="task:1", revision=4, execution_spec_cid="exec:1",
                   validation_spec_cid="validation:1")
    return dict(attempt_identity=attempt, control_claim=control,
                task_identity={**control, "revision": 5}, task_status="retrying",
                receipt={**attempt, "schema": RETRY_SCHEMA,
                         "validation_spec_cid": "validation:1", "retry_exhausted": False,
                         "operation": "database_retry_rearmed"})


@pytest.mark.parametrize("outcome", ["retry", "exhausted", "unknown"])
def test_committed_outcome(outcome):
    data = inputs()
    expected = "terminalized_for_retry"
    if outcome != "retry":
        data["task_status"] = "blocked"
        data["receipt"]["retry_exhausted"] = True
        data["receipt"]["operation"] = "database_retry_exhausted"
    if outcome == "unknown":
        data["receipt"].update(operation="database_unknown_outcome_blocked", forced_block=True)
        expected = "blocked_unknown_outcome"
    assert ordinary_finalized_disposition(**data) == expected
    before = deepcopy(data)
    assert ordinary_finalized_disposition(**data) == expected
    assert data == before


@pytest.mark.parametrize("section,key,value", [
    ("receipt", k, "changed") for k in inputs()["attempt_identity"]
] + [
    ("receipt", "fence_epoch", True), ("receipt", "attempt_number", True),
    ("receipt", "validation_spec_cid", "different"),
    ("receipt", "retry_exhausted", 0), ("receipt", "schema", "unknown"),
    ("receipt", "terminal_reconciliation", {"unverified": True}),
    ("receipt", "attempt_consumed", False), ("receipt", "operation", "unknown"),
    ("task_identity", "revision", 6), ("task_identity", "revision", True),
    ("task_identity", "execution_spec_cid", "new"),
    ("task_identity", "validation_spec_cid", "new"),
    ("control_claim", "extra", "unknown"), ("attempt_identity", "extra", "unknown"),
])
def test_rejects_changed_authority(section, key, value):
    data = inputs()
    data[section][key] = value
    assert ordinary_finalized_disposition(**data) == ""


@pytest.mark.parametrize("field", ["attempt_consumed", "forced_block"])
@pytest.mark.parametrize("value", [None, 0, 1, "", "false", [], {}])
def test_rejects_present_nonboolean_finalizer_flags(field, value):
    data = inputs()
    data["receipt"][field] = value
    assert ordinary_finalized_disposition(**data) == ""


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_present_saga_marker_never_becomes_ordinary_finalization(value):
    data = inputs()
    data["receipt"]["terminal_reconciliation"] = value
    assert ordinary_finalized_disposition(**data) == ""


@pytest.mark.parametrize("exhausted", [False, True])
def test_operation_must_agree_with_committed_retry_budget(exhausted):
    data = inputs()
    data["receipt"]["retry_exhausted"] = exhausted
    data["task_status"] = "blocked" if exhausted else "retrying"
    data["receipt"]["operation"] = (
        "database_retry_rearmed" if exhausted else "database_retry_exhausted"
    )
    assert ordinary_finalized_disposition(**data) == ""


def test_explicit_consumed_and_unforced_flags_retain_ordinary_receipt():
    data = inputs()
    data["receipt"].update(attempt_consumed=True, forced_block=False)
    assert ordinary_finalized_disposition(**data) == "terminalized_for_retry"
