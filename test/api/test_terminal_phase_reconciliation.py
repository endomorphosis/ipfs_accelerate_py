"""Consumed-credit and supersession fences for terminal phase mismatches."""
from copy import deepcopy
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.ordinary_finalizer_replay import (
    RETRY_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.terminal_phase_reconciliation import (
    CONSUMED_RETRY,
    INTENDED_SUPERSESSION,
    qualify_failed_phase,
    qualify_terminal_repair_batch,
)

CHECKOUT = Path(__file__).resolve().parents[2]
MODULE = CHECKOUT / "ipfs_accelerate_py/agent_supervisor/todo_daemon/terminal_phase_reconciliation.py"


def ordinary_finalizer():
    attempt = dict(
        attempt_id="attempt:f49b4e967c5c4fa7a358b4c2837094d4",
        claim_id="claim:5bec03a98d4341498cbeb897e92d6a58",
        task_cid="baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza",
        attempt_number=1, owner_session_id="session:1", fencing_token=1,
        fence_epoch=1, lease_id="lease:1",
    )
    control = dict(
        task_cid=attempt["task_cid"], revision=4,
        execution_spec_cid="exec:1", validation_spec_cid="validation:1",
    )
    return dict(
        attempt_identity=attempt, control_claim=control,
        task_identity={**control, "revision": 5}, task_status="retrying",
        receipt={
            **attempt, "schema": RETRY_SCHEMA,
            "validation_spec_cid": "validation:1", "retry_exhausted": False,
            "operation": "database_retry_rearmed", "attempt_consumed": True,
        },
    )


def pctdd_005_phases(**failed):
    row = dict(
        phase="failed", actual=CONSUMED_RETRY,
        intended=INTENDED_SUPERSESSION, attempt_consumed=None, **failed,
    )
    return (
        dict(phase="claimed", actual=None, intended=None, attempt_consumed=None),
        dict(phase="context", actual=None, intended=None, attempt_consumed=None),
        row,
    )


def test_imports_this_checkout():
    assert MODULE.is_file()
    assert "ipfs_accelerate_py.agent_supervisor.todo_daemon.terminal_phase_reconciliation" in (
        qualify_failed_phase.__module__,
    )


def test_matching_null_phases_are_reconciled():
    result = qualify_failed_phase(
        phase="claimed", actual=None, intended=None, attempt_consumed=None,
    )
    assert result["blocked"] is False
    assert result["reconciled"] is True
    assert result["skip_mutation"] is False
    assert result["independent_work_admitted"] is True
    assert result["preserved_actual"] is None


def test_matching_terminal_names_are_reconciled():
    result = qualify_failed_phase(
        phase="failed", actual=CONSUMED_RETRY, intended=CONSUMED_RETRY,
        attempt_consumed=True,
    )
    assert result["reconciled"] is True
    assert result["skip_mutation"] is False
    assert result["preserved_actual"] == CONSUMED_RETRY


def test_proven_consumed_retry_skips_supersession_mutation():
    data = ordinary_finalizer()
    before = deepcopy(data)
    result = qualify_failed_phase(
        phase="failed", actual=CONSUMED_RETRY, intended=INTENDED_SUPERSESSION,
        attempt_consumed=None, ordinary_finalizer=data,
    )
    assert result == {
        "blocked": False,
        "reconciled": False,
        "independent_work_admitted": True,
        "skip_mutation": True,
        "preserved_actual": CONSUMED_RETRY,
        "attempt_consumed": True,
        "reason": "preserved_consumed_retry_actual",
    }
    assert data == before
    assert result["preserved_actual"] != INTENDED_SUPERSESSION


def test_mismatch_without_ordinary_proof_stays_blocked():
    result = qualify_failed_phase(
        phase="failed", actual=CONSUMED_RETRY, intended=INTENDED_SUPERSESSION,
        attempt_consumed=None,
    )
    assert result["blocked"] is True
    assert result["independent_work_admitted"] is False
    assert result["skip_mutation"] is False
    assert result["preserved_actual"] == CONSUMED_RETRY
    assert result["reason"] == "terminal_phase_changed_its_actual_database_disposition"


def test_explicit_unconsumed_flag_cannot_skip_mutation():
    result = qualify_failed_phase(
        phase="failed", actual=CONSUMED_RETRY, intended=INTENDED_SUPERSESSION,
        attempt_consumed=False, ordinary_finalizer=ordinary_finalizer(),
    )
    assert result["blocked"] is True
    assert result["skip_mutation"] is False
    assert result["attempt_consumed"] is False


def test_explicit_consumed_flag_with_proof_skips_mutation():
    result = qualify_failed_phase(
        phase="failed", actual=CONSUMED_RETRY, intended=INTENDED_SUPERSESSION,
        attempt_consumed=True, ordinary_finalizer=ordinary_finalizer(),
    )
    assert result["skip_mutation"] is True
    assert result["attempt_consumed"] is True
    assert result["blocked"] is False
    assert result["reconciled"] is False


def test_inverse_mismatch_never_consumes_a_revoked_attempt():
    result = qualify_failed_phase(
        phase="failed", actual=INTENDED_SUPERSESSION, intended=CONSUMED_RETRY,
        attempt_consumed=None, ordinary_finalizer=ordinary_finalizer(),
    )
    assert result["blocked"] is True
    assert result["skip_mutation"] is False
    assert result["preserved_actual"] == INTENDED_SUPERSESSION


def test_saga_receipt_is_not_ordinary_finalizer_proof():
    data = ordinary_finalizer()
    data["receipt"]["terminal_reconciliation"] = {"unverified": True}
    result = qualify_failed_phase(
        phase="failed", actual=CONSUMED_RETRY, intended=INTENDED_SUPERSESSION,
        ordinary_finalizer=data,
    )
    assert result["blocked"] is True
    assert result["reason"] == "terminal_phase_changed_its_actual_database_disposition"


def test_batch_with_proven_skip_admits_independent_work():
    result = qualify_terminal_repair_batch(
        pctdd_005_phases(), ordinary_finalizer=ordinary_finalizer(),
    )
    assert result["blocked"] is False
    assert result["reconciled"] is False
    assert result["independent_work_admitted"] is True
    assert result["reconciliation_complete"] is False
    assert result["completion_authorized"] is False
    assert result["repair_batch_pending"] is False
    assert result["reason"] == "preserved_consumed_retry_actual"
    assert result["phases"][2]["skip_mutation"] is True
    assert result["phases"][2]["preserved_actual"] == CONSUMED_RETRY


def test_batch_without_proof_blocks_independent_work():
    result = qualify_terminal_repair_batch(pctdd_005_phases())
    assert result["blocked"] is True
    assert result["independent_work_admitted"] is False
    assert result["completion_authorized"] is False
    assert result["reason"] == "terminal_reconciliation_receipt_repair_failed"
    assert result["phases"][2]["reason"] == (
        "terminal_phase_changed_its_actual_database_disposition"
    )


def test_malformed_phase_row_blocks_the_batch():
    result = qualify_terminal_repair_batch([{"actual": CONSUMED_RETRY}])
    assert result["blocked"] is True
    assert result["independent_work_admitted"] is False
    assert result["reason"] == "terminal_reconciliation_receipt_repair_failed"


def test_skip_mutation_never_grants_completion_or_rewrites_inputs():
    phases = list(pctdd_005_phases())
    before = deepcopy(phases)
    result = qualify_terminal_repair_batch(
        phases, ordinary_finalizer=ordinary_finalizer(),
    )
    assert phases == before
    assert result["completion_authorized"] is False
    assert all(row.get("actual") != INTENDED_SUPERSESSION for row in phases)
