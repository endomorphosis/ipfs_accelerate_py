"""Qualified projection-wait recovery must retain all negative evidence gates."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.verification_deferral_recovery import (
    REASON, VerificationDeferralRecoveryError, digest, require_stopped, verify_blocked,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection


@pytest.fixture
def bound_wait(tmp_path):
    connection = open_duckdb_connection(tmp_path / "execution.duckdb")
    connection.execute("CREATE TABLE database_task_attempts (attempt_id VARCHAR, task_cid VARCHAR, "
        "task_alias VARCHAR, attempt_number BIGINT, status VARCHAR, revision BIGINT, "
        "claim_id VARCHAR, lease_id VARCHAR, owner_session_id VARCHAR, fencing_token BIGINT, fence_epoch BIGINT)")
    connection.execute("CREATE TABLE attempt_phases (attempt_id VARCHAR, phase VARCHAR, body_json VARCHAR)")
    identity = {"attempt_id": "attempt:2", "attempt_number": 2, "claim_id": "claim:2",
        "lease_id": "lease:2", "owner_session_id": "lane:1", "fencing_token": 2, "fence_epoch": 2}
    connection.execute("INSERT INTO database_task_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [identity['attempt_id'], 'task:1', 'SPAR-001', 2, 'failed', 3, 'claim:2', 'lease:2', 'lane:1', 2, 2])
    failed = {"attempt_consumed": False, "provider_dispatched": False, "reason": REASON}
    connection.execute("INSERT INTO attempt_phases VALUES (?, ?, ?)", ['attempt:2', 'failed', json.dumps(failed)])
    budget = {"exhausted": True, "task_cid": "task:1", "matching_attempts": [{
        "attempt_id": "attempt:2", "attempt_number": 2, "reason": REASON}]}
    budget['observation_id'] = digest(budget)
    receipt = {**identity, "operation": "database_portal_typed_deferral_budget_exhausted",
        "attempt_consumed": False, "control_expected_revision": 3, "execution_revision": 3,
        "retry_budget": budget}
    task = SimpleNamespace(task_cid='task:1', task_alias='SPAR-001', status='blocked', revision=4,
        body={"completion_receipt": receipt})
    yield task, connection
    connection.close()


def test_exact_unconsumed_wait_qualifies_without_mutating_execution(bound_wait):
    task, connection = bound_wait
    before = connection.execute("SELECT * FROM attempt_phases").fetchall()
    receipt, attempts = verify_blocked(task, connection)
    assert receipt == task.body['completion_receipt']
    assert attempts[0]['attempt_id'] == 'attempt:2'
    assert [tuple(row) for row in connection.execute("SELECT * FROM attempt_phases").fetchall()] == [tuple(row) for row in before]


@pytest.mark.parametrize('field,value', [('revision', 5), ('status', 'completed')])
def test_stale_or_completed_control_rejected(bound_wait, field, value):
    task, connection = bound_wait
    setattr(task, field, value)
    with pytest.raises(VerificationDeferralRecoveryError): verify_blocked(task, connection)


@pytest.mark.parametrize('body', [{'completion': 'manual'}, {'completion': {'mode': 'manual'}}, {'review_only': 'true'}])
def test_manual_task_cannot_be_reopened(bound_wait, body):
    task, connection = bound_wait
    task.body.update(body)
    with pytest.raises(VerificationDeferralRecoveryError, match='manual'): verify_blocked(task, connection)


@pytest.mark.parametrize('phase', ['provider', 'effect', 'validation', 'complete'])
def test_committed_effect_cannot_be_reopened_as_unconsumed(bound_wait, phase):
    task, connection = bound_wait
    connection.execute("INSERT INTO attempt_phases VALUES ('attempt:2', ?, '{}')", [phase])
    with pytest.raises(VerificationDeferralRecoveryError, match='committed an effect'): verify_blocked(task, connection)


@pytest.mark.parametrize('field', ['attempt_consumed', 'provider_dispatched'])
def test_consumed_or_dispatched_failed_phase_rejected(bound_wait, field):
    task, connection = bound_wait
    failure = {"attempt_consumed": False, "provider_dispatched": False, "reason": REASON, field: True}
    connection.execute("UPDATE attempt_phases SET body_json = ?", [json.dumps(failure)])
    with pytest.raises(VerificationDeferralRecoveryError, match='no provider dispatch'): verify_blocked(task, connection)


def test_foreign_budget_reason_is_not_recovery_authority(bound_wait):
    task, connection = bound_wait
    budget = task.body['completion_receipt']['retry_budget']
    budget['matching_attempts'][0]['reason'] = 'validation_failed'
    budget.pop('observation_id'); budget['observation_id'] = digest(budget)
    with pytest.raises(VerificationDeferralRecoveryError, match='foreign'): verify_blocked(task, connection)


def test_live_owner_rejected_even_when_status_claims_stopped(tmp_path):
    database = tmp_path / 'control.duckdb'
    status = tmp_path / 'status.json'
    ticks = int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19])
    status.write_text(json.dumps({'lifecycle': 'stopped', 'database_path': str(database),
        'identity': {'process_birth': {'pid': os.getpid(), 'start_time_ticks': ticks,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}}}))
    with pytest.raises(VerificationDeferralRecoveryError, match='still alive'):
        require_stopped(status, database)
