"""Qualified projection-wait recovery must retain all negative evidence gates."""
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


@pytest.fixture
def reconciled_snapshot(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    from ipfs_accelerate_py.agent_supervisor.rescue import verification_deferral_recovery as recovery
    from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event

    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True,
            capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    (repo / 'protected.txt').write_text('sealed\n')
    git('add', 'protected.txt')
    git('-c', 'user.name=Recovery Test', '-c', 'user.email=recovery-test@localhost', 'commit', '-qm', 'baseline')
    baseline = git('rev-parse', 'HEAD')
    runtime = repo / 'runtime'
    workspace = runtime / 'worktrees/candidate'
    git('worktree', 'add', '--detach', str(workspace), baseline)
    attempt_root = runtime / 'attempts'
    root = attempt_root / hashlib.sha256(b'attempt:1').hexdigest()[:24]
    root.mkdir(parents=True)
    binding = {'attempt_id': 'attempt:1', 'task_cid': 'task:1', 'claim_id': 'claim:1',
        'task_contract_digest': 'contract:1'}
    binding['binding_id'] = recovery.digest(binding)
    (root / 'database-attempt-binding.json').write_text(json.dumps(binding))
    event_file = root / 'portal-events.jsonl'
    append_jsonl_event(event_file, 'implementation_finished', {
        'task_id': 'SPAR-001', 'canonical_task_cid': 'task:1', 'attempt': 1,
        'attempt_consumed': False, 'provider_dispatched': True,
        'worktree_path': str(workspace), 'baseline_ref': baseline,
        'protected_path_violation': {'verification_deferred': True, 'reason': recovery.TIMEOUT}})
    append_jsonl_event(event_file, 'implementation_protected_path_snapshot_reconciled', {
        'task_id': 'SPAR-001', 'attempt': 1, 'reason': 'crash_reconciliation_unchanged'})
    monkeypatch.setattr(recovery, 'database_portal_task_contract_digest', lambda task: 'contract:1')
    monkeypatch.setattr(recovery.DatabasePortalExecutionBridge, '_verify_projection', lambda *args: '')
    kwargs = dict(repo=repo, runtime=runtime, attempt_root=attempt_root,
        task=SimpleNamespace(task_cid='task:1', task_alias='SPAR-001'),
        attempts=[{'attempt_id': 'attempt:2'}, {'attempt_id': 'attempt:1', 'claim_id': 'claim:1'}],
        config={'protected_paths': ['protected.txt']})
    return recovery, root, workspace, kwargs


def test_previously_reconciled_snapshot_default_is_read_only(reconciled_snapshot):
    recovery, root, workspace, kwargs = reconciled_snapshot
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    proof = recovery.qualify_snapshot(**kwargs)
    assert proof['target_attempt_id'] == 'attempt:2'
    assert {p: p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
    assert (workspace / 'protected.txt').read_text() == 'sealed\n'


def test_active_snapshot_default_never_invokes_reconciler(reconciled_snapshot, monkeypatch):
    recovery, root, _, kwargs = reconciled_snapshot
    active = root / 'implementation-protected-path-active.json'
    active.write_text('{}')
    monkeypatch.setattr(recovery, 'PortalImplementationDaemon',
        lambda **kw: pytest.fail('read-only qualification invoked a mutating native reconciler'))
    with pytest.raises(VerificationDeferralRecoveryError, match='requires --apply'):
        recovery.qualify_snapshot(**kwargs)
    assert active.read_text() == '{}'
    assert not (root / 'verification-deferral-recovery.json').exists()


def test_untracked_protected_file_cannot_reuse_old_reconciliation(reconciled_snapshot):
    recovery, _, workspace, kwargs = reconciled_snapshot
    (workspace / 'untracked.txt').write_text('unverified')
    kwargs['config']['protected_paths'].append('untracked.txt')
    with pytest.raises(VerificationDeferralRecoveryError, match='untracked'):
        recovery.qualify_snapshot(**kwargs, apply=True)


def test_changed_protected_content_rejects_old_reconciliation(reconciled_snapshot):
    recovery, _, workspace, kwargs = reconciled_snapshot
    (workspace / 'protected.txt').write_text('mutated')
    with pytest.raises(VerificationDeferralRecoveryError, match='local changes'):
        recovery.qualify_snapshot(**kwargs, apply=True)


def test_snapshot_rearms_only_one_exact_target_attempt(reconciled_snapshot):
    recovery, root, _, kwargs = reconciled_snapshot
    proof = recovery.qualify_snapshot(**kwargs, apply=True)
    assert json.loads((root / 'verification-deferral-recovery.json').read_text()) == proof
    assert recovery.qualify_snapshot(**kwargs, apply=True) == proof
    kwargs['attempts'][0]['attempt_id'] = 'attempt:3'
    with pytest.raises(VerificationDeferralRecoveryError, match='already rearmed'):
        recovery.qualify_snapshot(**kwargs, apply=True)


def test_recovery_receipt_preserves_exact_route_and_previous_authority():
    from ipfs_accelerate_py.agent_supervisor.rescue.verification_deferral_recovery import recovery_receipt
    previous = {'execution_route_binding': {'policy_id': 'route:1', 'task_cid': 'task:1'},
        'execution_route_policy_id': 'route:1', 'execution_route_origin_revision': 1}
    proof = {'receipt_id': 'snapshot:1'}
    receipt = recovery_receipt(previous, proof)
    assert all(receipt[key] == value for key, value in previous.items())
    assert receipt['source_receipt'] == previous
    assert receipt['qualification'] == proof
    assert receipt['completion_authority'] is False
    previous.pop('execution_route_origin_revision')
    with pytest.raises(VerificationDeferralRecoveryError, match='partial execution route'):
        recovery_receipt(previous, proof)
