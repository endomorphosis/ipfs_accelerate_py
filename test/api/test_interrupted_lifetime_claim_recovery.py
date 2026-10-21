"""Real native claim/lifecycle/event proofs retain spent lifetime accounting."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from test.api.test_agent_supervisor_implementation_protected_paths import (
    _persist_live_shaped_interrupted_attempt, _protected_git_worktree_daemon,
)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    monkeypatch.setenv(native.AUTOMATIC_RETRY_REARM_ENV, '0')
    # This matrix exercises claim proof, not the production two-second
    # maintenance latency policy. Cold DuckDB/filesystem setup can exceed that
    # limit under concurrent tests; keep the real fence protocol bounded at 10s.
    reconciler = native.CrashFenceReconciler
    monkeypatch.setattr(native, 'CrashFenceReconciler',
        lambda daemon: reconciler(daemon, max_hold_seconds=10.0))
    daemon, repo, workspace, _ = _protected_git_worktree_daemon(tmp_path)
    task, lock_path, claim_path = _persist_live_shaped_interrupted_attempt(
        daemon, workspace=workspace, attempt=1)
    state_dir = daemon.state_path.parent
    state_dir.chmod(0o700)
    state = native.PortalTaskState.load(daemon.state_path)
    cid = state.last_implementation_task_cid
    start = next(e for e in daemon._iter_events() if e.get('type') == 'implementation_started')
    # Reproduce the authoritative order through the native append API; this
    # fixture launches no worker, container, model or network request.
    daemon.events_path.unlink()
    daemon.events_path.with_name(daemon.events_path.name + '.manifest.json').unlink(missing_ok=True)
    fingerprint = 'sha256:' + hashlib.sha256(b'synthetic-prior-rearm').hexdigest()
    daemon._record_event('ordinary_task_validation_obsolescence_rearmed', {
        'rearmed': [{'task_id': task.task_id, 'canonical_task_cid': cid,
            'previous_attempt_count': 3, 'previous_canonical_attempt_count': 3,
            'previous_display_attempt_counts': {task.task_id: 3}, 'rearm_fingerprint': fingerprint}]})
    reserved = {'timestamp','type','stream_id','snapshot_id','sequence','event_id','previous_event_id'}
    daemon._record_event('implementation_started', {k:v for k,v in start.items() if k not in reserved})
    events = list(daemon._iter_events())
    rearm, start = events[-2:]
    state.implementation_attempts[task.task_id] = 4
    state.implementation_attempts_by_cid[cid] = 4
    state.validation_obsolescence_rearm_receipts[cid] = [fingerprint]
    state.save(daemon.state_path)
    # A real short-lived successor terminalizes the original dead owner;
    # unlike the test process, its captured terminal birth is actually dead.
    child_code = '''import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
s=WorktreeLifecycleStore(repo_root=Path(sys.argv[1]))
r=s.reclaim_dead_owner_for_controlled_restart(sys.argv[2],expected_state_dir=sys.argv[3],reason='controlled_shutdown_quiesced_owner')
print(json.dumps(r.to_dict()))
'''
    terminal = json.loads(subprocess.run([sys.executable, '-c', child_code,
        str(repo), str(workspace), str(state_dir)], check=True, capture_output=True,
        text=True, timeout=20).stdout)
    claim = json.loads(claim_path.read_text())
    claim['pid'] = terminal['owner']['pid']
    claim['owner_process_birth'] = terminal['owner']
    claim['dispatch_authority_cid'] = daemon._implementation_dispatch_authority_cid(claim)
    _write(claim_path, claim)
    # Real native shutdown proves dead ownership, terminalizes its lifecycle,
    # and clears active state. The existing terminal-task API correctly keeps
    # this pending task's ordinal1/lifetime4 claim fenced.
    cleanup = daemon.reconcile_quiesced_active_attempt()
    assert cleanup['blocked'] and cleanup['reason'] == 'task_claim_reconciliation_blocked'
    assert claim_path.exists()
    state = native.PortalTaskState.load(daemon.state_path)
    assert not state.implementation_in_progress and state.active_attempt == 0
    assert state.implementation_attempts_by_cid[cid] == 4
    claim = json.loads(claim_path.read_text())
    correction = {'schema': 'synthetic-lifetime-accounting-correction@1',
        'repo_root': str(repo.resolve()), 'state_dir': str(state_dir.resolve()),
        'task_id': task.task_id, 'canonical_task_cid': cid, 'plan_id': 'synthetic-plan',
        'original_post_rearm_local_ordinal': 1, 'prior_consumed_attempts': 3,
        'lifetime_consumed_attempts_after': 4, 'native_current_consumed_attempts_before': 1,
        'changed_fields': ['implementation_attempts', 'implementation_attempts_by_cid'],
        'rearm_history_preserved': True, 'local_ordinal_relabelled': False,
        'rearm_event_id': rearm['event_id'], 'post_rearm_dispatch_event_id': start['event_id'],
        'rearm_fingerprint': fingerprint,
        'event_log_sha256': hashlib.sha256(daemon.events_path.read_bytes()).hexdigest(),
        **{k:False for k in ('attempt_refunded','completion_authority','candidate_adoption_authority',
            'evidence_reuse_authority','new_attempt_authority','old_quarantine_released')}}
    launcher = {'identity': {'pid': claim['pid'], 'start_ticks': claim['owner_process_birth']['start_time_ticks'],
        'boot_id': claim['owner_process_birth']['boot_id']}, 'status': 'stopped',
        'owner_token': 'synthetic-owner-token', 'started_at': 1.0,
        'repo_root': str(repo.resolve()), 'state_dir': str(state_dir.resolve()), 'plan_id': 'synthetic-plan'}
    launcher_hash = _write(state_dir / 'launcher.json', launcher)
    shutdown = {'schema': 'synthetic-container-shutdown@1', 'repo_root': str(repo.resolve()),
        'state_dir': str(state_dir.resolve()), 'plan_id': 'synthetic-plan',
        'status': 'stopped', 'quiesced': True, 'reason': 'owned_container_closure_verified',
        'manual_holds_preserved': True, 'launcher_sha256': launcher_hash,
        'owner_token_sha256': hashlib.sha256(b'synthetic-owner-token').hexdigest(), 'started_at': 1.0,
        'event_log_sha256': correction['event_log_sha256'],
        'attempt_refunded': False, 'completion_authority': False, 'provider_callback_settled': False}
    correction_path, shutdown_path = state_dir / 'accounting.json', state_dir / 'shutdown.json'
    arguments = {'accounting_correction_receipt_path': correction_path,
        'accounting_correction_sha256': _write(correction_path, correction),
        'container_shutdown_receipt_path': shutdown_path,
        'container_shutdown_sha256': _write(shutdown_path, shutdown),
        'expected_recovery_runtime_revision': native.retry_budget_repair_runtime_revision(),
        'expected_claim_lease_id': claim['lease_id']}
    daemon._record_event('fixture_later_quiescence_observation', {'synthetic_only': True})
    return daemon, task, claim_path, correction, shutdown, arguments


def test_exact_pending_interrupted_claim_releases_without_refund_or_task_completion(interrupted):
    daemon, task, claim_path, _, _, arguments = interrupted
    before_state = asdict(native.PortalTaskState.load(daemon.state_path))
    board = daemon.todo_path.read_bytes()
    result = daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert result['reconciled'] and not result['blocked'], result.get('reason')
    assert result['attempt'] == 1 and result['lifetime_consumed_attempts'] == 4
    assert result['task_status'] == 'todo'
    assert all(result[k] is False for k in ('completion_authority','attempt_refunded',
        'provider_callback_settled','new_attempt_authority'))
    assert not claim_path.exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before_state
    assert daemon.todo_path.read_bytes() == board
    repeat = daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert repeat['reason'] == 'quiesced_interrupted_task_claim_already_released', repeat
    assert repeat['receipt_id'] == result['receipt_id']
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before_state


@pytest.mark.parametrize('damage', ['correction_hash','shutdown_hash','source_revision',
    'claim_lease','local_ordinal','lifetime_count','rearm_history','rearm_event',
    'start_event','history_prefix','launcher_bytes','launcher_live','claim_live',
    'protected_hold','task_hold','manual_hold','active_state','later_attempt','correction_authority',
    'malformed_tail','foreign_token','foreign_start','policy_enabled','unrelated_dead_launcher'])
def test_unproved_or_live_attempt_never_releases_claim(interrupted, monkeypatch, damage):
    daemon, task, claim_path, correction, shutdown, arguments = interrupted
    if damage == 'correction_hash': arguments['accounting_correction_sha256'] = '0' * 64
    elif damage == 'shutdown_hash': arguments['container_shutdown_sha256'] = '0' * 64
    elif damage == 'source_revision': arguments['expected_recovery_runtime_revision'] = 'sha256:' + '0' * 64
    elif damage == 'claim_lease': arguments['expected_claim_lease_id'] = 'foreign-lease'
    elif damage in ('local_ordinal','lifetime_count','rearm_event','start_event','history_prefix','correction_authority'):
        field, value = {'local_ordinal':('original_post_rearm_local_ordinal', 2),
            'lifetime_count':('lifetime_consumed_attempts_after',5),
            'rearm_event':('rearm_event_id','sha256:'+'0'*64),
            'start_event':('post_rearm_dispatch_event_id','sha256:'+'0'*64),
            'history_prefix':('event_log_sha256','0'*64),
            'correction_authority':('attempt_refunded',True)}[damage]
        correction[field] = value
        arguments['accounting_correction_sha256'] = _write(arguments['accounting_correction_receipt_path'], correction)
    elif damage == 'launcher_bytes':
        p=daemon.state_path.parent/'launcher.json';d=json.loads(p.read_text());d['status']='running';_write(p,d)
    elif damage == 'unrelated_dead_launcher':
        p=daemon.state_path.parent/'launcher.json';d=json.loads(p.read_text())
        d['identity']['pid']=2**30-17;d['identity']['start_ticks']=1
        _write(p,d)
    elif damage == 'launcher_live':
        p=daemon.state_path.parent/'launcher.json';d=json.loads(p.read_text())
        birth=native.current_process_birth();d['identity']={'pid':birth.pid,'start_ticks':birth.start_time_ticks,'boot_id':birth.boot_id}
        shutdown['launcher_sha256']=_write(p,d)
        arguments['container_shutdown_sha256']=_write(arguments['container_shutdown_receipt_path'],shutdown)
    elif damage == 'claim_live':
        d=json.loads(claim_path.read_text());birth=native.current_process_birth()
        d.update(pid=birth.pid,owner_process_birth=birth.to_dict());_write(claim_path,d)
    elif damage == 'protected_hold':
        p=daemon._implementation_protected_active_snapshot_path();p.parent.mkdir(parents=True,exist_ok=True);_write(p,{'synthetic_hold':True})
    elif damage == 'policy_enabled': monkeypatch.setenv(native.AUTOMATIC_RETRY_REARM_ENV, '1')
    elif damage == 'manual_hold':
        daemon.todo_path.write_text(daemon.todo_path.read_text() + '\n- Manual hold: true\n')
    elif damage == 'malformed_tail':
        with daemon.events_path.open('ab') as handle: handle.write(b'{broken tail')
    elif damage in ('foreign_token', 'foreign_start'):
        path=daemon.state_path.parent/'launcher.json';value=json.loads(path.read_text())
        value['owner_token' if damage=='foreign_token' else 'started_at']='foreign-token' if damage=='foreign_token' else 2.0
        _write(path,value)
    elif damage == 'task_hold':
        daemon.todo_path.write_text(daemon.todo_path.read_text().replace('- Status: ready','- Status: blocked'))
    elif damage == 'later_attempt':
        daemon._record_event('implementation_started',{'task_id':task.task_id,'canonical_task_cid':correction['canonical_task_cid'],'attempt':2})
    else:
        state=native.PortalTaskState.load(daemon.state_path)
        if damage == 'rearm_history': state.validation_obsolescence_rearm_receipts={}
        elif damage == 'active_state': state.active_attempt=1;state.implementation_in_progress=True
        state.save(daemon.state_path)
    before = claim_path.read_bytes()
    events_before = daemon.events_path.read_bytes()
    state_before=asdict(native.PortalTaskState.load(daemon.state_path))
    result=daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert result['blocked'] and not result['reconciled'], result
    assert claim_path.read_bytes()==before
    assert daemon.events_path.read_bytes()==events_before
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == state_before


def test_different_claim_is_never_deleted_by_replayed_recovery(interrupted):
    daemon, _, claim_path, _, _, arguments=interrupted
    saved=json.loads(claim_path.read_text())
    first=daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert first['reconciled'], first
    saved['lease_id']='new-owner-lease'
    _write(claim_path,saved)
    before=claim_path.read_bytes()
    second=daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert second['blocked'] and claim_path.read_bytes()==before


def test_release_receipt_write_interruption_resumes_exact_prepared_operation(interrupted, monkeypatch):
    daemon, _, claim_path, _, _, arguments=interrupted
    original=daemon._write_private_task_claim_release_receipt
    def interrupt(path,value):
        if value.get('operation')=='release_quiesced_interrupted_implementation_task_claim' and value.get('phase')=='released':
            raise OSError('synthetic interrupted final receipt write')
        return original(path,value)
    monkeypatch.setattr(daemon,'_write_private_task_claim_release_receipt',interrupt)
    first=daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert first['blocked'] and not claim_path.exists()
    monkeypatch.setattr(daemon,'_write_private_task_claim_release_receipt',original)
    resumed=daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert resumed['reconciled'], resumed
    assert json.loads(Path(resumed['receipt_path']).read_text())['recovered_after_claim_unlink'] is True


def test_dead_daemon_claim_is_bound_to_dead_launcher_lifecycle(interrupted):
    daemon, _, claim_path, _, _, arguments = interrupted
    claim = json.loads(claim_path.read_text())
    launcher_pid = claim['pid']
    claim['pid'] = 2**30 - 57
    claim['owner_process_birth'] = native.ProcessBirthIdentity(
        pid=claim['pid'], start_time_ticks=claim['owner_process_birth']['start_time_ticks'] + 1,
        boot_id=claim['owner_process_birth']['boot_id'], parent_pid=launcher_pid).to_dict()
    claim['dispatch_authority_cid'] = daemon._implementation_dispatch_authority_cid(claim)
    _write(claim_path, claim)
    result = daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert result['reconciled'] and not claim_path.exists(), result.get('reason')


def test_launcher_stop_metadata_may_advance_without_changing_owner(interrupted):
    daemon, _, claim_path, _, _, arguments = interrupted
    path = daemon.state_path.parent / 'launcher.json'
    launcher = json.loads(path.read_text())
    launcher['stopped_at'] = '2026-10-21T04:00:00+00:00'
    _write(path, launcher)
    result = daemon.reconcile_quiesced_interrupted_task_claim(**arguments)
    assert result['reconciled'] and not claim_path.exists(), result.get('reason')
