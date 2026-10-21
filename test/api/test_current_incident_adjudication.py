"""One ordinary incident can retire custody without changing task authority."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import current_incident_adjudication as adjudication
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import AUTONOMOUS_UNSTALL_STATE_SCHEMA
from test.api.test_agent_supervisor_implementation_protected_paths import (
    _persist_live_shaped_interrupted_attempt, _protected_git_worktree_daemon,
    _protected_git_worktree_supervisor,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def incident(tmp_path, monkeypatch):
    monkeypatch.setenv(native.AUTOMATIC_RETRY_REARM_ENV, '0')
    reconciler = native.CrashFenceReconciler
    monkeypatch.setattr(native, 'CrashFenceReconciler', lambda daemon: reconciler(daemon, max_hold_seconds=10))
    daemon, repo, workspace, _ = _protected_git_worktree_daemon(tmp_path)
    daemon.state_path = tmp_path / 'state' / 'portal_task_state.json'
    daemon.strategy_path = tmp_path / 'state' / 'portal_strategy.json'
    task, lease, claim_path = _persist_live_shaped_interrupted_attempt(daemon, workspace=workspace)
    state_dir = daemon.state_path.parent
    state_dir.chmod(0o700)
    state = native.PortalTaskState.load(daemon.state_path)
    start = next(e for e in daemon._iter_events() if e.get('type') == 'implementation_started')
    log = Path(start['log_path']); log.write_text('synthetic invocation header; no provider was launched\n'); log.chmod(0o600)
    os.utime(log, (1, 1))
    code = '''import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
s=WorktreeLifecycleStore(repo_root=Path(sys.argv[1]))
r=s.reclaim_dead_owner_for_controlled_restart(sys.argv[2],expected_state_dir=sys.argv[3],reason='controlled_shutdown_quiesced_owner')
print(json.dumps(r.to_dict()))
'''
    birth_code = 'import json; from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth; print(json.dumps(current_process_birth().to_dict()))'
    wrapper_code = """import json,subprocess,sys
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth
launcher=current_process_birth().to_dict()
dispatch=json.loads(subprocess.run([sys.executable,'-c',sys.argv[1]],check=True,capture_output=True,text=True).stdout)
terminal=json.loads(subprocess.run([sys.executable,'-c',sys.argv[2],*sys.argv[3:]],check=True,capture_output=True,text=True).stdout)
print(json.dumps({'launcher':launcher,'dispatch':dispatch,'terminal':terminal}))
"""
    tree = json.loads(subprocess.run([sys.executable, '-c', wrapper_code, birth_code, code,
        str(repo), str(workspace), str(state_dir)], check=True,capture_output=True,text=True,timeout=40).stdout)
    terminal = tree['terminal']
    assert tree['dispatch']['parent_pid'] == terminal['owner']['parent_pid'] == tree['launcher']['pid']
    assert tree['dispatch']['pid'] != terminal['owner']['pid']
    claim = json.loads(claim_path.read_text())
    claim.update(pid=tree['dispatch']['pid'], owner_process_birth=tree['dispatch'])
    claim['dispatch_authority_cid'] = daemon._implementation_dispatch_authority_cid(claim)
    write(claim_path, claim)
    lease_value=json.loads(lease.read_text())
    lease_value.update(pid=tree['dispatch']['pid'], owner_process_birth=tree['dispatch'])
    write(lease,lease_value)
    supervisor = _protected_git_worktree_supervisor(daemon, repo)
    supervisor.rewrite_strategy(state, 'heartbeat stale for active task ' + task.task_id)
    incident_cid = 'synthetic-incident-cid'
    reason = 'rescue_not_permitted_after_deterministic_exhaustion'
    scope = supervisor._quarantine_autonomous_unstall_scope(['lane:implementation'], incident_cid, reason)
    assert scope['task_id'] == task.task_id
    record = {'incident_cid': incident_cid, 'phase': 'quarantined', 'kind': 'stale_heartbeat',
              'reason': reason, 'target_ids': ['lane:implementation'], 'completion_authority': False,
              'updated_at_ms': 1}
    write(supervisor._autonomous_unstall_state_path(), {'schema': AUTONOMOUS_UNSTALL_STATE_SCHEMA,
        'incidents': {incident_cid: record}})
    # Native shutdown retires real lifecycle/protected leases but correctly
    # retains the pending task's global claim. No model is ever invoked.
    cleanup = daemon.reconcile_quiesced_active_attempt()
    assert cleanup['blocked'] and cleanup['reason'] == 'task_claim_reconciliation_blocked', cleanup
    assert claim_path.exists()
    state = native.PortalTaskState.load(daemon.state_path)
    assert not state.implementation_in_progress and not state.active_task_id
    launcher = {'identity': {'pid': tree['launcher']['pid'], 'start_ticks': tree['launcher']['start_time_ticks'],
        'boot_id': tree['launcher']['boot_id']}, 'status': 'stopped', 'owner_token': 'synthetic-secret-owner-token',
        'argv': ['private-launcher-argument-not-for-receipts'], 'started_at': 1.0,
        'repo_root': str(repo.resolve()), 'state_dir': str(state_dir.resolve()), 'plan_id': 'synthetic-plan'}
    write(state_dir / 'launcher.json', launcher)
    write(state_dir / 'container_shutdown.json', {'status': 'stopped', 'quiesced': True,
        'reason': 'owned_container_closure_verified', 'repo_root': str(repo.resolve()), 'state_dir': str(state_dir.resolve()),
        'plan_id': 'synthetic-plan', 'started_at': 1.0,
        'owner_token_sha256': hashlib.sha256(launcher['owner_token'].encode()).hexdigest(),
        'attempt_refunded': False, 'completion_authority': False, 'provider_callback_settled': False})
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
    monkeypatch.setattr(grok_cli_runner, '_docker_isolation_binary', lambda: '/synthetic-trusted-docker')
    real_run = subprocess.run
    def run(argv, **kwargs):
        if argv[0] == '/synthetic-trusted-docker':
            return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')
        return real_run(argv, **kwargs)
    monkeypatch.setattr(adjudication.subprocess, 'run', run)
    request = daemon.review_quiesced_ordinary_incident(task_id=task.task_id, incident_cid=incident_cid)
    return daemon, supervisor, task, workspace, claim_path, record, request


def test_pending_incident_abandons_only_exact_custody_preserving_native_state_and_history(incident):
    daemon, supervisor, task, workspace, claim_path, record, request = incident
    before_state = asdict(native.PortalTaskState.load(daemon.state_path))
    before = {name: Path(request['pins'][name]['path']).read_bytes() for name in
              ('board', 'incident', 'lifecycle', 'launcher', 'shutdown', 'log')}
    old_strategy = json.loads(daemon.strategy_path.read_text())
    result = daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['reconciled'] and not result['blocked'], result.get('reason')
    assert not claim_path.exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before_state
    assert all(Path(request['pins'][name]['path']).read_bytes() == raw for name, raw in before.items())
    strategy = json.loads(daemon.strategy_path.read_text())
    assert strategy['autonomous_unstall_quarantines'] == old_strategy['autonomous_unstall_quarantines']
    assert task.task_id not in strategy['blocked_tasks'] and task.task_id not in strategy['deprioritized_tasks']
    assert task.task_id not in daemon.load_strategy()['blocked_tasks']
    assert supervisor._autonomous_unstall_status()['latest']['phase'] == 'administratively_abandoned'
    receipt = Path(result['receipt_path']).read_text()
    assert 'synthetic-secret-owner-token' not in receipt
    assert 'private-launcher-argument-not-for-receipts' not in receipt
    assert all(result[k] is False for k in adjudication._FALSE)
    repeated = daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert repeated['reason'] == 'incident_already_adjudicated', repeated
    # The next scheduling ordinal is real native lifetime arithmetic.
    state = native.PortalTaskState.load(daemon.state_path)
    assert daemon._task_attempt(state, task) == 2
    assert state.implementation_attempts[task.task_id] == 1


@pytest.mark.parametrize('damage', ['source', 'state', 'claim_changed', 'task_revision', 'manual_hold',
    'waiting', 'blocked', 'reserved', 'protected_state', 'protected_file', 'budget',
    'capacity', 'live_launcher', 'unknown_owner', 'native_event_tail', 'supervisor_event_tail',
    'baseline_written', 'git_tree_error', 'foreign_claim_path', 'claim_symlink', 'live_lease',
    'unknown_lease', 'independent_strategy', 'different_quarantine', 'strategy_origin', 'log_changed'])
def test_unproved_or_independently_held_incident_preserves_all_custody(incident, monkeypatch, damage):
    daemon, supervisor, task, workspace, claim_path, record, request = incident
    if damage == 'source': request['runtime_revision'] = 'sha256:' + '0' * 64
    elif damage == 'state':
        s = native.PortalTaskState.load(daemon.state_path); s.active_attempt = 1; s.save(daemon.state_path)
    elif damage in ('waiting','blocked','reserved','protected_state'):
        s = native.PortalTaskState.load(daemon.state_path)
        if damage == 'protected_state': s.protected_implementation_attempts[request['task_cid']]={'task_id':task.task_id}
        else: setattr(s, {'waiting':'waiting_task_ids','blocked':'blocked_task_ids','reserved':'external_reserved_task_ids'}[damage], [task.task_id])
        s.save(daemon.state_path)
        # Review the new preimage: this must fail on the independent hold,
        # rather than merely on the state digest changing after review.
        request = daemon.review_quiesced_ordinary_incident(task_id=task.task_id, incident_cid=request['incident_cid'])
    elif damage == 'claim_changed':
        d=json.loads(claim_path.read_text()); d['lease_id']='other-lease'; write(claim_path,d)
    elif damage == 'task_revision': daemon.todo_path.write_text(daemon.todo_path.read_text().replace('src/example.py','src/other.py'))
    elif damage == 'manual_hold': daemon.todo_path.write_text(daemon.todo_path.read_text()+'\n- Manual hold: true\n')
    elif damage == 'protected_file': write(daemon._implementation_protected_active_snapshot_path(), {'held':True})
    elif damage == 'budget': daemon.max_task_attempts=1
    elif damage == 'capacity': monkeypatch.setattr(daemon,'_retained_provider_capacity_deferral',lambda: {'reason':'synthetic_hold'})
    elif damage == 'live_launcher':
        p=daemon.state_path.parent/'launcher.json';d=json.loads(p.read_text());birth=native.current_process_birth()
        d['identity']={'pid':birth.pid,'start_ticks':birth.start_time_ticks,'boot_id':birth.boot_id};write(p,d)
        request = daemon.review_quiesced_ordinary_incident(task_id=task.task_id, incident_cid=request['incident_cid'])
    elif damage == 'unknown_owner': monkeypatch.setattr(native,'owner_liveness',lambda *a,**kw:native.OwnerLiveness.UNKNOWN)
    elif damage in ('native_event_tail','supervisor_event_tail'):
        p=daemon.events_path if damage=='native_event_tail' else supervisor.config.events_path
        with p.open('ab') as h: h.write(b'{incomplete')
    elif damage == 'baseline_written':
        p=workspace/'src/example.py';p.parent.mkdir();p.write_text('operator changed pending source\n')
    elif damage == 'git_tree_error':
        previous=adjudication.subprocess.run
        def run(argv,**kw):
            if argv[0]=='git' and 'ls-tree' in argv: return subprocess.CompletedProcess(argv,128,b'',b'synthetic error')
            return previous(argv,**kw)
        monkeypatch.setattr(adjudication.subprocess,'run',run)
    elif damage == 'foreign_claim_path': request['pins']['claim']['path']=str(workspace/'foreign-claim.json')
    elif damage == 'claim_symlink':
        outside=workspace/'external-claim.json';outside.write_bytes(claim_path.read_bytes());outside.chmod(0o600)
        claim_path.unlink();claim_path.symlink_to(outside)
    elif damage in ('live_lease','unknown_lease'):
        value=daemon._build_implementation_lock_metadata(task,1,native.utc_now())
        value['owner_process_birth']=native.current_process_birth().to_dict()
        if damage=='unknown_lease':value['owner_process_birth']={}
        write(daemon._implementation_lock_path(),value)
    elif damage == 'independent_strategy':
        d=json.loads(daemon.strategy_path.read_text());d['manual_hold_tasks']=[task.task_id];write(daemon.strategy_path,d)
        request=daemon.review_quiesced_ordinary_incident(task_id=task.task_id, incident_cid=request['incident_cid'])
    elif damage == 'different_quarantine':
        d=json.loads(daemon.strategy_path.read_text());d['autonomous_unstall_quarantines'].append({**d['autonomous_unstall_quarantines'][0],'incident_cid':'other'})
        write(daemon.strategy_path,d);request=daemon.review_quiesced_ordinary_incident(task_id=task.task_id,incident_cid=request['incident_cid'])
    elif damage == 'strategy_origin':
        d=json.loads(daemon.strategy_path.read_text());d['last_rewrite_reason']='independent operator block';write(daemon.strategy_path,d)
        request=daemon.review_quiesced_ordinary_incident(task_id=task.task_id,incident_cid=request['incident_cid'])
    elif damage == 'log_changed': Path(request['pins']['log']['path']).write_text('different evidence')
    saved = {p:p.read_bytes() for p in (claim_path,daemon.strategy_path,daemon.todo_path,supervisor._autonomous_unstall_state_path())}
    state=asdict(native.PortalTaskState.load(daemon.state_path))
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and not result['reconciled'], (damage,result)
    assert all(p.read_bytes()==raw for p,raw in saved.items()),damage
    assert claim_path.exists()
    if damage=='claim_symlink': assert claim_path.is_symlink() and outside.exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path))==state


@pytest.mark.parametrize('stage',['prepared','strategy','released'])
def test_interruption_keeps_dispatch_blocked_and_exact_journal_can_resume(incident, monkeypatch, stage):
    daemon, supervisor, task, _, claim_path, _, request=incident
    original=daemon._write_private_task_claim_release_receipt
    def interrupted(path,value):
        if (stage in {'prepared','released'} and value.get('phase')==stage
                or stage=='strategy' and path==daemon.strategy_path):
            raise OSError('synthetic crash')
        return original(path,value)
    monkeypatch.setattr(daemon,'_write_private_task_claim_release_receipt',interrupted)
    before=asdict(native.PortalTaskState.load(daemon.state_path))
    first=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert first['blocked'],first
    assert task.task_id in daemon.load_strategy()['blocked_tasks']
    assert task.task_id in supervisor._load_strategy()['blocked_tasks']
    assert supervisor._autonomous_unstall_status()['latest']['phase']=='quarantined'
    monkeypatch.setattr(daemon,'_write_private_task_claim_release_receipt',original)
    resumed=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert resumed['reconciled'],resumed
    assert not claim_path.exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path))==before


def test_later_incident_preserves_settled_old_record_and_blocks_new_scope(incident):
    daemon, supervisor, task, _, _, old_record, request=incident
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['reconciled'],result
    path=supervisor._autonomous_unstall_state_path();d=json.loads(path.read_text())
    d['incidents']['new-incident']={**old_record,'incident_cid':'new-incident','updated_at_ms':2}
    write(path,d)
    strategy=json.loads(daemon.strategy_path.read_text())
    old_scope=strategy['autonomous_unstall_quarantines'][0]
    assert adjudication.settlement_valid(repo_root=daemon.repo_root,state_path=daemon.state_path,
        incident_cid=request['incident_cid'],quarantine=old_scope)
    new_scope={**old_scope,'incident_cid':'new-incident'}
    strategy['autonomous_unstall_quarantines'].append(new_scope);write(daemon.strategy_path,strategy)
    assert task.task_id in daemon.load_strategy()['blocked_tasks']
    assert supervisor._autonomous_unstall_status()['latest']['phase']=='quarantined'
    assert not adjudication.settlement_valid(repo_root=daemon.repo_root,state_path=daemon.state_path,
        incident_cid='new-incident',quarantine=new_scope)


@pytest.mark.parametrize('damage',['incomplete','corrupt','replayed_request','source_update','new_ordinal','new_claim','incident_record','task_cid'])
def test_invalid_or_replayed_sidecar_grants_no_new_cleanup(incident,monkeypatch,damage):
    daemon,supervisor,task,_,claim_path,_,request=incident
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['reconciled'],result
    path=Path(result['receipt_path']);receipt=json.loads(path.read_text())
    if damage=='incomplete':path.write_text('{incomplete')
    elif damage=='corrupt':receipt['receipt_id']='sha256:'+'0'*64;write(path,receipt)
    elif damage=='replayed_request':
        receipt['request']['incident_cid']='new-incident';write(path,receipt)
    elif damage=='source_update':monkeypatch.setattr(adjudication,'runtime_revision',lambda:'sha256:'+'f'*64)
    elif damage=='new_ordinal':
        s=native.PortalTaskState.load(daemon.state_path);s.implementation_attempts[task.task_id]=2;s.implementation_attempts_by_cid[request['task_cid']]=2;s.save(daemon.state_path)
    elif damage=='new_claim':write(claim_path,{**receipt['claim'],'lease_id':'next-claim'})
    elif damage=='incident_record':
        p=supervisor._autonomous_unstall_state_path();d=json.loads(p.read_text());d['incidents'][request['incident_cid']]['reason']='changed';write(p,d)
    elif damage=='task_cid':daemon.todo_path.write_text(daemon.todo_path.read_text().replace('src/example.py','src/other.py'))
    saved={p:p.read_bytes() for p in (daemon.todo_path,daemon.strategy_path,supervisor._autonomous_unstall_state_path())}
    claim=claim_path.read_bytes() if claim_path.exists() else None
    second=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert second['blocked'] and not second['reconciled'],(damage,second)
    assert all(p.read_bytes()==raw for p,raw in saved.items())
    assert (claim_path.read_bytes() if claim_path.exists() else None)==claim
    if damage!='new_ordinal':
        assert task.task_id in daemon.load_strategy()['blocked_tasks'],damage


@pytest.mark.parametrize('authority',['task_source','_database_attempt_authority','production_provider_policy',
    'production_owner_handoff_factory','manual_completion_authority_task_ids','manual_completion_authority_required_task_ids'])
def test_nonordinary_native_authority_cannot_release_ordinary_custody(incident,authority):
    daemon,_,_,_,claim_path,_,request=incident
    setattr(daemon,authority,object() if authority not in {'manual_completion_authority_task_ids','manual_completion_authority_required_task_ids'} else frozenset({'held'}))
    claim=claim_path.read_bytes();strategy=daemon.strategy_path.read_bytes()
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and result['reason']=='ordinary_pending_claim_required',result
    assert claim_path.read_bytes()==claim and daemon.strategy_path.read_bytes()==strategy
    assert not adjudication._journal_path(daemon.state_path.parent,request['incident_cid']).exists()


@pytest.mark.parametrize('field',['active_task_key','active_phase','active_phase_started_at','active_log_path','active_task_started_at','task_statuses'])
def test_partial_native_claim_or_status_hold_is_not_quiescent(incident,field):
    daemon,_,task,_,claim_path,_,request=incident
    s=native.PortalTaskState.load(daemon.state_path)
    setattr(s,field,{task.task_id:'waiting_user'} if field=='task_statuses' else 'retained-current-owner')
    s.save(daemon.state_path)
    request=daemon.review_quiesced_ordinary_incident(task_id=task.task_id,incident_cid=request['incident_cid'])
    before=claim_path.read_bytes()
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and claim_path.read_bytes()==before,result


@pytest.mark.parametrize('reason',['heartbeat stale','validation failed','merge failed'])
def test_same_task_fault_in_next_attempt_is_a_fresh_real_native_incident(tmp_path,reason):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import PortalImplementationSupervisor
    from test.api.test_agent_supervisor_autonomous_unstall import _coordinator
    first=native.PortalTaskState(active_task_id='PORTAL-001',active_task_cid='canonical-task-cid',active_attempt=1,
        active_task_started_at='2026-10-21T06:00:00+00:00',active_worktree_path=str(tmp_path/'attempt1'))
    second=deepcopy(first);second.active_attempt=2;second.active_task_started_at='2026-10-21T07:00:00+00:00';second.active_worktree_path=str(tmp_path/'attempt2')
    coordinator=_coordinator(tmp_path)
    one=coordinator.unstall(evidence=PortalImplementationSupervisor._autonomous_unstall_evidence(first,reason))
    two=coordinator.unstall(evidence=PortalImplementationSupervisor._autonomous_unstall_evidence(second,reason))
    assert one['incident_cid']!=two['incident_cid']
    assert one['status']=='quarantined' and two['status']=='quarantined'
    original=json.loads((tmp_path/'state'/'autonomous-unstall-state.json').read_text())
    assert original['incidents'][one['incident_cid']]['phase']=='quarantined'
    assert original['incidents'][two['incident_cid']]['phase']=='quarantined'


@pytest.mark.parametrize('consumed', [1, 2])
def test_distinct_native_attempt_two_may_retain_exact_old_history(incident, consumed):
    daemon,supervisor,task,_,claim_path,_,request=incident
    first=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert first['reconciled'],first
    old_receipt=Path(first['receipt_path']).read_bytes()
    s=native.PortalTaskState.load(daemon.state_path)
    assert daemon._task_attempt(s,task)==2
    s.implementation_attempts[task.task_id]=consumed;s.implementation_attempts_by_cid[request['task_cid']]=consumed
    s.active_task_id=task.task_id;s.active_task_cid=request['task_cid'];s.active_attempt=2 if consumed==2 else 0
    s.active_task_started_at='2026-10-21T09:00:00+00:00';s.implementation_in_progress=True;s.save(daemon.state_path)
    claim=daemon._build_implementation_task_claim_metadata(task,2,s.active_task_started_at)
    write(claim_path,claim)
    # The old settled record stays historical. This neither suppresses a new
    # incident nor releases the real later native claim.
    assert task.task_id not in daemon.load_strategy()['blocked_tasks']
    assert Path(first['receipt_path']).read_bytes()==old_receipt
    repeat=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert repeat['blocked'] and claim_path.exists()
    assert native.PortalTaskState.load(daemon.state_path).active_attempt==(2 if consumed==2 else 0)


def test_dangling_receipt_symlink_is_never_replaced(incident):
    daemon,_,_,workspace,claim_path,_,request=incident
    path=adjudication._journal_path(daemon.state_path.parent,request['incident_cid'])
    path.parent.mkdir(mode=0o700)
    target=workspace/'nonexistent-outside-receipt.json';path.symlink_to(target)
    before=claim_path.read_bytes(),daemon.strategy_path.read_bytes(),os.readlink(path)
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and path.is_symlink() and not target.exists()
    assert (claim_path.read_bytes(),daemon.strategy_path.read_bytes(),os.readlink(path))==before


def test_unknown_durable_state_preimage_cannot_be_discarded(incident):
    daemon,_,_,_,claim_path,_,request=incident
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import read_task_state_payload,save_portal_task_state
    body=read_task_state_payload(daemon.state_path)
    body['independent_future_authority']={'held':True}
    save_portal_task_state(daemon.state_path.stem,body,state_path=daemon.state_path)
    claim=claim_path.read_bytes();strategy=daemon.strategy_path.read_bytes()
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and result['reason']=='state_preimage_changed'
    assert claim_path.read_bytes()==claim and daemon.strategy_path.read_bytes()==strategy


def test_unrelated_dead_terminal_owner_is_not_stopped_launch_proof(incident):
    daemon,_,task,_,claim_path,_,request=incident
    path=Path(request['pins']['lifecycle']['path']);life=json.loads(path.read_text())
    life['owner']['parent_pid']=2**30-101
    write(path,life)
    request=daemon.review_quiesced_ordinary_incident(task_id=task.task_id,incident_cid=request['incident_cid'])
    claim=claim_path.read_bytes()
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and result['reason']=='owner_not_in_stopped_launch',result
    assert claim_path.read_bytes()==claim


@pytest.fixture
def settled_next_attempt(incident):
    daemon, supervisor, task, old_workspace, claim_path, _, request = incident
    first=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert first['reconciled'], first.get('reason')
    old_lifecycle=Path(request['pins']['lifecycle']['path'])
    original=old_lifecycle.read_bytes(),supervisor._autonomous_unstall_state_path().read_bytes()
    birth_code='import json; from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth; print(json.dumps(current_process_birth().to_dict()))'
    birth=native.ProcessBirthIdentity.from_dict(json.loads(subprocess.run(
        [sys.executable,'-c',birth_code],capture_output=True,text=True,check=True,timeout=20).stdout))
    assert native.owner_liveness(birth) is native.OwnerLiveness.DEAD
    workspace=daemon.worktree_root/'native-second-attempt'
    life=daemon.worktree_lifecycle.begin_preparing(task_id=task.task_id,canonical_task_cid=request['task_cid'],
        attempt=2,lane_id='synthetic-second-attempt',workspace_path=workspace,branch='lane-second',
        merge_target=daemon.resolved_merge_target_branch,state_dir=str(daemon.state_path.parent),owner=birth)
    subprocess.run(['git','worktree','add','-b','lane-second',str(workspace),'HEAD'],
        cwd=daemon.repo_root,capture_output=True,check=True,timeout=20)
    daemon.worktree_lifecycle.mark_active(workspace,lease_id=life.lease_id,expected_fence=life.fence)
    state=native.PortalTaskState.load(daemon.state_path)
    state.implementation_attempts[task.task_id]=2;state.implementation_attempts_by_cid[request['task_cid']]=2
    state.active_task_id=task.task_id;state.active_task_key=request['task_key'];state.active_task_cid=request['task_cid']
    state.active_task_started_at=native.utc_now();state.active_attempt=2;state.implementation_in_progress=True
    state.active_worktree_path=str(workspace);state.active_branch='lane-second';state.active_phase='implementing'
    state.save(daemon.state_path)
    claim=daemon._build_implementation_task_claim_metadata(task,2,state.active_task_started_at)
    claim.update(pid=birth.pid,owner_process_birth=birth.to_dict())
    claim['dispatch_authority_cid']=daemon._implementation_dispatch_authority_cid(claim)
    write(claim_path,claim)
    assert (old_lifecycle.read_bytes(),supervisor._autonomous_unstall_state_path().read_bytes())==original
    return daemon,supervisor,task,claim_path,request,first,state


def test_settled_old_quarantine_does_not_block_new_native_stale_recovery_snapshot(settled_next_attempt):
    daemon,supervisor,_,claim_path,_,first,state=settled_next_attempt
    before=asdict(native.PortalTaskState.load(daemon.state_path)),claim_path.read_bytes(),daemon.strategy_path.read_bytes(),Path(first['receipt_path']).read_bytes()
    snapshot,reason=supervisor._stale_recovery_snapshot(state)
    assert snapshot is not None,reason
    assert snapshot['claim']['attempt']==2
    assert (asdict(native.PortalTaskState.load(daemon.state_path)),claim_path.read_bytes(),daemon.strategy_path.read_bytes(),Path(first['receipt_path']).read_bytes())==before


@pytest.mark.parametrize('hold',['manual_strategy','new_quarantine','missing_receipt','corrupt_receipt','native_block','source_update'])
def test_settled_old_quarantine_never_suppresses_new_or_independent_stale_recovery_hold(settled_next_attempt,monkeypatch,hold):
    daemon,supervisor,task,claim_path,_,first,state=settled_next_attempt
    if hold in {'manual_strategy','new_quarantine'}:
        value=json.loads(daemon.strategy_path.read_text())
        if hold=='manual_strategy':value['blocked_tasks'].append(task.task_id)
        else:value['autonomous_unstall_quarantines'].append({**value['autonomous_unstall_quarantines'][0],'incident_cid':'new-unsettled-incident'})
        write(daemon.strategy_path,value)
    elif hold=='missing_receipt':Path(first['receipt_path']).unlink()
    elif hold=='corrupt_receipt':Path(first['receipt_path']).write_text('{incomplete')
    elif hold=='native_block':
        state.blocked_task_ids=[task.task_id];state.task_statuses[task.task_id]='blocked';state.save(daemon.state_path)
    elif hold=='source_update':monkeypatch.setattr(adjudication,'runtime_revision',lambda:'sha256:'+'0'*64)
    before=asdict(native.PortalTaskState.load(daemon.state_path)),claim_path.read_bytes(),daemon.strategy_path.read_bytes()
    snapshot,reason=supervisor._stale_recovery_snapshot(state)
    assert snapshot is None and reason in {'strategy_hold_preserved','task_hold_preserved'},(hold,reason)
    assert (asdict(native.PortalTaskState.load(daemon.state_path)),claim_path.read_bytes(),daemon.strategy_path.read_bytes())==before
