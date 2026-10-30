"""Native generated-block provenance is classified, never manually cleared."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import current_incident_adjudication as adjudication
from test.api.test_current_incident_adjudication import incident, write


@pytest.fixture
def projected(incident):
    daemon, supervisor, task, workspace, claim_path, record, request=incident
    daemon.implement=False
    before=claim_path.read_bytes()
    # This is a real synthetic native scheduler pass, not a hand-set blocked
    # map or fabricated daemon_pass. No provider command can run in this mode.
    result=daemon.run_once()
    state=native.PortalTaskState.load(daemon.state_path)
    assert state.blocked_task_ids==[task.task_id] and state.task_statuses[task.task_id]=='blocked',result
    assert claim_path.read_bytes()==before
    assert not state.implementation_in_progress and not state.active_task_id
    request=daemon.review_quiesced_ordinary_incident(task_id=task.task_id,incident_cid=request['incident_cid'])
    return daemon,supervisor,task,workspace,claim_path,record,request


def test_exact_generated_native_projection_is_preserved_until_normal_next_pass(projected):
    daemon,supervisor,task,_,claim_path,_,request=projected
    before=asdict(native.PortalTaskState.load(daemon.state_path))
    board=daemon.todo_path.read_bytes();incident=supervisor._autonomous_unstall_state_path().read_bytes()
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['reconciled'],result.get('reason')
    assert not claim_path.exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path))==before
    receipt=json.loads(Path(result['receipt_path']).read_text())
    proof=receipt['generated_quarantine_projection']
    assert proof['task_cid']==request['task_cid'] and proof['state_sha256']==request['state_sha256']
    assert proof['blocked_task_ids']==[task.task_id] and proof['task_statuses'][task.task_id]=='blocked'
    assert proof['daemon_pass_event_id'] in {e['event_id'] for e in daemon._iter_events() if e['type']=='daemon_pass'}
    daemon.run_once()
    after=native.PortalTaskState.load(daemon.state_path)
    assert after.task_statuses[task.task_id]=='ready' and task.task_id not in after.blocked_task_ids
    assert after.implementation_attempts==before['implementation_attempts']
    assert after.implementation_attempts_by_cid==before['implementation_attempts_by_cid']
    assert after.active_attempt==0 and not after.implementation_in_progress
    assert after.completed_task_ids==before['completed_task_ids']
    assert daemon.todo_path.read_bytes()==board and supervisor._autonomous_unstall_state_path().read_bytes()==incident
    assert len([e for e in daemon._iter_events() if e['type']=='implementation_started'])==1


@pytest.mark.parametrize('damage',['no_pass','canonical_cid','projection_status','generation','unknown_block',
    'manual_source','protected_source','merge_failure','transient_merge','queue_hold','repair_source','manual_runtime','dependency_source'])
def test_matching_generated_block_does_not_hide_independent_authority(projected,monkeypatch,damage):
    daemon,supervisor,task,_,claim_path,_,request=projected
    if damage in {'no_pass','canonical_cid','projection_status'}:
        events=list(daemon._iter_events())
        last=next(e for e in reversed(events) if e['type']=='daemon_pass')
        daemon.events_path.unlink();daemon.events_path.with_name(daemon.events_path.name+'.manifest.json').unlink(missing_ok=True)
        reserved={'timestamp','type','stream_id','snapshot_id','sequence','event_id','previous_event_id'}
        for event in events:
            if damage=='no_pass' and event['type']=='daemon_pass':continue
            body={k:v for k,v in event.items() if k not in reserved}
            if event['event_id']==last['event_id']:
                if damage=='canonical_cid':body['execution_slice_task_cids_by_id'][task.task_id]='foreign-cid'
                elif damage=='projection_status':body['execution_slice_task_statuses'][task.task_id]='ready'
            daemon._record_event(event['type'],body)
    elif damage in {'generation','unknown_block'}:
        s=native.PortalTaskState.load(daemon.state_path)
        if damage=='generation':s.strategy_generation+=1
        else:s.blocked_task_ids.append('independent-block');s.blocked_count+=1
        s.save(daemon.state_path)
    elif damage=='manual_source':daemon.todo_path.write_text(daemon.todo_path.read_text()+'\n- Manual hold: true\n')
    elif damage=='protected_source':monkeypatch.setattr(native,'task_implementation_protected_path_conflicts',lambda *a,**kw:('protected',))
    elif damage=='merge_failure':monkeypatch.setattr(daemon,'_unresolved_merge_failures_by_task',lambda **kw:{task.task_id:{'failed':True}})
    elif damage=='transient_merge':monkeypatch.setattr(daemon,'_transient_merge_deferrals_by_task',lambda **kw:{task.task_id:{'waiting':True}})
    elif damage=='queue_hold':monkeypatch.setattr(daemon,'_pending_queued_merge_task_ids',lambda *a,**kw:{task.task_id})
    elif damage=='repair_source':monkeypatch.setattr(native,'pending_retry_budget_repair_sources',lambda *a,**kw:{task.task_id})
    elif damage=='manual_runtime':daemon._manual_completion_authority_hard_blocked_task_ids=frozenset({task.task_id})
    elif damage=='dependency_source':monkeypatch.setattr(native,'dependency_satisfied_references',lambda *a,**kw:set());
    if damage=='dependency_source':
        daemon.todo_path.write_text(daemon.todo_path.read_text()+'\n- Depends on: PORTAL-DEPENDENCY\n')
    request=daemon.review_quiesced_ordinary_incident(task_id=task.task_id,incident_cid=request['incident_cid'])
    before={p:p.read_bytes() for p in (claim_path,daemon.todo_path,daemon.strategy_path,supervisor._autonomous_unstall_state_path())}
    state=asdict(native.PortalTaskState.load(daemon.state_path))
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and not result['reconciled'],(damage,result.get('reason'))
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert asdict(native.PortalTaskState.load(daemon.state_path))==state


def test_real_board_retry_repair_remains_a_hold_beneath_generated_quarantine(projected):
    daemon,supervisor,task,_,claim_path,_,request=projected
    with daemon.todo_path.open('a') as out:
        out.write('\n## PORTAL-099 Resolve implementation retry-budget failure for PORTAL-001\n'
            '\n- Status: ready\n- Completion: manual\n- Outputs: src/repair.py\n'
            '- Acceptance: release PORTAL-001 from strategy blocked_tasks\n')
    assert task.task_id in native.pending_retry_budget_repair_sources(daemon._load_tasks())
    request=daemon.review_quiesced_ordinary_incident(task_id=task.task_id,incident_cid=request['incident_cid'])
    before=claim_path.read_bytes(),daemon.strategy_path.read_bytes(),asdict(native.PortalTaskState.load(daemon.state_path))
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and result['reason']=='independent_retry_repair_blocker',result
    assert (claim_path.read_bytes(),daemon.strategy_path.read_bytes(),asdict(native.PortalTaskState.load(daemon.state_path)))==before


def test_real_same_key_merge_custody_is_not_hidden_by_different_display_or_revision(projected):
    daemon,_,task,_,claim_path,_,request=projected
    held=daemon.merge_queue.enqueue(branch_name='synthetic-other-candidate',task_id='PORTAL-FOREIGN',
        canonical_task_id='synthetic-foreign-cid',canonical_task_key=request['task_key'],
        commit_sha=request['baseline_ref'],target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch)
    assert held.canonical_task_key==request['task_key']
    before=claim_path.read_bytes(),daemon.strategy_path.read_bytes(),asdict(native.PortalTaskState.load(daemon.state_path))
    result=daemon.adjudicate_quiesced_ordinary_incident(request=request)
    assert result['blocked'] and result['reason']=='independent_merge_custody',result
    assert (claim_path.read_bytes(),daemon.strategy_path.read_bytes(),asdict(native.PortalTaskState.load(daemon.state_path)))==before
