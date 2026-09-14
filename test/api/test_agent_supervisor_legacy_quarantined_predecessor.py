"""Real event chains, DuckDB quarantine claims, Git refs and native queue locks."""
from __future__ import annotations

import copy
import fcntl
import json
import subprocess
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_legacy_verification_retry import history, inspect_existing
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import checkout_repository_id
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError, verify_database_portal_attempt_projection_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_quarantined_predecessor import (
    inspect_reconciled_legacy_verification_retry, hold_reconciled_legacy_retry_queue,
    hold_legacy_verification_retry_observation, inspect_legacy_retry_profile,
)


@pytest.fixture
def prior(history, tmp_path):
    repo = tmp_path / 'repo'; repo.mkdir()
    def git(*argv):
        return subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
            *argv], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    git('init','--quiet','--initial-branch=main')
    (repo/'candidate.py').write_text('value = 1\n')
    git('add','candidate.py');git('commit','--quiet','-m','Preserved candidate')
    commit, tree = git('rev-parse','HEAD'), git('rev-parse','HEAD^{tree}')
    branch = 'implementation/test-041-attempt-1'
    git('branch',branch)
    identity = verify_database_portal_attempt_projection_identity(history.paths.task_projection)
    cid = identity['portal_canonical_task_cid']
    target = checkout_repository_id(repo)
    q = MergeQueue(tmp_path/'queue',target_repository_id=target,target_branch='main')
    metadata = {'completion_task_cids':{'TEST-041':cid},'events_path':str(history.paths.events),
                'candidate_tree':tree,'failure_metadata':[]}
    queued = q.enqueue(branch_name=branch,task_id='TEST-041',canonical_task_cid=cid,
        canonical_task_key=identity['portal_canonical_task_key'],commit_sha=commit,metadata=metadata)
    merge = {'attempted':True,'merged':False,'merge_commit':'','returncode':2,
        'branch':branch,'target_branch':'main','main_worktree_path':str(repo),
        'started_at':'2026-01-01T00:00:00Z','finished_at':'2026-01-01T00:00:01Z',
        'dirty_paths':['external/runtime'],'reason':'main_checkout_dirty_conflict',
        'llm_merge_resolver':{'applied':False,'command':[],'llm_timeout':False,'llm_returncode':2,
                              'reason':'main_checkout_dirty_conflict'}}
    for key in ('submodule_merge_results','generated_submodule_reconciliation','identical_untracked_paths',
                'resolved_generated_conflicts','restored_generated_dirty_overlap','restored_incidental_gitlinks'):
        merge[key] = []
    failure = {'request_id':queued.request_id,'commit_sha':commit,'canonical_task_id':cid,
        'failure_count':1,'status':'quarantined','reason':'changed_submodule_merge_unverified',
        'accepted':False,'integrated':False,'acceptance_pending':False,'merged':False,
        'merge_result':copy.deepcopy(merge)}
    claim = q.claim_pending_request(queued,consumer_id='old-consumer')
    q.quarantine(claim,reason=failure['reason'],metadata=failure)
    common = {'task_id':'TEST-041','canonical_task_cid':cid,
              'canonical_task_key':identity['portal_canonical_task_key'],'attempt':1,
              'branch':branch,'implementation_commit':commit,'baseline_ref':'a'*40}
    queue_outcome = {'request_id':queued.request_id,'queued':True,'merged':False,
                     'reason':'merge_queued','target_branch':'main'}
    board = {'complete':False,'pending_merge':True}
    events = [
        ('implementation_started',{**common,'worktree_path':str(repo)}),
        ('merge_candidate_enqueued',{**common,**queue_outcome,'attempted':False,
            'target_repository_id':target,'queue_dir':str(q.queue_dir)}),
        ('implementation_pending_merge',{**common,'merge_result':queue_outcome,'board_completion':board}),
        ('implementation_finished',{**common,'worktree_path':str(repo),'merge_result':queue_outcome,
            'board_completion':board,'returncode':0,'provider_dispatched':True,'attempt_consumed':True}),
        ('merge_finished',merge),
        ('merge_reconciled',{**common,'request_id':queued.request_id,'request_status':'quarantined',
            'resolved':True,'reason':'stale_quarantined_merge','failure_reason':failure['reason'],
            'merge_result':{'request_id':queued.request_id,'attempted':False,'merged':False,
                            'reason':'stale_quarantined_merge'}}),
    ]
    for _, body in history.events:
        if 'attempt' in body: body['attempt'] = 2
    history.events = events + history.events
    return SimpleNamespace(history=history,repo=repo,git=git,queue=q,target=target,request=queued,
                           metadata=metadata,commit=commit,tree=tree,branch=branch)


def inspect(prior):
    h = prior.history
    for kind, body in h.events: append_jsonl_event(h.paths.events,kind,body)
    return inspect_reconciled_legacy_verification_retry(h.row,task_projection=h.paths.task_projection,
        allowed_attempt_root=h.attempt_root,retained_worktree_root=h.worktrees,expected_task_revision=4)


def guard(prior, evidence):
    return hold_reconciled_legacy_retry_queue(queue_dir=prior.queue.queue_dir,
        target_repository_id=prior.target,target_branch='main',repository_root=prior.repo,evidence=evidence)


def test_exact_native_quarantine_is_preserved_and_does_not_grant_retry(prior):
    evidence = inspect(prior)
    original = prior.queue.database_path.read_bytes()
    events = prior.history.paths.events.read_bytes()
    with (prior.queue.queue_dir/'.merge_queue.duckdb.lock').open('rb') as contender:
        with guard(prior,evidence) as receipt:
            assert receipt['matching_queue_rows'] == 1
            assert receipt['quarantined_predecessor_unchanged'] is True
            assert receipt['retry_authorized'] is receipt['completion_authority'] is False
            with pytest.raises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert evidence['fresh_attempt_number'] == 2 and evidence['attempt_refunded'] is False
    assert prior.queue.database_path.read_bytes() == original
    assert prior.history.paths.events.read_bytes() == events
    assert prior.history.workspace.joinpath('result.py').read_text() == '# unverified work must survive\n'
    with pytest.raises(DatabasePortalBridgeError):inspect_existing(prior.history)


def test_operator_composition_chooses_verified_predecessor_and_retains_queue_guard(prior):
    expected = inspect(prior)
    h = prior.history
    arguments = dict(task_projection=h.paths.task_projection,allowed_attempt_root=h.attempt_root,
                     retained_worktree_root=h.worktrees,expected_task_revision=4)
    assert inspect_legacy_retry_profile(h.row,**arguments) == expected
    with hold_legacy_verification_retry_observation(h.row,**arguments,queue_dir=prior.queue.queue_dir,
        target_repository_id=prior.target,target_branch='main',repository_root=prior.repo) as (evidence,receipt):
        assert evidence == expected and receipt['guard_retained'] is True
        assert receipt['retry_authorized'] is False


def test_profile_selection_preserves_original_single_lifecycle_evidence(history):
    from test.api.test_agent_supervisor_legacy_verification_retry import inspect as single
    expected = single(history)
    assert inspect_legacy_retry_profile(history.row,task_projection=history.paths.task_projection,
        allowed_attempt_root=history.attempt_root,retained_worktree_root=history.worktrees,
        expected_task_revision=4) == expected


@pytest.mark.parametrize('change',['extra_provider','extra_candidate','unknown_finish','missing_reconciliation',
    'applied_merge','partial_submodule','resolver_timeout','callback','changed_commit','changed_task'])
def test_unsettled_event_lifecycles_refuse_evidence(prior,change):
    events=prior.history.events
    if change=='extra_provider':events.append(copy.deepcopy(events[0]))
    elif change=='extra_candidate':events.insert(2,copy.deepcopy(events[1]))
    elif change=='unknown_finish':events[3][1]['provider_dispatched']=None
    elif change=='missing_reconciliation':events.pop(5)
    elif change=='applied_merge':events[4][1]['merged']=True
    elif change=='partial_submodule':events[4][1]['submodule_merge_results']=[{'merged':True}]
    elif change=='resolver_timeout':events[4][1]['llm_merge_resolver']['llm_timeout']=True
    elif change=='callback':events.append(('post_merge_completion',{}))
    elif change=='changed_commit':events[3][1]['implementation_commit']='b'*40
    else:events[1][1]['canonical_task_cid']='task:foreign'
    with pytest.raises(DatabasePortalBridgeError):inspect(prior)


@pytest.mark.parametrize('change',['pending','processing','completed','missing_row','extra_request',
    'accepted','merge_mismatch','missing_failures','wrong_tree','moved_branch','changed_events','changed_projection'])
def test_queue_or_artifact_drift_never_yields_a_recovery_receipt(prior,change):
    evidence=inspect(prior)
    q=prior.queue
    if change=='extra_request':
        q.enqueue(branch_name='implementation/extra',task_id='TEST-041',commit_sha='b'*40,
            canonical_task_cid=prior.request.canonical_task_id,metadata=prior.metadata)
    elif change=='moved_branch':
        (prior.repo/'candidate.py').write_text('value = 2\n')
        prior.git('add','candidate.py');prior.git('commit','--quiet','-m','Other candidate')
        prior.git('branch','--force',prior.branch,'HEAD')
    elif change=='changed_events':append_jsonl_event(prior.history.paths.events,'daemon_pass',{})
    elif change=='changed_projection':
        p=prior.history.paths.task_projection;p.write_text(p.read_text()+'\nchanged contract\n')
    else:
        with q._connect() as c:
            if change in ('pending','processing','completed'):
                c.execute('UPDATE merge_requests SET status=?',[change])
            elif change=='missing_row':c.execute('DELETE FROM merge_requests')
            else:
                raw=c.execute('SELECT metadata_json FROM merge_requests').fetchone()[0]
                m=json.loads(raw)
                if change=='accepted':m['quarantine']['accepted']=True
                elif change=='merge_mismatch':m['quarantine']['merge_result']['finished_at']='changed'
                elif change=='missing_failures':m.pop('failure_metadata')
                else:m['candidate_tree']='c'*40
                c.execute('UPDATE merge_requests SET metadata_json=?',[json.dumps(m)])
    with pytest.raises(DatabasePortalBridgeError):
        with guard(prior,evidence):pytest.fail('unsettled history must not yield')
