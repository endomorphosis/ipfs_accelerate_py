"""Actual Native reconciliation failure events, claims and accounting.

All repositories are authored TinyGit fixtures. No guard/claim/validation
method is substituted, no provider is enabled and no board completion occurs.
"""
from pathlib import Path
import json

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon, PortalTaskState,
)
from test.api.test_agent_supervisor_reconciliation_json_safe import _seed_candidate, _git


@pytest.mark.parametrize('failure', ['out_of_scope', 'declared_command'])
def test_actual_reconciliation_refuses_and_persists_failure_without_paid_attempt(tmp_path, failure):
    fixture = _seed_candidate(tmp_path)
    repo = Path(fixture['repo']); candidate = Path(fixture['worktree_path'])
    if failure == 'out_of_scope':
        (candidate/'outside.txt').write_text('authored out-of-scope candidate\n')
        _git(candidate,'add','outside.txt');_git(candidate,'commit','-m','Outside declared scope')
        fixture['candidate'] = _git(candidate,'rev-parse','HEAD')
    else:
        # This is the actual canonical Native Python command runner, which
        # fails rather than manufacturing a passing or skipped behavioral run.
        todo = Path(fixture['todo_path'])
        text = todo.read_text().replace('python -m py_compile feature.py', 'python -m py_compile absent.py')
        todo.write_text(text)
        _git(repo,'add','todo.md');_git(repo,'commit','-m','Authored failing declared command')
    state_dir = tmp_path/'native-state'
    daemon = PortalImplementationDaemon(todo_path=Path(fixture['todo_path']),
        state_path=state_dir/'task_state.json', strategy_path=state_dir/'strategy.json',
        events_path=state_dir/'events.jsonl', repo_root=repo, task_header_prefix='## ACCEL-',
        worktree_root=tmp_path/'worktrees', merge_target_branch='main', worktree_submodule_paths=[],
        implement=False)
    task = daemon._load_tasks()[0]
    identity = daemon._identity_for_task(task)
    before = PortalTaskState.load(daemon.state_path)
    paid = (dict(before.implementation_attempts), dict(before.implementation_attempts_by_cid))
    board = Path(fixture['todo_path']).read_bytes()
    result = daemon.reconcile_validated_worktree_candidate(worktree_path=candidate,
        branch_name=str(fixture['branch_name']),task=task,baseline_ref=str(fixture['baseline']),
        candidate_commit=str(fixture['candidate']),recovery_key='authored-json-failure-'+failure)
    assert result['returncode'] != 0 and result['validation_result']['passed'] is False
    assert result['provider_dispatched'] is False and result['attempt_consumed'] is False
    assert not result['merge_result'].get('merged')
    json.dumps(result,allow_nan=False)
    after = PortalTaskState.load(daemon.state_path)
    assert (after.implementation_attempts,after.implementation_attempts_by_cid)==paid
    assert not after.implementation_in_progress
    assert not daemon._implementation_task_claim_path(task.task_id,
        canonical_task_cid=identity.canonical_task_cid).exists()
    assert not daemon._implementation_lock_path().exists()
    assert Path(fixture['todo_path']).read_bytes()==board
    rows=[json.loads(line) for line in daemon.events_path.read_text().splitlines()]
    terminal=next(row for row in reversed(rows) if row.get('type')=='worktree_reconciliation_validation_finished')
    assert terminal['validation_result']==json.loads(json.dumps(result['validation_result'],allow_nan=False))
    assert 'proposal_validation' not in terminal['validation_result']
    assert terminal['provider_dispatched'] is False and terminal['attempt_consumed'] is False
    assert not any(row.get('type')=='implementation_finished' for row in rows)
    if failure=='out_of_scope':
        assert terminal['validation_result']['proposal_gate']['accepted'] is False


def _typed_result(tmp_path):
    import hashlib
    from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import collect_git_candidate_diff
    from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
        ImplementationProposal, ProposalValidationPolicy, validate_implementation_proposal)
    fixture=_seed_candidate(tmp_path)
    repo=Path(fixture['repo']);root=Path(fixture['worktree_path'])
    common=_git(repo,'rev-parse','--path-format=absolute','--git-common-dir')
    proposal=ImplementationProposal(task_id='ACCEL-010J',accepted_plan_id='plan:authored-json',
        repository_id='repository:sha256:'+hashlib.sha256(common.encode()).hexdigest(),
        repository_tree_id=str(fixture['baseline']),baseline_id=str(fixture['baseline']),
        objective_id='objective:authored-json',context_id='context:authored-json',replay_nonce='authored-json',
        declared_paths=('feature.py',),candidate_diff=tuple(collect_git_candidate_diff(root,base_revision=str(fixture['baseline']))))
    result=validate_implementation_proposal(proposal,policy=ProposalValidationPolicy(allowed_paths=('feature.py',)))
    assert result.accepted
    return result


def test_projection_is_nonmutating_and_preserves_stronger_failed_gate(tmp_path):
    typed=_typed_result(tmp_path)
    daemon=object.__new__(PortalImplementationDaemon)  # Pure projection; no state/claim API is called.
    raw={'passed':False,'returncode':78,'proposal_validation':typed,
        'proposal_gate':{'accepted':False,'reason':'authored_stronger_denial'}}
    projected=daemon._detach_in_process_proposal_validation(raw)
    assert raw['proposal_validation'] is typed
    assert projected['passed'] is False and projected['returncode']==78
    assert projected['proposal_gate']==raw['proposal_gate']
    assert 'proposal_validation' not in projected
    assert 'VALUE = 1' not in json.dumps(projected)
    from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
    events=tmp_path/'authored-events.jsonl'
    append_jsonl_event(events,'worktree_reconciliation_validation_finished',{'validation_result':projected})
    row=json.loads(events.read_text().splitlines()[-1])
    assert row['validation_result']==projected


def test_unknown_unrelated_object_remains_json_refused(tmp_path):
    typed=_typed_result(tmp_path)
    daemon=object.__new__(PortalImplementationDaemon)
    raw={'passed':False,'proposal_validation':typed,'unrelated':object()}
    projected=daemon._detach_in_process_proposal_validation(raw)
    assert raw['proposal_validation'] is typed and projected['unrelated'] is raw['unrelated']
    from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
    events=tmp_path/'unknown-events.jsonl'
    with pytest.raises(TypeError,match='not JSON serializable'):
        append_jsonl_event(events,'worktree_reconciliation_validation_finished',{'validation_result':projected})
    assert not events.exists() or not events.read_bytes()
