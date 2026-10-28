"""Explicit clearance of unchanged inputs after unrelated maintenance commits.

Real Git, real Native snapshots/incidents and an inert real live-lock owner;
no provider, task completion, counter repair or campaign state is involved.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from test.api.test_agent_supervisor_implementation_protected_paths import (
    POLICY_PATH, _git, _protected_git_worktree_daemon, _task,
)
SECOND_PATH='implementation_plan/other-plan.json'


@pytest.fixture
def forward_case(tmp_path):
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    second=repo/SECOND_PATH;second.parent.mkdir(parents=True,exist_ok=True);second.write_text('other protected input\n')
    _git(repo,'add',SECOND_PATH)
    _git(repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','second protected input')
    _git(workspace,'merge','--ff-only',_git(repo,'rev-parse','HEAD'))
    daemon.implementation_protected_paths=(POLICY_PATH,SECOND_PATH)
    task = _task(outputs=['src/forensic.py'])
    forensic = workspace/'src/forensic.py'
    forensic.parent.mkdir()
    forensic.write_text('FORENSIC = "unknown provider outcome"\n')
    _git(workspace, 'add', 'src/forensic.py')
    _git(workspace, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-m', 'preserved interrupted candidate')
    candidate = _git(workspace, 'rev-parse', 'HEAD')
    identity = daemon._identity_for_task(task)
    state = native.PortalTaskState(last_implementation_task_id=task.task_id,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_task_key=identity.canonical_task_key,
        implementation_attempts={task.task_id:3},
        implementation_attempts_by_cid={identity.canonical_task_cid:3})
    state.save(daemon.state_path)
    before = daemon._require_implementation_protected_snapshot(task=task, attempt=3, workspace_path=workspace)
    baseline_head = _git(repo, 'rev-parse', 'HEAD')
    replacement = protected.with_name('replacement.json')
    replacement.write_bytes(protected.read_bytes())
    replacement.chmod(protected.stat().st_mode & 0o777)
    replacement.replace(protected)
    (repo/'maintenance.txt').write_text('unrelated maintenance\n')
    _git(repo, 'add', 'maintenance.txt')
    _git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-m', 'unrelated maintenance')
    violation = daemon._implementation_protected_path_violation(
        task=task, attempt=3, workspace_path=workspace, before=before)
    assert violation['reason'] == 'implementation_protected_path_mutated'
    assert {m['scope'] for m in violation['mutations']} == {'shared_checkout'}
    return SimpleNamespace(daemon=daemon,repo=repo,workspace=workspace,protected=protected,
        task=task,before=before,baseline_head=baseline_head,candidate=candidate,forensic=forensic,
        state_before=asdict(native.PortalTaskState.load(daemon.state_path)))


def _preserved(c):
    return (c.forensic.read_bytes(), _git(c.workspace,'rev-parse','HEAD'),
        _git(c.workspace,'ls-files','--stage'),
        asdict(native.PortalTaskState.load(c.daemon.state_path)))


def test_public_operator_clearance_accepts_exact_inputs_after_unrelated_forward_head(forward_case):
    c=forward_case; before=_preserved(c)
    result=c.daemon.clear_implementation_protected_path_incident(
        operator_note='Reviewed the unchanged protected inputs after an atomic plan rewrite.')
    assert result['cleared'] is True,result
    assert result['approved_commits'] == [] and result['shared_checkout_rollback_confirmed'] is True
    receipt=json.loads(Path(result['receipt_path']).read_text())
    proof=receipt['shared_checkout_rollback_proof']
    assert proof['baseline_git_head'] == c.baseline_head
    assert proof['git_head'] == _git(c.repo,'rev-parse','HEAD')
    assert proof['history_kind'] == 'forward_head_without_protected_history'
    assert receipt['incident']['attempt'] == 3
    assert not c.daemon._implementation_protected_active_snapshot_path().exists()
    assert not c.daemon._implementation_protected_incident_path().exists()
    assert _preserved(c) == before and before[-1] == c.state_before
    assert not any(row['type']=='implementation_finished' for row in c.daemon._iter_events())
    again=c.daemon.clear_implementation_protected_path_incident(operator_note='Already reviewed.')
    assert again['cleared'] is False and again['already_clear'] is True
    assert _preserved(c) == before


def test_automatic_crash_fence_keeps_unproved_head_clean_incident_latched(forward_case):
    c=forward_case
    # Existing Native metadata-only auto-clearance is independent of the
    # missing hook. An actual untrusted content change must stay latched.
    c.protected.write_text('unapproved protected content\n')
    c.daemon._implementation_protected_path_violation(
        task=c.task,attempt=3,workspace_path=c.workspace,before=c.before)
    active=c.daemon._implementation_protected_active_snapshot_path()
    incident=c.daemon._implementation_protected_incident_path()
    before=(active.read_bytes(),incident.read_bytes(),_preserved(c))
    result=c.daemon._reconcile_implementation_protected_path_fence()
    assert result['blocked'] is True and result['reason']=='implementation_protected_path_incident_latched',result
    assert (active.read_bytes(),incident.read_bytes(),_preserved(c)) == before


@pytest.mark.parametrize('damage',['content','mode','hardlink','workspace_content','foreign_source',
    'foreign_workspace','unrelated_head','protected_commit','protected_change_revert',
    'protected_branch_change_revert','other_protected_revert','unexpected_approval','live_owner'])
def test_explicit_forward_clearance_refuses_changed_or_unproved_inputs(forward_case,damage):
    c=forward_case; daemon=c.daemon; child=None
    active_path=daemon._implementation_protected_active_snapshot_path()
    active=json.loads(active_path.read_text())
    approvals=[]
    if damage=='content': c.protected.write_text('changed policy\n')
    elif damage=='mode': c.protected.chmod(0o600 if c.protected.stat().st_mode&0o777 != 0o600 else 0o644)
    elif damage=='hardlink': os.link(c.protected,c.repo/'other-policy-link')
    elif damage=='workspace_content': (c.workspace/POLICY_PATH).write_text('workspace mutation\n')
    elif damage=='foreign_source':
        active['snapshot']['shared_checkout']['root']=str(c.repo.parent/'foreign')
        native.write_json_atomic(active_path,active)
    elif damage=='foreign_workspace':
        active['snapshot']['workspace']['root']=str(c.repo)
        native.write_json_atomic(active_path,active)
    elif damage=='unrelated_head':
        _git(c.repo,'checkout','--orphan','unrelated')
        _git(c.repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','unrelated root')
    elif damage in {'protected_commit','protected_change_revert','protected_branch_change_revert','other_protected_revert'}:
        original_branch=_git(c.repo,'branch','--show-current')
        if damage=='protected_branch_change_revert':_git(c.repo,'checkout','-b','changed-side-branch')
        relative=SECOND_PATH if damage=='other_protected_revert' else POLICY_PATH
        changed=c.repo/relative;original=changed.read_bytes()
        changed.write_text('committed protected change\n')
        _git(c.repo,'add',relative)
        _git(c.repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','protected change')
        if damage!='protected_commit':
            changed.write_bytes(original);_git(c.repo,'add',relative)
            _git(c.repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','revert protected change')
        if damage=='protected_branch_change_revert':
            _git(c.repo,'checkout',original_branch)
            _git(c.repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                'merge','--no-ff','changed-side-branch','-m','merge side history')
    elif damage=='unexpected_approval': approvals=[_git(c.repo,'rev-parse','HEAD')]
    elif damage=='live_owner':
        child=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(30)'],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        birth=supervisor_runtime.read_process_birth(child.pid)
        assert birth is not None
        native.write_json_atomic(daemon._implementation_lock_path(),{'kind':'implementation',
            'pid':child.pid,'owner_process_birth':birth.to_dict(),'repo_root':str(c.repo),
            'state_dir':str(daemon.state_path.parent),'lease_id':'real-inert-live-owner'})
    try:
        before=(active_path.read_bytes(),daemon._implementation_protected_incident_path().read_bytes(),_preserved(c))
        if damage not in {'unexpected_approval','live_owner'}:
            assert daemon._implementation_protected_shared_checkout_rollback_proof(
                active=json.loads(active_path.read_text()),
                incident=json.loads(daemon._implementation_protected_incident_path().read_text())) == {}
        result=daemon.clear_implementation_protected_path_incident(approved_commits=approvals,
            operator_note='This damaged scope must remain held.')
        assert result['cleared'] is False,(damage,result)
        assert (active_path.read_bytes(),daemon._implementation_protected_incident_path().read_bytes(),_preserved(c))==before
        if child is not None: assert child.poll() is None
    finally:
        if child is not None: child.terminate();child.wait(timeout=3)


@pytest.mark.parametrize('change',['head','protected_inode','protected_content'])
def test_forward_clearance_rechecks_complete_proof_before_receipt(forward_case,monkeypatch,change):
    c=forward_case;original=c.daemon._implementation_protected_shared_checkout_rollback_proof;calls=[]
    def racing(**kwargs):
        value=original(**kwargs);calls.append(value)
        if len(calls)==1:
            assert value
            if change=='head':
                (c.repo/'another.txt').write_text('concurrent head\n');_git(c.repo,'add','another.txt')
                _git(c.repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','concurrent')
            elif change=='protected_inode':
                other=c.protected.with_name('new-inode.json');other.write_bytes(c.protected.read_bytes())
                other.chmod(c.protected.stat().st_mode&0o777);other.replace(c.protected)
            else:c.protected.write_text('late changed content\n')
        return value
    monkeypatch.setattr(c.daemon,'_implementation_protected_shared_checkout_rollback_proof',racing)
    before=_preserved(c)
    result=c.daemon.clear_implementation_protected_path_incident(operator_note='Require a stable proof.')
    assert result['cleared'] is False,result
    assert c.daemon._implementation_protected_active_snapshot_path().exists()
    assert c.daemon._implementation_protected_incident_path().exists()
    assert _preserved(c)==before
