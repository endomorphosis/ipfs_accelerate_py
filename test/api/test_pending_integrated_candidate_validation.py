"""Already-integrated code still needs real validation and a Native callback."""
from dataclasses import asdict
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor, TodoSupervisorConfig, parse_args, supervisor_config_from_args,
)


def git(root, *args):
    result = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True, check=False)
    assert result.returncode == 0, (args, result.stderr)
    return result.stdout.strip()


def world(tmp_path, *, failure='', historical_start=False, completed=False, legacy_cli=False):
    repo=tmp_path/'repo'; repo.mkdir(mode=0o700)
    git(repo,'init','-b','main');git(repo,'config','user.name','Fixture');git(repo,'config','user.email','fixture@example.invalid')
    board=repo/'todo.md'
    board.write_text('# Tasks\n\n## EX-015 Validate the retained feature\n\n- Status: todo\n'
        '- Completion: validation\n- Priority: P0\n- Outputs: feature.py, helper.py\n'
        '- Validation: python -m pytest -q -p no:cacheprovider --confcutdir=. -c pytest.ini test_feature.py\n'
        '- Acceptance: The real feature returns the validated value.\n')
    if completed:
        board.write_text(board.read_text().replace('- Status: todo', '- Status: completed'))
    if failure=='ambiguous':
        board.write_text(board.read_text()+'\n## EX-016 A different task\n\n- Status: todo\n- Outputs: other.py\n')
    if legacy_cli:
        board.write_text(board.read_text().replace('EX-015','CIMP-UX-15').replace('EX-016','CIMP-UX-16'))
    task_id='CIMP-UX-15' if legacy_cli else 'EX-015'
    task_prefix='## CIMP-' if legacy_cli else '## EX-'
    (repo/'feature.py').write_text('VALUE = 1\n');(repo/'helper.py').write_text('HELPER = 0\n')
    (repo/'test_feature.py').write_text('def test_actual_effect():\n from feature import VALUE\n assert VALUE == 2\n')
    (repo/'pytest.ini').write_text('[pytest]\n')
    (repo/'.gitignore').write_text('__pycache__/\n.pytest_cache/\n')
    git(repo,'add','.');git(repo,'commit','-m','initial pending contract')
    baseline=git(repo,'rev-parse','HEAD')
    workroot=tmp_path/'worktrees';workroot.mkdir(mode=0o700)
    workspace=workroot/'candidate'
    branch='implementation/cimp-ux-15-operator-retained' if legacy_cli else 'implementation/ex-015-operator-retained'
    if failure=='ambiguous':branch='implementation/ex-015-and-ex-016-retained'
    git(repo,'worktree','add','-b',branch,str(workspace),baseline)
    state_dir=tmp_path/'state'
    # These are the exact source-selection flags emitted by the complaint Host.
    # An explicit legacy descriptor is a real program configuration object; it
    # does not select database-authoritative task discovery.
    program=None;board_namespace=''
    if legacy_cli:
        args=parse_args(['--todo-path',str(board),'--state-dir',str(state_dir),
            '--state-prefix','owned','--worktree-root',str(workroot),
            '--task-source-kind','legacy-markdown','--authority-mode','legacy_markdown',
            '--explicit-legacy-task-source','--task-prefix',task_prefix,
            '--board-namespace','complaint-gui-ux-overhaul'])
        parsed_config=supervisor_config_from_args(args,repo_root=repo)
        program=parsed_config.database_program;board_namespace=parsed_config.board_namespace
        assert program is not None and program.authority_mode=='legacy_markdown'
        assert program.task_source_kind=='legacy-markdown'
    config=TodoSupervisorConfig(todo_path=board,state_path=state_dir/'task-state.json',
        strategy_path=state_dir/'strategy.json',events_path=state_dir/'supervisor-events.jsonl',
        state_dir=state_dir,repo_root=repo,task_prefix=task_prefix,state_prefix='owned',
        database_program=program,board_namespace=board_namespace,
        worktree_root=workroot,worktree_submodule_paths=(),merge_target_branch='main',
        merge_queue_dir=state_dir/'merge-queue',max_task_attempts=3,implementation_timeout=45,
        implementation_protected_paths=() if failure=='cid' else ('todo.md',),
        worktree_reconciliation_enabled=True,worktree_reconciliation_max_merges=1)
    supervisor=TodoImplementationSupervisor(config)
    daemon=supervisor._build_worktree_reconciliation_daemon()
    task=next(task for task in daemon._load_tasks() if task.task_id==task_id)
    identity=daemon._identity_for_task(task)
    state=native.PortalTaskState(implementation_attempts={task.task_id:2},
        implementation_attempts_by_cid={identity.canonical_task_cid:2},
        task_identities={task.task_id:identity.to_dict()})
    state.save(config.state_path)
    if historical_start:
        # Synthetic historical input is emitted by the real event owner. It
        # supplies only baseline provenance; fresh validation/callbacks decide.
        daemon._record_event('implementation_started',{'task_id':task.task_id,
            'canonical_task_cid':identity.canonical_task_cid,
            'canonical_task_key':identity.canonical_task_key,'board_namespace':identity.board_namespace,
            'branch':branch,'baseline_ref':baseline,'worktree_path':str(workspace),
            'provider_dispatched':False,'attempt':2})
    (workspace/'feature.py').write_text('VALUE = '+('3' if failure=='validation' else '2')+'\n')
    if failure=='scope':(workspace/'foreign.py').write_text('FOREIGN = "must be preserved"\n')
    git(workspace,'add','.');git(workspace,'commit','-m','genuine retained implementation')
    candidate=git(workspace,'rev-parse','HEAD')
    if failure=='empty':
        git(workspace,'commit','--allow-empty','-m','empty history cannot prove task work')
        candidate=git(workspace,'rev-parse','HEAD')
    if failure=='merge':
        git(repo,'checkout','-b','second-parent',baseline)
        (repo/'helper.py').write_text('HELPER = 2\n');git(repo,'commit','-am','other declared change')
        git(repo,'checkout','main')
        git(workspace,'merge','--no-ff','second-parent','-m','candidate has two possible parents')
        candidate=git(workspace,'rev-parse','HEAD')
    git(repo,'merge','--no-ff',branch,'-m','integrate actual code without completing the pending task')
    assert native.parse_task_file(board,task_header_prefix=task_prefix)[0].status==('completed' if completed else 'todo')
    return SimpleNamespace(repo=repo,board=board,workspace=workspace,branch=branch,
        baseline=baseline,candidate=candidate,supervisor=supervisor,daemon=daemon,task=task,
        identity=identity,state_before=asdict(state))


def test_host_parsed_legacy_program_validates_pending_integrated_task15(tmp_path):
    c=world(tmp_path,legacy_cli=True);before=counts(c)
    assert c.supervisor.config.database_program is not None
    assert c.supervisor._database_authority_enabled() is False
    result=c.supervisor.reconcile_backlogged_worktrees()
    successful=[item for item in result['processed'] if item.get('merged')]
    assert len(successful)==1,result
    recovery=successful[0]['recovery_result']
    assert recovery['validation_result']['attempted'] and recovery['validation_result']['passed']
    assert recovery['provider_dispatched'] is False and recovery['attempt_consumed'] is False
    assert native.parse_task_file(c.board,'## CIMP-')[0].status=='completed'
    assert counts(c)==before and before[0]=={'CIMP-UX-15':2}


@pytest.mark.parametrize('kind',['completed','unknown'])
def test_host_parsed_legacy_program_keeps_completed_and_unknown_cleanup_authority(tmp_path,monkeypatch,kind):
    c=world(tmp_path,legacy_cli=True,completed=kind=='completed');before=counts(c)
    if kind=='unknown':
        git(c.workspace,'branch','-m','implementation/unknown-retained')
    monkeypatch.setattr(c.supervisor,'_build_worktree_reconciliation_daemon',
        lambda:pytest.fail('completed or unknown alias must not request a fresh callback'))
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert not result['processed'] and any(item.get('reason')=='already_merged_cleanup_pass'
        for item in result['skipped']),result
    cleanup=c.supervisor.cleanup_backlogged_worktrees()
    # Existing descriptor-based cleanup refuses mutation until its own guarded
    # runtime is qualified. This repair grants no extra cleanup authority.
    assert cleanup['removed_count']==0 and c.workspace.exists(),cleanup
    assert counts(c)==before


@pytest.mark.parametrize('failure',['validation','scope'])
def test_host_parsed_legacy_program_retains_rejected_task15_without_paid_retry(tmp_path,failure):
    c=world(tmp_path,legacy_cli=True,failure=failure);before=counts(c)
    code=(c.workspace/'feature.py').read_bytes();index=git(c.workspace,'ls-files','--stage')
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert result['processed'] and not any(item.get('merged') for item in result['processed']),result
    assert native.parse_task_file(c.board,'## CIMP-')[0].status=='todo'
    cleanup=c.supervisor.cleanup_backlogged_worktrees()
    assert c.workspace.exists() and git(c.repo,'rev-parse',c.branch)==c.candidate,cleanup
    assert (c.workspace/'feature.py').read_bytes()==code and git(c.workspace,'ls-files','--stage')==index
    assert counts(c)==before and before[0]=={'CIMP-UX-15':2}


def counts(c):
    state=native.PortalTaskState.load(c.supervisor.config.state_path)
    return state.implementation_attempts,state.implementation_attempts_by_cid


@pytest.mark.parametrize('historical_start',[False,True])
def test_autoscan_validates_integrated_pending_code_without_spending_last_attempt(tmp_path,monkeypatch,historical_start):
    c=world(tmp_path,historical_start=historical_start)
    before=counts(c)
    for key,value in {'GIT_AUTHOR_NAME':'Inherited CI','GIT_AUTHOR_EMAIL':'ci@example.invalid',
        'GIT_COMMITTER_NAME':'Inherited CI','GIT_COMMITTER_EMAIL':'ci@example.invalid'}.items():
        monkeypatch.setenv(key,value)
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert any(item.get('merged') is True for item in result['processed']),result
    assert native.parse_task_file(c.board,task_header_prefix='## EX-')[0].status=='completed'
    assert counts(c)==before and before[0]=={'EX-015':2}
    assert git(c.repo,'log','-1','--format=%ae%n%ce').splitlines()==[
        'implementation-daemon@example.invalid','implementation-daemon@example.invalid']
    processed=next(item for item in result['processed'] if item.get('merged'))
    recovery=processed['recovery_result']
    assert recovery['provider_dispatched'] is False and recovery['attempt_consumed'] is False
    assert recovery['validation_result']['attempted'] and recovery['validation_result']['passed']
    assert processed['merge_result']['already_merged'] is True
    assert git(c.repo,'show',c.candidate+':feature.py')=='VALUE = 2'


@pytest.mark.parametrize('failure',['validation','scope','empty','merge','ambiguous'])
def test_rejected_integrated_candidate_retains_work_and_never_completes_or_spends(tmp_path,failure):
    c=world(tmp_path,failure=failure)
    before=counts(c);index=git(c.workspace,'ls-files','--stage')
    code=(c.workspace/'feature.py').read_bytes()
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert result['processed'] and not any(item.get('merged') for item in result['processed']),result
    assert native.parse_task_file(c.board,task_header_prefix='## EX-')[0].status=='todo'
    assert counts(c)==before
    cleanup=c.supervisor.cleanup_backlogged_worktrees()
    assert c.workspace.is_dir() and git(c.repo,'rev-parse',c.branch)==c.candidate,cleanup
    assert (c.workspace/'feature.py').read_bytes()==code and git(c.workspace,'ls-files','--stage')==index
    assert counts(c)==before and git(c.workspace,'status','--porcelain')==''
    if failure=='scope':assert (c.workspace/'foreign.py').read_text()=='FOREIGN = "must be preserved"\n'


def test_task_cid_change_after_real_validation_retains_candidate(tmp_path,monkeypatch):
    c=world(tmp_path,failure='cid');before=counts(c)
    run=native.PortalImplementationDaemon._run_validation_commands
    def race(self,*args,**kwargs):
        result=run(self,*args,**kwargs)
        assert result['passed'] is True
        c.board.write_text(c.board.read_text().replace('the validated value','a revised acceptance value'))
        return result
    monkeypatch.setattr(native.PortalImplementationDaemon,'_run_validation_commands',race)
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert result['processed'] and not any(item.get('merged') for item in result['processed']),result
    assert counts(c)==before and native.parse_task_file(c.board,task_header_prefix='## EX-')[0].status=='todo'
    cleanup=c.supervisor.cleanup_backlogged_worktrees()
    assert c.workspace.exists() and git(c.repo,'rev-parse',c.branch)==c.candidate,cleanup
    assert counts(c)==before


def test_old_already_merged_scan_cache_cannot_hide_pending_validation(tmp_path):
    c=world(tmp_path)
    cache=c.supervisor._load_worktree_scan_cache()
    c.supervisor._store_worktree_scan_cache_entry(cache,phase='reconciliation',path=c.workspace,
        branch=c.branch,head=c.candidate,target_signature=git(c.repo,'rev-parse','HEAD'),
        classification='skip',payload={'path':str(c.workspace),'branch':c.branch,'head':c.candidate,
            'reason':'already_merged_cleanup_pass'})
    assert c.supervisor._write_worktree_scan_cache(cache)
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert any(item.get('merged') for item in result['processed']),result
    assert native.parse_task_file(c.board,task_header_prefix='## EX-')[0].status=='completed'
    assert counts(c)[0]=={'EX-015':2}


def test_completed_task_keeps_existing_cleanup_without_validation(tmp_path,monkeypatch):
    c=world(tmp_path,completed=True);before=counts(c)
    monkeypatch.setattr(c.supervisor,'_build_worktree_reconciliation_daemon',
        lambda:pytest.fail('completed lane must retain normal cleanup without validation'))
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert not result['processed'] and any(item.get('reason')=='already_merged_cleanup_pass' for item in result['skipped'])
    cleanup=c.supervisor.cleanup_backlogged_worktrees()
    assert not c.workspace.exists() and cleanup['removed_count']==1,cleanup
    assert counts(c)==before


@pytest.mark.parametrize('drift',['none','head','cid','head_during_cleanup','cid_during_cleanup'])
def test_same_pass_cleanup_requires_real_callback_and_fresh_candidate_binding(tmp_path,monkeypatch,drift):
    c=world(tmp_path,failure='cid');before=counts(c)
    # Defer only physical cleanup: proposal, real pytest, commit, protected
    # postconditions, ordinary train and callback still run without mocks.
    monkeypatch.setattr(native.PortalImplementationDaemon,'_cleanup_merged_worktree',
        lambda *args,**kwargs:{'cleaned':False,'reason':'synthetic cleanup deferred'})
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert any(item.get('merged') for item in result['processed']),result
    assert c.workspace.exists() and native.parse_task_file(c.board,task_header_prefix='## EX-')[0].status=='completed'
    if drift=='head':
        (c.workspace/'helper.py').write_text('HELPER = 9\n')
        git(c.workspace,'commit','-am','a different candidate generation')
        git(c.repo,'merge','--no-ff',c.branch,'-m','integrate a different generation')
    elif drift=='cid':
        c.board.write_text(c.board.read_text().replace('the validated value','a different completed contract'))
    elif drift.endswith('_during_cleanup'):
        status=c.supervisor._git_status_short; changed=[]
        def observe(path):
            result=status(path)
            if Path(path)==c.workspace and not changed:
                changed.append(True)
                if drift.startswith('head'):
                    (c.workspace/'helper.py').write_text('HELPER = 10\n')
                    git(c.workspace,'commit','-am','generation changed during cleanup observation')
                    git(c.repo,'merge','--no-ff',c.branch,'-m','integrate the changed generation')
                else:
                    c.board.write_text(c.board.read_text().replace('the validated value','a contract changed during cleanup'))
            return result
        monkeypatch.setattr(c.supervisor,'_git_status_short',observe)
    cleanup=c.supervisor.cleanup_backlogged_worktrees()
    if drift=='none':
        assert not c.workspace.exists() and cleanup['removed_count']==1,cleanup
    else:
        assert c.workspace.exists() and any(item.get('reason')=='pending_integrated_candidate_retained'
            for item in cleanup['skipped']),cleanup
    assert counts(c)==before


@pytest.mark.parametrize('mismatch',['cid','workspace'])
def test_conflicting_exact_branch_history_cannot_fall_back_to_a_parent(tmp_path,mismatch):
    c=world(tmp_path);before=counts(c)
    c.daemon._record_event('implementation_started',{'task_id':c.task.task_id,
        'canonical_task_cid':'foreign-task-cid' if mismatch=='cid' else c.identity.canonical_task_cid,
        'canonical_task_key':c.identity.canonical_task_key,'board_namespace':c.identity.board_namespace,
        'branch':c.branch,'baseline_ref':c.baseline,
        'worktree_path':str(c.workspace/'foreign') if mismatch=='workspace' else str(c.workspace),
        'provider_dispatched':False,'attempt':2})
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert result['processed'] and not any(item.get('merged') for item in result['processed']),result
    assert result['processed'][0]['merge_result']['reason']=='integrated_candidate_provenance_mismatch'
    c.supervisor.cleanup_backlogged_worktrees()
    assert c.workspace.exists() and counts(c)==before


@pytest.mark.parametrize('unavailable',['missing','unreadable'])
def test_recognized_pending_candidate_is_retained_without_current_task_source(tmp_path,unavailable):
    c=world(tmp_path,failure='validation');before=counts(c)
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert result['processed'] and not any(item.get('merged') for item in result['processed'])
    if unavailable=='missing':c.board.unlink()
    else:c.board.chmod(0)
    cleanup=c.supervisor.cleanup_backlogged_worktrees()
    assert c.workspace.exists() and any(item.get('reason')=='pending_integrated_candidate_retained'
        for item in cleanup['skipped']),cleanup
    assert counts(c)==before


def test_dirty_integrated_candidate_cannot_rescue_an_unchecked_foreign_head(tmp_path,monkeypatch):
    c=world(tmp_path);before=counts(c)
    foreign=c.workspace/'foreign-data.txt';foreign.write_bytes(b'preserve unreviewed foreign output\n')
    index=git(c.workspace,'ls-files','--stage')
    monkeypatch.setattr(c.supervisor,'_rescue_dirty_worktree',
        lambda *args,**kwargs:pytest.fail('immutable integrated proposal must not create a rescue HEAD'))
    monkeypatch.setattr(native.PortalImplementationDaemon,'_run_validation_commands',
        lambda *args,**kwargs:pytest.fail('unchecked dirty integrated candidate reached validation'))
    result=c.supervisor.reconcile_backlogged_worktrees()
    assert result['processed'][0]['merge_result']['reason']=='integrated_candidate_workspace_dirty',result
    c.supervisor.cleanup_backlogged_worktrees()
    assert foreign.read_bytes()==b'preserve unreviewed foreign output\n'
    assert git(c.workspace,'rev-parse','HEAD')==c.candidate and git(c.workspace,'ls-files','--stage')==index
    assert counts(c)==before and native.parse_task_file(c.board,task_header_prefix='## EX-')[0].status=='todo'
