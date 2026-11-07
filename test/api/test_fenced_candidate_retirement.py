"""Actual Native validation failure can be retired without rewriting UNKNOWN."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon, PortalImplementationDaemon, parse_task_text
from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import bind_task_state_control_plane
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge, _accepted_source_events
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import CampaignConfig, atomic_json
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_continuation import build_candidate_request
from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_migration as migration
from test.api.test_native_provider_failure_settlement import _repository


def test_native_fenced_retirement_preserves_unknown_and_paid_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_HOME', str(tmp_path / 'agent-home'))
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR',str(tmp_path / 'orchestration'))
    repo = tmp_path / 'repository'
    branch = _repository(repo)
    (repo / 'test_result.py').write_text('import result\ndef test_actual_result():\n    assert result.VALUE == 2\n')
    frame_relative = 'ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_continuation.py'
    frame_source = repo / frame_relative
    frame_source.parent.mkdir(parents=True)
    frame_source.write_text("def refuse_candidate(**kwargs):\n    raise ValueError('authored candidate preparation refusal')\n")
    subprocess.run(['git','add','test_result.py',frame_relative],cwd=repo,check=True)
    subprocess.run(['git','commit','-qm','Declare semantic fixture'],cwd=repo,check=True)
    state = tmp_path / 'campaign'; state.mkdir()
    (state / 'native').mkdir()
    plan = tmp_path / 'plan.json'
    atomic_json(plan, {'tasks':[{'id':'WWL-002','repository':'site'}]})
    config = CampaignConfig(site_root=repo,framework_root=repo,state_root=state,plan_path=plan,
                            config_path=tmp_path / 'host.json',run_id='fixture-fenced')
    owner = DatabaseImplementationDaemon(database_path=config.database,coordination_path=state / 'coordination.duckdb',
        execution_path=state / 'execution.duckdb',state_dir=state / 'native',state_prefix='wwl',
        owner_session_id=config.run_id,authority_mode='embedded_exclusive',require_real_execution=True,
        max_task_attempts=3,lease_ms=7200000).open()
    owner.materialize_population({'repository_tree_id':'fixture-tree',
        'objectives':[{'objective_id':'fixture','goal_cid':'fixture-goal','title':'Fixture'}],
        'tasks':[{'task_cid':'sha256:' + hashlib.sha256(b'fenced-candidate').hexdigest(),'task_id':'WWL-002',
                  'goal_cid':'fixture-goal','status':'ready','ordinal':1,'title':'Fenced candidate fixture',
                  'outputs':[{'path':'result.py'}],"validation":["python3 -m pytest -q test_result.py"],
                  'completion':'auto','acceptance':'Reject invalid actual Python source.'}]})
    portals = []
    execution_pool = [state / 'worktrees']
    def factory(paths, alias):
        bind_task_state_control_plane(paths.state,str(paths.root / 'portal-state.duckdb'))
        portal = PortalImplementationDaemon(todo_path=paths.task_projection,state_path=paths.state,
            strategy_path=paths.strategy,events_path=paths.events,repo_root=repo,task_header_prefix='## WWL-',
            board_namespace=config.run_id,implement=True,use_ephemeral_worktree=True,
            implementation_command=shlex.join([sys.executable,'-c',"from pathlib import Path; Path('result.py').write_text('VALUE = 1\\n')"]),
            implementation_timeout=300,worktree_root=execution_pool[0],worktree_submodule_paths=[],
            implementation_log_dir=paths.implementation_logs,merge_target_branch=branch,
            max_task_attempts=3,implementation_protected_paths=(),implementation_declared_outputs_only=True)
        portals.append(portal)
        return portal
    bridge = DatabasePortalExecutionBridge(task_source=owner.task_source,attempt_root=state / 'native/wwl_database_portal_attempts',
        portal_factory=factory,repo_root=repo,board_namespace=config.run_id,merge_target_branch=branch,max_task_attempts=3)
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import _RepositoryRoutedPortalBridge
    routed = _RepositoryRoutedPortalBridge(lambda alias:repo,lambda repository:bridge,
        attempt_root=bridge.attempt_root,max_task_attempts=3)
    owner.bind_execution_callbacks(provider_fn=routed.run_provider,effect_fn=routed.apply_effect,validation_fn=routed.validate_effect)
    # Simulate the historical lost ephemeral native-exit capability. Actual
    # provider/validation/cleanup/events run; no claimed process closure follows.
    monkeypatch.setattr(bridge,'_native_provider_exit_failure',lambda **kwargs: None)
    fence = {'owner_alive':False,'main_pid':0,'cgroup_empty':True,'unit':'authored-fixture.service'}
    monkeypatch.setattr(migration,'_registered_stopped_fence',lambda config:dict(fence))
    try:
        outcome = owner.run_once()
        assert outcome['implementation_result']['reason'] == 'provider_callback_outcome_unknown'
        attempt = owner.get_attempt(outcome['attempt_id'])
        assert attempt.status == 'running'
        paths = bridge._paths(attempt)
        binding = bridge._strict_binding(paths.binding)
        events,_ = _accepted_source_events(paths.events)
        final = [event for event in events if event.get('type') == 'implementation_finished'][-1]
        assert final['validation_result']['attempted'] is True
        assert final['validation_result']['passed'] is False
        retained = final['failed_preservation_result']['preserved_commit']
        candidate = state / 'worktrees/retained-for-validation'
        subprocess.run(['git','worktree','add','-b','retained-candidate',str(candidate),retained],cwd=repo,check=True,capture_output=True)
        task = parse_task_text(bridge._verify_projection(paths,binding),path=paths.task_projection,task_header_prefix='## WWL-')[0]
        request = build_candidate_request(run_id=config.run_id,task=task,predecessor_binding=binding,
            repository_root=repo,worktree_path=candidate,branch_name='retained-candidate',
            baseline_commit=final['baseline_ref'],candidate_commit=retained,predecessor_attempt_number=attempt.attempt_number)
        request_path = state / 'candidate-request.json'; atomic_json(request_path,request)
        before = owner.provider_invocation_recorded(attempt.attempt_id,idempotency_key='provider:' + attempt.attempt_id)
        probe = owner.retire_fenced_native_validation_candidate(config=config,candidate_request_path=request_path)
        assert probe['applied'] is False and owner.get_attempt(attempt.attempt_id).status == 'running'
        fence['main_pid'] = 123
        def refused(config):
            raise ValueError('registered supervisor must be stopped')
        monkeypatch.setattr(migration,'_registered_stopped_fence',refused)
        with pytest.raises(ValueError,match='must be stopped'):
            owner.retire_fenced_native_validation_candidate(config=config,candidate_request_path=request_path,apply=True)
        fence['main_pid'] = 0
        monkeypatch.setattr(migration,'_registered_stopped_fence',lambda config:dict(fence))
        original_failure = owner._fail_coordination_claim
        def interrupted_failure(*args,**kwargs):
            raise RuntimeError('authored failure after native FAILED CAS')
        monkeypatch.setattr(owner,'_fail_coordination_claim',interrupted_failure)
        with pytest.raises(RuntimeError,match='after native FAILED CAS'):
            owner.retire_fenced_native_validation_candidate(config=config,candidate_request_path=request_path,apply=True)
        assert owner.get_attempt(attempt.attempt_id).status == 'failed'
        monkeypatch.setattr(owner,'_fail_coordination_claim',original_failure)
        result = owner.retire_fenced_native_validation_candidate(config=config,candidate_request_path=request_path,apply=True)
        assert result['applied'] is True and result['provider_outcome'] == 'unknown'
        failed = owner.get_attempt(attempt.attempt_id)
        assert failed.status == 'failed' and failed.attempt_number == 1
        assert owner.task_source.get_task(attempt.task_cid).status == 'blocked'
        assert owner.provider_invocation_recorded(attempt.attempt_id,idempotency_key='provider:' + attempt.attempt_id) == before
        phases = owner.phase_history(attempt.attempt_id)
        original_rearm = owner.coordinator.rearm_failed_task
        def interrupted_rearm(*args,**kwargs):
            raise RuntimeError('authored interruption after canonical retrying CAS')
        monkeypatch.setattr(owner.coordinator,'rearm_failed_task',interrupted_rearm)
        with pytest.raises(RuntimeError,match='after canonical retrying CAS'):
            owner.rearm_fenced_native_validation_candidate(config=config,candidate_request_path=request_path,apply=True)
        assert owner.task_source.get_task(attempt.task_cid).status == 'retrying'
        pointer = json.loads((state / 'control/pending-candidate-continuation.json').read_text())
        assert pointer['request_path'] == str(request_path)
        monkeypatch.setattr(owner.coordinator,'rearm_failed_task',original_rearm)
        rearm = owner.rearm_fenced_native_validation_candidate(config=config,candidate_request_path=request_path,apply=True)
        assert rearm['applied'] is True and rearm['paid_attempt_budget_reset'] is False
        assert owner.phase_history(attempt.attempt_id) == phases
        # A marker alone cannot bypass the unknown-provider barrier.
        assert owner.claim_next() is None
        from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import CampaignRuntime
        execution_pool[0] = state / 'worktrees/routed-execution'
        execution_pool[0].mkdir()
        runtime = CampaignRuntime(config,{})
        runtime._daemon = owner
        runtime._bridge = routed
        bridge.candidate_reconciliation_loader = runtime._load_candidate_admission
        next_attempt = owner.claim_next()
        assert next_attempt is not None
        assert next_attempt.attempt_number == 2 and next_attempt.attempt_id != attempt.attempt_id
        assert owner.provider_invocation_recorded(attempt.attempt_id,idempotency_key='provider:' + attempt.attempt_id) == before
        assert len([event for event in events if event['type'] == 'implementation_started']) == 1
        assert not any(event['type'] == 'task_completed' for event in events)
        # The exact no-model route may fail before creating any execution
        # workspace. This leaves UNKNOWN intent, not permission to run a model.
        from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_continuation as continuation
        actual_reconcile = continuation.reconcile_candidate
        import importlib.util
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location('authored_preparation_fault',frame_source)
        authored = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(authored)
        monkeypatch.setattr(continuation,'reconcile_candidate',authored.refuse_candidate)
        monkeypatch.setattr(bridge,'_native_provider_exit_failure',lambda **kwargs:None)
        runtime._cached = {'active_task_id':next_attempt.task_alias,'stage':'context'}
        with patch.object(runtime,'capture'):
            runtime._run_pass()
        assert runtime._last_failure['reason'] == 'authored candidate preparation refusal'
        actual_failure = runtime._last_failure
        assert any(frame['path'] == frame_relative for frame in actual_failure['source_frames'])
        current_paths = bridge._paths(next_attempt)
        pending_callback = owner.provider_invocation_recorded(next_attempt.attempt_id,
            idempotency_key='provider:' + next_attempt.attempt_id)
        assert pending_callback['callback_state'] == 'started_outcome_unknown'
        assert not current_paths.events.exists()
        bridge.candidate_reconciliation_loader = None
        waiting = owner.run_once()
        assert waiting['implementation_result']['reason'] == 'provider_callback_outcome_unknown'
        assert not current_paths.events.exists()
        monkeypatch.setattr(continuation,'reconcile_candidate',actual_reconcile)
        bridge.candidate_reconciliation_loader = runtime._load_candidate_admission
        # The unchanged preparation defect cannot spend its sole retry. Only
        # a committed repair of a real captured fault frame admits that retry.
        unchanged = owner.run_once()
        assert unchanged['implementation_result']['reason'] == 'provider_callback_outcome_unknown'
        assert not current_paths.events.exists()
        frame_source.write_text('def refuse_candidate(**kwargs):\n    return None\n')
        subprocess.run(['git','add',frame_relative],cwd=repo,check=True)
        subprocess.run(['git','commit','-qm','Repair authored preparation source'],cwd=repo,check=True)
        runtime = CampaignRuntime(config,{})
        runtime._daemon = owner
        runtime._bridge = routed
        runtime._last_failure = actual_failure
        bridge.candidate_reconciliation_loader = runtime._load_candidate_admission
        resumed = owner.run_once()
        current_events,_ = _accepted_source_events(current_paths.events)
        assert any(event['type'] == 'candidate_continuation_workspace_prepared' for event in current_events)
        assert any(event['type'] == 'worktree_reconciliation_validation_finished' for event in current_events)
        assert not any(event['type'] == 'implementation_started' for event in current_events)
        assert owner.get_attempt(next_attempt.attempt_id).attempt_number == 2
        assert owner.get_attempt(attempt.attempt_id).status == 'failed'
        assert owner.phase_history(attempt.attempt_id) == phases
        assert owner.provider_invocation_recorded(attempt.attempt_id,idempotency_key='provider:' + attempt.attempt_id) == before
        admitted = owner._require_connection().execute(
            "SELECT body_json FROM daemon_execution_events WHERE attempt_id=? AND event_type=?",
            [next_attempt.attempt_id,'native_candidate_preparation_resume_admitted']).fetchall()
        assert len(admitted) == 1
        assert json.loads(admitted[0][0])['previous_callback'] == pending_callback
        # A repeated UNKNOWN pass cannot re-enter preparation or validation.
        prior_count = len(current_events)
        owner.run_once()
        assert len(_accepted_source_events(current_paths.events)[0]) == prior_count
    finally:
        for portal in portals:
            portal.close_event_runtime()
        owner.close()
