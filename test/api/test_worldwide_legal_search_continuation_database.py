"""A retained proposal must reach the ordinary native database completion CAS."""
from pathlib import Path
import importlib.util
import json
import hashlib
import subprocess
import sys
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import _RepositoryRoutedPortalBridge

spec=importlib.util.spec_from_file_location('wwl_native_continuation_fixture',Path(__file__).with_name('test_worldwide_legal_search_continuation.py'))
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
native_continuation=fixtures.native_continuation


def test_public_native_database_completes_exact_retained_candidate_without_model(native_continuation):
    portal,attempt,paths,binding,admission,owner=native_continuation
    bridge=DatabasePortalExecutionBridge(task_source=owner.task_source,attempt_root=paths.root.parent,
        portal_factory=lambda *_:portal,repo_root=portal.repo_root,board_namespace='fixture-continuation',merge_target_branch='main',
        prior_attempt_authority=owner.authorize_superseded_portal_attempt_binding,
        candidate_reconciliation_loader=lambda **_kwargs:admission)
    routed=_RepositoryRoutedPortalBridge(lambda _alias:portal.repo_root,lambda _root:bridge,
        attempt_root=paths.root.parent,max_task_attempts=3)
    owner.bind_execution_callbacks(provider_fn=routed.run_provider,effect_fn=routed.apply_effect,validation_fn=routed.validate_effect)
    before=owner.get_attempt(admission.request['predecessor_attempt_id'])
    result=owner.run_once()
    current=owner.get_attempt(attempt.attempt_id)
    task=owner.task_source.get_task(attempt.task_alias)
    assert current.status=='succeeded',json.dumps(result,sort_keys=True)
    assert task.status=='completed'
    assert owner.get_attempt(before.attempt_id).status=='failed'
    assert owner.get_attempt(before.attempt_id).attempt_number==before.attempt_number
    assert current.attempt_number==2
    assert task.body.get('completion_receipt')
    assert 'failed' not in [phase['phase'] for phase in owner.phase_history(current.attempt_id)]
    events=[json.loads(line) for line in paths.events.read_text().splitlines()]
    assert not any(event.get('type')=='implementation_started' for event in events)
    assert any(event.get('type')=='merge_finished' and event.get('merged') is True for event in events)
    assert any(event.get('type')=='task_completed' for event in events)


@pytest.mark.parametrize('interrupted_checkout', [False, True, 'startup_fenced', 'past_execution_boundary'])
def test_unknown_callback_recovers_only_real_native_acceptance_or_a_dead_preparing_candidate(native_continuation, interrupted_checkout):
    portal, attempt, paths, binding, admission, owner = native_continuation
    def interrupted(**_kwargs):
        raise RuntimeError('interrupted native candidate callback')
    bridge = DatabasePortalExecutionBridge(task_source=owner.task_source, attempt_root=paths.root.parent,
        portal_factory=lambda *_: portal, repo_root=portal.repo_root, board_namespace='fixture-continuation',
        merge_target_branch='main', prior_attempt_authority=owner.authorize_superseded_portal_attempt_binding,
        candidate_reconciliation_loader=interrupted)
    routed = _RepositoryRoutedPortalBridge(lambda _alias: portal.repo_root, lambda _root: bridge,
        attempt_root=paths.root.parent, max_task_attempts=3)
    owner.bind_execution_callbacks(provider_fn=routed.run_provider, effect_fn=routed.apply_effect,
        validation_fn=routed.validate_effect)
    with pytest.raises(RuntimeError, match='interrupted native candidate'):
        owner.run_once()
    pending = owner.provider_invocation_recorded(attempt.attempt_id, idempotency_key='provider:' + attempt.attempt_id)
    assert pending['callback_state'] == 'started_outcome_unknown'
    bridge.candidate_reconciliation_loader = None
    waiting = owner.run_once()
    assert waiting['implementation_result']['reason'] == 'provider_callback_outcome_unknown'
    assert owner.provider_invocation_recorded(attempt.attempt_id, idempotency_key='provider:' + attempt.attempt_id) == pending
    if interrupted_checkout:
        fresh_id = hashlib.sha256((attempt.attempt_id + admission.request['receipt_id']).encode()).hexdigest()[:24]
        fresh = portal.worktree_root / ('workspace_continuation_' + fresh_id)
        branch = ('implementation/wwl-001-' + fresh_id + '-attempt-1-' + str(attempt.started_at_ms // 1000))
        task = portal._load_tasks()[0]
        body = dict(repo_root=str(portal.repo_root), task_id=task.task_id,
            canonical_task_cid=portal._identity_for_task(task).canonical_task_cid,
            attempt=portal._worktree_lifecycle_attempt(task, 1), lane_id=portal._worktree_lifecycle_lane_id(),
            workspace_path=str(fresh), branch=branch, merge_target=portal.resolved_merge_target_branch,
            state_dir=str(paths.root.resolve()))
        code = '''
import json, subprocess, sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
body = json.loads(sys.argv[1]); root = Path(body.pop('repo_root'))
WorktreeLifecycleStore(root).begin_preparing(**body, allow_replace_stale=False)
subprocess.run(['git', 'worktree', 'add', '--no-checkout', '-b', body['branch'], body['workspace_path'], sys.argv[2]], cwd=root, check=True, capture_output=True)
'''
        subprocess.run([sys.executable, '-c', code, json.dumps(body), admission.request['candidate_commit']],
            check=True, timeout=30, capture_output=True)
        assert not (fresh / 'feature.py').exists()
        if interrupted_checkout in {'startup_fenced', 'past_execution_boundary'}:
            fenced = portal.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
                fresh, expected_state_dir=paths.root)
            assert fenced is not None and fenced.terminal_reason == 'controlled_restart_dead_owner'
        if interrupted_checkout == 'past_execution_boundary':
            portal._record_event('candidate_continuation_workspace_prepared', {
                'database_attempt_id': attempt.attempt_id, 'worktree_path': str(fresh),
                'candidate_admission_receipt_id': admission.request['receipt_id']})
        bridge.candidate_reconciliation_loader = lambda **_kwargs: admission
    else:
        fixtures.reconcile_candidate(portal_daemon=portal, attempt=attempt, paths=paths,
            binding=binding, admission=admission)
        # Leave the candidate callback unbound: reading accepted work must not
        # require permission to invoke any provider or candidate operation.
    result = owner.run_once()
    if interrupted_checkout == 'past_execution_boundary':
        assert result['implementation_result']['reason'] == 'provider_callback_outcome_unknown'
        assert owner.provider_invocation_recorded(attempt.attempt_id, idempotency_key='provider:' + attempt.attempt_id) == pending
        assert owner.get_attempt(attempt.attempt_id).status == 'running'
        return
    assert owner.get_attempt(attempt.attempt_id).status == 'succeeded', json.dumps(result)
    assert owner.task_source.get_task(attempt.task_alias).status == 'completed'
    assert owner.get_attempt(attempt.attempt_id).attempt_number == 2
    assert owner.get_attempt(admission.request['predecessor_attempt_id']).status == 'failed'
    assert not any(json.loads(line).get('type') == 'implementation_started' for line in paths.events.read_text().splitlines())
    if interrupted_checkout:
        assert fresh.exists() and not (fresh / 'feature.py').exists()  # Retained, never reset or removed.


def test_material_candidate_repair_revalidates_the_same_native_attempt_without_a_model(native_continuation):
    portal, attempt, paths, binding, admission, owner = native_continuation
    candidate = Path(admission.request['worktree_path'])
    def proposal(text):
        (candidate / 'feature.py').write_text(text)
        fixtures.git(candidate, 'add', 'feature.py')
        fixtures.git(candidate, 'commit', '-m', 'Authored candidate source revision')
        body = dict(admission.request)
        body['candidate_commit'] = fixtures.git(candidate, 'rev-parse', 'HEAD')
        body['candidate_tree'] = fixtures.git(candidate, 'rev-parse', 'HEAD^{tree}')
        body['changed_files'] = fixtures.candidate_manifest(candidate, body['baseline_commit'], body['candidate_commit'])
        from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_continuation import identity
        body['receipt_id'] = identity({k: v for k, v in body.items() if k != 'receipt_id'})
        return replace(admission, request=body)
    admitted = [proposal('VALUE =\n')]
    bridge = DatabasePortalExecutionBridge(task_source=owner.task_source, attempt_root=paths.root.parent,
        portal_factory=lambda *_: portal, repo_root=portal.repo_root, board_namespace='fixture-continuation',
        merge_target_branch='main', prior_attempt_authority=owner.authorize_superseded_portal_attempt_binding,
        candidate_reconciliation_loader=lambda **_kwargs: admitted[0])
    routed = _RepositoryRoutedPortalBridge(lambda _alias: portal.repo_root, lambda _root: bridge,
        attempt_root=paths.root.parent, max_task_attempts=3)
    owner.bind_execution_callbacks(provider_fn=routed.run_provider, effect_fn=routed.apply_effect,
        validation_fn=routed.validate_effect)
    first = owner.run_once()
    assert owner.get_attempt(attempt.attempt_id).status == 'running', json.dumps(first)
    before = paths.events.read_bytes()
    owner.run_once()
    assert paths.events.read_bytes() == before  # No unbounded replay of the rejected bytes.
    admitted[0] = proposal('VALUE = 2\n')
    result = owner.run_once()
    assert owner.get_attempt(attempt.attempt_id).status == 'succeeded', json.dumps(result)
    assert owner.task_source.get_task(attempt.task_alias).status == 'completed'
    assert owner.get_attempt(attempt.attempt_id).attempt_number == 2
    assert owner.get_attempt(admission.request['predecessor_attempt_id']).status == 'failed'
    assert not any(json.loads(line).get('type') == 'implementation_started' for line in paths.events.read_text().splitlines())
