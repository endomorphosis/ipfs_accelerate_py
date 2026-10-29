"""A retained proposal must reach the ordinary native database completion CAS."""
from pathlib import Path
import importlib.util
import json

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
