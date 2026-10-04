"""Exact native queue/claim/CAS boundaries for the bounded pre-effect seam."""
import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalBridgeDeferred
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon, DatabaseImplementationAuthorityError
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


def deferred(_):
    raise DatabasePortalBridgeDeferred('worktree_lifecycle_claim_exists', backoff_seconds=0)


@pytest.mark.parametrize('boundary', ['queue', 'cas'])
def test_closed_deferral_recovers_lost_response_without_provider_replay(tmp_path, monkeypatch, boundary):
    calls = []
    def provider(attempt):
        calls.append(attempt.attempt_id)
        deferred(attempt)
    daemon = _open_daemon(tmp_path, provider_fn=provider, max_task_attempts=4)
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        source = daemon.task_source
        name = 'record_queue_backoff' if boundary == 'queue' else 'compare_and_set_status'
        original = getattr(source, name)
        def lose_response(*args, **kwargs):
            result = original(*args, **kwargs)
            raise OSError('response lost after durable commit')
        with monkeypatch.context() as patch:
            patch.setattr(source, name, lose_response)
            with pytest.raises(OSError, match='response lost'):
                daemon._resume_attempt_without_process_crash(attempt)
        prior_queue = source.get_queue_entry(attempt.task_cid)
        assert daemon.get_attempt(attempt.attempt_id).status == 'failed'
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == 'accepted'
        successor = daemon.claim_next()
        assert successor is not None and successor.attempt_id != attempt.attempt_id
        assert calls == [attempt.attempt_id]
        assert source.get_queue_entry(attempt.task_cid) == prior_queue
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == 'released'
    finally:
        daemon.close()


@pytest.mark.parametrize('budget', [0, 1])
def test_attempt_safety_cap_is_durable_and_not_provider_token_accounting(tmp_path, budget):
    daemon = _open_daemon(tmp_path, provider_fn=deferred, max_task_attempts=budget)
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        prior = daemon.get_attempt(result['attempt_id'])
        task = daemon.task_source.get(prior.task_cid)
        assert task.status == 'blocked'
        receipt = task.body['completion_receipt']
        assert receipt['budget_kind'] == 'coordination_attempt_safety_cap'
        assert receipt['attempt_consumed'] is False
        assert receipt['provider_dispatched'] is False
        assert daemon.claim_next() is None
    finally:
        daemon.close()


def test_control_cas_race_preserves_claim_custody(tmp_path, monkeypatch):
    daemon = _open_daemon(tmp_path, provider_fn=deferred, max_task_attempts=4)
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        source = daemon.task_source
        original = source.compare_and_set_status
        def race(task_cid, **kwargs):
            original(task_cid, expected_revision=kwargs['expected_revision'], status='blocked', receipt={'operation': 'operator_hold'})
            return original(task_cid, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(source, 'compare_and_set_status', race)
            with pytest.raises(Exception):
                daemon._resume_attempt_without_process_crash(attempt)
        assert source.get(attempt.task_cid).body['completion_receipt'] == {'operation': 'operator_hold'}
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == 'accepted'
        with pytest.raises(DatabaseImplementationAuthorityError, match='current control receipt'):
            daemon.claim_next()
    finally:
        daemon.close()


@pytest.mark.parametrize('value', [True, -1, 10001, '4', None])
def test_daemon_attempt_safety_cap_rejects_malformed_values(tmp_path, value):
    with pytest.raises(DatabaseImplementationAuthorityError, match='max_task_attempts'):
        DatabaseImplementationDaemon(database_path=tmp_path/'unused.duckdb', authority_mode='embedded', max_task_attempts=value, install_schema=False)


def test_bootstrap_credentials_cannot_be_silently_ignored(tmp_path):
    with pytest.raises(DatabaseImplementationAuthorityError, match='bootstrap credentials'):
        DatabaseImplementationDaemon(database_path=tmp_path/'unused.duckdb', authority_mode='embedded', state_owner_bootstrap_credentials=object(), install_schema=False)


def test_native_bridge_unknown_callback_keeps_custody_even_after_lease_expiry(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge, DatabasePortalBridgeError
    now = [10000]
    daemon = _open_daemon(tmp_path, max_task_attempts=4, clock_ms=lambda: now[0], lease_ms=5000)
    calls = []
    def factory(*args):
        calls.append(True)
        raise DatabasePortalBridgeError('unresolved callback failure')
    try:
        daemon.materialize_population(_population(1))
        bridge = DatabasePortalExecutionBridge(task_source=daemon.task_source, attempt_root=tmp_path/'attempts', portal_factory=factory, max_task_attempts=4)
        daemon._provider_fn = bridge.run_provider
        attempt = daemon.claim_next()
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert result['reason'] == 'provider_callback_outcome_unknown'
        prior = dict(daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=f'provider:{attempt.attempt_id}'))
        assert prior['callback_state'] == 'started_outcome_unknown'
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == 'accepted'
        daemon._resume_attempt_without_process_crash(daemon.get_attempt(attempt.attempt_id))
        assert calls == [True]
        now[0] += 6000
        assert daemon.claim_next() is None
        after = dict(daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=f'provider:{attempt.attempt_id}'))
        assert after == prior
    finally:
        daemon.close()
