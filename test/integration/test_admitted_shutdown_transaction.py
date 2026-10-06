"""Native STOP reads prior custody without renewing mutable START admission."""
from __future__ import annotations

import time
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    AuthorizationBindingError, MutationRecoveryAction, MutationTransactionPhase,
    StaleLeaseError, TransactionConflictError,
)
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import LifecycleSagaPhase
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from test.integration.test_admitted_benchmark_runtime import admitted  # noqa: F401


def test_native_stop_survives_expired_start_replay_scope(admitted, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.header_intent_applicability import (
        local_benchmark_applicability_budget,
    )
    runtime, _owner, _prepared = admitted
    started = runtime.start()
    assert started.succeeded
    original_scope = runtime._replay_work_scope
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    # A real captured work budget expires; the independently issued run lease
    # remains live. This test makes no Source384 inference claim.
    with local_benchmark_applicability_budget(deadline_monotonic=time.monotonic() + .1) as scope:
        runtime._replay_work_scope = scope
    time.sleep(.12)
    try:
        with pytest.raises(TimeoutError, match='deadline expired'):
            runtime._startup_validate('control_validation')
        request = runtime._requests[started.request_id]
        assert runtime._start_transaction_for_shutdown(request).phase is MutationTransactionPhase.COMMITTED
        assert runtime.stop().succeeded
        assert not runtime.process.snapshot(runtime.profile).members
        assert all(child.poll() is not None for child in runtime._children)
    finally:
        runtime._replay_work_scope = original_scope


@pytest.mark.parametrize('mutation', ['foreign-request', 'foreign-transaction',
    'revoked-profile', 'stop-grant-removed', 'lease-refused'])
def test_shutdown_transaction_lookup_preserves_bound_authority(admitted, monkeypatch, mutation):
    runtime, _owner, _prepared = admitted
    request = runtime.request(Operation.START)
    store = runtime.service._state_store
    transaction = store.begin_mutation(request, now_ms=0)
    revoked = runtime.profile_dir / profile_authority.REVOKE_MARKER
    if mutation == 'foreign-request':
        request = replace(request, parameters={**request.parameters, 'run_id': 'foreign'})
        expected = TransactionConflictError
    elif mutation == 'foreign-transaction':
        other = replace(transaction, request_id='request:foreign', transaction_id='')
        monkeypatch.setattr(store, 'get_mutation', lambda _request: other)
        expected = TransactionConflictError
    elif mutation == 'revoked-profile':
        revoked.write_text('revoked for shutdown lookup qualification\n')
        revoked.chmod(0o600)
        expected = ValueError
    elif mutation == 'stop-grant-removed':
        issue = runtime.request
        def revoke_issued_permit(operation):
            issued = issue(operation)
            runtime._permits.pop(issued.authorization.decision_id)
            return issued
        monkeypatch.setattr(runtime, 'request', revoke_issued_permit)
        expected = AuthorizationBindingError
    else:
        # Exercise the real service's fail-closed lease-validation boundary;
        # the authored refusal grants no replacement lease or process effect.
        monkeypatch.setattr(runtime.service, '_lease_validator', lambda _request: False)
        expected = StaleLeaseError
    if mutation != 'foreign-transaction':
        monkeypatch.setattr(store, 'get_mutation', lambda _request:
            pytest.fail('transaction read preceded shutdown admission'))
    try:
        with pytest.raises(expected):
            runtime._start_transaction_for_shutdown(request)
        assert runtime._children == []
    finally:
        if mutation == 'revoked-profile':
            revoked.unlink()


def test_terminal_lifecycle_does_not_hide_control_repair_required(admitted, monkeypatch):
    runtime, _owner, _prepared = admitted
    started = runtime.start()
    assert started.succeeded
    request = runtime._requests[started.request_id]
    store = runtime.service._state_store
    committed = store.get_mutation(request)
    checkpoint = runtime.orchestrator.store.latest()[runtime.profile.target_id]
    assert checkpoint.phase is LifecycleSagaPhase.COMMITTED
    uncertain = replace(committed, phase=MutationTransactionPhase.REPAIR_REQUIRED,
        recovery_action=MutationRecoveryAction.REPAIR, failure_code='injected_lost_control_response')
    # Author an ambiguous control result while preserving the real native
    # committed lifecycle. The read must not turn terminal saga into permission
    # to skip the original repair authority checks.
    get_mutation = store.get_mutation
    calls = []
    recover = runtime.service.recover_mutation
    def observe_repair(*args, **kwargs):
        calls.append((args, kwargs))
        return recover(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(store, 'get_mutation', lambda selected:
            uncertain if selected.request_id == request.request_id else get_mutation(selected))
        patch.setattr(runtime.service, 'recover_mutation', observe_repair)
        with pytest.raises(TransactionConflictError, match='exact launched custody'):
            runtime.stop()
    assert len(calls) == 1
    assert calls[0][0] == (request,)
    assert calls[0][1] == {'expected_revision': uncertain.revision, 'action': MutationRecoveryAction.REPAIR}
    assert runtime.process.identity_alive(checkpoint.new_tree.roots[0])
    assert store.get_mutation(request) == committed
    assert runtime.stop().succeeded
