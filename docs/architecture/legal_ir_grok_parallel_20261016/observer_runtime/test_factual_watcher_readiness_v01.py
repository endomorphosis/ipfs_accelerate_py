"""Native selected/start publications for the factual observer's readiness gate.

All writes use the existing private-state observer's disposable DuckDB fixture.
No campaign store, resource ledger, model, native claim or task status is touched.
Run only inside the original finite native test owner with external pytest temp
storage explicitly charged by the caller.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import portal_task_state_control_plane as state_owner
from test_private_portal_state_observer_v01 import case, _observe


META = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('factual_watcher_readiness_under_test', META/'factual_root_watcher_v04.py')
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


class Refusal(ValueError):
    pass


class Validator:
    TASK = 'LIR-9001'

    @staticmethod
    def need(value, reason):
        if not value:
            raise Refusal(reason)


def _observed(body):
    return {'native_portal_task_cid': body['active_task_cid'],
            'native_portal_task_key': body['active_task_key']}


def _selected():
    # These are the real PortalTaskState defaults retained by native selection.
    state = native.PortalTaskState(
        active_task_id='LIR-9001', active_task_cid='fixture-local-cid',
        active_task_key='fixture-local-key', recommended_task_id='LIR-9001')
    return vars(state).copy()


def test_native_private_selected_state_waits_then_started_publication_is_ready(case, monkeypatch):
    state_path = case['paths'].state
    key = str(state_path.resolve())
    previous = state_owner._BOUND_STATE_TARGETS.get(key)
    state_owner.bind_task_state_control_plane(state_path, str(case['database']))
    monkeypatch.delenv(state_owner.ENV_DUCKLAKE, raising=False)
    identity = SimpleNamespace(canonical_task_cid=case['body']['active_task_cid'],
                               canonical_task_key=case['body']['active_task_key'])
    task = SimpleNamespace(task_id='LIR-9001', title='Private fixture', track='fixture')
    state = native.PortalTaskState(active_task_id=task.task_id,
        active_task_cid=identity.canonical_task_cid, active_task_key=identity.canonical_task_key,
        recommended_task_id=task.task_id)
    daemon = object.__new__(native.PortalImplementationDaemon)
    daemon.state_path = state_path
    daemon._identity_for_task = lambda _task: identity
    try:
        # Publish and read through the genuine native private DuckDB protocol.
        assert state.save(state_path) is True
        before = hashlib.sha256(case['database'].read_bytes()).hexdigest()
        raw, _ = _observe(case)
        selected = json.loads(raw)
        assert selected['active_attempt'] == 0 and selected['active_phase'] == ''
        assert selected['implementation_in_progress'] is False
        assert watcher._native_portal_attempt(Validator, selected, _observed(selected)) is None
        assert hashlib.sha256(case['database'].read_bytes()).hexdigest() == before

        # Execute the real publication method, using its validation-only option
        # so this fixture never creates or claims provider retry authority.
        native.PortalImplementationDaemon._mark_implementation_started(
            daemon, state, task=task, attempt=1, started_at='2026-10-17T00:00:00+00:00',
            log_path=case['root']/'fixture.log', worktree_path=case['root'],
            branch_name='fixture', consume_attempt=False)
        raw, _ = _observe(case)
        started = json.loads(raw)
        assert started['active_phase'] == 'implementing'
        assert started['implementation_in_progress'] is True
        assert watcher._native_portal_attempt(Validator, started, _observed(selected)) == 1
        assert not state_path.exists()  # DuckDB remains the actual state store.
    finally:
        if previous is None:
            state_owner._BOUND_STATE_TARGETS.pop(key, None)
        else:
            state_owner._BOUND_STATE_TARGETS[key] = previous


@pytest.mark.parametrize(('field', 'value'), [
    ('active_attempt', None), ('active_attempt', False), ('active_attempt', True),
    ('active_attempt', -1), ('active_attempt', '0'), ('active_attempt', 0.0),
    ('active_phase', 'implementing'), ('active_phase', 'validating_reconciled_candidate'),
    ('active_phase', 'completed'), ('active_phase', None),
    ('implementation_in_progress', True), ('implementation_in_progress', None),
    ('implementation_in_progress', 0),
])
def test_zero_or_malformed_attempt_with_contradictory_phase_is_refused(field, value):
    state = _selected()
    observed = _observed(state)
    state[field] = value
    with pytest.raises(Refusal, match='actual local Portal attempt ordinal missing'):
        watcher._native_portal_attempt(Validator, state, observed)


@pytest.mark.parametrize('field', ['active_attempt', 'active_phase', 'implementation_in_progress'])
def test_missing_native_readiness_field_is_refused(field):
    state = _selected()
    observed = _observed(state)
    del state[field]
    with pytest.raises(Refusal, match='actual local Portal attempt ordinal missing'):
        watcher._native_portal_attempt(Validator, state, observed)


@pytest.mark.parametrize('field', ['active_task_id', 'active_task_cid', 'active_task_key'])
@pytest.mark.parametrize('attempt', [0, 1])
def test_wrong_identity_refuses_before_waiting_even_for_selected_zero(field, attempt):
    state = _selected()
    observed = _observed(state)
    state[field] = 'different-fixture-identity'
    state['active_attempt'] = attempt
    with pytest.raises(Refusal, match='actual current Portal state identity differs'):
        watcher._native_portal_attempt(Validator, state, observed)


def test_readiness_observation_never_mutates_native_state_or_derived_identity():
    state = _selected()
    observed = _observed(state)
    before = copy.deepcopy((state, observed))
    assert watcher._native_portal_attempt(Validator, state, observed) is None
    assert (state, observed) == before
