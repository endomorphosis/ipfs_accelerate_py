"""Real prelaunch wait/event path; diagnostic output never changes decisions."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.maintenance_diagnostics import (
    prelaunch_maintenance_observation,
)
from test.api.test_agent_supervisor_database_portal_reload_gate import _config


def test_real_prelaunch_loop_records_original_projection_cause_without_replay(tmp_path, monkeypatch):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    cause = supervisor._database_portal_reload_inconclusive_projection(TypeError("'bool' object is not iterable"))
    initial = {'maintenance_blocked': True, 'reason': 'database_portal_owner_mutation_fence_unavailable',
               'database_portal_reload_projection': cause}
    before = deepcopy(initial)
    calls = []
    ready = {'maintenance_blocked': False, 'safe_to_restart': False, 'retained_unknown_outcome': True}
    def maintenance(*, include_refill):
        calls.append(include_refill)
        return ready
    monkeypatch.setattr(supervisor, 'run_once', maintenance)
    monkeypatch.setattr(supervisor, '_supervisor_loop_recovery_delay_seconds', lambda: 0)
    result = supervisor._await_prelaunch_maintenance(initial)
    rows = [json.loads(line) for line in supervisor.config.events_path.read_text().splitlines()]
    event = next(row for row in rows if row['type'] == 'supervisor_prelaunch_fenced_for_recovery')
    assert event['reason'] == initial['reason']
    assert event['maintenance_cause'] == {
        'diagnostic_only': True, 'source': 'database_portal_reload_projection',
        'reason': 'database_portal_projection_inconclusive', 'error_type': 'TypeError',
        'error': "'bool' object is not iterable"}
    assert calls == [False] and result == ready and initial == before
    assert result['safe_to_restart'] is False and result['retained_unknown_outcome'] is True


@pytest.mark.parametrize('message', [
    'Authorization: Bearer private-value', 'password=short',
    'access_token=private-value', 'api-key: private-value', 'token=private-value',
    'https://user:private-value@example.invalid/quack',
    '-----BEGIN PRIVATE KEY-----\nprivate-value',
])
def test_sensitive_cause_never_enters_new_observation(message):
    result = {'maintenance_blocked': True, 'error': message,
              'database_portal_reload_projection': {'error_type':'RuntimeError','error':message,
                                                    'quack_owner':{'token':'hidden-owner-field'}}}
    before = deepcopy(result)
    observation = prelaunch_maintenance_observation(result, delay_seconds=20)
    encoded = json.dumps(observation)
    assert 'private-value' not in encoded and 'hidden-owner-field' not in encoded
    assert observation['maintenance_cause']['error_type'] == 'RuntimeError'
    assert observation['maintenance_cause']['error'] == '[redacted sensitive diagnostic]'
    assert result == before


def test_cause_has_utf8_byte_bound_and_no_terminal_controls():
    result = {'database_portal_reload_projection': {'error_type':'IOException','error':'\x1b[31m\n'+'界'*5000}}
    observation = prelaunch_maintenance_observation(result, delay_seconds=20)
    text = observation['maintenance_cause']['error']
    assert len(text.encode()) <= 400 and '\x1b' not in text and '\n' not in text
    assert text.endswith('...')


def test_malformed_projection_does_not_coerce_types_or_invent_authority():
    observation = prelaunch_maintenance_observation({'reason':True, 'database_portal_reload_projection':{
        'error_type':False,'error':['not a string'],'authority_available':True,'safe_to_restart':True}},delay_seconds=20)
    assert observation == {'reason':'maintenance_blocked','delay_seconds':20}


def test_top_level_typed_unknown_outcome_stays_diagnostic_only():
    observation = prelaunch_maintenance_observation({'reason':'prelaunch_recovery_failed',
        'error_type':'IntentRepositoryUnknownOutcomeError','error':'remote outcome requires exact reconciliation',
        'safe_to_restart':False},delay_seconds=20)
    cause = observation['maintenance_cause']
    assert cause['source'] == 'maintenance_result'
    assert cause['error_type'] == 'IntentRepositoryUnknownOutcomeError'
    assert cause['diagnostic_only'] is True
    assert 'safe_to_restart' not in observation and 'safe_to_restart' not in cause
