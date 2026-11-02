"""Finite v3 metadata/call-site fixtures; actual public integration is separate."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker


def _plan(tmp_path):
    name = str(tmp_path / 'maintained.py')
    return {'selected_checks': [name], 'expected_nodeids': [name + '::test_a[x]', name + '::test_a[y]'],
        'selector_case_counts': {name: 2}, 'expected_cases': 2}


def test_v3_expansion_keeps_one_bounded_selector_and_exact_natural_cases(tmp_path):
    plan = _plan(tmp_path)
    assert worker.expected_natural_nodes(plan) == plan['expected_nodeids']
    assert len(plan['selected_checks']) == 1 and len(plan['expected_nodeids']) == 2


@pytest.mark.parametrize('damage', ['count', 'duplicate', 'ambiguous', 'foreign', 'over_bound'])
def test_v3_expansion_refuses_nonexact_or_unbounded_nodes(tmp_path, damage):
    plan = _plan(tmp_path)
    name = plan['selected_checks'][0]
    if damage == 'count':
        plan['selector_case_counts'][name] = 1
    elif damage == 'duplicate':
        plan['expected_nodeids'][1] = plan['expected_nodeids'][0]
    elif damage == 'ambiguous':
        plan['selected_checks'].append(name + '::test_a')
        plan['selector_case_counts'] = {name: 1, name + '::test_a': 1}
    elif damage == 'foreign':
        plan['expected_nodeids'][1] = str(tmp_path / 'other.py') + '::test_b'
    else:
        plan['expected_nodeids'] = [name + '::test_a[' + str(i) + ']' for i in range(1025)]
        plan['selector_case_counts'][name] = plan['expected_cases'] = 1025
    with pytest.raises(ValueError):
        worker.expected_natural_nodes(plan)


def test_legacy_exact_node_contract_remains_expansion_free(tmp_path):
    node = str(tmp_path / 'maintained.py') + '::test_a'
    plan = {'selected_checks': [node], 'expected_cases': 1}
    assert worker.expected_natural_nodes(plan) == [node]
    plan['selector_case_counts'] = {node: 1}
    with pytest.raises(ValueError, match='partial selector'):
        worker.expected_natural_nodes(plan)


def _scope(tmp_path, owner_code):
    root = tmp_path / 'external'
    root.mkdir()
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    pin = worker.descriptor(Path(__file__))
    scope = object.__new__(worker.TrustedFixtures)
    scope.root = root
    scope.current = pin['path'] + '::' + owner_code.co_name
    scope.profile = {'alias_slots': [], 'source_slots': []}
    scope.guard = SimpleNamespace(compiled_codes={owner_code: pin['path']}, denials={})
    scope.guard.denials = []
    scope.test_pins = {pin['path']: pin}
    scope.plan = {'source_pins': {pin['path']: pin['sha256']}, 'attempt_directory': str(attempt)}
    scope.alias_events, scope.source_events, scope.fixture_compiled = [], [], {}
    return scope, pin


def _slot(scope, pin, line, event='os.symlink'):
    return {'selector': scope.current, 'owner_source': pin,
        'function': scope.current.rpartition('::')[2], 'line': line, 'event': event}


def test_alias_authorization_uses_actual_original_python_frame_and_private_destination(tmp_path):
    scope, pin = _scope(tmp_path, inspect.currentframe().f_code)
    scope.profile['alias_slots'] = [_slot(scope, pin, inspect.currentframe().f_lineno + 1)]
    scope.alias('os.symlink', (str(scope.root / 'missing'), str(scope.root / 'link'), -1))
    assert len(scope.alias_events) == 1 and scope.alias_events[0]['source'] == pin
    assert scope.guard.denials == []


def test_alias_call_site_line_mismatch_remains_permanent_even_when_caught(tmp_path):
    scope, pin = _scope(tmp_path, inspect.currentframe().f_code)
    scope.profile['alias_slots'] = [_slot(scope, pin, 1)]
    with pytest.raises(PermissionError, match='compiled trusted call site'):
        scope.alias('os.symlink', ('missing', str(scope.root / 'link'), -1))
    assert scope.guard.denials


def test_authored_alias_slot_grants_no_outside_destination(tmp_path):
    scope, pin = _scope(tmp_path, inspect.currentframe().f_code)
    scope.profile['alias_slots'] = [_slot(scope, pin, inspect.currentframe().f_lineno + 2)]
    with pytest.raises(ValueError, match='outside original admitted outputs'):
        scope.alias('os.symlink', ('missing', str(tmp_path / 'outside'), -1))
    assert scope.alias_events == []


def test_hardlink_slot_checks_real_regular_source_descriptor(tmp_path):
    scope, pin = _scope(tmp_path, inspect.currentframe().f_code)
    source = scope.root / 'actual.txt'
    source.write_text('actual ordinary private source')
    scope.profile['alias_slots'] = [_slot(scope, pin, inspect.currentframe().f_lineno + 1, 'os.link')]
    scope.alias('os.link', (str(source), str(scope.root / 'alias'), -1, -1))
    assert scope.alias_events[0]['event'] == 'os.link'
    assert source.read_text() == 'actual ordinary private source'
