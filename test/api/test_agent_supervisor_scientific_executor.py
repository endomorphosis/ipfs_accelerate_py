"""Service control fixtures. These tests cannot establish live executor health."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as health
from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as executor


TASK = 'fixture:scientific-task'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if type(value) is bytes else (json.dumps(value, sort_keys=True) + '\n').encode()
    path.write_bytes(raw); path.chmod(0o600)
    return health.descriptor(path)


@pytest.fixture
def service_case(tmp_path, monkeypatch):
    root = tmp_path / 'proof'; root.mkdir(mode=0o700)
    seed = root / 'seed'; seed.mkdir(mode=0o700)
    python = str(Path(sys.executable).resolve())
    roles = {name: put(tmp_path / (name + '.py'), b'# explicit source-only fixture\n')
             for name in ('main_entrypoint', 'owner_source', 'driver_source')}
    roles['health_module'] = health.descriptor(Path(health.__file__).resolve())
    catalogue = {'board_namespace': 'fixture:scientific', 'plan_root_cid': 'fixture:plan',
        'repository_tree_id': 'fixture:tree', 'population_sha256': 'a' * 64,
        'database_uuid': 'fixture:uuid', 'store_generation': 1}
    database = put(tmp_path / 'explicit-store-preimage.bin', b'fixture-db')
    profile = {'schema': health.PROFILE_SCHEMA, 'catalogue': catalogue,
        'unit': {'name': 'fixture-scientific.service', 'argv': [python, '-B', roles['main_entrypoint']['path']],
            'interpreter': python, 'working_directory': str(tmp_path), 'environment': {'PYTHONDONTWRITEBYTECODE': '1'}},
        'source_descriptors': [*roles.values(), health.installed_tool_descriptor(python)], **roles,
        'task_scope': [TASK], 'max_age_seconds': 120, 'max_cycle_seconds': 300,
        'proof_root': str(root), 'receipt_path': str(root / 'current.json'),
        'cycle_contract': {'plan_schema': 'fixture-idle-plan@1', 'plan_operation': 'fixture_idle'}}
    plan = {'schema': 'fixture-idle-plan@1', 'operation': 'fixture_idle', 'task_scope': [TASK],
        'native_metadata_cli': {'result_path': str(seed / 'attempt/exports/result.json'),
            'observation_binding': {'database_preimage': database}}}
    plan_ref = put(seed / 'plan.json', plan)
    request = {'schema': 'legal-ir-original-native-control-request/v1', 'purpose': 'native_control_metadata',
        'timeout_seconds': 120, 'database_preimage': database,
        'attempt_directory': str(seed / 'attempt'),
        'source_sha256': {roles[field]['path']: roles[field]['sha256'] for field in ('owner_source', 'driver_source')}}
    request['source_sha256'][plan_ref['path']] = plan_ref['sha256']
    template = {'schema': executor.TEMPLATE_SCHEMA, 'kind': 'genuine_scoped_idle', 'seed_root': str(seed),
        'request_ref': put(seed / 'request.json', request), 'plan_ref': plan_ref,
        'links': {'result': 'attempt/exports/result.json', 'origins': 'attempt/exports/origins.json',
            'execution_request': 'attempt/execution-request.json'}}
    template_ref = put(tmp_path / 'template.json', template)
    config = {'schema': executor.CONFIG_SCHEMA, 'profile_path': str(tmp_path / 'profile.json'),
        'catalogue': {**{key: catalogue[key] for key in ('board_namespace', 'plan_root_cid', 'repository_tree_id', 'population_sha256')},
            'database_path': database['path'], 'population_path': str(tmp_path / 'population.json'),
            'board_id': 'legal-ir-scientific', 'max_age_seconds': 120, 'timeout_seconds': 15},
        'idle_template_ref': template_ref, 'execution_templates': [], 'poll_seconds': 1,
        'max_cycles': 2, 'max_proof_bytes': 1024**2}
    config_ref = put(tmp_path / 'config.json', config)
    profile['source_descriptors'].append(config_ref)
    profile_ref = put(tmp_path / 'profile.json', profile)
    native = {'schema': health.DIAGNOSTIC_SCHEMA, **catalogue, 'event_cursor': 5,
        'fresh_read': True, 'completion_authority': False, 'database_and_wal_metadata_unchanged': True,
        'checked_at_epoch': time.time(), 'ready_status_count': 0,
        'tasks': [{'task_cid': TASK, 'status': 'blocked', 'revision': 2}]}
    # Service control tests inject this gate. Separate tests below exercise the
    # real gate; no fixture claims to be an actual systemd MainPID.
    monkeypatch.setattr(executor, 'require_main_process', lambda *args: {'pid': 12345})
    return dict(root=root, seed=seed, config=config, config_ref=config_ref,
        profile=profile, profile_ref=profile_ref, native=native, template=template,
        template_ref=template_ref, plan=plan, request=request)


def test_relocation_preserves_external_and_similar_prefixes():
    value = {'/seed/a': ['/seed', '/seed/file', '/seeded/file', '/elsewhere']}
    assert executor.relocate(value, Path('/seed'), Path('/fresh')) == {
        '/fresh/a': ['/fresh', '/fresh/file', '/seeded/file', '/elsewhere']}
    assert value['/seed/a'][0] == '/seed'


def test_closed_configuration_and_preparation_reuse_original_sources(service_case,monkeypatch):
    case = service_case
    config, profile_ref, profile = executor.load_configuration(case['config_ref'])
    calls=[];actual_data_reader=health.database_descriptor
    def data_reader(path):
        calls.append(str(path));return actual_data_reader(path)
    monkeypatch.setattr(health,'database_descriptor',data_reader)
    cycle = case['root'] / 'cycle-preparation'; cycle.mkdir(mode=0o700)
    request_ref, plan_ref, links = executor.prepare_cycle(case['template_ref'], 'genuine_scoped_idle', config, profile, cycle)
    request, plan = health._document(request_ref), health._document(plan_ref)
    assert request['argv'] == [profile['unit']['argv'][0], '-B', profile['driver_source']['path'],
        '--plan', plan_ref['path'], '--plan-sha256', plan_ref['sha256']]
    assert request['database_preimage'] == plan['native_metadata_cli']['observation_binding']['database_preimage']
    assert plan['native_metadata_cli']['result_path'] == str(links['result'])
    assert profile_ref == case['profile_ref'] and not links['result'].exists()
    assert calls==[config['catalogue']['database_path']]


@pytest.mark.parametrize('status', ['blocked', 'completed', 'complete', 'done', 'skipped'],
    ids=['blocked', 'completed', 'complete', 'done', 'skipped'])
def test_only_actual_quiet_scoped_states_select_idle(service_case, status):
    case = service_case; case['native']['tasks'][0]['status'] = status
    assert executor.select_template(case['config'], case['profile'], case['native']) == (
        case['template_ref'], 'genuine_scoped_idle')


@pytest.mark.parametrize('status', ['ready', 'running', 'claimed', 'unknown', 'pending'],
    ids=['ready', 'running', 'claimed', 'unknown', 'pending'])
def test_global_zero_ready_does_not_bypass_nonquiet_scope(service_case, status):
    case = service_case; case['native']['tasks'][0]['status'] = status
    with pytest.raises(executor.ExecutorHold, match='no_explicit_execution_binding'):
        executor.select_template(case['config'], case['profile'], case['native'])


def test_ready_dispatch_needs_exact_finite_cid_and_current_revision(service_case):
    case = service_case; case['native']['tasks'][0]['status'] = 'ready'
    case['config']['execution_templates'] = [{'task_cid': TASK, 'expected_revision': 1, 'template_ref': {'fixture': 'old'}}]
    with pytest.raises(executor.ExecutorHold):
        executor.select_template(case['config'], case['profile'], case['native'])
    case['config']['execution_templates'][0]['expected_revision'] = 2
    assert executor.select_template(case['config'], case['profile'], case['native']) == ({'fixture': 'old'}, 'native_execution')


@pytest.mark.parametrize('field,value', [('poll_seconds', 0), ('poll_seconds', True),
    ('max_cycles', 1), ('max_cycles', 10001), ('max_proof_bytes', 0)],
    ids=['zero-poll', 'boolean-poll', 'one-cycle', 'unbounded-cycles', 'zero-storage'])
def test_configuration_limits_reject_before_observation(service_case, field, value):
    case = service_case; case['config'][field] = value
    config_ref = put(Path(case['config_ref']['path']), case['config'])
    case['profile']['source_descriptors'][-1] = config_ref
    put(Path(case['profile_ref']['path']), case['profile'])
    with pytest.raises(executor.ExecutorHold, match='finite_limits'):
        executor.load_configuration(config_ref)


@pytest.mark.parametrize('mutation', ['foreign-seed', 'parent-link', 'absolute-link', 'alias-links'],
    ids=['foreign-seed', 'parent-link', 'absolute-link', 'alias-links'])
def test_template_paths_are_closed_before_any_original_process(service_case, mutation):
    case = service_case; template = copy.deepcopy(case['template'])
    if mutation == 'foreign-seed': template['seed_root'] = str(case['root'].parent)
    elif mutation == 'parent-link': template['links']['result'] = '../outside.json'
    elif mutation == 'absolute-link': template['links']['result'] = '/outside.json'
    else: template['links']['result'] = template['links']['origins']
    with pytest.raises((executor.ExecutorHold, health.ExecutorHealthError)):
        executor.validate_template(template, case['profile'], 'genuine_scoped_idle')


def test_owner_pin_is_required_before_preparing_original_cycle(service_case):
    case = service_case
    case['request']['source_sha256'].pop(case['profile']['owner_source']['path'])
    ref = put(Path(case['template']['request_ref']['path']), case['request'])
    case['template']['request_ref'] = ref
    template_ref = put(Path(case['template_ref']['path']), case['template'])
    cycle = case['root'] / 'cycle-pin'; cycle.mkdir(mode=0o700)
    with pytest.raises(executor.ExecutorHold, match='original_source_binding'):
        executor.prepare_cycle(template_ref, 'genuine_scoped_idle', case['config'], case['profile'], cycle)


def test_persistent_hold_preserves_first_uncertain_cycle(service_case):
    case = service_case
    executor.persistent_hold(case['root'], case['root'] / 'unknown-cycle', 'original timeout')
    path = case['root'] / 'executor-held.json'; original = path.read_bytes()
    executor.persistent_hold(case['root'], case['root'] / 'later-cycle', 'new excuse')
    assert path.read_bytes() == original
    assert json.loads(original)['automatic_retry_admitted'] is False


@pytest.mark.parametrize('state', ['persistent-hold', 'unfinished-cycle'], ids=['persistent-hold', 'unfinished-cycle'])
def test_startup_refuses_unknown_prior_cycle_before_observation_or_spawn(service_case, monkeypatch, state):
    case = service_case
    if state == 'persistent-hold': executor.persistent_hold(case['root'], case['root'], 'unknown')
    else: (case['root'] / 'cycle-unknown').mkdir(mode=0o700)
    def forbidden(*args, **kwargs): pytest.fail('held startup cannot observe or spawn')
    monkeypatch.setattr(executor, 'observe_native', forbidden)
    monkeypatch.setattr(executor.subprocess, 'Popen', forbidden)
    with pytest.raises(executor.ExecutorHold, match='reconciliation'):
        executor.serve(case['config_ref'], threading.Event())


def test_selection_failure_is_persistently_held_without_dispatch(service_case, monkeypatch):
    case = service_case; case['native']['tasks'][0]['status'] = 'ready'
    monkeypatch.setattr(executor, 'observe_native', lambda config: case['native'])
    monkeypatch.setattr(executor, 'run_cycle', lambda *args: pytest.fail('unbound task must not dispatch'))
    assert executor.serve(case['config_ref'], threading.Event()) == 78
    assert (case['root'] / 'executor-held.json').exists()


def test_two_control_cycles_advance_sequence_and_stop_at_finite_budget(service_case, monkeypatch):
    case = service_case; seen = []
    monkeypatch.setattr(executor, 'observe_native', lambda config: case['native'])
    monkeypatch.setattr(executor, 'run_cycle', lambda *args: seen.append((args[4], args[5])))
    class NoWait:
        def is_set(self): return False
        def wait(self, seconds): return False
    assert executor.serve(case['config_ref'], NoWait()) == 78
    assert seen == [('genuine_scoped_idle', 1), ('genuine_scoped_idle', 2)]
    assert json.loads((case['root'] / 'executor-held.json').read_text())['reason'] == 'finite_cycle_budget_exhausted'


def test_failed_cycle_preparation_leaves_hold_and_no_original_process(service_case, monkeypatch):
    case = service_case
    def fail(*args): raise executor.ExecutorHold('fixture preparation failure')
    monkeypatch.setattr(executor, 'prepare_cycle', fail)
    monkeypatch.setattr(executor.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('prepare failure must not spawn'))
    with pytest.raises(executor.ExecutorHold, match='preparation'):
        executor.run_cycle(case['config'], case['profile_ref'], case['profile'], case['template_ref'],
            'genuine_scoped_idle', 1, threading.Event())
    assert (case['root'] / 'executor-held.json').exists()
    assert len(list(case['root'].glob('cycle-*'))) == 1


@pytest.mark.parametrize('failure', ['timeout', 'signal', 'nonzero'], ids=['timeout', 'signal', 'nonzero'])
def test_unknown_or_failed_original_cycle_is_terminated_held_and_never_retried(service_case, monkeypatch, failure):
    case = service_case; calls = []; stop = threading.Event()
    class FixtureProcess:
        def __init__(self): self.closed = failure == 'nonzero'; self.terminated = False
        def poll(self): return 1 if self.closed else None
        def wait(self, timeout=None): self.closed = True; return 1
        def terminate(self): self.terminated = True
    process = FixtureProcess()
    def fixture_spawn(*args, **kwargs): calls.append(args[0]); return process
    monkeypatch.setattr(executor.subprocess, 'Popen', fixture_spawn)
    monkeypatch.setattr(health, 'track_original_cycle', lambda *args, **kwargs: object())
    monkeypatch.setattr(health, 'publish_after_native_cycle', lambda *args, **kwargs: pytest.fail('failed cycle cannot publish'))
    if failure == 'signal': stop.set()
    elif failure == 'timeout':
        ticks = iter([0, 301]); monkeypatch.setattr(executor.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(executor.ExecutorHold):
        executor.run_cycle(case['config'], case['profile_ref'], case['profile'], case['template_ref'],
            'genuine_scoped_idle', 1, stop)
    assert len(calls) == 1
    assert process.terminated is (failure != 'nonzero')
    assert (case['root'] / 'executor-held.json').exists()
    monkeypatch.setattr(executor, 'observe_native', lambda *args: pytest.fail('held restart cannot observe'))
    with pytest.raises(executor.ExecutorHold, match='reconciliation'):
        executor.serve(case['config_ref'], threading.Event())
    assert len(calls) == 1


def test_terminal_result_cannot_substitute_foreign_request():
    ref = {'sha256': 'a' * 64}
    raw = json.dumps({'schema': 'legal-ir-original-native-control-terminal/v1', 'request_sha256': 'b' * 64}).encode()
    with pytest.raises(executor.ExecutorHold, match='missing_or_ambiguous'):
        executor.original_terminal(raw, ref)


def test_proof_budget_refuses_symlinks_without_creating_a_special_fixture():
    import stat
    from types import SimpleNamespace
    class FixturePath:
        def lstat(self): return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777)
    class FixtureRoot:
        def rglob(self, pattern): return [FixturePath()]
    with pytest.raises(executor.ExecutorHold, match='proof_symlink'):
        executor.check_proof_budget(FixtureRoot(), 1024**2)


def test_main_process_gate_reuses_canonical_identity_before_any_owner(monkeypatch):
    import os
    profile = {'main_entrypoint': {'path': str(Path(executor.__file__).resolve())},
        'health_module': {'path': str(Path(health.__file__).resolve())}}
    calls = []
    def observed(*args): calls.append(args); return {'pid': os.getpid(), 'birth_ticks': 123}
    monkeypatch.setattr(health, '_identity', observed)
    assert executor.require_main_process({'fixture': 'profile'}, profile) == {'pid': os.getpid(), 'birth_ticks': 123}
    assert len(calls) == 1


@pytest.mark.parametrize('failure', ['pid', 'entrypoint', 'health-origin'], ids=['pid', 'entrypoint', 'health-origin'])
def test_main_process_gate_rejects_foreign_identity_and_source(monkeypatch, failure):
    import os
    profile = {'main_entrypoint': {'path': str(Path(executor.__file__).resolve())},
        'health_module': {'path': str(Path(health.__file__).resolve())}}
    if failure == 'entrypoint': profile['main_entrypoint']['path'] += '.foreign'
    if failure == 'health-origin': profile['health_module']['path'] += '.foreign'
    monkeypatch.setattr(health, '_identity', lambda *args: {'pid': os.getpid() + 1})
    with pytest.raises(executor.ExecutorHold):
        executor.require_main_process({'fixture': 'profile'}, profile)



