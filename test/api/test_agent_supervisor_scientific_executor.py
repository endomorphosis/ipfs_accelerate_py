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


@pytest.fixture
def nonstarted_case(service_case, monkeypatch):
    """Explicit source-only historical data; no systemd/native/DB acceptance."""
    case = service_case; archive = case['root'] / 'reconciliation-v01'; archive.mkdir(mode=0o700)
    source_root = case['root'].parent / 'explicit-source'
    role_names = {'main_entrypoint': 'scientific_executor.py', 'health_module': 'native_executor_health.py',
        'owner_source': 'original_embedded_observation_owner.py', 'driver_source': 'native_embedded_observation_driver.py'}
    for role, name in role_names.items():
        old = case['profile'][role]
        new = put(source_root / 'ipfs_accelerate_py/agent_supervisor/rescue' / name, Path(old['path']).read_bytes())
        case['profile']['source_descriptors'] = [new if pin == old else pin for pin in case['profile']['source_descriptors']]
        case['profile'][role] = new
        case['profile']['unit']['argv'] = [new['path'] if arg == old['path'] else arg for arg in case['profile']['unit']['argv']]
        if role in ('owner_source', 'driver_source'):
            case['request']['source_sha256'].pop(old['path']); case['request']['source_sha256'][new['path']] = new['sha256']
    case['profile_ref'] = put(Path(case['profile_ref']['path']), case['profile'])
    prior_ref = dict(case['profile_ref']); prior = copy.deepcopy(case['profile'])
    prior_snapshot = put(archive / 'prior-profile.json', Path(prior_ref['path']).read_bytes())
    failed = case['root'] / ('cycle-00000001-' + '1' * 32); failed.mkdir(mode=0o700)
    attempt = failed / 'attempt'; attempt.mkdir(mode=0o700)
    policy = {'fixture_memory_mb': 1536, 'fixture_global_headroom_mb': 24922}
    state = {'config': policy, 'waiters': {}, 'leases': {}}
    scheduler = put(case['root'].parent / 'explicit-scheduler.json', state)
    request = dict(case['request'], request_id='fixture:failed-cycle-1',
        attempt_directory=str(attempt), scheduler_state_path=scheduler['path'],
        expected_scheduler_policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest())
    request_ref = put(failed / 'request.json', request)
    intent_ref = put(failed / 'intent.json', dict(kind='genuine_scoped_idle', sequence=1,
        request_ref=request_ref, automatic_retry_admitted=False))
    terminal = dict(schema='legal-ir-original-native-control-terminal/v1',
        request_sha256=request_ref['sha256'], owner_pid=1000002, owner_birth='2',
        completed=False, child_launch_attempted=False, child_reaped=False, child_returncode=None,
        control_effects='not_started', storage_status='not_requested', original_native_compute_closed=True,
        source_READ_closed=True, absolute_resource_operation_deadline_closed=True,
        cleanup_errors=[], received_signals=[], error_type='ResourceAdmissionDeadlineError',
        database_preimage_before_grants=request['database_preimage'],
        task_completion_authority_granted_by_resource_owner=False)
    stdout_ref = put(failed / 'owner.stdout', terminal)
    executor.persistent_hold(case['root'], failed, 'original_owner_failed')
    hold_ref = health.descriptor(case['root'] / 'executor-held.json'); hold = health._document(hold_ref)
    old_main = dict(unit=prior['unit']['name'], pid=1000001, birth_ticks=1,
        boot_id='explicit-old-source-fixture-boot', invocation_id='a' * 32,
        argv=prior['unit']['argv'], exe=prior['unit']['interpreter'], cwd=prior['unit']['working_directory'],
        environment={**prior['unit']['environment'], 'INVOCATION_ID': 'a' * 32,
            health.ENV_PROFILE_SHA: prior_ref['sha256']})
    plan = dict(files=[{}, {'after': {**prior_snapshot, 'path': '/historical/not-opened/profile.json'}}],
        main_loop_limits={key: case['config'][key] for key in ('max_cycles', 'poll_seconds', 'max_proof_bytes')})
    plan_ref = put(archive / 'activation-plan.json', plan)
    capture = dict(schema='legal-ir-scientific-first-install-actual-capture/v1', accepted=False,
        applied=True, source_READ_closed_before_start=True, start_requested=True,
        plan={**plan_ref, 'path': '/historical/not-opened/plan.json'}, opening_actual_main_identity=old_main)
    capture_ref = put(archive / 'activation-capture.json', capture)
    negative = dict(schema='scientific-first-activation-independent-negative-audit/v1',
        accepted_as_runtime_activation=False, negative_terminal_and_hold_verified=True,
        attempt_directory_empty=True, cycle_receipts_absent=True, original_terminal=terminal,
        persistent_hold=hold, failed_initial_activation={'opening_identity': old_main})
    negative_ref = put(archive / 'negative-audit.json', negative)
    dropin = put(case['root'].parent / 'explicit-memory.conf', b'[Service]\nMemoryHigh=infinity\n')
    correction = dict(schema='scientific-service-memory-context-actual-repair/v1', accepted=True,
        native_envelope_and_policy_changed=False, restart_or_cycle_replay=False,
        scientific_after={'Id': prior['unit']['name'], 'MainPID': '0', 'MemoryHigh': 'infinity'},
        persistent_manager_dropin={**dropin, 'text': '[Service]\nMemoryHigh=infinity\n'})
    correction_ref = put(archive / 'memory-correction.json', correction)
    inventory = dict(schema='legal-ir-scientific-source-inventory/v1', source_root=str(source_root),
        source_commit='c' * 40, source_tree='d' * 40, file_count=4,
        files=[case['profile'][field] for field in role_names])
    inventory_ref = put(archive / 'source-inventory.json', inventory)
    source_commitment = dict(schema='legal-ir-scientific-source-commitment/v1', source_root=str(source_root),
        source_commit='c' * 40, source_tree='d' * 40, source_inventory=inventory_ref, runtime_activation_or_task_authority=False)
    commitment = put(archive / 'source-commitment.json', source_commitment)
    recovery_path = archive / 'explicit-authorization.json'
    case['profile']['unit']['argv'].extend(['serve', '--config', case['config_ref']['path'],
        '--config-sha256', case['config_ref']['sha256'], '--known-nonstarted-recovery', str(recovery_path)])
    case['profile_ref'] = put(Path(case['profile_ref']['path']), case['profile'])
    value = dict(schema=executor.RECOVERY_SCHEMA, profile_ref=case['profile_ref'],
        prior_profile_ref=prior_ref, prior_profile_snapshot_ref=prior_snapshot,
        activation_capture_ref=capture_ref, activation_plan_ref=plan_ref, negative_audit_ref=negative_ref,
        memory_correction_ref=correction_ref, source_commitment_ref=commitment, hold_ref=hold_ref,
        intent_ref=intent_ref, request_ref=request_ref, owner_stdout_ref=stdout_ref, failed_cycle_root=str(failed),
        scheduler_snapshot_ref=put(archive / 'scheduler.snapshot.json', state), max_cycles=case['config']['max_cycles'],
        authorization='resume_genuine_scoped_idle', recorded_at=time.time(),
        manager_before={'Id': prior['unit']['name'], 'MainPID': '0', 'ActiveState': 'failed'})
    recovery_ref = put(recovery_path, value)
    put(case['root'] / 'scientific-executor.lock', b'')
    monkeypatch.setenv(executor.RECOVERY_SHA_ENV, recovery_ref['sha256'])
    opening = {'pid': 1000003, 'birth_ticks': 3, 'boot_id': 'explicit-new-fixture-boot', 'invocation_id': 'b' * 32}
    monkeypatch.setattr(executor, 'require_main_process', lambda *args: opening)
    case.update(recovery=value, recovery_ref=recovery_ref, failed=failed, archive=archive,
        terminal=terminal, negative=negative, correction=correction, scheduler_state=state,
        scheduler_ref=scheduler, inventory=inventory, source_commitment=source_commitment,
        old_hold_bytes=Path(hold_ref['path']).read_bytes(), opening=opening)
    return case


def _resume(case):
    return executor.reconcile_known_nonstarted(case['recovery_ref'], case['profile_ref'],
        case['profile'], case['config'], case['opening'])


def _resign_recovery(case, monkeypatch):
    case['recovery_ref'] = put(Path(case['recovery_ref']['path']), case['recovery'])
    monkeypatch.setenv(executor.RECOVERY_SHA_ENV, case['recovery_ref']['sha256'])


def test_explicit_nonstarted_ack_is_negative_same_main_idempotent_and_preserves_hold(nonstarted_case):
    case = nonstarted_case
    assert _resume(case) == 1
    ack = case['failed'] / 'nonstarted-reconciliation.json'; first = ack.read_bytes()
    assert _resume(case) == 1 and ack.read_bytes() == first
    document = json.loads(first)
    assert document['health_credit'] is False and document['completion_authority'] is False
    assert document['dispatch_authority'] is False and document['consuming_main_identity'] == case['opening']
    assert (case['root'] / 'executor-held.json').read_bytes() == case['old_hold_bytes']
    assert not (case['failed'] / 'closed.json').exists() and not Path(case['profile']['receipt_path']).exists()


@pytest.mark.parametrize('field,value', [('child_launch_attempted', True), ('child_reaped', True),
    ('child_returncode', 0), ('control_effects', 'started_reconcile_on_failure'), ('storage_status', 'active'),
    ('original_native_compute_closed', False), ('source_READ_closed', False),
    ('absolute_resource_operation_deadline_closed', False), ('cleanup_errors', ['unknown']),
    ('received_signals', [15]), ('completed', True), ('owner_birth', True)],
    ids=['launch-attempted', 'reaped-child', 'returncode', 'effects', 'storage', 'native-open',
        'source-open', 'deadline-open', 'cleanup-unknown', 'operator-signal', 'success-not-negative', 'boolean-birth'])
def test_recovery_refuses_nonclosed_or_launched_original_terminal(nonstarted_case, monkeypatch, field, value):
    case = nonstarted_case; case['terminal'][field] = value
    case['recovery']['owner_stdout_ref'] = put(Path(case['recovery']['owner_stdout_ref']['path']), case['terminal'])
    case['negative']['original_terminal'] = case['terminal']
    case['recovery']['negative_audit_ref'] = put(Path(case['recovery']['negative_audit_ref']['path']), case['negative'])
    _resign_recovery(case, monkeypatch)
    with pytest.raises((executor.ExecutorHold, health.ExecutorHealthError)):
        _resume(case)
    assert not (case['failed'] / 'nonstarted-reconciliation.json').exists()


@pytest.mark.parametrize('mutation', ['environment', 'profile', 'hold', 'request', 'negative',
    'correction', 'scope', 'waiter', 'lease', 'policy', 'other-cycle', 'execution', 'budget', 'unit-argv',
    'stale-commitment', 'new-main-sha'],
    ids=['environment', 'profile', 'hold', 'request', 'negative', 'correction', 'scope', 'waiter',
        'lease', 'policy', 'other-cycle', 'execution', 'budget', 'unit-argv', 'stale-commitment', 'new-main-sha'])
def test_recovery_refuses_foreign_or_live_state_before_ack(nonstarted_case, monkeypatch, mutation):
    case = nonstarted_case
    if mutation == 'environment': monkeypatch.setenv(executor.RECOVERY_SHA_ENV, 'f' * 64)
    elif mutation == 'profile': case['recovery']['profile_ref'] = dict(case['profile_ref'], sha256='f' * 64)
    elif mutation == 'hold': put(case['root'] / 'executor-held.json', {'changed': True})
    elif mutation == 'request': case['recovery']['request_ref'] = dict(case['recovery']['request_ref'], sha256='f' * 64)
    elif mutation == 'negative': case['negative']['negative_terminal_and_hold_verified'] = False; case['recovery']['negative_audit_ref'] = put(Path(case['recovery']['negative_audit_ref']['path']), case['negative'])
    elif mutation == 'correction': case['correction']['native_envelope_and_policy_changed'] = True; case['recovery']['memory_correction_ref'] = put(Path(case['recovery']['memory_correction_ref']['path']), case['correction'])
    elif mutation == 'scope': case['profile']['task_scope'] = ['fixture:foreign-cid']
    elif mutation in ('waiter', 'lease'):
        key = 'waiters' if mutation == 'waiter' else 'leases'
        case['scheduler_state'][key]['named-original'] = {'request_id': 'fixture:failed-cycle-1:root'}
        put(Path(case['scheduler_ref']['path']), case['scheduler_state'])
    elif mutation == 'policy': case['scheduler_state']['config']['fixture_memory_mb'] += 1; put(Path(case['scheduler_ref']['path']), case['scheduler_state'])
    elif mutation == 'other-cycle':
        extra = case['root'] / ('cycle-00000002-' + '2' * 32); extra.mkdir()
        put(extra / 'intent.json', {'sequence': 2})
    elif mutation == 'execution': case['config']['execution_templates'] = [{'task_cid': TASK}]
    elif mutation == 'budget': case['config']['max_cycles'] = 3; case['recovery']['max_cycles'] = 3
    elif mutation == 'unit-argv': case['profile']['unit']['argv'].remove('--known-nonstarted-recovery')
    elif mutation == 'stale-commitment':
        case['source_commitment']['source_commit'] = 'e' * 40
        case['recovery']['source_commitment_ref'] = put(Path(case['recovery']['source_commitment_ref']['path']), case['source_commitment'])
    else:
        case['inventory']['files'][0] = dict(case['inventory']['files'][0], sha256='f' * 64)
        case['source_commitment']['source_inventory'] = put(Path(case['source_commitment']['source_inventory']['path']), case['inventory'])
        case['recovery']['source_commitment_ref'] = put(Path(case['recovery']['source_commitment_ref']['path']), case['source_commitment'])
    if mutation != 'environment': _resign_recovery(case, monkeypatch)
    with pytest.raises((executor.ExecutorHold, health.ExecutorHealthError)):
        _resume(case)
    assert not (case['failed'] / 'nonstarted-reconciliation.json').exists()


@pytest.mark.parametrize('identity', ['old-main', 'old-owner', 'unknown'], ids=['old-main', 'old-owner', 'unknown'])
def test_recovery_refuses_live_or_unknown_old_processes(nonstarted_case, monkeypatch, identity):
    case = nonstarted_case
    def decision(old, current):
        if identity == 'unknown': raise health.ExecutorHealthError('previous_process_custody_unknown')
        return old['pid'] != (1000001 if identity == 'old-main' else 1000002)
    monkeypatch.setattr(health, '_definitely_dead_identity', decision)
    with pytest.raises((executor.ExecutorHold, health.ExecutorHealthError)):
        _resume(case)
    assert not (case['failed'] / 'nonstarted-reconciliation.json').exists()


def test_consumed_recovery_receipt_cannot_bind_another_main(nonstarted_case):
    case = nonstarted_case; assert _resume(case) == 1
    case['opening'] = dict(case['opening'], pid=1000004, birth_ticks=4, invocation_id='c' * 32)
    with pytest.raises(executor.ExecutorHold, match='already_consumed'):
        _resume(case)


def test_admin_authorization_is_metadata_only_and_requires_actual_no_main(nonstarted_case, monkeypatch):
    case = nonstarted_case
    spec = {key: value for key, value in case['recovery'].items()
        if key not in ('schema', 'recorded_at', 'manager_before')}
    spec_ref = put(case['archive'] / 'authorization-spec.json', spec)
    target = Path(case['recovery_ref']['path']); target.unlink()
    monkeypatch.setattr(health, '_manager', lambda unit: dict(Id=unit, MainPID='0', ActiveState='failed'))
    monkeypatch.setattr(executor.subprocess, 'Popen', lambda *a, **k: pytest.fail('admin metadata must not spawn'))
    created = executor.authorize_known_nonstarted_recovery(spec_ref, target)
    assert created['path'] == str(target)
    assert health._document(created)['manager_before']['MainPID'] == '0'
    assert not (case['failed'] / 'nonstarted-reconciliation.json').exists()
    assert (case['root'] / 'executor-held.json').read_bytes() == case['old_hold_bytes']


def test_admin_authorization_refuses_running_unit_before_publication(nonstarted_case, monkeypatch):
    case = nonstarted_case
    spec = {key: value for key, value in case['recovery'].items()
        if key not in ('schema', 'recorded_at', 'manager_before')}
    spec_ref = put(case['archive'] / 'authorization-spec.json', spec)
    monkeypatch.setattr(health, '_manager', lambda unit: dict(Id=unit, MainPID='12345', ActiveState='active'))
    target = case['archive'] / 'must-remain-absent.json'
    with pytest.raises(executor.ExecutorHold, match='requires_no_main'):
        executor.authorize_known_nonstarted_recovery(spec_ref, target)
    assert not target.exists()


def test_explicit_resume_counts_failed_cycle_and_keeps_first_health_sequence(nonstarted_case, monkeypatch):
    case = nonstarted_case; seen = []
    monkeypatch.setattr(executor, 'observe_native', lambda config: case['native'])
    monkeypatch.setattr(executor, 'run_cycle', lambda *args, **kwargs: seen.append((args[5], kwargs['cycle_number'])))
    class NoWait:
        def is_set(self): return False
        def wait(self, seconds): return False
    assert executor.serve(case['config_ref'], NoWait(), recovery_ref=case['recovery_ref']) == 78
    assert seen == [(1, 2)]
    assert (case['root'] / 'executor-held.json').read_bytes() == case['old_hold_bytes']


def test_resumed_observation_failure_is_recorded_without_overwriting_original_hold(nonstarted_case, monkeypatch):
    case = nonstarted_case
    def failure(config): raise executor.ExecutorHold('new observation is unknown')
    monkeypatch.setattr(executor, 'observe_native', failure)
    monkeypatch.setattr(executor.subprocess, 'Popen', lambda *a, **k: pytest.fail('no observation may spawn'))
    assert executor.serve(case['config_ref'], threading.Event(), recovery_ref=case['recovery_ref']) == 78
    assert (case['root'] / 'executor-held.json').read_bytes() == case['old_hold_bytes']
    later = list(case['root'].glob('cycle-00000002-*')); assert len(later) == 1
    assert json.loads((later[0] / 'cycle-held.json').read_text())['reason'] == 'new observation is unknown'
    with pytest.raises(executor.ExecutorHold, match='other_unfinished'):
        _resume(case)


def test_recovery_live_scheduler_group_writable_data_preserves_strict_archive(nonstarted_case):
    case = nonstarted_case; live = Path(case['scheduler_ref']['path']); before = live.read_bytes()
    live.chmod(0o664)
    with pytest.raises(health.ExecutorHealthError, match='unsafe_or_oversized_file'):
        health.descriptor(live)
    assert _resume(case) == 1
    assert live.read_bytes() == before and live.stat().st_mode & 0o777 == 0o664
    assert (case['root'] / 'executor-held.json').read_bytes() == case['old_hold_bytes']
    assert not (case['failed'] / 'closed.json').exists() and not Path(case['profile']['receipt_path']).exists()
    # The separately archived control snapshot remains source-fenced.
    archive = Path(case['recovery']['scheduler_snapshot_ref']['path']); archive.chmod(0o664)
    with pytest.raises(health.ExecutorHealthError, match='unsafe_or_oversized_file'):
        executor._recovery_scheduler(health._document(case['recovery']['request_ref']), case['recovery']['scheduler_snapshot_ref'])


def test_recovery_live_scheduler_world_writable_data_is_refused_before_ack(nonstarted_case):
    case = nonstarted_case; Path(case['scheduler_ref']['path']).chmod(0o666)
    with pytest.raises(health.ExecutorHealthError, match='unsafe_or_oversized_database_data'):
        _resume(case)
    assert not (case['failed'] / 'nonstarted-reconciliation.json').exists()
    assert (case['root'] / 'executor-held.json').read_bytes() == case['old_hold_bytes']


def test_recovery_live_scheduler_change_during_data_read_is_refused_before_ack(nonstarted_case, monkeypatch):
    case = nonstarted_case; live = Path(case['scheduler_ref']['path']); actual = health.database_descriptor
    calls = []
    def changed_after_first_descriptor(path):
        ref = actual(path)
        if Path(path) == live:
            calls.append(ref)
            if len(calls) == 1:
                value = json.loads(live.read_bytes()); value['non_authoritative_diagnostic'] = 'changed'
                live.write_bytes((json.dumps(value, sort_keys=True) + '\n').encode())
        return ref
    monkeypatch.setattr(health, 'database_descriptor', changed_after_first_descriptor)
    with pytest.raises(executor.ExecutorHold, match='recovery_scheduler_data_changed'):
        _resume(case)
    assert len(calls) == 1 and not (case['failed'] / 'nonstarted-reconciliation.json').exists()
    assert (case['root'] / 'executor-held.json').read_bytes() == case['old_hold_bytes']





@pytest.fixture
def closed_idle_case(nonstarted_case, monkeypatch):
    """Real private metadata and canonical _cycle parser; no engine or manager."""
    case = nonstarted_case; root = case['root']; archive = root / 'reconciliation-v02'; archive.mkdir(mode=0o700)
    config = case['config']; config['max_cycles'] = 10000
    config_ref = put(Path(case['config_ref']['path']), config)
    prior = copy.deepcopy(case['profile'])
    prior['source_descriptors'] = [config_ref if pin['path'] == config_ref['path'] else pin for pin in prior['source_descriptors']]
    argv = prior['unit']['argv']; argv[argv.index('--config-sha256') + 1] = config_ref['sha256']
    prior['cycle_contract'] = {'plan_schema': 'native-embedded-scoped-observation-plan/v1',
                               'plan_operation': 'native_embedded_scoped_observation_v1'}
    source_root = Path(prior['owner_source']['path']).parents[3]
    route = put(source_root / 'ipfs_accelerate_py/agent_implementation_route.py', b'# original compiled route fixture\n')
    observer = put(source_root / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/native_owner_bootstrap.py',
                   b'# original compiled read-only observer fixture\n')
    guard = put(source_root / 'ipfs_accelerate_py/agent_supervisor/rescue/project_import_source_guard.py', b'# prior resolution guard\n')
    datasets = root.parent / 'source-datasets'
    backend = put(datasets / 'ipfs_datasets_py/ipfs_backend_router.py', b'# real resolution-only metadata fixture\n')
    prior['source_descriptors'].extend([route, observer, guard, backend])
    prior_ref = put(Path(case['profile_ref']['path']), prior)
    prior_snapshot = put(archive / 'prior-profile.json', prior)
    old_main = dict(unit=prior['unit']['name'], pid=1000003, birth_ticks=3,
        boot_id='explicit-prior-closed-idle-fixture-boot', invocation_id='b' * 32,
        argv=list(prior['unit']['argv']), exe=prior['unit']['interpreter'], cwd=prior['unit']['working_directory'],
        environment={**prior['unit']['environment'], 'INVOCATION_ID': 'b' * 32, health.ENV_PROFILE_SHA: prior_ref['sha256']})
    old_auth = copy.deepcopy(case['recovery']); old_auth.update(profile_ref=prior_ref, max_cycles=10000)
    original_plan = health._document(old_auth['activation_plan_ref'])
    original_plan['main_loop_limits'] = {key: config[key] for key in ('max_cycles', 'poll_seconds', 'max_proof_bytes')}
    old_auth['activation_plan_ref'] = put(Path(old_auth['activation_plan_ref']['path']), original_plan)
    old_auth_ref = put(Path(case['recovery_ref']['path']), old_auth)
    old_ack = dict(schema='agent-supervisor/scientific-nonstarted-reconciliation@1',
        recovery_ref=old_auth_ref, hold_ref=old_auth['hold_ref'], failed_cycle_root=str(case['failed']),
        consuming_main_identity=old_main, health_credit=False, completion_authority=False, dispatch_authority=False)
    old_ack_ref = put(case['failed'] / 'nonstarted-reconciliation.json', old_ack)
    capture = dict(schema='scientific-explicit-known-nonstarted-recovery-actual-capture/v1',
        accepted=False, start_requested=True, source_leases_closed_before_start=True, hold_removed=False,
        native_or_task_or_epoch_or_policy_reset=False, recovery_authorization_ref=old_auth_ref, opening_identity=old_main)
    capture_ref = put(archive / 'prior-administrative-capture.json', capture)
    failed = root / ('cycle-00000002-' + '2' * 32); failed.mkdir(mode=0o700)
    attempt = failed / 'attempt'; (attempt / 'exports').mkdir(parents=True)
    now = time.time(); started = now - 4; finished = now - 1
    end_mono = time.monotonic_ns() - 10**9; start_mono = end_mono - 3 * 10**9
    native = copy.deepcopy(case['native']); native['checked_at_epoch'] = finished - 0.1
    task_rows = [{key: row[key] for key in ('task_cid', 'status', 'revision')} for row in native['tasks']]
    db = case['request']['database_preimage']
    binding = dict(schema='native-embedded-daemon-observation-binding@1', database_preimage=db,
        observer_source=observer, task_scope=prior['task_scope'], plan_root_cid=prior['catalogue']['plan_root_cid'],
        repository_tree_id=prior['catalogue']['repository_tree_id'],
        source_schema='ipfs_accelerate_py/agent-supervisor/database-task-source@1')
    plan = dict(schema=prior['cycle_contract']['plan_schema'], operation=prior['cycle_contract']['plan_operation'],
        accelerate_root=str(source_root), task_scope=prior['task_scope'], native_metadata_cli=dict(
            schema='native-embedded-scoped-observation-cli/v2', runtime_root=str(source_root),
            result_path=str(attempt / 'exports/result.json'), observation_binding=binding, pre_body_guard=guard,
            import_head_capsule_Git={'route_source': route}, project_import_plan=dict(datasets_root=str(datasets),
                source_pins=[route, observer, guard, backend]),
            source_sha256={pin['path']: pin['sha256'] for pin in prior['source_descriptors']}))
    plan_ref = put(failed / 'operation-plan.json', plan)
    request = dict(case['request'], request_id='fixture:closed-idle-cycle-2', attempt_directory=str(attempt),
        scheduler_state_path=case['scheduler_ref']['path'], expected_scheduler_policy_sha256=health._document(old_auth['request_ref'])['expected_scheduler_policy_sha256'],
        cpu_slots=1, memory_mb=1536, child_process_slots=8, storage_bytes=384 * 1024**2)
    request['source_sha256'] = {pin['path']: pin['sha256'] for pin in prior['source_descriptors']}
    request['source_sha256'][plan_ref['path']] = plan_ref['sha256']
    request['argv'] = [prior['unit']['argv'][0], '-B', prior['driver_source']['path'],
        '--plan', plan_ref['path'], '--plan-sha256', plan_ref['sha256']]
    request_ref = put(failed / 'request.json', request)
    put(failed / 'intent.json', dict(kind='genuine_scoped_idle', automatic_retry_admitted=False,
        sequence=1, cycle_number=2, request_ref=request_ref))
    execution_ref = put(attempt / 'execution-request.json', dict(schema='native-metadata-resource-owner-execution-request/v1',
        request=request_ref, owner_source=prior['owner_source'], disk_reservation_id='fixture:released-reservation',
        completion_authority=False, provider_authority=False))
    origins = dict(schema='legal-ir-pre-execution-project-import-origins/v1',
        project_origins_checked_before_module_execution=True, denials=[], project_imports=[
            dict(module='ipfs_accelerate_py.agent_implementation_route', **route,
                 execution_profile='compiled_exact_authenticated_source_bytes'),
            dict(module='ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap', **observer,
                 execution_profile='compiled_exact_authenticated_source_bytes'),
            dict(module='ipfs_datasets_py.ipfs_backend_router', **backend)])
    origins_ref = put(attempt / 'exports/origins.json', origins)
    result = dict(schema='native-embedded-daemon-observation@1', backend='embedded', database_path=db['path'],
        task_scope=prior['task_scope'], observer_source=observer, observer_callable='observe_embedded_database_without_dispatch',
        ready_task_count=0, selection_idle_reason='no_scoped_ready_tasks', unchanged=True, write_count=0,
        provider_dispatched=False, attempt_consumed=False, completion_authority=False,
        catalogue_binding=dict(database_preimage=db, event_cursor=native['event_cursor'], task_statuses=task_rows,
            plan_root_cid=binding['plan_root_cid'], repository_tree_id=binding['repository_tree_id'], source_schema=binding['source_schema']))
    result_ref = put(Path(plan['native_metadata_cli']['result_path']), result)
    native_ref = put(failed / 'native-diagnostic.json', native)
    owner_argv = [prior['unit']['argv'][0], '-B', prior['owner_source']['path'],
        '--request', request_ref['path'], '--expected-request-sha256', request_ref['sha256']]
    terminal = dict(schema='legal-ir-original-native-control-terminal/v1', request_sha256=request_ref['sha256'],
        execution_request=execution_ref, completed=True, source_READ_closed=True, original_native_compute_closed=True,
        child_launch_attempted=True, child_reaped=True, child_returncode=0, storage_status='released',
        cleanup_errors=[], received_signals=[], partial_or_failed_control_must_be_reconciled=False,
        owner_pid=1000004, owner_birth='4', elapsed_seconds=2.0, absolute_resource_operation_deadline_closed=True,
        control_effects='actor_exit_zero_independent_task_receipt_still_required',
        task_completion_authority_granted_by_resource_owner=False,
        database_preimage_before_grants=db, database_preimage_observed=db, database_postimage=db,
        reservation=dict(status='released', reservation_id='fixture:released-reservation', record=dict(
            status='released', reservation_id='fixture:released-reservation', owner_pid=1000004,
            native_lease_id='fixture:released-native', created_at=started + 0.5, released_at=finished - 0.25),
            resource_lease=dict(released=True, owner_pid=1000004, lease_id='fixture:released-native')),
        sampled_process_custody_final=dict(scope='sampled_birth_bound_descendants', live_identities=[],
            live_processes=0, rss_bytes=0, tracking_error='', observed_identities=[dict(pid=1000005, birth_ticks=5)]),
        process_spawn_receipts=dict(schema='native-metadata-stopped-process-receipts/v1', all_recorded_original_births_stopped=True))
    terminal_ref = put(failed / 'terminal.json', terminal); put(failed / 'owner.stdout', terminal)
    proof = dict(schema=health.CYCLE_SCHEMA, kind='genuine_scoped_idle', sequence=1,
        process={key: old_main[key] for key in ('pid', 'birth_ticks', 'boot_id', 'invocation_id')},
        owner_observation=dict(pid=1000004, birth_ticks=4, boot_id=old_main['boot_id'], argv=owner_argv,
            captured_at=started + 0.25, captured_monotonic_ns=start_mono + 250000000,
            ancestry=[dict(pid=1000004, birth_ticks=4, parent_pid=1000003), dict(pid=1000003, birth_ticks=3, parent_pid=1)]),
        started_at=started, finished_at=finished, started_monotonic_ns=start_mono, finished_monotonic_ns=end_mono,
        request_ref=request_ref, execution_request_ref=execution_ref, terminal_ref=terminal_ref, result_ref=result_ref,
        plan_ref=plan_ref, origins_ref=origins_ref, native_diagnostic_ref=native_ref)
    proof_ref = put(failed / 'cycle-proof.json', proof)
    cycle_hold_ref = put(failed / 'cycle-held.json', dict(schema='agent-supervisor/scientific-cycle-held@1',
        reason='original_import_origin_unbound', automatic_retry_admitted=False, recorded_at=finished + 0.1))
    negative = dict(schema='scientific-native-completed-idle-health-failure-independent-negative-audit/v1',
        actual_administrative_capture=capture_ref, actual_profile_ref=prior_ref, authorization_ref=old_auth_ref,
        negative_ack_ref=old_ack_ref, accepted_as_executor_health_or_activation=False,
        completed_native_read_only_cycle_independently_verified=True, native_source_process_storage_closed=True,
        task_or_provider_or_completion_effects=False, database_unchanged_during_original_cycle=True,
        exact_original_cycle_number=2, health_sequence_attempted=1, actual_origin_rows=3, compiled_authenticated_origin_rows=2,
        exact_unlabeled_origin=origins['project_imports'][-1], global_original_hold_ref=old_auth['hold_ref'],
        new_cycle_hold_ref=cycle_hold_ref, cycle_proof_ref=proof_ref,
        **{key: proof[key] for key in ('request_ref', 'execution_request_ref', 'terminal_ref', 'result_ref', 'plan_ref', 'origins_ref', 'native_diagnostic_ref')})
    negative_ref = put(archive / 'closed-idle-negative.json', negative)
    # Adopt a prospective new source generation, keeping the old profile as data.
    profile = copy.deepcopy(prior)
    for role in ('main_entrypoint', 'owner_source', 'driver_source'):
        old = profile[role]; new = put(Path(old['path']), b'# explicitly adopted replacement source fixture\n')
        profile[role] = new; profile['source_descriptors'] = [new if pin == old else pin for pin in profile['source_descriptors']]
    new_guard = put(Path(guard['path']), b'# adopted explicit resolution-only lifecycle guard\n')
    profile['source_descriptors'] = [new_guard if pin == guard else pin for pin in profile['source_descriptors']]
    future_plan = copy.deepcopy(plan); future_plan['native_metadata_cli']['pre_body_guard'] = new_guard
    future_plan_ref = put(Path(case['template']['plan_ref']['path']), future_plan)
    template = dict(case['template'], plan_ref=future_plan_ref)
    template_ref = put(Path(case['template_ref']['path']), template); config['idle_template_ref'] = template_ref
    config_ref = put(Path(case['config_ref']['path']), config)
    profile['source_descriptors'] = [config_ref if pin['path'] == config_ref['path'] else pin for pin in profile['source_descriptors']]
    new_auth_path = archive / 'known-closed-idle-authorization.json'; argv = profile['unit']['argv']
    index = argv.index('--known-nonstarted-recovery'); argv[index:index + 2] = ['--known-closed-idle-recovery', str(new_auth_path)]
    argv[argv.index('--config-sha256') + 1] = config_ref['sha256']
    profile_ref = put(Path(prior_ref['path']), profile)
    selected = [pin for pin in profile['source_descriptors'] if Path(pin['path']).is_relative_to(source_root)
                and Path(pin['path']).suffix == '.py']
    inventory = dict(schema='legal-ir-scientific-source-inventory/v1', source_root=str(source_root),
        source_commit='e' * 40, source_tree='f' * 40, file_count=len(selected), files=selected)
    inventory_ref = put(archive / 'current-source-inventory.json', inventory)
    commitment_ref = put(archive / 'current-source-commitment.json', dict(schema='legal-ir-scientific-source-commitment/v1',
        source_root=str(source_root), source_commit='e' * 40, source_tree='f' * 40,
        source_inventory=inventory_ref, runtime_activation_or_task_authority=False))
    current = copy.deepcopy(native); current['checked_at_epoch'] = time.time()
    value = dict(schema=executor.CLOSED_IDLE_SCHEMA, recovery_kind='known_closed_idle_health_rejection',
        profile_ref=profile_ref, prior_profile_ref=prior_ref, prior_profile_snapshot_ref=prior_snapshot,
        prior_authorization_ref=old_auth_ref, prior_ack_ref=old_ack_ref, prior_administrative_capture_ref=capture_ref,
        negative_audit_ref=negative_ref, source_commitment_ref=commitment_ref,
        scheduler_snapshot_ref=put(archive / 'scheduler.snapshot.json', case['scheduler_state']),
        current_native_ref=put(archive / 'current-native.json', current), global_hold_ref=old_auth['hold_ref'],
        cycle_hold_ref=cycle_hold_ref, cycle_proof_ref=proof_ref, max_cycles=10000,
        authorization='resume_genuine_scoped_idle', recorded_at=time.time(),
        manager_before=dict(Id=profile['unit']['name'], MainPID='0', ActiveState='failed'))
    recovery_ref = put(new_auth_path, value); monkeypatch.setenv(executor.CLOSED_IDLE_SHA_ENV, recovery_ref['sha256'])
    opening = dict(pid=1000006, birth_ticks=6, boot_id='explicit-next-fixture-boot', invocation_id='c' * 32)
    monkeypatch.setattr(executor, 'require_main_process', lambda *args: opening)
    case.update(profile=profile, profile_ref=profile_ref, config=config, config_ref=config_ref,
        closed_value=value, closed_ref=recovery_ref, closed_cycle=failed, closed_proof=proof,
        closed_negative=negative, closed_terminal=terminal, closed_result=result, closed_origins=origins,
        closed_native=current, prior_closed_profile=prior, prior_closed_main=old_main,
        closed_archive=archive, opening=opening,
        preserved_closed_history={ref['path']: Path(ref['path']).read_bytes() for ref in
            (old_auth_ref, old_ack_ref, old_auth['hold_ref'], cycle_hold_ref, proof_ref, origins_ref)})
    return case


def _closed_resume(case):
    return executor.reconcile_known_closed_idle(case['closed_ref'], case['profile_ref'],
        case['profile'], case['config'], case['opening'])


def _resign_closed(case, monkeypatch):
    case['closed_ref'] = put(Path(case['closed_ref']['path']), case['closed_value'])
    monkeypatch.setenv(executor.CLOSED_IDLE_SHA_ENV, case['closed_ref']['sha256'])


def _repin_closed_artifact(case, monkeypatch, key, value):
    proof = case['closed_proof']; proof[key] = put(Path(proof[key]['path']), value)
    if key == 'terminal_ref': put(case['closed_cycle'] / 'owner.stdout', value)
    case['closed_negative'][key] = proof[key]
    case['closed_value']['cycle_proof_ref'] = put(Path(case['closed_value']['cycle_proof_ref']['path']), proof)
    case['closed_negative']['cycle_proof_ref'] = case['closed_value']['cycle_proof_ref']
    case['closed_value']['negative_audit_ref'] = put(Path(case['closed_value']['negative_audit_ref']['path']), case['closed_negative'])
    _resign_closed(case, monkeypatch)


def test_closed_idle_negative_ack_uses_real_original_cycle_checks_and_preserves_history(closed_idle_case):
    case = closed_idle_case; assert _closed_resume(case) == 2
    ack = case['closed_cycle'] / 'closed-idle-reconciliation-v01.json'; first = ack.read_bytes()
    assert _closed_resume(case) == 2 and ack.read_bytes() == first
    value = json.loads(first)
    assert value['spent_cycle_count'] == 2 and value['next_cycle_number'] == 3 and value['first_health_sequence'] == 1
    assert all(value[key] is False for key in ('health_credit', 'completion_authority', 'dispatch_authority', 'historical_origins_relabelled'))
    assert all(Path(path).read_bytes() == raw for path, raw in case['preserved_closed_history'].items())
    assert not (case['closed_cycle'] / 'closed.json').exists() and not Path(case['profile']['receipt_path']).exists()
    assert 'execution_profile' not in case['closed_origins']['project_imports'][-1]


@pytest.mark.parametrize('field,value', [('completed', False), ('child_returncode', 1), ('storage_status', 'retained'),
    ('source_READ_closed', False), ('original_native_compute_closed', False), ('cleanup_errors', ['unknown']),
    ('received_signals', [15]), ('absolute_resource_operation_deadline_closed', False)],
    ids=['not-completed', 'child-failed', 'storage-retained', 'source-open', 'native-open', 'cleanup-unknown', 'signal', 'deadline-open'])
def test_closed_idle_refuses_incomplete_original_terminal_before_ack(closed_idle_case, monkeypatch, field, value):
    case = closed_idle_case; case['closed_terminal'][field] = value
    _repin_closed_artifact(case, monkeypatch, 'terminal_ref', case['closed_terminal'])
    with pytest.raises((executor.ExecutorHold, health.ExecutorHealthError)): _closed_resume(case)
    assert not (case['closed_cycle'] / 'closed-idle-reconciliation-v01.json').exists()


@pytest.mark.parametrize('field,value', [('write_count', 1), ('provider_dispatched', True), ('attempt_consumed', True),
    ('completion_authority', True), ('ready_task_count', 1), ('unchanged', False)],
    ids=['write', 'provider', 'coding-attempt', 'completion', 'ready', 'changed'])
def test_closed_idle_refuses_nonquiet_or_effectful_result_before_ack(closed_idle_case, monkeypatch, field, value):
    case = closed_idle_case; case['closed_result'][field] = value
    _repin_closed_artifact(case, monkeypatch, 'result_ref', case['closed_result'])
    with pytest.raises(executor.ExecutorHold, match='outcome_not_readonly'): _closed_resume(case)
    assert not (case['closed_cycle'] / 'closed-idle-reconciliation-v01.json').exists()


@pytest.mark.parametrize('mutation', ['another-unlabelled', 'wrong-module', 'failed-label', 'wrong-hash'],
    ids=['another-unlabelled', 'wrong-module', 'failed-label', 'wrong-hash'])
def test_closed_idle_ambiguous_origins_are_not_normalized_into_health(closed_idle_case, monkeypatch, mutation):
    case = closed_idle_case; rows = case['closed_origins']['project_imports']
    if mutation == 'another-unlabelled': rows[0].pop('execution_profile')
    elif mutation == 'wrong-module': rows[-1]['module'] = 'ipfs_datasets_py.other'
    elif mutation == 'failed-label': rows[-1]['execution_profile'] = 'failed'
    else: rows[-1]['sha256'] = 'f' * 64
    _repin_closed_artifact(case, monkeypatch, 'origins_ref', case['closed_origins'])
    with pytest.raises((executor.ExecutorHold, health.ExecutorHealthError)): _closed_resume(case)
    assert not (case['closed_cycle'] / 'closed-idle-reconciliation-v01.json').exists()
    assert not Path(case['profile']['receipt_path']).exists()


def test_closed_idle_database_change_is_refused_without_opening_an_engine(closed_idle_case):
    case = closed_idle_case; db = Path(case['request']['database_preimage']['path']); db.write_bytes(b'changed-data')
    with pytest.raises(executor.ExecutorHold, match='database_history_or_current_changed'): _closed_resume(case)
    assert not (case['closed_cycle'] / 'closed-idle-reconciliation-v01.json').exists()


def test_closed_idle_current_native_revision_change_is_refused(closed_idle_case, monkeypatch):
    case = closed_idle_case; native = case['closed_native']; native['tasks'][0]['revision'] += 1
    case['closed_value']['current_native_ref'] = put(Path(case['closed_value']['current_native_ref']['path']), native)
    _resign_closed(case, monkeypatch)
    with pytest.raises(executor.ExecutorHold, match='current_tasks_changed'): _closed_resume(case)


@pytest.mark.parametrize('family', ['first', 'second'], ids=['first', 'second'])
def test_closed_idle_residual_native_family_prevents_ack(closed_idle_case, monkeypatch, family):
    case = closed_idle_case
    request = health._document(case['recovery']['request_ref']) if family == 'first' else health._document(case['closed_proof']['request_ref'])
    state = case['scheduler_state']; state['waiters']['private-owned'] = dict(request_id=request['request_id'] + ':root')
    put(Path(case['scheduler_ref']['path']), state)
    with pytest.raises(executor.ExecutorHold, match='recovery_original_request_still_active'): _closed_resume(case)


@pytest.mark.parametrize('pid', [1000003, 1000004, 1000005], ids=['main', 'owner', 'descendant'])
def test_closed_idle_old_main_or_descendant_alive_refuses_ack(closed_idle_case, monkeypatch, pid):
    case = closed_idle_case
    monkeypatch.setattr(health, '_definitely_dead_identity', lambda old, current: old['pid'] != pid)
    with pytest.raises(executor.ExecutorHold, match='prior_.*_alive'): _closed_resume(case)


def test_closed_idle_consumed_authorization_cannot_bind_another_main(closed_idle_case):
    case = closed_idle_case; assert _closed_resume(case) == 2
    case['opening'] = dict(case['opening'], pid=1000007, birth_ticks=7)
    with pytest.raises(executor.ExecutorHold, match='consumed_by_other_main'): _closed_resume(case)


def test_closed_idle_admin_is_metadata_only_and_new_authorization_is_exclusive(closed_idle_case, monkeypatch):
    case = closed_idle_case; target = Path(case['closed_ref']['path']); target.unlink()
    spec = {key: value for key, value in case['closed_value'].items() if key not in ('schema', 'recorded_at', 'manager_before')}
    spec_ref = put(case['closed_archive'] / 'authorization-spec.json', spec)
    monkeypatch.setattr(health, '_manager', lambda unit: dict(Id=unit, MainPID='0', ActiveState='failed'))
    monkeypatch.setattr(executor.subprocess, 'Popen', lambda *a, **k: pytest.fail('admin cannot spawn'))
    created = executor.authorize_known_closed_idle_recovery(spec_ref, target)
    assert health._document(created)['schema'] == executor.CLOSED_IDLE_SCHEMA
    with pytest.raises(FileExistsError): executor.authorize_known_closed_idle_recovery(spec_ref, target)
    assert not (case['closed_cycle'] / 'closed-idle-reconciliation-v01.json').exists()
    assert all(Path(path).read_bytes() == raw for path, raw in case['preserved_closed_history'].items())


def test_closed_idle_new_unit_pin_is_required_before_any_ack(closed_idle_case, monkeypatch):
    case = closed_idle_case; monkeypatch.delenv(executor.CLOSED_IDLE_SHA_ENV)
    with pytest.raises(executor.ExecutorHold, match='receipt_not_unit_pinned'): _closed_resume(case)
    assert not (case['closed_cycle'] / 'closed-idle-reconciliation-v01.json').exists()


def test_closed_idle_result_revision_float_cannot_equal_a_real_integer(closed_idle_case, monkeypatch):
    case = closed_idle_case; case['closed_result']['catalogue_binding']['task_statuses'][0]['revision'] = 2.0
    _repin_closed_artifact(case, monkeypatch, 'result_ref', case['closed_result'])
    with pytest.raises(executor.ExecutorHold, match='catalogue_tasks_changed'): _closed_resume(case)


def test_closed_idle_result_cursor_float_cannot_equal_a_real_integer(closed_idle_case, monkeypatch):
    case = closed_idle_case; case['closed_result']['catalogue_binding']['event_cursor'] = 5.0
    _repin_closed_artifact(case, monkeypatch, 'result_ref', case['closed_result'])
    with pytest.raises(executor.ExecutorHold, match='catalogue_binding'): _closed_resume(case)


def test_closed_idle_admin_refuses_running_unit_without_changing_old_authorization(closed_idle_case, monkeypatch):
    case = closed_idle_case
    spec = {key: value for key, value in case['closed_value'].items() if key not in ('schema', 'recorded_at', 'manager_before')}
    spec_ref = put(case['closed_archive'] / 'running-authorization-spec.json', spec)
    monkeypatch.setattr(health, '_manager', lambda unit: dict(Id=unit, MainPID='12345', ActiveState='active'))
    target = Path(case['closed_ref']['path']); target.unlink()
    with pytest.raises(executor.ExecutorHold, match='requires_no_main'):
        executor.authorize_known_closed_idle_recovery(spec_ref, target)
    assert not target.exists()
    assert all(Path(path).read_bytes() == raw for path, raw in case['preserved_closed_history'].items())


def test_closed_idle_resume_starts_cycle_three_sequence_one_and_retains_two_spent(closed_idle_case, monkeypatch):
    case = closed_idle_case; stop = threading.Event(); seen = []
    monkeypatch.setattr(executor, 'observe_native', lambda config: case['closed_native'])
    def run(*args, **kwargs): seen.append((args[5], kwargs['cycle_number'])); stop.set()
    monkeypatch.setattr(executor, 'run_cycle', run)
    assert executor.serve(case['config_ref'], stop, recovery_ref=case['closed_ref']) == 0
    assert seen == [(1, 3)] and case['config']['max_cycles'] - 2 == 9998
    assert not (case['closed_cycle'] / 'closed.json').exists()
    assert all(Path(path).read_bytes() == raw for path, raw in case['preserved_closed_history'].items())


def test_closed_idle_and_nonstarted_cli_cannot_be_combined():
    with pytest.raises(SystemExit) as stopped:
        executor.main(['serve', '--config', '/private-not-opened.json', '--config-sha256', 'a' * 64,
            '--known-nonstarted-recovery', '/private-not-opened-old.json', '--known-closed-idle-recovery', '/private-not-opened-new.json'])
    assert stopped.value.code == 2



@pytest.mark.parametrize('field', ['poll_seconds', 'max_proof_bytes'], ids=['poll', 'proof-bytes'])
def test_closed_idle_original_poll_and_proof_budget_cannot_be_reset(closed_idle_case, field):
    case = closed_idle_case; case['config'][field] += 1
    with pytest.raises(executor.ExecutorHold, match='original_budget_changed'): _closed_resume(case)
    assert not (case['closed_cycle'] / 'closed-idle-reconciliation-v01.json').exists()
