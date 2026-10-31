"""Prospective source-only protocol tests, never scientific activation proof.

Injected manager/process observations below are explicit unit-test fixtures.
All receipt/profile/source bytes are ordinary private files. A passing cohort
cannot establish a live executor, native task admission, completion, training,
legal meaning or Lean acceptance.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as health


NOW = 1_000.0
PID = 271_828
BIRTH = 12_345
BOOT = '12345678-1234-1234-1234-123456789abc'
INVOCATION = 'a' * 32
TASK = 'baguqeerabt7pfpb5p6nuzmftacfjymnruurxqdsitliniynbxcf4na6ueyia'
UNIT = 'ipfs-taskboard-legal-ir-native-executor.service'


def _wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _ref(path):
    raw = Path(path).read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else _wire(value)
    path.write_bytes(raw)
    path.chmod(0o400)
    return _ref(path)


def _replace(path, value):
    path = Path(path)
    path.chmod(0o600)
    return _put(path, value)


def _blocked_native():
    return {'schema': 'native-catalogue-diagnostic/v1',
        'board_namespace': 'legal-ir-scientific-fixture', 'plan_root_cid': 'plan:fixture',
        'repository_tree_id': 'tree:fixture', 'population_sha256': 'b' * 64,
        'database_uuid': 'database:fixture', 'store_generation': 3,
        'event_cursor': 253, 'fresh_read': True, 'checked_at_epoch': NOW,
        'probe_observed_at_epoch': NOW, 'counts': {'blocked': 1},
        'task_count': 1, 'completion_authority': False,
        'database_and_wal_metadata_unchanged': True,
        'tasks': [{'task_cid': TASK, 'task_alias': 'LIR-0102', 'status': 'blocked', 'revision': 7}]}


def _base_profile(tmp_path):
    """Build the typed profile; the genuine-cycle fixture is supplied below."""
    sources = []
    roles = {}
    for name in ('main_entrypoint', 'owner_source', 'driver_source'):
        roles[name] = _put(tmp_path / 'source' / (name + '.py'), ('# Explicit source-only ' + name + ' fixture\n').encode())
        sources.append(roles[name])
    roles['health_module'] = _ref(Path(health.__file__).resolve())
    sources.append(roles['health_module'])
    interpreter = str(Path(sys.executable).resolve())
    sources.append(_ref(Path(interpreter)))
    profile = {'schema': 'agent-supervisor/native-executor-health-profile@1',
        'catalogue': {key: _blocked_native()[key] for key in
            ('board_namespace', 'plan_root_cid', 'repository_tree_id', 'population_sha256', 'database_uuid', 'store_generation')},
        'unit': {'name': UNIT, 'argv': [interpreter, '-B', roles['main_entrypoint']['path']],
            'interpreter': interpreter, 'working_directory': str(tmp_path),
            'environment': {'PYTHONDONTWRITEBYTECODE': '1'}},
        'receipt_path': str(tmp_path / 'proof' / 'current-cycle.json'), 'source_descriptors': sources,
        **roles, 'task_scope': [TASK],
        'max_age_seconds': 120, 'max_cycle_seconds': 300,
        'proof_root': str(tmp_path / 'proof'),
        'cycle_contract': {'plan_schema': 'native-metadata-original-cold-execution-plan-template/v15',
                           'plan_operation': 'native_metadata_signed_execution_0102_v1'}}
    manager = {'Id': UNIT, 'MainPID': str(PID), 'ActiveState': 'active', 'SubState': 'running',
        'InvocationID': INVOCATION, 'ExecMainStartTimestampMonotonic': '123450000',
        'ControlGroup': '/user.slice/fixture-native-executor.service'}
    process = {'pid': PID, 'birth_ticks': BIRTH, 'boot_id': BOOT,
        'argv': list(profile['unit']['argv']), 'exe': interpreter, 'cwd': str(tmp_path),
        'environment': {'PYTHONDONTWRITEBYTECODE': '1', 'INVOCATION_ID': INVOCATION}}
    return {'profile': profile, 'manager': manager, 'process': process,
            'profile_path': tmp_path / 'profile.json', 'native': _blocked_native()}


def _publish_profile(case):
    ref = _replace(case['profile_path'], case['profile']) if case['profile_path'].exists() else _put(case['profile_path'], case['profile'])
    case['process']['environment']['IPFS_NATIVE_EXECUTOR_PROFILE_SHA256'] = ref['sha256']
    return ref


def _verify(case, profile_ref=None, manager_reader=None, process_reader=None):
    profile_ref = _publish_profile(case) if profile_ref is None else profile_ref
    def manager(unit):
        assert unit == case['profile']['unit']['name']
        return copy.deepcopy(case['manager'])
    def process(pid):
        assert pid == PID
        return copy.deepcopy(case['process'])
    return health.verify_bound_executor(profile_ref, case['native'], now=NOW,
        manager_reader=manager if manager_reader is None else manager_reader,
        process_reader=process if process_reader is None else process_reader)


@pytest.mark.parametrize('descriptor', [None, {}, [], True,
    {'path': '/absent-profile.json', 'bytes': True, 'sha256': 'a' * 64},
    {'path': '/absent-profile.json', 'bytes': 1, 'sha256': 'x' * 64}])
def test_invalid_profile_descriptor_never_reads_manager_or_process(descriptor):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid immutable profile cannot reach manager/process observation')
    result = health.verify_bound_executor(descriptor, _blocked_native(), now=NOW,
        manager_reader=forbidden, process_reader=forbidden)
    assert result['verified'] is False
    assert result['scope'] == 'none'
    assert isinstance(result['reason'], str) and result['reason']


def test_missing_profile_is_unknown_without_creating_it(tmp_path):
    path = tmp_path / 'absent-profile.json'
    ref = {'path': str(path), 'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}
    def forbidden(*args, **kwargs):
        pytest.fail('missing profile must stop before manager/process IO')
    result = health.verify_bound_executor(ref, _blocked_native(), now=NOW,
        manager_reader=forbidden, process_reader=forbidden)
    assert result['verified'] is False and result['scope'] == 'none'
    assert not path.exists()


@pytest.mark.parametrize('mutation', ['schema', 'extra', 'age_boolean', 'age_zero', 'cycle_unbounded',
                                    'unknown_unit', 'missing_role_source'])
def test_closed_profile_rejects_changes_before_manager_observation(tmp_path, mutation):
    case = _base_profile(tmp_path)
    value = case['profile']
    if mutation == 'schema': value['schema'] = 'agent-report-with-no-authority'
    elif mutation == 'extra': value['activation_authority'] = True
    elif mutation == 'age_boolean': value['max_age_seconds'] = True
    elif mutation == 'age_zero': value['max_age_seconds'] = 0
    elif mutation == 'cycle_unbounded': value['max_cycle_seconds'] = 1_000_000
    elif mutation == 'unknown_unit': value['unit']['name'] = '--all.service'
    elif mutation == 'missing_role_source': value['source_descriptors'] = value['source_descriptors'][:-1]
    def forbidden(*args, **kwargs):
        pytest.fail('invalid closed profile must stop before manager observation')
    result = _verify(case, manager_reader=forbidden, process_reader=forbidden)
    assert result['verified'] is False and result['scope'] == 'none'


def test_changed_selected_source_refuses_before_live_observation(tmp_path):
    case = _base_profile(tmp_path)
    source = Path(case['profile']['owner_source']['path'])
    source.chmod(0o600)
    source.write_text('# Changed after original source selection\n')
    def forbidden(*args, **kwargs):
        pytest.fail('changed source must stop before manager observation')
    result = _verify(case, manager_reader=forbidden, process_reader=forbidden)
    assert result['verified'] is False and result['scope'] == 'none'


def test_heartbeat_only_cannot_substitute_for_genuine_native_cycle(tmp_path):
    case = _base_profile(tmp_path)
    _put(Path(case['profile']['receipt_path']), {'schema': 'native-typed-owner-live-observation@1',
        'sequence': 5, 'owner_read_succeeded': True, 'observed_at_ms': int(NOW * 1000),
        'process_birth': {'pid': PID, 'birth_ticks': BIRTH}, 'completion_authority': False})
    result = _verify(case)
    assert result['verified'] is False and result['scope'] == 'none'


def _document(case, name, value):
    path = case['root'] / 'proof' / (name + '.json')
    return _replace(path, value) if path.exists() else _put(path, value)


def _identity_fixture(case, profile_ref):
    """Expected identity is independent of later injected-reader mutations."""
    return {'unit': UNIT, 'pid': PID, 'birth_ticks': BIRTH, 'boot_id': BOOT,
        'invocation_id': INVOCATION, 'argv': list(case['profile']['unit']['argv']),
        'exe': case['profile']['unit']['interpreter'], 'cwd': str(case['root']),
        'environment': {'PYTHONDONTWRITEBYTECODE': '1', 'INVOCATION_ID': INVOCATION,
                        'IPFS_NATIVE_EXECUTOR_PROFILE_SHA256': profile_ref['sha256']}}


def _write_evidence(case):
    """Write explicit fixture links without pretending they are actual runs."""
    profile_ref = _publish_profile(case)
    identity = _identity_fixture(case, profile_ref)
    plan = copy.deepcopy(case['plan'])
    plan['native_metadata_cli']['result_path'] = str(case['root'] / 'proof/original-result.json')
    plan_ref = _document(case, 'original-plan', plan)
    request = copy.deepcopy(case['request'])
    request['argv'] = [case['profile']['unit']['interpreter'], '-B',
        case['profile']['driver_source']['path'], '--plan', plan_ref['path'], '--plan-sha256', plan_ref['sha256']]
    request_ref = _document(case, 'original-request', request)
    execution = copy.deepcopy(case['execution'])
    execution['request'] = request_ref
    execution_ref = _document(case, 'original-execution-request', execution)
    terminal = copy.deepcopy(case['terminal'])
    terminal.update(request_sha256=request_ref['sha256'], execution_request=execution_ref)
    terminal_ref = _document(case, 'original-terminal', terminal)
    refs = {'plan_ref': plan_ref, 'request_ref': request_ref, 'execution_request_ref': execution_ref,
        'terminal_ref': terminal_ref, 'result_ref': _document(case, 'original-result', case['result']),
        'origins_ref': _document(case, 'original-origins', case['origins']),
        'native_diagnostic_ref': _document(case, 'original-native', case['cycle_native'])}
    proof = copy.deepcopy(case['proof'])
    proof.update(refs)
    observation = copy.deepcopy(case['owner_observation'])
    observation['argv'] = [case['profile']['unit']['argv'][0], '-B',
        case['profile']['owner_source']['path'], '--request', request_ref['path'],
        '--expected-request-sha256', request_ref['sha256']]
    proof['owner_observation'] = observation
    proof_ref = _document(case, 'original-cycle-links', proof)
    receipt = copy.deepcopy(case['receipt'])
    receipt.update(profile_ref=profile_ref, identity=identity, cycle_proof_ref=proof_ref)
    path = Path(case['profile']['receipt_path'])
    _replace(path, receipt) if path.exists() else _put(path, receipt)
    case['profile_ref'], case['written_identity'] = profile_ref, identity
    return profile_ref


@pytest.fixture
def cycle_case(tmp_path, monkeypatch):
    """Genuine-cycle SHAPE fixture; no process or native executor is started."""
    case = _base_profile(tmp_path)
    case['root'] = tmp_path
    mono = 10**15
    monkeypatch.setattr(health.time, 'monotonic_ns', lambda: mono)
    pid_owner, owner_birth, reservation = 314_159, '99999000', 'f' * 32
    case['plan'] = {'schema': case['profile']['cycle_contract']['plan_schema'],
        'operation': case['profile']['cycle_contract']['plan_operation'],
        'task_cid': TASK, 'expected_ready_revision': 4, 'task_scope': [TASK],
        'native_metadata_cli': {'task_cid': TASK, 'expected_revision': 4,
            'cli_argv': [case['profile']['unit']['interpreter'], '-B', '-m',
                'ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon',
                '--once', '--implement', '--native-metadata-execution-task-cid', TASK,
                '--execution-slice-task-cid', TASK, '--native-metadata-execution-task-revision', '4']}}
    case['request'] = {'schema': 'legal-ir-original-native-control-request/v1',
        'purpose': 'native_control_metadata', 'source_sha256':
            {pin['path']: pin['sha256'] for pin in case['profile']['source_descriptors']}}
    case['execution'] = {'schema': 'native-metadata-resource-owner-execution-request/v1',
        'owner_source': case['profile']['owner_source'], 'disk_reservation_id': reservation}
    case['terminal'] = {'schema': 'legal-ir-original-native-control-terminal/v1',
        'completed': True, 'source_READ_closed': True, 'original_native_compute_closed': True,
        'child_reaped': True, 'child_returncode': 0, 'storage_status': 'released',
        'cleanup_errors': [], 'received_signals': [], 'partial_or_failed_control_must_be_reconciled': False,
        'owner_pid': pid_owner, 'owner_birth': owner_birth, 'elapsed_seconds': 8,
        'reservation': {'status': 'released', 'reservation_id': reservation,
            'record': {'status': 'released', 'reservation_id': reservation,
                       'owner_pid': pid_owner, 'owner_birth': owner_birth,
                       'native_lease_id': 'd' * 32, 'created_at': NOW - 9, 'released_at': NOW - 3},
            'resource_lease': {'released': True, 'owner_pid': pid_owner, 'lease_id': 'd' * 32}},
        'sampled_process_custody_final': {'scope': 'sampled_birth_bound_descendants',
            'live_identities': [], 'live_processes': 0, 'rss_bytes': 0, 'tracking_error': ''},
        'process_spawn_receipts': {'schema': 'native-metadata-stopped-process-receipts/v1',
            'all_recorded_original_births_stopped': True}}
    case['result'] = {'schema': 'native-metadata-cold-cli-result/v1', 'completed': True,
        'original_cli_calls': 1, 'original_main_returncode': 0, 'error': None,
        'provider_authority': False, 'completion_authority': False}
    case['origins'] = {'schema': 'legal-ir-pre-execution-project-import-origins/v1',
        'project_origins_checked_before_module_execution': True, 'denials': [],
        'all_reads_enforced_or_attested': False, 'guard_propagated_to_exec_subprocess': False,
        'project_imports': [{'module': 'fixture.original_driver', **case['profile']['driver_source'],
            'execution_profile': 'compiled_exact_authenticated_source_bytes'}]}
    case['cycle_native'] = copy.deepcopy(case['native'])
    case['cycle_native']['checked_at_epoch'] = NOW - 3
    case['cycle_native']['probe_observed_at_epoch'] = NOW - 3
    case['owner_observation'] = {'pid': pid_owner, 'birth_ticks': int(owner_birth),
        'boot_id': BOOT, 'captured_at': NOW - 8, 'captured_monotonic_ns': mono - 8 * 10**9,
        'ancestry': [{'pid': pid_owner, 'birth_ticks': int(owner_birth), 'parent_pid': PID},
                     {'pid': PID, 'birth_ticks': BIRTH, 'parent_pid': 1}]}
    case['proof'] = {'schema': 'agent-supervisor/native-executor-original-cycle-links@1',
        'kind': 'native_execution', 'sequence': 1, 'started_at': NOW - 10,
        'finished_at': NOW - 2, 'started_monotonic_ns': mono - 10 * 10**9,
        'finished_monotonic_ns': mono - 2 * 10**9,
        'process': {'pid': PID, 'birth_ticks': BIRTH, 'boot_id': BOOT, 'invocation_id': INVOCATION}}
    case['receipt'] = {'schema': 'agent-supervisor/native-executor-main-loop-health@1',
        'sequence': 1, 'observed_at': NOW - 1, 'observed_monotonic_ns': mono - 10**9,
        'catalogue': {**case['profile']['catalogue'], 'event_cursor': 253, 'task_scope': [TASK]},
        'origins': {'__main__': case['profile']['main_entrypoint'],
                    'fixture.health': case['profile']['health_module']},
        'previous_receipt_sha256': None}
    _write_evidence(case)
    return case


def test_bound_original_execution_shape_verifies_without_granting_task_authority(cycle_case):
    before = copy.deepcopy(cycle_case['native'])
    result = _verify(cycle_case)
    assert result['verified'] is True and result['scope'] == 'native_execution'
    assert result.get('completion_authority') is False
    assert result.get('dispatch_authority') is False
    assert result.get('complete', False) is False
    assert cycle_case['native'] == before


def test_bound_idle_manual_blocked_shape_remains_incomplete(cycle_case):
    case = cycle_case
    case['proof']['kind'] = 'genuine_scoped_idle'
    case['result'] = {'schema': 'native-typed-daemon-observation@1', 'ready_task_count': 0,
        'selection_idle_reason': 'manual_review_or_dependency_wait', 'unchanged': True,
        'write_count': 0, 'provider_dispatched': False, 'attempt_consumed': False,
        'completion_authority': False}
    _write_evidence(case)
    before = copy.deepcopy(case['native'])
    result = _verify(case)
    assert result['verified'] is True and result['scope'] == 'genuine_scoped_idle'
    assert result.get('completion_authority') is False and result.get('dispatch_authority') is False
    assert result.get('complete', False) is False
    assert case['native'] == before and case['native']['counts'] == {'blocked': 1}


def _embedded_idle_shape(case):
    """Explicit unit fixture, never evidence of an actual daemon observation."""
    case['proof']['kind'] = 'genuine_scoped_idle'
    observer = _put(case['root'] / 'source/native_owner_bootstrap.py', b'# synthetic observer source fixture\n')
    case['profile']['source_descriptors'].append(observer)
    case['request']['source_sha256'][observer['path']] = observer['sha256']
    route=_put(case['root']/'ipfs_accelerate_py/agent_implementation_route.py',b'# explicit source-only route fixture\n')
    case['profile']['source_descriptors'].append(route)
    case['request']['source_sha256'][route['path']]=route['sha256']
    case['origins']['project_imports'].append({'module':'ipfs_accelerate_py.agent_implementation_route',
        **route,'execution_profile':'compiled_exact_authenticated_source_bytes'})
    db = {'path': str(case['root'] / 'fixture.duckdb'), 'bytes': 8, 'sha256': 'd' * 64}
    case['request']['database_preimage'] = db
    case['request']['external_files'] = [{'path': db['path'], 'max_bytes': 128 * 1024**2}]
    binding = {'schema': 'native-embedded-daemon-observation-binding@1',
        'database_preimage': db, 'task_scope': [TASK], 'plan_root_cid': 'plan:fixture',
        'repository_tree_id': 'tree:fixture', 'observer_source': observer,
        'source_schema': 'ipfs_accelerate_py/agent-supervisor/database-task-source@1'}
    case['profile']['cycle_contract'] = {'plan_schema': 'native-embedded-scoped-observation-plan/v1',
        'plan_operation': 'native_embedded_scoped_observation_v1'}
    case['plan'] = {'schema': 'native-embedded-scoped-observation-plan/v1',
        'operation': 'native_embedded_scoped_observation_v1', 'accelerate_root': str(case['root']),
        'task_scope': [TASK], 'native_metadata_cli': {
            'schema': 'native-embedded-scoped-observation-cli/v2', 'runtime_root': str(case['root']),
            'cwd': str(case['root']), 'attempt_directory': str(case['root'] / 'proof'),
            'temporary_directory': str(case['root'] / 'proof/temporary'),
            'project_import_plan': {}, 'pre_body_guard': observer, 'result_path': '',
            'source_sha256': dict(case['request']['source_sha256']), 'observation_binding': binding,
            'import_head_capsule_Git': _import_git_fixture(case['root'],route)}}
    case['origins']['project_imports'].append({'module':
        'ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap',
        **observer, 'execution_profile': 'compiled_exact_authenticated_source_bytes'})
    case['result'] = {'schema': 'native-embedded-daemon-observation@1',
        'backend': 'embedded', 'database_path': db['path'], 'task_scope': [TASK],
        'observer_source': observer, 'observer_callable': 'observe_embedded_database_without_dispatch',
        'ready_task_count': 0, 'selection_idle_reason': 'no_scoped_ready_tasks',
        'unchanged': True, 'write_count': 0, 'provider_dispatched': False,
        'attempt_consumed': False, 'completion_authority': False,
        'catalogue_binding': {'database_preimage': db, 'plan_root_cid': 'plan:fixture',
            'repository_tree_id': 'tree:fixture', 'source_schema': binding['source_schema'],
            'event_cursor': 253, 'task_statuses': [{'task_cid': TASK, 'status': 'blocked', 'revision': 7}]}}
    return case


def test_genuine_embedded_idle_shape_is_distinct_and_source_bound(cycle_case):
    case = _embedded_idle_shape(cycle_case)
    before = copy.deepcopy(case['native'])
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is True and result['scope'] == 'genuine_scoped_idle'
    assert result['dispatch_authority'] is False and result['completion_authority'] is False
    assert case['native'] == before


@pytest.mark.parametrize('mutation', ['extra', 'backend', 'callable', 'scope', 'selection',
    'ready', 'boolean_ready', 'writes', 'provider', 'attempt', 'completion', 'unchanged',
    'database', 'db_preimage', 'observer_source', 'observer_unselected', 'observer_origin',
    'catalogue_extra', 'catalogue_root', 'catalogue_tree', 'catalogue_schema',
    'catalogue_cursor', 'task_revision', 'task_status', 'duplicate_task'],
    ids=['extra', 'backend', 'callable', 'scope', 'selection', 'ready', 'boolean-ready',
         'writes', 'provider', 'attempt', 'completion', 'unchanged', 'database',
         'db-preimage', 'observer-source', 'observer-unselected', 'observer-origin',
         'catalogue-extra', 'catalogue-root', 'catalogue-tree', 'catalogue-schema',
         'catalogue-cursor', 'task-revision', 'task-status', 'duplicate-task'])
def test_embedded_idle_cannot_substitute_backend_source_catalogue_or_authority(cycle_case, mutation):
    case = _embedded_idle_shape(cycle_case)
    result = case['result']; catalogue = result['catalogue_binding']
    if mutation == 'extra': result['trusted'] = True
    elif mutation == 'backend': result['backend'] = 'typed'
    elif mutation == 'callable': result['observer_callable'] = 'observe_typed_database_without_dispatch'
    elif mutation == 'scope': result['task_scope'] = []
    elif mutation == 'selection': result['selection_idle_reason'] = 'scoped_ready'
    elif mutation == 'ready': result['ready_task_count'] = 1
    elif mutation == 'boolean_ready': result['ready_task_count'] = False
    elif mutation == 'writes': result['write_count'] = 1
    elif mutation == 'provider': result['provider_dispatched'] = True
    elif mutation == 'attempt': result['attempt_consumed'] = True
    elif mutation == 'completion': result['completion_authority'] = True
    elif mutation == 'unchanged': result['unchanged'] = False
    elif mutation == 'database': result['database_path'] += '.foreign'
    elif mutation == 'db_preimage': catalogue['database_preimage'] = {**catalogue['database_preimage'], 'sha256': 'e' * 64}
    elif mutation == 'observer_source': result['observer_source'] = case['profile']['driver_source']
    elif mutation == 'observer_unselected': case['request']['source_sha256'].pop(result['observer_source']['path'])
    elif mutation == 'observer_origin': case['origins']['project_imports'].pop()
    elif mutation == 'catalogue_extra': catalogue['trusted'] = True
    elif mutation == 'catalogue_root': catalogue['plan_root_cid'] = 'foreign:root'
    elif mutation == 'catalogue_tree': catalogue['repository_tree_id'] = 'foreign:tree'
    elif mutation == 'catalogue_schema': catalogue['source_schema'] = 'foreign@1'
    elif mutation == 'catalogue_cursor': catalogue['event_cursor'] -= 1
    elif mutation == 'task_revision': catalogue['task_statuses'][0]['revision'] += 1
    elif mutation == 'task_status': catalogue['task_statuses'][0]['status'] = 'completed'
    else: catalogue['task_statuses'].append(copy.deepcopy(catalogue['task_statuses'][0]))
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['dead_pid', 'birth', 'boot', 'invocation', 'argv', 'cwd', 'exe', 'profile_environment'])
def test_current_process_mismatch_refuses_old_cycle(cycle_case, mutation):
    case = cycle_case
    if mutation == 'birth': case['process']['birth_ticks'] += 1
    elif mutation == 'boot': case['process']['boot_id'] = '87654321-4321-4321-4321-cba987654321'
    elif mutation == 'invocation': case['manager']['InvocationID'] = 'b' * 32
    elif mutation == 'argv': case['process']['argv'].append('--unbound-work')
    elif mutation == 'cwd': case['process']['cwd'] += '/foreign'
    elif mutation == 'exe': case['process']['exe'] = '/foreign/python'
    elif mutation == 'profile_environment': case['process']['environment']['IPFS_NATIVE_EXECUTOR_PROFILE_SHA256'] = '0' * 64
    process = (lambda pid: None) if mutation == 'dead_pid' else (lambda pid: copy.deepcopy(case['process']))
    result = _verify(case, profile_ref=case['profile_ref'], process_reader=process)
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['inactive', 'wrong_id', 'missing_pid', 'missing_invocation'])
def test_manager_mismatch_refuses_cycle(cycle_case, mutation):
    case = cycle_case
    if mutation == 'inactive': case['manager']['ActiveState'] = 'inactive'
    elif mutation == 'wrong_id': case['manager']['Id'] = 'foreign.service'
    elif mutation == 'missing_pid': case['manager']['MainPID'] = '0'
    else: case['manager']['InvocationID'] = ''
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['namespace', 'database', 'generation', 'tree', 'population', 'task_cid', 'cursor'])
def test_foreign_current_catalogue_or_regressed_cursor_refuses_cycle(cycle_case, mutation):
    case = cycle_case
    if mutation == 'namespace': case['native']['board_namespace'] = 'foreign'
    elif mutation == 'database': case['native']['database_uuid'] = 'foreign-database'
    elif mutation == 'generation': case['native']['store_generation'] += 1
    elif mutation == 'tree': case['native']['repository_tree_id'] = 'foreign-tree'
    elif mutation == 'population': case['native']['population_sha256'] = 'f' * 64
    elif mutation == 'task_cid': case['native']['tasks'][0]['task_cid'] = 'task:foreign'
    else: case['native']['event_cursor'] = 252
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['receipt_stale', 'receipt_future', 'cycle_stale', 'cycle_future',
                                    'monotonic_future', 'receipt_before_cycle', 'wrong_cycle_pid'])
def test_stale_future_or_wrong_process_cycle_refuses(cycle_case, mutation):
    case = cycle_case
    if mutation == 'receipt_stale': case['receipt']['observed_at'] = NOW - 121
    elif mutation == 'receipt_future': case['receipt']['observed_at'] = NOW + 1
    elif mutation == 'cycle_stale': case['proof']['finished_at'] = NOW - 121
    elif mutation == 'cycle_future': case['proof']['finished_at'] = NOW + 1
    elif mutation == 'monotonic_future': case['proof']['finished_monotonic_ns'] = 10**15 + 1
    elif mutation == 'receipt_before_cycle': case['receipt']['observed_at'] = NOW - 3
    else: case['proof']['process']['pid'] += 1
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['source_open', 'native_open', 'child_unreaped', 'child_failed',
    'storage_retained', 'cleanup_error', 'signal', 'partial', 'missing_lease_release',
    'live_descendant', 'unknown_processes', 'spawn_closure_unknown'])
def test_any_missing_original_closure_refuses_health(cycle_case, mutation):
    case = cycle_case
    terminal = case['terminal']
    if mutation == 'source_open': terminal['source_READ_closed'] = False
    elif mutation == 'native_open': terminal['original_native_compute_closed'] = False
    elif mutation == 'child_unreaped': terminal['child_reaped'] = False
    elif mutation == 'child_failed': terminal['child_returncode'] = 1
    elif mutation == 'storage_retained': terminal['storage_status'] = 'retained'
    elif mutation == 'cleanup_error': terminal['cleanup_errors'] = ['unresolved custody']
    elif mutation == 'signal': terminal['received_signals'] = [15]
    elif mutation == 'partial': terminal['partial_or_failed_control_must_be_reconciled'] = True
    elif mutation == 'missing_lease_release': terminal['reservation']['resource_lease'].pop('released')
    elif mutation == 'live_descendant': terminal['sampled_process_custody_final']['live_processes'] = 1
    elif mutation == 'unknown_processes': terminal['sampled_process_custody_final']['tracking_error'] = 'unknown'
    else: terminal['process_spawn_receipts']['all_recorded_original_births_stopped'] = False
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['result_schema', 'result_failed', 'result_has_provider_authority',
                                    'plan_task_cid', 'plan_revision', 'native_cli_task_cid', 'origin_source', 'origin_denial'])
def test_original_cli_result_plan_and_origin_scope_cannot_be_substituted(cycle_case, mutation):
    case = cycle_case
    if mutation == 'result_schema': case['result']['schema'] = 'agent-says-passed'
    elif mutation == 'result_failed': case['result']['original_main_returncode'] = 1
    elif mutation == 'result_has_provider_authority': case['result']['provider_authority'] = True
    elif mutation == 'plan_task_cid': case['plan']['task_cid'] = 'task:foreign'
    elif mutation == 'plan_revision': case['plan']['native_metadata_cli']['expected_revision'] = 8
    elif mutation == 'native_cli_task_cid':
        argv = case['plan']['native_metadata_cli']['cli_argv']
        argv[argv.index('--native-metadata-execution-task-cid') + 1] = 'task:foreign'
    elif mutation == 'origin_source': case['origins']['project_imports'][0]['sha256'] = '0' * 64
    else: case['origins']['denials'] = [{'kind': 'unknown source'}]
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


def test_process_replacement_during_verification_refuses_health(cycle_case):
    case = cycle_case
    calls = []
    def process(pid):
        calls.append(pid)
        observed = copy.deepcopy(case['process'])
        if len(calls) > 1:
            observed['birth_ticks'] += 1
        return observed
    result = _verify(case, profile_ref=case['profile_ref'], process_reader=process)
    assert result['verified'] is False and result['scope'] == 'none'
    assert calls == [PID, PID]


def test_cycle_ending_after_legacy_preclose_capture_is_not_accepted(cycle_case):
    """Actual v08 control-execution hadFalse; onlypostexit ownerstdout hadTrue."""
    case = cycle_case
    case['terminal']['source_READ_closed'] = False
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['boolean_cli_count', 'boolean_live_processes', 'resident_descendant_bytes',
    'foreign_owner_pid', 'foreign_execution_reservation', 'foreign_record_reservation',
    'unselected_original_owner', 'boolean_store_generation'])
def test_typed_original_process_and_reservation_joins_are_mandatory(cycle_case, mutation):
    case = cycle_case
    if mutation == 'boolean_cli_count': case['result']['original_cli_calls'] = True
    elif mutation == 'boolean_live_processes': case['terminal']['sampled_process_custody_final']['live_processes'] = False
    elif mutation == 'resident_descendant_bytes': case['terminal']['sampled_process_custody_final']['rss_bytes'] = 1
    elif mutation == 'foreign_owner_pid': case['terminal']['owner_pid'] += 1
    elif mutation == 'foreign_execution_reservation': case['execution']['disk_reservation_id'] = 'e' * 32
    elif mutation == 'foreign_record_reservation': case['terminal']['reservation']['record']['reservation_id'] = 'e' * 32
    elif mutation == 'unselected_original_owner':
        case['request']['source_sha256'].pop(case['profile']['owner_source']['path'])
    else:
        case['profile']['catalogue']['store_generation'] = 1
        case['cycle_native']['store_generation'] = 1
        case['receipt']['catalogue']['store_generation'] = 1
        case['native']['store_generation'] = True
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['process_environment', 'receipt_catalogue', 'receipt_origin',
                                    'terminal_reservation', 'terminal_process_custody'])
def test_malformed_linked_metadata_returns_closed_unknown_instead_of_raising(cycle_case, mutation):
    case = cycle_case
    if mutation == 'process_environment': case['process']['environment'] = None
    elif mutation == 'receipt_catalogue': case['receipt']['catalogue'] = None
    elif mutation == 'receipt_origin': case['receipt']['origins'] = {'untyped': None}
    elif mutation == 'terminal_reservation': case['terminal']['reservation'] = []
    else: case['terminal']['sampled_process_custody_final'] = []
    if mutation != 'process_environment':
        _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


def test_linked_native_snapshot_cannot_be_newer_than_finished_cycle(cycle_case):
    case = cycle_case
    case['cycle_native']['checked_at_epoch'] = NOW - 1
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


@pytest.mark.parametrize('mutation', ['pid', 'birth', 'boot', 'parent', 'ancestry_end', 'captured_after_cycle',
                                    'native_lease', 'released_after_cycle'])
def test_original_owner_capability_links_cannot_be_relabelled(cycle_case, mutation):
    case = cycle_case
    observation = case['owner_observation']
    if mutation == 'pid': observation['pid'] += 1
    elif mutation == 'birth': observation['birth_ticks'] += 1
    elif mutation == 'boot': observation['boot_id'] = 'foreign-boot'
    elif mutation == 'parent': observation['ancestry'][0]['parent_pid'] += 1
    elif mutation == 'ancestry_end': observation['ancestry'][-1]['birth_ticks'] += 1
    elif mutation == 'captured_after_cycle': observation['captured_at'] = NOW - 1
    elif mutation == 'native_lease': case['terminal']['reservation']['record']['native_lease_id'] = 'c' * 32
    else: case['terminal']['reservation']['record']['released_at'] = NOW - 1
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope'] == 'none'


def test_json_data_has_no_original_cycle_capability_to_publish(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('serializable fake handle cannot reach manager/process observation')
    monkeypatch.setattr(health, '_manager', forbidden)
    monkeypatch.setattr(health, '_process', forbidden)
    with pytest.raises(health.ExecutorHealthError, match='capability'):
        health.publish_after_native_cycle({}, {}, {}, sequence=1, cycle_handle={})
    assert not list(tmp_path.iterdir())


def test_fake_popen_data_cannot_start_original_cycle_tracking(tmp_path):
    class LooksLikeProcess:
        pid = PID
        def poll(self):
            return None
    with pytest.raises(health.ExecutorHealthError, match='live_popen'):
        health.track_original_cycle({}, LooksLikeProcess(), {}, sequence=1)
    assert not list(tmp_path.iterdir())


def _new_transition_shape(case, *, changed_epoch):
    """Pure transition-helper input after assumed current-cycle authentication.

    This helper never supplies an opaque cycle capability or publishes health.
    The separate capability negatives above exercise that production boundary.
    Old proof/request files remain byte-identical and are never rewritten here.
    """
    receipt_path = Path(case['profile']['receipt_path'])
    old_raw = receipt_path.read_bytes()
    identity = copy.deepcopy(case['written_identity'])
    if changed_epoch:
        identity.update(pid=PID + 1, birth_ticks=BIRTH + 100, invocation_id='b' * 32)
        identity['environment']['INVOCATION_ID'] = 'b' * 32
    proof = copy.deepcopy(case['proof'])
    proof.update(sequence=1 if changed_epoch else 2,
        started_monotonic_ns=10**15 + 190 * 10**9,
        request_ref=_put(case['root'] / 'proof/new-cycle-request.json',
            {'schema': 'explicit-transition-input-fixture', 'request_generation': 2}))
    native = {**case['profile']['catalogue'], 'event_cursor': 253, 'task_scope': [TASK]}
    return identity, proof, native, old_raw


def _transition(case, identity, proof, native):
    return health._previous_receipt_transition(case['profile_ref'], case['profile'], native,
        identity, proof, now=NOW + 200, monotonic_ns=10**15 + 200 * 10**9)


def test_same_epoch_genuine_next_shape_can_follow_a_stale_prior_observation(cycle_case):
    case = cycle_case
    identity, proof, native, old_raw = _new_transition_shape(case, changed_epoch=False)
    result = _transition(case, identity, proof, native)
    assert result['previous_receipt_sha256'] == hashlib.sha256(old_raw).hexdigest()
    assert result['archive_path'] is result['archive_raw'] is None
    assert Path(case['profile']['receipt_path']).read_bytes() == old_raw


@pytest.mark.parametrize('epoch', ['same_boot_dead_owner', 'new_boot_pid_collision'])
def test_new_manager_epoch_requires_new_cycle_and_retains_prior_receipt(cycle_case, monkeypatch, epoch):
    case = cycle_case
    identity, proof, native, old_raw = _new_transition_shape(case, changed_epoch=True)
    def disappeared(pid):
        assert pid == PID
        raise FileNotFoundError('explicit old-process fixture')
    if epoch == 'new_boot_pid_collision':
        identity.update(pid=PID, birth_ticks=BIRTH, boot_id='87654321-4321-4321-4321-cba987654321')
        def forbidden(pid):
            pytest.fail('old kernel boot identity cannot be resurrected by a new-boot PID collision')
        monkeypatch.setattr(health, '_stat_identity', forbidden)
    else:
        monkeypatch.setattr(health, '_stat_identity', disappeared)
    result = _transition(case, identity, proof, native)
    digest = hashlib.sha256(old_raw).hexdigest()
    assert result['previous_receipt_sha256'] == digest
    assert result['archive_raw'] == old_raw
    assert result['archive_path'] == str(health._epoch_archive_path(case['profile']['receipt_path'], digest))
    assert Path(case['profile']['receipt_path']).read_bytes() == old_raw
    assert not Path(result['archive_path']).exists()  # Pure planning, no publication.


def test_new_invocation_cannot_replace_an_old_owner_that_is_still_alive(cycle_case, monkeypatch):
    case = cycle_case
    identity, proof, native, old_raw = _new_transition_shape(case, changed_epoch=True)
    monkeypatch.setattr(health, '_stat_identity', lambda pid: {'pid': pid, 'birth_ticks': BIRTH, 'parent_pid': 1})
    with pytest.raises(health.ExecutorHealthError, match='still_alive'):
        _transition(case, identity, proof, native)
    assert Path(case['profile']['receipt_path']).read_bytes() == old_raw


def test_unknown_old_owner_custody_cannot_be_treated_as_dead(cycle_case, monkeypatch):
    case = cycle_case
    identity, proof, native, old_raw = _new_transition_shape(case, changed_epoch=True)
    def denied(pid):
        raise PermissionError('old process custody unavailable')
    monkeypatch.setattr(health, '_stat_identity', denied)
    with pytest.raises(health.ExecutorHealthError, match='custody_unknown'):
        _transition(case, identity, proof, native)
    assert Path(case['profile']['receipt_path']).read_bytes() == old_raw


@pytest.mark.parametrize('sequence', [0, 2, True])
def test_new_epoch_needs_exact_initial_sequence(cycle_case, monkeypatch, sequence):
    case = cycle_case
    identity, proof, native, old_raw = _new_transition_shape(case, changed_epoch=True)
    identity['boot_id'] = '87654321-4321-4321-4321-cba987654321'
    proof['sequence'] = sequence
    with pytest.raises(health.ExecutorHealthError):
        _transition(case, identity, proof, native)
    assert Path(case['profile']['receipt_path']).read_bytes() == old_raw


def test_new_process_without_a_new_manager_invocation_cannot_change_epoch(cycle_case):
    case = cycle_case
    identity, proof, native, old_raw = _new_transition_shape(case, changed_epoch=True)
    identity['invocation_id'] = INVOCATION
    identity['environment']['INVOCATION_ID'] = INVOCATION
    with pytest.raises(health.ExecutorHealthError, match='invocation_not_changed'):
        _transition(case, identity, proof, native)
    assert Path(case['profile']['receipt_path']).read_bytes() == old_raw


def test_new_epoch_cannot_relabel_the_old_original_request_as_new_work(cycle_case):
    case = cycle_case
    identity, proof, native, old_raw = _new_transition_shape(case, changed_epoch=True)
    old_receipt = json.loads(old_raw)
    original_proof = json.loads(Path(old_receipt['cycle_proof_ref']['path']).read_bytes())
    proof['request_ref'] = original_proof['request_ref']
    with pytest.raises(health.ExecutorHealthError, match='request_replayed'):
        _transition(case, identity, proof, native)
    assert Path(case['profile']['receipt_path']).read_bytes() == old_raw


def test_epoch_archive_preserves_exact_bytes_and_is_idempotent_without_replacement(cycle_case):
    case = cycle_case
    raw = Path(case['profile']['receipt_path']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    path = health._epoch_archive_path(case['profile']['receipt_path'], digest)
    first = health._archive_previous_receipt(path, raw)
    inode = path.stat().st_ino
    second = health._archive_previous_receipt(path, raw)
    assert first == second == _ref(path)
    assert path.read_bytes() == raw and path.stat().st_ino == inode
    assert path.stat().st_mode & 0o222 == 0
    assert Path(case['profile']['receipt_path']).read_bytes() == raw


def test_epoch_archive_collision_refuses_and_preserves_the_existing_bytes(cycle_case):
    case = cycle_case
    raw = Path(case['profile']['receipt_path']).read_bytes()
    path = health._epoch_archive_path(case['profile']['receipt_path'], hashlib.sha256(raw).hexdigest())
    _put(path, b'preexisting contradictory bytes\n')
    before = path.read_bytes()
    with pytest.raises(health.ExecutorHealthError, match='collision'):
        health._archive_previous_receipt(path, raw)
    assert path.read_bytes() == before
    assert Path(case['profile']['receipt_path']).read_bytes() == raw


@pytest.mark.parametrize('mutation', ['plan_schema', 'plan_operation', 'plan_extra', 'cli_schema',
    'cli_extra', 'cli_runtime_root', 'binding_extra', 'binding_schema', 'binding_none',
    'coordinated_source_schema', 'db_zero', 'db_boolean', 'db_negative', 'db_oversized',
    'db_float', 'db_digest', 'db_relative', 'db_extra', 'external_missing', 'external_none',
    'external_extra', 'external_duplicate', 'external_small', 'external_boolean', 'external_float',
    'external_oversized', 'external_foreign', 'task_revision_float', 'task_revision_boolean',
    'observer_bytes_float', 'cli_observer_unselected'])
def test_embedded_idle_closed_physical_and_protocol_joins_refuse_relabelled_metadata(cycle_case, mutation):
    case = _embedded_idle_shape(cycle_case)
    plan = case['plan']; command = plan['native_metadata_cli']
    binding = command['observation_binding']; result = case['result']
    db = case['request']['database_preimage']
    external = case['request']['external_files']
    if mutation == 'plan_schema':
        plan['schema'] = case['profile']['cycle_contract']['plan_schema'] = 'foreign-plan@1'
    elif mutation == 'plan_operation':
        plan['operation'] = case['profile']['cycle_contract']['plan_operation'] = 'foreign_operation'
    elif mutation == 'plan_extra': plan['grant'] = True
    elif mutation == 'cli_schema': command['schema'] = 'foreign-cli@1'
    elif mutation == 'cli_extra': command['grant'] = True
    elif mutation == 'cli_runtime_root': command['runtime_root'] += '/foreign'
    elif mutation == 'binding_extra': binding['grant'] = True
    elif mutation == 'binding_schema': binding['schema'] = 'foreign-binding@1'
    elif mutation == 'binding_none': command['observation_binding'] = None
    elif mutation == 'coordinated_source_schema':
        binding['source_schema'] = result['catalogue_binding']['source_schema'] = 'foreign-source@1'
    elif mutation.startswith('db_'):
        if mutation == 'db_zero': db['bytes'] = 0
        elif mutation == 'db_boolean': db['bytes'] = True
        elif mutation == 'db_negative': db['bytes'] = -1
        elif mutation == 'db_oversized': db['bytes'] = 128 * 1024**2 + 1
        elif mutation == 'db_float': db['bytes'] = 8.0
        elif mutation == 'db_digest': db['sha256'] = 'invalid-physical-digest'
        elif mutation == 'db_relative':
            db['path'] = result['database_path'] = external[0]['path'] = 'relative.duckdb'
        else: db['grant'] = True
    elif mutation == 'external_missing': case['request']['external_files'] = []
    elif mutation == 'external_none': case['request']['external_files'] = None
    elif mutation == 'external_extra': external[0]['grant'] = True
    elif mutation == 'external_duplicate': external.append(dict(external[0]))
    elif mutation == 'external_small': external[0]['max_bytes'] = db['bytes'] - 1
    elif mutation == 'external_boolean':
        db['bytes'] = 1; external[0]['max_bytes'] = True
    elif mutation == 'external_float': external[0]['max_bytes'] = float(128 * 1024**2)
    elif mutation == 'external_oversized': external[0]['max_bytes'] = 128 * 1024**2 + 1
    elif mutation == 'external_foreign': external[0]['path'] += '.foreign'
    elif mutation == 'task_revision_float': result['catalogue_binding']['task_statuses'][0]['revision'] = 7.0
    elif mutation == 'task_revision_boolean':
        case['native']['tasks'][0]['revision'] = case['cycle_native']['tasks'][0]['revision'] = 1
        result['catalogue_binding']['task_statuses'][0]['revision'] = True
    elif mutation == 'observer_bytes_float':
        changed = {**result['observer_source'], 'bytes': float(result['observer_source']['bytes'])}
        result['observer_source'] = binding['observer_source'] = changed
    else: command['source_sha256'].pop(result['observer_source']['path'])
    _write_evidence(case)
    verified = _verify(case, profile_ref=case['profile_ref'])
    assert verified['verified'] is False and verified['scope'] == 'none'
    assert verified['dispatch_authority'] is False and verified['completion_authority'] is False


def test_embedded_idle_health_validation_never_opens_the_mutable_database(cycle_case, monkeypatch):
    case = _embedded_idle_shape(cycle_case)
    database = case['request']['database_preimage']['path']
    original_raw = health._raw
    def guarded_raw(path, maximum=health.MAX_JSON):
        assert str(path) != database, 'health verification must consume the closed observation, not reopen its DB'
        return original_raw(path, maximum)
    monkeypatch.setattr(health, '_raw', guarded_raw)
    _write_evidence(case)
    verified = _verify(case, profile_ref=case['profile_ref'])
    assert verified['verified'] is True and verified['scope'] == 'genuine_scoped_idle'
    assert verified['dispatch_authority'] is False and verified['completion_authority'] is False


def test_installed_interpreter_accepts_actual_elf_in_distinct_bounded_role(tmp_path):
    case = _base_profile(tmp_path)
    interpreter = Path(sys.executable).resolve()
    tool = health.installed_tool_descriptor(interpreter)
    assert tool == _ref(interpreter)
    with interpreter.open('rb') as stream:
        assert stream.read(4) == b'\x7fELF'
    assert health.MAX_SOURCE == health.MAX_JSON == 4 * 1024**2
    assert health.MAX_INSTALLED_TOOL == 64 * 1024**2
    assert health._profile(_publish_profile(case)) == case['profile']



@pytest.mark.parametrize('role', ['main_entrypoint', 'health_module', 'owner_source', 'driver_source'],
    ids=['main-entrypoint', 'health-module', 'owner-source', 'driver-source'])
def test_project_role_cannot_borrow_installed_interpreter_allowance(tmp_path, role):
    case = _base_profile(tmp_path)
    case['profile'][role] = health.installed_tool_descriptor(Path(sys.executable).resolve())
    with pytest.raises(health.ExecutorHealthError, match='project_role_aliases_installed_tool'):
        health._profile(_publish_profile(case))



@pytest.mark.parametrize('declared_bytes', [67108865, True], ids=['over-64-mib', 'boolean-bytes'])
def test_installed_tool_metadata_bound_is_checked_before_read(declared_bytes):
    ref = {'path': str(Path(sys.executable).resolve()), 'bytes': declared_bytes, 'sha256': 'f' * 64}
    with pytest.raises(health.ExecutorHealthError, match='descriptor_bounds'):
        health._bound_installed_tool(ref)



def test_ordinary_project_source_retains_four_mib_bound(tmp_path):
    case = _base_profile(tmp_path)
    source = _put(tmp_path / 'source/oversized_project.py', b'#' + b'x' * (4 * 1024**2))
    case['profile']['source_descriptors'].append(source)
    with pytest.raises(health.ExecutorHealthError, match='descriptor_bounds'):
        health._profile(_publish_profile(case))



def test_installed_binary_is_rejected_as_an_ordinary_project_source(tmp_path):
    case = _base_profile(tmp_path)
    tool = health.installed_tool_descriptor(Path(sys.executable).resolve())
    with pytest.raises(health.ExecutorHealthError, match='installed_tool_is_not_project_source'):
        health._bound_project_source(tool, case['profile'])



def test_python_source_cannot_be_declared_an_installed_elf_tool(tmp_path):
    source = _put(tmp_path / 'pretend_interpreter.py', b'# explicit Python source fixture\n')
    with pytest.raises(health.ExecutorHealthError, match='installed_tool_not_elf'):
        health.installed_tool_descriptor(source['path'])



def test_project_import_origin_cannot_be_laundered_through_interpreter_role(tmp_path):
    case = _base_profile(tmp_path)
    tool = health.installed_tool_descriptor(Path(sys.executable).resolve())
    origins = {'schema': 'legal-ir-pre-execution-project-import-origins/v1',
        'project_origins_checked_before_module_execution': True, 'denials': [],
        'project_imports': [{'module': 'ipfs_accelerate_py.fixture', **tool,
            'execution_profile': 'compiled_exact_authenticated_source_bytes'}]}
    with pytest.raises(health.ExecutorHealthError, match='installed_tool_is_not_project_source'):
        health._origins(origins, case['profile'])



def test_process_identity_rechecks_installed_tool_pin(tmp_path):
    case = _base_profile(tmp_path)
    profile_ref = _publish_profile(case)
    case['process']['environment']['IPFS_NATIVE_EXECUTOR_PROFILE_SHA256'] = profile_ref['sha256']
    for source in case['profile']['source_descriptors']:
        if source['path'] == case['profile']['unit']['interpreter']:
            source['sha256'] = 'f' * 64
    with pytest.raises(health.ExecutorHealthError, match='descriptor_changed'):
        health._identity(profile_ref, case['profile'], lambda _: case['manager'], lambda _: case['process'])



def test_receipt_project_origins_cannot_claim_installed_interpreter(cycle_case):
    case = cycle_case
    case['receipt']['origins']['ipfs_accelerate_py.fake_tool_module'] = health.installed_tool_descriptor(Path(sys.executable).resolve())
    _write_evidence(case)
    result = _verify(case, profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['reason'] == 'producer_import_origins_unbound'



def _import_git_fixture(root, route):
    relative='ipfs_accelerate_py/agent_implementation_route.py'
    sql=['ipfs_accelerate_py/agent_supervisor/task_sources/sql/'+name for name in (
        '0001_control_plane.sql','0002_causal_event_federation_core.sql',
        '0003_state_server_restart_identity.sql','0004_hash_observations.sql','0005_tasks_index_repair.sql')]
    blobs=[{'relative_path':relative,'object_id':'0'*40,'bytes':route['bytes']}]
    blobs.extend({'relative_path':name,'object_id':str(index)*40,'bytes':1} for index,name in enumerate(sql,2))
    return {'blobs':blobs,
        'ls_tree_tail':['ls-tree','-r','-z','1'*40,'--','ipfs_accelerate_py',
            'scripts/ops/agent_supervisor/configured_board_scheduler.py',
            'scripts/ops/agent_supervisor/implementation_supervisor_entry.py'],
        'payload_bytes':route['bytes']+5,'required_relative_files':[relative,*sql],'route_source':route,
        'scope':'immutable HEAD bytes archived/hashed by original import capsule; no import or execution of omitted live source bodies',
        'selected_file_count':6,'source_head':'1'*40,'source_root':str(root),'unique_blob_count':6}


@pytest.mark.parametrize('mutation',['head','route_source','old_cli_schema'])
def test_embedded_idle_requires_the_original_import_source_generation_contract(cycle_case,mutation):
    case=_embedded_idle_shape(cycle_case)
    command=case['plan']['native_metadata_cli'];git=command['import_head_capsule_Git']
    if mutation=='head':git['source_head']='not-an-immutable-head'
    elif mutation=='route_source':case['request']['source_sha256'].pop(git['route_source']['path'])
    else:command['schema']='native-embedded-scoped-observation-cli/v1'
    _write_evidence(case)
    result=_verify(case,profile_ref=case['profile_ref'])
    assert result['verified'] is False and result['scope']=='none'
    assert result['dispatch_authority'] is False and result['completion_authority'] is False



def test_group_writable_database_data_does_not_relax_control_or_source_readers(tmp_path):
    path=tmp_path/'owner-data.json';raw=b'{"explicit_data_fixture":true}\n'
    path.write_bytes(raw);path.chmod(0o664)
    ref=health.database_descriptor(path)
    assert ref=={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    assert health.MAX_DATABASE_DATA_BYTES==128*1024**2
    with pytest.raises(health.ExecutorHealthError,match='unsafe_or_oversized_file'):
        health._document(ref)
    with pytest.raises(health.ExecutorHealthError,match='unsafe_or_oversized_file'):
        health.descriptor(path)
    path.chmod(0o666)
    with pytest.raises(health.ExecutorHealthError,match='unsafe_or_oversized_database_data'):
        health.database_descriptor(path)


@pytest.mark.parametrize('case',['relative','directory','alias','oversized'])
def test_database_data_descriptor_reuses_path_and_bounded_regular_file_fences(tmp_path,monkeypatch,case):
    path=tmp_path/'data.bin';path.write_bytes(b'explicit small byte fixture')
    if case=='relative':
        offered='relative-data.bin';reason='aliased_or_relative_path'
    elif case=='directory':
        offered=tmp_path;reason='unsafe_or_oversized_database_data'
    elif case=='alias':
        # Exercise the existing alias fence without adding a second special
        # symlink fixture outside the original owning harness's admitted pair.
        original=Path.is_symlink
        monkeypatch.setattr(Path,'is_symlink',lambda item:True if item==path else original(item))
        offered=path;reason='aliased_or_relative_path'
    else:
        monkeypatch.setattr(health,'MAX_DATABASE_DATA_BYTES',len(path.read_bytes())-1)
        offered=path;reason='unsafe_or_oversized_database_data'
    with pytest.raises(health.ExecutorHealthError,match=reason):
        health.database_descriptor(offered)
