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
