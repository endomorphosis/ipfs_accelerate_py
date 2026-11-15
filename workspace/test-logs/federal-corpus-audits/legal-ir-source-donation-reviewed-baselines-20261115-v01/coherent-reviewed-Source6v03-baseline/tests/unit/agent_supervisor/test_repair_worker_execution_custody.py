"""Actual import and original-Popen regression fixtures for worker closeout.

These are protocol/resource owning cases, not public D/native admission or full
supervisor integration. Process primitives are real. Only /proc metadata reads
are fault-injected; every child is genuinely waited and all outputs retained.
"""
from contextlib import contextmanager
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker


@contextmanager
def actual_guarded_module(tmp_path, name, source):
    root = tmp_path / 'candidate'
    root.mkdir()
    path = root / (name + '.py')
    path.write_text(source)
    pin = worker.descriptor(path)
    guard = worker.PinnedSources({str(path): pin['sha256']},
        [root, tmp_path / 'datasets', tmp_path / 'trusted'])
    original_path, original_meta = list(sys.path), list(sys.meta_path)
    assert name not in sys.modules
    try:
        sys.path.insert(0, str(root))
        sys.meta_path.insert(0, guard)
        yield path, pin, guard
    finally:
        sys.modules.pop(name, None)
        sys.path[:] = original_path
        sys.meta_path[:] = original_meta
        importlib.invalidate_caches()


def test_original_module_body_error_records_exact_executed_buffer_and_failure_attribution(tmp_path):
    name = 'private_candidate_body_error'
    with actual_guarded_module(tmp_path, name, 'raise RuntimeError("actual candidate body failure")\n') as (path, pin, guard):
        with pytest.raises(RuntimeError, match='actual candidate body failure') as caught:
            importlib.import_module(name)
        assert guard.executed == {str(path): pin['sha256']}
        frame = caught.value.__traceback__
        module_frames = []
        while frame is not None:
            if guard.compiled_codes.get(frame.tb_frame.f_code) == str(path):
                module_frames.append(frame.tb_frame.f_code)
            frame = frame.tb_next
        assert len(module_frames) == 1 and module_frames[0].co_name == '<module>'
        row = worker.original_pytest_failure_attribution(guard, error=caught.value,
            report_path=__file__, nodeid=__file__ + '::' + test_original_module_body_error_records_exact_executed_buffer_and_failure_attribution.__name__,
            when='call', selected_files={__file__}, required_bodies={str(path)},
            source_pins={str(path): pin['sha256']}, collector_error_type=type(None),
            importer_code=None, importer_globals=None, wrapper_pin=None)
        assert row['kind'] == 'changed_candidate_runtime_error'
        assert row['candidate_traceback'] == [{'path': str(path), 'sha256': pin['sha256'], 'function': '<module>'}]
        assert guard.syntax_failures == [] and guard.denials == []
        assert name not in sys.modules


def test_original_preexec_audit_refusal_never_asserts_module_body_execution(tmp_path):
    name = 'private_candidate_audit_refusal'
    with actual_guarded_module(tmp_path, name, 'body_started = True\n') as (path, pin, guard):
        active = [True]
        def refuse_original_exec(event, args):
            if active[0] and event == 'exec' and args[0].co_filename == str(path):
                raise RuntimeError('actual preexec audit refusal')
        sys.addaudithook(refuse_original_exec)
        try:
            with pytest.raises(RuntimeError, match='actual preexec audit refusal'):
                importlib.import_module(name)
        finally:
            active[0] = False
        assert any(code.co_filename == str(path) for code in guard.compiled_codes)
        assert str(path) not in guard.executed
        assert guard.syntax_failures == [] and name not in sys.modules


def test_original_compile_syntax_failure_still_has_no_body_execution_witness(tmp_path):
    name = 'private_candidate_syntax_error'
    with actual_guarded_module(tmp_path, name, 'def broken(:\n') as (path, pin, guard):
        with pytest.raises(SyntaxError):
            importlib.import_module(name)
        assert str(path) not in guard.executed
        assert guard.syntax_failures[0]['body_executed'] is False
        assert guard.syntax_failures[0]['sha256'] == pin['sha256']
        assert name not in sys.modules


def custody_scope(tmp_path):
    scope = object.__new__(worker.TrustedFixtures)
    scope.root = tmp_path / 'original-root'
    scope.root.mkdir(mode=0o700)
    scope.identity = worker._actual_identity(scope.root)
    scope.current = __file__ + '::original_popen_closeout_protocol'
    scope.processes, scope.child_events = [], []
    scope.guard = SimpleNamespace(denials=[])
    scope.children_ready = False
    scope.cleaning_children = False
    request = tmp_path / 'execution-request.json'
    request.write_bytes(b'{"explicit_protocol_fixture":true}\n')
    scope.plan = {'public_request_path': str(request)}
    scope.request_sha256 = hashlib.sha256(request.read_bytes()).hexdigest()
    parent = os.getppid()
    birth = Path('/proc/' + str(parent) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
    scope.original_owner = {'pid': parent, 'birth': birth}
    scope.control_receipts = []
    return scope


ACTUAL_CHILD_SOURCE = 'import time; time.sleep(20)\n'


def spawn_actual_child(tmp_path):
    # This is an original real process, not a mocked Popen result.
    return worker._ORIGINAL_POPEN([sys.executable, '-I', '-B', '-c', ACTUAL_CHILD_SOURCE],
        cwd=tmp_path, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)


def row_for(process):
    return {'nodeid': __file__, 'kind': 'explicit_original_popen_closeout_fixture',
        'pid': process.pid, 'birth': None, 'start_new_session': True,
        'genuine_original_Popen_returned': True, 'actual_wait_completed': False}


def finish_actual_children(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def test_original_popen_handle_survives_first_birth_read_failure_and_is_really_waited(tmp_path, monkeypatch):
    scope = custody_scope(tmp_path)
    process = spawn_actual_child(tmp_path)
    row = row_for(process)
    original = Path.read_text
    target = Path('/proc/' + str(process.pid) + '/stat')
    remaining = [1]
    def fail_one_birth_read(path, *args, **kwargs):
        if path == target and remaining[0]:
            remaining[0] -= 1
            raise OSError('injected original birth metadata read failure')
        return original(path, *args, **kwargs)
    try:
        with monkeypatch.context() as change:
            change.setattr(Path, 'read_text', fail_one_birth_read)
            with pytest.raises(OSError, match='original birth metadata'):
                scope.register_child(process, row)
            assert scope.processes == [(process, row)] and scope.child_events == [row]
            assert row['birth'] is None and row['actual_wait_completed'] is False
            assert row['initial_identity_error']['type'] == 'OSError' and scope.guard.denials
            scope.close_children()
        assert process.returncode is not None and row['actual_wait_completed'] is True
        assert row['birth'] is None  # Do not replace unknown original identity.
        assert row['cleanup_observed_identity']['pid'] == process.pid
        assert row['cleanup_observed_identity']['birth'].isdigit()
        assert scope.guard.denials  # Cleanup does not grant semantic acceptance.
    finally:
        finish_actual_children([process])


def test_cleanup_attempts_other_real_children_after_one_metadata_closeout_error(tmp_path, monkeypatch):
    scope = custody_scope(tmp_path)
    processes = []
    try:
        first = spawn_actual_child(tmp_path)
        processes.append(first)
        second = spawn_actual_child(tmp_path)
        processes.append(second)
        rows = [row_for(first), row_for(second)]
        scope.register_child(first, rows[0])
        scope.register_child(second, rows[1])
        original = Path.read_text
        target = Path('/proc/' + str(first.pid) + '/stat')
        def fail_first_child_metadata(path, *args, **kwargs):
            if path == target:
                raise OSError('injected first child closeout metadata failure')
            return original(path, *args, **kwargs)
        with monkeypatch.context() as change:
            change.setattr(Path, 'read_text', fail_first_child_metadata)
            with pytest.raises(RuntimeError, match='cleanup incomplete'):
                scope.close_children()
        assert rows[0]['actual_wait_completed'] is False
        assert rows[0]['cleanup_error']['type'] == 'OSError'
        assert rows[1]['actual_wait_completed'] is True and second.returncode is not None
        assert scope.guard.denials and scope.cleaning_children is False
    finally:
        finish_actual_children(processes)


def test_missing_initial_birth_after_actual_wait_remains_unknown_but_cleanup_closes(tmp_path, monkeypatch):
    scope = custody_scope(tmp_path)
    process = spawn_actual_child(tmp_path)
    row = row_for(process)
    original = Path.read_text
    target = Path('/proc/' + str(process.pid) + '/stat')
    def fail_original_birth(path, *args, **kwargs):
        if path == target:
            raise OSError('injected missing initial birth')
        return original(path, *args, **kwargs)
    try:
        with monkeypatch.context() as change:
            change.setattr(Path, 'read_text', fail_original_birth)
            with pytest.raises(OSError, match='missing initial birth'):
                scope.register_child(process, row)
        process.terminate()
        actual_code = process.wait(timeout=3)
        scope.close_children()
        assert row['actual_wait_completed'] is True and row['returncode'] == actual_code
        assert row['birth'] is None and 'cleanup_observed_identity' not in row
        assert scope.guard.denials
    finally:
        finish_actual_children([process])


@pytest.mark.parametrize('error_type', [OSError, ValueError, RuntimeError])
def test_normal_original_finder_failure_remains_denied_after_real_import_restoration(tmp_path, monkeypatch, error_type):
    name = 'private_original_finder_restoration'
    with actual_guarded_module(tmp_path, name, 'body_executed = True\n') as (path, pin, guard):
        original = importlib.machinery.PathFinder.find_spec
        remaining = [1]
        def fail_one_original_lookup(cls, fullname, path=None, target=None):
            if fullname == name and remaining[0]:
                remaining[0] -= 1
                raise error_type('actual original finder metadata refusal')
            return original(fullname, path, target)
        with monkeypatch.context() as change:
            change.setattr(importlib.machinery.PathFinder, 'find_spec', classmethod(fail_one_original_lookup))
            with pytest.raises(ImportError, match='normal source specification authentication failed'):
                importlib.import_module(name)
            assert str(path) not in guard.executed and guard.denials
            refusal = list(guard.denials)
            assert error_type.__name__ + ': actual original finder metadata refusal' in refusal[0]['reason']
            actual = importlib.import_module(name)
            assert actual.body_executed is True and Path(actual.__file__) == path
            assert guard.executed == {str(path): pin['sha256']}
            assert guard.denials == refusal  # Later real execution grants no clean qualification.
            assert guard.syntax_failures == []


# Ordinary immutable delegation controls. The SDK proof-reader body below is
# explicitly modeled DATA: these cases exercise worker transport/custody gates,
# never real parent FDs, Source42 fairness, Native admission or SDK integration.
_ORDINARY_PARENT_READER_DATA = b'''import hashlib
import json
from pathlib import Path
observations = []
state = {'refuse': False, 'summary_fault': None}
def original_execution_request(invocation):
    return json.loads((invocation / 'execution-request-publication.json').read_bytes())['request']
def _modeled_parent_reader(raw, *, enforce=True, **arguments):
    if enforce and state['refuse']:
        raise ValueError('modeled original parent no longer holds Source')
    value = json.loads(raw)
    observations.append(arguments)
    result = {'observation_sha256': hashlib.sha256(raw).hexdigest(),
        'request_sha256': arguments['expected_request_sha256'],
        'nonce_sha256': arguments['expected_nonce_sha256'],
        'parent_identity': arguments['expected_parent_identity'],
        'source_fds': value['source_fds'],
        'source_root_bindings': arguments['expected_source_root_bindings']}
    if state['summary_fault'] == 'nonce':
        result['nonce_sha256'] = '0' * 64
    elif state['summary_fault'] == 'extra':
        result['source_closed'] = True
    return result
def verify_immutable_source_parent_observation_v1(raw, **arguments):
    return _modeled_parent_reader(raw, **arguments)
'''


def _ordinary_parent_put_DATA(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if type(value) is bytes else worker.canonical(value) + b'\n'
    path.write_bytes(raw)
    path.chmod(0o400)
    return worker.descriptor(path)


def _ordinary_parent_rewrite_DATA(path, value):
    path.chmod(0o600)
    return _ordinary_parent_put_DATA(path, value)


def _ordinary_parent_replace_request_DATA(scope, request):
    """Keep transport joins coherent while damaging only the request contract."""
    import json
    local = _ordinary_parent_rewrite_DATA(Path(scope.local_request['path']), request)
    public = _ordinary_parent_rewrite_DATA(Path(scope.public_request['path']), request)
    publication = json.loads(scope.publication_path.read_bytes())
    publication.update(request=public, invocation_copy=local)
    _ordinary_parent_rewrite_DATA(scope.publication_path, publication)
    observation = json.loads(Path(scope.observation_pin['path']).read_bytes())
    observation['request'] = public
    _ordinary_parent_rewrite_DATA(Path(scope.observation_pin['path']), observation)


@pytest.fixture
def ordinary_parent_worker_DATA(tmp_path, monkeypatch):
    """Real sealed files, modeled SDK proof seam, no Source/Native operations."""
    control = tmp_path / 'control'
    datasets = tmp_path / 'datasets'
    primary_d, primary_c = tmp_path / 'primary-D', tmp_path / 'primary-C'
    invocation, attempt = tmp_path / 'invocation', tmp_path / 'attempt'
    for path in (control, datasets, primary_d, primary_c, invocation, attempt):
        path.mkdir()
    relatives = {
        'ipfs_accelerate_py/__init__.py': b'',
        'ipfs_accelerate_py/agent_supervisor/__init__.py': b'',
        'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py': b'',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py': b'# modeled metadata only\n',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py': _ORDINARY_PARENT_READER_DATA,
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py': Path(worker.__file__).read_bytes(),
        'ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py': b'modeled_Main_subject = True\n',
        'ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py': b'modeled_health_subject = True\n',
        'tests/unit/test_selected_DATA.py': b'def test_selected_DATA():\n    assert True\n',
    }
    files = [_ordinary_parent_put_DATA(control / relative, body) for relative, body in relatives.items()]
    files.sort(key=lambda row: row['path'])
    manifest = _ordinary_parent_put_DATA(control / 'repair-control-manifest.json', {
        'schema': 'agent-supervisor/repair-control-source-release@1', 'root': str(control),
        'files': files, 'namespace_profiles': []})
    population = {str(Path(row['path']).relative_to(control)):
        dict(bytes=row['bytes'], sha256=row['sha256'], executable=False) for row in files}
    commitment = hashlib.sha256(worker.canonical({'source_root': str(control), 'manifest': population})).hexdigest()
    snapshot = _ordinary_parent_put_DATA(tmp_path / 'source-snapshot.json', {
        'schema': 'agent-supervisor/immutable-source-snapshot@1',
        'binding': {'namespace': 'ptr_lane_0', 'board_id': 'legal-ir', 'attempt_id': 'a' * 32},
        'source_root': str(control), 'manifest': population, 'source_commitment': commitment})
    required = [row for row in files if Path(row['path']).name in ('scientific_executor.py', 'native_executor_health.py')]
    profile = {'schema': 'agent-supervisor/immutable-source-qualification@2', 'snapshot': snapshot,
        'required_source_bodies': required, 'source_custody_mode': 'delegated-original-parent-read/v1'}
    selected = [str(control / 'tests/unit/test_selected_DATA.py') + '::test_selected_DATA']
    validation = dict(trusted_tests=[next(row for row in files if row['path'].endswith('test_selected_DATA.py'))],
        selected_checks=selected, pytest_module={'modeled_pytest_origin': True}, import_sources=[],
        attempt_root=str(tmp_path / 'accounted'), cpu_slots=1, memory_mb=1536, storage_bytes=16777216,
        timeout_seconds=300, child_process_slots=4, scheduler_state_path=str(tmp_path / 'scheduler.json'),
        expected_scheduler_policy_sha256='1' * 64, roots=[str(tmp_path)],
        resource_ledger=str(tmp_path / 'resource-ledger.json'),
        ordinary_scratch_profile={'schema': 'owned-validation-ordinary-scratch-live/v1'}, source_qualification=profile)
    owner_path = control / 'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py'
    worker_path = owner_path.with_name('repair_validation_worker.py')
    policy = {'schema': 'agent-supervisor/repair-public-owner-policy@2',
        'control_release': {'root': str(control), 'manifest': manifest}, 'datasets_root': str(datasets),
        'python': sys.executable, 'source_descriptors': [worker.descriptor(owner_path), worker.descriptor(worker_path)],
        'validation': validation, 'adoption': {}}
    policy_pin = _ordinary_parent_put_DATA(invocation / 'owner-policy.json', policy)
    nonce = 'b' * 64
    public_path = Path(validation['attempt_root']) / 'control-requests/legal_ir_unattended_repair/legal-ir' / ('a' * 32) / 'execution-request.json'
    bindings = [{'physical_root': str(datasets), 'canonical_root': str(primary_d)},
        {'physical_root': str(control), 'canonical_root': str(primary_c)}]
    plan = {'schema': worker.WORK_SCHEMA, 'attempt_directory': str(attempt), 'candidate_root': str(control),
        'expected_cases': 1, 'freeze': snapshot, 'normal_namespace_profiles': [],
        'ordinary_scratch_profile': validation['ordinary_scratch_profile'],
        'original_public_control_pins': {row['path']: row['sha256'] for row in files},
        'policy': policy_pin, 'required_candidate_bodies': [],
        'required_source_bodies': [row['path'] for row in required], 'selected_checks': selected,
        'source_commitment': commitment, 'source_pins': {row['path']: row['sha256'] for row in files},
        'source_qualification': profile, 'parent_identity': {'pid': 4242, 'birth': '17'},
        'parent_nonce_sha256': hashlib.sha256(bytes.fromhex(nonce)).hexdigest(),
        'parent_source_observation_path': str(invocation / 'parent-source-observation.json'),
        'execution_request_path': str(public_path), 'source_root_bindings': bindings}
    plan_pin = _ordinary_parent_put_DATA(invocation / 'validation-plan.json', plan)
    args = SimpleNamespace(plan=Path(plan_pin['path']), plan_sha256=plan_pin['sha256'])
    request = {'schema': 'owned-validation-request/v5',
        'argv': [sys.executable, '-I', '-B', str(worker_path), '--plan', str(args.plan),
            '--plan-sha256', args.plan_sha256], 'cwd': str(control), 'selected_checks': selected,
        'source_sha256': {**plan['original_public_control_pins'], plan_pin['path']: plan_pin['sha256']},
        'timeout_seconds': 300, 'storage_bytes': 16777216,
        'memory_mb': 1536, 'cpu_slots': 1, 'child_process_slots': 4, 'attempt_directory': str(attempt),
        'outputs': {name: str(attempt / name) for name in ('cache', 'db', 'exports', 'logs', 'pytest', 'tmp')},
        'declared_external_writable_paths': [], 'process_scope': 'sampled_birth_bound_descendants',
        'scheduler_state_path': validation['scheduler_state_path'],
        'expected_scheduler_policy_sha256': validation['expected_scheduler_policy_sha256'],
        'roots': validation['roots'], 'ordinary_scratch_profile': validation['ordinary_scratch_profile']}
    local_request = _ordinary_parent_put_DATA(invocation / 'execution-request.json', request)
    public_request = _ordinary_parent_put_DATA(public_path, Path(local_request['path']).read_bytes())
    publication_path = invocation / 'execution-request-publication.json'
    _ordinary_parent_put_DATA(publication_path, {'schema': 'agent-supervisor/public-validation-request-publication@1',
        'request': public_request, 'invocation_copy': local_request, 'plan': plan_pin, 'freeze': snapshot,
        'owner_policy': policy_pin, 'source_commitment': commitment})
    observation = {'schema': 'agent-supervisor/immutable-source-parent-observation@1',
        'request': public_request, 'parent_identity': plan['parent_identity'], 'nonce': nonce,
        'source_receipts': [{'modeled_receipt_only': True}] * 2,
        'source_fds': [{'modeled_FD_only': True}, {'modeled_FD_only': True}],
        'sdk_argv': [], 'source_root_bindings': bindings}
    observation_pin = _ordinary_parent_put_DATA(Path(plan['parent_source_observation_path']), observation)
    remembered = []
    guard = SimpleNamespace(executed={}, remember_code=lambda code, path: remembered.append((code, path)))
    monkeypatch.setattr(worker, '__file__', str(worker_path))
    monkeypatch.setitem(sys.modules, '_ordinary_immutable_parent_source_proof_owner', None)
    return SimpleNamespace(plan=plan, policy=policy, args=args, guard=guard, control=control, datasets=datasets,
        owner_path=owner_path, worker_path=worker_path, observation=observation, observation_pin=observation_pin,
        request=request, local_request=local_request, public_request=public_request, publication_path=publication_path,
        remembered=remembered, attempt=attempt, invocation=invocation)


def test_ordinary_immutable_parent_worker_binds_public_request_without_child_Source(ordinary_parent_worker_DATA):
    from contextlib import ExitStack
    s = ordinary_parent_worker_DATA
    lease_calls = []
    def forbidden_lease(*args, **kwargs):
        lease_calls.append((args, kwargs))
        raise AssertionError('delegated worker must not acquire either Source lease')
    assert worker.checked_source_qualification_plan(s.plan, s.policy) == s.plan['source_qualification']
    with ExitStack() as stack:
        parent, entered = worker.enter_ordinary_worker_source_custody_v1(
            s.plan, s.policy, s.args, s.guard, stack, forbidden_lease)
        parent.verify_parent()
        parent.verify_parent()
    assert entered == 0 and lease_calls == []
    assert parent.parent_observation == s.observation_pin
    assert len(parent.helper.observations) == 3
    for arguments in parent.helper.observations:
        assert arguments['expected_sdk_argv'][5] == s.public_request['path']
        assert arguments['expected_sdk_argv'][5] != s.local_request['path']
        assert arguments['expected_request_sha256'] == s.public_request['sha256']
        assert arguments['expected_source_root_bindings'] == s.plan['source_root_bindings']
    assert parent.proof_reader.__code__.co_filename == str(s.owner_path)
    assert s.guard.executed == {str(s.owner_path): worker.descriptor(s.owner_path)['sha256']}
    assert not (s.datasets / 'workspace/control-plane/source-leases').exists()
    assert not (s.control / 'workspace/control-plane/source-leases').exists()


@pytest.mark.parametrize('damage', ['missing_observation', 'unsealed_observation', 'local_only', 'wrong_public_path'])
def test_ordinary_parent_source_refusal_never_retries_child_leases(ordinary_parent_worker_DATA, damage):
    from contextlib import ExitStack
    s = ordinary_parent_worker_DATA
    if damage == 'missing_observation':
        Path(s.observation_pin['path']).unlink()
    elif damage == 'unsealed_observation':
        Path(s.observation_pin['path']).chmod(0o600)
    elif damage == 'local_only':
        s.publication_path.unlink()
    else:
        publication = __import__('json').loads(s.publication_path.read_bytes())
        publication['request'] = s.local_request
        _ordinary_parent_rewrite_DATA(s.publication_path, publication)
    calls = []
    def forbidden_lease(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('failed parent proof must not fall back to Source admission')
    with ExitStack() as stack, pytest.raises((ValueError, OSError)):
        worker.enter_ordinary_worker_source_custody_v1(s.plan, s.policy, s.args, s.guard, stack, forbidden_lease)
    assert calls == []


def test_ordinary_legacy_worker_still_enters_and_closes_two_original_source_contexts(tmp_path):
    from contextlib import ExitStack
    events = []
    @contextmanager
    def modeled_lease(root, **arguments):
        events.append(('enter', root, arguments))
        try:
            yield
        finally:
            events.append(('exit', root))
    policy = {'datasets_root': str(tmp_path / 'D'), 'validation': {'timeout_seconds': 300}}
    plan = {'candidate_root': str(tmp_path / 'C')}
    with ExitStack() as stack:
        parent, entered = worker.enter_ordinary_worker_source_custody_v1(plan, policy, None, None, stack, modeled_lease)
        assert parent is None and entered == 2
    assert [row[:2] for row in events] == [('enter', tmp_path / 'D'), ('enter', tmp_path / 'C'),
        ('exit', tmp_path / 'C'), ('exit', tmp_path / 'D')]
    assert all(row[2] == {'mode': 'read', 'timeout_seconds': 300} for row in events if row[0] == 'enter')


@pytest.mark.parametrize('damage', ['mode', 'missing_mode', 'extra_profile', 'boolean_pid', 'crossed_roots',
    'extra_root_field', 'relative_lock_root', 'missing_runtime', 'extra_runtime', 'native_schema', 'retry_profile'])
def test_ordinary_parent_profile_rejects_mixed_or_incomplete_delegation(ordinary_parent_worker_DATA, damage):
    import copy
    s = ordinary_parent_worker_DATA
    plan, policy = copy.deepcopy(s.plan), copy.deepcopy(s.policy)
    profile = policy['validation']['source_qualification']
    if damage == 'mode': profile['source_custody_mode'] = 'reviewed-native420'
    elif damage == 'missing_mode': del profile['source_custody_mode']
    elif damage == 'extra_profile': profile['native_operation'] = {}
    elif damage == 'boolean_pid': plan['parent_identity']['pid'] = True
    elif damage == 'crossed_roots': plan['source_root_bindings'].reverse()
    elif damage == 'extra_root_field': plan['source_root_bindings'][0]['lock_path'] = '/tmp/foreign.lock'
    elif damage == 'relative_lock_root': plan['source_root_bindings'][0]['canonical_root'] = 'foreign'
    elif damage == 'missing_runtime': del plan['parent_nonce_sha256']
    elif damage == 'extra_runtime': plan['parent_nonce'] = 'f' * 64
    elif damage == 'native_schema': plan['schema'] = worker.REVIEWED_NATIVE_WORK_SCHEMA
    else: policy['validation']['validation_admission_retry'] = {}
    with pytest.raises(ValueError):
        worker.checked_source_qualification_plan(plan, policy)


@pytest.mark.parametrize('field', ['parent_identity', 'parent_nonce_sha256', 'parent_source_observation_path',
    'execution_request_path', 'source_root_bindings'])
def test_schema1_ordinary_plan_cannot_smuggle_delegated_fields(ordinary_parent_worker_DATA, field):
    import copy
    s = ordinary_parent_worker_DATA
    plan, policy = copy.deepcopy(s.plan), copy.deepcopy(s.policy)
    profile = policy['validation']['source_qualification']
    profile['schema'] = 'agent-supervisor/immutable-source-qualification@1'
    del profile['source_custody_mode']
    for key in ('parent_identity', 'parent_nonce_sha256', 'parent_source_observation_path',
            'execution_request_path', 'source_root_bindings'):
        if key != field:
            del plan[key]
    with pytest.raises(ValueError, match='legacy plan cannot carry parent delegation'):
        worker.checked_source_qualification_plan(plan, policy)


@pytest.mark.parametrize('damage', ['boolean_cpu', 'request_v6', 'foreign_cwd', 'other_selected',
    'outside_output', 'other_pins', 'other_process_scope', 'unsealed_request', 'unsealed_public', 'unsealed_plan',
    'changed_plan', 'changed_policy', 'changed_SDK', 'changed_public_bytes', 'wrong_observation_request'])
def test_ordinary_parent_worker_rejects_crossed_or_changed_original_inputs(ordinary_parent_worker_DATA, damage):
    import copy
    s = ordinary_parent_worker_DATA
    request = copy.deepcopy(s.request)
    if damage == 'boolean_cpu': request['cpu_slots'] = True
    elif damage == 'request_v6': request['schema'] = 'owned-validation-request/v6'
    elif damage == 'foreign_cwd': request['cwd'] = str(s.datasets)
    elif damage == 'other_selected': request['selected_checks'] = []
    elif damage == 'outside_output': request['outputs']['logs'] = str(s.control)
    elif damage == 'other_pins': request['source_sha256'] = {}
    elif damage == 'other_process_scope': request['process_scope'] = 'caller_asserted'
    elif damage == 'unsealed_request': Path(s.local_request['path']).chmod(0o600)
    elif damage == 'unsealed_public': Path(s.public_request['path']).chmod(0o600)
    elif damage == 'unsealed_plan': s.args.plan.chmod(0o600)
    elif damage == 'changed_plan': _ordinary_parent_rewrite_DATA(s.args.plan, b'{}\n')
    elif damage == 'changed_policy': _ordinary_parent_rewrite_DATA(Path(s.plan['policy']['path']), b'{}\n')
    elif damage == 'changed_SDK': _ordinary_parent_rewrite_DATA(s.owner_path, b'# source drift\n')
    elif damage == 'changed_public_bytes': _ordinary_parent_rewrite_DATA(Path(s.public_request['path']), b'{}\n')
    else:
        observation = copy.deepcopy(s.observation)
        observation['request'] = s.local_request
        _ordinary_parent_rewrite_DATA(Path(s.observation_pin['path']), observation)
    if damage in {'boolean_cpu', 'request_v6', 'foreign_cwd', 'other_selected', 'outside_output', 'other_pins', 'other_process_scope'}:
        _ordinary_parent_replace_request_DATA(s, request)
    with pytest.raises((ValueError, OSError)):
        worker.OrdinaryImmutableParentSourceVerifierV1(s.plan, s.policy, s.args, s.guard)


@pytest.mark.parametrize('damage', ['reader_binding', 'reader_code', 'core_code', 'core_keyword_default',
    'extra_global', 'global_binding'])
def test_ordinary_parent_pinned_reader_code_and_globals_remain_bound(ordinary_parent_worker_DATA, damage):
    s = ordinary_parent_worker_DATA
    parent = worker.OrdinaryImmutableParentSourceVerifierV1(s.plan, s.policy, s.args, s.guard)
    if damage == 'reader_binding': parent.helper.verify_immutable_source_parent_observation_v1 = lambda *a, **k: {}
    elif damage == 'reader_code': parent.proof_reader.__code__ = (lambda *a, **k: {}).__code__
    elif damage == 'core_code': parent.helper._modeled_parent_reader.__code__ = (lambda *a, **k: {}).__code__
    elif damage == 'core_keyword_default': parent.helper._modeled_parent_reader.__kwdefaults__['enforce'] = False
    elif damage == 'extra_global': parent.helper.unprovided_proof_authority = True
    else: parent.helper.json = None
    with pytest.raises(ValueError, match='Source proof helper changed'):
        parent.verify_parent()


@pytest.mark.parametrize('damage', ['parent_lost', 'different_nonce_summary', 'extra_summary', 'observation_bytes'])
def test_ordinary_parent_loss_or_changed_observation_at_later_boundary_refuses(ordinary_parent_worker_DATA, damage):
    s = ordinary_parent_worker_DATA
    parent = worker.OrdinaryImmutableParentSourceVerifierV1(s.plan, s.policy, s.args, s.guard)
    parent.verify_parent()  # Earlier modeled selected-phase boundary succeeds.
    if damage == 'parent_lost': parent.helper.state['refuse'] = True
    elif damage == 'different_nonce_summary': parent.helper.state['summary_fault'] = 'nonce'
    elif damage == 'extra_summary': parent.helper.state['summary_fault'] = 'extra'
    else:
        path = Path(s.observation_pin['path'])
        _ordinary_parent_rewrite_DATA(path, path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        parent.verify_parent()
    assert parent.parent_observation == s.observation_pin



def test_ordinary_parent_profile_accepts_positive_container_PID1_metadata(ordinary_parent_worker_DATA):
    """Parser compatibility only; real parent FD/nonce/ancestry proof is separate."""
    import copy
    s = ordinary_parent_worker_DATA
    plan = copy.deepcopy(s.plan)
    plan['parent_identity']['pid'] = 1
    assert worker.checked_source_qualification_plan(plan, s.policy) == plan['source_qualification']
    assert s.guard.executed == {}  # No SDK proof reader or child Source operation ran.


def test_ordinary_parent_profile_rejects_zero_PID_metadata(ordinary_parent_worker_DATA):
    import copy
    s = ordinary_parent_worker_DATA
    plan = copy.deepcopy(s.plan)
    plan['parent_identity']['pid'] = 0
    with pytest.raises(ValueError, match='parent identity and nonce'):
        worker.checked_source_qualification_plan(plan, s.policy)
