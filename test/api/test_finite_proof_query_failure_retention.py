"""Failure-boundary custody controls; no native/prover/model qualification.

These tests execute an explicitly declared AST subset of the actual fixture and
scheduler sources. The private native body is replaced by a controlled delegate.
Receipt writing and connection-close priority use the real retained code. No
native authority, Docker, prover, model, pressure sampler or SQL owner is invoked.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE_RELATIVE = Path('benchmarks/agent_supervisor/container_coding/finite_proof_query_worker_experiment.py')
SCHEDULER_RELATIVE = Path('ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py')
# Exact owner-approved failure definitions; the private native _run is excluded.
FIXTURE_DEFINITIONS = (
    '_failure_identity', '_failure_write', '_failure_attribute',
    '_failure_exception', '_failure_read', '_FailureRetentionContext',
    '_failure_annotate', '_retain_fixture_failure', 'run', 'run_all', 'main',
)
FIXTURE_ASSIGNMENTS = (
    'SCHEMA', 'FAILURE_CUSTODY_SCHEMA', '_FAILURE_FILE_LIMIT',
    '_FAILURE_TRACE_LIMIT', 'POST_BIRTH_CONTROLS',
)
SCHEDULER_DEFINITIONS = ('ResourceSchedulerError', 'LeaseTimeoutError', 'LeaseCancelledError')


def _source_paths():
    fixture = os.environ.get('FINITE_FAILURE_FIXTURE_SOURCE')
    scheduler = os.environ.get('FINITE_FAILURE_SCHEDULER_SOURCE')
    assert bool(fixture) == bool(scheduler), 'captured failure tests require both exact source paths'
    if fixture:
        paths = Path(fixture), Path(scheduler)
        assert all(path.is_absolute() and path.resolve(strict=True) == path for path in paths)
        return (*paths, True)
    # Ordinary local development is explicitly separate from captured tests.
    paths = REPOSITORY / FIXTURE_RELATIVE, REPOSITORY.parent / 'ipfs_datasets' / SCHEDULER_RELATIVE
    if not all(path.is_file() for path in paths):
        pytest.skip('both real local source repositories are required')
    return (*paths, False)


def _extract(path, definitions, monkeypatch, *, assignments=()):
    before = path.stat()
    source = path.read_bytes()
    after = path.stat()
    key = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    assert key(before) == key(after)
    tree = ast.parse(source, filename=str(path))
    selected = []
    selected_names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(alias.name.split('.')[0] in sys.stdlib_module_names for alias in node.names):
                selected.append(node)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or '').split('.')[0] in {*sys.stdlib_module_names, '__future__'}:
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in definitions:
            selected.append(node)
            selected_names.append(node.name)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in assignments for target in node.targets):
            selected.append(node)
    assert len(selected_names) == len(definitions) and set(selected_names) == set(definitions), 'exact actual failure subset missing'
    name = '_finite_failure_custody_ast_' + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    module = ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), module.__dict__)
    module._custody_test_source = {'path': str(path), 'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest(), 'actual_AST_definitions': list(definitions), 'actual_AST_assignments': list(assignments), 'whole_module_import_attested': False}
    return module


@pytest.fixture
def boundary(monkeypatch):
    fixture_path, scheduler_path, captured = _source_paths()
    fixture = _extract(fixture_path, FIXTURE_DEFINITIONS, monkeypatch,
        assignments=FIXTURE_ASSIGNMENTS)
    exceptions = _extract(scheduler_path, SCHEDULER_DEFINITIONS, monkeypatch)
    writer = _extract(fixture_path.parent / 'finite_repository_admission_experiment.py',
        ('_wire', '_write'), monkeypatch)
    fixture._write = writer._write
    return SimpleNamespace(fixture=fixture, exceptions=exceptions, writer=writer,
        captured=captured)


def _receipt(output):
    path = output / 'failure-custody/failure-receipt.json'
    return json.loads(path.read_bytes())


def _retain_test_witness(tmp_path, boundary, case, **facts):
    value = {'schema': 'finite-proof-query-failure-boundary-test-witness@1', 'case': case, 'source_subset': [boundary.fixture._custody_test_source, boundary.exceptions._custody_test_source, boundary.writer._custody_test_source], 'explicit_captured_source_pair': boundary.captured, 'controlled_private_body': True, 'native_worker_or_proof_model_training_qualification': False, 'whole_global_job_counts': None, **facts}
    (tmp_path / 'test-witness.json').write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _invoke(boundary, output, **controls):
    return boundary.fixture.run(output, python_executable=Path(sys.executable),
        lean_executable=Path('/not-invoked/lean'), handoff_root=output.parent / 'handoffs',
        worktree_root=output.parent / 'worktrees', **controls)


def _owned_delegate(boundary, monkeypatch, error, *, before_failure=None):
    calls = []

    def controlled_private_body(output, *, _failure_context, **kwargs):
        output = Path(output)
        output.mkdir(mode=0o700)
        _failure_context.attach_owned_output(output)
        calls.append({'output': str(output), 'arguments': {key: str(value)
            if isinstance(value, Path) else value for key, value in kwargs.items()}})
        if before_failure is not None:
            before_failure(output, _failure_context)
        raise error

    monkeypatch.setattr(boundary.fixture, '_run', controlled_private_body, raising=False)
    return calls


def _assert_failed_receipt(receipt):
    assert receipt['schema'] == 'finite-proof-query-native-failure-custody@1'
    assert receipt['status'] == 'failure_observed'
    assert receipt['same_escaping_exception_preserved'] is True
    assert receipt['terminal_timeout_cause_inferred'] is None
    assert receipt['runtime_STOP_cleanup'] is receipt['kernel_process_cleanup'] is None
    assert receipt['global_training_or_process_counts'] is None
    assert receipt['native_positive_or_post_birth_qualification'] is False
    assert receipt['new_SQL_sampling_scheduler_or_prover_operations'] == 0


def test_original_timeout_identity_message_and_trace_are_retained(boundary, tmp_path, monkeypatch):
    error = boundary.exceptions.LeaseTimeoutError('original admission failure\nsecond line')
    calls = _owned_delegate(boundary, monkeypatch, error)
    output = tmp_path / 'owned'
    with pytest.raises(boundary.exceptions.LeaseTimeoutError) as caught:
        _invoke(boundary, output)
    assert caught.value is error and caught.value.args == ('original admission failure\nsecond line',)
    receipt = _receipt(output)
    _assert_failed_receipt(receipt)
    primary = receipt['primary_exception']
    raw = (output / 'failure-custody' / primary['traceback']['name']).read_bytes()
    assert b'controlled_private_body' in raw and b'original admission failure' in raw
    assert primary['message'] == str(error) and primary['traceback']['truncated'] is False
    assert hashlib.sha256(raw).hexdigest() == primary['traceback']['sha256']
    assert not (output / 'result.json').exists() and len(calls) == 1
    _retain_test_witness(tmp_path, boundary, 'original-timeout', receipt=receipt,
        exact_escaping_instance=True, original_arguments=list(error.args))


@pytest.mark.parametrize('field', ['admission_observation', 'timeout_decision'])
@pytest.mark.parametrize('presence', ['absent', 'null', 'present'])
def test_terminal_field_presence_is_not_filled_from_global_history(
    boundary, tmp_path, monkeypatch, field, presence,
):
    own = {'schema': 'controlled-terminal-field@1', 'waiter_id': 'own-waiter',
           'exact_blocking_predicate': None}
    rival = {'schema': 'controlled-historical-refusal@1', 'waiter_id': 'rival-waiter'}
    error = boundary.exceptions.LeaseTimeoutError('same original timeout')
    if presence == 'absent':
        delattr(error, field)
    else:
        setattr(error, field, None if presence == 'null' else own)

    def state(output, context):
        private = output / 'private'
        private.mkdir()
        (private / 'resource-admission.json').write_text(json.dumps({'last_proof_refusal': rival}))

    _owned_delegate(boundary, monkeypatch, error, before_failure=state)
    output = tmp_path / 'owned'
    with pytest.raises(boundary.exceptions.LeaseTimeoutError) as caught:
        _invoke(boundary, output)
    receipt = _receipt(output)
    observed = receipt['primary_exception']['exception_attributes'][field]
    assert caught.value is error and observed['present'] is (presence != 'absent')
    assert observed['value'] == (own if presence == 'present' else None)
    assert observed['access'] == 'exception_instance_dict'
    assert receipt['persisted_global_last_proof_refusal'] == rival
    assert receipt['persisted_global_history_may_describe_another_request'] is True
    _assert_failed_receipt(receipt)
    _retain_test_witness(tmp_path, boundary, field + '-' + presence, receipt=receipt,
        exact_escaping_instance=True, controlled_history=rival)


@pytest.mark.parametrize('field,invalid', [
    ('admission_observation', {'unsupported': {1, 2}}),
    ('timeout_decision', {'nonfinite': float('nan')}),
])
def test_nonjson_terminal_fields_are_unknown_without_masking_primary(
    boundary, tmp_path, monkeypatch, field, invalid,
):
    error = boundary.exceptions.LeaseTimeoutError('primary remains unchanged')
    setattr(error, field, invalid)
    _owned_delegate(boundary, monkeypatch, error)
    output = tmp_path / 'owned'
    with pytest.raises(boundary.exceptions.LeaseTimeoutError) as caught:
        _invoke(boundary, output)
    receipt = _receipt(output)
    observed = receipt['primary_exception']['exception_attributes'][field]
    assert caught.value is error and observed['present'] is True and observed['value'] is None
    assert observed['serialization_error']['type'] in {'TypeError', 'ValueError'}
    assert receipt['retention_complete'] is False
    assert 'exception_serialization_or_trace_bound' in receipt['diagnostic_errors']
    _assert_failed_receipt(receipt)
    _retain_test_witness(tmp_path, boundary, 'nonjson-' + field, receipt=receipt,
        exact_escaping_instance=True)


def test_partial_stages_raw_journal_and_existing_result_are_preserved(
    boundary, tmp_path, monkeypatch,
):
    error = boundary.exceptions.LeaseTimeoutError('controlled failure after partial stages')
    stages = [{'stage': 'controlled_prepare_returned', 'seconds': 0.01},
              {'stage': 'controlled_observation_returned', 'seconds': 0.02}]
    journal = b''.join((json.dumps(row) + '\n').encode() for row in stages)
    original_result = b'{"status":"controlled_incomplete","native_qualified":false}\n'

    def files(output, context):
        (output / 'stage-timings.json').write_text(json.dumps(stages))
        (output / 'stage-timings.jsonl').write_bytes(journal)
        (output / 'result.json').write_bytes(original_result)

    _owned_delegate(boundary, monkeypatch, error, before_failure=files)
    output = tmp_path / 'owned'
    with pytest.raises(boundary.exceptions.LeaseTimeoutError):
        _invoke(boundary, output)
    receipt = _receipt(output)
    assert receipt['observed_stage_prefix'] == stages
    assert receipt['snapshots']['persisted_resources']['status'] == 'missing'
    assert (output / 'result.json').read_bytes() == original_result
    raw_name = receipt['snapshots']['stage_journal']['retained_raw']['name']
    assert (output / 'failure-custody' / raw_name).read_bytes() == journal
    assert receipt['retention_complete'] is True
    _assert_failed_receipt(receipt)
    _retain_test_witness(tmp_path, boundary, 'partial-progress', receipt=receipt,
        original_result_sha256=hashlib.sha256(original_result).hexdigest())


def test_redirected_resource_snapshot_is_unknown_and_foreign_bytes_unchanged(
    boundary, tmp_path, monkeypatch,
):
    foreign = tmp_path / 'foreign-resources.json'
    raw = b'{"last_proof_refusal":{"waiter_id":"foreign"}}\n'
    foreign.write_bytes(raw)
    error = boundary.exceptions.LeaseTimeoutError('primary timeout')

    def redirected(output, context):
        private = output / 'private'
        private.mkdir()
        (private / 'resource-admission.json').symlink_to(foreign)

    _owned_delegate(boundary, monkeypatch, error, before_failure=redirected)
    output = tmp_path / 'owned'
    with pytest.raises(boundary.exceptions.LeaseTimeoutError) as caught:
        _invoke(boundary, output)
    receipt = _receipt(output)
    assert caught.value is error and foreign.read_bytes() == raw
    assert receipt['snapshots']['persisted_resources']['status'] == 'unavailable'
    assert receipt['persisted_global_last_proof_refusal'] is None
    assert receipt['retention_complete'] is False
    _assert_failed_receipt(receipt)
    _retain_test_witness(tmp_path, boundary, 'redirected-resource-snapshot', receipt=receipt)


@pytest.mark.parametrize('kind', ['keyboard', 'system-exit', 'scheduler-cancel'])
def test_baseexceptions_and_actual_scheduler_cancellation_escape_unchanged(
    boundary, tmp_path, monkeypatch, kind,
):
    error = {'keyboard': KeyboardInterrupt('controlled interrupt'),
             'system-exit': SystemExit(17),
             'scheduler-cancel': boundary.exceptions.LeaseCancelledError('controlled cancellation')}[kind]
    calls = []

    def close():
        calls.append('close')
        raise SystemExit(29)

    def attach(output, context):
        context.attach_connection(SimpleNamespace(close=close))

    _owned_delegate(boundary, monkeypatch, error, before_failure=attach)
    output = tmp_path / 'owned'
    with pytest.raises(type(error)) as caught:
        _invoke(boundary, output)
    receipt = _receipt(output)
    assert caught.value is error and receipt['primary_exception']['type'] == type(error).__name__
    assert calls == ['close'] and receipt['connection_cleanup_errors'][0]['type'] == 'SystemExit'
    assert receipt['owned_connection_cleanup']['close_returned'] is False
    _assert_failed_receipt(receipt)
    _retain_test_witness(tmp_path, boundary, kind, receipt=receipt,
        exact_escaping_instance=True)


def test_primary_timeout_wins_over_actual_context_connection_close_failure(
    boundary, tmp_path, monkeypatch,
):
    error = boundary.exceptions.LeaseTimeoutError('original admission failure')
    cleanup = OSError('controlled original connection close error')
    calls = []

    def close():
        calls.append('close')
        raise cleanup

    def attach(output, context):
        context.attach_connection(SimpleNamespace(close=close))

    _owned_delegate(boundary, monkeypatch, error, before_failure=attach)
    output = tmp_path / 'owned'
    with pytest.raises(boundary.exceptions.LeaseTimeoutError) as caught:
        _invoke(boundary, output)
    receipt = _receipt(output)
    assert caught.value is error and calls == ['close']
    assert receipt['owned_connection_cleanup'] == {'attached': True,
        'close_attempted': True, 'close_returned': False}
    assert len(receipt['connection_cleanup_errors']) == 1
    assert receipt['connection_cleanup_errors'][0]['type'] == 'OSError'
    assert receipt['connection_cleanup_errors'][0]['message'] == str(cleanup)
    _assert_failed_receipt(receipt)
    _retain_test_witness(tmp_path, boundary, 'primary-before-close-error', receipt=receipt,
        close_call_count=len(calls), exact_escaping_instance=True,
        cleanup_scope='actual context.close_connection only; native STOP/runtime.close unknown')


def test_actual_close_helper_failure_without_primary_remains_the_failure(
    boundary, tmp_path, monkeypatch,
):
    error = OSError('close failed after controlled body success')
    calls = []

    def close():
        calls.append('close')
        raise error

    def controlled_private_body(output, *, _failure_context, **kwargs):
        output.mkdir()
        _failure_context.attach_owned_output(output)
        _failure_context.attach_connection(SimpleNamespace(close=close))
        _failure_context.close_connection()
        raise AssertionError('failed actual close must not return success')

    monkeypatch.setattr(boundary.fixture, '_run', controlled_private_body, raising=False)
    output = tmp_path / 'owned'
    with pytest.raises(OSError) as caught:
        _invoke(boundary, output)
    receipt = _receipt(output)
    assert caught.value is error and calls == ['close']
    assert receipt['primary_exception']['type'] == 'OSError'
    assert receipt['owned_connection_cleanup']['close_returned'] is False
    _assert_failed_receipt(receipt)
    _retain_test_witness(tmp_path, boundary, 'close-without-primary', receipt=receipt,
        exact_escaping_instance=True, close_call_count=len(calls))


def test_setup_gap_attached_connection_is_closed_once_by_public_wrapper(
    boundary, tmp_path, monkeypatch,
):
    calls = []
    error = RuntimeError('controlled setup failure')

    def attach(output, context):
        context.attach_connection(SimpleNamespace(close=lambda: calls.append('close')))

    _owned_delegate(boundary, monkeypatch, error, before_failure=attach)
    output = tmp_path / 'owned'
    with pytest.raises(RuntimeError) as caught:
        _invoke(boundary, output)
    receipt = _receipt(output)
    assert caught.value is error and calls == ['close']
    assert receipt['owned_connection_cleanup'] == {'attached': True,
        'close_attempted': True, 'close_returned': True}
    _assert_failed_receipt(receipt)
    _retain_test_witness(tmp_path, boundary, 'setup-gap-close-once', receipt=receipt)


def test_retention_write_failure_keeps_primary_and_partial_trace(
    boundary, tmp_path, monkeypatch, capsys,
):
    error = boundary.exceptions.LeaseTimeoutError('original timeout')
    original = boundary.fixture._failure_write

    def unavailable(directory_fd, name, raw):
        if name == 'failure-receipt.json':
            raise OSError('controlled receipt write failure')
        return original(directory_fd, name, raw)

    monkeypatch.setattr(boundary.fixture, '_failure_write', unavailable)
    _owned_delegate(boundary, monkeypatch, error)
    output = tmp_path / 'owned'
    with pytest.raises(boundary.exceptions.LeaseTimeoutError) as caught:
        _invoke(boundary, output)
    assert caught.value is error and not (output / 'failure-custody/failure-receipt.json').exists()
    trace = output / 'failure-custody/primary.traceback.txt'
    assert trace.is_file() and b'original timeout' in trace.read_bytes()
    assert 'fixture failure custody failed: OSError' in capsys.readouterr().err
    _retain_test_witness(tmp_path, boundary, 'receipt-write-failure',
        exact_escaping_instance=True, receipt_available=False,
        partial_primary_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())


@pytest.mark.parametrize('kind', ['existing', 'redirected', 'before-mkdir'])
def test_failure_without_owned_output_never_writes_foreign_custody(
    boundary, tmp_path, monkeypatch, kind,
):
    output = tmp_path / 'not-owned'
    foreign = tmp_path / 'foreign'
    sentinel = b'original foreign bytes\n'
    if kind == 'existing':
        output.mkdir()
        (output / 'sentinel').write_bytes(sentinel)
    elif kind == 'redirected':
        foreign.mkdir()
        (foreign / 'sentinel').write_bytes(sentinel)
        output.symlink_to(foreign, target_is_directory=True)

    def controlled_private_body(path, *, _failure_context, **kwargs):
        if kind == 'before-mkdir':
            raise RuntimeError('controlled failure before ownership')
        path.mkdir()
        raise AssertionError('existing/redirected output must not be acquired')

    monkeypatch.setattr(boundary.fixture, '_run', controlled_private_body, raising=False)
    with pytest.raises(RuntimeError if kind == 'before-mkdir' else FileExistsError):
        _invoke(boundary, output)
    assert not (output / 'failure-custody').exists()
    if kind != 'before-mkdir':
        assert (output / 'sentinel').read_bytes() == sentinel
    else:
        assert not output.exists()
    _retain_test_witness(tmp_path, boundary, 'unowned-' + kind,
        failure_custody_written=False, foreign_bytes_preserved=True)


def test_replaced_owned_directory_is_not_written_as_if_still_owned(
    boundary, tmp_path, monkeypatch, capsys,
):
    error = RuntimeError('primary before replaced output')
    moved = tmp_path / 'original-moved'
    sentinel = b'new foreign directory\n'

    def replace(output, context):
        output.rename(moved)
        output.mkdir()
        (output / 'sentinel').write_bytes(sentinel)

    _owned_delegate(boundary, monkeypatch, error, before_failure=replace)
    output = tmp_path / 'owned'
    with pytest.raises(RuntimeError) as caught:
        _invoke(boundary, output)
    assert caught.value is error and (output / 'sentinel').read_bytes() == sentinel
    assert not (output / 'failure-custody').exists() and not (moved / 'failure-custody').exists()
    assert 'owned fixture output was replaced' in capsys.readouterr().err
    _retain_test_witness(tmp_path, boundary, 'owned-directory-replaced',
        exact_escaping_instance=True, failure_custody_written=False)


def test_success_returns_original_object_and_preserves_existing_result(
    boundary, tmp_path, monkeypatch,
):
    result = {'schema': boundary.fixture.SCHEMA, 'status': 'completed', 'controlled_delegate': True}
    raw = (json.dumps(result, sort_keys=True) + '\n').encode()
    arguments = []

    def controlled_private_body(output, *, _failure_context, **kwargs):
        output.mkdir()
        _failure_context.attach_owned_output(output)
        (output / 'result.json').write_bytes(raw)
        arguments.append(kwargs)
        return result

    monkeypatch.setattr(boundary.fixture, '_run', controlled_private_body, raising=False)
    output = tmp_path / 'owned'
    assert _invoke(boundary, output, child_control='proof') is result
    assert (output / 'result.json').read_bytes() == raw
    assert arguments[0]['child_control'] == 'proof' and arguments[0]['post_birth_control'] is None
    assert not (output / 'failure-custody').exists()
    _retain_test_witness(tmp_path, boundary, 'success-unchanged',
        same_return_object=True, original_result_sha256=hashlib.sha256(raw).hexdigest())


def test_original_all_collection_keeps_only_original_three_routes(
    boundary, tmp_path, monkeypatch,
):
    calls = []

    def selected(output, **kwargs):
        calls.append((output.name, kwargs['child_control'], kwargs.get('post_birth_control')))
        return {'status': 'completed', 'active_leases': 0, 'waiting_requests': 0}

    monkeypatch.setattr(boundary.fixture, 'run', selected)
    handoffs = tmp_path / 'handoffs'
    handoffs.mkdir()
    boundary.fixture.run_all(tmp_path / 'collection', python_executable=Path(sys.executable),
        lean_executable=Path('/not-invoked/lean'), handoff_root=handoffs,
        worktree_root=tmp_path / 'worktrees')
    assert calls == [('positive', None, None), ('child-late-proof', 'proof', None),
                     ('child-late-epoch', 'epoch', None)]
    _retain_test_witness(tmp_path, boundary, 'original-default-routes',
        observed_delegations=calls, controlled_return_values_not_native_results=True)


@pytest.mark.parametrize('control', [
    'isolated_after_real_birth_callback_error', 'isolated_real_birth_application_ACK_drop',
])
def test_explicit_cli_postbirth_control_is_forwarded_without_native_execution(
    boundary, tmp_path, monkeypatch, capsys, control,
):
    calls = []

    def selected(output, **kwargs):
        calls.append((output, kwargs))
        return {'schema': boundary.fixture.SCHEMA, 'status': 'completed'}

    output = tmp_path / 'not-created'
    monkeypatch.setattr(boundary.fixture, 'run', selected)
    monkeypatch.setattr(sys, 'argv', ['fixture', 'run', str(output), '--post-birth-control', control])
    boundary.fixture.main()
    assert len(calls) == 1 and calls[0][1]['post_birth_control'] == control
    assert calls[0][1]['child_control'] is None and not output.exists()
    assert json.loads(capsys.readouterr().out) == {'schema': boundary.fixture.SCHEMA, 'status': 'completed'}
    _retain_test_witness(tmp_path, boundary, 'explicit-CLI-' + control,
        observed_control=control, controlled_return_value_not_native_result=True)


def test_all_cli_refuses_explicit_postbirth_control(boundary, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fixture', 'run-all', str(tmp_path / 'not-created'),
        '--post-birth-control', 'isolated_after_real_birth_callback_error'])
    with pytest.raises(SystemExit) as caught:
        boundary.fixture.main()
    assert caught.value.code == 2 and not (tmp_path / 'not-created').exists()
    _retain_test_witness(tmp_path, boundary, 'all-CLI-refusal', actual_parser_exit=2)
