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
