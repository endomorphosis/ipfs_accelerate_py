"""Cold refusal/source controls only. No genuine-owner fakes or runtime jobs."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).absolute().parent
RUNTIME = HERE/'proposed/ipfs_accelerate_py/agent_supervisor/runtime/ordinary_success_handoff.py'
HOST = HERE/'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_native_host.py'


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module;spec.loader.exec_module(module)
    return module


H = load('cold_H4_source_controls',RUNTIME)
N = load('cold_native_host_source_controls',HOST)


def method(name):
    tree = ast.parse(HOST.read_text())
    host = next(row for row in tree.body if isinstance(row,ast.ClassDef) and row.name == 'ReportOnlyNativeHost')
    return next(row for row in host.body if isinstance(row,ast.FunctionDef) and row.name == name)


def test_unknown_host_cannot_enable_daemon():
    with pytest.raises(N.NativeHostRefusal,match='explicit_same_process_native_host_required'):
        N.require_report_only_native_host(None)


def test_native_pending_bypasses_generic_Exception_handlers():
    assert issubclass(N.NativeHostCustodyPending,BaseException)
    assert not issubclass(N.NativeHostCustodyPending,Exception)


def test_callback_capture_cannot_be_constructed():
    with pytest.raises(H.OrdinaryHandoffRefusal,match='internal_authenticated_callback_constructor_only'):
        H.AuthenticatedOrdinaryCallback()


def test_callback_authentication_requires_actual_active_callback():
    with pytest.raises(H.OrdinaryHandoffRefusal,match='fresh_original_callback_authentication_required'):
        H.authenticate_ordinary_callback_for_freezer(None)


def test_unknown_capture_cannot_enter_freezer_scope():
    with pytest.raises(H.OrdinaryHandoffRefusal,match='actual_authenticated_callback_capture_required'):
        with H.authenticated_ordinary_freezer_scope(None,freezer_owner=None):
            raise AssertionError('unknown capture entered')


def test_unknown_ordinary_pending_cannot_release_consumers():
    with pytest.raises(H.OrdinaryHandoffRefusal,match='actual_retained_ordinary_pending_required'):
        H.acknowledge_stopped_ordinary_custody(None)


def test_unknown_capture_cannot_be_discarded_as_authenticated():
    with pytest.raises(H.OrdinaryHandoffRefusal,match='actual_unused_same_callback_capture_required'):
        H.discard_unused_authenticated_callback(None)


def test_unknown_outcome_cannot_authenticate_success():
    with pytest.raises(H.OrdinaryHandoffRefusal,match='actual_active_original_success_callback_required'):
        H.require_active_ordinary_success(None)


def test_no_new_scheduler_root_or_reconstructed_lease_in_host():
    calls = [row for row in ast.walk(ast.parse(HOST.read_text())) if isinstance(row,ast.Call)]
    assert not any(isinstance(row.func,ast.Attribute) and row.func.attr == 'acquire' for row in calls)
    assert not any(isinstance(row.func,ast.Name) and row.func.id in {'ResourceLease','GlobalResourceScheduler'} for row in calls)


def test_host_cap_guard_precedes_original_pass_without_setting_cap():
    text = ast.unparse(method('guard_pass_entry'))
    assert 'daemon.max_task_attempts == 3' in text and 'daemon.preserve_task_attempts is True' in text
    assert 'daemon.use_ephemeral_worktree is True' in text and 'daemon.production_provider_policy is None' in text
    tree = ast.parse(HOST.read_text())
    assert not any(isinstance(row,(ast.Assign,ast.AnnAssign,ast.AugAssign))
        and any(isinstance(target,ast.Attribute) and target.attr == 'max_task_attempts'
                for target in (row.targets if isinstance(row,ast.Assign) else [row.target])) for row in ast.walk(tree))


def test_original_command_and_binary_are_checked_before_preflight_owner():
    body = method('preflight_candidate').body
    owner_index = next(i for i,row in enumerate(body) if 'self._owner(' in ast.unparse(row))
    earlier = '\n'.join(ast.unparse(row) for row in body[:owner_index])
    assert 'parse_runner_arguments' in earlier and '_resolve_trusted_grok_bin' in earlier
    assert 'actual_trusted_binary_pin_required_before_consumption' in earlier
    assert 'check_primary_only_options' in earlier and 'grok-4.7' in earlier


def test_callback_does_not_rediscover_idle_roots_after_consumption():
    text = ast.unparse(method('_freeze'))
    assert 'self._storage(owner, self.storage_roots)' in text
    assert 'report_only_preflight_storage_roots' not in text


def test_Git_pending_is_retained_inside_scope_without_infinite_poll():
    function = method('_freeze')
    assert not any(isinstance(row,ast.While) for row in ast.walk(function))
    assert 'self.git_pending = error' in ast.unparse(function)
    assert 'authenticated_ordinary_freezer_scope' in ast.unparse(function)


def test_cleanup_uses_exact_role_objects_not_owner_indices():
    text = ast.unparse(method('observe_pending'))
    assert "('freezer', 'preflight')" in text and "self.role_owners['provider']" in text
    assert 'self.owners[' not in text


def test_actual_stopped_public_owner_release_is_used():
    text = ast.unparse(method('_release'))
    assert 'owner.release(artifacts_durable=True)' in text
    assert 'owner.finalize(' not in text and 'owner.native_lease' not in text


def test_Docker_query_is_guarded_before_actual_query():
    tree = ast.parse(RUNTIME.read_text())
    lifecycle = next(row for row in tree.body if isinstance(row,ast.ClassDef) and row.name == '_Lifecycle')
    query = next(row for row in lifecycle.body if isinstance(row,ast.FunctionDef) and row.name == 'query')
    assert ast.unparse(query.body[0]) == '_ordinary_admin_capacity(self)'


def test_native_group_capture_precedes_watchdog_and_transport_attach():
    text = RUNTIME.read_text()
    assert text.index('_capture_native_group(self, self.watchdog_owner, lease._watchdog)') < text.index('child_pid=lease._watchdog.pid')
    assert text.index('_capture_native_group(self, self.owner, process)') < text.index('child_pid=process.pid')


def test_partial_stopped_release_does_not_reauthenticate_unreleased_pair():
    tree = ast.parse(RUNTIME.read_text())
    functions = {row.name:row for row in tree.body if isinstance(row,ast.FunctionDef)}
    text = ast.unparse(functions['acknowledge_stopped_ordinary_custody'])
    assert 'held_native_consumers' in text and 'require_bound_ordinary_custody_hold' not in text
    assert 'if not consumer.released' in text
    unstarted = ast.unparse(functions['acknowledge_unstarted_custody_hold'])
    assert '_UNSTARTED_ACKNOWLEDGMENTS' in unstarted and 'if not consumer.released' in unstarted


def test_retained_pending_cannot_unregister_original_borrow_after_stopped_observation():
    execute = method('execute_daemon_once')
    handler = next(row for row in ast.walk(execute) if isinstance(row,ast.ExceptHandler))
    guard = handler.body[1]
    assert isinstance(guard,ast.If) and 'NativeHostCustodyPending' in ast.unparse(guard.test)
    assert 'self.pending is not None' in ast.unparse(guard.test)
    assert ast.unparse(guard.body[-1]) == 'self._retain(error)'
    finish = method('_finish_borrow')
    assert 'self.pending is None' in ast.unparse(finish.body[0])
    assert 'original_finalizer_settlement_required_before_borrow_exit' in ast.unparse(finish.body[0])
