"""Cold refusal and exact source preservation only; no fake positive host.

Every executable probe below MUST refuse or raise its original failure. Probes
cannot complete identity/profile/native validation, create a host or grant a
pass. Successful actual same-pass operation needs a separate genuine fixture.
"""
import ast
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
SOURCE = HERE / 'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_native_host.py'
PREVIOUS = Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-settlement-20261017-01/native-host-v03/proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_native_host.py')
PREVIOUS_SHA = '0b76675f07959164adb5ef442e7df2218148674bad739379adb528ec1b2cc574'


@lru_cache(maxsize=1)
def target():
    # This module imports only stdlib. No product import/ctor or method granting
    # a positive binding/phase is executed. The enclosing job gates this import.
    spec = importlib.util.spec_from_file_location('source_only_sticky_host_refusals', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OriginalProbeFailure(BaseException):
    pass


class NegativeOnlyProbe:
    def __init__(self):
        self.pending, self.disposition_pending = None, False
        self._execution_attempted = True
        self._pass_entry_attempted = self._pass_entry_validated = False
        self.bound, self.finished, self.token = False, False, None
        self.daemon = object()
        self.effects = []
        self.failure = OriginalProbeFailure('original negative probe failure')

    def _forbidden(self, operation):
        self.effects.append(operation)
        raise self.failure

    def _validate_in_pass_profile(self, _daemon):
        self._forbidden('profile')

    def _binding_identity(self, _daemon):
        self._forbidden('binding')

    def _identity(self, _daemon):
        self._forbidden('database identity')

    def _phase_live(self):
        self._forbidden('native renewal')

    def _require_validated_in_pass(self, daemon):
        return target().ReportOnlyNativeHost._require_validated_in_pass(self, daemon)


@pytest.mark.parametrize('condition', ['pending', 'disposition', 'attempted', 'validated', 'not_executed'])
def test_public_guard_refuses_before_profile_or_identity(condition):
    module, probe = target(), NegativeOnlyProbe()
    original_pending = object()
    if condition == 'pending': probe.pending = original_pending
    elif condition == 'disposition': probe.disposition_pending = True
    elif condition == 'attempted': probe._pass_entry_attempted = True
    elif condition == 'validated': probe._pass_entry_validated = True
    else: probe._execution_attempted = False
    with pytest.raises(module.NativeHostRefusal):
        module.ReportOnlyNativeHost.guard_pass_entry(probe, probe.daemon)
    assert probe.effects == []
    if condition == 'pending': assert probe.pending is original_pending
    if condition == 'disposition': assert probe.disposition_pending is True


def test_failed_first_public_check_is_sticky_and_preserves_failure():
    module, probe = target(), NegativeOnlyProbe()
    with pytest.raises(OriginalProbeFailure) as observed:
        module.ReportOnlyNativeHost.guard_pass_entry(probe, probe.daemon)
    assert observed.value is probe.failure
    assert probe._pass_entry_attempted is True and probe._pass_entry_validated is False
    with pytest.raises(module.NativeHostRefusal, match='single_public_pass_already_attempted'):
        module.ReportOnlyNativeHost.guard_pass_entry(probe, probe.daemon)
    assert probe.effects == ['profile']


@pytest.mark.parametrize('condition', ['pending', 'disposition', 'not_attempted', 'not_validated', 'not_executed'])
def test_shared_productive_guard_refuses_before_binding_or_database(condition):
    module, probe = target(), NegativeOnlyProbe()
    probe._pass_entry_attempted = probe._pass_entry_validated = True
    if condition == 'pending': probe.pending = object()
    elif condition == 'disposition': probe.disposition_pending = True
    elif condition == 'not_attempted': probe._pass_entry_attempted = False
    elif condition == 'not_validated': probe._pass_entry_validated = False
    else: probe._execution_attempted = False
    with pytest.raises(module.NativeHostRefusal):
        module.ReportOnlyNativeHost._require_validated_in_pass(probe, probe.daemon)
    assert probe.effects == []


@pytest.mark.parametrize('operation', ['attempt', 'dispatch', 'owner', 'storage', 'freeze'])
def test_productive_entrypoints_refuse_pending_before_effects(operation):
    module, probe = target(), NegativeOnlyProbe()
    retained = probe.pending = object()
    probe._pass_entry_attempted = probe._pass_entry_validated = True
    cls = module.ReportOnlyNativeHost
    with pytest.raises(module.NativeHostRefusal):
        if operation == 'attempt': cls.require_attempt_profile(probe, probe.daemon, None, None)
        elif operation == 'dispatch':
            cls.dispatch_provider(probe, probe.daemon, state=None, task=None, attempt=2,
                prompt='', environment={}, log_path=None, command=())
        elif operation == 'owner': cls._owner(probe, 'forbidden', 1, 1, 1)
        elif operation == 'storage': cls._storage(probe, None, ())
        else: cls._freeze(probe, None)
    assert probe.effects == [] and probe.pending is retained


@pytest.mark.parametrize('condition', ['pending', 'disposition', 'attempted', 'finished'])
def test_outer_execute_refuses_before_native_renewal(condition):
    module, probe = target(), NegativeOnlyProbe()
    probe._execution_attempted = False
    if condition == 'pending': probe.pending = object()
    elif condition == 'disposition': probe.disposition_pending = True
    elif condition == 'attempted': probe._execution_attempted = True
    else: probe.finished = True
    with pytest.raises(module.NativeHostRefusal):
        module.ReportOnlyNativeHost.execute_daemon_once(probe)
    assert probe.effects == []


def test_outer_execute_first_phase_failure_cannot_renew_again():
    module, probe = target(), NegativeOnlyProbe()
    probe._execution_attempted = False
    with pytest.raises(OriginalProbeFailure) as observed:
        module.ReportOnlyNativeHost.execute_daemon_once(probe)
    assert observed.value is probe.failure and probe._execution_attempted is True
    with pytest.raises(module.NativeHostRefusal, match='fresh_single_host_pass_required'):
        module.ReportOnlyNativeHost.execute_daemon_once(probe)
    assert probe.effects == ['native renewal']


def test_missing_actual_binding_remains_refused():
    module = target()
    assert module._HOST.get() is None
    with pytest.raises(module.NativeHostRefusal, match='explicit_same_process_native_host_required'):
        module.require_report_only_native_host(object())


def dump(node):
    return ast.dump(node, include_attributes=False)


@lru_cache(maxsize=1)
def trees():
    previous = PREVIOUS.read_bytes()
    assert hashlib.sha256(previous).hexdigest() == PREVIOUS_SHA
    return ast.parse(previous), ast.parse(SOURCE.read_bytes())


def host_methods(tree):
    cls = next(row for row in tree.body if isinstance(row, ast.ClassDef) and row.name == 'ReportOnlyNativeHost')
    return {row.name: row for row in cls.body if isinstance(row, ast.FunctionDef)}


def function(tree, name):
    return next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name)


def calls(node, name):
    return [row for row in ast.walk(node) if isinstance(row, ast.Call)
            and getattr(row.func, 'attr', getattr(row.func, 'id', None)) == name]


def test_original_daemon_pin_and_exact_constructor_checks_are_preserved():
    before, after = trees()
    constant = lambda tree: next(row.value.value for row in tree.body if isinstance(row, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == 'DAEMON_SHA' for name in row.targets))
    assert constant(before) == constant(after) == '4984370456bd304dd9d89550c05e2d364fd9f0ca4849db560cb10caddfe3f407'
    old, new = host_methods(before)['__init__'], host_methods(after)['__init__']
    flags = {'_execution_attempted', '_pass_entry_attempted', '_pass_entry_validated'}
    retained = [row for row in new.body if not (isinstance(row, ast.Assign)
        and any(isinstance(name, ast.Attribute) and name.attr in flags for name in row.targets))]
    assert [dump(row) for row in old.body] == [dump(row) for row in retained]


def test_all_unedited_host_methods_and_native_cleanup_are_preserved():
    before, after = (host_methods(tree) for tree in trees())
    edited = {'__init__', '_identity', 'execute_daemon_once', 'guard_pass_entry', 'require_attempt_profile',
              '_owner', '_storage', 'dispatch_provider', '_freeze'}
    added = {'_binding_identity', '_validate_in_pass_profile', '_require_validated_in_pass'}
    assert set(after) == set(before) | added
    assert all(dump(row) == dump(after[name]) for name, row in before.items() if name not in edited)
    for name in ('observe_pending', 'settle_retained_finalizers', 'handle_attempt_exception',
                 '_drain_original_finalizers', '_run_original_finalizer', '_finish_borrow'):
        assert dump(before[name]) == dump(after[name])
    assert dump(function(trees()[0], 'run_original_finalizer')) == dump(function(trees()[1], 'run_original_finalizer'))


def test_original_profile_checks_moved_intact_to_repeatable_helper():
    before, after = (host_methods(tree) for tree in trees())
    assert [dump(row) for row in before['guard_pass_entry'].body] == [
        dump(row) for row in after['_validate_in_pass_profile'].body[1:]]
    old_tail = before['require_attempt_profile'].body[1:]
    new_tail = after['require_attempt_profile'].body[2:]
    assert [dump(row) for row in old_tail] == [dump(row) for row in new_tail]
    assert not calls(after['require_attempt_profile'], 'guard_pass_entry')


def test_exact_binding_identity_and_borrowed_connection_validation_are_preserved():
    before, after = (host_methods(tree) for tree in trees())
    assert [dump(row) for row in after['_binding_identity'].body] == [dump(before['_identity'].body[0])]
    assert [dump(row) for row in after['_identity'].body[1:]] == [dump(row) for row in before['_identity'].body[1:]]
    assert not calls(after['_binding_identity'], '_module')
    assert not calls(after['_binding_identity'], '_phase_live')


def test_public_accessor_has_no_database_or_phase_effect_before_guard():
    node = function(trees()[1], 'require_report_only_native_host')
    assert len(calls(node, '_binding_identity')) == 1
    assert not calls(node, '_identity') and not calls(node, '_module') and not calls(node, '_phase_live')
    assert dump(node.body[1]) == dump(function(trees()[0], 'require_report_only_native_host').body[1])


def test_latches_are_monotonic_and_precede_fallible_first_checks():
    methods = host_methods(trees()[1])
    flags = {'_execution_attempted', '_pass_entry_attempted', '_pass_entry_validated'}
    writes = {name: [] for name in flags}
    for method_name, node in methods.items():
        for row in ast.walk(node):
            if isinstance(row, ast.Assign):
                for name in row.targets:
                    if isinstance(name, ast.Attribute) and name.attr in flags:
                        writes[name.attr].append((method_name, ast.literal_eval(row.value)))
    assert writes['_execution_attempted'] == [('__init__', False), ('execute_daemon_once', True)]
    assert writes['_pass_entry_attempted'] == [('__init__', False), ('guard_pass_entry', True)]
    assert writes['_pass_entry_validated'] == [('__init__', False), ('guard_pass_entry', True)]
    for method_name, flag, fallible in (('execute_daemon_once', '_execution_attempted', '_phase_live'),
                                       ('guard_pass_entry', '_pass_entry_attempted', '_validate_in_pass_profile')):
        node = methods[method_name]
        latch = next(row for row in node.body if isinstance(row, ast.Assign)
                     and isinstance(row.targets[0], ast.Attribute) and row.targets[0].attr == flag)
        assert latch.lineno < calls(node, fallible)[0].lineno
        assert all(getattr(row.func, 'id', None) == 'need'
            or isinstance(row.func, ast.Attribute) and row.func.attr == 'get'
               and isinstance(row.func.value, ast.Name) and row.func.value.id == '_HOST'
            for row in ast.walk(ast.Module(
            body=[statement for statement in node.body if statement.end_lineno < latch.lineno], type_ignores=[]))
            if isinstance(row, ast.Call))


def test_productive_effect_producers_are_unchanged_after_early_deny():
    before, after = (host_methods(tree) for tree in trees())
    for name in ('_owner', '_storage', 'dispatch_provider', '_freeze'):
        new = after[name]
        assert len(calls(new.body[0], '_require_validated_in_pass')) == 1
        assert [dump(row) for row in new.body[1:]] == [dump(row) for row in before[name].body]
    helper = after['_require_validated_in_pass']
    assert len(calls(helper, '_binding_identity')) == 1
    assert not calls(helper, '_identity') and not calls(helper, '_phase_live') and not calls(helper, '_module')


def test_callback_and_source_native_validators_are_preserved():
    before, after = trees()
    assert dump(function(before, '_success_callback')) == dump(function(after, '_success_callback'))
    old, new = host_methods(before), host_methods(after)
    for name in ('_sources', '_module', '_phase_live', '_release'):
        assert dump(old[name]) == dump(new[name])
