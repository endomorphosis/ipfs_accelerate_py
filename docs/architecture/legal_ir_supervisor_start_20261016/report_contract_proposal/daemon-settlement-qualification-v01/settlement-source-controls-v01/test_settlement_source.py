"""Authored source controls; no target daemon import or positive runtime grant."""
import ast
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
from types import CodeType


ROOT = Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-settlement-20261017-01')
BUILDER = ROOT / 'daemon-integration-v03/build_patch.py'
HOST = ROOT / 'native-host-v03/proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_native_host.py'
AUDIT = ROOT / 'settlement-audit-v01'
HELPER = AUDIT / 'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/original_finalizer_continuations.py'
EXPECTED_BUILDER = 'f5199a384dfc8ab642582bb68fe78765f61b6958471597180aab4f25c270006e'
EXPECTED_HOST = '0b76675f07959164adb5ef442e7df2218148674bad739379adb528ec1b2cc574'
EXPECTED_COMPOSED = '4984370456bd304dd9d89550c05e2d364fd9f0ca4849db560cb10caddfe3f407'


def load_source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def sources():
    assert hashlib.sha256(BUILDER.read_bytes()).hexdigest() == EXPECTED_BUILDER
    assert hashlib.sha256(HOST.read_bytes()).hexdigest() == EXPECTED_HOST
    builder = load_source_module(BUILDER, 'settlement_source_assembler_only')
    original, previous, proposed = builder.assemble()
    assert hashlib.sha256(proposed.encode()).hexdigest() == EXPECTED_COMPOSED
    return original, previous, proposed


@lru_cache(maxsize=1)
def daemon_trees():
    return tuple(ast.parse(source) for source in sources())


@lru_cache(maxsize=1)
def host_tree():
    return ast.parse(HOST.read_text())


def cls(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def method(tree, name, class_name='PortalImplementationDaemon'):
    return next(node for node in cls(tree, class_name).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def host_method(name):
    return method(host_tree(), name, 'ReportOnlyNativeHost')


def dump(node):
    return ast.dump(node, include_attributes=False)


def calls(node, attribute):
    return [row for row in ast.walk(node) if isinstance(row, ast.Call)
            and isinstance(row.func, ast.Attribute) and row.func.attr == attribute]


@lru_cache(maxsize=1)
def body_comparison():
    checker = load_source_module(AUDIT/'check_finalizer_body_preservation.py',
                                 'source_body_comparison_only')
    original, _previous, proposed = sources()
    return checker.compare(original.encode(), proposed.encode())


def test_original_claim_finalizer_body_is_preserved():
    rows = {row['stage']: row for row in body_comparison()['checks']}
    assert rows['implementation_claim_release']['unchanged_original_body'] is True


def test_original_dispatch_intent_finalizer_body_is_preserved():
    rows = {row['stage']: row for row in body_comparison()['checks']}
    assert rows['pass_dispatch_intent_reconciliation']['unchanged_original_body'] is True


def test_all_other_daemon_class_definitions_are_unchanged_from_V02():
    _original, previous, proposed = daemon_trees()
    before = cls(previous, 'PortalImplementationDaemon')
    after = cls(proposed, 'PortalImplementationDaemon')
    names = {'_run_implementation', '_run_once'}
    select = lambda node: [dump(row) for row in node.body
                           if not isinstance(row, ast.FunctionDef) or row.name not in names]
    assert select(before) == select(after)


def test_original_counter_and_refund_producers_are_unchanged():
    original, _previous, proposed = daemon_trees()
    names = ('_record_task_attempt', '_restore_task_attempt', '_release_unfinished_active_attempt')
    assert all(dump(method(original, name)) == dump(method(proposed, name)) for name in names)


def test_candidate_acceptance_and_completion_producers_are_unchanged():
    original, _previous, proposed = daemon_trees()
    names = ('_board_completion_decision', '_mark_task_or_bundle_completed_in_todo')
    assert all(dump(method(original, name)) == dump(method(proposed, name)) for name in names)


def test_original_ephemeral_kernel_and_provider_order_is_unchanged_from_V02():
    _original, previous, proposed = daemon_trees()
    name = '_run_implementation_in_ephemeral_worktree'
    assert dump(method(previous, name)) == dump(method(proposed, name))


def test_continuations_are_created_in_original_finally_sites_only():
    _original, _previous, proposed = daemon_trees()
    expected = {'_run_implementation': '_settle_report_only_implementation_claims',
                '_run_once': '_settle_report_only_dispatch_intents'}
    for name, nested_name in expected.items():
        node = method(proposed, name)
        sites = [row for row in ast.walk(node) if isinstance(row, ast.Try)
                 and row.finalbody and isinstance(row.finalbody[0], ast.FunctionDef)
                 and row.finalbody[0].name == nested_name]
        assert len(sites) == 1
        assert len(sites[0].finalbody) == 2
        branch = sites[0].finalbody[1]
        assert isinstance(branch, ast.If) and dump(branch.test) == dump(ast.parse('self.report_only_candidate_freezing', mode='eval').body)
        assert len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.Expr)
        assert isinstance(branch.orelse[0].value, ast.Call)
        assert isinstance(branch.orelse[0].value.func, ast.Name)
        assert branch.orelse[0].value.func.id == nested_name


def test_original_claim_cell_assignment_uses_only_required_nonlocal():
    _original, _previous, proposed = daemon_trees()
    node = method(proposed, '_run_implementation')
    nested = next(row for row in ast.walk(node) if isinstance(row, ast.FunctionDef)
                  and row.name == '_settle_report_only_implementation_claims')
    declarations = [row.names for row in nested.body if isinstance(row, ast.Nonlocal)]
    assert declarations == [['acquired_resource_claims']]


def test_original_finalizer_frame_carriers_are_preserved():
    _original, previous, proposed = daemon_trees()
    for name in ('_run_implementation', '_run_once'):
        old = calls(method(previous, name), 'before_original_finalizer')
        assert len(old) == 1
        old_frame = next(row.value for row in old[0].keywords if row.arg == 'original_frame')
        new = [row for row in ast.walk(method(proposed, name)) if isinstance(row, ast.Call)
               and isinstance(row.func, ast.Name) and row.func.id == 'run_original_finalizer']
        assert len(new) == 1
        new_frame = next(row.value for row in new[0].keywords if row.arg == 'original_frame')
        assert dump(old_frame) == dump(new_frame)


def test_compiled_original_source_has_exact_zero_argument_closure_constants():
    # Compile only. No target top-level bytecode, decorator or method is executed.
    compiled = compile(sources()[2], '<source-only-composed-daemon>', 'exec')
    class_code = next(value for value in compiled.co_consts if type(value) is CodeType
                      and value.co_name == 'PortalImplementationDaemon')
    for method_name, nested_name in (
        ('_run_implementation', '_settle_report_only_implementation_claims'),
        ('_run_once', '_settle_report_only_dispatch_intents')):
        parent = next(value for value in class_code.co_consts if type(value) is CodeType
                      and value.co_name == method_name)
        nested = [value for value in parent.co_consts if type(value) is CodeType
                  and value.co_name == nested_name]
        assert len(nested) == 1 and nested[0].co_argcount == nested[0].co_kwonlyargcount == 0
        assert 'self' in nested[0].co_freevars


def test_public_wrapper_registers_directly_after_retaining_original_frame():
    node = next(row for row in host_tree().body if isinstance(row, ast.FunctionDef)
                and row.name == 'run_original_finalizer')
    register = calls(node, 'register')
    assert len(register) == 1
    append = calls(node, 'append')
    assert len(append) == 1 and append[0].lineno < register[0].lineno
    caller = next(row for row in node.body if isinstance(row, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'caller' for target in row.targets))
    assert isinstance(caller.value, ast.Attribute) and caller.value.attr == 'f_back'
    assert caller.lineno < append[0].lineno
    assert len(register[0].args) == 4 and [row.id for row in register[0].args] == ['stage', 'continuation', 'caller', 'original_frame']
    assert not calls(host_method('_run_original_finalizer'), 'register')


def test_registration_exception_retains_original_host():
    node = next(row for row in host_tree().body if isinstance(row, ast.FunctionDef)
                and row.name == 'run_original_finalizer')
    protected = [row for row in node.body if isinstance(row, ast.Try) and calls(row, 'register')]
    assert len(protected) == 1 and len(protected[0].handlers) == 1
    handler = protected[0].handlers[0]
    assert isinstance(handler.type, ast.Name) and handler.type.id == 'BaseException'
    assert len(calls(handler, '_retain')) == 1
    assert len(calls(node, '_run_original_finalizer')) == 1
    assert calls(node, '_run_original_finalizer')[0].lineno > protected[0].end_lineno


def test_original_host_checks_stop_before_any_finalizer_drain():
    node = host_method('_run_original_finalizer')
    identity = calls(node, '_identity')
    latch = calls(node, '_latch_disposition')
    observe = calls(node, 'observe_pending')
    drain = calls(node, '_drain_original_finalizers')
    assert len(identity) == len(latch) == len(observe) == len(drain) == 1
    assert identity[0].lineno < latch[0].lineno < observe[0].lineno < drain[0].lineno
    stopped_branch = next(row for row in node.body if isinstance(row, ast.If) and calls(row.test, 'observe_pending'))
    assert isinstance(stopped_branch.test, ast.UnaryOp) and isinstance(stopped_branch.test.op, ast.Not)
    assert len(calls(stopped_branch, '_retain')) == 1


def test_borrow_exit_guard_and_unregister_failures_retain():
    node = host_method('_finish_borrow')
    protected = next(row for row in node.body if isinstance(row, ast.Try))
    required = {'original_finalizer_settlement_required_before_borrow_exit',
                'original_attempt_disposition_required_before_borrow_exit',
                'all_original_finalizers_required_before_borrow_exit',
                'actual_host_must_be_stopped_before_borrow_exit'}
    found = {row.value for row in ast.walk(ast.Module(body=protected.body, type_ignores=[]))
             if isinstance(row, ast.Constant) and type(row.value) is str}
    assert required <= found and len(calls(protected, '__exit__')) == 1
    handler = protected.handlers[0]
    assert isinstance(handler.type, ast.Name) and handler.type.id == 'BaseException'
    assert len(calls(handler, 'append')) == len(calls(handler, '_retain')) == 1
    assert all(row.lineno > protected.end_lineno for row in calls(node, 'reset') + calls(node, 'pop'))


def test_disposition_is_monotonic_and_only_constructor_initializes_false():
    owner = cls(host_tree(), 'ReportOnlyNativeHost')
    assignments = []
    for method_node in owner.body:
        if not isinstance(method_node, ast.FunctionDef):
            continue
        for row in ast.walk(method_node):
            if isinstance(row, ast.Assign) and any(isinstance(target, ast.Attribute)
                    and target.attr == 'disposition_pending' for target in row.targets):
                assignments.append((method_node.name, row.value))
    assert {name for name, _ in assignments} == {'__init__', '_latch_disposition'}
    latch = next(value for name, value in assignments if name == '_latch_disposition')
    assert isinstance(latch, ast.Call) and isinstance(latch.func, ast.Name) and latch.func.id == 'bool'
    assert isinstance(latch.args[0], ast.BoolOp) and isinstance(latch.args[0].op, ast.Or)
    first = latch.args[0].values[0]
    assert isinstance(first, ast.Attribute) and first.attr == 'disposition_pending'


def test_retained_settlement_returns_only_borrow_retained_diagnostics():
    node = host_method('settle_retained_finalizers')
    assert not calls(node, '_finish_borrow')
    assert not calls(node, 'dispatch_provider') and not calls(node, '_freeze') and not calls(node, '_owner')
    results = [row.value for row in ast.walk(node) if isinstance(row, ast.Return)]
    assert len(results) == 2
    for result in results:
        assert isinstance(result, ast.Call) and isinstance(result.func, ast.Name) and result.func.id == 'dict'
        pairs = {item.arg: item.value for item in result.keywords}
        assert isinstance(pairs['original_borrow_retained'], ast.Constant) and pairs['original_borrow_retained'].value is True
        assert isinstance(pairs['acceptance_authority'], ast.Constant) and pairs['acceptance_authority'].value is False
        assert isinstance(pairs['canonical_tasks_accepted'], ast.Constant) and pairs['canonical_tasks_accepted'].value == 0


def test_host_does_not_close_original_caller_connection():
    node = cls(host_tree(), 'ReportOnlyNativeHost')
    forbidden = [row for row in calls(node, 'close') if isinstance(row.func.value, ast.Attribute)
                 and row.func.value.attr == 'connection']
    assert forbidden == []


def test_retained_finalizer_helper_is_bound_to_actual_original_module_and_daemon():
    node = host_method('__init__')
    construction = calls(node, 'OriginalFinalizerContinuations')
    assert len(construction) == 1
    assert [row.id for row in construction[0].args] == ['module', 'daemon']
    assert len(calls(node, '_sources')) == 1 and calls(node, '_sources')[0].lineno < construction[0].lineno


def test_original_dispatch_and_acceptance_routes_are_not_called_by_settlement():
    for name in ('_run_original_finalizer', '_drain_original_finalizers', 'settle_retained_finalizers'):
        node = host_method(name)
        forbidden = {'_record_task_attempt', '_restore_task_attempt', '_mark_implementation_started',
                     '_mark_implementation_finished', '_board_completion_decision',
                     '_mark_task_or_bundle_completed_in_todo', '_decision_runtime_completion',
                     'run_ordinary_with_success_handoff', 'freeze_report_only_candidate'}
        assert all(not calls(node, name) for name in forbidden)


def test_finalizer_pending_cause_is_retained_and_not_reset_by_drain():
    node = host_method('_drain_original_finalizers')
    assert len(calls(node, 'drain')) == 1
    handler = next(row for row in ast.walk(node) if isinstance(row, ast.ExceptHandler))
    assert isinstance(handler.type, ast.Name) and handler.type.id == 'BaseException'
    assert len(calls(handler, 'append')) == len(calls(handler, '_retain')) == 1
