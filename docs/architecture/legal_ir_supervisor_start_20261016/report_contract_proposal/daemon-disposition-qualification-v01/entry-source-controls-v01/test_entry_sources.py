"""Source composition checks only; target daemon bytecode is never executed."""
import ast
import copy
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).absolute().parent.parent
PREV = ROOT.parent/'legal-ir-daemon-settlement-20261017-01'


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def sources():
    builder = source_module(ROOT/'daemon-integration-v04/build_patch.py', 'entry_assembler_only')
    return builder.assemble()


def dump(node):
    return ast.dump(node, include_attributes=False)


def constant(tree, name):
    return next(row.value.value for row in tree.body if isinstance(row, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in row.targets))


def method(tree, name):
    cls = next(row for row in tree.body if isinstance(row, ast.ClassDef)
               and row.name == 'PortalImplementationDaemon')
    return next(row for row in cls.body if isinstance(row, ast.FunctionDef) and row.name == name)


def test_only_two_exact_readonly_entry_checks_change_the_complete_daemon():
    _original, previous, proposed = sources()
    tree = ast.parse(proposed)
    for name in ('run_once', '_run_implementation'):
        node = method(tree, name)
        guards = [row for row in node.body if isinstance(row, ast.If)
                  and dump(row.test) == dump(ast.parse('self.report_only_candidate_freezing', mode='eval').body)]
        assert len(guards) == 1
        guard = guards[0]
        assert len(guard.body) == 3 and isinstance(guard.body[0], ast.ImportFrom)
        assert guard.body[0].module == 'report_only_disposition_gate'
        assert dump(guard.body[1]) == dump(ast.parse('require_clear_report_only_disposition(self)').body[0])
        guard.body = guard.body[2:]
    # This covers all original counters, acceptance, SQL, kernel and default
    # behavior, rather than enumerating only a favorable subset of methods.
    assert dump(tree) == dump(ast.parse(previous))


def test_both_original_finally_bodies_are_still_exact():
    checker = source_module(PREV/'settlement-source-controls-v02/check_finalizer_body_preservation.py',
                            'exact_original_finally_comparison')
    original, _previous, proposed = sources()
    result = checker.compare(original.encode(), proposed.encode())
    assert len(result['checks']) == 2
    assert all(row['unchanged_original_body'] for row in result['checks'])


def test_public_and_private_body_deny_precedes_host_and_original_dispatch():
    tree = ast.parse(sources()[2])
    for name in ('run_once', '_run_implementation'):
        node = method(tree, name)
        body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else node.body
        assert isinstance(body[0], ast.If)
        assert len(body[0].body) == 3
        assert body[0].body[1].value.func.id == 'require_clear_report_only_disposition'
        host_call = body[0].body[2].value
        assert isinstance(host_call.func.value, ast.Call)
        assert host_call.func.value.func.attr == '_require_report_only_native_host'
        assert isinstance(body[1], ast.Assign)
        assert body[1].value.func.attr == '_retained_provider_capacity_deferral'
    # Private decoration is explicitly preserved, not claimed to run after its
    # new body guard. The public run_once guard precedes that private boundary.
    assert dump(method(tree, '_run_implementation').decorator_list[0]) == dump(
        method(ast.parse(sources()[1]), '_run_implementation').decorator_list[0])


def test_baked_pin_successor_preserves_all_qualified_host_code():
    suffix = 'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_native_host.py'
    before = ast.parse((ROOT/'native-host-v04'/suffix).read_bytes())
    after = ast.parse((ROOT/'native-host-v05'/suffix).read_bytes())
    proposed_sha = hashlib.sha256(sources()[2].encode()).hexdigest()
    assert constant(after, 'DAEMON_SHA') == proposed_sha
    for row in before.body:
        if isinstance(row, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DAEMON_SHA' for t in row.targets):
            row.value = ast.Constant(proposed_sha)
    assert dump(before) == dump(after)


def test_gate_requires_actual_source_scope_and_never_returns_a_grant():
    path = ROOT/'entry-gate-v01/proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_disposition_gate.py'
    tree = ast.parse(path.read_bytes())
    fn = next(row for row in tree.body if isinstance(row, ast.FunctionDef))
    calls = [row for row in ast.walk(fn) if isinstance(row, ast.Call)]
    assert len([row for row in calls if getattr(row.func, 'attr', '') == 'OriginalDispositionScope']) == 1
    assert len([row for row in calls if getattr(row.func, 'attr', '') == 'assert_no_denials_for_state']) == 1
    forbidden = {'connect', 'execute', 'Popen', 'acquire_child', 'publish', 'quarantine_current_owner',
                 'mark_terminal', 'close', '_mark_implementation_started'}
    assert not {getattr(row.func, 'attr', '') for row in calls}.intersection(forbidden)
    returns = [row for row in ast.walk(fn) if isinstance(row, ast.Return)]
    assert len(returns) == 1 and isinstance(returns[0].value, ast.Constant) and returns[0].value.value is None


def test_gate_expected_source_pins_match_the_composed_modules():
    tree = ast.parse((ROOT/'entry-gate-v01/proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_disposition_gate.py').read_bytes())
    helper = ROOT/'attempt-disposition-v01/proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/attempt_disposition.py'
    assert constant(tree, 'DAEMON_SHA') == hashlib.sha256(sources()[2].encode()).hexdigest()
    assert constant(tree, 'DISPOSITION_SHA') == hashlib.sha256(helper.read_bytes()).hexdigest()
    assert constant(tree, 'LIFECYCLE_SHA') == 'b7f8508b4b4b50d5bc259c186b55c4c6f87e938d99689a913383a8a35130249f'
