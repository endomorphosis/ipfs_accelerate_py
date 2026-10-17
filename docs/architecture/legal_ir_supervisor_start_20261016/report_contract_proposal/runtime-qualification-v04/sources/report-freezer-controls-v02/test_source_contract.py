"""Inspect the real exact-baseline proposal without importing product code."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

import pytest

ROOT = Path('/home/barberb/lift_coding/artifacts/legal-ir-callable-handoff-20261017-01/report-freezer-v02')
SPEC = importlib.util.spec_from_file_location('report_freezer_patch_source_only', ROOT/'build_patch.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)
BASELINE = json.loads((ROOT/'BASELINE.json').read_text())
INPUTS = {Path(row['path']).name:row for row in BASELINE['inputs']}
HELPER = ROOT/'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_candidate_freezer.py'


def source(name):
    row = INPUTS[name]
    raw = Path(row['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['sha256']
    return raw.decode()


def daemon_tree():
    return ast.parse(b.daemon_successor(source('implementation_daemon.py')))


def methods(tree):
    daemon = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PortalImplementationDaemon')
    return {node.name:node for node in daemon.body if isinstance(node, ast.FunctionDef)}


def helper_functions():
    return {node.name:node for node in ast.parse(HELPER.read_text()).body if isinstance(node, ast.FunctionDef)}


def attribute_calls(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == name]


def test_patch_targets_exact_original_sources_and_successor_bytes():
    for name, successor in [('implementation_daemon.py', b.daemon_successor), ('git_checkout_transaction.py', b.lock_successor)]:
        row = INPUTS[name]
        proposed = successor(source(name)).encode()
        assert hashlib.sha256(proposed).hexdigest() == row['proposed_sha256']
        assert len(proposed) == row['proposed_bytes']
        ast.parse(proposed)


def test_public_api_cannot_take_caller_task_baseline_binding_or_git_facade():
    function = methods(daemon_tree())['freeze_report_only_candidate']
    assert [arg.arg for arg in function.args.kwonlyargs] == [
        'outcome', 'freezer_owner', 'protected_source_sha256', 'declared_external_storage']
    assert not function.decorator_list
    assert not attribute_calls(function, '_run_git')
    assert not attribute_calls(function, '_commit_worktree_changes_unchecked')


def test_original_generic_commit_uses_extracted_original_tail():
    nodes = methods(daemon_tree())
    assert len(attribute_calls(nodes['_commit_worktree_changes_unchecked'], '_commit_staged_worktree_changes')) == 1
    original = methods(ast.parse(source('implementation_daemon.py')))['_commit_worktree_changes_unchecked']
    # ast.walk is breadth first: its final commit call is the nested
    # submodule --allow-empty rescue, not the original top-level tail.
    original_tail = [statement.value for statement in original.body
        if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call)
        and isinstance(statement.value.func,ast.Attribute) and statement.value.func.attr=='_run_git'
        and statement.value.args and isinstance(statement.value.args[0],ast.List)
        and any(isinstance(x,ast.Constant) and x.value=='commit' for x in statement.value.args[0].elts)]
    assert len(original_tail)==1
    last_commit = original_tail[0]
    tail = nodes['_commit_staged_worktree_changes']
    copied = [n for n in ast.walk(tail) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_git'
              and n.args and isinstance(n.args[0],ast.List) and any(isinstance(x,ast.Constant) and x.value=='commit' for x in n.args[0].elts)]
    assert len(copied)==1 and ast.dump(copied[0].args[0])==ast.dump(last_commit.args[0])
    assert not attribute_calls(tail, '_commit_worktree_submodule_changes')
    # Preserve the original generic nested submodule path as well; the
    # report-only producer never enters it, and the extraction must not edit it.
    def rescues(node):
        return [ast.dump(call,include_attributes=False) for call in attribute_calls(node,'_run_git')
            if call.args and isinstance(call.args[0],ast.List)
            and any(isinstance(x,ast.Constant) and x.value=='--allow-empty' for x in call.args[0].elts)]
    assert len(rescues(original))==1
    assert rescues(original)==rescues(nodes['_commit_worktree_changes_unchecked'])


def test_opt_in_default_is_false_and_capture_precedes_attempt_consumption():
    nodes = methods(daemon_tree())
    init = nodes['__init__']
    parameters = [arg.arg for arg in init.args.kwonlyargs]
    assert ast.literal_eval(init.args.kw_defaults[parameters.index('report_only_candidate_freezing')]) is False
    start = nodes['_mark_implementation_started']
    captures = [n.lineno for n in ast.walk(start) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='capture_before_attempt']
    consumes = [n.lineno for n in attribute_calls(start, '_record_task_attempt')]
    assert len(captures)==len(consumes)==1 and captures[0] < consumes[0]


def test_dispatch_capture_happens_after_original_save_and_needs_real_objects():
    node = methods(daemon_tree())['_mark_provider_launch_boundary']
    saves = [n.lineno for n in attribute_calls(node, 'save')]
    captures = [n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='capture_dispatch']
    assert len(saves)==len(captures)==1 and saves[0] < captures[0].lineno
    assert [arg.id for arg in captures[0].args][-3:] == ['baseline_ref','callable_invocation','control_plane_connection']


def test_native_entry_uses_one_original_decision_callback_and_factored_lock():
    functions = helper_functions()
    freeze = functions['freeze']
    assert len(attribute_calls(freeze, '_decision_runtime_mutation')) == 1
    scopes = [node for node in ast.walk(freeze) if isinstance(node,ast.With)]
    assert len(scopes)==1 and isinstance(scopes[0].items[0].context_expr.func, ast.Name)
    assert scopes[0].items[0].context_expr.func.id == 'lock'
    assert '_freeze_locked' in ast.unparse(scopes[0])
    assert not attribute_calls(freeze, '_run_git')


def test_factored_lock_reuses_identical_original_lock_body():
    original = ast.parse(source('git_checkout_transaction.py'))
    proposed = ast.parse(b.lock_successor(source('git_checkout_transaction.py')))
    old = next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='checkout_git_mutation')
    new = next(n for n in proposed.body if isinstance(n,ast.FunctionDef) and n.name=='checkout_git_directory_mutation')
    def tail(node):
        start = next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='key' for t in n.targets))
        return [ast.dump(n,include_attributes=False) for n in node.body[start:]]
    assert tail(old)==tail(new)
    wrapper = next(n for n in proposed.body if isinstance(n,ast.FunctionDef) and n.name=='checkout_git_mutation')
    assert 'git_checkout_directory' in ast.unparse(wrapper) and 'checkout_git_directory_mutation' in ast.unparse(wrapper)


def test_only_explicit_report_add_and_shared_tail_in_locked_path():
    node = helper_functions()['_freeze_locked']
    adds=[]
    for call in attribute_calls(node, 'run'):
        if call.args and isinstance(call.args[0], ast.List):
            values=[]
            for arg in call.args[0].elts:
                values.append(arg.value if isinstance(arg,ast.Constant) else ast.unparse(arg))
            if values and values[0]=='add': adds.append(values)
    assert adds == [['add','--sparse','-f','--','REPORT']]
    assert len(attribute_calls(node, '_commit_staged_worktree_changes'))==1
    text=ast.unparse(node)
    for forbidden in ('_commit_worktree_changes_unchecked','_validated_existing_worktree_commit','submodule','reset','checkout','merge'):
        assert forbidden not in [call.func.attr for call in ast.walk(node) if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)]
    assert text.index('_require_callback') < text.index("['add'") < text.index('_commit_staged_worktree_changes')


def test_current_database_projection_has_select_only_no_backend_creation_or_reconnect():
    node=helper_functions()['_read_bound_control_plane']
    commands=[]
    for call in attribute_calls(node,'execute'):
        commands.append(ast.literal_eval(call.args[0]))
    assert commands == [
        'SELECT payload_json FROM artefact_generic WHERE board_namespace = ? AND kind = ? AND artefact_id = ?',
        'SELECT body_json FROM portal_task_state WHERE lane_key = ?']
    text=ast.unparse(node)
    assert 'with connection._execution_condition:' in text
    for marker in ('_poisoned is False','_transaction_active is False','_owner_binding_retired is False','_default_catalog is None'):
        assert marker in text
    for call in ast.walk(node):
        if isinstance(call,ast.Call):
            name=call.func.attr if isinstance(call.func,ast.Attribute) else call.func.id if isinstance(call.func,ast.Name) else ''
            assert name not in {'open','connect','_connect','open_duckdb_connection','load_portal_task_state','save','reconnect_exclusive_owner'}


def test_no_validation_completion_or_task_state_write_in_freezer_source():
    tree=ast.parse(HELPER.read_text())
    for call in ast.walk(tree):
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute):
            assert call.func.attr not in {'save','mark_completed','accept','merge','reset','issue_handle','create_worktree'}
    text=HELPER.read_text()
    for marker in ('"native_validation_qualified": False','"validation_passed": False','"completion_authority": False',
                   '"canonical_tasks_accepted": 0'):
        assert marker in text


def test_four_slot_profile_and_automatic_git_children_disabled():
    text=HELPER.read_text()
    assert 'lease.get("child_process_slots") == 4' in text
    for value in ('gc.auto=0','maintenance.auto=false','core.hooksPath=/dev/null','commit.gpgsign=false',
                  'tag.gpgsign=false','diff.external=','core.fsmonitor=false','core.untrackedCache=false'):
        assert value in text
    assert 'actual_git_group_exceeded_four_slots' in text
    assert 'git_config_include_or_external_filter_refused' in text


def test_single_actual_git_launch_is_attached_before_gate_release():
    tree=ast.parse(HELPER.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='_Git')
    run=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='run')
    launches=attribute_calls(run,'Popen')
    assert len(launches)==1 and isinstance(launches[0].func.value,ast.Name) and launches[0].func.value.id=='subprocess'
    attaches=[node for node in attribute_calls(run,'check_usage') if any(k.arg=='child_pid' for k in node.keywords)]
    gates=[node for node in attribute_calls(run,'write') if node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value==b'G']
    assert len(attaches)==len(gates)==1 and launches[0].lineno < attaches[0].lineno < gates[0].lineno
    assert not any(isinstance(node,ast.For) and launches[0] in list(ast.walk(node)) for node in ast.walk(run))
