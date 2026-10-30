"""Cold necessary-condition/source cases only; no native/Git/provider effects."""
import ast
import importlib.util
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).absolute().parent
SOURCE = HERE / 'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_candidate_freezer.py'
QUALIFIED = Path('/home/barberb/lift_coding/artifacts/legal-ir-callable-handoff-20261017-01/report-freezer-v03/proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_candidate_freezer.py')
SPEC = importlib.util.spec_from_file_location('cold_dual_board_freezer_controls', SOURCE)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def definitions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def test_relative_board_is_derived_from_original_repository():
    assert M.board_relative_path('/original/repo', '/original/repo/docs/tasks.todo.md') == 'docs/tasks.todo.md'


@pytest.mark.parametrize('repo,board,reason', [
    ('/original/repo', '/outside/tasks.todo.md', 'authoritative_board_outside_original_repository'),
    ('/original/repo', '/original/repository/tasks.todo.md', 'authoritative_board_outside_original_repository'),
    ('original/repo', '/original/repo/tasks.todo.md', 'canonical_original_board_layout_required'),
    ('/original/repo', 'tasks.todo.md', 'canonical_original_board_layout_required'),
    ('/original/repo', '/original/repo/../tasks.todo.md', 'canonical_original_board_layout_required'),
    ('/original/repo', '/original/repo', 'supported_authoritative_board_relative_path_required'),
    ('/original/repo', '/original/repo/.git/tasks.todo.md', 'supported_authoritative_board_relative_path_required'),
    ('/original/repo', '/original/repo/docs/task board.md', 'supported_authoritative_board_relative_path_required'),
])
def test_changed_or_unsupported_original_board_relation_refuses(repo, board, reason):
    with pytest.raises(M.ReportFreezerRefusal, match=reason):
        M.board_relative_path(repo, board)


FLAGS = b'H a\0H docs/tasks.todo.md\0'
STAGE = b'100644 ' + b'1' * 40 + b' 0\ta\0' + b'100644 ' + b'2' * 40 + b' 0\tdocs/tasks.todo.md\0'


def test_regular_unique_zero_stage_index_shape_is_only_necessary():
    assert M.check_supported_index(FLAGS, STAGE) is None


@pytest.mark.parametrize('flags,stage,reason', [
    (FLAGS[:-1], STAGE, 'bounded_nul_terminated_index_inventory_required'),
    (FLAGS, STAGE[:-1], 'bounded_nul_terminated_index_inventory_required'),
    ('not bytes', STAGE, 'bounded_nul_terminated_index_inventory_required'),
    (FLAGS, None, 'bounded_nul_terminated_index_inventory_required'),
    (FLAGS.replace(b'H a', b'S a'), STAGE, 'hidden_index_or_submodule_flag_refused'),
    (FLAGS.replace(b'H a', b'h a'), STAGE, 'hidden_index_or_submodule_flag_refused'),
    (FLAGS, STAGE.replace(b'100644', b'120000', 1), 'submodule_symlink_or_unmerged_index_refused'),
    (FLAGS, STAGE.replace(b'100644', b'160000', 1), 'submodule_symlink_or_unmerged_index_refused'),
    (FLAGS, STAGE.replace(b' 0\t', b' 1\t', 1), 'submodule_symlink_or_unmerged_index_refused'),
    (FLAGS, STAGE.replace(b'1' * 40, b'1' * 39, 1), 'submodule_symlink_or_unmerged_index_refused'),
    (FLAGS + b'H a\0', STAGE, 'exact_unique_tracked_index_inventory_required'),
    (FLAGS, STAGE + STAGE.split(b'\0')[0] + b'\0', 'exact_unique_tracked_index_inventory_required'),
    (b'H a\0', STAGE, 'exact_unique_tracked_index_inventory_required'),
])
def test_hidden_submodule_unmerged_or_malformed_index_refuses(flags, stage, reason):
    with pytest.raises(M.ReportFreezerRefusal, match=reason):
        M.check_supported_index(flags, stage)


@pytest.mark.parametrize('name', [b'/absolute', b'../escape', b'a/../escape', b'a//b', b'.git/config', b'a\nb', b'a\tb'])
def test_aliased_tracked_names_refuse(name):
    with pytest.raises(M.ReportFreezerRefusal, match='unaliased_tracked_index_paths_required'):
        M.check_supported_index(b'H ' + name + b'\0', b'100644 ' + b'1' * 40 + b' 0\t' + name + b'\0')


def test_symbolic_original_branch_shape_is_only_necessary():
    assert M.check_symbolic_head(b'ref: refs/heads/work/legal-ir-r03\n', 'work/legal-ir-r03') is None


@pytest.mark.parametrize('raw,branch', [
    (b'1' * 40 + b'\n', 'main'),
    (b'ref: refs/heads/other\n', 'main'),
    (b'ref: refs/heads/main', 'main'),
    ('ref: refs/heads/main\n', 'main'),
    (b'ref: refs/heads/../other\n', '../other'),
    (b'ref: refs/heads/a//b\n', 'a//b'),
    (b'ref: refs/heads/\n', ''),
])
def test_changed_detached_or_aliased_symbolic_head_refuses(raw, branch):
    with pytest.raises(M.ReportFreezerRefusal, match='supported_original_symbolic_branch_head_required'):
        M.check_symbolic_head(raw, branch)


def test_qualified_native_git_topology_and_pending_observer_remain_exact():
    before, after = definitions(QUALIFIED), definitions(SOURCE)
    for name in ('_Git', 'GitCustodyPending', '_git_process', 'check_git_anchor_projection',
                 '_git_group_members', 'acknowledge_stopped_git_custody'):
        assert ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False)
    get = lambda path: {n.targets[0].id: n.value for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    for name in ('BARRIER', 'ANCHOR', 'MAX_GIT_CHILDREN'):
        assert ast.dump(get(QUALIFIED)[name], include_attributes=False) == ast.dump(get(SOURCE)[name], include_attributes=False)


def test_normative_counter_report_source_storage_guards_remain_exact():
    before, after = definitions(QUALIFIED), definitions(SOURCE)
    for name in ('budget_projection', '_task_projection', 'strict_report', 'parse_tree', 'parse_diff',
                 'parse_commit', 'allowed_dirty_paths', '_guard_owner', '_verify_sources',
                 '_discover_git_storage', '_storage_roots', '_check_storage', '_index_snapshot'):
        assert ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False)


def test_dispatch_accepts_no_caller_board_baseline_or_workspace():
    node = definitions(SOURCE)['capture_dispatch']
    assert [a.arg for a in node.args.args] == ['daemon', 'state', 'task', 'attempt', 'invocation', 'control_plane_connection']
    text = ast.unparse(node)
    assert 'preflight.baseline' in text and 'preflight.workspace' in text and 'preflight.board_bytes' in text


def test_mark_requires_exact_preflight_before_original_mutations():
    node = definitions(SOURCE)['capture_before_attempt']
    assert {a.arg for a in node.args.kwonlyargs} == {'consume_attempt', 'worktree_path', 'branch_name'}
    text = ast.unparse(node)
    assert 'before_consumption=True' in text and 'canonical(worktree_path) == preflight.workspace' in text
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in
                   {'save', '_record_task_attempt'} for n in ast.walk(node))


def test_preflight_checks_source_storage_and_genuine_owner_before_any_git():
    node = definitions(SOURCE)['preflight_report_only_candidate']
    calls = {name: min(n.lineno for n in ast.walk(node) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == name) for name in
        ('_verify_sources', '_guard_preflight_owner', '_check_storage', '_Git')}
    assert calls['_verify_sources'] < calls['_Git'] and calls['_guard_preflight_owner'] < calls['_Git']
    assert calls['_check_storage'] < calls['_Git']
    assignment = next(n for n in node.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == '_report_only_preflight' for t in n.targets))
    with_node = next(n for n in node.body if isinstance(n, ast.With))
    assert assignment.lineno > with_node.end_lineno


def test_metadata_roots_never_start_git_or_acquire_resources():
    names = ('report_only_preflight_storage_roots', '_metadata_storage', '_board_layout', '_original_idle_binding')
    for name in names:
        node = definitions(SOURCE)[name]
        assert not any(isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == '_Git') or
            (isinstance(n.func, ast.Attribute) and n.func.attr in {'Popen', 'acquire', 'acquire_child', 'save', 'close'}))
            for n in ast.walk(node))


def test_original_connection_is_required_in_every_control_plane_read():
    node = definitions(SOURCE)['_read_bound_control_plane']
    text = ast.unparse(node)
    assert '_original_connection(daemon, connection)' in text
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in {'connect', 'close', 'save', 'load', '_connect'} for n in ast.walk(node))


def test_before_consumption_checks_refs_and_index_without_new_git_child():
    node = definitions(SOURCE)['_require_preflight']
    text = ast.unparse(node)
    assert '_head_ref_snapshot' in text and '_index_snapshot' in text and 'before_consumption' in text
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_Git'
                   for n in ast.walk(node))


def test_candidate_retains_both_board_and_original_blob_correspondence():
    node = definitions(SOURCE)['_freeze_locked']
    text = ast.unparse(node)
    for item in ('authoritative_repository_root', 'authoritative_board_path', 'authoritative_board_sha256',
                 'board_relative_path', 'baseline_board_blob_oid'):
        assert item in text
    assert 'board_oid == capture.preflight.board_oid' in text
