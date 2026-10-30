"""Cold necessary-condition/source checks; no native/model/Git processes."""
import ast
import importlib.util
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).absolute().parent
OLD = HERE.parent / 'report-freezer-v02/proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_candidate_freezer.py'
NEW = HERE / 'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_candidate_freezer.py'
NATIVE = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py')
SPEC = importlib.util.spec_from_file_location('cold_git_anchor_source_controls', NEW)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def defs(path):
    return {node.name: node for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def test_pure_valid_projection_is_necessary_shape_only():
    assert M.check_git_anchor_projection(dict(pid=10, birth='200', state='S', parent_pid=5,
        group_pid=10, session_pid=4), pid=10, birth='200', parent_pid=5, session_pid=4) is None


@pytest.mark.parametrize('changed', [
    {'pid': 11}, {'birth': '201'}, {'state': 'Z'}, {'parent_pid': 6},
    {'group_pid': 5}, {'session_pid': 10}, None,
])
def test_pure_changed_anchor_correspondence_refuses(changed):
    row = dict(pid=10, birth='200', state='S', parent_pid=5, group_pid=10, session_pid=4)
    row = None if changed is None else {**row, **changed}
    with pytest.raises(M.ReportFreezerRefusal, match='actual_original_git_anchor_identity_required'):
        M.check_git_anchor_projection(row, pid=10, birth='200', parent_pid=5, session_pid=4)


def test_original_guards_and_locked_commit_body_keep_exact_AST():
    before, after = defs(OLD), defs(NEW)
    for name, node in before.items():
        if name not in {'_Git', 'freeze'}:
            assert ast.dump(node, include_attributes=False) == ast.dump(after[name], include_attributes=False)


def test_original_commit_runner_keeps_exact_AST():
    before, after = defs(OLD)['_Git'], defs(NEW)['_Git']
    pick = lambda node: next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'commit_runner')
    assert ast.dump(pick(before), include_attributes=False) == ast.dump(pick(after), include_attributes=False)


def test_original_full_success_path_requires_29_git_commands():
    old = defs(OLD)
    calls = lambda node: [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'git' and n.func.attr == 'run']
    assert len(calls(old['freeze'])) == 3
    assert len(calls(old['_freeze_locked'])) == 22
    loops = [n for n in ast.walk(old['_freeze_locked']) if isinstance(n, ast.For)]
    assert len(loops) == 1 and isinstance(loops[0].iter, (ast.Tuple, ast.List)) and len(loops[0].iter.elts) == 3
    # The sole loop repeats one counted site twice more; shared original tail
    # adds exactly commit and rev-parse. No fake runtime/owner is supplied.
    assert 3 + 22 + 2 + 2 == 29


def test_actual_SDK_limit_and_documented_session_variance():
    assignments = {n.targets[0].id: n.value for n in ast.parse(NATIVE.read_text()).body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    assert ast.literal_eval(assignments['MAX_CHILDREN_PER_RESERVATION']) == 16
    text = NEW.read_text()
    assert 'process_group=0' in text and 'process_group=self.anchor.pid' in text
    assert "SDK doc's start_new_session=True" in text


def test_run_count_is_inside_terminal_cleanup_and_close_is_idempotent():
    methods = {n.name: n for n in defs(NEW)['_Git'].body if isinstance(n, ast.FunctionDef)}
    run = methods['run']
    protected = next(n for n in run.body if isinstance(n, ast.Try))
    assert any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute)
        and n.target.attr == 'count' for n in protected.body)
    close = methods['close']
    assert isinstance(close.body[0], ast.If) and ast.unparse(close.body[0].test) == 'self._closed'
