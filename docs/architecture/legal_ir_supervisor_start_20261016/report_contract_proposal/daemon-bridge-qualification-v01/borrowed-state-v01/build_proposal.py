"""Source-only assembler; never imports the product or opens a database."""
import ast
import difflib
import hashlib
import os
from pathlib import Path

HERE = Path(__file__).absolute().parent
BASE = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
RELATIVE = Path('ipfs_accelerate_py/agent_supervisor/todo_daemon/portal_task_state_control_plane.py')


def assemble(original, fragment):
    source = original.replace('import os\n', 'import os\nimport stat\nimport threading\nfrom contextlib import contextmanager\n', 1)
    marker = '_BOUND_STATE_TARGETS: dict[str, str] = {}'
    assert source.count(marker) == 1
    source = source.replace(marker, fragment + '\n\n' + marker, 1)

    def change(name, transform):
        nonlocal source
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        lines = source.splitlines(True)
        before = ''.join(lines[node.lineno - 1:node.end_lineno])
        after = transform(before)
        assert before != after, name
        source = ''.join(lines[:node.lineno - 1]) + after + ''.join(lines[node.end_lineno:])

    def bind(text):
        key = '    _BOUND_STATE_TARGETS[str(Path(state_path).expanduser().resolve())] = text'
        return text.replace(key, '''    if _BORROWED_BY_STATE:
        key = _borrow_path(state_path)
        with _BORROWED_REGISTRY_LOCK:
            _borrow_need(key not in _BORROWED_BY_STATE, "borrowed_state_binding_mutation_refused")
        _refuse_borrowed_target_open(text)
''' + key)
    change('bind_task_state_control_plane', bind)
    change('resolve_control_plane', lambda text: text.replace('    selected = control_plane_target(state_path)', '''    selected = control_plane_target(state_path)
    binding = _borrowed_binding(state_path, selected)
    if binding is not None:
        require_borrowed_task_state_connection(state_path, connection=binding.connection)
        return str(binding.target), binding.source
    # Implicit legacy resolution can adopt/move catalogs. No such resolution
    # may occur incidentally inside an active borrowed attempt.
    _borrow_need(not _BORROWED_BY_STATE or selected is not None,
                 "borrowed_state_implicit_resolution_during_scope_refused")''', 1))
    change('_connect', lambda text: text.replace('    from ipfs_accelerate_py.', '    _refuse_borrowed_target_open(target)\n    from ipfs_accelerate_py.', 1))

    def state_context(text):
        old = '    connection = _connect(target)\n    try:\n'
        assert text.count(old) == 1
        text = text.replace(old, '    with _state_connection(target, state_path=state_path) as connection:\n')
        end = '    finally:\n        connection.close()\n'
        assert text.count(end) == 1
        return text.replace(end, '')

    def save_board(text):
        text = text.replace('    source: str,\n', '    source: str,\n    *, state_path: Path | str | None = None,\n', 1)
        text = state_context(text)
        return text.replace('_project_ducklake(\n            key, body_json, updated_ms, updated_at, target=target',
                            '_state_ducklake_projection(\n            state_path, target, key, body_json, updated_ms, updated_at')

    def load_board(text):
        text = text.replace('*, create: bool)', '*, create: bool, state_path: Path | str | None = None)', 1)
        text = state_context(text)
        return missing_row(text)

    def missing_row(text):
        return text.replace('    if row is None:\n        return {}', '''    if row is None:
        if state_path is not None and task_state_connection_is_borrowed(state_path):
            raise PortalTaskStateControlPlaneError("borrowed_state_original_authoritative_row_required")
        return {}''', 1)

    change('_save_board_task_state', save_board)
    change('_load_board_task_state', load_board)

    def save_lane(text):
        text = text.replace('return _save_board_task_state(target, key, body_json, updated_at, source)',
                            'return _save_board_task_state(target, key, body_json, updated_at, source, state_path=state_path)')
        text = state_context(text)
        return text.replace('_project_ducklake(\n            key, body_json, revision, updated_at, target=target',
                            '_state_ducklake_projection(\n            state_path, target, key, body_json, revision, updated_at')

    def load_lane(text):
        text = text.replace('return _load_board_task_state(target, key, create=create)',
                            'return _load_board_task_state(target, key, create=create, state_path=state_path)')
        return missing_row(state_context(text))

    change('save_portal_task_state', save_lane)
    change('load_portal_task_state', load_lane)
    change('read_task_state_payload', lambda text: text.replace('    if not state_path.is_file():', '''    if task_state_connection_is_borrowed(state_path):
        raise PortalTaskStateControlPlaneError("borrowed_state_legacy_json_fallback_refused")
    if not state_path.is_file():''', 1))
    ast.parse(source)
    # Preserve the four original SQL programs, not just their string spelling.
    def execute_calls(tree, name):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        return [ast.dump(n, include_attributes=False) for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'execute']
    before, after = ast.parse(original), ast.parse(source)
    for name in ('_save_board_task_state', '_load_board_task_state', 'save_portal_task_state', 'load_portal_task_state'):
        assert execute_calls(before, name) == execute_calls(after, name), name
    return source


def main():
    original = (BASE / RELATIVE).read_text()
    source = assemble(original, (HERE / 'borrowed_scope.fragment.txt').read_text())
    target = HERE / 'proposed' / RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    for path, raw in (
        (target, source.encode()),
        (HERE / 'portal_task_state_control_plane.patch', ''.join(difflib.unified_diff(
            original.splitlines(True), source.splitlines(True),
            fromfile='a/' + RELATIVE.as_posix(), tofile='b/' + RELATIVE.as_posix())).encode()),
    ):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print({'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
           'proposed_sha256': hashlib.sha256(source.encode()).hexdigest(),
           'original_four_execute_call_ASTs_preserved': True})


if __name__ == '__main__':
    main()
