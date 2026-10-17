"""Pure necessary-policy controls. No native/daemon/provider/Git fixtures.

Positive pure structures are never claims of successful public freezing.
"""
from pathlib import Path
import importlib.util
import sys

import pytest

ROOT = Path(__file__).absolute().parent
HELPER = ROOT / 'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_candidate_freezer.py'
SPEC = importlib.util.spec_from_file_location('report_freezer_memory_under_test', HELPER)
f = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = f
SPEC.loader.exec_module(f)  # This standalone helper has stdlib-only top-level imports.


def budget(**changes):
    values = dict(attempt=3, cap=3, task_id='LIR-9001', task_cid='actual-cid',
        before_by_id={'LIR-9001': 2, 'LIR-0101': 1}, before_by_cid={'actual-cid': 2, 'peer-cid': 1},
        current_by_id={'LIR-9001': 3, 'LIR-0101': 1}, current_by_cid={'actual-cid': 3, 'peer-cid': 1})
    values.update(changes)
    return values


def test_two_consumed_attempts_and_peer_counters_preserved():
    assert f.budget_projection(**budget()) is None


@pytest.mark.parametrize('change', [
    {'attempt': 2}, {'attempt': True}, {'cap': 4}, {'task_id': 'LIR-9002'},
    {'before_by_id': {'LIR-9001': 1, 'LIR-0101': 1}},
    {'before_by_cid': {'actual-cid': 0, 'peer-cid': 1}},
    {'current_by_id': {'LIR-9001': 3, 'LIR-0101': 0}},
    {'current_by_cid': {'actual-cid': 2, 'peer-cid': 1}},
    {'current_by_id': {'LIR-9001': 3}},
    {'current_by_cid': {'actual-cid': 3, 'peer-cid': True}},
])
def test_attempt_reset_extra_cap_and_peer_changes_refused(change):
    with pytest.raises(f.ReportFreezerRefusal):
        f.budget_projection(**budget(**change))


def test_report_is_only_structurally_json():
    assert f.strict_report(b'{"native_validation_qualified":false,"items":[1,2]}')['items'] == [1, 2]


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":{"b":1,"b":2}}', b'{"x":NaN}',
    b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e9999}', b'[]', b'null', b'{', b'\xff'])
def test_ambiguous_nonfinite_nonobject_invalid_json_refused(raw):
    with pytest.raises(f.ReportFreezerRefusal):
        f.strict_report(raw)


def test_single_original_parent_structural_check():
    raw = b'tree ' + b'a' * 40 + b'\nparent ' + b'b' * 40 + b'\nauthor x\n\nsubject\n'
    assert f.parse_commit(raw, 'b' * 40) == 'a' * 40


@pytest.mark.parametrize('parents', [b'', b'parent '+b'c'*40+b'\n',
    b'parent '+b'b'*40+b'\nparent '+b'c'*40+b'\n', b'parent '+b'b'*39+b'\n'])
def test_root_merge_wrong_or_malformed_parent_refused(parents):
    with pytest.raises(f.ReportFreezerRefusal):
        f.parse_commit(b'tree '+b'a'*40+b'\n'+parents+b'author x\n\nsubject', 'b'*40)


def raw_diff(old_mode=b':000000', new_mode=b'100644', old=b'0'*40, new=b'a'*40, status=b'A', name=None):
    return b' '.join((old_mode, new_mode, old, new, status)) + b'\0' + (name or f.REPORT.encode()) + b'\0'


@pytest.mark.parametrize('raw', [raw_diff(), raw_diff(b':100644', old=b'b'*40, status=b'M')])
def test_exact_report_regular_addition_or_modification(raw):
    assert f.parse_diff(raw) == 'a'*40


@pytest.mark.parametrize('raw', [raw_diff(new_mode=b'100755'), raw_diff(new_mode=b'120000'),
    raw_diff(name=b'.gitignore'), raw_diff(status=b'R100'), raw_diff()+raw_diff(),
    raw_diff(b':100644', old=b'a'*40, status=b'M')])
def test_extra_rename_mode_and_nochange_candidate_refused(raw):
    with pytest.raises(f.ReportFreezerRefusal):
        f.parse_diff(raw)


@pytest.mark.parametrize('raw', [b'', f.REPORT.encode()+b'\0'])
def test_only_declared_dirty_path_structure_allowed(raw):
    f.allowed_dirty_paths(raw)


@pytest.mark.parametrize('raw', [b'.gitignore\0', f.REPORT.encode()+b'\0extra.py\0',
    b'../handoff-reconciliation.json\0', f.REPORT.encode(), f.REPORT.encode()+b'\n\0'])
def test_extra_dirty_unterminated_and_path_tricks_refused(raw):
    with pytest.raises(f.ReportFreezerRefusal):
        f.allowed_dirty_paths(raw)


def test_no_capture_refuses_before_native_source_provider_or_git_calls():
    if not sys.dont_write_bytecode:
        pytest.fail('controls must run under python -B')
    with pytest.raises(f.ReportFreezerRefusal, match='actual_callable_dispatch_bridge_capture_absent'):
        f.freeze(object(), outcome=object(), freezer_owner=object(), protected_source_sha256={},
                 declared_external_storage={})


@pytest.mark.parametrize('role,mode', [('git_metadata',0o644),('git_metadata',0o664),
    ('report',0o644),('pinned_source',0o664)])
def test_explicit_regular_role_modes(role,mode):
    import stat
    assert f.check_regular_policy(mode=stat.S_IFREG|mode,nlink=1,uid=1000,owner_uid=1000,role=role) is None


@pytest.mark.parametrize('role,mode,nlink,uid', [('report',0o664,1,1000),('report',0o666,1,1000),
    ('git_metadata',0o666,1,1000),('git_metadata',0o644,2,1000),('git_metadata',0o644,1,1234),
    ('git_metadata',0o755,1,1000),('pinned_source',0o666,1,1000)])
def test_report_groupwrite_and_metadata_worldwrite_alias_owner_refused(role,mode,nlink,uid):
    import stat
    with pytest.raises(f.ReportFreezerRefusal):
        f.check_regular_policy(mode=stat.S_IFREG|mode,nlink=nlink,uid=uid,owner_uid=1000,role=role)


def test_symlink_metadata_cannot_be_a_regular_git_input():
    import stat
    with pytest.raises(f.ReportFreezerRefusal):
        f.check_regular_policy(mode=stat.S_IFLNK|0o644,nlink=1,uid=1000,owner_uid=1000,role='git_metadata')


def test_actual_frozen_board_0664_is_readonly_fixed_sha_input():
    board=Path('/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/acceptance-adapter/adapter-v02/tests/fixtures/bootstrap/tasks.todo.md')
    import stat
    assert stat.S_IMODE(board.lstat().st_mode)==0o664
    raw=f.read_regular(board,source_sha=f.BOARD_SHA256)
    assert f.sha(raw)==f.BOARD_SHA256 and raw.count(b'\n## LIR-')==64
