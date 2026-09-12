"""Preserve the immutable transaction protocol while reads avoid index writes."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import run_agent_supervisor_efficiency_state_hardening as op

HISTORICAL_GUARD = {'branch_lock_identity': [66306, 22071209, 33204, 1000, 1, 0, 1787801484465514608, 1787801484465514608], 'branch_ref': 'refs/heads/agent/agent-supervisor-efficiency-and-state-hardening-v1', 'candidate_head': '34e71e42fbe9b02b8a3f8d6b6d3d0521f6fd7e77', 'candidate_tree': '4ff8126a210b8f39f78b1746c9c41e552a641b03', 'guard_cid': 'sha256:41641daeafed602f2b3014ecee623f07be16585274160c4c180299d9235c26a0', 'head_lock_identity': [66306, 30315015, 33204, 1000, 1, 0, 1787801484353513103, 1787801484353513103], 'index_lock_identity': [66306, 30315018, 33204, 1000, 1, 0, 1787801485738531709, 1787801485738531709], 'index_pid': 4027881, 'index_process_birth': {'boot_id': 'fe7ef8ca-8b86-4280-a74e-f37d621c2f96', 'parent_pid': 4022781, 'pid': 4027881, 'start_time_ticks': 37139540}, 'owner_pid': 4022781, 'owner_thread_id': 259096036142336, 'packed_refs_lock_absent_before_guard_start': True, 'prepared_at_ns': 1787801485746600633, 'protocol_cid': 'sha256:cdf11f400fad534798b865d9593cf454d10d57b248b3d6539faa34420d2b634c', 'reference_pid': 4024490, 'reference_process_birth': {'boot_id': 'fe7ef8ca-8b86-4280-a74e-f37d621c2f96', 'parent_pid': 4022781, 'pid': 4024490, 'start_time_ticks': 37139380}, 'schema': 'ipfs_accelerate_py/agent-supervisor/aseh-r30-candidate-git-guard@1', 'session_id': 4022758, 'state': 'prepared_verify_only'}


def test_actual_historical_protocol_and_guard_cid_remain_exact():
    guard = copy.deepcopy(HISTORICAL_GUARD)
    protocol = op._r30_candidate_git_guard_protocol(guard['candidate_head'])
    assert op._identity(protocol) == guard['protocol_cid']
    assert protocol['git_environment'] == op._trusted_git_guard_environment()
    changed = {**protocol, 'git_environment': op._trusted_git_environment()}
    assert op._identity(changed) != guard['protocol_cid']
    if os.geteuid() == guard['head_lock_identity'][3]:
        assert op._validate_r30_candidate_git_guard_record(guard,
            candidate_head=guard['candidate_head'], candidate_tree=guard['candidate_tree']) == guard
    else:
        with pytest.raises(op.OperatorError):
            op._validate_r30_candidate_git_guard_record(guard,
                candidate_head=guard['candidate_head'], candidate_tree=guard['candidate_tree'])


@pytest.mark.parametrize('change', ['protocol', 'birth', 'candidate', 'guard-cid'])
def test_historical_tamper_is_not_an_environment_compatibility_exception(change):
    guard = copy.deepcopy(HISTORICAL_GUARD)
    for key in ('head_lock_identity', 'branch_lock_identity', 'index_lock_identity'):
        guard[key][3] = os.geteuid()
    if change == 'protocol':
        protocol = op._r30_candidate_git_guard_protocol(guard['candidate_head'])
        protocol['git_environment']['GIT_CONFIG_VALUE_1'] = 'true'
        guard['protocol_cid'] = op._identity(protocol)
    if change == 'birth': guard['index_process_birth']['parent_pid'] += 1
    if change == 'candidate': guard['candidate_head'] = 'f' * 40
    guard['guard_cid'] = op._identity({k:v for k,v in guard.items() if k != 'guard_cid'})
    if change == 'guard-cid': guard['guard_cid'] = 'sha256:' + '0' * 64
    with pytest.raises(op.OperatorError, match='record differs'):
        op._validate_r30_candidate_git_guard_record(guard,
            candidate_head=HISTORICAL_GUARD['candidate_head'], candidate_tree=HISTORICAL_GUARD['candidate_tree'])


def test_writer_environment_is_only_used_by_the_three_exact_guard_sites():
    tree = ast.parse(Path(op.__file__).read_text())
    callers = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == '_trusted_git_guard_environment':
                    callers.append(node.name)
    assert sorted(callers) == sorted(['_r30_candidate_git_guard_protocol', '_r30_git_guard_launcher', '_prepared_candidate_git_guard'])
    assert op._trusted_git_environment()['GIT_OPTIONAL_LOCKS'] == '0'
    assert op._trusted_git_environment()['GIT_CONFIG_VALUE_2'] == 'false'
    assert 'GIT_OPTIONAL_LOCKS' not in op._trusted_git_guard_environment()
    assert op._trusted_git_guard_environment()['GIT_CONFIG_COUNT'] == '2'


def test_actual_fresh_guard_exec_environment_and_index_closure(tmp_path, monkeypatch):
    repository = tmp_path / 'source'; repository.mkdir()
    def git(*arguments):
        p = subprocess.run(['/usr/bin/git', *arguments], cwd=repository, text=True,
            capture_output=True, timeout=20, check=True)
        return p.stdout.strip()
    git('init', '-q', '--initial-branch=aseh')
    git('config', 'user.name', 'Qualification'); git('config', 'user.email', 'qualification@example.invalid')
    tracked = repository / 'source.py'; tracked.write_text('value = 1\n')
    git('add', 'source.py'); git('commit', '-qm', 'qualified source')
    head, tree = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD^{tree}')
    index = repository / '.git/index'; original = hashlib.sha256(index.read_bytes()).hexdigest()
    monkeypatch.setattr(op, 'ROOT', repository)
    with op._prepared_candidate_git_guard(candidate_head=head, candidate_tree=tree) as guard:
        expected = op._trusted_git_guard_environment()
        for child in (guard.reference_process, guard.index_process):
            raw = Path(f'/proc/{child.pid}/environ').read_bytes()
            actual = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
            assert actual == expected
        assert op._r30_candidate_git_guard_protocol(head)['git_environment'] == expected
        assert op._validate_candidate_git_guard_health(guard, boundary='fresh exact environment') == guard.record
        assert op._validate_r30_candidate_git_guard_record(guard.record, candidate_head=head, candidate_tree=tree) == guard.record
        assert op._git('status', '--porcelain') == ''
        assert op._git('diff', '--check') == ''
        assert hashlib.sha256(index.read_bytes()).hexdigest() == original
    assert guard.reference_process.returncode == 0 and guard.index_process.returncode == 0
    assert all(not path.exists() for path in (guard.head_lock_path, guard.branch_lock_path, guard.index_lock_path))
    assert hashlib.sha256(index.read_bytes()).hexdigest() == original
