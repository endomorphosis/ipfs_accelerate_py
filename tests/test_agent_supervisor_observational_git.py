"""Actual Git regression; AST isolates functions from native actor imports."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'ipfs_accelerate_py/agent_supervisor/'
spec = importlib.util.spec_from_file_location(
    '_candidate_git_environment', ROOT / PREFIX / 'git_environment.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def load_functions(relative, names):
    """Compile unchanged function bodies; native package import is excluded."""
    tree = ast.parse((ROOT / PREFIX / relative).read_text())
    selected = [node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(selected) == len(names)
    module = types.ModuleType('_fixture_' + relative.replace('/', '_'))
    sys.modules[module.__name__] = module
    module.__dict__.update(os=os, subprocess=subprocess, Path=Path, signal=signal,
        dataclass=dataclasses.dataclass, git_subprocess_environment=helper.git_subprocess_environment)
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(ROOT / PREFIX / relative), 'exec'), module.__dict__)
    return module


def index_identity(repo):
    path = repo / '.git/index'
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid, st.st_nlink,
            st.st_size, st.st_mtime_ns, st.st_ctime_ns, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    root.mkdir()
    env = dict(os.environ)
    for key in tuple(env):
        if key.upper().startswith('GIT_'):
            monkeypatch.delenv(key, raising=False)
            env.pop(key)
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    def git(*args):
        return subprocess.run(['/usr/bin/git', *args], cwd=root, env=env,
                              capture_output=True, check=True)
    git('init', '-q')
    git('config', 'user.email', 'private-fixture@example.invalid')
    git('config', 'user.name', 'Private fixture')
    (root / 'tracked.py').write_text('answer = 42\n')
    git('add', 'tracked.py')
    git('commit', '-qm', 'private initial commit')
    file = root / 'tracked.py'
    st = file.stat()
    file.chmod(st.st_mode ^ 0o100)
    file.chmod(st.st_mode)
    assert file.stat().st_mtime_ns == st.st_mtime_ns
    monkeypatch.setenv('GIT_OPTIONAL_LOCKS', '1')
    return root


def test_environment_preserves_selected_routing_without_mutating_input(monkeypatch):
    supplied = {'GIT_OPTIONAL_LOCKS':'1', 'GIT_INDEX_FILE':'explicit-index',
                'GIT_SSH_COMMAND':'custom-auth', 'COUNT':3}
    original = dict(supplied)
    result = helper.git_subprocess_environment(supplied)
    assert supplied == original
    assert result == {**{k:str(v) for k,v in supplied.items()}, 'GIT_OPTIONAL_LOCKS':'0'}
    monkeypatch.setenv('GIT_OPTIONAL_LOCKS', '1')
    assert helper.git_subprocess_environment()['GIT_OPTIONAL_LOCKS'] == '0'
    assert helper.git_subprocess_environment({}) == {'GIT_OPTIONAL_LOCKS':'0'}


def test_observational_status_skips_untracked_walk_on_retained_indexes():
    assert helper.observational_status_arguments(retain_index=False) == (
        'status', '--porcelain=v1', '--untracked-files=all')
    assert helper.observational_status_arguments(retain_index=True) == (
        'status', '--porcelain=v1', '--untracked-files=no')


def test_unprotected_original_status_reproduces_raw_index_refresh(repo):
    before = index_identity(repo)
    completed = subprocess.run(['git', 'status', '--porcelain=v1'], cwd=repo,
                               text=True, capture_output=True, check=True)
    assert completed.stdout == ''
    after = index_identity(repo)
    assert after != before
    # Git may publish identical index bytes when its stat-cache update is a
    # no-op; replacing the inode alone still invalidates a retained raw pin.
    assert after[1] != before[1]


@pytest.mark.parametrize('caller', ['checkpoint', 'changed_paths', 'ledger', 'daemon', 'engine', 'absolute_engine'])
def test_protected_actual_git_observations_preserve_complete_index(repo, caller):
    before = index_identity(repo)
    if caller == 'checkpoint':
        module = load_functions('runtime/interrupted_validation_checkpoint.py', ['_run_git'])
        observe = lambda: module._run_git(repo, 'status', '--porcelain=v1')
    elif caller == 'changed_paths':
        module = load_functions('runtime/interrupted_validation_checkpoint.py', ['_changed_paths'])
        observe = lambda: module._changed_paths(repo)
    elif caller == 'ledger':
        module = load_functions('runtime/pytest_item_ledger.py', ['dirty_source_paths'])
        observe = lambda: module.dirty_source_paths(repo)
    elif caller == 'daemon':
        module = load_functions('todo_daemon/implementation_daemon.py', ['_run_git'])
        observe = lambda: module._run_git(None, ['status', '--porcelain=v1'], cwd=repo).stdout
    else:
        module = load_functions('todo_daemon/engine.py', ['CommandResult', 'run_command'])
        command = '/usr/bin/git' if caller == 'absolute_engine' else 'git'
        observe = lambda: module.run_command([command, 'status', '--porcelain=v1'],
            cwd=repo, timeout=5, environment=dict(os.environ)).stdout
    for _ in range(3):
        assert observe() in ('', ())
        assert index_identity(repo) == before


def test_required_git_writes_still_work(repo):
    module = load_functions('todo_daemon/engine.py', ['CommandResult', 'run_command'])
    (repo / 'new.py').write_text('added = True\n')
    before = index_identity(repo)
    assert module.run_command(['git', 'add', 'new.py'], cwd=repo, timeout=5).returncode == 0
    assert index_identity(repo) != before
    result = module.run_command(['git', 'diff', '--cached', '--name-only'], cwd=repo, timeout=5)
    assert result.stdout == 'new.py\n'


def test_non_git_commands_keep_existing_environment_contract(tmp_path):
    module = load_functions('todo_daemon/engine.py', ['CommandResult', 'run_command'])
    command = [sys.executable, '-c', 'import os;print(os.environ.get("GIT_OPTIONAL_LOCKS"))']
    result = module.run_command(command, cwd=tmp_path, timeout=5, environment={'GIT_OPTIONAL_LOCKS':'1'})
    assert result.returncode == 0 and result.stdout == '1\n'


def test_all_candidate_files_compile_and_import_the_shared_helper():
    for relative in ('git_environment.py', 'todo_daemon/engine.py',
                     'todo_daemon/implementation_daemon.py',
                     'runtime/interrupted_validation_checkpoint.py',
                     'runtime/pytest_item_ledger.py'):
        file = ROOT / PREFIX / relative
        tree = ast.parse(file.read_bytes())
        compile(tree, str(file), 'exec')
        if file.name != 'git_environment.py':
            assert any(isinstance(node, ast.ImportFrom) and node.module == 'git_environment'
                       and node.level == 2 and any(alias.name == 'git_subprocess_environment'
                                                  for alias in node.names)
                       for node in tree.body)
