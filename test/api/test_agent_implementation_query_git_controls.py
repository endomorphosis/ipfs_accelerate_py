"""Real private Git queries; no product substitutes, native grants or providers."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

import ipfs_accelerate_py.agent_implementation_route as subject


def _git(root, *arguments, optional_locks='1'):
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith('GIT_')}
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS=optional_locks,
        GIT_NO_LAZY_FETCH='1', LC_ALL='C', LANG='C')
    result = subprocess.run(['git', '-c', 'protocol.allow=never',
        '-c', 'user.name=Private Query Test', '-c', 'user.email=query-test@example.invalid',
        '-c', 'commit.gpgSign=false', *arguments], cwd=root, env=environment,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=15, check=False)
    assert result.returncode == 0, result.stderr.decode('utf-8', 'replace')[-4096:]
    assert len(result.stdout) <= 1024*1024 and len(result.stderr) <= 65536
    return result.stdout


def _repository_with_stale_tracked_stat(root):
    root.mkdir(mode=0o700)
    _git(root, 'init', '--quiet', '--initial-branch=main')
    tracked = root/'tracked.txt'
    tracked.write_text('ordinary source bytes, unchanged during the query\n')
    _git(root, 'add', '--', 'tracked.txt')
    _git(root, 'commit', '--quiet', '-m', 'Private fixture baseline')
    before = tracked.stat()
    os.utime(tracked, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
    assert tracked.stat().st_mtime_ns != before.st_mtime_ns
    return root


def _index_identity(root):
    path = root/'.git/index'
    before = path.stat()
    body = path.read_bytes()
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) == (
        after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
    return dict(device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns,
                bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def test_real_status_query_preserves_index_identity_despite_conflicting_caller_flags(tmp_path, monkeypatch):
    # Prove this real Git/filesystem combination actually has a stale-stat
    # refresh to perform, using a different disposable repository.
    control = _repository_with_stale_tracked_stat(tmp_path/'positive-control')
    control_before = _index_identity(control)
    assert _git(control, 'status', '--porcelain=v1', '-z', '--untracked-files=all',
                optional_locks='1') == b''
    assert _index_identity(control) != control_before

    root = _repository_with_stale_tracked_stat(tmp_path/'query-under-test')
    before = _index_identity(root)
    monkeypatch.setenv('GIT_OPTIONAL_LOCKS', '1')
    monkeypatch.setenv('GIT_NO_LAZY_FETCH', '0')
    observed = subject._agent_git_output(root,
        ('status', '--porcelain=v1', '-z', '--untracked-files=all'))
    assert observed == b''
    assert _index_identity(root) == before
    assert not (root/'.git/index.lock').exists()
    assert (root/'tracked.txt').read_text() == 'ordinary source bytes, unchanged during the query\n'


@pytest.mark.parametrize('caller_flags', ['conflicting', 'absent'])
def test_real_git_alias_subprocess_observes_explicit_readonly_no_lazy_controls(tmp_path, monkeypatch, caller_flags):
    root = _repository_with_stale_tracked_stat(tmp_path/'real-environment-query')
    for key, value in (('GIT_OPTIONAL_LOCKS', '1'), ('GIT_NO_LAZY_FETCH', '0')):
        if caller_flags == 'conflicting':
            monkeypatch.setenv(key, value)
        else:
            monkeypatch.delenv(key, raising=False)
    # An actual Git command-local shell alias launches the actual interpreter.
    # It reads only two public flags. Git and the product function are genuine.
    script = "import json,os;print(json.dumps({key:os.environ.get(key) for key in ('GIT_OPTIONAL_LOCKS','GIT_NO_LAZY_FETCH')}))"
    alias = '!'+shlex.quote(sys.executable)+' -B -c '+shlex.quote(script)
    before = _index_identity(root)
    observed = subject._agent_git_output(root,
        ('-c', 'alias.query-public-environment='+alias, 'query-public-environment'))
    assert json.loads(observed) == {'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_LAZY_FETCH': '1'}
    assert _index_identity(root) == before
    assert not (root/'.git/index.lock').exists()
