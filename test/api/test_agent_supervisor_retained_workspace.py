from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.verification import retained_workspace as target


def legacy_digest(workspace):
    digest = hashlib.sha256()
    for root, directories, files in os.walk(workspace, followlinks=False):
        directories.sort()
        files.sort()
        for name in (*directories, *files):
            path = Path(root) / name
            digest.update(path.relative_to(workspace).as_posix().encode() + b'\0')
            digest.update(str(path.lstat().st_mode).encode() + b'\0')
            if path.is_symlink():
                digest.update(b'L' + os.readlink(path).encode())
            elif path.is_dir():
                digest.update(b'D')
            else:
                digest.update(b'F' + path.read_bytes())
            digest.update(b'\0')
    return 'sha256:' + digest.hexdigest()


def test_matches_legacy_complete_tree_including_ignored_files(tmp_path):
    (tmp_path / '.gitignore').write_text('ignored\n')
    (tmp_path / 'ignored').write_bytes(b'ignored contents')
    (tmp_path / 'sub').mkdir()
    script = tmp_path / 'sub' / 'run'
    script.write_bytes(b'#!/bin/sh\n')
    script.chmod(0o755)
    (tmp_path / 'external-link').symlink_to('/does/not/exist')
    result = target.fingerprint_retained_tree(tmp_path)
    assert result['content_digest'] == legacy_digest(tmp_path)
    assert result['entry_count'] == 5
    (tmp_path / 'ignored').write_bytes(b'changed contents')
    assert target.fingerprint_retained_tree(tmp_path) != result


def test_source_over_old_512_mib_limit_is_fully_streamed(tmp_path):
    path = tmp_path / 'toolchain.bin'
    size = 512 * 1024 * 1024 + 1
    with path.open('wb') as stream:
        stream.truncate(size)
    with pytest.raises(RuntimeError, match='byte budget'):
        target.fingerprint_retained_tree(tmp_path, max_bytes=512 * 1024 * 1024)
    first = target.fingerprint_retained_tree(tmp_path)
    assert first['content_bytes'] == size
    with path.open('r+b') as stream:
        stream.seek(size - 1)
        stream.write(b'!')
    assert target.fingerprint_retained_tree(tmp_path)['content_digest'] != first['content_digest']


@pytest.mark.parametrize('kind', ['file', 'link'])
def test_byte_budget_includes_symlink_targets(tmp_path, kind):
    path = tmp_path / 'entry'
    if kind == 'file':
        path.write_bytes(b'12345')
    else:
        path.symlink_to('12345')
    with pytest.raises(RuntimeError, match='byte budget'):
        target.fingerprint_retained_tree(tmp_path, max_bytes=4)


def test_entry_budget(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').touch()
    with pytest.raises(RuntimeError, match='entry budget'):
        target.fingerprint_retained_tree(tmp_path, max_entries=1)


def test_deadline(tmp_path, monkeypatch):
    ticks = iter([0, 2])
    monkeypatch.setattr(target.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(RuntimeError, match='time budget'):
        target.fingerprint_retained_tree(tmp_path, max_seconds=1)


def test_special_file_denied_without_blocking(tmp_path):
    os.mkfifo(tmp_path / 'fifo')
    with pytest.raises(RuntimeError, match='special file'):
        target.fingerprint_retained_tree(tmp_path)


def test_symlink_root_denied(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(RuntimeError, match='not a directory'):
        target.fingerprint_retained_tree(link)


def test_walk_error_is_not_silently_omitted(tmp_path, monkeypatch):
    def failed_walk(*args, **kwargs):
        kwargs['onerror'](PermissionError('denied subtree'))
        yield
    monkeypatch.setattr(target.os, 'fwalk', failed_walk)
    with pytest.raises(RuntimeError, match='fingerprint failed'):
        target.fingerprint_retained_tree(tmp_path)


@pytest.mark.parametrize('change', ['bytes', 'add', 'replace'])
def test_change_after_first_walk_denied(tmp_path, monkeypatch, change):
    path = tmp_path / 'file'
    path.write_bytes(b'original')
    walk = target.os.fwalk
    calls = 0
    def changing_walk(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            if change == 'bytes':
                path.write_bytes(b'modified')
            elif change == 'add':
                (tmp_path / 'new').touch()
            else:
                path.unlink()
                path.symlink_to('/dev/zero')
        yield from walk(*args, **kwargs)
    monkeypatch.setattr(target.os, 'fwalk', changing_walk)
    with pytest.raises(RuntimeError, match='changed during fingerprint'):
        target.fingerprint_retained_tree(tmp_path)


@pytest.mark.parametrize('limits', [{'max_bytes': True}, {'max_entries': 0},
                                  {'max_seconds': float('inf')}, {'max_seconds': 0}])
def test_invalid_limits(tmp_path, limits):
    with pytest.raises(ValueError, match='invalid'):
        target.fingerprint_retained_tree(tmp_path, **limits)
