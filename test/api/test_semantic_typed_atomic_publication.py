"""Authored inode-custody controls using actual Linux filesystem operations.

These controls exercise publication and failure cleanup, not source verification
or semantic/proof authority. Native source-bound SIGKILL probes are separate.
"""
import errno
import os
from pathlib import Path
import stat

import pytest
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_source_context as wrapper


class ObservedOS:
    def __getattr__(self, name):
        return getattr(os, name)


def test_atomic_publication_exposes_only_complete_single_link_inode(tmp_path, monkeypatch):
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    observed = ObservedOS()
    raw = b"authored-publication-body" * 4096
    snapshots = []
    def link(source, name, **kwargs):
        assert source.startswith("/proc/self/fd/") and kwargs["follow_symlinks"] is True
        descriptor = int(source.rsplit("/", 1)[1])
        before = os.fstat(descriptor)
        assert before.st_nlink == 0 and stat.S_IMODE(before.st_mode) == 0o400
        assert before.st_size == len(raw) and list(tmp_path.iterdir()) == []
        os.link(source, name, **kwargs)
        snapshots.append(os.fstat(descriptor).st_nlink)
    observed.link = link
    monkeypatch.setattr(wrapper, "os", observed)
    try:
        wrapper._typed_cache_atomic_file(parent, "record.json", raw)
        value, info = wrapper._typed_cache_read_file(parent, "record.json", len(raw))
        assert value == raw and snapshots == [1]
        assert [path.name for path in tmp_path.iterdir()] == ["record.json"]
    finally:
        os.close(parent)


def test_atomic_collision_preserves_existing_inode_and_bytes(tmp_path):
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        wrapper._typed_cache_atomic_file(parent, "record.json", b"first")
        before = (tmp_path / "record.json").stat()
        wrapper._typed_cache_atomic_file(parent, "record.json", b"replacement refused")
        after = (tmp_path / "record.json").stat()
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
        assert (tmp_path / "record.json").read_bytes() == b"first" and after.st_nlink == 1
        assert [path.name for path in tmp_path.iterdir()] == ["record.json"]
    finally:
        os.close(parent)


@pytest.mark.parametrize("phase", ["write", "link", "directory_fsync"])
def test_atomic_failure_has_no_named_or_two_link_residue(tmp_path, monkeypatch, phase):
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    observed = ObservedOS()
    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "authored publication failure")
    if phase in {"write", "link"}:
        setattr(observed, phase, fail)
    else:
        def fsync(descriptor):
            if descriptor == parent:
                fail()
            return os.fsync(descriptor)
        observed.fsync = fsync
    monkeypatch.setattr(wrapper, "os", observed)
    try:
        with pytest.raises(OSError, match="authored publication failure"):
            wrapper._typed_cache_atomic_file(parent, "record.json", b"complete")
        paths = list(tmp_path.iterdir())
        if phase == "directory_fsync":
            assert [path.name for path in paths] == ["record.json"]
            assert paths[0].read_bytes() == b"complete" and paths[0].stat().st_nlink == 1
        else:
            assert paths == []
    finally:
        os.close(parent)


def test_unsupported_anonymous_publication_leaves_readonly_cache_miss(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    observed = ObservedOS()
    observed.O_TMPFILE = None
    monkeypatch.setattr(wrapper, "os", observed)
    binding, proposal = {"authored_input": "one"}, {"authored_proposal": "not verification"}
    wrapper._publish_cached_typed_proposal(repository, binding, proposal, 10000)
    directory = repository / ".runtime" / wrapper._TYPED_CACHE_DIRECTORY
    assert list(directory.iterdir()) == []
    assert wrapper._load_cached_typed_proposal(repository, binding, 10000) is None
    assert list(directory.iterdir()) == []


def test_concurrent_publishers_leave_one_immutable_winner(tmp_path):
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    pids = []
    try:
        for raw in (b"first" * 20000, b"second" * 20000):
            pid = os.fork()
            if pid == 0:
                try:
                    wrapper._typed_cache_atomic_file(parent, "record.json", raw)
                except BaseException:
                    os._exit(1)
                os._exit(0)
            pids.append(pid)
        for pid in pids:
            assert os.waitpid(pid, 0)[1] == 0
        paths = list(tmp_path.iterdir())
        assert [path.name for path in paths] == ["record.json"]
        assert paths[0].read_bytes() in (b"first" * 20000, b"second" * 20000)
        info = paths[0].stat()
        assert info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o400
    finally:
        os.close(parent)
