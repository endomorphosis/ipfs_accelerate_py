"""Hash pressure and admission regression tests, without real hashing workloads."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py import _hash_resources as pressure


_QUIET_PSI = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"


@pytest.fixture
def quiet_host(monkeypatch):
    files = {
        "/proc/loadavg": "0.0 0.0 0.0 1/100 42\n",
        "/proc/meminfo": "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\nSwapTotal: 0 kB\n",
        "/proc/self/cgroup": "0::/team/job\n",
        "/proc/self/mountinfo": "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        **{f"/proc/pressure/{kind}": _QUIET_PSI for kind in ("cpu", "io", "memory")},
    }

    def read(path):
        try:
            return files[str(path)]
        except KeyError:
            raise FileNotFoundError(str(path)) from None

    monkeypatch.setattr(pressure, "_read_proc", read)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    monkeypatch.delenv("IPFS_HASH_MAX_WORKERS", raising=False)
    return files


def test_default_configured_and_requested_caps(quiet_host, monkeypatch):
    assert pressure.hash_worker_limit() == 2
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "99")
    assert pressure.hash_worker_limit(1000) == 4
    assert pressure.hash_worker_limit(1) == 1
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "1")
    assert pressure.hash_worker_limit(4) == 1
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "garbage")
    assert pressure.hash_worker_limit() == 1


def test_affinity_and_ancestor_cpu_quota(quiet_host, monkeypatch):
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "4")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 2})
    assert pressure.hash_worker_limit() == 2
    quiet_host["/sys/fs/cgroup/team/cpu.max"] = "150000 100000"
    assert pressure.hash_worker_limit() == 1
    quiet_host["/sys/fs/cgroup/team/cpu.max"] = "max 100000"
    quiet_host["/sys/fs/cgroup/team/job/cpu.max"] = "25000 100000"
    assert pressure.hash_worker_limit() == 1


@pytest.mark.parametrize("limit_file", ["memory.max", "memory.high"])
def test_cgroup_ancestor_memory_headroom(quiet_host, limit_file):
    quiet_host[f"/sys/fs/cgroup/team/{limit_file}"] = str(4 * 1024**3)
    quiet_host["/sys/fs/cgroup/team/memory.current"] = str(3 * 1024**3)
    assert pressure.host_hash_pressure() == (1, "memory_headroom")


@pytest.mark.parametrize("kind,value", [("cpu", 30), ("io", 7), ("memory", 2)])
@pytest.mark.parametrize("cgroup", [False, True])
def test_cpu_io_memory_pressure(quiet_host, kind, value, cgroup):
    location = f"/sys/fs/cgroup/team/job/{kind}.pressure" if cgroup else f"/proc/pressure/{kind}"
    quiet_host[location] = f"some avg10={value} avg60=0 avg300=0 total=1\n"
    source = "cgroup" if cgroup else "host"
    assert pressure.host_hash_pressure() == (1, f"{source}_{kind}_pressure")


def test_unknown_pressure_and_host_load_are_conservative(quiet_host):
    quiet_host["/proc/loadavg"] = "7 0 0 1/100 42"
    assert pressure.host_hash_pressure() == (1, "host_cpu_load")
    quiet_host["/proc/loadavg"] = "nan 0 0 1/100 42"
    assert pressure.hash_worker_limit() == 1
    quiet_host.pop("/proc/loadavg")
    assert pressure.hash_worker_limit() == 1


def test_delegated_mount_and_cgroup_namespace(quiet_host):
    quiet_host["/proc/self/mountinfo"] = "1 0 0:1 /team /delegated rw - cgroup2 cgroup rw\n"
    assert pressure._cgroup_directories() == (Path("/delegated/job"), Path("/delegated"))
    quiet_host["/proc/self/cgroup"] = "0::/\n"
    assert pressure._cgroup_directories() == (Path("/delegated"),)


def test_lock_identity_ignores_worktree_kind_and_runtime_env(monkeypatch):
    path = pressure.hash_lock_path("sealed-bundle")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/different/runtime")
    monkeypatch.setenv("TMPDIR", "/different/tmp")
    assert pressure.hash_lock_path("native-sha256") == path
    assert path.parent == Path("/tmp")


@pytest.fixture
def isolated_lock(tmp_path, monkeypatch):
    lock = tmp_path / "heavy-hash.lock"
    monkeypatch.setattr(pressure, "hash_lock_path", lambda kind="file-hash": lock)
    return lock


def test_reentrant_admission_across_kinds_and_fresh_pressure(quiet_host, isolated_lock):
    with pressure.hashing_lock(kind="archive", exclusive=False) as outer:
        assert outer == 2
        quiet_host["/proc/pressure/io"] = "some avg10=10 avg60=0 avg300=0 total=1\n"
        with pressure.hashing_lock(kind="interpreter", exclusive=True, timeout=0) as inner:
            assert inner == 1
            assert pressure.hash_worker_limit(4) == 1
    quiet_host["/proc/pressure/io"] = _QUIET_PSI
    with pressure.hashing_lock(timeout=0) as fresh:
        assert fresh == 2


def test_exception_releases_admission(quiet_host, isolated_lock):
    with pytest.raises(RuntimeError, match="hash failed"):
        with pressure.hashing_lock():
            raise RuntimeError("hash failed")
    with pressure.hashing_lock(timeout=0):
        pass


def test_pressure_is_resampled_after_waiting(quiet_host, isolated_lock):
    entered = threading.Event()
    result = []

    def contender():
        entered.set()
        with pressure.hashing_lock(timeout=2) as workers:
            result.append(workers)

    with pressure.hashing_lock():
        thread = threading.Thread(target=contender)
        thread.start()
        assert entered.wait(1)
        quiet_host["/proc/pressure/io"] = "some avg10=10 avg60=0 avg300=0 total=1\n"
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result == [1]


def test_thread_wait_is_bounded(quiet_host, isolated_lock):
    errors = []

    def contender():
        try:
            with pressure.hashing_lock(timeout=0.05):
                errors.append("unexpected admission")
        except pressure.HashingResourceTimeout:
            errors.append("timeout")

    with pressure.hashing_lock():
        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert errors == ["timeout"]


@pytest.mark.skipif(pressure.fcntl is None, reason="requires cross-process flock")
def test_process_contention_even_when_idle_and_kinds_differ(quiet_host, isolated_lock):
    # Load the file as a separate checkout would, with no shared Python state.
    script = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('isolated_pressure', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.hash_lock_path = lambda kind='file-hash': pathlib.Path(sys.argv[2])
module.host_hash_pressure = lambda: (4, 'admitted')
try:
    with module.hashing_lock(kind='different-kind', exclusive=False, timeout=0.1):
        print('admitted')
except module.HashingResourceTimeout:
    print('timeout')
"""
    args = [sys.executable, "-c", script, pressure.__file__, str(isolated_lock)]
    environment = dict(os.environ, XDG_RUNTIME_DIR="/other/runtime")
    with pressure.hashing_lock(kind="outer-kind", exclusive=False):
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, env=environment)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "timeout"
    result = subprocess.run(args, capture_output=True, text=True, timeout=5, env=environment)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "admitted"


@pytest.mark.skipif(pressure.fcntl is None, reason="requires flock")
def test_symlink_lock_fails_closed(quiet_host, tmp_path, isolated_lock):
    target = tmp_path / "unrelated"
    target.touch()
    isolated_lock.symlink_to(target)
    with pytest.raises(OSError):
        with pressure.hashing_lock(timeout=0):
            pytest.fail("unsafe lock was admitted")


@pytest.mark.skipif(pressure.fcntl is None, reason="requires flock")
def test_replaced_lock_fails_closed(quiet_host, isolated_lock, monkeypatch):
    original_flock = pressure.fcntl.flock

    def replacing_flock(fd, flags):
        original_flock(fd, flags)
        if flags & pressure.fcntl.LOCK_EX:
            isolated_lock.unlink()
            isolated_lock.touch(mode=0o600)

    monkeypatch.setattr(pressure.fcntl, "flock", replacing_flock)
    with pytest.raises(PermissionError, match="changed while waiting"):
        with pressure.hashing_lock(timeout=0):
            pytest.fail("replaced lock was admitted")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_child_unwind_preserves_reused_fd_and_parent_lock(isolated_lock):
    # Unlike an os._exit inside the context, normal child unwinding executes
    # both inherited context managers, after atfork has closed their lock fd.
    script = """
import importlib.util, os, pathlib, sys
spec = importlib.util.spec_from_file_location('fork_pressure', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.hash_lock_path = lambda kind='file-hash': pathlib.Path(sys.argv[2])
module.host_hash_pressure = lambda: (2, 'admitted')
def fork_in_admission():
    with module.hashing_lock():
        with module.hashing_lock(kind='nested'):
            old_fd = next(iter(module._open_fds))
            child = os.fork()
            if child == 0:
                replacement = os.open(os.devnull, os.O_RDONLY)
                if replacement != old_fd:
                    os.dup2(replacement, old_fd)
                    os.close(replacement)
            else:
                _, status = os.waitpid(child, 0)
                assert os.waitstatus_to_exitcode(status) == 0, status
    return child, old_fd
child, old_fd = fork_in_admission()
if child == 0:
    os.fstat(old_fd)
    try:
        with module.hashing_lock(timeout=0.1):
            raise AssertionError('child bypassed parent admission')
    except module.HashingResourceTimeout:
        pass
    print('child-ok', flush=True)
    os._exit(0)
with module.hashing_lock(timeout=0):
    print('parent-ok', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, pressure.__file__, str(isolated_lock)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["child-ok", "parent-ok"]


def test_no_flock_fallback_is_serial(quiet_host, monkeypatch, isolated_lock):
    monkeypatch.setattr(pressure, "fcntl", None)
    with pressure.hashing_lock() as workers:
        assert workers == 1
        with pressure.hashing_lock(timeout=0) as nested:
            assert nested == 1


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_timeout_rejected(quiet_host, isolated_lock, timeout):
    with pytest.raises(ValueError):
        with pressure.hashing_lock(timeout=timeout):
            pytest.fail("invalid timeout admitted")
