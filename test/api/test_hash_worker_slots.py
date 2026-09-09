"""Cross-process file hashing slots, without large files or live owner writes."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py import _hash_resources as resources

pytestmark = pytest.mark.skipif(resources.fcntl is None, reason="requires flock")


@pytest.fixture
def isolated_slots(tmp_path, monkeypatch):
    path = tmp_path / "heavy-hash.lock"
    monkeypatch.setattr(resources, "hash_lock_path", lambda kind="file-hash": path)
    monkeypatch.setattr(resources, "host_hash_pressure", lambda: (4, "admitted"))
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "2")
    yield path
    assert not resources._open_fds


_CHILD = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('child_resources', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.hash_lock_path = lambda kind='file-hash': pathlib.Path(sys.argv[2])
module.host_hash_pressure = lambda: (4, 'admitted')
admission = module.hashing_worker_slot if sys.argv[3] == 'worker' else module.hashing_lock
try:
    with admission(timeout=float(sys.argv[4])):
        print('entered', flush=True)
        sys.stdin.readline()
    print('released', flush=True)
except module.HashingResourceTimeout:
    print('timeout', flush=True)
"""


def _start(path: Path, *, kind: str = "worker", timeout: float = 3) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-B", "-c", _CHILD, resources.__file__, str(path), kind, str(timeout)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )


def _line(process: subprocess.Popen, timeout: float = 3) -> str:
    assert select.select([process.stdout], [], [], timeout)[0], "child did not respond"
    line = process.stdout.readline().strip()
    assert line, process.stderr.read() if process.poll() is not None else "child closed stdout"
    return line


def _release(process: subprocess.Popen) -> None:
    process.stdin.write("\n")
    process.stdin.flush()


def _finish(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                _release(process)
                process.wait(timeout=4)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=2)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def test_two_processes_overlap_and_third_waits(isolated_slots):
    processes = []
    try:
        first = _start(isolated_slots)
        processes.append(first)
        assert _line(first) == "entered"
        second = _start(isolated_slots)
        processes.append(second)
        assert _line(second) == "entered"  # First is still inside its slot.
        third = _start(isolated_slots)
        processes.append(third)
        assert not select.select([third.stdout], [], [], 0.15)[0]
        _release(first)
        assert _line(first) == "released"
        assert _line(third) == "entered"
    finally:
        _finish(processes)


@pytest.mark.parametrize("parent_kind", ["worker", "exclusive"])
def test_heavy_batch_and_worker_exclude_each_other(isolated_slots, parent_kind):
    parent = resources.hashing_worker_slot if parent_kind == "worker" else resources.hashing_lock
    child_kind = "exclusive" if parent_kind == "worker" else "worker"
    processes = []
    try:
        with parent():
            child = _start(isolated_slots, kind=child_kind, timeout=0.1)
            processes.append(child)
            assert _line(child) == "timeout"
    finally:
        _finish(processes)


def test_two_threads_overlap_without_process_mutex_serialization(isolated_slots):
    both_entered = threading.Barrier(2)
    errors = []

    def worker():
        try:
            with resources.hashing_worker_slot(timeout=1) as count:
                assert count == 1
                both_entered.wait(timeout=1)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert not errors


def test_pressure_drop_waits_for_occupied_higher_slot(isolated_slots, monkeypatch):
    processes = []
    try:
        with resources.hashing_worker_slot():  # Occupies slot 0.
            second = _start(isolated_slots)
            processes.append(second)
            assert _line(second) == "entered"  # Occupies slot 1.
        monkeypatch.setattr(resources, "host_hash_pressure", lambda: (1, "memory_headroom"))
        # Slot 0 is free, but admitting into it would exceed the reduced ceiling.
        with pytest.raises(resources.HashingResourceTimeout):
            with resources.hashing_worker_slot(timeout=0.1):
                pytest.fail("pressure reduction admitted around occupied high slot")
        _release(second)
        assert _line(second) == "released"
        with resources.hashing_worker_slot(timeout=0):
            pass
    finally:
        _finish(processes)


def test_hard_cap_allows_four_but_never_five(isolated_slots, monkeypatch):
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "99")
    processes = []
    try:
        for _ in range(4):
            process = _start(isolated_slots)
            processes.append(process)
            assert _line(process) == "entered"
        with pytest.raises(resources.HashingResourceTimeout):
            with resources.hashing_worker_slot(timeout=0.05):
                pytest.fail("fifth worker exceeded hard cap")
    finally:
        _finish(processes)


def test_reentrant_worker_and_exclusive_to_worker(isolated_slots):
    with resources.hashing_worker_slot(timeout=0):
        with resources.hashing_worker_slot(timeout=0) as count:
            assert count == 1
        with pytest.raises(resources.HashingResourceUpgradeError, match="leave the worker slot"):
            with resources.hashing_lock(timeout=0):
                pytest.fail("unsafe upgrade admitted")
    with resources.hashing_lock(timeout=0):
        with resources.hashing_worker_slot(timeout=0):
            with resources.hashing_lock(timeout=0):
                pass  # The original admission is still exclusive.


def test_exception_releases_both_slot_and_global_lock(isolated_slots):
    with pytest.raises(RuntimeError, match="read failed"):
        with resources.hashing_worker_slot():
            raise RuntimeError("read failed")
    assert not resources._open_fds
    with resources.hashing_lock(timeout=0):
        pass


@pytest.mark.parametrize("which", ["global", "gate", "slot"])
def test_symlink_lock_fails_closed(isolated_slots, tmp_path, which):
    global_path, gate, slots = resources._worker_slot_paths()
    path = {"global": global_path, "gate": gate, "slot": slots[0]}[which]
    target = tmp_path / "other"
    target.touch(mode=0o600)
    path.symlink_to(target)
    with pytest.raises(OSError):
        with resources.hashing_worker_slot(timeout=0):
            pytest.fail("unsafe lock admitted")


def test_replaced_slot_inode_fails_closed(isolated_slots, monkeypatch):
    actual = resources.fcntl.flock
    slot = resources._worker_slot_paths()[2][0]

    def replace_after_lock(fd, flags):
        actual(fd, flags)
        if flags == resources.fcntl.LOCK_EX | resources.fcntl.LOCK_NB:
            if Path(f"/proc/self/fd/{fd}").resolve() == slot:
                slot.unlink()
                slot.touch(mode=0o600)

    monkeypatch.setattr(resources.fcntl, "flock", replace_after_lock)
    with pytest.raises(PermissionError, match="changed while waiting"):
        with resources.hashing_worker_slot(timeout=0):
            pytest.fail("replaced slot admitted")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_child_unwind_keeps_reused_fd_and_parent_slot(isolated_slots, monkeypatch):
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "1")
    script = """
import importlib.util, os, pathlib, sys
spec = importlib.util.spec_from_file_location('fork_slots', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.hash_lock_path = lambda kind='file-hash': pathlib.Path(sys.argv[2])
module.host_hash_pressure = lambda: (4, 'admitted')
def fork_in_slot():
    with module.hashing_worker_slot():
        with module.hashing_worker_slot():
            old_fd = min(module._open_fds)
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
child, old_fd = fork_in_slot()
if child == 0:
    os.fstat(old_fd)
    assert not module._open_fds
    try:
        with module.hashing_worker_slot(timeout=0.1):
            raise AssertionError('child bypassed occupied parent slot')
    except module.HashingResourceTimeout:
        pass
    print('child-ok', flush=True)
    os._exit(0)
with module.hashing_lock(timeout=0):
    print('parent-ok', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, resources.__file__, str(isolated_slots)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["child-ok", "parent-ok"]


def test_no_flock_fallback_stays_serial_and_rejects_upgrade(isolated_slots, monkeypatch):
    monkeypatch.setattr(resources, "fcntl", None)
    with resources.hashing_worker_slot() as count:
        assert count == 1
        with resources.hashing_worker_slot(timeout=0):
            pass
        with pytest.raises(resources.HashingResourceUpgradeError):
            with resources.hashing_lock(timeout=0):
                pytest.fail("fallback upgrade admitted")
    with resources.hashing_lock(timeout=0):
        pass


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(isolated_slots, timeout):
    with pytest.raises(ValueError):
        with resources.hashing_worker_slot(timeout=timeout):
            pytest.fail("invalid timeout admitted")
