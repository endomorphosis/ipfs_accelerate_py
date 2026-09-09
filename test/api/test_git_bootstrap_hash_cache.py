"""PID-local Git observations before broker handoff; no real executable edits."""

from __future__ import annotations

import hashlib
import json
import os
import select
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import shared_hashing
from test.api.test_agent_supervisor_configured_typed_grant_handoff import aseh_operator

from ipfs_accelerate_py import _hash_resources as resources


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class ExecutableOS:
    """Keep real descriptor/path behavior, model root ownership hermetically."""

    def __init__(self):
        self.overrides = {"st_uid": 0}
        self.pid = None

    def __getattr__(self, name):
        return getattr(os, name)

    def metadata(self, metadata):
        fields = {name: getattr(metadata, name) for name in dir(metadata) if name.startswith("st_")}
        return SimpleNamespace(**(fields | self.overrides))

    def lstat(self, path):
        return self.metadata(os.lstat(path))

    def fstat(self, descriptor):
        return self.metadata(os.fstat(descriptor))

    def getpid(self):
        return os.getpid() if self.pid is None else self.pid


@pytest.fixture
def git_probe(tmp_path, monkeypatch):
    path = tmp_path / "git"
    path.write_bytes(b"hermetic executable bytes\n")
    path.chmod(0o755)
    clock = Clock()
    executable_os = ExecutableOS()
    calls = []
    admissions = []
    original_hash = shared_hashing.hash_descriptor

    def hash_file(descriptor, **kwargs):
        calls.append(kwargs)
        return original_hash(descriptor, **kwargs)

    def admit(descriptor, *, expected_path):
        assert expected_path == path and os.fstat(descriptor).st_ino == path.stat().st_ino
        admissions.append(descriptor)
        return {"path": str(path)}

    monkeypatch.setattr(aseh_operator, "TRUSTED_GIT", path)
    monkeypatch.setattr(aseh_operator, "os", executable_os)
    monkeypatch.setattr(aseh_operator, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(aseh_operator, "_TRUSTED_GIT_IDENTITY", None)
    monkeypatch.setattr(aseh_operator, "_TRUSTED_GIT_BOOTSTRAP_OBSERVATION", None)
    monkeypatch.setattr(aseh_operator, "_r16_admit_unprivileged_executable_fd", admit)
    monkeypatch.setattr(aseh_operator, "hash_descriptor", hash_file)
    monkeypatch.setattr(shared_hashing, "default_hash_connection", lambda: None)
    monkeypatch.delenv("IPFS_HASH_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "2")
    monkeypatch.setattr(resources, "hash_lock_path", lambda kind="file-hash": tmp_path / "hash.lock")
    monkeypatch.setattr(resources, "host_hash_pressure", lambda: (4, "test"))
    yield SimpleNamespace(
        path=path, clock=clock, os=executable_os, calls=calls, admissions=admissions,
        hash=hash_file,
    )
    assert not resources._open_fds


def test_default_24h_bootstrap_hit_never_renews_and_expiry_reads_again(git_probe):
    probe = git_probe
    assert aseh_operator._trusted_git_executable() == str(probe.path)
    observed = aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION
    assert observed[:2] == (os.getpid(), str(probe.path))
    assert len(observed[2]) == 9 and observed[2][4] == probe.path.stat().st_gid
    assert observed[3] == hashlib.sha256(probe.path.read_bytes()).hexdigest()
    assert observed[4] == 1000.0
    probe.clock.now += 86_399.0
    assert aseh_operator._trusted_git_executable() == str(probe.path)
    assert len(probe.calls) == 1 and len(probe.admissions) == 2
    assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION is observed
    probe.clock.now += 1.0
    assert aseh_operator._trusted_git_executable() == str(probe.path)
    assert len(probe.calls) == 2 and len(probe.admissions) == 3
    assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION[4] == probe.clock.now


def test_bootstrap_ttl_starts_before_the_first_hash_or_resource_wait(git_probe, monkeypatch):
    probe = git_probe

    def slow(descriptor, **kwargs):
        result = probe.hash(descriptor, **kwargs)
        probe.clock.now += 10
        return result

    monkeypatch.setattr(aseh_operator, "hash_descriptor", slow)
    monkeypatch.setenv("IPFS_HASH_CACHE_TTL_SECONDS", "10")
    aseh_operator._trusted_git_executable()
    assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION[4] == 1000.0
    aseh_operator._trusted_git_executable()
    assert len(probe.calls) == 2  # It expired while the first read was running.


@pytest.mark.parametrize("strict,ttl", [(True, "86400"), (False, "0")])
def test_strict_or_zero_ttl_always_rereads_without_renewing_local_cache(
    git_probe, monkeypatch, strict, ttl,
):
    probe = git_probe
    aseh_operator._trusted_git_executable()
    cached = aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION
    monkeypatch.setenv("IPFS_HASH_CACHE_TTL_SECONDS", ttl)
    for _ in range(2):
        aseh_operator._trusted_git_executable(strict=strict)
    assert len(probe.calls) == 3
    assert all(call["strict"] is True for call in probe.calls)
    assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION is cached


def test_same_size_preserved_mtime_edit_cannot_reuse_digest_or_escape_drift_check(git_probe):
    probe = git_probe
    aseh_operator._trusted_git_executable()
    cached = aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION
    before = probe.path.stat()
    probe.path.write_bytes(b"!" * before.st_size)
    os.utime(probe.path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert probe.path.stat().st_mtime_ns == before.st_mtime_ns
    assert probe.path.stat().st_ctime_ns != before.st_ctime_ns
    with pytest.raises(aseh_operator.OperatorError, match="identity drifted"):
        aseh_operator._trusted_git_executable()
    assert len(probe.calls) == 2
    assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION is cached


def test_gid_is_part_of_cache_and_process_drift_witness(git_probe):
    probe = git_probe
    aseh_operator._trusted_git_executable()
    probe.os.overrides["st_gid"] = probe.path.stat().st_gid + 1
    with pytest.raises(aseh_operator.OperatorError, match="identity drifted"):
        aseh_operator._trusted_git_executable()
    assert len(probe.calls) == 2


def test_configured_owner_bypasses_even_a_poisoned_local_observation(git_probe, monkeypatch):
    probe = git_probe
    aseh_operator._trusted_git_executable()
    monkeypatch.setattr(aseh_operator, "_TRUSTED_GIT_BOOTSTRAP_OBSERVATION", ("malformed",))
    requests = []

    class Owner:
        def hash_observation(self, request):
            requests.append(request)
            return {"status": "hit", "sha256": hashlib.sha256(probe.path.read_bytes()).hexdigest()}

    owner = Owner()
    monkeypatch.setattr(shared_hashing, "default_hash_connection", lambda: owner)
    assert aseh_operator._trusted_git_executable() == str(probe.path)
    assert len(probe.calls) == 2 and probe.calls[-1]["connection"] is owner
    assert len(requests) == 1
    assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION == ("malformed",)


def test_configured_owner_error_does_not_fall_back_to_warm_local_cache(git_probe, monkeypatch):
    probe = git_probe
    aseh_operator._trusted_git_executable()

    class Owner:
        def hash_observation(self, request):
            raise RuntimeError("owner transport unavailable")

    monkeypatch.setattr(shared_hashing, "default_hash_connection", lambda: Owner())
    with pytest.raises(RuntimeError, match="owner transport unavailable"):
        aseh_operator._trusted_git_executable()
    assert len(probe.calls) == 2


def test_strict_does_not_require_a_broker_even_if_handoff_is_broken(git_probe, monkeypatch):
    def broken():
        raise RuntimeError("broken owner handoff")

    monkeypatch.setattr(shared_hashing, "default_hash_connection", broken)
    aseh_operator._trusted_git_executable(strict=True)
    assert len(git_probe.calls) == 1


@pytest.mark.parametrize("replacement", [
    (), [], ("wrong",), (True, "path", (), "a" * 64, 1000.0),
    (1, "path", (1,) * 9, "A" * 64, 1000.0),
    (1, "path", (1,) * 9, "a" * 64, float("nan")),
    (1, "path", (1,) * 9, "a" * 64, float("inf")),
    (1, "path", (1,) * 9, "a" * 64, -1.0),
    (1, "path", (1,) * 9, "a" * 64, 10**1000),
    (1, "path", [1] * 9, "a" * 64, 1000.0),
    (1, "path", (True,) * 9, "a" * 64, 1000.0),
])
def test_malformed_local_observations_fail_closed(git_probe, monkeypatch, replacement):
    monkeypatch.setattr(aseh_operator, "_TRUSTED_GIT_BOOTSTRAP_OBSERVATION", replacement)
    with pytest.raises(aseh_operator.OperatorError, match="observation is malformed"):
        aseh_operator._trusted_git_executable()
    assert git_probe.calls == []


def test_pid_change_invalidates_observation(git_probe):
    probe = git_probe
    aseh_operator._trusted_git_executable()
    probe.os.pid = os.getpid() + 1
    aseh_operator._trusted_git_executable()
    assert len(probe.calls) == 2
    assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION[0] == probe.os.pid


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_real_fork_child_rehashes_inherited_bootstrap_observation(git_probe):
    aseh_operator._trusted_git_executable()
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            aseh_operator._trusted_git_executable()
            record = {
                "calls": len(git_probe.calls), "pid": os.getpid(),
                "cache_pid": aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION[0],
            }
            os.write(write_fd, json.dumps(record).encode("ascii"))
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(write_fd)
    finished = False
    try:
        assert select.select([read_fd], [], [], 3)[0], "forked bootstrap did not finish"
        record = json.loads(os.read(read_fd, 4096))
        waited, status = os.waitpid(child, 0)
        finished = True
        assert waited == child and os.waitstatus_to_exitcode(status) == 0
        assert record == {"calls": 2, "pid": child, "cache_pid": child}
        assert len(git_probe.calls) == 1
        assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION[0] == os.getpid()
    finally:
        os.close(read_fd)
        if not finished:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)


def test_cache_hit_still_checks_executable_admission_path(git_probe, monkeypatch):
    aseh_operator._trusted_git_executable()
    monkeypatch.setattr(
        aseh_operator, "_r16_admit_unprivileged_executable_fd",
        lambda descriptor, **kwargs: {"path": "/foreign/git"},
    )
    with pytest.raises(aseh_operator.OperatorError, match="identity drifted"):
        aseh_operator._trusted_git_executable()
    assert len(git_probe.calls) == 1


def test_post_hash_metadata_drift_does_not_publish_bootstrap_cache(git_probe, monkeypatch):
    probe = git_probe

    def mutate(descriptor, **kwargs):
        result = probe.hash(descriptor, **kwargs)
        probe.path.chmod(0o700)
        return result

    monkeypatch.setattr(aseh_operator, "hash_descriptor", mutate)
    with pytest.raises(aseh_operator.OperatorError, match="identity drifted"):
        aseh_operator._trusted_git_executable()
    assert aseh_operator._TRUSTED_GIT_BOOTSTRAP_OBSERVATION is None
    assert aseh_operator._TRUSTED_GIT_IDENTITY is None


def test_concurrent_bootstrap_misses_do_not_hold_a_mutex_during_hashing(git_probe, monkeypatch):
    barrier = threading.Barrier(2, timeout=2)

    def overlap(descriptor, **kwargs):
        barrier.wait()
        return git_probe.hash(descriptor, **kwargs)

    monkeypatch.setattr(aseh_operator, "hash_descriptor", overlap)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(aseh_operator._trusted_git_executable) for _ in range(2)]
        assert [future.result(timeout=3) for future in futures] == [str(git_probe.path)] * 2
    assert len(git_probe.calls) == 2
    aseh_operator._trusted_git_executable()
    assert len(git_probe.calls) == 2


def test_monotonic_rollback_forces_fresh_observation(git_probe):
    aseh_operator._trusted_git_executable()
    git_probe.clock.now -= 1
    aseh_operator._trusted_git_executable()
    assert len(git_probe.calls) == 2
