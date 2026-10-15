"""Real child census and private publication, with authored container guards.

The fixture authors only the privileged boundary/ancestor admission. Sealed
stdin, PID births, waits, event files, modes and hash joins remain real. No
provider, retry, database authority, or second coding launch is invoked.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_worker_attempt as module
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _sealed(raw, *, seal=True):
    descriptor = os.memfd_create("authored-worker-attempt", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    os.write(descriptor, raw)
    if seal:
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_SEAL | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW)
    return descriptor


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "owner"
    root.mkdir(mode=0o750)
    parent = root / "worker-observations"
    parent.mkdir(mode=0o750)
    boundary = root / "boundary.json"
    boundary.write_bytes(b"authored container boundary\n")
    boundary.chmod(0o444)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt = b"Keep every original requirement, including opaque text: \xff\n"
    descriptor = _sealed(prompt)
    os.lseek(descriptor, 3, os.SEEK_SET)

    def owner_boundary(path):
        assert Path(path) == boundary
        return dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), _sha(boundary.read_bytes())

    def protected_directory(path, *, owner, group):
        # /tmp and fixture UID ownership are authored rather than privileged
        # container evidence. The private leaf still has real fixed custody.
        info = path.lstat()
        if (path.resolve(strict=True) != path or not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o750
                or info.st_uid != owner or info.st_gid != group):
            raise ValueError("authored private observation directory changed")

    monkeypatch.setattr(guard, "_owner_boundary", owner_boundary)
    monkeypatch.setattr(guard, "_protected_directory", protected_directory)
    value = SimpleNamespace(boundary=boundary, workspace=workspace, parent=parent,
        input_fd=descriptor, prompt=prompt)
    yield value
    os.close(descriptor)


def _prepare(case, **changes):
    return module.prepare_owner_worker_attempt(**dict(boundary=case.boundary,
        workspace=case.workspace, input_fd=case.input_fd,
        absolute_deadline=time.monotonic() + 60) | changes)


def _path(case, attempt, event):
    return case.parent / ("worker-attempt-" + attempt.attempt_id + "-" + event + ".json")


def _event(case, attempt, event):
    path = _path(case, attempt, event)
    return path, path.read_bytes(), json.loads(path.read_bytes())


@pytest.fixture
def children():
    population = []
    def launch(code=7):
        child = subprocess.Popen([sys.executable, "-c",
            "import sys;sys.stdin.buffer.read();raise SystemExit(int(sys.argv[1]))", str(code)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        population.append(child)
        return child
    yield launch
    for child in population:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        if child.stdin is not None:
            child.stdin.close()


def test_real_child_has_closed_private_events_and_original_hash_chain(case, children):
    deadline = time.monotonic() + 60
    attempt = _prepare(case, absolute_deadline=deadline)
    assert os.lseek(case.input_fd, 0, os.SEEK_CUR) == 3
    prepared_path, prepared_raw, prepared = _event(case, attempt, "prepared")
    assert set(prepared) == module.COMMON_FIELDS
    assert len(attempt.attempt_id) == 32 and int(attempt.attempt_id, 16) >= 0
    assert prepared["native_prompt_sha256"] == _sha(case.prompt)
    assert prepared["workspace_identity_sha256"] == _sha(str(case.workspace).encode())
    assert prepared["boundary_manifest_sha256"] == _sha(case.boundary.read_bytes())
    assert prepared["absolute_deadline_ns"] == int(deadline * 1_000_000_000)
    assert prepared["owner_pid"] == os.getpid() and prepared["owner_start_ticks"] > 0
    child = children()
    attempt.note_spawned(child)
    spawned_path, spawned_raw, spawned = _event(case, attempt, "spawned")
    assert set(spawned) == module.SPAWNED_FIELDS
    assert spawned["prepared_sha256"] == _sha(prepared_raw)
    assert spawned["child_pid"] == child.pid and spawned["child_start_ticks"] > 0
    child.communicate(b"go", timeout=10)
    result = attempt.finish(waited=True, cleanup_succeeded=True, brokers_closed=True)
    retired_path, retired_raw, retired = _event(case, attempt, "retired")
    assert result == retired and set(retired) == module.RETIRED_FIELDS
    assert retired["spawned_sha256"] == _sha(spawned_raw)
    assert retired["prepared_sha256"] == _sha(prepared_raw)
    assert retired["waited"] is True and retired["child_returncode"] == 7
    assert retired["cleanup_succeeded"] is retired["brokers_closed"] is True
    for path, raw in ((prepared_path, prepared_raw), (spawned_path, spawned_raw), (retired_path, retired_raw)):
        value = json.loads(raw)
        assert value["schema"] == module.SCHEMA and value["purpose"] == "coding"
        assert value["observation_only"] is True
        assert value["execution_authority"] is value["completion_authority"] is value["retry_authority"] is False
        assert stat.S_IMODE(path.stat().st_mode) == 0o400 and path.stat().st_nlink == 1
        assert case.prompt not in raw and str(case.workspace).encode() not in raw
        assert json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() == raw
        for name in module.COMMON_FIELDS - {"event"}:
            assert value[name] == prepared[name]
    result["waited"] = False
    assert json.loads(retired_path.read_bytes())["waited"] is True


def test_two_preparations_have_independent_ids_and_no_adoption(case):
    first, second = _prepare(case), _prepare(case)
    assert first.attempt_id != second.attempt_id
    assert _path(case, first, "prepared").exists() and _path(case, second, "prepared").exists()
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        first.attempt_id = second.attempt_id
    with pytest.raises(ValueError, match="live owner"):
        module.OwnerWorkerAttempt(token=object(), common={}, boundary=case.boundary,
            workspace=case.workspace, workspace_identity=(), parent=case.parent, parent_identity=())


def test_prepared_without_a_child_can_retire_as_explicit_unstarted(case):
    attempt = _prepare(case)
    with pytest.raises(ValueError, match="settled real child"):
        attempt.finish(waited=True, cleanup_succeeded=True, brokers_closed=True)
    result = attempt.finish(waited=False, cleanup_succeeded=True, brokers_closed=True)
    assert result["spawned_sha256"] is result["child_pid"] is result["child_start_ticks"] is None
    assert result["child_returncode"] is None and result["waited"] is False
    assert not _path(case, attempt, "spawned").exists()
    with pytest.raises(ValueError, match="consumed"):
        attempt.finish(waited=False, cleanup_succeeded=True, brokers_closed=True)


def test_live_unwaited_child_preserves_unknown_returncode_and_cleanup(case, children):
    attempt, child = _prepare(case), children()
    attempt.note_spawned(child)
    with pytest.raises(ValueError, match="settled real child"):
        attempt.finish(waited=True, cleanup_succeeded=True, brokers_closed=True)
    result = attempt.finish(waited=False, cleanup_succeeded=False, brokers_closed=False)
    assert result["child_pid"] == child.pid and result["spawned_sha256"] is not None
    assert result["child_returncode"] is None and result["waited"] is False
    assert result["cleanup_succeeded"] is result["brokers_closed"] is False


@pytest.mark.parametrize("deadline", [None, True, False, "120", float("nan"), float("inf"),
    float("-inf"), 10 ** 500, -1])
def test_malformed_deadline_never_publishes(case, deadline):
    with pytest.raises(ValueError):
        _prepare(case, absolute_deadline=deadline)
    assert not list(case.parent.iterdir())


def test_expired_preparation_and_birth_do_not_reset_deadline(case, children, monkeypatch):
    with pytest.raises(TimeoutError, match="exhausted"):
        _prepare(case, absolute_deadline=time.monotonic() - 1)
    attempt = _prepare(case)
    deadline = json.loads(_path(case, attempt, "prepared").read_bytes())["absolute_deadline_ns"]
    child = children()
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: deadline)
    with pytest.raises(TimeoutError, match="exhausted"):
        attempt.note_spawned(child)
    assert not _path(case, attempt, "spawned").exists()
    with pytest.raises(ValueError, match="incomplete"):
        attempt.finish(waited=False, cleanup_succeeded=True, brokers_closed=True)


def test_retirement_is_persisted_after_original_deadline(case, children, monkeypatch):
    attempt, child = _prepare(case), children(code=0)
    attempt.note_spawned(child)
    child.communicate(b"go", timeout=10)
    deadline = json.loads(_path(case, attempt, "prepared").read_bytes())["absolute_deadline_ns"]
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: deadline + 1)
    result = attempt.finish(waited=True, cleanup_succeeded=True, brokers_closed=True)
    assert result["child_returncode"] == 0 and _path(case, attempt, "retired").exists()


@pytest.mark.parametrize("sealed,raw", [(False, b"unsealed"), (True, b"x" * (module.MAX_PROMPT_BYTES + 1))])
def test_unsealed_or_oversized_input_never_publishes(case, sealed, raw):
    fd = _sealed(raw, seal=sealed)
    try:
        with pytest.raises(ValueError):
            _prepare(case, input_fd=fd)
    finally:
        os.close(fd)
    assert not list(case.parent.iterdir())


@pytest.mark.parametrize("field", ["waited", "cleanup_succeeded", "brokers_closed"])
@pytest.mark.parametrize("bad", [None, 0, 1, "yes"])
def test_retirement_flags_are_exact_booleans(case, field, bad):
    attempt = _prepare(case)
    flags = dict(waited=False, cleanup_succeeded=True, brokers_closed=True)
    flags[field] = bad
    with pytest.raises(ValueError, match="boolean"):
        attempt.finish(**flags)
    assert not _path(case, attempt, "retired").exists()


def test_fake_child_cannot_record_a_birth(case):
    attempt = _prepare(case)
    with pytest.raises(ValueError, match="real Popen"):
        attempt.note_spawned(SimpleNamespace(pid=os.getpid(), returncode=None, _child_created=True))
    assert not _path(case, attempt, "spawned").exists()


@pytest.mark.parametrize("failure", ["kernel_birth_unavailable", "already_waited"])
def test_recognized_real_child_failure_can_never_retire_as_no_spawn(case, children, monkeypatch, failure):
    attempt, child = _prepare(case), children()
    if failure == "already_waited":
        child.communicate(b"go", timeout=10)
    else:
        original = module._birth
        def unavailable(pid):
            if pid == child.pid:
                raise OSError("authored kernel observation unavailable")
            return original(pid)
        monkeypatch.setattr(module, "_birth", unavailable)
    with pytest.raises((ValueError, OSError)):
        attempt.note_spawned(child)
    if child.returncode is None:
        child.communicate(b"go", timeout=10)
    with pytest.raises(ValueError, match="incomplete"):
        attempt.finish(waited=False, cleanup_succeeded=True, brokers_closed=True)
    assert not _path(case, attempt, "retired").exists()


def test_child_birth_is_one_use(case, children):
    attempt, first, second = _prepare(case), children(), children()
    attempt.note_spawned(first)
    raw = _path(case, attempt, "spawned").read_bytes()
    with pytest.raises(ValueError, match="consumed"):
        attempt.note_spawned(second)
    assert _path(case, attempt, "spawned").read_bytes() == raw


@pytest.mark.parametrize("change", ["bytes", "mode", "replacement", "hardlink"])
def test_prior_immutable_event_tampering_refuses_another_event(case, children, change):
    attempt = _prepare(case)
    path = _path(case, attempt, "prepared")
    original = path.read_bytes()
    if change == "hardlink":
        os.link(path, case.parent / "foreign-hardlink")
    elif change == "replacement":
        path.rename(case.parent / "retained-original")
        path.write_bytes(original)
        path.chmod(0o400)
    else:
        path.chmod(0o600)
        if change == "bytes":
            path.write_bytes(b"foreign replacement bytes")
            path.chmod(0o400)
    child = children()
    with pytest.raises(ValueError, match="immutable"):
        attempt.note_spawned(child)
    assert not _path(case, attempt, "spawned").exists()
    assert path.exists()


@pytest.mark.parametrize("event", ["spawned", "retired"])
def test_existing_event_is_never_overwritten_or_removed(case, children, event):
    attempt, child = _prepare(case), children()
    if event == "retired":
        attempt.note_spawned(child)
        child.communicate(b"go", timeout=10)
    path = _path(case, attempt, event)
    path.write_bytes(b"foreign retained observation")
    path.chmod(0o400)
    with pytest.raises(FileExistsError):
        if event == "spawned":
            attempt.note_spawned(child)
        else:
            attempt.finish(waited=True, cleanup_succeeded=True, brokers_closed=True)
    assert path.read_bytes() == b"foreign retained observation"


def test_directory_replacement_is_not_adopted(case):
    attempt = _prepare(case)
    original = case.parent.with_name("retained-observations")
    case.parent.rename(original)
    case.parent.mkdir(mode=0o750)
    with pytest.raises(ValueError, match="directory changed"):
        attempt.finish(waited=False, cleanup_succeeded=True, brokers_closed=True)
    assert not list(case.parent.iterdir())
    assert len(list(original.iterdir())) == 1


def test_boundary_drift_refuses_retirement(case):
    attempt = _prepare(case)
    case.boundary.chmod(0o644)
    case.boundary.write_bytes(b"different boundary")
    with pytest.raises(ValueError, match="boundary changed"):
        attempt.finish(waited=False, cleanup_succeeded=True, brokers_closed=True)
    assert not _path(case, attempt, "retired").exists()


def test_publish_failure_after_birth_preserves_incomplete_chain(case, children, monkeypatch):
    attempt, child = _prepare(case), children()
    original = module.os.write
    def fail(fd, raw):
        original(fd, raw[:5])
        raise OSError("authored partial write")
    monkeypatch.setattr(module.os, "write", fail)
    with pytest.raises(OSError, match="partial write"):
        attempt.note_spawned(child)
    assert _path(case, attempt, "spawned").read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        attempt.finish(waited=False, cleanup_succeeded=True, brokers_closed=True)
    assert not _path(case, attempt, "retired").exists()
