"""Focused owning tests for opt-in regular supervisor log provenance.

Collection imports the generic runtime/loop only; full implementation-daemon
startup remains covered by the separate genuine cold qualification.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import SupervisorLoop
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SupervisedChildSpec,
    adopt_supervised_child,
    launch_supervised_child,
)


# Regular latest-log provenance is explicitly opt-in; legacy alias behavior stays
# the default. These unit cases do not create productive child processes.
def _regular_log_marker_spec(tmp_path, monkeypatch, *, mode="regular-json"):
    monkeypatch.delenv(supervisor_runtime.SUPERVISED_CHILD_LOG_MARKER_MODE_ENV, raising=False)
    repo = tmp_path / "repo"
    (repo / "state").mkdir(parents=True)
    return SupervisedChildSpec(
        repo_root=repo, command=(sys.executable, "-c", "pass"),
        log_path=Path("state/current.log"), child_pid_path=Path("state/child.pid"),
        latest_log_path=Path("state/latest.log"),
        env={} if mode is None else {
            supervisor_runtime.SUPERVISED_CHILD_LOG_MARKER_MODE_ENV: mode},
    )


def _regular_log_marker_fake_launcher(monkeypatch):
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b"child run log bytes\n")
        kwargs["stdout"].flush()
        return SimpleNamespace(pid=2468, poll=lambda: None)
    monkeypatch.setattr(supervisor_runtime, "launch_process_child", launch)
    return calls


def _regular_log_marker_live_child(spec, monkeypatch):
    spec.resolve(spec.child_pid_path).write_text("2468\n", encoding="ascii")
    monkeypatch.setattr(supervisor_runtime, "pid_alive", lambda pid: pid == 2468)
    monkeypatch.setattr(supervisor_runtime, "process_args", lambda pid: " ".join(spec.command))


def _regular_log_marker_payload(spec, name="prior.log"):
    command = json.dumps({"command": list(spec.command)}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"schema": "ipfs_accelerate_py.agent_supervisor.supervised-child-log-marker@1",
            "command_id": "sha256:" + hashlib.sha256(command).hexdigest(),
            "log_basename": name}


def _regular_log_marker_write_fixture(spec, payload):
    marker = spec.resolve(spec.latest_log_path)
    marker.write_text(json.dumps(payload), encoding="utf-8")
    marker.chmod(0o600)
    return marker


def test_supervisor_runtime_regular_log_marker_launch_preserves_actual_output(tmp_path, monkeypatch):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    child = launch_supervised_child(spec)
    marker = spec.resolve(spec.latest_log_path)
    assert len(calls) == 1
    assert marker.is_file() and not marker.is_symlink()
    assert marker.stat().st_nlink == 1
    assert marker.stat().st_mode & 0o777 == 0o600
    assert json.loads(marker.read_text()) == _regular_log_marker_payload(spec, "current.log")
    assert child.log_path == spec.resolve(spec.log_path)
    assert child.latest_log_path == child.log_path
    assert child.log_path.read_bytes() == b"child run log bytes\n"
    assert child.child_pid_path.read_text() == "2468\n"
    assert not list(marker.parent.glob(".*.tmp"))


def test_supervisor_runtime_default_log_marker_keeps_legacy_symlink_call(tmp_path, monkeypatch):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch, mode=None)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    aliases = []
    # Inspect the exact legacy call without creating a relative symlink in a
    # strict evidence attempt. Cold integration separately exercises defaults.
    monkeypatch.setattr(Path, "symlink_to", lambda path, target: aliases.append((path, target)))
    child = launch_supervised_child(spec)
    assert len(calls) == 1
    assert aliases == [(spec.resolve(spec.latest_log_path), "current.log")]
    assert child.latest_log_path == spec.resolve(spec.latest_log_path)


@pytest.mark.parametrize("value", ["", "json", "REGULAR-JSON", "regular-json ", "0", "1"])
@pytest.mark.parametrize("operation", ["launch", "adopt"])
def test_supervisor_runtime_log_marker_invalid_mode_refuses_before_child(tmp_path, monkeypatch, value, operation):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch, mode=value)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    with pytest.raises(RuntimeError, match="invalid supervised child log marker mode"):
        (launch_supervised_child if operation == "launch" else adopt_supervised_child)(spec)
    assert calls == []
    assert list(spec.resolve(spec.log_path).parent.iterdir()) == []


def test_supervisor_runtime_log_marker_explicit_spec_mode_overrides_parent(tmp_path, monkeypatch):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    monkeypatch.setenv(supervisor_runtime.SUPERVISED_CHILD_LOG_MARKER_MODE_ENV, "invalid-parent-mode")
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    launch_supervised_child(spec)
    assert len(calls) == 1
    assert not spec.resolve(spec.latest_log_path).is_symlink()


def test_supervisor_runtime_log_marker_parent_opt_in_reaches_original_owner(tmp_path, monkeypatch):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch, mode=None)
    monkeypatch.setenv(supervisor_runtime.SUPERVISED_CHILD_LOG_MARKER_MODE_ENV, "regular-json")
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    child = launch_supervised_child(spec)
    assert len(calls) == 1
    assert json.loads(spec.resolve(spec.latest_log_path).read_text())["log_basename"] == "current.log"
    assert child.latest_log_path == child.log_path


@pytest.mark.parametrize("layout", ["missing-marker-config", "same-name", "other-directory"])
def test_supervisor_runtime_regular_log_marker_requires_contained_distinct_paths(tmp_path, monkeypatch, layout):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    if layout == "missing-marker-config":
        spec = replace(spec, latest_log_path=None)
    elif layout == "same-name":
        spec = replace(spec, latest_log_path=spec.log_path)
    else:
        spec = replace(spec, latest_log_path=Path("other/latest.log"))
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    with pytest.raises(RuntimeError):
        launch_supervised_child(spec)
    assert calls == []
    assert not spec.resolve(spec.log_path).exists()


def test_supervisor_runtime_regular_log_marker_adoption_retains_prior_run(tmp_path, monkeypatch):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    prior = spec.resolve(spec.log_path).with_name("prior.log")
    prior.write_bytes(b"original daemon output\n")
    prior.chmod(0o600)
    marker = _regular_log_marker_write_fixture(spec, _regular_log_marker_payload(spec))
    _regular_log_marker_live_child(spec, monkeypatch)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    before = marker.read_bytes()
    child = supervisor_runtime.adopt_or_launch_supervised_child(spec, launch_lock_path=marker.parent / "launch.lock")
    assert child is not None and child.pid == 2468
    assert child.log_path == prior and child.latest_log_path == prior
    assert marker.read_bytes() == before
    assert not spec.resolve(spec.log_path).exists()
    assert calls == []


@pytest.mark.parametrize("fault", [
    "missing", "invalid-json", "duplicate-field", "foreign-schema", "foreign-command",
    "extra-field", "array", "missing-basename", "traversal", "absolute", "backslash",
    "self", "null-basename", "empty-basename", "control-basename", "missing-target", "oversize",
])
def test_supervisor_runtime_regular_log_marker_bad_adoption_never_launches(tmp_path, monkeypatch, fault):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    prior = spec.resolve(spec.log_path).with_name("prior.log")
    prior.write_bytes(b"existing output\n")
    prior.chmod(0o600)
    payload = _regular_log_marker_payload(spec)
    if fault == "foreign-schema":
        payload["schema"] = "foreign/v1"
    elif fault == "foreign-command":
        payload["command_id"] = "sha256:" + "0" * 64
    elif fault == "extra-field":
        payload["extra"] = True
    elif fault == "array":
        payload = []
    elif fault == "missing-basename":
        del payload["log_basename"]
    elif fault in {"traversal", "absolute", "backslash", "self", "null-basename", "empty-basename", "control-basename", "missing-target"}:
        payload["log_basename"] = {"traversal": "../outside.log", "absolute": str(prior),
            "backslash": "..\\outside.log", "self": "latest.log", "null-basename": None,
            "empty-basename": "", "control-basename": "bad\nname.log", "missing-target": "absent.log"}[fault]
    marker = _regular_log_marker_write_fixture(spec, payload)
    if fault == "missing":
        marker.unlink()
    elif fault == "invalid-json":
        marker.write_text("{", encoding="utf-8")
    elif fault == "duplicate-field":
        marker.write_text(json.dumps(payload)[:-1] + ', "log_basename": "prior.log"}', encoding="utf-8")
    elif fault == "oversize":
        marker.write_bytes(b" " * (supervisor_runtime.SUPERVISED_CHILD_LOG_MARKER_MAX_BYTES + 1))
    _regular_log_marker_live_child(spec, monkeypatch)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    with pytest.raises(RuntimeError):
        supervisor_runtime.adopt_or_launch_supervised_child(spec, launch_lock_path=marker.parent / "launch.lock")
    assert calls == []
    assert prior.read_bytes() == b"existing output\n"
    assert not spec.resolve(spec.log_path).exists()


@pytest.mark.parametrize("fault", ["permissions", "foreign-owner", "hardlink"])
def test_supervisor_runtime_regular_log_marker_rejects_untrusted_marker_metadata(tmp_path, monkeypatch, fault):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    prior = spec.resolve(spec.log_path).with_name("prior.log")
    prior.write_text("prior output\n")
    prior.chmod(0o600)
    marker = _regular_log_marker_write_fixture(spec, _regular_log_marker_payload(spec))
    link = marker.with_name("marker-hardlink")
    if fault == "permissions":
        marker.chmod(0o640)
    elif fault == "foreign-owner":
        owner = os.geteuid()
        monkeypatch.setattr(supervisor_runtime.os, "geteuid", lambda: owner + 1)
    else:
        os.link(marker, link)
    _regular_log_marker_live_child(spec, monkeypatch)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    try:
        with pytest.raises(RuntimeError):
            adopt_supervised_child(spec)
        assert calls == []
    finally:
        link.unlink(missing_ok=True)


@pytest.mark.parametrize("alias", ["marker", "run-log"])
def test_supervisor_runtime_regular_log_marker_rejects_internal_absolute_alias(tmp_path, monkeypatch, alias):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    prior = spec.resolve(spec.log_path).with_name("prior.log")
    marker = spec.resolve(spec.latest_log_path)
    if alias == "marker":
        prior.write_text("prior output\n")
        prior.chmod(0o600)
        target = marker.with_name("marker-target.json")
        target.write_text(json.dumps(_regular_log_marker_payload(spec)), encoding="utf-8")
        target.chmod(0o600)
        alias_path = marker
    else:
        _regular_log_marker_write_fixture(spec, _regular_log_marker_payload(spec))
        target = prior.with_name("prior-target.log")
        target.write_text("prior output\n")
        target.chmod(0o600)
        alias_path = prior
    assert not alias_path.exists() and not alias_path.is_symlink()
    alias_path.symlink_to(target.absolute())
    _regular_log_marker_live_child(spec, monkeypatch)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    try:
        with pytest.raises(RuntimeError):
            supervisor_runtime.adopt_or_launch_supervised_child(spec, launch_lock_path=marker.parent / "launch.lock")
        assert calls == []
        assert target.is_file()
    finally:
        alias_path.unlink(missing_ok=True)


def test_supervisor_runtime_regular_log_marker_atomic_failure_preserves_prior_pointer(tmp_path, monkeypatch):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    marker = _regular_log_marker_write_fixture(spec, _regular_log_marker_payload(spec))
    before = marker.read_bytes()
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    def fail_replace(*args, **kwargs):
        raise OSError("injected marker replacement failure")
    monkeypatch.setattr(supervisor_runtime.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected marker replacement failure"):
        launch_supervised_child(spec)
    assert calls == []
    assert marker.read_bytes() == before
    assert not list(marker.parent.glob(".*.tmp"))


@pytest.mark.parametrize("fresh_log", [False, True])
def test_supervisor_runtime_regular_log_marker_watchdog_uses_output_mtime(tmp_path, monkeypatch, fresh_log):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_loop as loop_module
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    _regular_log_marker_fake_launcher(monkeypatch)
    child = launch_supervised_child(spec)
    marker = spec.resolve(spec.latest_log_path)
    now = time.time()
    os.utime(child.log_path, (now, now) if fresh_log else (1, 1))
    os.utime(marker, (1, 1) if fresh_log else (now, now))
    monkeypatch.setattr(loop_module, "pid_alive", lambda pid: True)
    loop = object.__new__(SupervisorLoop)
    loop.config = SimpleNamespace(watchdog_stale_after_seconds=60)
    observed = loop._quiescent_child_log_activity(child)
    assert observed["child_log_path"] == str(child.log_path)
    assert observed["child_log_fresh"] is fresh_log



def test_supervisor_runtime_regular_log_marker_broad_umask_still_creates_private_files(tmp_path, monkeypatch):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    previous = os.umask(0o002)
    try:
        child = launch_supervised_child(spec)
    finally:
        os.umask(previous)
    assert len(calls) == 1
    assert child.log_path.stat().st_mode & 0o777 == 0o600
    assert spec.resolve(spec.latest_log_path).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("operation", ["launch", "adopt"])
def test_supervisor_runtime_regular_log_marker_refuses_writable_existing_log_without_chmod(tmp_path, monkeypatch, operation):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    log = spec.resolve(spec.log_path)
    log.write_text("existing writable log\n")
    log.chmod(0o664)
    marker = _regular_log_marker_write_fixture(spec, _regular_log_marker_payload(spec, "current.log"))
    before = log.read_bytes()
    if operation == "adopt":
        _regular_log_marker_live_child(spec, monkeypatch)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    with pytest.raises(RuntimeError, match="untrusted supervised child run log"):
        (launch_supervised_child if operation == "launch" else adopt_supervised_child)(spec)
    assert calls == []
    assert log.read_bytes() == before
    assert log.stat().st_mode & 0o777 == 0o664
    assert not marker.is_symlink()



@pytest.mark.parametrize("liveness", ["alive", "dead"])
@pytest.mark.parametrize("pointer", ["missing", "malformed"])
def test_supervisor_runtime_regular_log_marker_orphan_refusal_precedes_identity_effects(tmp_path, monkeypatch, liveness, pointer):
    spec = _regular_log_marker_spec(tmp_path, monkeypatch)
    identity_path = spec.resolve(spec.child_pid_path).with_name("child.identity.json")
    identity_path.write_text("retained owner fixture\n", encoding="utf-8")
    identity = SimpleNamespace(command=spec.command, owner_scope={"role": "test"},
                               process_birth=SimpleNamespace(pid=2468))
    spec = replace(spec, env={**spec.env,
        supervisor_runtime.SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
        supervisor_runtime.SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps({"role": "test"})})
    if pointer == "malformed":
        marker = spec.resolve(spec.latest_log_path)
        marker.write_text("{", encoding="utf-8")
        marker.chmod(0o600)
    monkeypatch.setattr(supervisor_runtime, "pid_alive", lambda pid: False)
    monkeypatch.setattr(supervisor_runtime, "load_supervised_child_identity", lambda path: identity)
    monkeypatch.setattr(supervisor_runtime, "supervised_child_identity_liveness",
                        lambda item: supervisor_runtime.OwnerLiveness.ALIVE if liveness == "alive" else supervisor_runtime.OwnerLiveness.DEAD)
    monkeypatch.setattr(supervisor_runtime, "read_process_command_argv", lambda pid: spec.command)
    monkeypatch.setattr(supervisor_runtime, "_supervised_child_identity_session_quiesced", lambda item: True)
    calls = _regular_log_marker_fake_launcher(monkeypatch)
    parent = identity_path.parent
    before = {path.name: (path.read_bytes(), path.stat().st_ino) for path in parent.iterdir()}
    with pytest.raises(RuntimeError):
        adopt_supervised_child(spec)
    assert calls == []
    assert {path.name: (path.read_bytes(), path.stat().st_ino) for path in parent.iterdir()} == before
    assert not spec.resolve(spec.child_pid_path).exists()
