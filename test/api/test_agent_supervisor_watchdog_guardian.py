"""Guardian tests use synthetic /proc entries and never start real workers."""

from __future__ import annotations

import json
import errno
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import watchdog_guardian as guardian_module


@pytest.fixture
def setup_guardian(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "repo"
    directory = repo / "data" / "watchdog"
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({
        "report_path": "data/watchdog/report.json",
        "lanes": [{"bundle_key": "test", "state_dir": "data/watchdog/lane",
                   "state_prefix": "test", "log_path": "data/watchdog/lane/supervisor.log"}],
    }))
    source = repo / "external" / "ipfs_accelerate"
    module_path = (source / Path(*guardian_module.WATCHDOG_MODULE.split("."))).with_suffix(".py")
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# synthetic test module\n")
    proc = tmp_path / "proc"
    proc.mkdir()
    calls = []

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=701, poll=lambda: None)

    monkeypatch.setattr(guardian_module.subprocess, "Popen", spawn)
    guardian = guardian_module.WatchdogGuardian(manifest, repo, python="/test/venv/bin/python",
                                              apply=True, proc_root=proc, restart_cooldown=20)
    return guardian, proc, calls


def owner(proc: Path, guardian, pid: int = 123, *, state: str = "S", module: str | None = None,
          manifest: Path | None = None, repo: Path | None = None, uid: int | None = None):
    directory = proc / str(pid)
    directory.mkdir()
    fields = [state] + ["0"] * 18 + ["123456"]
    (directory / "stat").write_text(f"{pid} (python) " + " ".join(fields))
    argv = [guardian.python, "-m", module or guardian_module.WATCHDOG_MODULE,
            "--manifest-path", str(manifest or guardian.manifest),
            "--repo-root", str(repo or guardian.repo)]
    (directory / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    uid = os.getuid() if uid is None else uid
    (directory / "status").write_text("Uid:\t" + "\t".join([str(uid)] * 4) + "\n")
    (directory / "cwd").symlink_to(guardian.repo, target_is_directory=True)
    return directory


def fresh_report(guardian, pid=123, now=1000):
    path = guardian.manifest.parent / "report.json"
    path.write_text(json.dumps({"status": {"pid": pid, "state": "healthy"}}))
    os.utime(path, (now, now))


def test_adopts_verified_live_owner_and_preserves_progress(setup_guardian):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    fresh_report(guardian)
    lane = guardian.manifest.parent / "lane"
    lane.mkdir()
    (lane / "test_status.json").write_text(json.dumps({
        "pid": 456, "state": "healthy", "phase": "deterministic_command",
        "active_task_id": "VOG-046", "heartbeat_at_ms": 999000,
        "last_task_transition_at_ms": 900000,
    }))
    log = lane / "supervisor.log"
    log.write_text("live-click 224 app #save\nlive-click 225 app #next\n")
    os.utime(log, (998, 998))
    result = guardian.inspect(now=1000)
    assert result["action"] == "owner_present"
    assert result["owner_pid"] == 123 and result["owner_alive"] is True
    assert result["lanes"][0]["last_logged_control_index"] == 225
    assert result["lanes"][0]["log_age_seconds"] == 2
    assert result["lanes"][0]["heartbeat_age_seconds"] == 1
    assert not calls
    assert result["work_complete"] is result["completion_authority"] is False


def test_absent_owner_launches_detached_with_checkout_pythonpath(setup_guardian):
    guardian, _proc, calls = setup_guardian
    result = guardian.inspect(now=1000)
    assert result["action"] == "launched"
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv[:3] == ["/test/venv/bin/python", "-m", guardian_module.WATCHDOG_MODULE]
    assert argv[3:7] == ["--manifest-path", str(guardian.manifest), "--repo-root", str(guardian.repo)]
    assert options["start_new_session"] is True and options["close_fds"] is True
    assert options["cwd"] == guardian.repo
    assert options["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(guardian.repo / "external/ipfs_accelerate")
    assert result["next_launch_after"] == 1020


@pytest.mark.parametrize("state", ["Z", "X"])
def test_zombie_or_dead_owner_does_not_prevent_replacement(setup_guardian, state):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian, state=state)
    result = guardian.inspect(now=1000)
    assert result["action"] == "launched" and len(calls) == 1


@pytest.mark.parametrize("mismatch", ["module", "manifest", "repo", "uid"])
def test_pid_files_and_mismatched_processes_are_not_authority(setup_guardian, mismatch):
    guardian, proc, calls = setup_guardian
    kwargs = {mismatch: guardian.repo / "other"} if mismatch in {"manifest", "repo"} else {}
    if mismatch == "module":
        kwargs["module"] = "unrelated.supervisor_watchdog"
    if mismatch == "uid":
        kwargs["uid"] = os.getuid() + 1
    directory = owner(proc, guardian, **kwargs)
    (guardian.manifest.parent / "watchdog.pid").write_text("123\n")
    assert guardian.inspect(now=1000)["action"] == "launched"
    assert len(calls) == 1


def test_explicit_absolute_repo_from_foreign_cwd_is_adopted(setup_guardian):
    guardian, proc, calls = setup_guardian
    directory = owner(proc, guardian)
    (directory / "cwd").unlink()
    (directory / "cwd").symlink_to(guardian.repo.parent, target_is_directory=True)
    assert guardian.inspect(now=1000)["owner_pid"] == 123
    assert not calls


@pytest.mark.parametrize("flags", [["-uB"], ["-X", "dev"], ["-P"]])
def test_unfamiliar_python_flags_block_duplicate_launch(setup_guardian, flags):
    guardian, proc, calls = setup_guardian
    directory = owner(proc, guardian)
    argv = (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    argv[1:1] = flags
    (directory / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    assert guardian.inspect(now=1000)["action"] == "discovery_uncertain"
    assert not calls


@pytest.mark.parametrize("flags", [[], ["-u"], ["-u", "-B"], ["-uB"], ["-X", "dev"], ["-W", "ignore"], ["--"]])
def test_direct_source_file_invocation_blocks_duplicate_launch(setup_guardian, flags):
    guardian, proc, calls = setup_guardian
    directory = owner(proc, guardian)
    source = guardian.repo / "external/ipfs_accelerate" / Path(*guardian_module.WATCHDOG_MODULE.split("."))
    argv = [guardian.python, *flags, str(source.with_suffix(".py")), "--manifest-path", str(guardian.manifest)]
    (directory / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    assert guardian.inspect(now=1000)["action"] == "discovery_uncertain"
    assert not calls


@pytest.mark.parametrize("invocation", [
    ["-m", "ipfs_accelerate_py.agent_supervisor.rescue.repair_dispatcher", "--evidence-path", "{source}"],
    ["-u", "-m", "other.module", "--evidence-path", "{source}"],
    ["-X", "dev", "-m", "other.module", "--evidence-path", "{source}"],
    ["-m", "other.module", "-m", "{watchdog_module}", "--manifest-path", "{manifest}", "--repo-root", "{repo}", "{source}"],
    ["-mother.module", "--evidence-path", "{source}"],
    ["-uBm", "other.module", "--evidence-path", "{source}"],
    ["-umother.module", "--evidence-path", "{source}"],
    ["tool.py", "--evidence-path", "{source}"],
    ["tool.py", "-m", "{watchdog_module}", "--manifest-path", "{manifest}", "--repo-root", "{repo}"],
    ["-c", "print('not a watchdog')", "{source}"],
    ["-c", "print('not a watchdog')", "-m", "{watchdog_module}", "--manifest-path", "{manifest}", "--repo-root", "{repo}"],
    ["-", "{source}"],
    ["--", "tool.py", "{source}"],
    ["-W", "{source}", "tool.py"],
    ["-uW", "{source}", "tool.py"],
    ["-uX", "{source}", "tool.py"],
])
def test_python_application_data_cannot_impersonate_watchdog_invocation(setup_guardian, invocation):
    guardian, proc, calls = setup_guardian
    directory = owner(proc, guardian)
    source = (guardian.repo / "external/ipfs_accelerate" / Path(*guardian_module.WATCHDOG_MODULE.split("."))).with_suffix(".py")
    substitutions = dict(source=str(source), watchdog_module=guardian_module.WATCHDOG_MODULE,
                         manifest=str(guardian.manifest), repo=str(guardian.repo))
    argv = [guardian.python, *(argument.format(**substitutions) for argument in invocation)]
    (directory / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    result = guardian.inspect(now=1000)
    assert result["errors"] == [], result
    assert result["owners"] == []
    assert result["action"] == "launched"
    assert len(calls) == 1


def test_repair_monitor_evidence_path_does_not_taint_verified_watchdog_owner(setup_guardian):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    fresh_report(guardian)
    monitor = owner(proc, guardian, pid=124, module="ipfs_accelerate_py.agent_supervisor.rescue.repair_dispatcher")
    argv = (monitor / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    source = (guardian.repo / "external/ipfs_accelerate" / Path(*guardian_module.WATCHDOG_MODULE.split("."))).with_suffix(".py")
    argv.extend(["--evidence-path", str(source), "--apply"])
    (monitor / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    result = guardian.inspect(now=1000)
    assert result["action"] == "owner_present"
    assert result["owner_pid"] == 123
    assert result["errors"] == []
    assert not calls


def test_unrelated_editor_mentioning_source_file_does_not_block_recovery(setup_guardian):
    guardian, proc, calls = setup_guardian
    directory = owner(proc, guardian)
    source = guardian.repo / "external/ipfs_accelerate" / Path(*guardian_module.WATCHDOG_MODULE.split("."))
    argv = ["/usr/bin/tail", "-f", str(source.with_suffix(".py"))]
    (directory / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    assert guardian.inspect(now=1000)["action"] == "launched"
    assert len(calls) == 1


def test_duplicate_owners_are_reported_without_signals_or_launch(setup_guardian):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    owner(proc, guardian, pid=124)
    result = guardian.inspect(now=1000)
    assert result["action"] == "duplicates_detected"
    assert result["owner_pid"] is None
    assert [item["pid"] for item in result["owners"]] == [123, 124]
    assert not calls


def test_adopts_optional_repo_root_and_python_flags_and_refreshes_pid(setup_guardian):
    guardian, proc, calls = setup_guardian
    directory = owner(proc, guardian)
    argv = [guardian.python, "-u", "-B", "-m", guardian_module.WATCHDOG_MODULE,
            "--manifest-path", str(guardian.manifest)]
    (directory / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    pid_path = guardian.manifest.parent / "watchdog.pid"
    pid_path.write_text("999\n")
    result = guardian.inspect(now=1000)
    assert result["owner_pid"] == 123 and not calls
    assert pid_path.read_text() == "123\n"


@pytest.mark.parametrize("option", ["--manifest-path", "--repo-root"])
@pytest.mark.parametrize("extra", ["duplicate", "empty"])
def test_duplicate_or_empty_identity_options_are_rejected(setup_guardian, option, extra):
    guardian, proc, _calls = setup_guardian
    directory = owner(proc, guardian)
    argv = (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    if extra == "duplicate":
        argv.extend([option + "=" + str(guardian.repo if option == "--repo-root" else guardian.manifest)])
    else:
        argv[argv.index(option) + 1] = ""
    (directory / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    owners, errors = guardian_module.discover_watchdogs(guardian.manifest, guardian.repo, proc_root=proc)
    assert owners == [] and errors == ["identity_options_invalid:123"]
    assert guardian.inspect(now=1000)["action"] == "discovery_uncertain"


def test_observation_does_not_rewrite_watchdog_pid(setup_guardian):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    pid_path = guardian.manifest.parent / "watchdog.pid"
    pid_path.write_text("999\n")
    guardian.inspect(now=1000, once=True)
    assert pid_path.read_text() == "999\n" and not calls


@pytest.mark.parametrize("marker", guardian_module.STOP_MARKERS)
@pytest.mark.parametrize("location", ["repo", "manifest"])
def test_operator_hold_prevents_launch(setup_guardian, marker, location):
    guardian, _proc, calls = setup_guardian
    directory = guardian.repo if location == "repo" else guardian.manifest.parent
    (directory / marker).touch()
    result = guardian.inspect(now=1000)
    assert result["action"] == "held" and result["hold_markers"] == [str(directory / marker)]
    assert not calls


@pytest.mark.parametrize("once,apply", [(True, True), (False, False)])
def test_once_and_default_dry_run_never_launch(setup_guardian, once, apply):
    guardian, _proc, calls = setup_guardian
    guardian.apply = apply
    assert guardian.inspect(now=1000, once=once)["action"] == "would_launch"
    assert not calls


def test_live_unresponsive_owner_is_reported_without_replacement(setup_guardian):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    fresh_report(guardian, now=100)
    assert guardian.inspect(now=1000)["action"] == "owner_unresponsive"
    fresh_report(guardian, pid=999, now=1000)
    assert guardian.inspect(now=1000)["report_health"] == "different_owner"
    assert not calls


def test_spawn_failure_backoff_survives_guardian_restart(setup_guardian, monkeypatch):
    guardian, proc, calls = setup_guardian

    def fail(*args, **kwargs):
        calls.append(args)
        raise OSError(11, "test spawn unavailable")

    monkeypatch.setattr(guardian_module.subprocess, "Popen", fail)
    assert guardian.inspect(now=1000)["action"] == "launch_failed"
    resumed = guardian_module.WatchdogGuardian(guardian.manifest, guardian.repo, apply=True,
                                              proc_root=proc, restart_cooldown=20)
    assert resumed.inspect(now=1001)["action"] == "restart_cooldown"
    result = resumed.inspect(now=1020)
    assert result["action"] == "launch_failed" and result["next_launch_after"] == 1060
    assert len(calls) == 2


def test_singleton_lock_is_released_only_when_owner_closes(tmp_path):
    path = tmp_path / "guardian.lock"
    first = guardian_module.acquire_lock(path)
    assert first is not None
    try:
        assert guardian_module.acquire_lock(path) is None
    finally:
        first.close()
    second = guardian_module.acquire_lock(path)
    assert second is not None
    second.close()


def test_pid_reuse_during_discovery_fails_closed(setup_guardian, monkeypatch):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    identities = iter([("S", "123"), ("S", "124")])
    monkeypatch.setattr(guardian_module, "_stat_identity", lambda directory: next(identities))
    result = guardian.inspect(now=1000)
    assert result["action"] == "discovery_uncertain"
    assert result["errors"] == ["identity_changed:123"]
    assert not calls


def test_malformed_manifest_and_partial_report_fail_closed(setup_guardian):
    guardian, proc, calls = setup_guardian
    guardian.manifest.write_text('{"lanes": [null], "report_path": []}')
    result = guardian.inspect(now=1000)
    assert result["action"] == "configuration_error" and not calls
    saved = json.loads(guardian.status_path.read_text())
    assert saved["work_complete"] is saved["completion_authority"] is False


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EDQUOT])
def test_storage_admission_does_not_launch_or_consume_budget_and_recovers(setup_guardian, monkeypatch, number):
    guardian, _proc, calls = setup_guardian
    emergency, cleared = [], []
    monkeypatch.setattr(guardian_module, "publish_emergency_status", lambda *args, **kwargs: emergency.append(args[1]))
    monkeypatch.setattr(guardian_module, "clear_emergency_status", cleared.append)
    original = guardian_module.check_storage_capacity
    monkeypatch.setattr(guardian_module, "check_storage_capacity", lambda *_, **_kwargs: {
        "ok": False, "storage_pressure": True, "reason": errno.errorcode[number], "errno": number})
    for now in [1000, 1030, 1060]:
        result = guardian.inspect(now=now)
        assert result["action"] == "storage_blocked"
        assert result["work_complete"] is result["completion_authority"] is False
        assert guardian.launch_attempts == 0 and guardian.next_launch_after == 0
    assert not calls and emergency and not cleared
    monkeypatch.setattr(guardian_module, "check_storage_capacity", original)
    assert guardian.inspect(now=1090)["action"] == "launched"
    assert len(calls) == 1 and guardian.launch_attempts == 1
    assert cleared == [guardian.status_path]


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EDQUOT])
def test_prelaunch_persistence_failure_keeps_guardian_budget_and_does_not_spawn(setup_guardian, monkeypatch, number):
    guardian, _proc, calls = setup_guardian
    original = guardian_module.atomic_json
    emergency = []
    monkeypatch.setattr(guardian_module, "publish_emergency_status", lambda *args, **kwargs: emergency.append(args[1]))

    def full(*_args):
        raise OSError(number, "injected capacity failure")

    monkeypatch.setattr(guardian_module, "atomic_json", full)
    assert guardian.inspect(now=1000)["action"] == "storage_blocked"
    assert guardian.inspect(now=1030)["action"] == "storage_blocked"
    assert guardian.launch_attempts == 0 and guardian.next_launch_after == 0
    assert not calls and emergency
    monkeypatch.setattr(guardian_module, "atomic_json", original)
    assert guardian.inspect(now=1060)["action"] == "launched"
    assert len(calls) == 1


def test_telemetry_failure_retains_the_real_guardian_lock_and_existing_owner(setup_guardian, monkeypatch):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    fresh_report(guardian)
    emergency = []
    monkeypatch.setattr(guardian_module, "publish_emergency_status", lambda *args, **kwargs: emergency.append(args[1]))
    original = guardian_module.atomic_json

    def full(*_args):
        raise OSError(errno.ENOSPC, "injected full filesystem")

    lock = guardian_module.acquire_lock(guardian.lock_path)
    try:
        monkeypatch.setattr(guardian_module, "atomic_json", full)
        for now in [1000, 1030]:
            result = guardian.inspect(now=now)
            assert result["action"] == "storage_blocked" and result["owner_pid"] == 123
            assert guardian_module.acquire_lock(guardian.lock_path) is None
        assert not calls and emergency
        monkeypatch.setattr(guardian_module, "atomic_json", original)
        assert guardian.inspect(now=1060)["action"] == "owner_present"
        assert guardian_module.acquire_lock(guardian.lock_path) is None
    finally:
        lock.close()


def test_guardian_does_not_hide_an_unrelated_telemetry_io_error(setup_guardian, monkeypatch):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    fresh_report(guardian)

    def fail(*_args):
        raise OSError(errno.EIO, "unrelated disk error")

    monkeypatch.setattr(guardian_module, "atomic_json", fail)
    with pytest.raises(OSError) as error:
        guardian.inspect(now=1000)
    assert error.value.errno == errno.EIO and not calls


def test_guardian_main_keeps_polling_and_owns_lock_through_telemetry_pressure(setup_guardian, monkeypatch):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    fresh_report(guardian, now=time.time())
    guardian.poll_seconds = 1
    actual_inspect, actual_atomic = guardian.inspect, guardian_module.atomic_json
    actions, polls = [], []
    monkeypatch.setattr(guardian_module, "WatchdogGuardian", lambda *_args, **_kwargs: guardian)
    monkeypatch.setattr(guardian_module, "publish_emergency_status", lambda *_args, **_kwargs: None)

    def full(*_args):
        raise OSError(errno.ENOSPC, "injected persistent telemetry failure")

    def inspect(**kwargs):
        result = actual_inspect(**kwargs)
        actions.append(result["action"])
        return result

    def poll(seconds):
        polls.append(seconds)
        assert guardian_module.acquire_lock(guardian.lock_path) is None
        if len(polls) == 1:
            monkeypatch.setattr(guardian_module, "atomic_json", actual_atomic)
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(guardian_module, "atomic_json", full)
    monkeypatch.setattr(guardian, "inspect", inspect)
    monkeypatch.setattr(guardian_module.time, "sleep", poll)
    with pytest.raises(KeyboardInterrupt):
        guardian_module.main(["--manifest-path", str(guardian.manifest), "--repo-root", str(guardian.repo), "--apply"])
    assert actions == ["storage_blocked", "owner_present"]
    assert polls == [1, 1] and not calls
    lock = guardian_module.acquire_lock(guardian.lock_path)
    assert lock is not None
    lock.close()


def test_guardian_probes_declared_output_filesystems_even_when_metadata_is_writable(setup_guardian, monkeypatch):
    guardian, _proc, calls = setup_guardian
    data = json.loads(guardian.manifest.read_text())
    data["storage_paths"] = ["artifacts"]
    data["lanes"][0]["storage_paths"] = ["captures"]
    guardian.manifest.write_text(json.dumps(data))
    observed = []

    def capacity(paths, **kwargs):
        observed.extend(paths)
        return {"ok": False, "storage_pressure": True, "reason": "output_full"}

    monkeypatch.setattr(guardian_module, "check_storage_capacity", capacity)
    monkeypatch.setattr(guardian_module, "publish_emergency_status", lambda *_args, **_kwargs: None)
    assert guardian.inspect(now=1000)["action"] == "storage_blocked"
    assert guardian.repo / "artifacts" in observed and guardian.repo / "captures" in observed
    assert not calls and guardian.launch_attempts == 0


def test_guardian_reports_fresh_owner_storage_wait_without_claiming_healthy(setup_guardian, monkeypatch):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    fresh_report(guardian)
    path = guardian.manifest.parent / "report.json"
    path.write_text(json.dumps({"status": {"pid": 123, "state": "degraded", "phase": "storage_pressure"},
                                "reports": [{"bundle_key": "test", "action": "storage_wait"}]}))
    os.utime(path, (1000, 1000))
    monkeypatch.setattr(guardian_module, "publish_emergency_status", lambda *_args, **_kwargs: None)
    result = guardian.inspect(now=1000)
    assert result["action"] == "storage_blocked" and result["report_health"] == "storage_pressure"
    assert result["reported_state"] == "degraded" and not calls
    os.utime(path, (0, 0))
    assert guardian.inspect(now=1000)["action"] == "owner_unresponsive"


def test_guardian_uses_only_expected_verified_watchdog_emergency_record(setup_guardian, monkeypatch):
    guardian, proc, calls = setup_guardian
    owner(proc, guardian)
    fresh_report(guardian, now=1)
    report_path = guardian.manifest.parent / "report.json"

    def emergency(path, *, expected_pid, identity_check, **_kwargs):
        if path != report_path:
            return None
        assert expected_pid == 123 and identity_check(123) and not identity_check(999)
        return {"pid": 123, "observed_at_ms": 1000000,
                "payload": {"status": {"pid": 123, "state": "degraded"}, "storage_pressure": True}}

    monkeypatch.setattr(guardian_module, "read_emergency_status", emergency)
    monkeypatch.setattr(guardian_module, "publish_emergency_status", lambda *_args, **_kwargs: None)
    result = guardian.inspect(now=1000)
    assert result["action"] == "storage_blocked" and result["report_health"] == "storage_pressure"
    assert result["report_age_seconds"] == 0 and result["reported_state"] == "degraded"
    assert not calls
