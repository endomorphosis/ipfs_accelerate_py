"""A leased watchdog waits through disk pressure without spending recovery."""
import copy
import errno
import json
from contextlib import nullcontext

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import leased_service_watchdog as watch
from ipfs_accelerate_py.agent_supervisor.rescue import storage_pressure as storage
from test_agent_supervisor_leased_service_watchdog import case


@pytest.fixture(autouse=True)
def private_emergency_runtime(tmp_path, monkeypatch):
    directory = tmp_path / "watchdog-private-runtime"
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(storage, "_runtime_directory", lambda: directory)


def emergency(path):
    return storage.read_emergency_status(path, expected_pid=storage.os.getpid(),
                                         identity_check=lambda pid: pid == storage.os.getpid())


def test_capacity_pressure_defers_all_observation_and_preserves_durable_budget(case, monkeypatch):
    case.observe()
    prior = case.watchdog.state_path.read_bytes()
    memory = copy.deepcopy(case.watchdog.state)
    monkeypatch.setattr(case.watchdog, "_capacity", lambda: {"ok": False, "reason": "insufficient_free_bytes"})
    monkeypatch.setattr(case.watchdog, "_owners", lambda: pytest.fail("storage admission must precede owner/action work"))
    result = case.observe(now=3000, apply=True)
    assert result["action"] == "storage_wait" and result["storage_pressure"] is True
    assert result["durable_persistence"] is False
    assert case.watchdog.state == memory and case.watchdog.state_path.read_bytes() == prior
    assert case.host.restarts == 0
    assert emergency(case.watchdog.status_path)["payload"]["completion_authority"] is False


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EDQUOT])
@pytest.mark.parametrize("target", ["state", "status", "incident"])
def test_full_telemetry_is_explicit_wait_then_recovers_same_daemon(case, monkeypatch, number, target):
    case.observe()
    real_write = watch.atomic_json
    def failed(path, value):
        match = path == getattr(case.watchdog, target + "_path", None) if target != "incident" else path.name.startswith("incident-")
        if match:
            raise OSError(number, "injected persistent volume pressure")
        return real_write(path, value)
    monkeypatch.setattr(watch, "atomic_json", failed)
    result = case.watchdog._finish({"action": "attention_required", "phase": "observe",
                                  "incidents": [{"code": "required_barrier"}],
                                  "work_complete": False, "completion_authority": False})
    assert result["action"] == "storage_wait" and result["durable_persistence"] is False
    marker = emergency(case.watchdog.status_path)
    assert marker["errno"] == number and marker["payload"]["work_complete"] is False
    assert case.host.restarts == 0 and case.watchdog.state["restart_attempts"] == 0
    monkeypatch.setattr(watch, "atomic_json", real_write)
    recovered = case.observe(now=1001)
    assert recovered["action"] == "active_within_deadline" and recovered["durable_persistence"] is True
    assert emergency(case.watchdog.status_path) is None
    assert case.host.restarts == 0 and case.watchdog.state["restart_attempts"] == 0


@pytest.mark.parametrize("number", [errno.EACCES, errno.EIO])
def test_permission_and_unrelated_io_are_not_transient_storage_pressure(case, monkeypatch, number):
    def fail(*args):
        raise OSError(number, "non-capacity failure")
    monkeypatch.setattr(watch, "atomic_json", fail)
    with pytest.raises(OSError) as error:
        case.observe()
    assert error.value.errno == number
    assert emergency(case.watchdog.status_path) is None and case.host.restarts == 0


def test_pressure_arriving_at_restart_fence_spends_nothing(case, monkeypatch):
    case.observe()
    original = case.watchdog.state_path.read_bytes()
    capacities = iter([{"ok": True}, {"ok": False, "reason": "EDQUOT"}])
    monkeypatch.setattr(case.watchdog, "_capacity", lambda: next(capacities))
    result = case.observe(now=1030, apply=True)
    assert result["action"] == "storage_wait"
    assert case.host.restarts == 0 and case.watchdog.state["restart_attempts"] == 0
    assert case.watchdog.state_path.read_bytes() == original
    assert "pending_recovery" not in case.watchdog.state


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EDQUOT])
def test_failed_durable_admission_never_dispatches_or_spends_then_real_recovery_can_run(case, monkeypatch, number):
    case.observe()
    original = case.watchdog.state_path.read_bytes()
    real_write = watch.atomic_json
    def fail_admission(path, value):
        if path == case.watchdog.state_path and value.get("restart_attempts", 0) > 0:
            raise OSError(number, "restart admission could not persist")
        return real_write(path, value)
    monkeypatch.setattr(watch, "atomic_json", fail_admission)
    result = case.observe(now=1030, apply=True)
    assert result["action"] == "storage_wait" and case.host.restarts == 0
    assert case.watchdog.state["restart_attempts"] == 0 and "pending_recovery" not in case.watchdog.state
    assert case.watchdog.state_path.read_bytes() == original
    monkeypatch.setattr(watch, "atomic_json", real_write)
    recovered = case.observe(now=1031, apply=True)
    assert recovered["action"] == "service_restart_requested"
    assert case.host.restarts == 1 and case.watchdog.state["restart_attempts"] == 1
    assert json.loads(case.watchdog.state_path.read_text())["restart_attempts"] == 1


def test_verified_foreground_emergency_handles_unwritten_canonical_status_without_restart(case, monkeypatch):
    case.config.status_path.unlink()
    checks = []
    def marker(path, *, expected_pid, identity_check):
        assert path == case.config.status_path and expected_pid == 102
        checks.append(identity_check(expected_pid))
        assert identity_check(103) is False
        changed = case.host.processes[102]["start_ticks"]
        case.host.processes[102]["start_ticks"] = "reused"
        assert identity_check(expected_pid) is False
        case.host.processes[102]["start_ticks"] = changed
        return {"payload": {"active_task_id": "TEST-001", "storage_pressure": True}}
    monkeypatch.setattr(watch, "read_emergency_status", marker)
    result = case.observe(now=3000, apply=True)
    assert checks == [True] and result["action"] == "storage_wait"
    assert result["reason"] == "verified_foreground_storage_pressure" and result["owner_storage_telemetry"] is True
    assert result["durable_persistence"] is True and case.host.restarts == 0


def test_foreign_or_unverified_emergency_does_not_suppress_real_deadline(case, monkeypatch):
    monkeypatch.setattr(watch, "read_emergency_status", lambda *args, **kwargs: None)
    result = case.observe(now=1030, apply=True)
    assert result["action"] == "service_restart_requested" and case.host.restarts == 1


def test_missing_status_never_borrows_unrelated_matching_foreground(case, monkeypatch):
    case.config.status_path.unlink()
    foreign = copy.deepcopy(case.host.processes.pop(102))
    foreign.update(pid=902, ppid=1, start_ticks="9020")
    case.host.processes[902] = foreign
    monkeypatch.setattr(watch, "read_emergency_status", lambda *args, **kwargs: pytest.fail("no independently verified foreground"))
    owners = case.watchdog._owners()
    assert owners["child"] is None


def test_missing_status_multiple_matching_protected_descendants_are_ambiguous(case, monkeypatch):
    case.config.status_path.unlink()
    extra = copy.deepcopy(case.host.processes[102])
    extra.update(pid=106, start_ticks="1060")
    case.host.processes[106] = extra
    monkeypatch.setattr(watch, "read_emergency_status", lambda *args, **kwargs: pytest.fail("ambiguous foreground"))
    result = case.observe(now=3000, apply=True)
    assert result["action"] == "observation_blocked" and case.host.restarts == 0


def test_operator_hold_remains_authoritative_even_when_telemetry_volume_is_full(case, monkeypatch):
    (case.root / "HOLD").touch()
    monkeypatch.setattr(watch, "atomic_json", lambda *args: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))
    result = case.observe(now=3000, apply=True)
    assert result["action"] == "operator_hold" and result["storage_pressure"] is True
    assert case.host.restarts == 0


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EDQUOT])
def test_startup_lock_capacity_failure_waits_instead_of_exit_78(case, monkeypatch, capsys, number):
    monkeypatch.setattr(watch.Config, "from_dict", classmethod(lambda cls, value: case.config))
    monkeypatch.setattr(watch, "check_storage_capacity", lambda paths: {"ok": True})
    monkeypatch.setattr(watch, "acquire_lock", lambda path: (_ for _ in ()).throw(OSError(number, "full")))
    monkeypatch.setattr(watch, "LeasedServiceWatchdog", lambda *args: pytest.fail("must not create unlocked actor"))
    assert watch.main(["--config", str(case.config.inventory_path), "--once"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "storage_wait" and result["completion_authority"] is False
    assert emergency(case.watchdog.status_path)["errno"] == number


def test_startup_capacity_pressure_automatically_recovers_without_process_restart(case, monkeypatch, capsys):
    monkeypatch.setattr(watch.Config, "from_dict", classmethod(lambda cls, value: case.config))
    capacity = iter([{"ok": False, "reason": "insufficient_free_bytes"}, {"ok": True}])
    monkeypatch.setattr(watch, "check_storage_capacity", lambda paths: next(capacity))
    locks = []
    monkeypatch.setattr(watch, "acquire_lock", lambda path: locks.append(path) or nullcontext())
    calls = []
    class EndTest(BaseException):
        pass
    class Actor:
        def __init__(self, config):
            assert len(locks) == 1
        def observe(self, *, apply):
            calls.append(apply)
            return {"action": "active_within_deadline"}
    monkeypatch.setattr(watch, "LeasedServiceWatchdog", Actor)
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise EndTest
    monkeypatch.setattr(watch.time, "sleep", sleep)
    with pytest.raises(EndTest):
        watch.main(["--config", str(case.config.inventory_path), "--apply"])
    outputs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["action"] for item in outputs] == ["storage_wait", "active_within_deadline"]
    assert calls == [True] and len(locks) == 1 and len(sleeps) == 2


@pytest.mark.parametrize("number", [errno.EACCES, errno.EIO])
def test_startup_non_capacity_errors_keep_configuration_failure(case, monkeypatch, capsys, number):
    monkeypatch.setattr(watch.Config, "from_dict", classmethod(lambda cls, value: case.config))
    monkeypatch.setattr(watch, "check_storage_capacity", lambda paths: (_ for _ in ()).throw(OSError(number, "not capacity")))
    assert watch.main(["--config", str(case.config.inventory_path), "--once"]) == 78
    assert json.loads(capsys.readouterr().out)["action"] == "configuration_blocked"
