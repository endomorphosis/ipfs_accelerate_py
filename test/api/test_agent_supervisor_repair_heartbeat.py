"""A live main loop pings systemd without renewing a coding-job deadline."""
import importlib.util
import math
import os
from pathlib import Path
import socket
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import systemd_heartbeat as heartbeat


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class Datagram:
    def __init__(self, calls, failure=""):
        self.calls = calls
        self.failure = failure
        self.closed = False

    def settimeout(self, value):
        assert value == 0.2
        if self.failure == "timeout":
            raise OSError("socket refused timeout")

    def sendto(self, body, address):
        self.calls.append((body, address))
        if self.failure == "send":
            raise OSError("notification receiver unavailable")
        return 0 if self.failure == "short" else len(body)

    def close(self):
        self.closed = True
        if self.failure == "close":
            raise OSError("close interrupted")


def environment(**changes):
    return {"NOTIFY_SOCKET": "/run/user/1000/systemd/notify", "WATCHDOG_USEC": "2000000",
            "WATCHDOG_PID": str(os.getpid()), **changes}


def make_heartbeat(clock=None, *, env=None, failure=""):
    clock = clock or Clock()
    calls, sockets = [], []

    def factory(family, kind):
        assert (family, kind) == (socket.AF_UNIX, socket.SOCK_DGRAM)
        if failure == "create":
            raise OSError("socket unavailable")
        client = Datagram(calls, failure)
        sockets.append(client)
        return client

    result = heartbeat.SystemdHeartbeat.from_environment(
        environment() if env is None else env, clock=clock, socket_factory=factory)
    return result, clock, calls, sockets


def test_absent_systemd_environment_preserves_one_original_wait():
    h, _, calls, sockets = make_heartbeat(env={})
    assert not h.enabled and h.disabled_reason == "watchdog_environment_absent"

    class Process:
        def __init__(self):
            self.timeouts = []

        def wait(self, timeout):
            self.timeouts.append(timeout)
            return 17

    child = Process()
    assert not h.notify_due(force=True)
    assert h.wait_for_process(child, 2400) == 17
    assert child.timeouts == [2400]
    assert calls == sockets == []


@pytest.mark.parametrize("changes,reason", [
    ({"WATCHDOG_USEC": "0"}, "watchdog_interval_invalid"),
    ({"WATCHDOG_USEC": "-1"}, "watchdog_interval_invalid"),
    ({"WATCHDOG_USEC": "1.5"}, "watchdog_interval_invalid"),
    ({"WATCHDOG_USEC": "nan"}, "watchdog_interval_invalid"),
    ({"WATCHDOG_USEC": True}, "watchdog_interval_invalid"),
    ({"WATCHDOG_USEC": "2" * 1000}, "watchdog_interval_invalid"),
    ({"WATCHDOG_USEC": "٢"}, "watchdog_interval_invalid"),
    ({"WATCHDOG_PID": "0"}, "watchdog_pid_mismatch"),
    ({"WATCHDOG_PID": "invalid"}, "watchdog_pid_mismatch"),
    ({"NOTIFY_SOCKET": "relative/notify"}, "notify_socket_invalid"),
    ({"NOTIFY_SOCKET": "@"}, "notify_socket_invalid"),
    ({"NOTIFY_SOCKET": "/tmp/notify\nother"}, "notify_socket_invalid"),
])
def test_malformed_service_environment_cannot_send(changes, reason):
    h, _, calls, sockets = make_heartbeat(env=environment(**changes))
    assert not h.enabled and h.disabled_reason == reason
    assert h.notify_due(force=True) is False
    assert calls == sockets == []


def test_main_pid_and_half_interval_bind_the_notification():
    h, clock, calls, sockets = make_heartbeat()
    assert h.enabled and h.interval_seconds == 1
    assert h.notify_due() is True
    assert calls == [(b"WATCHDOG=1", "/run/user/1000/systemd/notify")]
    assert sockets[0].closed
    clock.now = 0.999
    assert h.notify_due() is False and len(calls) == 1
    clock.now = 1
    assert h.notify_due() is True and len(calls) == 2


def test_abstract_unix_address_and_optional_pid_are_supported():
    env = environment(NOTIFY_SOCKET="@supervisor-notify")
    env.pop("WATCHDOG_PID")
    h, _, calls, _ = make_heartbeat(env=env)
    assert h.notify_due() is True
    assert calls == [(b"WATCHDOG=1", "\0supervisor-notify")]


def test_inherited_heartbeat_cannot_notify_from_a_different_process(monkeypatch):
    h, _, calls, _ = make_heartbeat()
    original = os.getpid()
    monkeypatch.setattr(heartbeat.os, "getpid", lambda: original + 1)
    assert h.notify_due(force=True) is False
    assert calls == []


@pytest.mark.parametrize("failure", ["create", "timeout", "send", "short"])
def test_notification_failure_does_not_fake_success_or_crash_loop(failure):
    h, clock, calls, sockets = make_heartbeat(failure=failure)
    assert h.notify_due() is False
    assert all(client.closed for client in sockets)
    before = len(sockets)
    clock.now = 0.5
    assert h.notify_due() is False and len(sockets) == before
    clock.now = 1
    assert h.notify_due() is False


def test_socket_close_error_does_not_fail_a_successful_main_loop_tick():
    h, _, calls, sockets = make_heartbeat(failure="close")
    assert h.notify_due() is True
    assert len(calls) == 1 and sockets[0].closed


class WaitingProcess:
    args = ["bounded-coding-job"]

    def __init__(self, clock):
        self.clock = clock
        self.timeouts = []

    def wait(self, timeout):
        self.timeouts.append(timeout)
        self.clock.now += timeout
        raise subprocess.TimeoutExpired(self.args, timeout)


def test_wait_heartbeats_never_extend_the_absolute_job_budget():
    h, clock, calls, _ = make_heartbeat()
    child = WaitingProcess(clock)
    with pytest.raises(subprocess.TimeoutExpired):
        h.wait_for_process(child, 2.5, poll_seconds=0.25)
    assert clock.now == 2.5
    assert len(child.timeouts) == 10
    assert len(calls) == 3  # At0,1,2; no ping grants another child timeout.


def test_supplied_original_deadline_can_only_shorten_wait():
    h, clock, _, _ = make_heartbeat()
    child = WaitingProcess(clock)
    with pytest.raises(subprocess.TimeoutExpired):
        h.wait_for_process(child, 2400, deadline=1.25, poll_seconds=0.5)
    assert clock.now == 1.25
    assert child.timeouts == [0.5, 0.5, 0.25]


def test_longer_supplied_deadline_cannot_replenish_a_shorter_timeout():
    h, clock, _, _ = make_heartbeat()
    child = WaitingProcess(clock)
    with pytest.raises(subprocess.TimeoutExpired):
        h.wait_for_process(child, 1, deadline=2400, poll_seconds=0.5)
    assert clock.now == 1 and child.timeouts == [0.5, 0.5]


def test_disabled_watchdog_still_honors_a_shorter_original_deadline():
    clock = Clock(10)
    h, _, _, _ = make_heartbeat(clock, env={})
    child = WaitingProcess(clock)
    with pytest.raises(subprocess.TimeoutExpired):
        h.wait_for_process(child, 2400, deadline=11)
    assert child.timeouts == [1] and clock.now == 11


def test_real_child_exit_code_is_preserved_without_a_second_wait():
    h, clock, calls, _ = make_heartbeat()

    class Child:
        def wait(self, timeout):
            clock.now += 0.1
            return 23

    assert h.wait_for_process(Child(), 2400) == 23
    assert clock.now == 0.1 and len(calls) == 1


@pytest.mark.parametrize("value", [True, -1, math.inf, math.nan, "300"])
def test_unbounded_or_malformed_process_timeout_is_refused_before_wait(value):
    h, _, calls, _ = make_heartbeat()

    class Child:
        def wait(self, timeout):
            pytest.fail("invalid timeout reached a process wait")

    with pytest.raises(ValueError, match="timeout"):
        h.wait_for_process(Child(), value)
    assert calls == []


def installer():
    path = Path(__file__).resolve().parents[2] / "scripts/ops/agent_supervisor/install_fleet_watchdog.py"
    spec = importlib.util.spec_from_file_location("_repair_heartbeat_installer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_pairs_main_process_watchdog_with_bundled_heartbeat(tmp_path):
    module = installer()
    assert "systemd_heartbeat" in module.MODULES
    assert "repair_candidate" in module.MODULES
    assert "repair_candidate_runtime" in module.MODULES
    for suffix, name, mode in [("watchdog", "fleet_watchdog", ["--apply"]),
                               ("repair", "fleet_repair", ["run"])]:
        unit = module.generic_service_unit("/usr/bin/python3", tmp_path / "source", tmp_path / "release",
                                           tmp_path / "fleet.json", suffix, name, mode).decode()
        assert "Type=simple\nNotifyAccess=main\nWatchdogSec=300\n" in unit
        assert "Restart=always\nRestartSec=30\n" in unit
        assert "TimeoutStopSec=330\nKillMode=control-group\n" in unit


def test_installer_cannot_prepare_watchdog_release_without_heartbeat_source(tmp_path):
    module = installer()
    source, home = tmp_path / "source", tmp_path / "home"
    directory = source / "ipfs_accelerate_py/agent_supervisor/rescue"
    directory.mkdir(parents=True)
    for name in module.MODULES:
        if name != "systemd_heartbeat":
            (directory / (name + ".py")).write_text("# selected source\n")
    for relative in module.PACKAGE_MODULES.values():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# package source\n")
    with pytest.raises(FileNotFoundError, match="systemd_heartbeat"):
        module._enrollment_runtime(source, home)
    assert not home.exists()
