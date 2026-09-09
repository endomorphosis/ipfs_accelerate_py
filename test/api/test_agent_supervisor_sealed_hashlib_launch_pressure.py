"""Host-pressure gate for sealed hashlib capsule births."""

from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)


def test_command_detects_sealed_hashlib_bootstrap() -> None:
    command = [
        "/usr/bin/python3.12",
        "-I",
        "-c",
        runner.SEALED_CONTROL_PLANE_BOOTSTRAP,
        "8",
    ]
    isolated = [
        "/usr/bin/python3.12",
        "-I",
        "-S",
        "-B",
        "-c",
        runner.SEALED_CONTROL_PLANE_BOOTSTRAP,
        "8",
    ]
    assert runner.command_is_sealed_hashlib_bootstrap(command) is True
    assert runner.command_is_sealed_hashlib_bootstrap(isolated) is True
    assert runner.command_is_sealed_hashlib_bootstrap(["python3", "-m", "x"]) is False
    assert runner.command_is_sealed_hashlib_bootstrap(
        ["/usr/bin/python3.12", "-c", runner.SEALED_CONTROL_PLANE_BOOTSTRAP]
    ) is False


def test_bootstrap_serializes_hashlib_behind_exclusive_lock() -> None:
    source = runner.SEALED_CONTROL_PLANE_BOOTSTRAP
    assert "IPFS_ACCELERATE_SEALED_HASHLIB_LOCK" in source
    assert "fcntl.LOCK_EX" in source
    assert source.index("fcntl.LOCK_EX") < source.index("hashlib.sha256()")
    assert source.index("runpy.run_module") < source.index("fcntl.LOCK_UN")


@pytest.mark.parametrize(
    ("loadavg", "meminfo", "reason"),
    (
        (
            "40.0 1.0 1.0 1/1 1\n",
            "MemTotal:        10485760 kB\nMemAvailable:     5242880 kB\n",
            "host_cpu_load",
        ),
        (
            "0.1 0.1 0.1 1/1 1\n",
            "MemTotal:        15728640 kB\nMemAvailable:     1800000 kB\n",
            "host_memory_headroom",
        ),
        (
            "0.1 0.1 0.1 1/1 1\n",
            "MemTotal:        33554432 kB\nMemAvailable:    16777216 kB\n"
            "SwapTotal:        1048576 kB\nSwapFree:              16 kB\n",
            "host_swap_exhaustion",
        ),
        (
            "0.1 0.1 0.1 1/1 1\n",
            "MemTotal:        33554432 kB\nMemAvailable:     4194304 kB\n"
            "SwapTotal:        1048576 kB\nSwapFree:         1048576 kB\n",
            "host_memory_percent",
        ),
    ),
)
def test_sealed_hashlib_pressure_refuses_saturated_host(
    monkeypatch: pytest.MonkeyPatch,
    loadavg: str,
    meminfo: str,
    reason: str,
) -> None:
    monkeypatch.setattr(runner, "_read_proc_loadavg", lambda: loadavg)
    monkeypatch.setattr(runner, "_read_proc_meminfo", lambda: meminfo)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 8)
    admitted, observed = runner.sealed_hashlib_host_pressure()
    assert admitted is False
    assert observed == reason


def test_sealed_hashlib_pressure_admits_idle_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_read_proc_loadavg",
        lambda: "0.2 0.2 0.2 1/1 1\n",
    )
    monkeypatch.setattr(
        runner,
        "_read_proc_meminfo",
        lambda: (
            "MemTotal:        33554432 kB\n"
            "MemAvailable:    16777216 kB\n"
            "SwapTotal:        1048576 kB\n"
            "SwapFree:         1048576 kB\n"
        ),
    )
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 8)
    admitted, observed = runner.sealed_hashlib_host_pressure()
    assert admitted is True
    assert observed == "admitted"


def test_wait_fail_closes_when_pressure_does_not_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "sealed_hashlib_host_pressure",
        lambda: (False, "host_cpu_load"),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    monotonic = iter((0.0, 0.0, 120.0))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(runner.SealedHashlibHostPressureError, match="host_cpu_load"):
        runner.wait_for_sealed_hashlib_launch_headroom(timeout_seconds=1.0)


def test_apply_controls_skips_non_hashlib_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []
    monkeypatch.setattr(
        runner,
        "wait_for_sealed_hashlib_launch_headroom",
        lambda **_kwargs: called.append(True) or "admitted",
    )
    env: dict[str, str] = {}
    runner.apply_sealed_hashlib_launch_controls(
        ["python3", "-m", "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner"],
        env,
    )
    assert called == []
    assert runner.SEALED_HASHLIB_LOCK_ENV not in env


def test_apply_controls_waits_and_sets_lock_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []
    monkeypatch.setattr(
        runner,
        "wait_for_sealed_hashlib_launch_headroom",
        lambda **_kwargs: called.append(True) or "admitted",
    )
    env: dict[str, str] = {}
    command = [
        "/usr/bin/python3.12",
        "-I",
        "-c",
        runner.SEALED_CONTROL_PLANE_BOOTSTRAP,
        "8",
    ]
    runner.apply_sealed_hashlib_launch_controls(command, env)
    assert called == [True]
    assert env[runner.SEALED_HASHLIB_LOCK_ENV] == runner.sealed_hashlib_lock_path()
    called.clear()
    env.clear()
    isolated = [
        "/usr/bin/python3.12",
        "-I",
        "-S",
        "-B",
        "-c",
        runner.SEALED_CONTROL_PLANE_BOOTSTRAP,
        "8",
    ]
    runner.apply_sealed_hashlib_launch_controls(isolated, env)
    assert called == [True]
    assert env[runner.SEALED_HASHLIB_LOCK_ENV] == runner.sealed_hashlib_lock_path()
