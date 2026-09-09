"""Hash worker counts must shrink under live CPU and memory pressure."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime import hash_pressure


def test_hash_worker_limit_collapses_on_cpu_load(monkeypatch) -> None:
    monkeypatch.setattr(
        hash_pressure,
        "_read_proc",
        lambda path: (
            "32.0 1.0 1.0 1/1 1\n"
            if path.endswith("loadavg")
            else (
                "MemTotal:        33554432 kB\n"
                "MemAvailable:    16777216 kB\n"
                "SwapTotal:        1048576 kB\n"
                "SwapFree:         1048576 kB\n"
            )
        ),
    )
    monkeypatch.setattr(hash_pressure.os, "cpu_count", lambda: 8)
    workers, reason = hash_pressure.host_hash_pressure()
    assert workers == 1
    assert reason == "host_cpu_load"
    assert hash_pressure.hash_worker_limit(16) == 1


def test_hash_worker_limit_collapses_on_memory_headroom(monkeypatch) -> None:
    monkeypatch.setattr(
        hash_pressure,
        "_read_proc",
        lambda path: (
            "0.1 0.1 0.1 1/1 1\n"
            if path.endswith("loadavg")
            else (
                "MemTotal:        16777216 kB\n"
                "MemAvailable:     1000000 kB\n"
                "SwapTotal:        1048576 kB\n"
                "SwapFree:         1048576 kB\n"
            )
        ),
    )
    monkeypatch.setattr(hash_pressure.os, "cpu_count", lambda: 8)
    workers, reason = hash_pressure.host_hash_pressure()
    assert workers == 1
    assert reason == "host_memory_headroom"


def test_hash_worker_limit_collapses_on_swap_exhaustion(monkeypatch) -> None:
    monkeypatch.setattr(
        hash_pressure,
        "_read_proc",
        lambda path: (
            "0.1 0.1 0.1 1/1 1\n"
            if path.endswith("loadavg")
            else (
                "MemTotal:        33554432 kB\n"
                "MemAvailable:    16777216 kB\n"
                "SwapTotal:       15728640 kB\n"
                "SwapFree:              16 kB\n"
            )
        ),
    )
    monkeypatch.setattr(hash_pressure.os, "cpu_count", lambda: 8)
    workers, reason = hash_pressure.host_hash_pressure()
    assert workers == 1
    assert reason == "host_swap_exhaustion"


def test_hash_worker_limit_caps_idle_host(monkeypatch) -> None:
    monkeypatch.setattr(
        hash_pressure,
        "_read_proc",
        lambda path: (
            "0.2 0.2 0.2 1/1 1\n"
            if path.endswith("loadavg")
            else (
                "MemTotal:        33554432 kB\n"
                "MemAvailable:    16777216 kB\n"
                "SwapTotal:        1048576 kB\n"
                "SwapFree:         1048576 kB\n"
            )
        ),
    )
    monkeypatch.setattr(hash_pressure.os, "cpu_count", lambda: 8)
    workers, reason = hash_pressure.host_hash_pressure()
    assert reason == "admitted"
    assert 1 <= workers <= hash_pressure.MAX_WORKERS_CAP
    assert hash_pressure.hash_worker_limit(64) == workers
    assert hash_pressure.hash_worker_limit(1) == 1
