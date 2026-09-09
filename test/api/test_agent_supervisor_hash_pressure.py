"""Hash worker counts shrink under CPU, memory, and swap pressure."""

from __future__ import annotations

import pytest
from ipfs_accelerate_py import _hash_resources as resources
from ipfs_accelerate_py.agent_supervisor.runtime import hash_pressure


@pytest.mark.parametrize(
    "load,available_kb,swap_free_kb,reason",
    [
        (32.0, 16777216, 1048576, "host_cpu_load"),
        (0.1, 1000000, 1048576, "memory_headroom"),
        (0.1, 16777216, 16, "host_swap_exhaustion"),
        (0.2, 16777216, 1048576, "admitted"),
    ],
)
def test_hash_worker_limit_tracks_host_pressure(
    monkeypatch, load, available_kb, swap_free_kb, reason
) -> None:
    def read_proc(path):
        path = str(path)
        if path == "/proc/loadavg":
            return f"{load} 0.1 0.1 1/1 1\n"
        if path == "/proc/meminfo":
            return (
                "MemTotal: 33554432 kB\n"
                f"MemAvailable: {available_kb} kB\n"
                "SwapTotal: 1048576 kB\n"
                f"SwapFree: {swap_free_kb} kB\n"
            )
        if path.startswith("/proc/pressure/"):
            return "some avg10=0 avg60=0 avg300=0 total=0\n"
        raise AssertionError(f"unexpected host measurement: {path}")

    # The public compatibility API delegates to the shared resource module.
    # Supply host measurements there, with no cgroup quota or PSI pressure.
    monkeypatch.setattr(resources, "_read_proc", read_proc)
    monkeypatch.setattr(resources, "_cgroup_directories", lambda: ())
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "4")
    assert hash_pressure.host_hash_pressure is resources.host_hash_pressure
    workers, actual_reason = hash_pressure.host_hash_pressure()
    assert actual_reason == reason
    if reason == "admitted":
        assert workers == hash_pressure.MAX_WORKERS_CAP
    else:
        assert workers == 1
    assert hash_pressure.hash_worker_limit(64) == workers
    assert hash_pressure.hash_worker_limit(1) == 1
