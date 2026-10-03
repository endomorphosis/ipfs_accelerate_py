"""Opt-in Linux per-process limits for an already admitted pipeline phase.

The existing datasets runner owns prlimit and process-group cleanup. These
limits do not create another host reservation or provide aggregate cgroup,
device, PID or filesystem enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys

MIB = 1024 * 1024
SCHEMA = "repository-linux-process-limits@1"


@dataclass(frozen=True)
class RepositoryProcessLimits:
    """Exact CPU seconds and virtual address space, applied soft=hard.

    CPU time is per process, including its threads. Forked children inherit
    the ceiling but have independent CPU counters. Address space is virtual
    memory per process; aggregate resident memory remains sampled.
    """

    cpu_seconds: int
    address_space_bytes: int

    def __post_init__(self):
        if type(self.cpu_seconds) is not int or not 1 <= self.cpu_seconds <= 3600:
            raise ValueError("CPU limit must be an exact integer in 1..3600 seconds")
        if (type(self.address_space_bytes) is not int
                or not 16 * MIB <= self.address_space_bytes <= 2**40):
            raise ValueError("address-space limit must be an exact integer in 16MiB..1TiB")

    def for_phase(self, *, memory_mb, threads_per_process, timeout_seconds, disk_bytes):
        if not sys.platform.startswith("linux"):
            raise ValueError("explicit hard process limits require Linux prlimit")
        if (type(memory_mb) is not int or memory_mb <= 0
                or type(threads_per_process) is not int or threads_per_process <= 0
                or type(disk_bytes) is not int or disk_bytes <= 0
                or type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("exact bounded native phase dimensions required")
        if self.address_space_bytes > memory_mb * MIB:
            raise ValueError("per-process address-space limit exceeds phase RAM declaration")
        if self.cpu_seconds > max(1, math.ceil(timeout_seconds * threads_per_process)):
            raise ValueError("per-process CPU limit exceeds phase thread-time declaration")
        return {
            "schema": SCHEMA,
            "backend": "existing_datasets_linux_prlimit",
            "kernel_per_process": {
                "cpu_seconds": self.cpu_seconds,
                "address_space_bytes": self.address_space_bytes,
                "file_size_bytes": disk_bytes,
                "core_bytes": 0,
                "soft_equals_hard": True,
            },
            "sampled_process_tree_rss_bytes": memory_mb * MIB,
            "aggregate_kernel_enforcement": False,
            "kernel_process_count_enforcement": False,
            "filesystem_quota": False,
            "device_enforcement": False,
            "host_resource_authority": False,
            "execution_authority": False,
            "completion_authority": False,
            "limitations": [
                "limits apply to each process, not the sum of descendant allocations or CPU time",
                "file-size limit applies to each file, not the sum of files or a filesystem quota",
                "aggregate RSS, process count and named-root disk usage remain sampled",
                "thread environment is cooperative; this is not a sandbox for privileged code",
            ],
        }


__all__ = ["RepositoryProcessLimits", "SCHEMA"]
