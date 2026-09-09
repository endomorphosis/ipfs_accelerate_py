"""Supervisor compatibility API for cross-process hashing admission."""

from ipfs_accelerate_py._hash_resources import (
    BYTES_PER_WORKER,
    CPU_LOAD_LIMIT,
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    DEFAULT_MAX_WORKERS,
    MAX_WORKERS_CAP,
    MIN_AVAILABLE_BYTES,
    HashingResourceTimeout,
    HashingResourceUpgradeError,
    hash_lock_path,
    hash_worker_limit,
    hashing_lock,
    hashing_worker_slot,
    host_hash_pressure,
)

# Retain legacy public constants for callers importing the old helper API.
# Admission now uses actual memory headroom, cgroup limits, and live PSI.
MEMORY_PERCENT_LIMIT = 80
SWAP_FLOOR_BYTES = 256 * 1024 * 1024

__all__ = [
    "BYTES_PER_WORKER", "CPU_LOAD_LIMIT", "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_MAX_WORKERS", "MAX_WORKERS_CAP", "MIN_AVAILABLE_BYTES",
    "MEMORY_PERCENT_LIMIT", "SWAP_FLOOR_BYTES", "HashingResourceTimeout",
    "HashingResourceUpgradeError", "hash_lock_path", "hash_worker_limit",
    "hashing_lock", "hashing_worker_slot", "host_hash_pressure",
]
