"""Bind extended coding budgets to the installer and router in a runtime archive.

This is an archive compatibility check, not execution or completion authority.
Legacy archives remain usable with the original at-most-300-second profiles.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .benchmark_resource_profile import coding_timeout_seconds

KEY = "router_worker_capability"
SCHEMA = "terminal-router-worker-capability@1"
LEGACY_MAX_TIMEOUT_SECONDS = 300
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
)


def _current_capability():
    from .container_worker_deployment import WORKER_ENTRY
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
        MAX_IMPLEMENTATION_TIMEOUT_SECONDS,
    )

    root = Path(__file__).resolve().parents[3]
    return {
        "schema": SCHEMA,
        "max_timeout_seconds": MAX_IMPLEMENTATION_TIMEOUT_SECONDS,
        "worker_entry_sha256": hashlib.sha256(WORKER_ENTRY.encode()).hexdigest(),
        "source_sha256": {
            "source/" + name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS
        },
    }


def _matching_inventory(inventory, capability):
    if type(inventory) is not list:
        return False
    for name, digest in capability["source_sha256"].items():
        rows = [row for row in inventory if type(row) is dict and row.get("path") == name]
        if len(rows) != 1 or rows[0].get("sha256") != digest:
            return False
    return True


def worker_capability_for_inventory(inventory):
    """Advertise only the exact executing installer's reviewed source bytes.

    Partial fixtures and archives of other source versions get no new capability.
    They can still serve legacy budgets, but cannot silently claim the extension.
    """
    expected_names = {"source/" + name for name in SOURCE_PATHS}
    if type(inventory) is not list or not expected_names.issubset(
        row.get("path") for row in inventory if type(row) is dict
    ):
        return None
    capability = _current_capability()
    return capability if _matching_inventory(inventory, capability) else None


def require_worker_capability(manifest, resource_profile):
    """Reject an unsupported extended budget before planning or deployment."""
    selected = coding_timeout_seconds(resource_profile)
    if selected <= LEGACY_MAX_TIMEOUT_SECONDS:
        return None
    if type(manifest) is not dict:
        raise ValueError("extended coding budget requires a worker archive capability")
    expected = _current_capability()
    capability = manifest.get(KEY)
    if (type(capability) is not dict or capability != expected
            or type(capability.get("max_timeout_seconds")) is not int
            or selected > capability["max_timeout_seconds"]
            or not _matching_inventory(manifest.get("files"), expected)):
        raise ValueError("extended coding budget requires the matching worker archive capability")
    return dict(capability)
