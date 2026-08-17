#!/usr/bin/env python3
"""Append LGSWF integer reservation helpers onto the existing resource scheduler."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MARKER = "def lgswf_reserve_vector"
EXTENSION = '''
LGSWF_RESOURCE_VECTOR = (
    "cpu_ms",
    "cpu_concurrency",
    "ram_mib",
    "gpu_memory_mib",
    "disk_mib",
    "network",
    "subprocesses",
    "worktrees",
    "model_input_tokens",
    "model_output_tokens",
    "provider_quota_units",
    "provider_concurrency",
)


class LgswfReservationError(ValueError):
    """Leased reservation was refused."""


def lgswf_reserve_vector(available, demand, *, owner):
    reserved = {}
    for key in LGSWF_RESOURCE_VECTOR:
        need = int(demand.get(key, 0) or 0)
        have = int(available.get(key, 0) or 0)
        if need < 0 or have < 0:
            raise LgswfReservationError(f"{key} must be non-negative")
        if need > have:
            raise LgswfReservationError(f"insufficient {key}")
        reserved[key] = need
    return {
        "owner": owner,
        "reserved": reserved,
        "leased": True,
        "released": False,
    }


def lgswf_release_vector(lease):
    if not lease.get("leased"):
        raise LgswfReservationError("lease is not active")
    released = dict(lease)
    released["leased"] = False
    released["released"] = True
    return released
'''
EXPORTS = (
    '    "LGSWF_RESOURCE_VECTOR",\n'
    '    "LgswfReservationError",\n'
    '    "lgswf_release_vector",\n'
    '    "lgswf_reserve_vector",\n'
)


def apply(dest: Path) -> dict[str, object]:
    src_root = Path(__file__).resolve().parents[1]
    module = dest / "ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py"
    test_dst = dest / "test/api/test_agent_supervisor_lgswf_resource_scheduler.py"
    text = module.read_text(encoding="utf-8")
    if MARKER not in text:
        text = text.replace("\n__all__ = [\n", "\n" + EXTENSION + "\n__all__ = [\n", 1)
        text = text.replace(
            '    "sample_host_resources",\n]',
            '    "sample_host_resources",\n' + EXPORTS + "]",
            1,
        )
        module.write_text(text, encoding="utf-8")
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.write_text(
        '''from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    lgswf_release_vector,
    lgswf_reserve_vector,
)

def test_integer_vector_lease_and_release():
    lease = lgswf_reserve_vector({"cpu_ms": 10, "ram_mib": 8}, {"cpu_ms": 4, "ram_mib": 2}, owner="T1")
    assert lease["leased"] is True
    assert lease["reserved"]["cpu_ms"] == 4
    released = lgswf_release_vector(lease)
    assert released["released"] is True
''',
        encoding="utf-8",
    )
    outputs = [
        "ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py",
        "test/api/test_agent_supervisor_lgswf_resource_scheduler.py",
    ]
    add = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *outputs],
        cwd=dest,
        text=True,
        capture_output=True,
        check=False,
    )
    return {"applied": MARKER in module.read_text(encoding="utf-8"), "staged": add.returncode == 0}


if __name__ == "__main__":
    print(json.dumps(apply(Path.cwd()), indent=2, sort_keys=True))
