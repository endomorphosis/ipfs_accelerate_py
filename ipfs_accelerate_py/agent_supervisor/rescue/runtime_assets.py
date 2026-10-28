"""Preflight configured launcher assets before publishing a unit binding.

This checks availability only. Immutable runtime manifests provide integrity;
asset existence never authorizes an owner stop, migration or source transition.
"""
from __future__ import annotations

import stat
from pathlib import Path
from typing import Any


def schema_runtime_assets(overlay: str | Path, *, board_id: str) -> dict[str, Any]:
    root = Path(overlay)
    names = [
        "ipfs_accelerate_py/agent_supervisor/rescue/sealed_board_supervisor_launch.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py",
    ]
    if board_id == "sawm":
        names.extend([
            "ipfs_accelerate_py/agent_supervisor/rescue/sawm_supervise_with_heal_overlay.sh",
            "ipfs_accelerate_py/agent_supervisor/rescue/live_owner_ready.py",
        ])
    missing = []
    for name in names:
        try:
            metadata = (root / name).lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    return {"valid": not missing,
            "reason": "runtime_assets_present" if not missing else "supervisor_runtime_assets_unavailable",
            "missing_assets": missing, "completion_authority": False,
            "owner_restart_authority": False}
