"""Preflight configured launcher assets before publishing a unit binding.

This checks availability only. Immutable runtime manifests provide integrity;
asset existence never authorizes an owner stop, migration or source transition.
"""
from __future__ import annotations

import stat
from pathlib import Path
from typing import Any


def schema_runtime_assets(
    overlay: str | Path, *, board_id: str, source_root: str | Path | None = None,
) -> dict[str, Any]:
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
    result = {"valid": not missing,
              "reason": "runtime_assets_present" if not missing else "supervisor_runtime_assets_unavailable",
              "missing_assets": missing, "completion_authority": False,
              "owner_restart_authority": False}
    if not missing and board_id == "sawm" and source_root is not None:
        source_missing = _sawm_source_assets(Path(source_root))
        result["missing_source_assets"] = source_missing
        if source_missing:
            result.update(valid=False, reason="supervisor_source_assets_unavailable")
    return result


def _sawm_source_assets(root: Path) -> list[str]:
    """Check the direct native entrypoints used by the selected SAWM launcher.

    The operator binds its two observation modules to files below REPO_ROOT.
    An overlay or nested external checkout cannot substitute for those files.
    This finite availability check does not validate source seals, the full
    transitive import graph, database state, or permission to start an owner.
    """
    if not root.is_absolute() or ".." in root.parts:
        return ["source_root"]
    try:
        if not all(stat.S_ISDIR(path.lstat().st_mode) for path in (root, *root.parents)):
            return ["source_root"]
    except OSError:
        return ["source_root"]
    names = (
        ("scripts/ops/agent_supervisor/semantic_addressed_world_model.py", False),
        ("config/agent_supervisor_semantic_addressed_world_model_scheduler.json", False),
        ("ipfs_accelerate_py/__init__.py", True),
        ("ipfs_accelerate_py/agent_supervisor/__init__.py", True),
        ("ipfs_accelerate_py/agent_supervisor/task_sources/__init__.py", True),
        ("ipfs_accelerate_py/agent_supervisor/runtime/__init__.py", True),
        ("ipfs_accelerate_py/agent_supervisor/runtime/native_dispatch_drain.py", False),
        ("ipfs_accelerate_py/agent_supervisor/runtime/owner_status_observation.py", False),
    )
    missing = []
    for name, empty_allowed in names:
        path = root / name
        try:
            # Check each actual ancestor below the exact configured root;
            # lstat keeps a package-directory symlink from satisfying it.
            ancestors = [path.parent]
            while ancestors[-1] != root:
                ancestors.append(ancestors[-1].parent)
            metadata = path.lstat()
            if (not all(stat.S_ISDIR(parent.lstat().st_mode) for parent in ancestors)
                    or not stat.S_ISREG(metadata.st_mode)
                    or (metadata.st_size == 0 and not empty_allowed)):
                missing.append(name)
        except OSError:
            missing.append(name)
    return missing
