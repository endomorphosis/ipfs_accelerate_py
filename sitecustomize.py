"""Load overlay supervisor heals before a sealed board script hides PYTHONPATH."""

from __future__ import annotations

from pathlib import Path

_OVERLAY = Path(__file__).resolve().parent
if (_OVERLAY / "ipfs_accelerate_py" / "agent_supervisor").is_dir():
    from ipfs_accelerate_py.agent_supervisor.rescue.overlay_sys_path import (
        pin_overlay_sys_path,
    )

    pin_overlay_sys_path(str(_OVERLAY))
    import ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server  # noqa: F401
    import ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state  # noqa: F401
    import ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_command  # noqa: F401
