"""Keep this overlay's ipfs_accelerate_py first when it is on PYTHONPATH.

Board extra-gate scripts insert a sealed checkout at sys.path[0], which hides
PYTHONPATH and loads a package without supervisor heals. This hook runs at
interpreter start so one exclusive owner still loads overlay heals without
wrapping ExecStart.
"""

from __future__ import annotations

from pathlib import Path

_OVERLAY = Path(__file__).resolve().parent
if (_OVERLAY / "ipfs_accelerate_py" / "agent_supervisor").is_dir():
    from ipfs_accelerate_py.agent_supervisor.rescue.overlay_sys_path import (
        pin_overlay_sys_path,
    )

    pin_overlay_sys_path(str(_OVERLAY))
    # Preload overlay extra-gate heals before a board script inserts a sealed
    # checkout. Token republish and owner-side unstall then run in this process.
    import ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server  # noqa: F401
    import ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_command  # noqa: F401
    import ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state  # noqa: F401
