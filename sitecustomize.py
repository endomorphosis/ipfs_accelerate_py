"""Keep this overlay's ipfs_accelerate_py first when it is on PYTHONPATH.

Board extra-gate scripts insert a sealed checkout at sys.path[0], which hides
PYTHONPATH and loads a package without supervisor heals. This hook runs at
interpreter start so one exclusive owner still loads overlay heals without
wrapping ExecStart.
"""

from __future__ import annotations

from pathlib import Path

_OVERLAY = Path(__file__).resolve().parent
if str(_OVERLAY / "ipfs_accelerate_py" / "agent_supervisor").is_dir():
    from ipfs_accelerate_py.agent_supervisor.rescue.overlay_sys_path import (
        pin_overlay_sys_path,
    )

    pin_overlay_sys_path(str(_OVERLAY))
