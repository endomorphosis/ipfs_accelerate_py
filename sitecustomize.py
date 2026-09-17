"""Keep overlay imports first when a sealed checkout inserts itself.

Do not wrap exclusive-owner ExecStart. Pinning sys.path only restores the
fleet overlay ahead of nested extra-gate inserts so leftover-attempt fairness
and missing-cooldown skips load in managed daemons.
"""

try:
    from ipfs_accelerate_py.agent_supervisor.rescue.overlay_sys_path import (
        pin_overlay_sys_path,
    )

    pin_overlay_sys_path()
except Exception:
    pass
