"""Rescue package for agent_supervisor (ASREF).

Owns rescue orchestration/planning, recovery diagnostics, codex failure
policy, and supervisor recovery/watchdog surfaces. Higher packages may
depend on ``rescue``; ``rescue`` must not form cycles with
``todo_daemon``, ``self_improvement``, or ``integrations``.

Modules owned by bundle ``asref/rescue`` (see
``docs/architecture/asref/move_map.json``) move into this package via
``git mv`` without long-lived re-export stubs at the former flat paths.

First-batch modules are dual-copied under this package (flat originals
remain until ASREF-G090 cutover). Prefer package imports for landed
modules; remaining owned stems still live at flat paths until child
batches land. Import landed modules via::

    from ipfs_accelerate_py.agent_supervisor.rescue.<module> import ...
"""

from __future__ import annotations

from typing import Final

__all__: Final[tuple[str, ...]] = (
    "RESCUE_LANDED_MODULES",
    "RESCUE_PACKAGE_NAME",
    "RESCUE_OWNED_MODULES",
    "RESCUE_FORBIDDEN_DEPENDENTS",
)

RESCUE_PACKAGE_NAME: Final[str] = "ipfs_accelerate_py.agent_supervisor.rescue"

# Stems owned by asref/rescue in docs/architecture/asref/move_map.json.
RESCUE_OWNED_MODULES: Final[tuple[str, ...]] = (
    "codex_failure_policy",
    "recovery_diagnostics",
    "rescue_orchestrator",
    "rescue_planner",
    "supervisor_recovery",
    "supervisor_watchdog",
)

# Dual-copied under this package in the current ASREF-011 batch.
RESCUE_LANDED_MODULES: Final[tuple[str, ...]] = (
    "rescue_orchestrator",
    "codex_failure_policy",
)

# Packages that must not be imported by rescue (DAG / cycle guard).
RESCUE_FORBIDDEN_DEPENDENTS: Final[tuple[str, ...]] = (
    "todo_daemon",
    "self_improvement",
    "integrations",
)
