"""Runtime package for agent_supervisor (ASREF).

Owns multi-supervisor runners, artifact store, event log, CAS, temporal
monitoring, and resource/scheduler surfaces. Higher packages may depend
on ``runtime``; ``runtime`` must not form cycles with ``todo_daemon``,
``self_improvement``, or ``integrations``.

Modules owned by bundle ``asref/runtime`` (see
``docs/architecture/asref/move_map.json``) move into this package via
``git mv`` without long-lived re-export stubs at the former flat paths.

Until those modules land under this directory, import them from their
current flat locations. After each move, callers must use::

    from ipfs_accelerate_py.agent_supervisor.runtime.<module> import ...

Post-move console entry point (pyproject / setup) retargets to::

    ipfs_accelerate_py.agent_supervisor.runtime.artifact_store:main
"""

from __future__ import annotations

from typing import Final

__all__: Final[tuple[str, ...]] = (
    "RUNTIME_PACKAGE_NAME",
    "RUNTIME_OWNED_MODULES",
    "RUNTIME_FORBIDDEN_DEPENDENTS",
    "RUNTIME_ENTRY_POINT_TARGETS",
)

RUNTIME_PACKAGE_NAME: Final[str] = "ipfs_accelerate_py.agent_supervisor.runtime"

# Stems owned by asref/runtime in docs/architecture/asref/move_map.json.
RUNTIME_OWNED_MODULES: Final[tuple[str, ...]] = (
    "artifact_store",
    "event_log",
    "multi_supervisor_runner",
    "provider_batch_scheduler",
    "resource_scheduler",
    "runtime_cas",
    "runtime_temporal_monitor",
    "scheduler_metrics",
)

# Packages that must not be imported by runtime (DAG / cycle guard).
RUNTIME_FORBIDDEN_DEPENDENTS: Final[tuple[str, ...]] = (
    "todo_daemon",
    "self_improvement",
    "integrations",
)

# Intended post-move entry-point module targets (ASREF-G070).
RUNTIME_ENTRY_POINT_TARGETS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.runtime.artifact_store:main",
)
