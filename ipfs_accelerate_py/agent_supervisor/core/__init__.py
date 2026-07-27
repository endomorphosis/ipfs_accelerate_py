"""Shared foundation package for agent_supervisor (ASREF core).

This package is the bottom of the agent_supervisor package DAG. Higher
packages (control, task_sources, context, analysis, proof, and layers above
them) may depend on ``core``; ``core`` must not depend on ``todo_daemon``,
``runtime``, ``merge``, ``rescue``, or ``self_improvement``.

Modules owned by bundle ``asref/core`` (see
``docs/architecture/asref/move_map.json``) move into this package via
``git mv`` without long-lived re-export stubs at the former flat paths:

* ``conflict_graph``
* ``external_completion``
* ``program_behavior``
* ``submodule_degradation``
* ``wrapper_utils``

Until those modules land under this directory, import them from their current
flat locations. After each move, callers must use::

    from ipfs_accelerate_py.agent_supervisor.core.<module> import ...

Package metadata and the public module list are intentional surface area so
importers and tooling can discover the core contract without loading optional
supervisor providers.
"""

from __future__ import annotations

from typing import Final

__all__: Final[tuple[str, ...]] = (
    "CORE_PACKAGE_NAME",
    "CORE_OWNED_MODULES",
    "CORE_FORBIDDEN_DEPENDENTS",
)

CORE_PACKAGE_NAME: Final[str] = "ipfs_accelerate_py.agent_supervisor.core"

# Stems owned by asref/core in docs/architecture/asref/move_map.json.
CORE_OWNED_MODULES: Final[tuple[str, ...]] = (
    "conflict_graph",
    "external_completion",
    "program_behavior",
    "submodule_degradation",
    "wrapper_utils",
)

# Packages that must not be imported by core (DAG / cycle guard).
CORE_FORBIDDEN_DEPENDENTS: Final[tuple[str, ...]] = (
    "todo_daemon",
    "runtime",
    "merge",
    "rescue",
    "self_improvement",
    "integrations",
)
