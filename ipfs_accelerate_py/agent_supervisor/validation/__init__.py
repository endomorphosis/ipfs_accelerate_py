"""Validation package for agent_supervisor (ASREF).

Owns proposal validation, scope adjudication, and validation runtime /
scheduler / command surfaces. Higher packages may depend on
``validation``; ``validation`` must not form cycles with
``todo_daemon``, ``runtime``, ``merge``, ``rescue``, or
``self_improvement``.

Modules owned by bundle ``asref/validation`` (see
``docs/architecture/asref/move_map.json``) move into this package via
``git mv`` without long-lived re-export stubs at the former flat paths.

Until those modules land under this directory, import them from their
current flat locations. After each move, callers must use::

    from ipfs_accelerate_py.agent_supervisor.validation.<module> import ...
"""

from __future__ import annotations

from typing import Final

__all__: Final[tuple[str, ...]] = (
    "VALIDATION_PACKAGE_NAME",
    "VALIDATION_OWNED_MODULES",
    "VALIDATION_FORBIDDEN_DEPENDENTS",
)

VALIDATION_PACKAGE_NAME: Final[str] = "ipfs_accelerate_py.agent_supervisor.validation"

# Stems owned by asref/validation in docs/architecture/asref/move_map.json.
VALIDATION_OWNED_MODULES: Final[tuple[str, ...]] = (
    "proposal_validation",
    "scope_adjudication",
    "validation_commands",
    "validation_runtime",
    "validation_scheduler",
)

# Packages that must not be imported by validation (DAG / cycle guard).
VALIDATION_FORBIDDEN_DEPENDENTS: Final[tuple[str, ...]] = (
    "todo_daemon",
    "runtime",
    "merge",
    "rescue",
    "self_improvement",
    "integrations",
)
