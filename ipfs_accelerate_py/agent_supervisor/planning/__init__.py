"""Planning package for agent_supervisor (ASREF).

Owns adaptive / formal planners, plan evaluators, and task proposal
routing. Higher packages may depend on ``planning``; ``planning`` must
not form cycles with ``todo_daemon``, ``runtime``, ``merge``,
``rescue``, or ``self_improvement``.

Modules owned by bundle ``asref/planning`` (see
``docs/architecture/asref/move_map.json``) move into this package via
``git mv`` without long-lived re-export stubs at the former flat paths.

Until those modules land under this directory, import them from their
current flat locations. After each move, callers must use::

    from ipfs_accelerate_py.agent_supervisor.planning.<module> import ...
"""

from __future__ import annotations

from typing import Final

__all__: Final[tuple[str, ...]] = (
    "PLANNING_PACKAGE_NAME",
    "PLANNING_OWNED_MODULES",
    "PLANNING_FORBIDDEN_DEPENDENTS",
)

PLANNING_PACKAGE_NAME: Final[str] = "ipfs_accelerate_py.agent_supervisor.planning"

# Stems owned by asref/planning in docs/architecture/asref/move_map.json.
PLANNING_OWNED_MODULES: Final[tuple[str, ...]] = (
    "adaptive_planner",
    "formal_plan_compiler",
    "formal_plan_conformance",
    "formal_plan_context",
    "formal_plan_validator",
    "formal_planning_adversarial",
    "formal_planning_contracts",
    "formal_planning_metrics",
    "formal_planning_rollout",
    "formal_replanner",
    "plan_evaluator",
    "plan_failure_memory",
    "proof_carrying_planner",
    "task_proposal_router",
    "task_quality",
)

# Packages that must not be imported by planning (DAG / cycle guard).
PLANNING_FORBIDDEN_DEPENDENTS: Final[tuple[str, ...]] = (
    "todo_daemon",
    "runtime",
    "merge",
    "rescue",
    "self_improvement",
    "integrations",
)
