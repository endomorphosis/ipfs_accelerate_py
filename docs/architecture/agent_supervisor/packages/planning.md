# agent_supervisor.planning

## Purpose

Adaptive and formal planning (non-daemon): plan compile/validate/conformance, adaptive planner/refiner, formal replanner, proof-carrying planner orchestration, and planning metrics.

## When to use this package

You are changing plan IR, conformance rules, or replanning budgets—not the multi-lane process supervisor.

## Public modules

| Module | Role |
| --- | --- |
| `formal_plan_compiler` | Compile formal plans / preconditions |
| `formal_plan_validator` | Validate plans |
| `formal_plan_conformance` | Conformance vs execution events |
| `formal_plan_context` | Plan context assembly |
| `formal_planning_contracts` | Planning contracts |
| `formal_replanner` | Counterexample-guided repair |
| `proof_carrying_planner` | Proof-carrying execution DAG |
| `adaptive_planner` | Adaptive planning |
| `adaptive_goal_refiner` | Goal refinement from evidence |
| `plan_evaluator` | Plan evaluation helpers |
| `formal_planning_metrics` | Planning benchmarks |
| `formal_planning_rollout` | Planning rollout gates |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.planning import ...
# or
from ipfs_accelerate_py.agent_supervisor.planning.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Objectives, control plan operations, proof gates. |
| **Outbound** | `proof`, `context`, `core`. |
| **Forbidden** | Unbounded replanning; treating candidates as completions. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.
