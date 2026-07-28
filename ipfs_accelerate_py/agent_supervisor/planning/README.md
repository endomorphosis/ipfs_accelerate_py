# agent_supervisor.planning

**Layer:** Mid · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Adaptive and formal planning: compile, validate, replan, metrics, and rollout helpers (non-daemon).

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, `objectives` contracts, proof contracts as needed |
| **Typical dependents** | todo_daemon, validation, proof-carrying flows |

## Modules

| Module | Path |
| --- | --- |
| `adaptive_planner` | `planning/adaptive_planner.py` |
| `formal_plan_compiler` | `planning/formal_plan_compiler.py` |
| `formal_plan_conformance` | `planning/formal_plan_conformance.py` |
| `formal_plan_context` | `planning/formal_plan_context.py` |
| `formal_plan_validator` | `planning/formal_plan_validator.py` |
| `formal_planning_adversarial` | `planning/formal_planning_adversarial.py` |
| `formal_planning_contracts` | `planning/formal_planning_contracts.py` |
| `formal_planning_metrics` | `planning/formal_planning_metrics.py` |
| `formal_planning_rollout` | `planning/formal_planning_rollout.py` |
| `formal_replanner` | `planning/formal_replanner.py` |
| `plan_evaluator` | `planning/plan_evaluator.py` |
| `plan_failure_memory` | `planning/plan_failure_memory.py` |
| `proof_carrying_planner` | `planning/proof_carrying_planner.py` |
| `task_proposal_router` | `planning/task_proposal_router.py` |
| `task_quality` | `planning/task_quality.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.planning.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/planning.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
