# agent_supervisor.objectives

**Layer:** Mid · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Durable intent lifecycle: objective heap parse, daemons, trackers, backlog refinery, bundles, and goal quality/completion.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, `task_sources` |
| **Typical dependents** | todo_daemon, planning, self_improvement, runtime |

## Modules

| Module | Path |
| --- | --- |
| `adaptive_goal_refiner` | `objectives/adaptive_goal_refiner.py` |
| `backlog_refinery` | `objectives/backlog_refinery.py` |
| `bundle_optimizer` | `objectives/bundle_optimizer.py` |
| `bundle_supervisor` | `objectives/bundle_supervisor.py` |
| `goal_completion` | `objectives/goal_completion.py` |
| `goal_coverage` | `objectives/goal_coverage.py` |
| `goal_development_contracts` | `objectives/goal_development_contracts.py` |
| `goal_quality` | `objectives/goal_quality.py` |
| `goal_refinement_verification` | `objectives/goal_refinement_verification.py` |
| `objective_daemon` | `objectives/objective_daemon.py` |
| `objective_graph` | `objectives/objective_graph.py` |
| `objective_task_janitor` | `objectives/objective_task_janitor.py` |
| `objective_tracker` | `objectives/objective_tracker.py` |
| `scan_receipts` | `objectives/scan_receipts.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.objectives.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/objectives.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
