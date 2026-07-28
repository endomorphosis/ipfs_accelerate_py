# agent_supervisor.objectives

**Code:** `ipfs_accelerate_py/agent_supervisor/objectives/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/objectives/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


## Purpose

Objective heap lifecycle: parsing, tracking, daemon/CLI bridge, janitor, goal completion/coverage/quality, backlog refinery, and objective graph scanning.

## When to use this package

You are changing how goals are refined, scanned for evidence, or projected into tasks.

## Public modules

| Module | Role |
| --- | --- |
| `objective_tracker` | Durable objective heap |
| `objective_graph` | Goal graph parse/scan/proposals |
| `objective_daemon` | CLI/runtime bridge for scans |
| `objective_task_janitor` | Task janitor policies |
| `goal_completion` | Goal lifecycle authority |
| `goal_coverage` | Coverage projections |
| `goal_quality` | Goal quality metrics |
| `backlog_refinery` | Bounded refill / repair work |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.objectives import ...
# or
from ipfs_accelerate_py.agent_supervisor.objectives.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Operators, control objective operations, runtime scanners. |
| **Outbound** | `task_sources`, `analysis`, `core`. |
| **Forbidden** | Mutating protected objective heaps without operator policy. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.