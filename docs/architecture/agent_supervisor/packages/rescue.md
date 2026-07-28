# agent_supervisor.rescue

## Purpose

Rescue and recovery: planners, orchestrators, diagnostics, watchdog hooks, and recovery paths when lanes stall or fail policy.

## When to use this package

You are improving automatic recovery without expanding authority beyond rescue policies.

## Public modules

| Module | Role |
| --- | --- |
| `rescue_orchestrator` | Rescue orchestration |
| `rescue_planner` | Rescue planning |
| `recovery_diagnostics` | Diagnostics for failed runs |
| `supervisor_recovery` | Supervisor recovery helpers |
| `supervisor_watchdog` | Watchdog hooks |
| `implementation_failure_review` | Review implementation failures |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.rescue import ...
# or
from ipfs_accelerate_py.agent_supervisor.rescue.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Runtime supervisors, operators replaying failures. |
| **Outbound** | `validation`, `merge`, `planning` as needed. |
| **Forbidden** | Using rescue to grant completion without fresh evidence. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.
