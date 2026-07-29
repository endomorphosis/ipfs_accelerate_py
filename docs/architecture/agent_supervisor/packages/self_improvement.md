# agent_supervisor.self_improvement

**Code:** `ipfs_accelerate_py/agent_supervisor/self_improvement/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/self_improvement/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


## Purpose

Bounded self-improvement epochs: completion evaluation, successor refill, v2 efficiency/token/state models, and rollout helpers for the self-improvement program.

## When to use this package

You are changing epoch gates, refill materialization, or supervisor efficiency accounting—not general task implementation.

## Public modules

| Module | Role |
| --- | --- |
| `self_improvement` | Core self-improvement API |
| `self_improvement_completion` | Epoch completion evaluation |
| `self_improvement_rollout` | Rollout helpers |
| `self_improvement_v2` | v2 contracts |
| `self_improvement_v2_rollout` | v2 rollout |
| `supervisor_efficiency_metrics` | Efficiency metrics |
| `supervisor_token_ledger` | Token ledger |
| `supervisor_state_model` | Supervisor state model |
| `supervisor_v2_contracts` | v2 contracts surface |
| `supervisor_v2_benchmark` | v2 benchmarks |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.self_improvement import ...
# or
from ipfs_accelerate_py.agent_supervisor.self_improvement.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Self-improvement program operators and control profiles that enable refill. |
| **Outbound** | `objectives`, `task_sources`, `runtime`. |
| **Forbidden** | Unbounded automatic self-modification without rollout mode gates. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.