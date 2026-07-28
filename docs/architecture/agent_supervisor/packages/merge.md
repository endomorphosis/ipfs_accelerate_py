# agent_supervisor.merge

## Purpose

Merge train, queue, checkpoints, conflict repair, checkout locks, leases, and git hygiene used by multi-lane work.

## When to use this package

You are changing how implementation branches land on the merge target or how conflicts are repaired.

## Public modules

| Module | Role |
| --- | --- |
| `merge_train` | Merge train orchestration |
| `merge_queue` | Merge queue |
| `merge_checkpoint` | Merge checkpoints |
| `merge_resolver` | Merge resolution |
| `merge_conflict_repair` | Conflict repair |
| `checkout_lock` | Checkout locking |
| `lease_coordination` | Leases across lanes |
| `leased_lane` | Leased lane helpers |
| `git_gc` | Git hygiene |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.merge import ...
# or
from ipfs_accelerate_py.agent_supervisor.merge.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Runtime multi-supervisor, implementation supervisors. |
| **Outbound** | `core`, git; LLM merge fallback only via integrations. |
| **Forbidden** | Force-pushing protected history; skipping leases under contention. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.
