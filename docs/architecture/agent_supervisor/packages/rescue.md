# agent_supervisor.rescue

**Code:** `ipfs_accelerate_py/agent_supervisor/rescue/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/rescue/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


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

## Retained candidates after a dead provider

A native dead-admission quarantine remains a stop until recovery reproduces
its exact canonical admission, process birth, claim tuple, and retained
committed-candidate proof. The original lane must observe the historical
provider process as dead. A second source and liveness check precedes the
owner's receipt-bound retry CAS and cooldown write. The retry carries the
same immutable candidate through the existing fresh validation and merge
path; this transition grants no completion and invokes no provider.

Legacy auxiliary merge events that omit a canonical task key may be ignored
only when the same verified event chain contains an exact validation start,
worktree, enqueue request, candidate, and later fully bound implementation
result. That later result supplies completion evidence through the ordinary
verifier. Auxiliary events never acquire an inferred identity or become
completion receipts, and nonempty mismatched keys remain errors.

A terminal receiver key mismatch can enter only retained callback
reconciliation. It must reproduce the exact original terminal revision and
claim history before and inside the native retry transaction. The handoff
carries the original queue, binding, candidate, and target qualification in
a fenced completion-recovery seed; it does not redispatch provider work or
accept a task solely because files have landed.

Changed or dirty candidate worktrees, unknown or live process identities,
stale revision history, foreign lanes, and missing candidate evidence retain
the quarantine. A clean-baseline/no-effect classification alone is not
admitted by this recovery path.

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.