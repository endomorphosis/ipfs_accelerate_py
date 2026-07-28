# agent_supervisor.validation

**Code:** `ipfs_accelerate_py/agent_supervisor/validation/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/validation/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


## Purpose

Proposal validation and validation runtime: command selection, schedulers, scope adjudication hooks, and pre-merge validation policy.

## When to use this package

You are changing what must pass before a patch is accepted or merged.

## Public modules

| Module | Role |
| --- | --- |
| `proposal_validation` | Proposal validation policy |
| `validation_commands` | Validation command selection |
| `validation_scheduler` | Schedule validation work |
| `validation_runtime` | Run validation commands |
| `scope_adjudication` | Scope adjudication |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.validation import ...
# or
from ipfs_accelerate_py.agent_supervisor.validation.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Implementation daemon, merge path, control validation-replay. |
| **Outbound** | `core`, repo tooling; not optional model providers for policy decisions. |
| **Forbidden** | Accepting model prose as a substitute for validation commands. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.