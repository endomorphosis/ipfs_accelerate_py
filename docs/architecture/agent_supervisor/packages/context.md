# agent_supervisor.context

**Code:** `ipfs_accelerate_py/agent_supervisor/context/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/context/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


## Purpose

Context compilation and decision runtime: obligation-first capsules, decision contracts/context, IR constraint compilation hooks, and runtime CAS for decision artifacts.

## When to use this package

You are shaping what an agent is allowed to see (capsules, proof_delta retries) or how decision contexts are assembled.

## Public modules

| Module | Role |
| --- | --- |
| `context_compiler` | Compile context capsules |
| `context_contracts` | Context envelope contracts |
| `decision_context` | Decision core fields |
| `decision_contracts` | Decision request/result contracts |
| `decision_runtime` | Decision runtime engine |
| `ir_constraint_compiler` | IR constraint compilation |
| `runtime_cas` | Content-addressed decision/runtime artifacts |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.context import ...
# or
from ipfs_accelerate_py.agent_supervisor.context.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Planning, proof query paths, prompt admission, control-assisted decision tools. |
| **Outbound** | `core`, proof query surfaces as needed without cycles. |
| **Forbidden** | Granting merge/completion authority from context alone. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.