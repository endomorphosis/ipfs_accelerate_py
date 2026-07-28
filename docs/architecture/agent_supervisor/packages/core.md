# agent_supervisor.core

## Purpose

Shared foundation utilities at the bottom of the package dependency DAG: conflict graphs, external completion receipts, program behavior helpers, submodule degradation, and wrapper utilities.

## When to use this package

You need leaf-safe helpers shared by control, proof, runtime, and higher packages without creating import cycles.

## Public modules

| Module | Role |
| --- | --- |
| `conflict_graph` | Conflict / resource conflict modeling |
| `external_completion` | External completion receipt binding |
| `program_behavior` | Program behavior identities |
| `submodule_degradation` | Submodule health / degradation signals |
| `wrapper_utils` | Small wrapper helpers for process boundaries |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.core import ...
# or
from ipfs_accelerate_py.agent_supervisor.core.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | All higher packages may import `core`. |
| **Outbound** | Prefer stdlib only. Must not import `todo_daemon`, `runtime`, `merge`, `rescue`, or optional providers. |
| **Forbidden** | `todo_daemon`, `runtime`, `merge`, `rescue`, `self_improvement`, MCP/provider integrations. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.
