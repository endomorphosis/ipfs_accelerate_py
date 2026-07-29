# agent_supervisor.analysis

**Code:** `ipfs_accelerate_py/agent_supervisor/analysis/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/analysis/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


## Purpose

Repository analysis pipeline: AST indexes, analysis cache, retrieval, consensus, transport, and integrated analysis orchestration.

## When to use this package

You are improving how the supervisor discovers symbols, text, or structural evidence for objective scanning.

## Public modules

| Module | Role |
| --- | --- |
| `analysis_pipeline` | Integrated analysis orchestration |
| `analysis_ast_index` | AST / symbol index |
| `analysis_cache` | Analysis result cache |
| `analysis_retrieval` | Retrieval over analysis artifacts |
| `analysis_consensus` | Consensus across analysis channels |
| `analysis_contracts` | Analysis contracts and schemas |
| `analysis_transport` | Transport for analysis jobs |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.analysis import ...
# or
from ipfs_accelerate_py.agent_supervisor.analysis.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Objectives scanner, backlog refinery, planning aids. |
| **Outbound** | `core`, optional dataset providers via integrations only when explicit. |
| **Forbidden** | Treating analysis hits as kernel-level proofs. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.