# agent_supervisor.task_sources

**Code:** `ipfs_accelerate_py/agent_supervisor/task_sources/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/task_sources/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


## Purpose

Durable task projections and storage: Markdown and DuckDB task sources, taskboard store, persistent queues, task identity, dataset store, and todo vector indexes.

## When to use this package

You are changing how boards are parsed, queued, sharded, or projected—not how they are implemented by an LLM.

## Public modules

| Module | Role |
| --- | --- |
| `task_source` | Backend-neutral task source protocol |
| `markdown_task_source` | Markdown board projection |
| `duckdb_task_source` | DuckDB board projection |
| `taskboard_store` | Taskboard persistence |
| `persistent_task_queue` | Queue materialization |
| `task_identity` | Stable task identity helpers |
| `dataset_store` | Dataset backing for coverage/bundles |
| `todo_vector_index` | Vector index for todos |
| `duckdb_state` | DuckDB state locking helpers |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.task_sources import ...
# or
from ipfs_accelerate_py.agent_supervisor.task_sources.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | `objectives`, `planning`, `runtime`, `todo_daemon`, and related packages. |
| **Outbound** | `core`; protocol types should not be re-defined in implementations. |
| **Forbidden** | Re-defining control authority or proof trust in the taskboard layer. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.