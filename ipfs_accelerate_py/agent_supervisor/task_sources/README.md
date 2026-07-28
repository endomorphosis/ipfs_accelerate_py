# agent_supervisor.task_sources

**Layer:** Foundation · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Task projection and storage: Markdown/DuckDB taskboards, queues, indexes, and task identity used by daemons and planners.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core` |
| **Typical dependents** | objectives, planning, todo_daemon, runtime |

## Modules

| Module | Path |
| --- | --- |
| `dataset_store` | `task_sources/dataset_store.py` |
| `duckdb_state` | `task_sources/duckdb_state.py` |
| `duckdb_task_source` | `task_sources/duckdb_task_source.py` |
| `markdown_task_source` | `task_sources/markdown_task_source.py` |
| `persistent_task_queue` | `task_sources/persistent_task_queue.py` |
| `task_identity` | `task_sources/task_identity.py` |
| `task_source` | `task_sources/task_source.py` |
| `taskboard_store` | `task_sources/taskboard_store.py` |
| `todo_vector_index` | `task_sources/todo_vector_index.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.task_sources.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/task_sources.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
