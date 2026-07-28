# agent_supervisor.task_sources

Markdown and DuckDB task projections, taskboard store, queues, and indexes
(`ASREF-G040` / bundle `asref/task-sources`).

## Purpose

`task_sources` owns durable task-board projections and related storage:

- Markdown task-source projection and materialization
- DuckDB task-source projection and state locking
- Taskboard store and persistent task queue
- Task identity helpers and the backend-neutral task-source protocol
- Dataset store and todo vector index used by coverage / bundling

It sits **above** `core` in the package dependency DAG. Higher packages
(objectives, planning, runtime, todo_daemon, …) may depend on it without
forming cycles.

## Protocol dependence on core / task_source

ASREF-G040 refinement: **keep the protocol in `core`; keep implementations in
`task_sources`.**

| Layer | Module | Role |
|---|---|---|
| Protocol (design home: `core.task_source`) | `task_source` | Backend-neutral `CanonicalTaskSource`, dual projection, parity, migration receipts |
| Implementation | `markdown_task_source` | Markdown board projection |
| Implementation | `duckdb_task_source` | DuckDB board projection |
| Storage helpers | `taskboard_store`, `duckdb_state`, `persistent_task_queue`, `dataset_store`, `todo_vector_index`, `task_identity` | Supporting surfaces |

Until `task_source` is relocated under `core/`, the protocol module co-resides
here so dual-projection code and callers stay coherent. Implementations must
not re-define the protocol types; they implement or adapt to them.

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
    MarkdownTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_task_source import (
    DuckDBTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    CanonicalTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_store import (
    TaskboardStore,
)
```

Relative imports inside `task_sources/` stay package-local
(`from .taskboard_store import ...`). Outbound imports to still-flat siblings
use one parent level (`from ..prompt.prompt_workflow import ...`). Imports of
core-owned modules use the core package
(`from ..core.conflict_graph import ...`).

## Public modules (move_map ownership)

These modules are owned by `asref/task-sources` and live under this package
(inventory: `docs/architecture/asref/move_map.json`):

| Module | Path |
|---|---|
| `dataset_store` | `task_sources/dataset_store.py` |
| `duckdb_state` | `task_sources/duckdb_state.py` |
| `duckdb_task_source` | `task_sources/duckdb_task_source.py` |
| `markdown_task_source` | `task_sources/markdown_task_source.py` |
| `persistent_task_queue` | `task_sources/persistent_task_queue.py` |
| `task_identity` | `task_sources/task_identity.py` |
| `task_source` | `task_sources/task_source.py` |
| `taskboard_store` | `task_sources/taskboard_store.py` |
| `todo_vector_index` | `task_sources/todo_vector_index.py` |

## Allowed dependents

Packages **may** import from `ipfs_accelerate_py.agent_supervisor.task_sources`:

- `objectives`, `planning`, `validation`, `prompt`
- `merge`, `rescue`, `runtime`, `self_improvement`
- `todo_daemon.*`, `integrations`
- tests, scripts, and console entry points that previously imported the
  flat module paths listed above

## Forbidden dependencies

`task_sources` **must not** import:

- `todo_daemon` or any `todo_daemon.*` subpackage
- `runtime` (higher DAG layer)
- `merge`, `rescue`, `self_improvement`
- optional provider / MCP integration surfaces that would invert the DAG

Allowed inbound dependency: `core` (and still-flat siblings until their
packages land). Prefer stdlib and same-package symbols where practical.

## Import policy

1. New and updated callers must import from `task_sources.<module>`.
2. Do **not** leave thin re-export stubs at former flat paths.
3. After callers are rewritten, remove the temporary flat copies of
   task_sources-owned modules (if still present) without reintroducing stubs.
4. Prefer absolute imports:
   `from ipfs_accelerate_py.agent_supervisor.task_sources.<module> import ...`

## Validation (owning goal)

```bash
python -m pytest test/api/test_agent_supervisor_markdown_task_source.py \
  test/api/test_agent_supervisor_duckdb_task_source.py -q
```

After full caller cutover, also assert absence of old flat imports, e.g.:

```bash
rg -n 'agent_supervisor\.(markdown_task_source|duckdb_task_source|taskboard_store|persistent_task_queue|todo_vector_index|task_identity|task_source|dataset_store|duckdb_state)\b' \
  --glob '!**/__pycache__/**' --glob '!docs/architecture/asref/**' --glob '!**/task_sources/**'
```

And assert no flat leftovers for moved modules:

```bash
# expected: no hits after cutover
ls ipfs_accelerate_py/agent_supervisor/{markdown_task_source,duckdb_task_source,taskboard_store,persistent_task_queue,todo_vector_index,task_identity,task_source,dataset_store,duckdb_state}.py
```

## Status

- Package scaffold (`__init__.py`, this README): present
- Task-sources-owned modules under `task_sources/`: present (ASREF-005)
- Caller rewrites + removal of temporary flat copies: follow-on when
  edit scope authorizes import-update paths (proposal expansion limit is 8;
  full cutover exceeds that bound and cannot delete out-of-scope paths)
- Protocol relocation into `core.task_source`: follow-on once core edit scope
  includes the protocol module (G040 refinement)
