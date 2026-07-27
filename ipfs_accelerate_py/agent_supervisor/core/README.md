# agent_supervisor.core

Shared foundation package for the agent supervisor module layout
(`ASREF-G020` / bundle `asref/core`).

## Purpose

`core` is the bottom of the package dependency DAG. It owns reusable
identity, conflict, completion, and wrapper utilities that higher packages
depend on without forming cycles.

## Public modules (move_map ownership)

These modules are owned by `asref/core` and live under this package
(inventory: `docs/architecture/asref/move_map.json`):

| Module | Path |
|---|---|
| `conflict_graph` | `core/conflict_graph.py` |
| `external_completion` | `core/external_completion.py` |
| `program_behavior` | `core/program_behavior.py` |
| `submodule_degradation` | `core/submodule_degradation.py` |
| `wrapper_utils` | `core/wrapper_utils.py` |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ...
from ipfs_accelerate_py.agent_supervisor.core.external_completion import ...
from ipfs_accelerate_py.agent_supervisor.core.program_behavior import ...
from ipfs_accelerate_py.agent_supervisor.core.submodule_degradation import ...
from ipfs_accelerate_py.agent_supervisor.core.wrapper_utils import ...
```

Relative imports inside `core/` stay package-local (`from .x import y`).
Outbound imports to still-flat siblings use one parent level
(`from ..task_identity import ...`).

## Allowed dependents

Packages **may** import from `ipfs_accelerate_py.agent_supervisor.core`:

- `control`, `task_sources`, `context`, `analysis`, `proof`
- `objectives`, `planning`, `validation`, `prompt`
- `merge`, `rescue`, `runtime`, `self_improvement`
- `todo_daemon.*`, `integrations`
- tests, scripts, and console entry points that previously imported the
  flat module paths listed above

## Forbidden dependencies

`core` **must not** import:

- `todo_daemon` or any `todo_daemon.*` subpackage
- `runtime` (higher DAG layer)
- `merge`, `rescue`, `self_improvement`
- optional provider / MCP / dataset integration surfaces

Prefer stdlib and same-package symbols. Cross-package imports from `core`
are limited to other leaf-safe helpers only when the package DAG remains
acyclic.

## Import policy

1. New and updated callers must import from `core.<module>`.
2. Do **not** leave thin re-export stubs at former flat paths.
3. After callers are rewritten, remove the temporary flat copies of
   core-owned modules (if still present) without reintroducing stubs.
4. Prefer absolute imports:
   `from ipfs_accelerate_py.agent_supervisor.core.<module> import ...`

## Validation (owning goal)

```bash
python -m pytest test/api/test_agent_supervisor_event_driven_runtime.py -q --collect-only
```

After full caller cutover, also assert absence of old flat imports, e.g.:

```bash
rg -n 'agent_supervisor\.(conflict_graph|external_completion|program_behavior|submodule_degradation|wrapper_utils)\b' \
  --glob '!**/__pycache__/**' --glob '!docs/architecture/asref/**' --glob '!**/core/**'
```

## Status

- Package scaffold (`__init__.py`, this README): present
- Core-owned modules under `core/`: present (ASREF-003)
- Caller rewrites + removal of temporary flat copies: follow-on when
  edit scope authorizes import-update paths (proposal expansion limit is 8;
  full cutover exceeds that bound and cannot delete out-of-scope paths)
