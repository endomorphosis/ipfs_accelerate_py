# agent_supervisor.core

Shared foundation package for the agent supervisor module layout
(`ASREF-G020` / bundle `asref/core`).

## Purpose

`core` is the bottom of the package dependency DAG. It owns reusable
identity, conflict, completion, and wrapper utilities that higher packages
depend on without forming cycles.

This package was introduced by the ASREF-015 validation retry-budget repair
after ASREF-003 exhausted three attempts with **no repository changes**
(`declared validation plan requires changed paths`). Downstream package
moves (ASREF-003 and dependents) consume this package tree.

## Public modules (move_map ownership)

These modules are owned by `asref/core` and land under this package via
`git mv` (no long-lived re-export stubs at the old flat paths):

| Source (flat) | Target |
|---|---|
| `conflict_graph.py` | `core/conflict_graph.py` |
| `external_completion.py` | `core/external_completion.py` |
| `program_behavior.py` | `core/program_behavior.py` |
| `submodule_degradation.py` | `core/submodule_degradation.py` |
| `wrapper_utils.py` | `core/wrapper_utils.py` |

Inventory source: `docs/architecture/asref/move_map.json`.

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

1. After each module moves into `core/`, update **all** callers in the same
   change (package code, tests, scripts, entry points).
2. Do **not** leave thin re-export stubs at the former flat path.
3. Prefer absolute imports:
   `from ipfs_accelerate_py.agent_supervisor.core.<module> import ...`
4. Relative imports inside `core/` stay package-local (`from .x import y`).

## Validation (owning goal)

```bash
python -m pytest test/api/test_agent_supervisor_event_driven_runtime.py -q --collect-only
```

After full module moves, also assert absence of old flat imports, e.g.:

```bash
rg -n 'agent_supervisor\.(conflict_graph|external_completion|program_behavior|submodule_degradation|wrapper_utils)\b' \
  --glob '!**/__pycache__/**' --glob '!docs/architecture/asref/**'
```

## Status

- Package scaffold (`__init__.py`, this README): present
- Module `git mv` + caller rewrites: owned by ASREF-003 once released from
  strategy `blocked_tasks` after this repair completes
