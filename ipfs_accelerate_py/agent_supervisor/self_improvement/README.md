# agent_supervisor.self_improvement

Bounded self-improvement epoch contracts, successor refill, and supervisor
v2 efficiency surfaces (`ASREF-G070` / bundle `asref/self-improvement`).

## Purpose

`self_improvement` owns epoch evaluation, healthy-exhaustion evidence,
successor refill materialization, rollout/benchmark helpers, and supervisor
v2 state/token/efficiency models.

This package scaffold was introduced by the ASREF-017 validation
retry-budget repair after ASREF-011 exhausted three attempts with **no
repository changes** (`declared validation plan requires changed paths`).

## Temporary flat-module shadowing

Python prefers the package directory over the sibling flat module
`self_improvement.py`. Until ASREF-011 moves modules under this package:

- Package `__init__.py` re-exports the flat module public API so
  `from ipfs_accelerate_py.agent_supervisor.self_improvement import ...`
  continues to work.
- That re-export is **temporary compatibility only**, not a permanent
  old-path stub policy for other packages.
- Sibling modules (`self_improvement_v2`, `self_improvement_rollout`, …)
  remain importable at their current flat paths until moved.

## Public modules (move_map ownership)

| Source (flat) | Target |
|---|---|
| `self_improvement.py` | `self_improvement/` package modules (stem lands as package core) |
| `self_improvement_completion.py` | `self_improvement/self_improvement_completion.py` |
| `self_improvement_rollout.py` | `self_improvement/self_improvement_rollout.py` |
| `self_improvement_v2.py` | `self_improvement/self_improvement_v2.py` |
| `self_improvement_v2_rollout.py` | `self_improvement/self_improvement_v2_rollout.py` |
| `supervisor_efficiency_metrics.py` | `self_improvement/supervisor_efficiency_metrics.py` |
| `supervisor_state_model.py` | `self_improvement/supervisor_state_model.py` |
| `supervisor_token_ledger.py` | `self_improvement/supervisor_token_ledger.py` |
| `supervisor_v2_benchmark.py` | `self_improvement/supervisor_v2_benchmark.py` |
| `supervisor_v2_contracts.py` | `self_improvement/supervisor_v2_contracts.py` |

Inventory source: `docs/architecture/asref/move_map.json`.

## Forbidden dependencies

`self_improvement` **must not** import `todo_daemon` or optional
integration surfaces in a way that forms package DAG cycles.

## Import policy

After each module moves into `self_improvement/`, update all callers in the
same change. Prefer::

    from ipfs_accelerate_py.agent_supervisor.self_improvement.<module> import ...

Remove the temporary flat re-export path once `self_improvement.py` content
lives only under this package and no callers depend on package-root symbols
from the old flat module.

## Status

- Package scaffold (`__init__.py`, this README): present
- Dual-copied this batch: `self_improvement_completion.py`
- Remaining owned modules: see `objectives/ASREF_G070_CHILD_GOALS.md` (proposal-gate size batches)
- Flat dual-copies remain until ASREF-G090 cutover; prefer `agent_supervisor.self_improvement.<module>` for landed modules
- Entry-point retargets: when task Outputs include `pyproject.toml` / `setup.py` and target modules are landed
