# agent_supervisor.objectives

Objectives, goal ownership, backlog refinery, and bundle supervision
(`ASREF-G070` / bundle `asref/objectives`).

## Purpose

`objectives` owns the objective heap, goal completion/coverage, backlog
refinery admission, and bundle supervisor planning surfaces that higher
packages and `todo_daemon` consume without forming cycles.

This package scaffold was introduced by the ASREF-017 validation
retry-budget repair after ASREF-011 exhausted three attempts with **no
repository changes** (`declared validation plan requires changed paths`).
Downstream module moves (ASREF-011 and dependents) consume this package tree.

## Public modules (move_map ownership)

These modules are owned by `asref/objectives` and land under this package via
`git mv` (no long-lived re-export stubs at the old flat paths):

| Source (flat) | Target |
|---|---|
| `adaptive_goal_refiner.py` | `objectives/adaptive_goal_refiner.py` |
| `backlog_refinery.py` | `objectives/backlog_refinery.py` |
| `bundle_optimizer.py` | `objectives/bundle_optimizer.py` |
| `bundle_supervisor.py` | `objectives/bundle_supervisor.py` |
| `goal_completion.py` | `objectives/goal_completion.py` |
| `goal_coverage.py` | `objectives/goal_coverage.py` |
| `goal_development_contracts.py` | `objectives/goal_development_contracts.py` |
| `goal_quality.py` | `objectives/goal_quality.py` |
| `goal_refinement_verification.py` | `objectives/goal_refinement_verification.py` |
| `objective_daemon.py` | `objectives/objective_daemon.py` |
| `objective_graph.py` | `objectives/objective_graph.py` |
| `objective_task_janitor.py` | `objectives/objective_task_janitor.py` |
| `objective_tracker.py` | `objectives/objective_tracker.py` |
| `scan_receipts.py` | `objectives/scan_receipts.py` |

Inventory source: `docs/architecture/asref/move_map.json`.

## Entry points (post-move)

| Console script | Target |
|---|---|
| `ipfs-accelerate-agent-objective-daemon` | `objectives.objective_daemon:main` |
| `ipfs-accelerate-agent-backlog-refinery` | `objectives.backlog_refinery:main` |
| `ipfs-accelerate-agent-bundle-supervisor` | `objectives.bundle_supervisor:main` |

## Forbidden dependencies

`objectives` **must not** import:

- `todo_daemon` or any `todo_daemon.*` subpackage
- `runtime`, `merge`, `rescue`, `self_improvement`
- optional provider / MCP / dataset integration surfaces

## Import policy

1. After each module moves into `objectives/`, update **all** callers in the
   same change (package code, tests, scripts, entry points).
2. Do **not** leave thin re-export stubs at the former flat path.
3. Prefer absolute imports:
   `from ipfs_accelerate_py.agent_supervisor.objectives.<module> import ...`

## Status

- Package scaffold (`__init__.py`, this README): present
- Dual-copied this batch: `objective_graph.py`, `objective_daemon.py`, `backlog_refinery.py`
- Remaining owned modules: see `objectives/ASREF_G070_CHILD_GOALS.md` (proposal-gate size batches)
- Flat dual-copies remain until ASREF-G090 cutover; prefer `agent_supervisor.objectives.<module>` for landed modules
- Entry-point retargets: when task Outputs include `pyproject.toml` / `setup.py` and target modules are landed
