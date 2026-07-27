# agent_supervisor.planning

Formal and adaptive planning surfaces for the agent supervisor
(`ASREF-G070` / bundle `asref/planning`).

## Purpose

`planning` owns adaptive planners, formal plan compile/validate/conformance
pipelines, plan evaluation, failure memory, proof-carrying planners, and task
proposal quality routing.

This package scaffold was introduced by the ASREF-017 validation
retry-budget repair after ASREF-011 exhausted three attempts with **no
repository changes** (`declared validation plan requires changed paths`).

## Public modules (move_map ownership)

| Source (flat) | Target |
|---|---|
| `adaptive_planner.py` | `planning/adaptive_planner.py` |
| `formal_plan_compiler.py` | `planning/formal_plan_compiler.py` |
| `formal_plan_conformance.py` | `planning/formal_plan_conformance.py` |
| `formal_plan_context.py` | `planning/formal_plan_context.py` |
| `formal_plan_validator.py` | `planning/formal_plan_validator.py` |
| `formal_planning_adversarial.py` | `planning/formal_planning_adversarial.py` |
| `formal_planning_contracts.py` | `planning/formal_planning_contracts.py` |
| `formal_planning_metrics.py` | `planning/formal_planning_metrics.py` |
| `formal_planning_rollout.py` | `planning/formal_planning_rollout.py` |
| `formal_replanner.py` | `planning/formal_replanner.py` |
| `plan_evaluator.py` | `planning/plan_evaluator.py` |
| `plan_failure_memory.py` | `planning/plan_failure_memory.py` |
| `proof_carrying_planner.py` | `planning/proof_carrying_planner.py` |
| `task_proposal_router.py` | `planning/task_proposal_router.py` |
| `task_quality.py` | `planning/task_quality.py` |

Inventory source: `docs/architecture/asref/move_map.json`.

## Forbidden dependencies

`planning` **must not** import `todo_daemon`, `runtime`, `merge`, `rescue`,
`self_improvement`, or optional integration surfaces.

## Import policy

After each module moves into `planning/`, update all callers in the same
change. Do not leave thin re-export stubs at former flat paths. Prefer::

    from ipfs_accelerate_py.agent_supervisor.planning.<module> import ...

## Status

- Package scaffold (`__init__.py`, this README): present
- Dual-copied this batch: `plan_failure_memory.py`, `formal_planning_metrics.py`, `formal_planning_rollout.py`
- Remaining owned modules: see `objectives/ASREF_G070_CHILD_GOALS.md` (proposal-gate size batches)
- Flat dual-copies remain until ASREF-G090 cutover; prefer `agent_supervisor.planning.<module>` for landed modules
- Entry-point retargets: when task Outputs include `pyproject.toml` / `setup.py` and target modules are landed
