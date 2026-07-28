# agent_supervisor.runtime

Multi-supervisor runtime, artifact store, event log, and schedulers
(`ASREF-G070` / bundle `asref/runtime`).

## Purpose

`runtime` owns durable event/CAS surfaces, artifact query entry points,
provider/resource schedulers, multi-supervisor runners, and temporal
monitoring used by control and implementation lanes.

This package scaffold was introduced by the ASREF-017 validation
retry-budget repair after ASREF-011 exhausted three attempts with **no
repository changes** (`declared validation plan requires changed paths`).

## Public modules (move_map ownership)

| Source (flat) | Target |
|---|---|
| `artifact_store.py` | `runtime/artifact_store.py` |
| `event_log.py` | `runtime/event_log.py` |
| `multi_supervisor_runner.py` | `runtime/multi_supervisor_runner.py` |
| `provider_batch_scheduler.py` | `runtime/provider_batch_scheduler.py` |
| `resource_scheduler.py` | `runtime/resource_scheduler.py` |
| `runtime_cas.py` | `runtime/runtime_cas.py` |
| `runtime_temporal_monitor.py` | `runtime/runtime_temporal_monitor.py` |
| `scheduler_metrics.py` | `runtime/scheduler_metrics.py` |

Inventory source: `docs/architecture/asref/move_map.json`.

## Entry points (post-move)

| Console script | Target |
|---|---|
| `ipfs-accelerate-agent-artifact-query` | `runtime.artifact_store:main` |

## Forbidden dependencies

`runtime` **must not** import `todo_daemon`, `self_improvement`, or optional
integration surfaces.

## Import policy

After each module moves into `runtime/`, update all callers in the same
change. Do not leave thin re-export stubs at former flat paths. Prefer::

    from ipfs_accelerate_py.agent_supervisor.runtime.<module> import ...

## Status

- Package scaffold (`__init__.py`, this README): present
- Dual-copied this batch: `multi_supervisor_runner.py`
- Remaining owned modules: see `objectives/ASREF_G070_CHILD_GOALS.md` (proposal-gate size batches)
- Flat dual-copies remain until ASREF-G090 cutover; prefer `agent_supervisor.runtime.<module>` for landed modules
- Entry-point retargets: when task Outputs include `pyproject.toml` / `setup.py` and target modules are landed
