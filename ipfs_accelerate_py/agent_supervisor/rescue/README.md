# agent_supervisor.rescue

Rescue orchestration, recovery diagnostics, and supervisor watchdog
surfaces (`ASREF-G070` / bundle `asref/rescue`).

## Purpose

`rescue` owns failure policy, rescue planning/orchestration, recovery
diagnostics, and supervisor recovery/watchdog loops used when
implementation lanes stall or providers fail.

This package scaffold was introduced by the ASREF-017 validation
retry-budget repair after ASREF-011 exhausted three attempts with **no
repository changes** (`declared validation plan requires changed paths`).

## Public modules (move_map ownership)

| Source (flat) | Target |
|---|---|
| `codex_failure_policy.py` | `rescue/codex_failure_policy.py` |
| `recovery_diagnostics.py` | `rescue/recovery_diagnostics.py` |
| `rescue_orchestrator.py` | `rescue/rescue_orchestrator.py` |
| `rescue_planner.py` | `rescue/rescue_planner.py` |
| `supervisor_recovery.py` | `rescue/supervisor_recovery.py` |
| `supervisor_watchdog.py` | `rescue/supervisor_watchdog.py` |

Inventory source: `docs/architecture/asref/move_map.json`.

## Forbidden dependencies

`rescue` **must not** import `todo_daemon`, `self_improvement`, or optional
integration surfaces.

## Import policy

After each module moves into `rescue/`, update all callers in the same
change. Do not leave thin re-export stubs at former flat paths. Prefer::

    from ipfs_accelerate_py.agent_supervisor.rescue.<module> import ...

## Status

- Package scaffold (`__init__.py`, this README): present
- Dual-copied this batch: `rescue_orchestrator.py`, `codex_failure_policy.py`
- Remaining owned modules: see `objectives/ASREF_G070_CHILD_GOALS.md` (proposal-gate size batches)
- Flat dual-copies remain until ASREF-G090 cutover; prefer `agent_supervisor.rescue.<module>` for landed modules
- Entry-point retargets: when task Outputs include `pyproject.toml` / `setup.py` and target modules are landed
