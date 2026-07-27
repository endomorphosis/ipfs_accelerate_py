# agent_supervisor.validation

Proposal validation, scope adjudication, and validation scheduling
(`ASREF-G070` / bundle `asref/validation`).

## Purpose

`validation` owns pre-merge proposal gates, scope adjudication, validation
command construction, runtime execution, and scheduler surfaces used by the
implementation daemon and merge pipeline.

This package scaffold was introduced by the ASREF-017 validation
retry-budget repair after ASREF-011 exhausted three attempts with **no
repository changes** (`declared validation plan requires changed paths`).

## Public modules (move_map ownership)

| Source (flat) | Target |
|---|---|
| `proposal_validation.py` | `validation/proposal_validation.py` |
| `scope_adjudication.py` | `validation/scope_adjudication.py` |
| `validation_commands.py` | `validation/validation_commands.py` |
| `validation_runtime.py` | `validation/validation_runtime.py` |
| `validation_scheduler.py` | `validation/validation_scheduler.py` |

Inventory source: `docs/architecture/asref/move_map.json`.

## Forbidden dependencies

`validation` **must not** import `todo_daemon`, `runtime`, `merge`, `rescue`,
`self_improvement`, or optional integration surfaces.

## Import policy

After each module moves into `validation/`, update all callers in the same
change. Do not leave thin re-export stubs at former flat paths. Prefer::

    from ipfs_accelerate_py.agent_supervisor.validation.<module> import ...

## Status

- Package scaffold (`__init__.py`, this README): present
- Dual-copied this batch: `proposal_validation.py`
- Remaining owned modules: see `objectives/ASREF_G070_CHILD_GOALS.md` (proposal-gate size batches)
- Flat dual-copies remain until ASREF-G090 cutover; prefer `agent_supervisor.validation.<module>` for landed modules
- Entry-point retargets: when task Outputs include `pyproject.toml` / `setup.py` and target modules are landed
