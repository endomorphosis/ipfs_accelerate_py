# agent_supervisor.rescue

**Layer:** Ops · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Failure recovery: rescue planners/orchestrators, failure policy, and watchdog/recovery hooks.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, merge/runtime hooks carefully |
| **Typical dependents** | todo_daemon, runtime |

## Modules

| Module | Path |
| --- | --- |
| `codex_failure_policy` | `rescue/codex_failure_policy.py` |
| `recovery_diagnostics` | `rescue/recovery_diagnostics.py` |
| `rescue_orchestrator` | `rescue/rescue_orchestrator.py` |
| `rescue_planner` | `rescue/rescue_planner.py` |
| `supervisor_recovery` | `rescue/supervisor_recovery.py` |
| `supervisor_watchdog` | `rescue/supervisor_watchdog.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.rescue.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/rescue.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
