# agent_supervisor.validation

**Layer:** Mid · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Pre-merge and proposal validation: schedulers, runtimes, commands, and scope adjudication gates.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, planning/proposal contracts |
| **Typical dependents** | todo_daemon, merge, control mutations |

## Modules

| Module | Path |
| --- | --- |
| `proposal_validation` | `validation/proposal_validation.py` |
| `scope_adjudication` | `validation/scope_adjudication.py` |
| `validation_commands` | `validation/validation_commands.py` |
| `validation_runtime` | `validation/validation_runtime.py` |
| `validation_scheduler` | `validation/validation_scheduler.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.validation.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/validation.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
