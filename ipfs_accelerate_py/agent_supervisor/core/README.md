# agent_supervisor.core

**Layer:** Foundation · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Shared foundation at the bottom of the package DAG. Identity helpers, conflict graphs, external completion receipts, and wrapper utilities that higher packages depend on without cycles.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | stdlib and carefully chosen leaf helpers only; never daemons/runtime/merge/rescue |
| **Typical dependents** | all higher packages, tests, scripts |

## Modules

| Module | Path |
| --- | --- |
| `conflict_graph` | `core/conflict_graph.py` |
| `external_completion` | `core/external_completion.py` |
| `program_behavior` | `core/program_behavior.py` |
| `submodule_degradation` | `core/submodule_degradation.py` |
| `wrapper_utils` | `core/wrapper_utils.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.core.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/core.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
