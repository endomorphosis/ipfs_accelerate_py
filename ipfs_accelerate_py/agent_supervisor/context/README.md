# agent_supervisor.context

**Layer:** Foundation · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Context compilation and decision runtime: obligation-first capsules, decision contracts, and runtime/rollout helpers for proof-directed work.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, limited analysis/proof contracts |
| **Typical dependents** | planning, proof pipelines, prompt/decision rollouts |

## Modules

| Module | Path |
| --- | --- |
| `context_compiler` | `context/context_compiler.py` |
| `context_contracts` | `context/context_contracts.py` |
| `decision_context` | `context/decision_context.py` |
| `decision_contracts` | `context/decision_contracts.py` |
| `decision_runtime` | `context/decision_runtime.py` |
| `decision_runtime_benchmark` | `context/decision_runtime_benchmark.py` |
| `decision_runtime_rollout` | `context/decision_runtime_rollout.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.context.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/context.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
