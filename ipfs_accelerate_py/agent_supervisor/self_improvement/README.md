# agent_supervisor.self_improvement

**Layer:** Ops · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Self-improvement program support: epoch contracts, refill, v2 evaluation/metrics/rollout models.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, objectives/runtime contracts |
| **Typical dependents** | todo_daemon, ops scripts |

## Modules

| Module | Path |
| --- | --- |
| `self_improvement` | `self_improvement/self_improvement.py` |
| `self_improvement_completion` | `self_improvement/self_improvement_completion.py` |
| `self_improvement_rollout` | `self_improvement/self_improvement_rollout.py` |
| `self_improvement_v2` | `self_improvement/self_improvement_v2.py` |
| `self_improvement_v2_rollout` | `self_improvement/self_improvement_v2_rollout.py` |
| `supervisor_efficiency_metrics` | `self_improvement/supervisor_efficiency_metrics.py` |
| `supervisor_state_model` | `self_improvement/supervisor_state_model.py` |
| `supervisor_token_ledger` | `self_improvement/supervisor_token_ledger.py` |
| `supervisor_v2_benchmark` | `self_improvement/supervisor_v2_benchmark.py` |
| `supervisor_v2_contracts` | `self_improvement/supervisor_v2_contracts.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.self_improvement.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/self_improvement.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
