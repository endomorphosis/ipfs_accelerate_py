# agent_supervisor.merge

**Layer:** Ops · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Landing completed work: merge queue/train/resolver, checkout locks, leases, and git hygiene helpers.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, runtime-safe helpers |
| **Typical dependents** | todo_daemon, runtime, rescue |

## Modules

| Module | Path |
| --- | --- |
| `checkout_lock` | `merge/checkout_lock.py` |
| `git_gc` | `merge/git_gc.py` |
| `lease_coordination` | `merge/lease_coordination.py` |
| `leased_lane` | `merge/leased_lane.py` |
| `merge_checkpoint` | `merge/merge_checkpoint.py` |
| `merge_conflict_repair` | `merge/merge_conflict_repair.py` |
| `merge_queue` | `merge/merge_queue.py` |
| `merge_resolver` | `merge/merge_resolver.py` |
| `merge_train` | `merge/merge_train.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.merge.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/merge.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
