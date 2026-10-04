# agent_supervisor.runtime

**Layer:** Ops · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Multi-lane execution fabric: multi-supervisor runners, event log, CAS, and resource/batch schedulers.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, merge/rescue coordination surfaces |
| **Typical dependents** | todo_daemon, ops scripts |

## Modules

| Module | Path |
| --- | --- |
| `artifact_store` | `runtime/artifact_store.py` |
| `configured_board_scheduler` | `runtime/configured_board_scheduler.py` |
| `event_log` | `runtime/event_log.py` |
| `intent_requirement_context` | `runtime/intent_requirement_context.py` |
| `intent_requirement_observation` | `runtime/intent_requirement_observation.py` |
| `learning_checkpoint` | `runtime/learning_checkpoint.py` |
| `local_planning_admission` | `runtime/local_planning_admission.py` |
| `multi_supervisor_runner` | `runtime/multi_supervisor_runner.py` |
| `operational_campaign` | `runtime/operational_campaign.py` |
| `provider_batch_scheduler` | `runtime/provider_batch_scheduler.py` |
| `resource_scheduler` | `runtime/resource_scheduler.py` |
| `router_public_instruction` | `runtime/router_public_instruction.py` |
| `runtime_cas` | `runtime/runtime_cas.py` |
| `runtime_temporal_monitor` | `runtime/runtime_temporal_monitor.py` |
| `scheduler_metrics` | `runtime/scheduler_metrics.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.runtime.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

`intent_requirement_context` projects exact accepted source and task bindings for
worker delivery. `router_public_instruction` verifies their signed public
admission without owner keys and includes them after semantic encoding.
`intent_requirement_observation` reads actual native owner validation records,
checks event order and source/revision/attempt freshness, and reports public
checks and output presence separately. These projections do not prove source
meaning, expand task scope, change canonical state or grant completion authority.

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/runtime.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
