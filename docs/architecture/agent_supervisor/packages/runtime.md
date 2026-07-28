# agent_supervisor.runtime

## Purpose

Multi-lane runtime: multi-supervisor runners, bundle supervisor entry, artifact store, event log, resource/provider schedulers, temporal monitors, and scheduler metrics.

## When to use this package

You are changing how many lanes run, how events are recorded, or how resources are admitted.

## Public modules

| Module | Role |
| --- | --- |
| `multi_supervisor_runner` | Multi-lane master runner |
| `bundle_supervisor` | Bundle-oriented supervisor |
| `artifact_store` | Artifact query/store |
| `event_log` | Durable event log |
| `resource_scheduler` | Resource admission |
| `provider_batch_scheduler` | Provider batching |
| `runtime_temporal_monitor` | Temporal monitoring |
| `scheduler_metrics` | Scheduler metrics projections |
| `implementation_daemon_runner` | Daemon runner helpers |
| `implementation_supervisor_runner` | Supervisor runner helpers |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.runtime import ...
# or
from ipfs_accelerate_py.agent_supervisor.runtime.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Operators launching multi-lane programs; control start/drain operations. |
| **Outbound** | `todo_daemon` for process loops; `merge`/`rescue` for lifecycle. |
| **Forbidden** | Embedding board-prefix-specific business logic that belongs in domain modules. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.
