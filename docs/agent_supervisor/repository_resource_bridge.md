# Shared repository resource bridge

`runtime.repository_resource_bridge` provides an opt-in CPU parent envelope for
repository work. The existing supervisor `ResourceScheduler` retains local lane
and policy bookkeeping; the existing default datasets `GlobalResourceScheduler`
is the shared host CPU, memory and process admission authority. No second host
capacity is created.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_bridge import (
    RepositoryResourceBridge, RepositoryResourceBudget, RepositoryPhaseDemand,
)

with RepositoryResourceBridge(supervisor_resource_scheduler).reserve(
    repository_id="example", workspace=repository,
    budget=RepositoryResourceBudget(wall_time_ms=120000),
    cancel_event=cancellation,
) as parent:
    with parent.phase(RepositoryPhaseDemand("semantic_index")) as phase:
        # Pass these to a datasets consumer that accepts native parent leases.
        consumer(**phase.native_options())
    receipt = parent.receipt()
```

The supported phase names are `scan`, `sql`, `parquet`, `semantic_index`,
`inference`, `training`, `search`, `proof`, `validation`, `persistence`, and
`cleanup`. Each phase acquires a native child lease and checks the remaining
parent deadline. A consumer can acquire a further child under that phase; it
must use the supplied parent instead of creating another root reservation.
`native_options()` returns `parent_lease`, `cancel_event`, `timeout_seconds`, and
`memory_mb`; callers should pass only keywords their existing consumer accepts.
For an API without timeout or memory keywords, the enclosing phase still bounds
admission, while the caller must propagate cancellation and its native parent.

Use `phase.thread_environment()` with a bounded subprocess launcher when thread
caps are needed. The bridge does not mutate process-wide environments. The
receipt reports the shared owner, state path, policy digest, parent identities,
phase wait/execution times and queue counts. It contains no lease keys and grants
no execution, completion or proof authority.

Cancellation combines the caller signal, native cancellation and continued
existence of the supervisor lease. Closing cancels and waits for cooperating
phases to drain. If they do not drain within five seconds, it refuses to report
cleanup and retains reservations. Native stale-owner recovery reclaims lease
accounting after a process dies; terminating arbitrary untracked orphan
processes is outside this bridge.

The profile preserves existing safety/backoff and the global validation lane
reservation. That reservation does not create reserved capacity within every
parent: training can occupy a complete parent envelope and delay its sibling
validation. Disk budgets are local declarations plus free-space sampling, and
queue byte bounds cover bridge request metadata. CPU/memory/process leases are
admission accounting, not kernel hard enforcement. GPU/per-device/provider
admission is rejected; complete shared disk and queue authority, gradual
recovery, and all production call-site integration remain open under RPI-022.

The [qualification evidence](evidence/repository-resource-bridge-20261002/README.md)
records 18 passing tests, including three with the real default scheduler and
live host telemetry. Earlier resource refusals are retained as diagnostics.
