# Entrypoints

`agent_supervisor.entrypoints` is the highest package in the supervisor
dependency graph. It owns the prompt-first Python, CLI, MCP, and MCP++
composition boundary. Domain behavior remains in the existing lower packages.

```text
core and domain packages
          ↑
todo_daemon and integrations
          ↑
entrypoints
```

The arrow means “may import.” `entrypoints` may lazily compose lower packages;
no lower package may import `entrypoints`.

## Cold-import contract

Importing this package may load only provider-free contract definitions and
package metadata. Import alone must never:

- inspect or scan a repository;
- resolve a provider, service, credential, authority, or profile;
- import implementation daemons or runtime factories;
- open SQLite, DuckDB, Parquet, IPLD, CAR, or IPFS storage;
- create files, threads, child processes, sockets, or leases.

Convenience facades added later must use module-level lazy resolution. Parser
construction, `--help`, MCP discovery, and Python API introspection have the
same cold behavior.

## Public surface

The current eager API is the reviewed
`ENTRYPOINT_CONTRACT_EXPORTS` population from `contracts.py`. Package exports
refer to the exact same class and enum objects as direct module imports:

```python
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    SupervisorInvocationRequest,
    TargetResolutionReceipt,
    LaunchPlan,
    RunHandle,
)
```

`ENTRYPOINT_LAZY_FACADE_EXPORTS` is intentionally empty until facade delivery
tasks land. Adding a name requires a boundary test and cannot make import-time
capability claims.

## Storage and authority

The contract layer keeps mutable coordination distinct from immutable
distribution:

- one elected owner writes each DuckDB coordination shard and grants fenced
  claims through transactional compare-and-swap;
- Parquet epochs and DAG-JSON/IPLD/CAR records replicate immutable history;
- IPFS availability never grants a lease, authority, or permission;
- prompt bodies, bearer UCANs, credentials, and environment values remain
  outside durable records.

## Placement rule

Put transport-neutral control behavior in `control/`, prompt planning in
`prompt/`, task projections in `task_sources/`, coordination primitives in
`merge/` or `runtime/`, and daemon execution in `todo_daemon/`. Put only the
high-level composition facade and its provider-free request/result contracts
here.
