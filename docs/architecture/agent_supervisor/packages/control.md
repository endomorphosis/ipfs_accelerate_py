# agent_supervisor.control

## Purpose

Transport-neutral control plane: closed operation vocabulary, request/result contracts, capability reports, control service, CLI registration, lifecycle orchestration, execution permits, and authorization policy.

## When to use this package

You are adding or changing operations exposed to Python, CLI (`ipfs-accelerate agent`), or MCP—or tightening allowlists, dry-run, idempotency, or fencing.

## Public modules

| Module | Role |
| --- | --- |
| `control_contracts` | Operations, schemas, capabilities, discovery |
| `control_plane` | SupervisorControlService and backends |
| `control_cli` | CLI dispatch for unified agent command |
| `lifecycle_orchestrator` | Process lifecycle inside control mutations |
| `execution_permit` | Short-lived mutation permits |
| `authorization_logic` | Deterministic authz / delegation evaluation |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.control import ...
# or
from ipfs_accelerate_py.agent_supervisor.control.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Python embeddings, CLI, MCP adapters, higher packages that need control types. |
| **Outbound** | May use `core` and stable contracts; must not pull implementation daemons for ordinary reads. |
| **Forbidden** | Loading optional provers/providers on cold import of control contracts. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.
