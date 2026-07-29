# agent_supervisor.control

**Layer:** Foundation · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Transport-neutral control plane. Python, CLI, and MCP build the same OperationRequest records and invoke SupervisorControlService without embedding policy in each transport.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, contracts-level `proof`/`validation` as needed without cycles |
| **Typical dependents** | objectives, planning, validation, prompt, merge, rescue, runtime, self_improvement, todo_daemon, integrations |

## Modules

| Module | Path |
| --- | --- |
| `authorization_logic` | `control/authorization_logic.py` |
| `control_cli` | `control/control_cli.py` |
| `control_contracts` | `control/control_contracts.py` |
| `control_plane` | `control/control_plane.py` |
| `execution_permit` | `control/execution_permit.py` |
| `lifecycle_orchestrator` | `control/lifecycle_orchestrator.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.control.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/control.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)

## CLI entry surfaces

| Surface | Target |
|---|---|
| Unified CLI register | `control.control_cli:register_agent_cli` |
| Unified CLI run | `control.control_cli:run_agent_cli` |
| Control service | `control.control_plane:SupervisorControlService` |

`ipfs_accelerate_py.cli` should call into `control.control_cli` after full
caller cutover. During dual-path cutover, temporary flat imports may still
resolve the pre-move modules.
