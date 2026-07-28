# agent_supervisor.control

Control plane, CLI adapter, contracts, lifecycle orchestration, and execution
permits (`ASREF-G030` / bundle `asref/control`).

## Purpose

`control` is the shared, transport-neutral authority surface for the agent
supervisor. Python, CLI (`ipfs-accelerate agent`), and MCP adapters construct
the same immutable `OperationRequest` records and invoke
`SupervisorControlService` without embedding supervisor policy in each
transport.

This package owns:

- Typed operation catalog, request/result contracts, and capability reports
- Control service (allowlists, authz freshness, leases/fencing, idempotency,
  dry-run, audit receipts)
- CLI registration and dispatch for the unified agent command surface
- Process lifecycle orchestration inside control mutations
- Short-lived execution permits for mutation-capable operations
- Deterministic authorization / delegation policy evaluation

## Public modules (move_map ownership)

These modules are owned by `asref/control` and live under this package
(inventory: `docs/architecture/asref/move_map.json`):

| Module | Path |
|---|---|
| `authorization_logic` | `control/authorization_logic.py` |
| `control_cli` | `control/control_cli.py` |
| `control_contracts` | `control/control_contracts.py` |
| `control_plane` | `control/control_plane.py` |
| `execution_permit` | `control/execution_permit.py` |
| `lifecycle_orchestrator` | `control/lifecycle_orchestrator.py` |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    Operation,
    OperationRequest,
    OperationResult,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    register_agent_cli,
    run_agent_cli,
)
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import ...
from ipfs_accelerate_py.agent_supervisor.control.execution_permit import ...
from ipfs_accelerate_py.agent_supervisor.control.authorization_logic import ...
```

Relative imports inside `control/` stay package-local (`from .control_contracts import ...`).
Outbound imports to still-flat siblings use one parent level
(`from ..proof.formal_verification_contracts import ...`).

## CLI entry surfaces

| Surface | Target |
|---|---|
| Unified CLI register | `control.control_cli:register_agent_cli` |
| Unified CLI run | `control.control_cli:run_agent_cli` |
| Control service | `control.control_plane:SupervisorControlService` |

`ipfs_accelerate_py.cli` should call into `control.control_cli` after full
caller cutover. During dual-path cutover, temporary flat imports may still
resolve the pre-move modules.

## Allowed dependents

Packages **may** import from `ipfs_accelerate_py.agent_supervisor.control`:

- `objectives`, `planning`, `validation`, `prompt`
- `merge`, `rescue`, `runtime`, `self_improvement`
- `todo_daemon.*`, `integrations`
- tests, scripts, MCP tools, and console entry points that previously imported
  the flat module paths listed above

`control` itself may depend on `core` and other leaf-safe still-flat helpers
that do not create package cycles (for example formal contracts used by
control contracts, or objective/task board readers used as backends).

## Forbidden dependencies

`control` **must not** import:

- `todo_daemon` or any `todo_daemon.*` subpackage as a hard package dependency
  of the control contract surface (optional backends remain unregistered until
  a deployment wires them)
- `self_improvement` (higher / peer layer that must not cycle)
- optional provider / MCP / dataset integration packages as eager imports
- shell-string translation of control operations (backends are Python callables)

## Import policy

1. New and updated callers must import from `control.<module>`.
2. Do **not** leave thin re-export stubs at former flat paths.
3. After callers are rewritten, remove temporary flat copies of control-owned
   modules (if still present) without reintroducing stubs. Acceptance for this
   package requires absence of old `control_*` flat modules once edit scope
   authorizes those deletions and import rewrites.
4. Prefer absolute imports:
   `from ipfs_accelerate_py.agent_supervisor.control.<module> import ...`

## Package dependency DAG position

```text
core
  ↑
control   ← this package (with task_sources, context, analysis, proof)
  ↑
objectives, planning, validation, prompt
  ↑
merge, rescue, runtime, self_improvement
```

## Validation (owning goal)

```bash
python -m pytest \
  test/api/test_agent_supervisor_control_conformance_v2.py \
  test/api/test_agent_supervisor_lifecycle_orchestrator.py \
  -q
```

After full caller cutover, also assert absence of old flat imports, e.g.:

```bash
rg -n 'agent_supervisor\.(control_cli|control_contracts|control_plane|lifecycle_orchestrator|execution_permit|authorization_logic)\b' \
  --glob '!**/__pycache__/**' --glob '!docs/architecture/asref/**' --glob '!**/control/**'
```

And absence of flat files:

```bash
# Expected: no matches after cutover
ls ipfs_accelerate_py/agent_supervisor/control_*.py 2>/dev/null
```

## Status

- Package scaffold (`__init__.py`, this README): present (ASREF-004)
- Control-owned modules under `control/` with package-relative imports: present
- CLI entry targets documented in `CONTROL_CLI_ENTRY_TARGETS` / table above
- Temporary flat copies of control-owned modules may still exist at the former
  root paths when this package lands under a narrow `control/`-only edit scope
  (same dual-copy pattern as `core/`). Prefer `control.<module>` for all new
  code. A follow-on import-rewrite pass must update callers (`cli.py`, MCP tools,
  tests, sibling packages), then delete the flat copies **without** reintroducing
  stubs.
