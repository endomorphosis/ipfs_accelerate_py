# agent_supervisor.entrypoints

**Code:** `ipfs_accelerate_py/agent_supervisor/entrypoints/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/entrypoints/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md) · [Prompt-first runtime guide](../PROMPT_FIRST_RUNTIME.md)

## Purpose

Highest-layer **prompt-first composition** package: provider-free request/result
contracts, target and profile inference, transient prompt brokering, durable run
registry, plan lint, steering classification, and verified IPLD admission.

Domain behavior (control ops, prompt planning, daemons, merge, rescue) stays in
lower packages. This package **composes** them; it does not re-implement the
control plane.

**Status today:** leaf resolvers, broker, registry, contracts, lint/explain,
steering contracts, and verified IPLD helpers are **landed** / **implemented**.
The high-level `Supervisor.open()` / product CLI / MCP lifecycle is **planned**
/ **not yet** exported (`ENTRYPOINT_LAZY_FACADE_EXPORTS` is empty).

## When to use this package

- You are adding prompt-only inference receipts, run-handle storage, or steering
  classification.  
- You are wiring a future Python/CLI/MCP facade that must stay cold on import.  
- You need to keep prompt bodies, credentials, and UCANs out of durable records.

Do **not** put ordinary control operations, taskboard parsers, or daemon loops
here—use `control/`, `task_sources/`, and `todo_daemon/` respectively.

## Public modules

| Module | Role | Status |
| --- | --- | --- |
| `contracts` | Closed invocation, receipt, profile, launch, run, result schemas | **Landed** (eager package export) |
| `target_resolver` | Repository, checkout, scope, dirty-tree resolution | **Landed** |
| `state_resolver` | Platform state root, namespace, run-candidate classification | **Landed** |
| `objective_resolver` | Objective, plan, task-source, output binding | **Landed** |
| `authority_resolver` | Principal, policy, authority source, effect ceiling | **Landed** |
| `capability_resolver` | Provider/resource/lane/validation/topology evidence | **Landed** |
| `profile_resolver` | Precedence merge → `TargetResolutionReceipt` + profile | **Landed** |
| `inference_explain` | Body-free human/JSON provenance for receipts | **Landed** |
| `plan_lint` | Read-only goal/task/profile lint | **Landed** |
| `prompt_broker` | Transient, capability-protected prompt bodies | **Landed** |
| `run_registry` | Durable handles, CAS heads, restart reconstruction | **Landed** |
| `steering_contracts` | Steering request/event/result + closed classification | **Landed** (apply path **planned**) |
| `verified_ipld_backend` | Strict CIDv1/IPLD admission and rehash verification | **Landed** |
| Lazy `Supervisor` / CLI / MCP facades | Product lifecycle | **Planned** / **not yet** |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    SupervisorInvocationRequest,
    TargetResolutionReceipt,
    RunHandle,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.prompt_broker import (
    PromptBodyBroker,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import (
    RunRegistry,
)
```

Package `__init__` re-exports only `ENTRYPOINT_CONTRACT_EXPORTS` today. Do not
assume `Supervisor` is importable from the package root until a facade delivery
task fills `ENTRYPOINT_LAZY_FACADE_EXPORTS`.

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Product CLI, Python embeddings, MCP/MCP++ adapters, future facades only |
| **Outbound** | May **lazily** compose `control`, `prompt`, `objectives`, `planning`, `runtime`, `rescue`, `todo_daemon`, `integrations`, and foundation packages |
| **Forbidden** | Any lower domain package importing `entrypoints` upward; eager provider/daemon/import side effects on package import |

Cold import may load only provider-free contracts and package metadata—no
repository scans, DuckDB/Parquet/IPLD opens, credential resolution, or process
starts.

## Durable data rules

| Store | Allowed | Forbidden |
| --- | --- | --- |
| Contracts / receipts / run registry | CIDs, opaque refs, handles, decision provenance | Prompt bodies, capability tokens, raw UCANs, API keys |
| Prompt broker | Process memory or encrypted artifacts under capability | Writing body or token into ordinary state/logs |
| Launch profile | Env **names**, credential **handles** | Embedding secret values in argv or profile CID payload |

See [PROMPT_FIRST_RUNTIME.md](../PROMPT_FIRST_RUNTIME.md) for the full durable
versus transient table, resolver precedence, CAS/restart semantics, and the
landed-versus-planned product matrix.

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).  
2. New facade symbols must be **lazy** and listed in
   `ENTRYPOINT_LAZY_FACADE_EXPORTS` with cold-import tests.  
3. Use **semantic** symbol names; do not name public APIs after board prefixes
   (ASE-… IDs are historical evidence only).  
4. Update this page and the runtime guide when modules move from planned to
   landed—**source and tests** are the status authority, not the ASE board.  
5. Add focused tests under `test/api/` next to existing
   `test_agent_supervisor_*_resolver.py` / broker / registry suites.

## Program evidence (optional)

The prompt-only entrypoints program
([plan](../../AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md),
[objectives](../../agent_supervisor_prompt_only_entrypoints.objectives.md),
[board](../../agent_supervisor_prompt_only_entrypoints.todo.md)) funds this
package. Board ticket status may lag the tree; product docs should cite modules
and tests as current behavior.
