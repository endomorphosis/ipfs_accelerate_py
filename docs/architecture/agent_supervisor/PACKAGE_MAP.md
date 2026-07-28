# Agent Supervisor package map

This map describes the **domain package layout** for
`ipfs_accelerate_py.agent_supervisor` as landed on current `main` (domain layout
program). Feature branches may still carry a transitional flat tree; treat this
map as the target and merge destination.

## Dependency DAG (allowed direction)

```text
core
  ↑
control, task_sources, context, analysis, proof
  ↑
objectives, planning, validation, prompt
  ↑
merge, rescue, runtime, self_improvement
  ↑
todo_daemon (implementation), integrations
```

**Forbidden:** `core` importing `todo_daemon`; proof/runtime cycles;
reintroducing long-lived flat re-export stubs at retired paths.

## Domain packages

| Package | Purpose | Typical contents |
| --- | --- | --- |
| `core/` | Shared foundation: identity, conflict graph, wrappers, external completion | Bottom of the DAG |
| `control/` | Transport-neutral control plane, CLI/MCP contracts, lifecycle, permits | `SupervisorControlService`, operations catalog |
| `task_sources/` | Markdown/DuckDB task sources, taskboards, queues, indexes | Board parse, queues |
| `context/` | Context compiler/contracts, decision runtime | Obligation-first capsules |
| `prompt/` | Prompt workflow, scanning, plan admission | Bootstrap / rescue prompts |
| `analysis/` | Analysis pipeline, AST, cache, retrieval, consensus | Integrated analysis |
| `proof/` | Formal verification, provers, attestation, proof cache | Codebase-proof, kernels |
| `objectives/` | Objective heap parse, daemon, tracker, janitor, goal quality | Intent reconciliation |
| `planning/` | Adaptive and formal plan compile/validate (non-daemon) | Plan IR, conformance |
| `validation/` | Proposal validation, validation scheduler/runtime/commands | Pre-merge gates |
| `merge/` | Merge queue/train/resolver, checkout locks, leases, git hygiene | Lane merge |
| `rescue/` | Rescue planner/orchestrator, recovery, watchdog hooks | Failure recovery |
| `runtime/` | Multi-supervisor runners, event log, CAS, schedulers | Lane orchestration |
| `self_improvement/` | Epoch contracts, refill, v2 efficiency/token models | Self-improvement program |
| `integrations/` | LLM merge fallback, goose/meta adapters, dataset providers | Optional providers |
| `todo_daemon/` | Executable implementation/supervisor daemons and git worktree helpers | Drain boards |

Root `__init__.py` re-exports only intentional public symbols. Prefer **domain
imports** for new code:

```python
# Preferred
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)

# Avoid (retired flat paths after domain layout cutover)
# from ipfs_accelerate_py.agent_supervisor.control_plane import ...
```

## Where new work goes

| If you are adding… | Put it in… |
| --- | --- |
| A new control operation or CLI binding | `control/` |
| A new taskboard parser or queue backend | `task_sources/` |
| A prover, attestation, or proof-cache API | `proof/` |
| Context capsule / decision-core fields | `context/` |
| Implementation daemon behavior | `todo_daemon/` (and runners in `runtime/` if multi-lane) |
| Self-improvement epoch logic | `self_improvement/` |
| A new *program* (board + objectives only) | `docs/architecture/` boards + bundles; code still lands in domain packages |

## Public API stability

- **v1 compatibility surface:** existing operation names, request/result records, CLI/MCP tool names.
- **v2 stable exports:** package-root manifests (`AGENT_SUPERVISOR_PUBLIC_API_EXPORTS` (alias `AGENT_SUPERVISOR_V2_STABLE_EXPORTS`) and related layout constants) for generation-2 contracts.
- **Domain layout constants (semantic):** package-root names prefer product roles, not board prefixes:
  - `AGENT_SUPERVISOR_DOMAIN_PACKAGES`, `AGENT_SUPERVISOR_CORE_PACKAGES`, `AGENT_SUPERVISOR_CONTROL_PACKAGES`, …
  - `AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS` / `AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS`
  - `AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES`, `AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_*`
  - Board-prefix spellings (`AGENT_SUPERVISOR_G020_*`, `AGENT_SUPERVISOR_EVIDENCE_CLUSTER_*`) remain as **deprecated aliases**.
  - Board IDs (`ASREF-G0xx`, `ASREF-0xx`) stay as **string values** for scanners and receipts.
- Import success is **not** a capability signal; run discovery then capability probes.

Per-package semantic READMEs (purpose, modules, dependency rules):
[packages/README.md](packages/README.md).

Details: [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md),
[Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md),
[Contributor guide](FOR_CONTRIBUTORS.md).
