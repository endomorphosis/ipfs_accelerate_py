# agent_supervisor

**Status:** Current

**Owner:** agent-supervisor maintainers

**Audience:** Developers embedding or extending the supervisor

**Sources:** package modules under this directory; package `__init__.py` export
inventories; supervisor console scripts in `pyproject.toml`;
`test/api/test_agent_supervisor_*.py`

**Last-verified:** 2026-08-06 — domain-layout residual flat-module land; root holds only `__init__.py` + domain packages; `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE` covers production stems
test paths rechecked

**Freshness triggers:** domain-module moves, public-export or console-script
changes, and supervisor test relocation

Control plane for **objective-driven, evidence-bounded software work**.

Models propose plans and edits. Deterministic validation, leases, allowlists,
typed receipts, and Git isolation decide whether work may advance. This package
is the product surface: domain modules, public contracts, and runnable daemons.

| If you want… | Go to… |
| --- | --- |
| Mental model | [Design philosophy](../../docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md) |
| Hands-on developer guide | [DEVELOPER_GUIDE.md](../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md) |
| Architecture (deep) | [AGENT_SUPERVISOR_ARCHITECTURE.md](../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md) |
| Operator / CLI / MCP | [Operator guide](../../docs/guides/AGENT_SUPERVISOR_GUIDE.md) |
| Package ownership map | [PACKAGE_MAP.md](../../docs/architecture/agent_supervisor/PACKAGE_MAP.md) |
| Doc hub | [docs/architecture/agent_supervisor/](../../docs/architecture/agent_supervisor/README.md) |
| Implementation-agent checklist | [FOR_AGENTS.md](../../docs/architecture/agent_supervisor/FOR_AGENTS.md) |

## What the system is

```text
  Objectives (durable intent)
        │  project / refill
        ▼
  Taskboard (schedulable work)
        │  claim + worktree
        ▼
  Implementation lane (model proposes)
        │  validate + evidence
        ▼
  Merge train / recovery
        │
        ▼
  Event log · receipts · metrics
```

Seven pillars (short form):

1. **Objectives are durable intent; todos are projections.**
2. **Models propose; policies admit.**
3. **Evidence is typed** (tests ≠ solver candidates ≠ kernel proofs).
4. **Isolation by default** (worktrees, leases, protected paths).
5. **One contract, three transports** (Python / CLI / MCP).
6. **Domain packages encode ownership** (acyclic DAG).
7. **Programs layer on the control plane** (boards ≠ package trees).

## Quick start

### Preferred imports

Import **domain packages** for application code. Use the package root only for
the reviewed control surface and layout inventories.

```python
# Control plane (transport-neutral)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    Operation,
    OperationRequest,
    OperationResult,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)

# Domain modules (examples)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ConflictGraph
from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import (
    GoalCompletionDecision,
    evaluate_goal_completion,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)

# Reviewed package-root inventories (semantic names)
from ipfs_accelerate_py.agent_supervisor import (
    AGENT_SUPERVISOR_PUBLIC_API_EXPORTS,
    AGENT_SUPERVISOR_DOMAIN_PACKAGES,
    AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE,
    AGENT_SUPERVISOR_CORE_PACKAGES,
    AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS,
)
```

**Avoid** retired flat imports such as
`from ipfs_accelerate_py.agent_supervisor.control_plane import …` after domain
layout cutover. Root `__getattr__` may still resolve some historical stems for
compatibility; new code must not depend on that.

### Control service sketch

```python
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    Operation,
    OperationRequest,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)

service = SupervisorControlService(...)  # wire backend + allowlists per guide
# Always discover / probe capabilities before assuming a provider is live.
manifest = service  # use discovery helpers — see operator guide
```

Cold-import of `ipfs_accelerate_py.agent_supervisor` must not start processes,
touch the network, or load optional provers. Provider adapters resolve **lazily**.

## Architecture at a glance

### Domain package DAG (allowed direction)

```text
core
  ↑
control · task_sources · context · analysis · proof
  ↑
objectives · planning · validation · prompt
  ↑
merge · rescue · runtime · self_improvement
  ↑
todo_daemon · integrations
```

**Forbidden examples:** `core` → `todo_daemon`; `proof` ↔ `runtime` cycles;
long-lived re-export stubs at retired flat paths.

### Package roles

| Package | Role |
| --- | --- |
| `core/` | Shared foundation: conflict graph, wrappers, external completion, multiformats identity, layout evidence |
| `control/` | Transport-neutral ops, CLI binding, permits, authorization |
| `task_sources/` | Markdown/DuckDB boards, queues, indexes, task identity, finding task source, symbolic finding refill |
| `context/` | Context compiler, decision runtime, obligation capsules |
| `prompt/` | Prompt workflow, scanning, plan admission, workflow benchmark/rollout |
| `analysis/` | AST/index/cache/retrieval/consensus pipeline, program graph, repository forest/corpus |
| `proof/` | Formal verification, provers, attestation, code/CVE/security contracts, MCP++ witnesses |
| `objectives/` | Objective heap, tracker, backlog, goal quality/completion |
| `planning/` | Adaptive + formal planning (compile/validate/metrics), task proposal router |
| `validation/` | Proposal validation, schedulers, pre-merge gates, failure review, evidence output scope |
| `merge/` | Merge queue/train/resolver, locks, leases, git hygiene, worktree lifecycle |
| `rescue/` | Failure policy, rescue orchestrator, recovery/watchdog |
| `runtime/` | Multi-lane runners, event log, CAS, schedulers, provider execution/usage, durable process, release evidence |
| `self_improvement/` | Self-improvement epochs, v2 contracts, refill |
| `integrations/` | Optional bridges (LLM merge fallback, Goose, datasets program graph/analysis) |
| `todo_daemon/` | Implementation / supervisor daemons, git helpers, implementation timeout |
| `entrypoints/` | Prompt-first composition facade (contracts + lazy transport facades) |
| `contract_analysis/` | Bounded contract-analysis execution profiles and cache adapters |

Per-package detail: each directory’s `README.md`, plus semantic pages under
[`docs/architecture/agent_supervisor/packages/`](../../docs/architecture/agent_supervisor/packages/README.md).

### Runtime shape

| Component | Owns |
| --- | --- |
| `SupervisorControlService` | Authoritative mutations and discovery |
| Objective / task sources | Intent → schedulable tasks |
| Implementation daemon | Claim task, run agent in worktree, validate |
| Multi-supervisor runtime | Lanes, heartbeats, shared merge target |
| Merge train | Land completed work onto integrate/main |
| Rescue / watchdog | Quarantine and recovery paths |
| Event log + artifacts | Audit, resume, metrics |


## Domain layout status (flat-module cutover)

The package root is **domain packages only** plus `__init__.py`. Residual
production modules that previously accumulated as flat root files have been
moved into permanent owners:

| Cluster | Owner package | Example stems |
| --- | --- | --- |
| Program graph / forest / corpus | `analysis/` | `program_graph`, `repository_forest`, `program_ast_adapters` |
| Code/CVE/security contracts | `proof/` | `code_contract_*`, `contract_*`, `cve_*`, `security_contract_analysis` |
| Provider execution & usage | `runtime/` | `provider_execution`, `provider_usage`, `grok_cli_runner`, `durable_process` |
| Worktree fencing | `merge/` | `worktree_lifecycle` |
| Finding → task projection | `task_sources/` | `finding_task_source`, `symbolic_finding_refill` |
| Datasets program bridges | `integrations/` | `ipfs_datasets_program_*_provider` |
| Prompt workflow gates | `prompt/` | `prompt_workflow_benchmark`, `prompt_workflow_rollout` |
| Layout evidence / CID bridge | `core/` | `asref_layout_evidence`, `multiformats_identity` |

**Preferred imports** always use the domain package path:

```python
from ipfs_accelerate_py.agent_supervisor.analysis.program_graph import ProgramGraph
from ipfs_accelerate_py.agent_supervisor.proof.code_contract_prover import CodeContractProver
from ipfs_accelerate_py.agent_supervisor.runtime.provider_execution import ProviderExecutionGateway
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
```

Historical flat submodule names (`import ipfs_accelerate_py.agent_supervisor.program_graph`)
continue to resolve through package-root `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE`
aliasing for compatibility. **Do not add new callers on flat paths.** Do not
reintroduce long-lived re-export stub files at the package root.

## Public API surface

### Root exports (what is intentional)

The package root re-exports a **reviewed** set only:

1. **Control contracts & service** — `Operation*`, `SupervisorControlService`, …
2. **Generation-2 stable manifest** — `AGENT_SUPERVISOR_PUBLIC_API_EXPORTS`
   (alias: `AGENT_SUPERVISOR_V2_STABLE_EXPORTS`)
3. **Domain layout inventories** — semantic package/stem maps (below)
4. Selected objective / proof / planning helpers used by transports

Everything else is domain-internal or lazy. **Import success ≠ capability.**
Run discovery, then capability probes, before routing work to a provider.

### Layout inventories (semantic names)

| Constant | Meaning |
| --- | --- |
| `AGENT_SUPERVISOR_DOMAIN_PACKAGES` | Ordered domain package names |
| `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE` | Landed stem → owning package |
| `AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE` | Planned stem map (scan evidence; not import aliases) |
| `AGENT_SUPERVISOR_CORE_PACKAGES` … `_INTEGRATIONS_DAEMON_PACKAGES` | Packages by product role |
| `AGENT_SUPERVISOR_CORE_STEMS` / `_CONTROL_STEMS` / `_TASK_SOURCES_STEMS` | Landed stem inventories |
| `AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS` | Foundation layout goal-id strings |
| `AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS` | Operations layout goal-id strings |
| `AGENT_SUPERVISOR_DOMAIN_LAYOUT_GOAL_IDS` | Full layout evidence goal-id set |
| `AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES` | Goal-id string → package tuple |
| `AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_*` | Domain-layout cutover identity |

Board-prefix spellings (`AGENT_SUPERVISOR_G020_PACKAGES`,
`AGENT_SUPERVISOR_EVIDENCE_CLUSTER_*`, `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS`, …)
remain **compatibility aliases** only. Board IDs (`"ASREF-G020"`, …) stay as
**string values** for scanners and receipts—not as Python API names.

Full preferred import example:

```python
from ipfs_accelerate_py.agent_supervisor import (
    Operation,
    OperationRequest,
    OperationResult,
    SupervisorControlService,
    AGENT_SUPERVISOR_PUBLIC_API_EXPORTS,
    AGENT_SUPERVISOR_DOMAIN_PACKAGES,
    AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE,
    AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE,
    AGENT_SUPERVISOR_DOMAIN_LAYOUT_GOAL_IDS,
    AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES,
    AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS,
    AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS,
    AGENT_SUPERVISOR_CORE_PACKAGES,
    AGENT_SUPERVISOR_CONTROL_PACKAGES,
    AGENT_SUPERVISOR_TASK_SOURCES_PACKAGES,
    AGENT_SUPERVISOR_CONTEXT_PROMPT_PACKAGES,
    AGENT_SUPERVISOR_ANALYSIS_PROOF_PACKAGES,
    AGENT_SUPERVISOR_OPERATIONS_PACKAGES,
    AGENT_SUPERVISOR_INTEGRATIONS_DAEMON_PACKAGES,
    AGENT_SUPERVISOR_CORE_STEMS,
    AGENT_SUPERVISOR_CONTROL_STEMS,
    AGENT_SUPERVISOR_TASK_SOURCES_STEMS,
    AGENT_SUPERVISOR_CONTEXT_PROMPT_PLANNED_MODULES,
    AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_GOAL_ID,
    AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_TASK_ID,
    AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_PACKET_TASK_IDS,
)
```

## How work flows (implementation loop)

1. **Objectives** describe durable goals and evidence expectations.
2. **Taskboards** project those goals into drainable tasks (`## PREFIX-###`).
3. A **lane** claims a ready task, creates/reuses a worktree, and runs an
   implementation provider (Grok, Codex, …).
4. **Validation** runs the task’s declared commands; proposal validation and
   protected-path policy gate completion.
5. **Merge** lands the branch onto the shared target when policy allows.
6. **Events / receipts** record what happened for resume, rescue, and audit.

Operators launch multi-lane supervisors via
`scripts/ops/agent_supervisor/` (see the [operator guide](../../docs/guides/AGENT_SUPERVISOR_GUIDE.md)).

## Extending the system

| You want to… | Put code in… | Also update… |
| --- | --- | --- |
| Add a control operation | `control/` contracts + plane | Discovery manifests, CLI/MCP parity, tests |
| Parse a new board format | `task_sources/` | Package README, parser tests |
| Add a prover / attestation | `proof/` | Capability probe paths, tests |
| Change merge/lease behavior | `merge/` | Runtime + daemon integration tests |
| Change implementation loop | `todo_daemon/` | Failure review / rescue hooks as needed |
| Add a *program* (board only) | `docs/architecture/*.todo.md` | [PROGRAMS.md](../../docs/architecture/agent_supervisor/PROGRAMS.md); **code still lands in domain packages** |

Rules of thumb:

- Prefer **semantic names** (`code_proof_*`) over board prefixes (`cbp_*` only).
- Do not encode taskboard IDs into public API names.
- Keep the dependency DAG acyclic.
- Do not rewrite protected paths (boards, sealed plans) unless the task owns them.

Longer walkthrough: [DEVELOPER_GUIDE.md](../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md).

## Testing

Focused suites live under `test/api/` (and package-local tests where present).

```bash
# Layout + public API evidence
python -m pytest test/api/test_agent_supervisor_asref_layout_evidence.py \
  test/api/test_agent_supervisor_semantic_layout_exports.py -q

# Control / contracts
python -m pytest test/api/test_agent_supervisor_control_conformance_v2.py \
  test/api/test_agent_supervisor_control_plane.py -q
```

When changing a domain package, run the nearest module tests plus any
validation/daemon suites that import your surface.

## Console entry points

Post-domain-layout targets (see `pyproject.toml` / `setup.py` for the live list):

| Surface | Module path |
| --- | --- |
| Unified agent CLI | `control.control_cli` |
| Objective daemon | `objectives.objective_daemon` |
| Implementation daemon | `todo_daemon.implementation_daemon` |
| Merge resolver | `merge.merge_resolver` |

## Documentation map

| Doc | Audience |
| --- | --- |
| This README | Developers landing in the package |
| [DEVELOPER_GUIDE.md](../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md) | Contributors extending the system |
| [PACKAGE_MAP.md](../../docs/architecture/agent_supervisor/PACKAGE_MAP.md) | Ownership & placement |
| [Architecture](../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md) | Deep subsystem contracts |
| [Philosophy](../../docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md) | Design principles |
| [Operator guide](../../docs/guides/AGENT_SUPERVISOR_GUIDE.md) | Run, discover, authorize, recover |
| [FOR_CONTRIBUTORS.md](../../docs/architecture/agent_supervisor/FOR_CONTRIBUTORS.md) | PR / doc checklist |
| [FOR_AGENTS.md](../../docs/architecture/agent_supervisor/FOR_AGENTS.md) | Implementation-agent invariants |
| [PROGRAMS.md](../../docs/architecture/agent_supervisor/PROGRAMS.md) | Board prefixes ↔ product names |
| [LAYOUT_CUTOVER_EVIDENCE.md](../../docs/architecture/agent_supervisor/LAYOUT_CUTOVER_EVIDENCE.md) | Historical domain-layout cutover evidence |

## Related plans (program docs)

Sealed or long-form plans under `docs/architecture/` (often protected during
implementation) include module refactor, self-improvement, codebase-proof,
formal planning, and prompt/usage rollouts. Prefer **product docs above** for
day-to-day development; open plans only when you own that program’s work.

## Historical layout program (ASREF-G090)

Domain packages were landed under the **domain-layout / ASREF** program. The
public cutover goal **ASREF-G090** (packet tasks ASREF-012 / ASREF-013 /
ASREF-014) published this package map, root hygiene, and no-old-import gate.

Day-to-day developers should use **semantic** layout constants and the docs
linked above. Historical evidence tables (package goals ASREF-G020–G080,
stem inventories, cutover witnesses) remain in
[LAYOUT_CUTOVER_EVIDENCE.md](../../docs/architecture/agent_supervisor/LAYOUT_CUTOVER_EVIDENCE.md).

---

*Root hygiene note:* nested product trees and git submodules next to this package
are intentional. See [NESTED_PACKAGES.md](../../docs/NESTED_PACKAGES.md). Do not
move `agent_supervisor` packages into nested products as part of layout work.
