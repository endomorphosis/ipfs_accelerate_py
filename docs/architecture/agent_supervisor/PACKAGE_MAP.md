# Agent Supervisor package map

Ownership map for `ipfs_accelerate_py.agent_supervisor` on current `main`.

**Audience:** developers placing code and reviewers checking the dependency DAG.

Related: [Developer guide](DEVELOPER_GUIDE.md) · [Doc hub](README.md) ·
[Package README](../../../ipfs_accelerate_py/agent_supervisor/README.md) ·
[Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md).

---

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
  ↑
entrypoints (prompt-first composition facade)
```

### Rules

| Rule | Detail |
| --- | --- |
| Acyclic | No package cycles |
| Bottom-up only | Higher layers may import lower; not the reverse |
| No daemon in core | `core` must not import `todo_daemon`, `runtime`, `merge`, `rescue` |
| No upward entrypoint import | No lower domain package may import `entrypoints`; only the product/transport edge imports it |
| Cold composition | `entrypoints` may compose lower packages lazily, but import/discovery may load only provider-free contracts |
| No long-lived flat stubs | Do not reintroduce retired root-level re-export modules |
| Domain imports | `from ipfs_accelerate_py.agent_supervisor.<pkg>.<mod> import …` |

---

## Domain packages

| Package | Purpose | Typical contents | Layer |
| --- | --- | --- | --- |
| `core/` | Shared foundation | Conflict graph, wrappers, external completion | Foundation |
| `control/` | Transport-neutral control plane | Ops catalog, service, CLI, permits, authz | Foundation |
| `task_sources/` | Task projection & storage | Markdown/DuckDB boards, queues, indexes | Foundation |
| `context/` | Decision context | Compiler, contracts, decision runtime | Foundation |
| `prompt/` | Prompt workflows | Scanner, plan admission, bootstrap/rescue | Mid |
| `analysis/` | Code analysis pipeline | AST, cache, retrieval, consensus | Mid |
| `proof/` | Formal assurance | Provers, attestation, proof cache | Mid |
| `objectives/` | Intent lifecycle | Heap, tracker, backlog, goal quality | Mid |
| `planning/` | Plan compile/validate | Adaptive + formal planning, metrics | Mid |
| `validation/` | Pre-merge gates | Proposal validation, schedulers | Mid |
| `merge/` | Land completed work | Queue/train/resolver, locks, leases | Ops |
| `rescue/` | Failure recovery | Orchestrator, watchdog hooks | Ops |
| `runtime/` | Multi-lane execution | Runners, event log, CAS, schedulers | Ops |
| `self_improvement/` | Meta-improvement | Epochs, refill, v2 contracts | Ops |
| `integrations/` | Optional bridges | LLM merge fallback, Goose, datasets | Edge |
| `todo_daemon/` | Executable daemons | Implementation/supervisor loops, git helpers | Edge |
| `entrypoints/` | Prompt-first composition | Closed request/result contracts; lazy Python, CLI, MCP and MCP++ facades | Facade |

Semantic pages: [packages/README.md](packages/README.md).  
Code-tree READMEs: `ipfs_accelerate_py/agent_supervisor/<package>/README.md`.

---

## Where new work goes

| If you are adding… | Put it in… |
| --- | --- |
| A control operation or CLI binding | `control/` |
| A taskboard parser or queue backend | `task_sources/` |
| A prover, attestation, or proof-cache API | `proof/` |
| Context capsule / decision-core fields | `context/` |
| Formal plan IR / conformance | `planning/` |
| Proposal or validation policy | `validation/` |
| Merge/lease/checkout behavior | `merge/` |
| Multi-lane runner / event log | `runtime/` |
| Implementation daemon behavior | `todo_daemon/` |
| Optional external tool bridge | `integrations/` |
| Self-improvement epoch logic | `self_improvement/` |
| Prompt-only Python/CLI/MCP facade or inference receipt | `entrypoints/` |
| A new *program* (board + objectives only) | `docs/architecture/` boards; **code still in domain packages** |

---

## Public API stability

### Control surface

- **v1 compatibility:** existing operation names, request/result records, CLI/MCP tool names.
- **v2 stable exports:** `AGENT_SUPERVISOR_PUBLIC_API_EXPORTS` (aliases:
  `AGENT_SUPERVISOR_V2_STABLE_EXPORTS`, `V2_STABLE_EXPORTS`). Closed set — review
  before expanding.
- Import success is **not** a capability signal; run discovery then capability probes.

### Domain layout inventories (semantic)

Prefer product-role names:

| Constant family | Role |
| --- | --- |
| `AGENT_SUPERVISOR_DOMAIN_PACKAGES` | Full package list |
| `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE` | Stem → package (landed) |
| `AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE` | Stem → package (planned / scan only) |
| `AGENT_SUPERVISOR_*_PACKAGES` | Packages by role (`CORE_`, `CONTROL_`, …) |
| `AGENT_SUPERVISOR_*_STEMS` | Stem inventories |
| `AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS` / `_OPERATIONS_…` | Layout evidence clusters |
| `AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES` | Goal-id string → packages |
| `AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_*` | Cutover identity |

Board-prefix spellings (`AGENT_SUPERVISOR_G020_*`,
`AGENT_SUPERVISOR_EVIDENCE_CLUSTER_*`, `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS`)
are **deprecated aliases**. Board IDs remain string **values** for scanners.

```python
# Preferred
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)

# Avoid (retired flat paths after domain layout cutover)
# from ipfs_accelerate_py.agent_supervisor.control_plane import ...
```

---

## Import style

| Pattern | When |
| --- | --- |
| `from ipfs_accelerate_py.agent_supervisor.<pkg>.<mod> import X` | Default for app & library code |
| `from .<mod> import X` | Inside the same domain package |
| Package-root re-export | Only for reviewed public control/layout symbols |

Historical flat stems may resolve via package-root aliasing for compatibility;
do not add new callers on those paths.

### Entrypoint composition surface

`agent_supervisor.entrypoints` is the sole highest-layer composition package.
Its eager public inventory is `ENTRYPOINT_CONTRACT_EXPORTS`; package exports
preserve identity with `entrypoints.contracts`. Runtime/service facades must be
lazy and listed in `ENTRYPOINT_LAZY_FACADE_EXPORTS`.

The boundary is intentionally one-way:

```text
product CLI / Python / MCP / MCP++
                |
                v
           entrypoints
                |
                v
      existing domain packages
```

An `entrypoints` import cannot scan a repository, load an implementation
provider, import a daemon/runtime factory, open DuckDB/Parquet/IPLD/IPFS, or
start a process. Lower domain packages cannot import upward. DuckDB remains
the mutable single-writer coordination shard; Parquet/IPLD/CAR/IPFS remain
immutable replication formats and do not convey authority.

---

## Historical layout cutover

Domain-layout program evidence tables (ASREF-G0xx package goals) live in
[LAYOUT_CUTOVER_EVIDENCE.md](LAYOUT_CUTOVER_EVIDENCE.md). That document is
**historical / scanner-oriented**, not the day-to-day developer map.

---


---

## Flat-module residual land (2026-08-06)

Package root no longer hosts production modules other than `__init__.py`.
Previously flat stems (program graph/forest, code/CVE contracts, provider
execution, worktree lifecycle, finding task sources, etc.) live under their
domain packages. The authoritative stem → package map is
`AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE` in the package root `__init__.py`.
Historical flat import paths resolve via package-root aliasing only.

## Related

- [Developer guide](DEVELOPER_GUIDE.md)
- [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md)
- [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Contributor guide](FOR_CONTRIBUTORS.md)
- [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md)
