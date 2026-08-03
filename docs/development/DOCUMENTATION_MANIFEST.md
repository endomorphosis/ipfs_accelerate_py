# Documentation manifest

**Status:** Current
**Owner:** documentation-governance
**Audience:** maintainers, documentation authors, implementation agents, and
readers who need to know which documents are maintained versus archival
**Scope:** Status classification of the documentation tree with owner, status,
primary source anchors, last-verified baseline, and supersession notes. Lists
**maintained** Current/Reference surfaces explicitly; summarizes Plan,
Historical, Generated, and Vendored bulk without claiming every file is
reviewed as Current.
**Non-goals:** Replacing leaf guide content; mass-moving or deleting archives;
editing top-level `docs/README.md` / `docs/INDEX.md` (DOC-028); inventing CI
gates that do not exist; resolving code-owned version/CLI contradictions in
prose.
**Sources:** [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md);
[DOCUMENTATION_CURRENT_STATE.md](DOCUMENTATION_CURRENT_STATE.md);
[DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md);
[architecture/README.md](../architecture/README.md);
[architecture/GLOSSARY.md](../architecture/GLOSSARY.md);
maintained architecture guides and ADRs; `docs/` tree layout.
**Last-verified:** 2026-08-03; classifications aligned with lifecycle policy
and landed documentation-refresh leaf guides (DOC-002 through DOC-027 outputs).
**Interface:** DocumentationManifest@1
**Freshness triggers:** new Current guide; ADR accept/supersede; package or
entrypoint renames; archive moves; closeout of DOC-028 navigation.

This manifest is the **DocumentationManifest@1** inventory for the
documentation-refresh program. It does **not** claim that every Markdown file
under `docs/` is maintained. Unlisted files default to the lifecycle fail-closed
rules (often **Plan**, **Historical**, **Generated**, or **Vendored**).

Authority order (summary): package metadata and live code → executable help →
tests → **Current** docs → **Reference** docs → **Plan** → **Historical** /
**Generated** / **Vendored**. Full matrix:
[DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md).

---

## Status vocabulary (closed)

| Status | Normative for behavior? | Use in this manifest |
| --- | --- | --- |
| **Current** | Yes (subject to code/test override) | Explicitly maintained operational or policy narrative |
| **Reference** | Partial | Glossary, maps, Accepted ADRs, stable orientation |
| **Plan** | No | Design, rollout, objectives, taskboards |
| **Historical** | No | Point-in-time reports, archives, completion summaries |
| **Generated** | No (unless regenerated) | Exports, derived dumps |
| **Vendored** | External wins | Nested products, third-party trees |

Readers must never be routed to a **Plan** or **Historical** document as if it
were **Current** without an explicit status label.

---

## How to read rows

| Column | Meaning |
| --- | --- |
| Path | Repo-relative documentation path |
| Status | Closed vocabulary above |
| Owner | Role responsible for freshness |
| Sources | Primary code or policy anchors |
| Last-verified | Baseline date or note; re-check after source churn |
| Supersession | What replaces this page, or “—” if still authoritative in class |

---

## 1. Current — product and development policy

| Path | Status | Owner | Sources | Last-verified | Supersession |
| --- | --- | --- | --- | --- | --- |
| [docs/api/overview.md](../api/overview.md) | Current | package maintainers | `pyproject.toml`, `__init__.py`, CLI modules | 2026-08-03 | — |
| [docs/architecture/overview.md](../architecture/overview.md) | Current | architecture | package layout, `cli_entry`, MCP/supervisor paths | 2026-08-03 | — |
| [docs/architecture/SYSTEM_CONTEXT.md](../architecture/SYSTEM_CONTEXT.md) | Current | architecture | same + supervisor domain packages | 2026-08-03 | — |
| [docs/architecture/INFERENCE_RUNTIME.md](../architecture/INFERENCE_RUNTIME.md) | Current | inference | routers, backends, workers | 2026-08-03 | — |
| [docs/architecture/MODEL_SERVICE_ROUTING.md](../architecture/MODEL_SERVICE_ROUTING.md) | Current | inference / routing | `model_catalog/`, `endpoint_usage/`, routers | 2026-08-03 | Supersedes plan prose in endpoint-usage plan for *landed* behavior |
| [docs/architecture/AI_SERVICE_CATALOG.md](../architecture/AI_SERVICE_CATALOG.md) | Current | catalog | `model_catalog/` | 2026-08-03 | — |
| [docs/architecture/MCP_RUNTIME.md](../architecture/MCP_RUNTIME.md) | Current | mcp | `mcp_server/`, `mcp/`, `mcplusplus_module/` | 2026-08-03 | Supersedes unification-plan claims for current runtime shape |
| [docs/architecture/DISTRIBUTED_RUNTIME.md](../architecture/DISTRIBUTED_RUNTIME.md) | Current | storage / p2p | `ipfs_backend_router.py`, `p2p_tasks/`, multiformats | 2026-08-03 | — |
| [docs/architecture/INTEGRATION_BOUNDARIES.md](../architecture/INTEGRATION_BOUNDARIES.md) | Current | integrations | `.gitmodules`, adapters | 2026-08-03 | — |
| [docs/architecture/GUIDE_CONVENTIONS.md](../architecture/GUIDE_CONVENTIONS.md) | Current | documentation-governance | guide contract | 2026-08-03 | — |
| [docs/development/DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md) | Current | documentation-governance | lifecycle policy | 2026-08-03 | — |
| [docs/development/DOCUMENTATION_CURRENT_STATE.md](DOCUMENTATION_CURRENT_STATE.md) | Current | documentation-governance | navigation + packaging | 2026-08-03 | Refined by this manifest; full index refresh is DOC-028 |
| [docs/development/DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md) | Current | documentation-governance | review checklist, workflows | 2026-08-03 | — |
| [docs/development/testing.md](testing.md) | Current | documentation-governance / qa | `pytest.ini`, `test/`, supervisor modules | 2026-08-03 | — |
| [docs/development/DOCUMENTATION_MANIFEST.md](DOCUMENTATION_MANIFEST.md) | Current | documentation-governance | this file | 2026-08-03 | — |
| [docs/guides/AGENT_SUPERVISOR_GUIDE.md](../guides/AGENT_SUPERVISOR_GUIDE.md) | Current | agent-supervisor | control, daemons, CLI | 2026-08-03 | — |
| [docs/guides/MCP_SETUP_GUIDE.md](../guides/MCP_SETUP_GUIDE.md) | Current | mcp | `mcp_server`, setup paths | treat as maintained; revalidate flags via `--help` | — |
| [docs/guides/getting-started/](../guides/getting-started/) | Current | package maintainers | install + first run | journey refresh wave | — |
| [docs/guides/QUICKSTART.md](../guides/QUICKSTART.md) | Current | package maintainers | CLI/Python examples | revalidate commands | — |
| [docs/guides/cli/README_CLI.md](../guides/cli/README_CLI.md) | Current | package maintainers | `cli.py`, `cli_entry.py` | revalidate `--help` | — |

---

## 2. Current — agent-supervisor architecture guides

| Path | Status | Owner | Sources | Last-verified | Supersession |
| --- | --- | --- | --- | --- | --- |
| [docs/architecture/agent_supervisor/README.md](../architecture/agent_supervisor/README.md) | Current | agent-supervisor | domain packages, hub docs | 2026-08-03 | — |
| [docs/architecture/agent_supervisor/CONTROL_PLANE.md](../architecture/agent_supervisor/CONTROL_PLANE.md) | Current | agent-supervisor | `control/` | 2026-08-03 | — |
| [docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md](../architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md) | Current | agent-supervisor | `todo_daemon/`, `merge/`, `rescue/` | 2026-08-03 | — |
| [docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md](../architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md) | Current | agent-supervisor | `planning/`, `proof/`, `validation/` | 2026-08-03 | — |
| [docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md](../architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md) | Current | agent-supervisor | `entrypoints/`, `prompt/` | 2026-08-03 | Distinguishes landed vs planned facades in-body |
| [docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md](../architecture/agent_supervisor/DEVELOPER_GUIDE.md) | Current | agent-supervisor | package layout | 2026-08-03 | — |
| [docs/architecture/agent_supervisor/FOR_AGENTS.md](../architecture/agent_supervisor/FOR_AGENTS.md) | Current | agent-supervisor | protected paths, invariants | 2026-08-03 | — |

---

## 3. Reference — vocabulary, maps, ADRs, hubs

| Path | Status | Owner | Sources | Last-verified | Supersession |
| --- | --- | --- | --- | --- | --- |
| [docs/architecture/GLOSSARY.md](../architecture/GLOSSARY.md) | Reference | documentation-governance | ADRs 0001–0006, Current guides | 2026-08-03 | — |
| [docs/architecture/README.md](../architecture/README.md) | Reference | architecture | hub routes | 2026-08-03 | — |
| [docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md](../architecture/AGENT_SUPERVISOR_PHILOSOPHY.md) | Reference | agent-supervisor | control-plane pillars | maintained product vocabulary | — |
| [docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md](../architecture/AGENT_SUPERVISOR_ARCHITECTURE.md) | Reference | agent-supervisor | full package map narrative | deep map; prefer hub for routing | — |
| [docs/architecture/agent_supervisor/PACKAGE_MAP.md](../architecture/agent_supervisor/PACKAGE_MAP.md) | Reference | agent-supervisor | domain DAG | layout changes | — |
| [docs/architecture/agent_supervisor/PROGRAMS.md](../architecture/agent_supervisor/PROGRAMS.md) | Reference | agent-supervisor | board prefixes | new programs | — |
| [docs/architecture/agent_supervisor/FOR_CONTRIBUTORS.md](../architecture/agent_supervisor/FOR_CONTRIBUTORS.md) | Reference | agent-supervisor | PR conventions | — | — |
| [docs/architecture/agent_supervisor/packages/](../architecture/agent_supervisor/packages/) | Reference | agent-supervisor | per-package pages | package renames | — |
| [docs/architecture/decisions/README.md](../architecture/decisions/README.md) | Reference | architecture | ADR index | new ADRs | — |
| [docs/architecture/decisions/0001-objectives-and-task-projections.md](../architecture/decisions/0001-objectives-and-task-projections.md) | Reference (Accepted ADR) | architecture | `objectives/`, `task_sources/` | 2026-08-03 | — |
| [docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md](../architecture/decisions/0002-model-proposals-and-evidence-admission.md) | Reference (Accepted ADR) | architecture | validation, merge, proof | 2026-08-03 | — |
| [docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md](../architecture/decisions/0003-capabilities-catalogs-and-routing.md) | Reference (Accepted ADR) | architecture | catalog, usage, routers | 2026-08-03 | — |
| [docs/architecture/decisions/0004-worktrees-leases-and-fencing.md](../architecture/decisions/0004-worktrees-leases-and-fencing.md) | Reference (Accepted ADR) | architecture | worktrees, leases | 2026-08-03 | — |
| [docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md](../architecture/decisions/0005-mutable-coordination-and-immutable-replication.md) | Reference (Accepted ADR) | architecture | DuckDB vs IPLD/IPFS | 2026-08-03 | — |
| [docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md](../architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md) | Reference (Accepted ADR) | architecture | domain packages | 2026-08-03 | — |
| [docs/architecture/decisions/0000-template.md](../architecture/decisions/0000-template.md) | Reference (template) | architecture | n/a | — | Not a decision |
| [docs/NESTED_PACKAGES.md](../NESTED_PACKAGES.md) | Reference | integrations | gitlink inventory hygiene | revalidate `.gitmodules` | Not runtime authority |

---

## 4. Plan — design and execution records (non-normative)

These documents sequence work or record intent. **Do not** treat them as
Current API, CLI, MCP, or install contracts.

| Path pattern / example | Status | Owner | Notes |
| --- | --- | --- | --- |
| [docs/architecture/DOCUMENTATION_REFRESH_PLAN_2026_08.md](../architecture/DOCUMENTATION_REFRESH_PLAN_2026_08.md) | Plan | operator (protected) | Program plan; not product runtime |
| [docs/architecture/documentation_refresh.objectives.md](../architecture/documentation_refresh.objectives.md) | Plan | operator (protected) | Objective heap |
| [docs/architecture/documentation_refresh.todo.md](../architecture/documentation_refresh.todo.md) | Plan | operator (protected) | Taskboard |
| [docs/architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md](../architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md) | Plan | routing program | Landed behavior → MODEL_SERVICE_ROUTING |
| [docs/architecture/MCP_SERVER_UNIFICATION_PLAN.md](../architecture/MCP_SERVER_UNIFICATION_PLAN.md) | Plan | mcp program | Landed behavior → MCP_RUNTIME |
| [docs/architecture/AGENT_SUPERVISOR_*_PLAN.md](../architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md) and other `*_PLAN*.md` | Plan | program owners | Delivery sequencing |
| `docs/architecture/*.objectives.md`, `*.todo.md` | Plan | program owners | Heaps and boards |
| [docs/architecture/IMPLEMENTATION_PLAN.md](../architecture/IMPLEMENTATION_PLAN.md) | Plan | historical program | Prefer Current guides |
| [docs/MCP_TRIO_ROADMAP.md](../MCP_TRIO_ROADMAP.md) | Plan | mcp | Roadmap, not Current runtime |

---

## 5. Historical — archives, summaries, cutover evidence

**Historical** material is retained for context. It must be labelled when linked
from navigation. It never overrides Current docs or live code.

### 5.1 Explicit historical trees (bulk)

| Location | Status | Owner | Approx. role | Supersession |
| --- | --- | --- | --- | --- |
| [docs/archive/](../archive/) | Historical | documentation-governance | Session and implementation archives | Prefer Current guides |
| [docs/development_history/](../development_history/) | Historical | documentation-governance | Delivery and verification session logs | Prefer Current guides |
| [docs/summaries/](../summaries/) | Historical | documentation-governance | Point-in-time summaries | Prefer Current guides |
| [docs/project/status/](../project/status/), [dashboard/](../project/dashboard/), [summaries/](../project/summaries/), [phases/](../project/phases/) | Historical | project records | Phase and status snapshots | Prefer Current guides |
| [docs/exports/](../exports/) | Generated / Historical | tooling | HTML/PDF exports | Regenerate; not normative |

### 5.2 Named historical / evidence architecture files

| Path | Status | Notes |
| --- | --- | --- |
| [docs/architecture/agent_supervisor/LAYOUT_CUTOVER_EVIDENCE.md](../architecture/agent_supervisor/LAYOUT_CUTOVER_EVIDENCE.md) | Historical | Domain-layout cutover evidence; start at package map instead |
| [docs/architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md](../architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md) | Historical | Pre-implementation inventory |
| [docs/architecture/asref/](../architecture/asref/) | Historical | Cutover receipts and move maps |
| [docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | Historical (frozen audit) | Wave-0 evidence; not a living API guide |
| Files matching `*SUMMARY*`, `*COMPLETE*`, `*FINAL*`, `*phase*`, `API_INTEGRATIONS_COMPLETE`, `CACHE_*SUMMARY*`, `COMPREHENSIVE_*` under architecture or docs root | Historical (default) | Lifecycle default; reclassify only after code-verified Current rewrite |

### 5.3 Archive debt summary

| Debt class | Observation | Remedy (not claimed done here) |
| --- | --- | --- |
| Mixed navigation | Top-level [INDEX.md](../INDEX.md) still links some **Plan** pages beside Current guides without always labelling them | DOC-028 navigation closeout |
| Filename-implied currency | “COMPLETE”, “100%”, “FINAL” titles read as guarantees | Keep **Historical**; link only with labels |
| Duplicate feature write-ups | Multiple cache/pipeline/router narratives | Prefer Current architecture guides; leave archives in place |
| Empty gitlink / vendored paths | Nested product docs may be absent offline | **Vendored**; capability-gated |
| Volume | Hundreds of Markdown files under `docs/` | Only rows in §§1–3 are maintained Current/Reference inventory |

This manifest **does not** enumerate every Historical file. Absence from §§1–3
means “not claimed Current/Reference,” not “deleted.”

---

## 6. Generated and Vendored

| Path pattern | Status | Rule |
| --- | --- | --- |
| `docs/exports/**` | Generated | Stale when source diagrams change |
| Nested product / submodule READMEs (e.g. under gitlink slots, `mcp-python-sdk/`, vendored trees) | Vendored | Upstream contract wins; do not redefine here |
| `mcpplusplus/` workspace checklists | Plan / evidence (vendored-adjacent) | Spec evidence; not `mcp_server` runtime API |
| Auto-generated inventories (if any under scripts output) | Generated | Cite generator command and input commit |

---

## 7. Feature guides and mid-tree pages (default policy)

Many paths under `docs/features/`, `docs/guides/infrastructure/`,
`docs/guides/github/`, `docs/guides/p2p/`, `docs/guides/docker/`, and root-level
feature READMEs are **useful** but not all revalidated in the documentation-
refresh wave. Classification:

| Condition | Treat as |
| --- | --- |
| Linked from Current journey and revalidated in DOC-021–026 | Current (when task evidence says so) |
| No recent verification; describes optional stacks | Reference until verified, or Historical if completion-summary style |
| Contradicts Current architecture guide | Record drift; **Current architecture + code win** |

Do **not** promote an entire directory to Current by path alone.

---

## 8. Operator-protected program inputs

These exact paths are **read-only** for implementation agents (operator
directive). Status **Plan** / execution:

- `docs/architecture/documentation_refresh.todo.md`
- `docs/architecture/documentation_refresh.objectives.md`
- `docs/architecture/DOCUMENTATION_REFRESH_PLAN_2026_08.md`

---

## 9. Related interfaces

| Interface | Version | Role |
| --- | --- | --- |
| DocumentationManifest@1 | 1 | This inventory |
| ProductGlossary@1 | 1 | [GLOSSARY.md](../architecture/GLOSSARY.md) |
| ArchitectureAudienceRouter@1 | 1 | [architecture/README.md](../architecture/README.md) |
| DocumentationLifecyclePolicy@1 | 1 | [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md) |

---

## 10. See also

- [Architecture hub](../architecture/README.md) — concern and audience routes
- [Product glossary](../architecture/GLOSSARY.md) — catalog/usage/router, MCP/MCP++, objective/task, discovery/capability/proof, CID/cache key, merge/acceptance, coordination/replication
- [Lifecycle policy](DOCUMENTATION_LIFECYCLE.md) — authority and freshness rules
- [Current state snapshot](DOCUMENTATION_CURRENT_STATE.md) — short maintained-surface list
- [Maintenance checklist](DOCUMENTATION_MAINTENANCE.md) — PR review without suppressing drift
- [Drift audit (Historical evidence)](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) — frozen Wave-0 inventory
