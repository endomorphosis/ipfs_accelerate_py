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
**Last-verified:** 2026-08-03 @ `114838662e`; classifications checked against
the lifecycle policy and landed documentation-refresh leaf guides available at
the DOC-027 source baseline.
**Interface:** DocumentationManifest@1
**Freshness triggers:** new Current guide; ADR accept/supersede; package or
entrypoint renames; archive moves; closeout of DOC-028 navigation.
**Supersedes:** none declared
**Superseded-by:** none

This manifest is the **DocumentationManifest@1** inventory for the
documentation-refresh program. It does **not** claim that every Markdown file
under `docs/` is maintained. Unlisted files default to the lifecycle fail-closed
rules (often **Plan**, **Historical**, **Generated**, or **Vendored**).

At the initial integrated DOC-027 result (`362545ebd`):

- `git ls-files docs | wc -l` reports **508 tracked index entries**.
- `git ls-files 'docs/*.md' 'docs/**/*.md' | wc -l` reports **492 tracked
  Markdown files**.

Those are inventory counts, not Current-document counts.

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
| Path | Exact repo-relative file or bounded set; directory rows never promote every descendant |
| Lifecycle | Exactly one closed status value from this manifest |
| Internal status | Separate workflow state, such as ADR `Accepted`; never a lifecycle value |
| Declared owner | Literal metadata from the document; missing values remain metadata debt |
| Policy route | Maintenance route from the lifecycle policy; not a claim that the leaf declares that owner |
| Sources | Concrete code, policy, help, or guide anchors used for classification |
| Leaf Last-verified | The document's own marker, preserved verbatim enough to retain its baseline |
| Supersession / debt | Declared replacement or a precise missing-metadata caveat |

All classifications below were checked at `114838662e`. A row does not replace
its leaf's source verification marker.

---

## 1. Current — reviewed architecture and governance

| Path(s) | Lifecycle | Internal status | Declared owner | Policy route | Sources | Leaf Last-verified | Supersession / debt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [overview](../architecture/overview.md), [system context](../architecture/SYSTEM_CONTEXT.md), [inference](../architecture/INFERENCE_RUNTIME.md), [model routing](../architecture/MODEL_SERVICE_ROUTING.md), [MCP](../architecture/MCP_RUNTIME.md), [distributed runtime](../architecture/DISTRIBUTED_RUNTIME.md), [integration boundaries](../architecture/INTEGRATION_BOUNDARIES.md) | Current | — | not declared — metadata debt | architecture maintainers | each page's Source anchors; `ipfs_accelerate_py/`, `pyproject.toml`, `.gitmodules` | `e559ff0046c639ba1dadabe02ea0ea91d9877e20` (2026-08-03) | no leaf supersession declared; Owner/Freshness fields absent |
| [control plane](../architecture/agent_supervisor/CONTROL_PLANE.md), [planning and assurance](../architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md), [execution and recovery](../architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md) | Current | — | not declared — metadata debt | agent-supervisor maintainers | `ipfs_accelerate_py/agent_supervisor/{control,planning,proof,validation,todo_daemon,merge,rescue}/` | `e559ff0046c639ba1dadabe02ea0ea91d9877e20` (2026-08-03) | no leaf supersession declared; Owner/Freshness fields absent |
| [prompt-first runtime](../architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md) | Current | — | not declared — metadata debt | agent-supervisor maintainers | `ipfs_accelerate_py/agent_supervisor/{entrypoints,prompt}/` | `6eb3525aae8143eb56993a6cd96eb9e3fff684e0` (2026-08-03) | no leaf supersession declared; Owner/Freshness fields absent |
| [guide conventions](../architecture/GUIDE_CONVENTIONS.md) | Current | — | not declared — metadata debt | documentation governance | lifecycle policy; maintained guide contract and source-anchor patterns | `d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15` (2026-08-03) | no leaf owner or supersession declared |
| [documentation lifecycle](DOCUMENTATION_LIFECYCLE.md) | Current | — | `documentation-governance` | documentation governance | `docs/INDEX.md`, `docs/README.md`, docs tree, `pyproject.toml` | `5f572b7391eccc5a1c3975e6c2f9fb4946e0d85e` (2026-08-03) | no supersession declared |
| [documentation manifest](DOCUMENTATION_MANIFEST.md) | Current | — | `documentation-governance` | documentation governance | lifecycle policy; reviewed DOC-005–027 outputs; docs index | `114838662e` (2026-08-03 source baseline) | none declared |

The eleven primary Current architecture guides above declare lifecycle status
and exact source baselines, but do not declare Owner, Freshness triggers, or
supersession fields. This manifest records that debt; it does not silently add
leaf metadata.

---

## 2. Current — reviewed user, operator and contributor pages

| Path(s) | Lifecycle | Internal status | Declared owner | Policy route | Sources | Leaf Last-verified | Supersession / debt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [getting-started README](../guides/getting-started/README.md), [installation](../guides/getting-started/installation.md), [quickstart](../guides/QUICKSTART.md) | Current | — | `package maintainers` | package maintainers | `pyproject.toml`, `setup.py`, install manifests, CLI/Python entrypoints | `b128cceef` (2026-08-03) | lowercase `installation.md` supersedes removed `INSTALLATION.md`; no directory-wide promotion |
| [API overview](../api/overview.md) | Current | — | `package maintainers` | package maintainers | package `__init__.py`, `pyproject.toml`, CLI modules, public export manifests | `b128cceef` (2026-08-03) | none declared |
| [CLI reference](../guides/cli/README_CLI.md) | Current | — | `CLI and package maintainers` | package maintainers | `cli.py`, `cli_entry.py`, control CLI, help/tests | `b128cceef` (2026-08-03) | none declared |
| [MCP server](../MCP_SERVER.md), [MCP setup](../guides/MCP_SETUP_GUIDE.md), [MCP quick start](../guides/QUICK_START_MCP.md) | Current | — | `MCP maintainers` | MCP maintainers | `ipfs_accelerate_py/mcp_server/`, `mcp/`, configuration and transport modules | `d5f3aa5c6` (2026-08-03) | compatibility routes are labelled in the pages |
| [supervisor operator guide](../guides/AGENT_SUPERVISOR_GUIDE.md), [developer guide](../architecture/agent_supervisor/DEVELOPER_GUIDE.md), [agent capsule](../architecture/agent_supervisor/FOR_AGENTS.md) | Current | — | `agent-supervisor maintainers` | agent-supervisor maintainers | operation catalog, domain packages, daemons, prompt workflow | `8e940eb01` (2026-08-03) | none declared |
| [supervisor documentation hub](../architecture/agent_supervisor/README.md) | Current | — | `agent-supervisor maintainers` | agent-supervisor maintainers | domain packages and maintained supervisor guides | `d5f3aa5c6` (2026-08-03) | none declared |
| [deployment](../guides/deployment/README.md), [hardware](../guides/hardware/overview.md), [troubleshooting](../guides/troubleshooting/faq.md) | Current | — | `package maintainers` | package maintainers | `deployments/`, compose files, `install/`, hardware probes, source-owned diagnostics | `d5f3aa5c6` (2026-08-03) | none declared |
| [P2P journey](../guides/p2p/README.md) | Current | — | `package maintainers / distributed-runtime maintainers` | package / distributed-runtime maintainers | `p2p_tasks/`, P2P workflow modules, distributed-runtime guide | `d5f3aa5c6` (2026-08-03) | none declared |
| [testing](testing.md) | Current | — | `documentation-governance / package maintainers` | documentation governance / package maintainers | `pytest.ini`, `test/`, `tests/`, supervisor module entrypoints | `d5f3aa5c6` (2026-08-03) | none declared |
| [documentation maintenance](DOCUMENTATION_MAINTENANCE.md) | Current | — | `documentation-governance` | documentation governance | lifecycle policy, docs checker, workflow limitations | `efb030db743bff50afb939e89fcaa2c650d1c055` (2026-08-03) | none declared |

---

## 3. Reference — vocabulary, maps, ADRs and fail-closed debt

| Path(s) | Lifecycle | Internal status | Declared owner | Policy route | Sources | Leaf Last-verified | Supersession / debt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [product glossary](../architecture/GLOSSARY.md) | Reference | — | documentation governance / architecture maintainers | documentation governance / architecture maintainers | Current guides, ADR-0001–0006, identity and routing packages | `114838662e` (2026-08-03 source baseline) | none declared |
| [architecture hub](../architecture/README.md) | Reference | — | architecture maintainers / documentation governance | architecture maintainers | exact routes listed in the hub | `114838662e` (2026-08-03 source baseline) | none declared |
| [current-state snapshot](DOCUMENTATION_CURRENT_STATE.md) | Reference | pre-DOC-028 snapshot | not declared — metadata debt | documentation governance | existing navigation and packaging inventory | not declared | DOC-028 owns revalidation; do not treat its old inventory as current evidence |
| [AI service catalog](../architecture/AI_SERVICE_CATALOG.md) | Reference | — | not declared — metadata debt | architecture / catalog maintainers | `ipfs_accelerate_py/model_catalog/` | not declared | metadata and source revalidation required before Current promotion |
| [supervisor philosophy](../architecture/AGENT_SUPERVISOR_PHILOSOPHY.md), [deep architecture map](../architecture/AGENT_SUPERVISOR_ARCHITECTURE.md), [package map](../architecture/agent_supervisor/PACKAGE_MAP.md), [program glossary](../architecture/agent_supervisor/PROGRAMS.md), [contributor guide](../architecture/agent_supervisor/FOR_CONTRIBUTORS.md) | Reference | — | inspect leaf; several do not declare an owner | agent-supervisor maintainers | `ipfs_accelerate_py/agent_supervisor/`; semantic package READMEs | inspect each leaf | orientation/deep maps; Current guides own landed behavior |
| [package pages](../architecture/agent_supervisor/packages/README.md) | Reference | — | not declared on family | agent-supervisor maintainers | corresponding `ipfs_accelerate_py/agent_supervisor/<package>/` paths | varies | `packages/entrypoints.md` lacks lifecycle front matter and therefore remains Reference |
| [ADR index](../architecture/decisions/README.md) | Reference | index | inspect leaf | architecture maintainers | ADR-0001–0006 and template | inspect leaf | new ADRs trigger refresh |
| [ADR-0001](../architecture/decisions/0001-objectives-and-task-projections.md), [0002](../architecture/decisions/0002-model-proposals-and-evidence-admission.md), [0003](../architecture/decisions/0003-capabilities-catalogs-and-routing.md), [0004](../architecture/decisions/0004-worktrees-leases-and-fencing.md), [0005](../architecture/decisions/0005-mutable-coordination-and-immutable-replication.md) | Reference | Accepted | not declared — Deciders are not Owner | architecture maintainers | each ADR's Source anchors | 2026-08-03; no commit declared — metadata debt | each declares `none` / `none` |
| [ADR-0006](../architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md) | Reference | Accepted | `architecture maintainers` | architecture maintainers | semantic packages, compatibility manifests and tests | `b128cceef` (2026-08-03) | none / none declared |
| [ADR template](../architecture/decisions/0000-template.md) | Reference | template | not declared | architecture maintainers | lifecycle and ADR conventions | inspect leaf | not a decision |
| [nested-packages inventory](../NESTED_PACKAGES.md) | Reference | inventory | inspect leaf | integration maintainers | `.gitmodules`, gitlink pins | inspect leaf | not runtime authority |

Rows in §§1–3 are the reviewed maintained pages or explicitly routed
Reference debt. An unlisted file is **not promoted to Current**: apply the
lifecycle policy's fail-closed patterns and verify its own metadata and sources.

---

## 4. Plan — design and execution records (non-normative)

These documents sequence work or record intent. **Do not** treat them as
Current API, CLI, MCP, or install contracts.

| Path pattern / example | Lifecycle | Policy route / declared owner | Notes |
| --- | --- | --- | --- |
| [docs/architecture/DOCUMENTATION_REFRESH_PLAN_2026_08.md](../architecture/DOCUMENTATION_REFRESH_PLAN_2026_08.md) | Plan | operator (protected) | Program plan; not product runtime |
| [docs/architecture/documentation_refresh.objectives.md](../architecture/documentation_refresh.objectives.md) | Plan | operator (protected) | Objective heap |
| [docs/architecture/documentation_refresh.todo.md](../architecture/documentation_refresh.todo.md) | Plan | operator (protected) | Taskboard |
| [docs/architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md](../architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md) | Plan | routing program | Landed behavior → MODEL_SERVICE_ROUTING |
| [docs/architecture/MCP_SERVER_UNIFICATION_PLAN.md](../architecture/MCP_SERVER_UNIFICATION_PLAN.md) | Plan | mcp program | Landed behavior → MCP_RUNTIME |
| [docs/architecture/AGENT_SUPERVISOR_*_PLAN.md](../architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md) and other `*_PLAN*.md` | Plan | program operators (leaf metadata varies) | Delivery sequencing |
| `docs/architecture/*.objectives.md`, `*.todo.md` | Plan | program operators (leaf metadata varies) | Heaps and boards |
| [docs/architecture/IMPLEMENTATION_PLAN.md](../architecture/IMPLEMENTATION_PLAN.md) | Plan | historical program | Prefer Current guides |
| [docs/MCP_TRIO_ROADMAP.md](../MCP_TRIO_ROADMAP.md) | Plan | mcp | Roadmap, not Current runtime |

---

## 5. Historical — archives, summaries, cutover evidence

**Historical** material is retained for context. It must be labelled when linked
from navigation. It never overrides Current docs or live code.

### 5.1 Explicit historical trees (bulk)

| Location | Lifecycle | Policy route / declared owner | Approx. role | Supersession |
| --- | --- | --- | --- | --- |
| [docs/archive/](../archive/) | Historical | no active owner (lifecycle policy) | Session and implementation archives | Prefer Current guides |
| [docs/development_history/](../development_history/) | Historical | project historians (policy route; leaf metadata varies) | Delivery and verification session logs | Prefer Current guides |
| [docs/summaries/](../summaries/) | Historical | project historians (policy route; leaf metadata varies) | Point-in-time summaries | Prefer Current guides |
| [docs/project/status/](../project/status/), [dashboard/](../project/dashboard/), [summaries/](../project/summaries/), [phases/](../project/phases/) | Historical | project historians (policy route; leaf metadata varies) | Phase and status snapshots | Prefer Current guides |
| [docs/exports/](../exports/) | Generated | tooling (policy route; leaf metadata varies) | HTML/PDF exports | Regenerate; not normative |

### 5.2 Named historical / evidence architecture files

| Path | Lifecycle | Notes / internal status |
| --- | --- | --- |
| [docs/architecture/agent_supervisor/LAYOUT_CUTOVER_EVIDENCE.md](../architecture/agent_supervisor/LAYOUT_CUTOVER_EVIDENCE.md) | Historical | Domain-layout cutover evidence; start at package map instead |
| [docs/architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md](../architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md) | Historical | Pre-implementation inventory |
| [docs/architecture/asref/](../architecture/asref/) | Historical | Cutover receipts and move maps |
| [docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | Historical | Frozen Wave-0 audit evidence; not a living API guide |
| Files matching `*SUMMARY*`, `*COMPLETE*`, `*FINAL*`, `*phase*`, `API_INTEGRATIONS_COMPLETE`, `CACHE_*SUMMARY*`, `COMPREHENSIVE_*` under architecture or docs root | Historical | Fail-closed default; reclassify only after code-verified Current rewrite |

### 5.3 Archive debt summary

| Debt class | Observation | Remedy (not claimed done here) |
| --- | --- | --- |
| Mixed navigation | Top-level [INDEX.md](../INDEX.md) still links some **Plan** pages beside Current guides without always labelling them | DOC-028 navigation closeout |
| Filename-implied currency | “COMPLETE”, “100%”, “FINAL” titles read as guarantees | Keep **Historical**; link only with labels |
| Duplicate feature write-ups | Multiple cache/pipeline/router narratives | Prefer Current architecture guides; leave archives in place |
| Empty gitlink / vendored paths | Nested product docs may be absent offline | **Vendored**; capability-gated |
| Volume | Hundreds of Markdown files under `docs/` | Only rows in §§1–3 are maintained Current/Reference inventory |

This manifest **does not** enumerate every Historical file. Absence from the
tables means “not claimed Current”; apply the fail-closed lifecycle rules. It
does not mean “deleted.”

---

## 6. Generated and Vendored

| Path pattern | Lifecycle | Rule |
| --- | --- | --- |
| `docs/exports/**` | Generated | Stale when source diagrams change |
| Gitlinks [docs/fastmcp](../fastmcp) and [docs/mcp-python-sdk](../mcp-python-sdk), plus their upstream READMEs when checked out | Vendored | Upstream contract wins; do not redefine here |
| MCP++ plans or evidence checklists linked from this docs tree | Plan | Evidence/internal state belongs in notes; it is not `mcp_server` runtime API |
| Auto-generated inventories (if any under scripts output) | Generated | Cite generator command and input commit |

---

## 7. Feature guides and mid-tree pages (default policy)

Many paths under `docs/features/`, `docs/guides/infrastructure/`,
`docs/guides/github/`, `docs/guides/p2p/`, `docs/guides/docker/`, and root-level
feature READMEs are **useful** but not all revalidated in the documentation-
refresh wave. Classification:

| Condition | Lifecycle | Notes |
| --- | --- | --- |
| Exact file is listed in §2 with its leaf metadata and revalidation evidence | Current | No sibling or directory-wide promotion follows |
| No recent verification; describes optional stacks without completion-summary form | Reference | Revalidate sources before Current promotion |
| Completion-summary or point-in-time form | Historical | Retain for context and label every navigation route |
| Contradicts a Current architecture guide or code | Reference | Record drift; code and Current architecture win |

Do **not** promote an entire directory to Current by path alone.

---

## 8. Operator-protected program inputs

These exact paths are **read-only** for implementation agents (operator
directive). Lifecycle status is **Plan**; execution is separate internal state:

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
