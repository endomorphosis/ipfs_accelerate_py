# Architecture documentation hub

**Status:** Reference
**Owner:** architecture maintainers / documentation-governance
**Audience:** developers, operators, integrators, security reviewers, and
implementation agents choosing a maintained architecture path
**Scope:** Audience and concern router for architecture documentation;
pointers to Current guides, Reference maps/ADRs, and explicitly labelled
Plan/Historical material. Does not restate subsystem detail.
**Non-goals:** Top-level `docs/README.md` / `docs/INDEX.md` ownership (DOC-028);
editing leaf guides or ADRs; inventing APIs; treating plans as landed behavior.
**Sources:** maintained guides listed below; [GUIDE_CONVENTIONS.md](GUIDE_CONVENTIONS.md);
[DOCUMENTATION_LIFECYCLE.md](../development/DOCUMENTATION_LIFECYCLE.md);
[GLOSSARY.md](GLOSSARY.md); [DOCUMENTATION_MANIFEST.md](../development/DOCUMENTATION_MANIFEST.md).
**Last-verified:** 2026-08-03; routes checked against landed Current guides and
Accepted ADRs on this tree.
**Interface:** ArchitectureAudienceRouter@1

Use this page to **pick a maintained path**. Code, schemas, and executable help
remain authoritative when prose and implementation disagree.

**Rules for readers and agents:**

1. Prefer **Current** guides for operational architecture narrative.
2. Use **Reference** pages (glossary, package maps, Accepted ADRs) for
   vocabulary and *why*; revalidate volatile claims against code.
3. **Plan** and **Historical** links below are labelled. Never treat a plan,
   completion summary, archive, or `*.todo.md` / `*.objectives.md` board as a
   Current API or runtime contract.
4. Optional hardware, providers, IPFS, P2P, MCP, and provers require
   **capability language**—import ≠ capability ≠ proof
   ([GLOSSARY.md](GLOSSARY.md)).

---

## Start here

| If you need… | Open first | Status |
| --- | --- | --- |
| One-screen product map | [overview.md](overview.md) | Current |
| Actors, trust, two planes | [SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md) | Current |
| Shared term definitions | [GLOSSARY.md](GLOSSARY.md) | Reference |
| Doc status / what is maintained | [DOCUMENTATION_MANIFEST.md](../development/DOCUMENTATION_MANIFEST.md) | Current |
| How to write architecture guides | [GUIDE_CONVENTIONS.md](GUIDE_CONVENTIONS.md) | Current |
| Why decisions exist (ADRs) | [decisions/README.md](decisions/README.md) | Reference |

User/operator journeys live under [`docs/guides/`](../guides/) and
[`docs/api/overview.md`](../api/overview.md) (**Current**). This hub is for
**architecture** depth.

---

## By audience

| Audience | Start | Then |
| --- | --- | --- |
| **New developer** | [overview.md](overview.md) | [SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md) · [INFERENCE_RUNTIME.md](INFERENCE_RUNTIME.md) |
| **Inference / routing engineer** | [INFERENCE_RUNTIME.md](INFERENCE_RUNTIME.md) | [MODEL_SERVICE_ROUTING.md](MODEL_SERVICE_ROUTING.md) · [AI_SERVICE_CATALOG.md](AI_SERVICE_CATALOG.md) |
| **MCP integrator** | [MCP_RUNTIME.md](MCP_RUNTIME.md) | [GLOSSARY.md](GLOSSARY.md) (MCP vs MCP++) · package README in `ipfs_accelerate_py/mcp_server/` |
| **IPFS / P2P / identity** | [DISTRIBUTED_RUNTIME.md](DISTRIBUTED_RUNTIME.md) | [INTEGRATION_BOUNDARIES.md](INTEGRATION_BOUNDARIES.md) · ADR-0005 |
| **Cross-repo integrator** | [INTEGRATION_BOUNDARIES.md](INTEGRATION_BOUNDARIES.md) | [NESTED_PACKAGES.md](../NESTED_PACKAGES.md) (**Reference** inventory) |
| **Agent-supervisor developer** | [agent_supervisor/README.md](agent_supervisor/README.md) | [PACKAGE_MAP.md](agent_supervisor/PACKAGE_MAP.md) · [DEVELOPER_GUIDE.md](agent_supervisor/DEVELOPER_GUIDE.md) |
| **Supervisor operator** | [AGENT_SUPERVISOR_GUIDE.md](../guides/AGENT_SUPERVISOR_GUIDE.md) (**Current**) | [EXECUTION_AND_RECOVERY.md](agent_supervisor/EXECUTION_AND_RECOVERY.md) |
| **Implementation agent** | [FOR_AGENTS.md](agent_supervisor/FOR_AGENTS.md) | [GLOSSARY.md](GLOSSARY.md) · [AGENT_SUPERVISOR_PHILOSOPHY.md](AGENT_SUPERVISOR_PHILOSOPHY.md) |
| **Architect / reviewer** | [SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md) | [decisions/](decisions/README.md) · philosophy + package map |
| **Documentation author** | [GUIDE_CONVENTIONS.md](GUIDE_CONVENTIONS.md) | [DOCUMENTATION_LIFECYCLE.md](../development/DOCUMENTATION_LIFECYCLE.md) · manifest |

---

## By concern (product architecture)

Maintained **Current** product guides from the documentation-refresh wave:

| Concern | Guide | Primary sources |
| --- | --- | --- |
| Runtime layers and public package boundary | [overview.md](overview.md) | `__init__.py`, `ipfs_accelerate.py`, `pyproject.toml` |
| Actors, planes, trust/failure | [SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md) | package layout, CLI scripts, supervisor packages |
| Inference request lifecycle | [INFERENCE_RUNTIME.md](INFERENCE_RUNTIME.md) | routers, backends, workers |
| Catalog, usage, modality routing | [MODEL_SERVICE_ROUTING.md](MODEL_SERVICE_ROUTING.md) | `model_catalog/`, `endpoint_usage/`, routers |
| Service catalog identity plane | [AI_SERVICE_CATALOG.md](AI_SERVICE_CATALOG.md) | `model_catalog/` |
| MCP and MCP++ runtime | [MCP_RUNTIME.md](MCP_RUNTIME.md) | `mcp_server/`, `mcp/`, `mcplusplus_module/` |
| IPFS, CIDs vs cache keys, P2P | [DISTRIBUTED_RUNTIME.md](DISTRIBUTED_RUNTIME.md) | `ipfs_backend_router.py`, `p2p_tasks/` |
| Sibling repos and gitlinks | [INTEGRATION_BOUNDARIES.md](INTEGRATION_BOUNDARIES.md) | `.gitmodules`, adapters |

Related **Current** API/journey pages (outside this directory):
[api/overview.md](../api/overview.md),
[guides/MCP_SETUP_GUIDE.md](../guides/MCP_SETUP_GUIDE.md),
[guides/getting-started/](../guides/getting-started/).

---

## By concern (agent supervisor)

Start at the **supervisor hub** (audience router for that subsystem):

→ **[agent_supervisor/README.md](agent_supervisor/README.md)** (**Current**)

| Concern | Document | Status |
| --- | --- | --- |
| Design pillars and authority ladder | [AGENT_SUPERVISOR_PHILOSOPHY.md](AGENT_SUPERVISOR_PHILOSOPHY.md) | Reference |
| Implementation map and contracts | [AGENT_SUPERVISOR_ARCHITECTURE.md](AGENT_SUPERVISOR_ARCHITECTURE.md) | Reference / deep map |
| Package DAG and placement | [agent_supervisor/PACKAGE_MAP.md](agent_supervisor/PACKAGE_MAP.md) | Reference |
| Control operations and transports | [agent_supervisor/CONTROL_PLANE.md](agent_supervisor/CONTROL_PLANE.md) | Current |
| Lanes, daemons, merge, rescue | [agent_supervisor/EXECUTION_AND_RECOVERY.md](agent_supervisor/EXECUTION_AND_RECOVERY.md) | Current |
| Plans, proofs, assurance | [agent_supervisor/PLANNING_AND_ASSURANCE.md](agent_supervisor/PLANNING_AND_ASSURANCE.md) | Current |
| Prompt-first composition | [agent_supervisor/PROMPT_FIRST_RUNTIME.md](agent_supervisor/PROMPT_FIRST_RUNTIME.md) | Current |
| Extend code | [agent_supervisor/DEVELOPER_GUIDE.md](agent_supervisor/DEVELOPER_GUIDE.md) | Current |
| PR checklist | [agent_supervisor/FOR_CONTRIBUTORS.md](agent_supervisor/FOR_CONTRIBUTORS.md) | Reference |
| Agent fail-closed capsule | [agent_supervisor/FOR_AGENTS.md](agent_supervisor/FOR_AGENTS.md) | Current |
| Board prefix → program name | [agent_supervisor/PROGRAMS.md](agent_supervisor/PROGRAMS.md) | Reference |
| Per-package semantics | [agent_supervisor/packages/](agent_supervisor/packages/README.md) | Reference |
| Operator runbook | [guides/AGENT_SUPERVISOR_GUIDE.md](../guides/AGENT_SUPERVISOR_GUIDE.md) | Current |

Code-tree entry:
[`ipfs_accelerate_py/agent_supervisor/README.md`](../../ipfs_accelerate_py/agent_supervisor/README.md).

---

## Architectural decision records

Accepted ADRs explain **why** a boundary exists. They are **Reference** for
intent; Current guides and code remain authoritative for *what* the tree does.

| ADR | Topic | Status |
| --- | --- | --- |
| [0001](decisions/0001-objectives-and-task-projections.md) | Objectives vs task projections | Accepted |
| [0002](decisions/0002-model-proposals-and-evidence-admission.md) | Models propose; evidence admits; merge ≠ acceptance | Accepted |
| [0003](decisions/0003-capabilities-catalogs-and-routing.md) | Capability, catalog, usage, routing planes | Accepted |
| [0004](decisions/0004-worktrees-leases-and-fencing.md) | Worktrees, leases, fencing | Accepted |
| [0005](decisions/0005-mutable-coordination-and-immutable-replication.md) | Coordination vs replication | Accepted |
| [0006](decisions/0006-domain-packages-and-compatibility-boundaries.md) | Domain packages and compatibility | Accepted |

Index and template: [decisions/README.md](decisions/README.md).

---

## Plan and historical material (labelled)

The paths below are useful for sequencing or archaeology. They are **not**
Current product contracts. Prefer the Current guides above when they disagree.

### Plans (non-normative for runtime)

| Document | Topic | Status |
| --- | --- | --- |
| [DOCUMENTATION_REFRESH_PLAN_2026_08.md](DOCUMENTATION_REFRESH_PLAN_2026_08.md) | Documentation program sequencing | **Plan** (operator-protected) |
| [documentation_refresh.objectives.md](documentation_refresh.objectives.md) | Objective heap for doc refresh | **Plan** / execution (operator-protected) |
| [documentation_refresh.todo.md](documentation_refresh.todo.md) | Taskboard for doc refresh | **Plan** / execution (operator-protected) |
| [ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md](ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md) | Usage-aware routing delivery | **Plan** (see also Current [MODEL_SERVICE_ROUTING.md](MODEL_SERVICE_ROUTING.md)) |
| [MCP_SERVER_UNIFICATION_PLAN.md](MCP_SERVER_UNIFICATION_PLAN.md) | MCP unification delivery | **Plan** (see Current [MCP_RUNTIME.md](MCP_RUNTIME.md)) |
| [AGENT_SUPERVISOR_*_PLAN.md](AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md) and sibling `*_PLAN*.md` | Supervisor program delivery | **Plan** |
| [IPFS_KIT_*_PLAN.md](IPFS_KIT_INTEGRATION_PLAN.md), [WEBGPU_WEBNN_MIGRATION_PLAN.md](WEBGPU_WEBNN_MIGRATION_PLAN.md) | Integration / migration sequencing | **Plan** |
| `*.objectives.md` / `*.todo.md` under this tree | Program heaps and boards | **Plan** / execution records |

### Historical / evidence (context only)

| Location or document | Status |
| --- | --- |
| [agent_supervisor/LAYOUT_CUTOVER_EVIDENCE.md](agent_supervisor/LAYOUT_CUTOVER_EVIDENCE.md) | **Historical** cutover evidence |
| [agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md](agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md) | **Historical** / inventory baseline |
| [asref/](asref/) | **Historical** layout cutover receipts |
| `*SUMMARY*`, `*COMPLETE*`, `*FINAL*`, phase reports under architecture | **Historical** by default (lifecycle policy) |
| [`docs/archive/`](../archive/), [`docs/development_history/`](../development_history/), [`docs/summaries/`](../summaries/), [`docs/project/`](../project/) | **Historical** trees |
| Cache/pipeline “production ready” write-ups (e.g. `COMPREHENSIVE_*`) | Treat as **Historical** until revalidated against Current guides |

Full classification and archive debt summary:
[DOCUMENTATION_MANIFEST.md](../development/DOCUMENTATION_MANIFEST.md).

---

## Two planes (orientation)

```text
  Inference / data plane                    Agent-supervisor control plane
  -----------------------                   ------------------------------
  API / CLI / MCP tools                     Objectives → tasks → lanes
  catalog · usage · routers                 leases · validation · merge
  IPFS / P2P optional                       evidence · rescue · receipts
              \                              /
               adapters only (typed calls)
```

Collapsing these planes upgrades untrusted model prose into merge power.
Details: [SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md), [GLOSSARY.md](GLOSSARY.md).

---

## Governance companions

| Document | Role | Status |
| --- | --- | --- |
| [GUIDE_CONVENTIONS.md](GUIDE_CONVENTIONS.md) | ArchitectureGuideContract@1 | Current |
| [DOCUMENTATION_LIFECYCLE.md](../development/DOCUMENTATION_LIFECYCLE.md) | Status and authority policy | Current |
| [DOCUMENTATION_MAINTENANCE.md](../development/DOCUMENTATION_MAINTENANCE.md) | Review checklist | Current |
| [DOCUMENTATION_CURRENT_STATE.md](../development/DOCUMENTATION_CURRENT_STATE.md) | Maintained surface snapshot | Current |
| [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](../development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | Wave-0 drift inventory | **Historical** frozen audit (evidence) |
| [testing.md](../development/testing.md) | Validation command selection | Current |

---

## See also

- Product glossary: [GLOSSARY.md](GLOSSARY.md)
- Documentation index (top-level navigation; may still mix plan links—prefer
  status labels here until DOC-028 closeout): [docs/INDEX.md](../INDEX.md)
- Documentation orientation: [docs/README.md](../README.md)
