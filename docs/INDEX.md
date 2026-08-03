# IPFS Accelerate Python Documentation Index

**Status:** Current
**Owner:** documentation-governance
**Audience:** Readers and agents selecting a maintained document by task or
subsystem
**Scope:** Canonical navigation that separates **Current**, **Reference**,
**Plan**, and **Historical** surfaces. Current documentation is maintained
against the checked-out Python package.
**Non-goals:** Reclassifying every file under `docs/`; restating leaf guide
content; resolving code-owned packaging contradictions.
**Sources:** [DOCUMENTATION_MANIFEST.md](development/DOCUMENTATION_MANIFEST.md);
[DOCUMENTATION_LIFECYCLE.md](development/DOCUMENTATION_LIFECYCLE.md);
[architecture/README.md](architecture/README.md);
Current guides listed below.
**Last verified:** 2026-08-03; routes checked against the DOC-027 manifest and
Current guide headers. Exact commit and offline checks:
[DOCUMENTATION_VALIDATION_2026_08.md](development/DOCUMENTATION_VALIDATION_2026_08.md).
**Interface:** DocumentationNavigation@2

Documents under `archive/` and `development_history/` preserve earlier
implementations and are **not** normative API references. **Plan** pages
sequence work; they are never Current contracts.

---

## Start here (Current)

- [Getting started](guides/getting-started/README.md): install the package and
  run a first inference or MCP server.
- [Installation](guides/getting-started/installation.md): published and editable
  installs, extras, offline verification.
- [Quick start](guides/QUICKSTART.md): short command-line and Python examples.
- [API overview](api/overview.md): current Python exports and supported entry
  points.
- [CLI guide](guides/cli/README_CLI.md): unified `ipfs-accelerate` product CLI.
- [Architecture overview](architecture/overview.md): runtime layers and the
  inference vs supervisor plane split.
- [System context](architecture/SYSTEM_CONTEXT.md): actors, trust, and
  boundaries.
- [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md): operate the
  stable Python, CLI, and MCP control interfaces.
- [Agent Supervisor doc hub](architecture/agent_supervisor/README.md):
  audience router (operators, contributors, agents, programs).
- [Architecture hub](architecture/README.md): concern and audience router for
  all architecture guides.
- [Product glossary](architecture/GLOSSARY.md) (**Reference**): catalog/usage/
  router, MCP/MCP++, objective/task, merge/acceptance, and related terms.
- [Documentation current state](development/DOCUMENTATION_CURRENT_STATE.md):
  which surfaces are maintained and how to audit drift.
- [Documentation orientation](README.md): short choose-a-path page.

---

## User and operator guides (Current)

Reviewed journeys from the documentation-refresh wave. Sibling files under
`docs/guides/` that are **not** listed here are not promoted to Current by
path alone (see the [manifest](development/DOCUMENTATION_MANIFEST.md)).

- [Installation](guides/getting-started/installation.md)
- [Getting started](guides/getting-started/README.md)
- [Quick start](guides/QUICKSTART.md)
- [CLI](guides/cli/README_CLI.md)
- [MCP setup](guides/MCP_SETUP_GUIDE.md)
- [MCP server reference](MCP_SERVER.md)
- [Hardware support and tuning](guides/hardware/overview.md)
- [Deployment](guides/deployment/README.md)
- [P2P workflows](guides/p2p/README.md)
- [Troubleshooting FAQ](guides/troubleshooting/faq.md)
- [Testing](development/testing.md)
- [Examples](../examples/README.md) (executable samples; capability-gated)

---

## Product architecture (Current)

Start at the [architecture hub](architecture/README.md). Leaf guides:

| Concern | Guide |
| --- | --- |
| Runtime layers and public boundary | [overview.md](architecture/overview.md) |
| Actors, planes, trust/failure | [SYSTEM_CONTEXT.md](architecture/SYSTEM_CONTEXT.md) |
| Inference request lifecycle | [INFERENCE_RUNTIME.md](architecture/INFERENCE_RUNTIME.md) |
| Catalog, usage, modality routing | [MODEL_SERVICE_ROUTING.md](architecture/MODEL_SERVICE_ROUTING.md) |
| MCP and MCP++ runtime | [MCP_RUNTIME.md](architecture/MCP_RUNTIME.md) |
| IPFS, CIDs vs cache keys, P2P | [DISTRIBUTED_RUNTIME.md](architecture/DISTRIBUTED_RUNTIME.md) |
| Sibling repos and gitlinks | [INTEGRATION_BOUNDARIES.md](architecture/INTEGRATION_BOUNDARIES.md) |

Related **Reference** (metadata revalidation pending before Current promotion):

- [AI Service Catalog](architecture/AI_SERVICE_CATALOG.md) — service identities
  and resolution; revalidate against `model_catalog/` before treating as
  sole Current authority (landed routing narrative:
  [MODEL_SERVICE_ROUTING.md](architecture/MODEL_SERVICE_ROUTING.md)).

---

## Agent supervisor

Start with the **doc hub** and **operator guide** for day-to-day use.
Philosophy and package maps are **Reference** vocabulary.

### Current

- [Documentation hub (audience router)](architecture/agent_supervisor/README.md)
- [Operator guide, profiles, and migration](guides/AGENT_SUPERVISOR_GUIDE.md)
- [Control plane](architecture/agent_supervisor/CONTROL_PLANE.md)
- [Planning and assurance](architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md)
- [Execution and recovery](architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md)
- [Prompt-first runtime](architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md)
- [Developer guide](architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Agent capsule (fail-closed invariants)](architecture/agent_supervisor/FOR_AGENTS.md)

### Reference

- [Design philosophy](architecture/AGENT_SUPERVISOR_PHILOSOPHY.md)
- [Architecture map (deep)](architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Package map (domain packages + DAG)](architecture/agent_supervisor/PACKAGE_MAP.md)
- [Program glossary (board prefixes → semantic names)](architecture/agent_supervisor/PROGRAMS.md)
- [Contributor guide](architecture/agent_supervisor/FOR_CONTRIBUTORS.md)
- [Per-package semantics](architecture/agent_supervisor/packages/README.md)

### Plan (non-normative delivery records)

These sequence program work. Prefer Current guides above when they disagree.

- [Self-improvement rollout plan](architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md)
- [Formal planning and prover matrix](architecture/AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md)
- [Formal verification plan](architecture/AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
- [Leanstral goal development and benchmark](architecture/AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md)
- [Codebase-aware plan creation and steering](architecture/AGENT_SUPERVISOR_PLAN_CREATE_AND_STEER_PLAN.md)
- [Prompt-only entrypoints plan](architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md)
- [Endpoint usage-aware routing plan](architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md)
  (landed behavior: [MODEL_SERVICE_ROUTING.md](architecture/MODEL_SERVICE_ROUTING.md))
- Objective heaps and task boards (`*.objectives.md`, `*.todo.md`) under
  `docs/architecture/`

---

## Developer, operator, and governance references

| Document | Lifecycle | Role |
| --- | --- | --- |
| [Testing](development/testing.md) | Current | Test selection and capability-gated suites |
| [Documentation current state](development/DOCUMENTATION_CURRENT_STATE.md) | Current | Maintained-surface matrix |
| [Documentation manifest](development/DOCUMENTATION_MANIFEST.md) | Current | Status inventory |
| [Documentation lifecycle](development/DOCUMENTATION_LIFECYCLE.md) | Current | Authority and freshness policy |
| [Documentation maintenance](development/DOCUMENTATION_MAINTENANCE.md) | Current | PR review checklist |
| [Validation closeout](development/DOCUMENTATION_VALIDATION_2026_08.md) | Reference | Offline check receipt |
| [Guide conventions](architecture/GUIDE_CONVENTIONS.md) | Current | Architecture writing contract |
| [Product glossary](architecture/GLOSSARY.md) | Reference | Shared terms |
| [ADR index](architecture/decisions/README.md) | Reference | Why decisions exist |
| [LLM router](LLM_ROUTER.md) | Inspect leaf | Includes Goose CLI operator notes |
| [IPFS backend router](IPFS_BACKEND_ROUTER.md) | Inspect leaf | Prefer [DISTRIBUTED_RUNTIME.md](architecture/DISTRIBUTED_RUNTIME.md) for architecture |
| [Nested packages inventory](NESTED_PACKAGES.md) | Reference | Gitlink pins; not runtime authority |
| [Canonical MCP server README](../ipfs_accelerate_py/mcp_server/README.md) | Current (package) | Code-tree MCP entry |
| [Contributing](../CONTRIBUTING.md) | Project | Contribution process |

Accepted ADRs (Reference for *why*; Current guides and code for *what*):
[0001](architecture/decisions/0001-objectives-and-task-projections.md),
[0002](architecture/decisions/0002-model-proposals-and-evidence-admission.md),
[0003](architecture/decisions/0003-capabilities-catalogs-and-routing.md),
[0004](architecture/decisions/0004-worktrees-leases-and-fencing.md),
[0005](architecture/decisions/0005-mutable-coordination-and-immutable-replication.md),
[0006](architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md).

---

## Feature areas (default: not wholesale Current)

Feature write-ups under `docs/features/` are useful but not all revalidated in
the documentation-refresh wave. Prefer Current architecture guides when they
disagree. Treat completion-summary titles as **Historical** until revalidated.

- [IPFS integration](features/ipfs/IPFS.md)
- [WebNN/WebGPU](features/webnn-webgpu/WEBNN_WEBGPU_README.md)
- [Auto-healing](features/auto-healing/README.md)
- [HuggingFace model server](features/hf-model-server/README.md)
- [GitHub cache integration](features/github-cache/overview.md)

---

## Project records and archives (Historical)

- [Project documentation hub](project/README.md)
- [Status records](project/status/)
- [Dashboard records](project/dashboard/)
- [Migration records](project/migration/MIGRATION_GUIDE.md)
- [Historical session summaries](archive/sessions/)
- [Documentation audit history](development_history/README.md)
- [Drift audit (frozen Wave-0 evidence)](development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md)
- [Summaries](summaries/)

Historical reports may contain point-in-time scores, paths, test counts, or
planned work. Use the Current guides and source code for present behavior.

---

## By task

| Need | Start with | Lifecycle |
| --- | --- | --- |
| Install or verify the package | [Installation](guides/getting-started/installation.md) | Current |
| Run inference | [Quick start](guides/QUICKSTART.md) | Current |
| Use the product CLI | [CLI guide](guides/cli/README_CLI.md) | Current |
| Configure Goose CLI | [LLM router — Goose CLI](LLM_ROUTER.md#goose-cli) | Inspect leaf |
| Start MCP | [MCP setup](guides/MCP_SETUP_GUIDE.md) | Current |
| Understand MCP runtime internals | [MCP runtime](architecture/MCP_RUNTIME.md) | Current |
| Discover or route models/services | [Model/service routing](architecture/MODEL_SERVICE_ROUTING.md) | Current |
| Catalog identity plane (metadata pending) | [AI Service Catalog](architecture/AI_SERVICE_CATALOG.md) | Reference |
| Use catalog or router tools over MCP | [MCP server AI tools](MCP_SERVER.md) | Current |
| Deploy or run in containers | [Deployment](guides/deployment/README.md) | Current |
| Enable optional P2P / IPFS workflows | [P2P guide](guides/p2p/README.md) | Current |
| Operate agent-supervisor workflows | [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md) | Current |
| Learn supervisor design vocabulary | [Philosophy](architecture/AGENT_SUPERVISOR_PHILOSOPHY.md) | Reference |
| Orient as an implementation agent | [Agent capsule](architecture/agent_supervisor/FOR_AGENTS.md) | Current |
| Extend supervisor packages | [Developer guide](architecture/agent_supervisor/DEVELOPER_GUIDE.md) | Current |
| Trace merge vs acceptance | [Execution and recovery](architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md) | Current |
| Understand assurance and provers | [Planning and assurance](architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md) | Current |
| Run tests | [Testing](development/testing.md) | Current |
| Audit documentation drift | [Current state](development/DOCUMENTATION_CURRENT_STATE.md) | Current |
| Reproduce closeout validation | [Validation closeout](development/DOCUMENTATION_VALIDATION_2026_08.md) | Reference |
| Troubleshoot | [FAQ](guides/troubleshooting/faq.md) | Current |

---

**Documentation baseline:** 2026-08-03 (documentation-refresh closeout). Update
this page when a maintained entry point or canonical architecture document
changes. Exact tree identity and offline check results:
[DOCUMENTATION_VALIDATION_2026_08.md](development/DOCUMENTATION_VALIDATION_2026_08.md).
