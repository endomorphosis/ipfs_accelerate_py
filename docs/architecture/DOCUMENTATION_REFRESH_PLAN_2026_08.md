# IPFS Accelerate documentation refresh and architecture guide plan

**Status:** active execution plan
**Baseline:** `d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15` (2026-08-03)
**Program:** `ipfs-accelerate-documentation-refresh-v1`
**Task prefix:** `DOC-`
**Objective heap:** [documentation_refresh.objectives.md](documentation_refresh.objectives.md)
**Executable board:** [documentation_refresh.todo.md](documentation_refresh.todo.md)

## Outcome

Create a maintainable documentation set that is accurate for the checked-out
implementation and explains both **what the bespoke system does** and **why its
important boundaries exist**. The program covers the ordinary inference
runtime, service/model routing, MCP and MCP++, IPFS/P2P execution, cross-repo
integration, and the agent-supervisor control plane.

This plan is intentionally executable by the agent supervisor. Durable intent
lives in the objective heap; atomic work lives in the task board; the
dependency and file-ownership rules below allow independent worktrees to make
progress in parallel without silently overwriting one another.

## Why this refresh is necessary

The current documentation index declares a 2026-07-28 baseline. Since that
baseline, the repository has added or materially changed prompt-first
entrypoints, target/profile inference, run and state registries, steering,
repository/corpus analysis, MCP contract assurance, proof and attestation
adapters, authoritative completion, implementation routing, and supervisor
recovery behavior. The code delta is much larger than the corresponding
current-state documentation delta.

The documentation tree is also large and uneven: current guides, historical
completion reports, plans, and duplicated topic guides are adjacent. Readers
and agents need explicit status, source-of-truth, and archive rules so an old
summary is not mistaken for a supported contract.

## Documentation principles

Every maintained guide produced or refreshed by this program must:

1. state its audience, scope, status, and last-verified commit/date;
2. identify code, configuration, schema, or executable-help sources of truth;
3. distinguish current behavior from planned behavior and historical records;
4. explain the rationale and rejected simplifications behind bespoke choices;
5. show control flow, data flow, trust boundaries, and failure semantics when
   those relationships matter;
6. use capability language for optional providers, hardware, IPFS, P2P, MCP,
   browser, and prover integrations;
7. avoid volatile counts or performance claims without a dated reproducible
   command and environment;
8. keep secrets, credentials, raw prompts, and private provider output out of
   documentation and generated inventories;
9. prefer semantic product names over delivery-ticket or board-prefix names;
10. include a short verification recipe that can be rerun against a later
    checkout.

## Audiences and reading journeys

| Audience | Questions the refreshed set must answer |
| --- | --- |
| New user | What can I install, what is optional, and how do I run the first useful operation? |
| Application developer | Which Python/CLI/MCP surface is canonical, and what are the compatibility boundaries? |
| Integrator | How do model, endpoint, IPFS, P2P, MCP++, and sibling repositories connect? |
| Maintainer | Where does new code belong, which contracts are stable, and what must be tested? |
| Operator/SRE | What is authoritative state, how is health measured, and how is a failed run recovered? |
| Security reviewer | Where are trust, identity, authorization, lease, validation, and evidence boundaries? |
| Implementation agent | Which documents are normative, which paths may it change, and how does it prove completion? |

## Target information architecture

```text
docs/
├── README.md and INDEX.md                 audience routing and canonical entrypoints
├── architecture/
│   ├── overview.md                       one-screen maintained overview
│   ├── SYSTEM_CONTEXT.md                 actors, boundaries, containers, flows
│   ├── INFERENCE_RUNTIME.md               inference and router lifecycle
│   ├── MODEL_SERVICE_ROUTING.md           catalog, resolution, usage and fallback
│   ├── MCP_RUNTIME.md                     MCP/MCP++ transports, tools and policy
│   ├── DISTRIBUTED_RUNTIME.md             IPFS, storage, P2P and task execution
│   ├── INTEGRATION_BOUNDARIES.md          sibling repos and optional adapters
│   ├── agent_supervisor/                  control-plane architecture by concern
│   └── decisions/                         durable ADRs explaining why
├── guides/                                task-oriented user/operator journeys
├── api/                                   supported public entrypoints
├── development/                           freshness, testing and maintenance policy
├── project/                               point-in-time project records
├── archive/                               explicitly historical material
└── development_history/                   delivery/session history
```

Historical documents are retained unless a separate task proves they can be
moved safely. Navigation and status labels, rather than mass deletion, are the
first-line remedy for ambiguity.

## Goal and subgoal map

```text
DOC-G000  Accurate, rationale-rich documentation
├── DOC-G010  Evidence baseline and documentation governance
│   ├── DOC-G011  Reproducible drift inventory
│   ├── DOC-G012  Lifecycle, status and source-of-truth policy
│   └── DOC-G013  Architecture-writing and decision-record conventions
├── DOC-G020  Maintained product architecture guides
│   ├── DOC-G021  System context and inference runtime
│   ├── DOC-G022  Model/service and MCP control surfaces
│   └── DOC-G023  IPFS/P2P and cross-repository boundaries
├── DOC-G030  Agent-supervisor architecture guides
│   ├── DOC-G031  Intent, control and authority
│   ├── DOC-G032  Planning, context, analysis and proof
│   ├── DOC-G033  Scheduling, execution, merge and recovery
│   └── DOC-G034  Prompt-first runtime, state and steering
├── DOC-G040  Architectural decision records
│   ├── DOC-G041  Intent, trust and evidence decisions
│   ├── DOC-G042  Capability and isolation decisions
│   └── DOC-G043  Persistence, package and compatibility decisions
├── DOC-G050  Current user, developer and operator journeys
│   ├── DOC-G051  Installation, quickstart, Python and CLI
│   ├── DOC-G052  MCP and supervisor operation
│   └── DOC-G053  Deployment, hardware, P2P, testing and troubleshooting
└── DOC-G060  Navigation, verification and sustainable closeout
    ├── DOC-G061  Glossary, audience routes and archive taxonomy
    └── DOC-G062  Link/example verification and freshness publication
```

The complete goal contracts and acceptance evidence are in the objective heap.
The board contains 28 bounded implementation tasks.

## Execution waves and parallelism

| Wave | Tasks | Maximum useful width | Gate to next wave |
| --- | --- | ---: | --- |
| 0 — evidence and conventions | `DOC-001`–`DOC-004` | 4 | Drift baseline, lifecycle policy, guide contract, ADR template exist. |
| 1 — architecture guides | `DOC-005`–`DOC-014` | 4 | Each subsystem has current source anchors, flows, rationale and failure semantics. |
| 2 — decisions | `DOC-015`–`DOC-020` | 4 | Six ADRs record the durable design choices used by the guides. |
| 3 — user/operator refresh | `DOC-021`–`DOC-026` | 4 | Supported journeys agree with architecture and executable surfaces. |
| 4 — navigation and closeout | `DOC-027`–`DOC-028` | 1–2 | Canonical indexes, status matrix and validation report are current. |

Dependencies are evidence dependencies, not a demand for serial execution.
Within a wave, tasks own disjoint paths. A later task may link to an earlier
artifact but must not rewrite it unless its task record explicitly owns that
path.

## File ownership and conflict rules

| Work family | Owned paths | Shared paths intentionally deferred |
| --- | --- | --- |
| Baseline/governance | `docs/development/DOCUMENTATION_*`, `docs/architecture/GUIDE_CONVENTIONS.md` | `docs/README.md`, `docs/INDEX.md` |
| Product architecture | Named new files under `docs/architecture/` and `overview.md` only for `DOC-005` | Indexes and user guides |
| Supervisor architecture | Named files under `docs/architecture/agent_supervisor/` | Existing hub, operator guide and package READMEs |
| Decisions | One numbered ADR per task under `docs/architecture/decisions/` | ADR index is created by `DOC-004` and finalized by `DOC-027` |
| User/operator refresh | Exact guide files declared by each task | Top-level indexes |
| Closeout | `docs/README.md`, `docs/INDEX.md`, architecture hubs and current-state page | All earlier content is read-only evidence |

The plan, objective heap, and task board are operator-owned inputs and are
protected from implementation agents. The daemon may update task status only
through its normal post-merge completion path.

## Guide content contract

Architecture guides use the following common outline where applicable:

1. status, audience, scope and non-goals;
2. context and component map;
3. canonical entrypoints and source anchors;
4. control flow and data flow;
5. state and identity model;
6. trust, authorization and failure boundaries;
7. why the design exists and what simpler alternatives would break;
8. extension points and compatibility constraints;
9. operational signals and recovery;
10. verification recipe and related ADRs.

Diagrams use Mermaid only when the repository renderer supports it; otherwise
they use fenced monospace diagrams that remain readable in plain Markdown.
Names in diagrams must match live packages or be explicitly labelled as a
conceptual boundary.

## Sources of truth

| Claim area | Primary evidence |
| --- | --- |
| Package metadata, extras, scripts | `pyproject.toml`, `setup.py`, installed `--help` |
| Python API | package `__init__.py`, reviewed export manifests, focused import tests |
| CLI | `cli_entry.py`, `cli.py`, command parsers, subprocess help tests |
| Inference and routing | `ipfs_accelerate.py`, routers, `model_catalog/`, `endpoint_usage/`, provider adapters |
| MCP/MCP++ | `mcp_server/`, compatibility layer under `mcp/`, interface descriptors and conformance tests |
| IPFS/P2P | backend router, `datasets_integration/`, `p2p_tasks/`, workflow discovery/scheduler, sibling adapters |
| Agent supervisor | domain packages, package READMEs, control contracts, task sources, runtime and daemon tests |
| Deployment and hardware | installer inputs, Docker/deployment manifests, capability probes, focused hardware tests |
| History | Git log and dated records; history does not override current code |

An implementation agent should cite repo-relative source paths and symbols,
not paste large source excerpts. If code and prose disagree, the task updates
prose to verified current behavior or records a clearly labelled defect; it
must not invent an API to make an old guide appear correct.

## Validation strategy

Each task runs its declared focused checks plus `git diff --check`. Closeout
also verifies:

- all current-guide relative links resolve;
- referenced CLI commands exist in executable help or are labelled legacy;
- Python examples use intentional public imports;
- optional integrations are described as capabilities, not universal facts;
- top-level indexes route by audience and do not present archives as current;
- primary supervisor docs pass `scripts/docs/check_agent_supervisor_docs.py`;
- every architecture guide has status, source anchors, rationale, failure
  semantics, verification instructions and a baseline marker.

Network calls and live provider inference are not required for documentation
acceptance. Examples that need external services are validated structurally
and labelled with their prerequisites.

## Supervisor launch and anti-blocking policy

The program runs from a clean dedicated worktree and integration branch. Four
strict deterministic shards are used so two agents cannot claim the same task.
The program inputs are protected, work happens in ephemeral worktrees, and
accepted branches merge only into the dedicated documentation branch.

Runtime safeguards:

- provider route is exact `grok-4.5` first in the forced-Docker quota route;
  exact `gpt-5.6-terra` at `medium` is eligible only after the exact durable
  Grok 402 balance-exhausted record is independently reproduced;
- `max-task-attempts=3` and separate implementation/validation/merge retry
  budgets prevent infinite failure loops;
- heartbeats are checked every 30 seconds with a 600-second startup grace;
- a two-hour ordinary task timeout and three-hour hard ceiling cover long doc
  audits without allowing silent hangs;
- log-stall detection recycles an implementation attempt after 20 minutes of
  no output;
- merge reconciliation is bounded to one merge per pass;
- objective refill, goal migration, and codebase refill are disabled for this
  sealed 28-task tranche, preventing unscheduled scope growth;
- state, logs, worktrees and merge queues live outside the source checkout and
  are preserved for recovery.

The exact command and current runtime paths are recorded in the closeout
section after launch. Operators inspect task-state JSON, supervisor events,
implementation logs, process ancestry and merge receipts; a PID alone is not
treated as health.

## Completion definition

The program is complete only when:

1. all 28 tasks have authoritative completed status or an explicit reviewed
   scope disposition;
2. the maintained architecture set covers every subsystem named above;
3. each bespoke design choice is connected to a rationale-bearing guide or
   ADR and to current source anchors;
4. supported installation, Python, CLI, MCP, supervisor and deployment
   journeys agree with executable code;
5. canonical indexes distinguish maintained, planned, point-in-time and
   archived documents;
6. local link/example/style checks pass on the merge target;
7. the final current-state page records the commit, date, commands, known
   limitations and next audit trigger.

## Explicit non-goals

- Rewriting source code to preserve a stale documentation claim.
- Declaring optional hardware, provider, network or prover support healthy
  without a capability probe.
- Deleting historical records as part of the first refresh pass.
- Publishing generated API prose without review of stable versus internal
  symbols.
- Treating a model-generated explanation as architecture evidence by itself.
- Updating sibling repositories' documentation in this program.
