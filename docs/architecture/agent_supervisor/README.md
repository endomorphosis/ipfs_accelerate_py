# Agent Supervisor documentation hub

**Status:** Current

**Owner:** agent-supervisor maintainers

**Audience:** Developers, operators, architects, contributors, and agents

**Sources:** `ipfs_accelerate_py/agent_supervisor/`; the maintained guides in
this directory; `docs/guides/AGENT_SUPERVISOR_GUIDE.md`;
`scripts/docs/check_agent_supervisor_docs.py`

**Last-verified:** 2026-08-03 @ `d5f3aa5c6`; navigation, package map,
operation count, and primary-document vocabulary rechecked

**Freshness triggers:** maintained guide moves, package-layout or operation
catalog changes, and documentation-guard policy changes

Product documentation for `ipfs_accelerate_py.agent_supervisor` — the control
plane for **objective-driven, evidence-bounded software work**.

Models propose plans and edits; deterministic validation, leases, allowlists,
and typed evidence decide whether work may advance.

## Start here by role

| Audience | Start | Then |
| --- | --- | --- |
| **Developer (new)** | [Developer guide](DEVELOPER_GUIDE.md) | [Package map](PACKAGE_MAP.md) · [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) |
| **Architect / deep dive** | [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) | [CONTROL_PLANE](CONTROL_PLANE.md) · [EXECUTION_AND_RECOVERY](EXECUTION_AND_RECOVERY.md) · [PLANNING_AND_ASSURANCE](PLANNING_AND_ASSURANCE.md) |
| **Operator / SRE** | [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) | Profiles, recovery, prompt workflow |
| **Contributor (PR)** | [Contributor guide](FOR_CONTRIBUTORS.md) | Developer guide · package map |
| **Implementation agent** | [Agent capsule](FOR_AGENTS.md) | Philosophy fail-closed sections |
| **Program / board audit** | [Program glossary](PROGRAMS.md) | [programs/](programs/README.md) · sealed boards |

**Code-tree entry:**
[`ipfs_accelerate_py/agent_supervisor/README.md`](../../../ipfs_accelerate_py/agent_supervisor/README.md)

## Architecture overview (one screen)

```text
Objectives ──► Taskboard ──► Lane (worktree + agent)
                                 │
                    validation · evidence · leases
                                 │
                    merge train · rescue · event log
```

| Layer | Responsibility | Primary packages |
| --- | --- | --- |
| Intent | Durable goals & evidence expectations | `objectives/` |
| Projection | Schedulable tasks & queues | `task_sources/` |
| Control | Transport-neutral ops & policy | `control/` |
| Assurance | Plans, proofs, analysis | `planning/`, `proof/`, `analysis/` |
| Prompt | Bootstrap / rescue from prompts | `prompt/` |
| Execution | Daemons, multi-lane runtime | `todo_daemon/`, `runtime/` |
| Landing | Merge, recovery | `merge/`, `rescue/` |
| Learning | Self-improvement epochs | `self_improvement/` |
| Facade | Prompt-first composition | `entrypoints/` |

Dependency DAG and placement rules: [PACKAGE_MAP.md](PACKAGE_MAP.md).

## Two namespaces (important)

| Namespace | Meaning | Examples |
| --- | --- | --- |
| **Product / domain** | What the system *is* | `proof/`, control plane, `workflow_preview` |
| **Program / board** | How work was *scheduled and evidenced* | `## PREFIX-123` task headers on a program board |

Primary documentation teaches the **product** namespace. Board IDs stay on
taskboards, objective heaps, and optional evidence footers—not in public API
names or intro prose. The guard script
`scripts/docs/check_agent_supervisor_docs.py` fails if board-prefix ticket IDs
leak into primary hub docs.

## Documentation map

### Core product docs

| Doc | Purpose |
| --- | --- |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Extend, import, test, place code |
| [PACKAGE_MAP.md](PACKAGE_MAP.md) | Domain packages & DAG |
| [CONTROL_PLANE.md](CONTROL_PLANE.md) | Operation catalog, transports, authz |
| [EXECUTION_AND_RECOVERY.md](EXECUTION_AND_RECOVERY.md) | Lanes, daemons, rescue |
| [PLANNING_AND_ASSURANCE.md](PLANNING_AND_ASSURANCE.md) | Plans, proofs, assurance |
| [PROMPT_FIRST_RUNTIME.md](PROMPT_FIRST_RUNTIME.md) | Landed entrypoint composition vs planned facades |
| [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) | Design pillars & authority ladder |
| [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) | Implementation map & contracts |
| [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) | Discover, authorize, run, recover |
| Package README | Developer entry in the code tree |

### Reference

| Doc | Purpose |
| --- | --- |
| [packages/](packages/README.md) | Per-package semantic pages |
| Domain `*/README.md` under the code tree | Module tables & import examples |
| [PROGRAMS.md](PROGRAMS.md) | Board prefix → product name |
| [programs/](programs/README.md) | Program indexes |
| [PROMPT_ENTRYPOINT_BASELINE.md](PROMPT_ENTRYPOINT_BASELINE.md) | Pre-facade friction inventory |
| [LAYOUT_CUTOVER_EVIDENCE.md](LAYOUT_CUTOVER_EVIDENCE.md) | Historical domain-layout cutover only |
| [NESTED_PACKAGES.md](../../NESTED_PACKAGES.md) | Nested trees & submodule policy |

### Long-form plans (program-owned)

Under `docs/architecture/` — module refactor, self-improvement, codebase-proof,
formal planning, prompt/usage rollouts, etc. Open these when you **own** that
program’s work; they are often protected during implementation lanes. They are
**not** the primary product vocabulary for operators or API consumers.

For current prompt-first **behavior**, prefer
[PROMPT_FIRST_RUNTIME.md](PROMPT_FIRST_RUNTIME.md) over sealed program plans.

## Domain packages

On current `main`, code lives under domain packages:

- **Foundation:** `core`, `control`, `task_sources`, `context`, `prompt`
- **Assurance:** `analysis`, `proof`, `objectives`, `planning`, `validation`
- **Operations:** `merge`, `rescue`, `runtime`, `self_improvement`
- **Edges:** `todo_daemon`, `integrations`
- **Facade:** `entrypoints` (prompt-first composition; product facades may lag)

See [PACKAGE_MAP.md](PACKAGE_MAP.md) and [packages/README.md](packages/README.md).

## Programs layered on the control plane

Self-improvement, codebase-proof, domain layout, AI service catalog, and
related efforts are **programs** that use the supervisor—they are not alternate
supervisors. Map board prefixes in [PROGRAMS.md](PROGRAMS.md).

## Control surface snapshot

The closed `Operation` catalog has **31** members. Proposal-class prompt ops
include `workflow_preview` and `rescue_preview`; mutation-class companions
include `workflow_materialize`, `restart`, and `rescue`. Full operator journey:
[AGENT_SUPERVISOR_GUIDE.md](../../guides/AGENT_SUPERVISOR_GUIDE.md).
