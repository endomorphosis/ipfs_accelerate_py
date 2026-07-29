# Agent Supervisor documentation hub

Product documentation for `ipfs_accelerate_py.agent_supervisor` — the control
plane for **objective-driven, evidence-bounded software work**.

Models propose plans and edits; deterministic validation, leases, allowlists,
and typed evidence decide whether work may advance.

## Start here by role

| Audience | Start | Then |
| --- | --- | --- |
| **Developer (new)** | [Developer guide](DEVELOPER_GUIDE.md) | [Package map](PACKAGE_MAP.md) · [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) |
| **Architect / deep dive** | [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) | Subsystem sections · package READMEs |
| **Operator / SRE** | [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) | Profiles, recovery, rollout |
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
| Execution | Daemons, multi-lane runtime | `todo_daemon/`, `runtime/` |
| Landing | Merge, recovery | `merge/`, `rescue/` |
| Learning | Self-improvement epochs | `self_improvement/` |

Dependency DAG and placement rules: [PACKAGE_MAP.md](PACKAGE_MAP.md).

## Two namespaces (important)

| Namespace | Meaning | Examples |
| --- | --- | --- |
| **Product / domain** | What the system *is* | `proof/`, control plane, codebase-proof pipeline |
| **Program / board** | How work was *scheduled and evidenced* | `## ASI-170`, `## ASREF-G020` |

Primary documentation teaches the **product** namespace. Board IDs stay on
taskboards, objective heaps, and optional evidence footers—not in public API
names or intro prose.

## Documentation map

### Core product docs

| Doc | Purpose |
| --- | --- |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Extend, import, test, place code |
| [PACKAGE_MAP.md](PACKAGE_MAP.md) | Domain packages & DAG |
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
| [LAYOUT_CUTOVER_EVIDENCE.md](LAYOUT_CUTOVER_EVIDENCE.md) | Historical domain-layout cutover only |
| [NESTED_PACKAGES.md](../../NESTED_PACKAGES.md) | Nested trees & submodule policy |

### Long-form plans (program-owned)

Under `docs/architecture/` — module refactor, self-improvement, codebase-proof,
formal planning, prompt/usage rollouts, etc. Open these when you **own** that
program’s work; they are often protected during implementation lanes.

## Domain packages

On current `main`, code lives under domain packages:

- **Foundation:** `core`, `control`, `task_sources`, `context`, `prompt`
- **Assurance:** `analysis`, `proof`, `objectives`, `planning`, `validation`
- **Operations:** `merge`, `rescue`, `runtime`, `self_improvement`
- **Edges:** `todo_daemon`, `integrations`

See [PACKAGE_MAP.md](PACKAGE_MAP.md) and [packages/README.md](packages/README.md).

## Programs layered on the control plane

Self-improvement, codebase-proof, domain layout, AI service catalog, Goose, and
related efforts are **programs** that use the supervisor—they are not alternate
supervisors. Map board prefixes in [PROGRAMS.md](PROGRAMS.md).

## Inventory

Machine-generated prefix inventories used during the semantic documentation
refresh live under [`_inventory/`](_inventory/). Treat them as tooling output,
not product narrative.
