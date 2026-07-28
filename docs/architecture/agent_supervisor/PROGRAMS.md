# Agent Supervisor programs glossary

Programs are **scheduled work products** on top of the shared control plane.
They use objective heaps and Markdown taskboards with a stable **prefix**.
Product documentation should prefer the **semantic name**; use the prefix only
when operating a board or citing evidence.

| Prefix | Semantic program name | Purpose | Typical board / plan docs |
| --- | --- | --- | --- |
| `ASREF-` | **Domain layout** | Reorganize `agent_supervisor` into domain packages with an acyclic DAG and package READMEs | `agent_supervisor_module_refactor.*`, `AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md` |
| `ASI-` | **Self-improvement** | Bounded epoch self-improvement, refill, and v2 efficiency surfaces | `agent_supervisor_self_improvement.*`, `AGENT_SUPERVISOR_SELF_IMPROVEMENT_*.md` |
| `CBP-` | **Codebase proof** | Proof-carrying control for code change: catalog → claims → obligations → cache → context → edits → re-proof | `agent_supervisor_codebase_proof.*`, `AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md` |
| `AICAT-` | **AI service catalog** | Catalog discovery, routers, and service surfaces | `ai_service_catalog.*` |
| `PLAT-` / `PLAT2-` | **SRT plateau holdout** | Semantic-roundtrip plateau break / blind holdout promotion | SRT taskboards under datasets/benchmark worktrees |
| `GOOSE-` | **Goose CLI integration** | Goose / Meta Spark CLI integration | `goose_cli_integration.*` |
| `IRF-` | **IR family refactor** | Intent-IR family refactor (often datasets-adjacent boards) | IR family todos under `ipfs_datasets_py` |
| `REF-` | **Formal-planning evidence tags** (historical) | Older formal planning / prover matrix evidence labels | Formal planning plans; prefer module names in new prose |

## How to write about a program

**Good (product docs):**

> The codebase-proof pipeline compiles reviewed properties into obligations,
> proves them through the trust-aware proof cache, and materializes edit packets.

**Good (operator / agent on a board):**

> Drain taskboard prefix `## CBP-` from `agent_supervisor_codebase_proof.todo.md`.

**Avoid in API or package names:**

> `has_cbp_public_bindings`, `Plat2HoldoutRegistry` as the *only* public names
> (prefer semantic names; keep board IDs in evidence footers).

## Relationship to the control plane

```text
Programs (boards)  →  objectives / task_sources  →  control + runtime lanes
                              ↓
                     planning / proof / context / validation
```

A program may add modules under the appropriate **domain package** (for example
codebase-proof modules under `proof/` and `context/`). It should not invent a
parallel supervisor package named after the board prefix.
