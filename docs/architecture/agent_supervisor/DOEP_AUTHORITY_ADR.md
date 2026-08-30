# DOEP Authority ADR: Current objective, planner, task, event, and state authorities

- **Status:** Accepted
- **Date:** 2026-08-30
- **Last verified:** 2026-08-30 against sealed base forest for `DOEP-PLAN-V5`
  (`ipfs_accelerate_py` commit `87715e9295626e7918f7fc8a7b1a1531ab04208f`,
  tree `1c9a399cc7a599d5904e5be2ae58c6be3650cff7`)
- **Deciders:** DOEP bootstrap inventory (`DOEP-000`) under plan revision
  `DOEP-PLAN-V5`
- **Scope:** Inventory of the canonical objective, planner, task, event, and
  state/CAS authorities that Direct Objective and Event-Driven Planning (DOEP)
  reuses. Records cross-repository ownership and deferred hazards discovered
  during inventory. Does not implement new runtime behavior.
- **Non-goals:** Creating a competing objective, planner, task, event, or state
  subsystem; disposing deferred hazards as new campaign tasks; treating
  DuckLake, Markdown boards, model claims, or worker receipts as completion
  authority; activating objective/codebase refill at bootstrap.
- **Supersedes:** none
- **Superseded-by:** none
- **Related guides:**
  - `docs/architecture/AGENT_SUPERVISOR_DIRECT_OBJECTIVE_AND_EVENT_DRIVEN_PLANNING_V1_PLAN.md`
    (sealed plan; operator-protected)
  - `docs/architecture/agent_supervisor/CONTROL_PLANE.md`
  - `docs/architecture/agent_supervisor/PACKAGE_MAP.md`
  - `docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md`
  - `docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md`
  - `docs/architecture/decisions/0001-objectives-and-task-projections.md`
- **Source anchors:** packages and interfaces listed under
  [Authority inventory](#authority-inventory)
- **Plan binding:** `DOEP-PLAN-V5` /
  plan CID `sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4` /
  task `DOEP-000` /
  board namespace `agent-supervisor-direct-objective-and-event-driven-planning-v1`

## Context

DOEP makes the existing agent supervisor directly invokable from a high-level
idea. Before contracts, compiler stages, or event consolidation land, the
campaign must freeze which landed modules already own objective, planner, task,
event, and state authority. Without that inventory, later tasks risk forking a
second planner family, a second task database, or a second event path.

Forces:

- Multi-writer DuckDB safety requires one authenticated Quack owner.
- Datasets owns semantic identity; Kit owns exact durable bytes; Accelerate owns
  operational admission, validation, merge, recovery, and terminalization.
- Worker or model assertions are never completion authority.
- Bootstrap refill is disabled; the sealed 85-task board is the campaign ceiling.

## Decision

DOEP reuses the existing canonical authorities named by the sealed plan's
**Canonical consolidation decision**. No competing subsystem is created by
`DOEP-000`.

1. **Objective path.** Existing `objectives` heap/refinery plus
   `entrypoints.intent_service` (`SupervisorIntentService`) and
   `entrypoints.plan_materializer`.
2. **Planner.** Existing formal plan compiler, validator, and replanner together
   with `adaptive_planner`. There is no second planner family.
3. **Operational task authority.** `DatabaseTaskSource@1` and its DuckDB
   `IntentRepository`, served to concurrent workers only by one authenticated
   `QuackStateServer@1` owner.
4. **Event path.** Existing `DatabaseEventLog@1`, runtime event log, runtime CAS,
   and durable Kit storage adapters.
5. **State / CAS.** DuckDB intent rows and Quack exclusive ownership for mutable
   coordination; Accelerate `runtime_cas` for operational CAS; Kit for exact
   durable bytes, CIDs, WAL, recovery, and current-root CAS.
6. **Cross-repository split.** Datasets = semantic identity, schemas, ContextPack
   meaning, formal translation, and proof relationships. Kit = exact durable
   bytes and immutable storage. Accelerate = operational admission, execution,
   validation, merge, recovery, and terminalization.
7. **DuckLake.** Optional, rebuildable, append-only history/analytics projection.
   Never scheduling, completion, policy, or proof authority.
8. **Prompt-first facade.** The dormant entrypoints LaunchPlan surface is not
   bootstrap authority: on the sealed base it fails closed without a production
   intent factory and complete launch plan. DOEP consolidates and qualifies that
   intended surface rather than inventing a parallel facade.

The reviewed bootstrap route remains: sealed objective/board → canonical JSON
materialization → DuckDB → exclusive Quack owner → existing configured
multi-lane supervisor.

## Authority inventory

Paths below are relative to `ipfs_accelerate_py/agent_supervisor/` unless noted.

### Objective authority

| Field | Value |
| --- | --- |
| Role | Durable goal/intent lifecycle and direct objective submission |
| Disposition | **reuse** |
| Primary paths | `objectives/objective_tracker.py`, `objectives/objective_graph.py`, `objectives/goal_completion.py`, `objectives/backlog_refinery.py`, `entrypoints/intent_service.py`, `entrypoints/plan_materializer.py` |
| Key symbols / interfaces | `ObjectiveTracker`, `ObjectiveGoal` / objective graph helpers, `evaluate_goal_completion`, `SupervisorIntentService`, `PromptProgramMaterializer` / `materialize` |
| Library vs DOEP wiring | Library capability is landed. DOEP bootstrap materializes the sealed board into DuckDB; direct high-level submission is qualified by later `DOEP-G020` tasks over these same modules. |
| Negative assertions | Markdown `*.objectives.md` projections are not mutation authority. Model output is not goal completion. No second objective heap is introduced. |

### Planner authority

| Field | Value |
| --- | --- |
| Role | Formal and adaptive plan compile, validate, and replan |
| Disposition | **reuse** (single planner family) |
| Primary paths | `planning/formal_plan_compiler.py`, `planning/formal_plan_validator.py`, `planning/formal_replanner.py`, `planning/adaptive_planner.py`, `planning/formal_plan_context.py`, `planning/formal_planning_contracts.py` |
| Key symbols / interfaces | `FormalPlanCompiler`, `FormalPlanValidator`, `FormalDeltaReplanner`, `AdaptivePlanner` |
| Library vs DOEP wiring | Landed. Deterministic-first route-ladder extensions reuse these modules; they do not replace them. |
| Negative assertions | No second planner family, no DOEP-only plan IR, and no model-authored plan as admitted authority. |

### Task authority

| Field | Value |
| --- | --- |
| Role | Operational taskboard, claims, leases, fencing, and idempotent transitions |
| Disposition | **reuse** |
| Primary paths | `task_sources/database_task_source.py`, `task_sources/intent_repository.py`, `task_sources/typed_database_task_source.py`, `task_sources/duckdb_state.py`, `task_sources/task_identity.py`, `runtime/quack_state_server.py` |
| Key symbols / interfaces | `DatabaseTaskSource@1`, `IntentRepository`, `TypedDatabaseTaskSource@1`, `QuackStateServer@1` |
| Library vs DOEP wiring | DOEP scheduler uses DuckDB task source with Quack authority mode. Concurrent workers talk only through the authenticated Quack owner. |
| Negative assertions | Markdown/`*.todo.md` boards are schedulable projections, not completion authority. Workers cannot terminalize tasks. No second task database is created. |

### Event authority

| Field | Value |
| --- | --- |
| Role | Durable domain events, operational event history, and artifact offload |
| Disposition | **reuse** |
| Primary paths | `runtime/database_event_log.py`, `runtime/event_log.py`, `runtime/artifact_store.py`, `runtime/database_artifact_store.py`, `runtime/runtime_cas.py` |
| Key symbols / interfaces | `DatabaseEventLog@1`, runtime JSONL `event_log`, `runtime-cas@1` |
| Library vs DOEP wiring | Landed. Later `DOEP-G040` tasks consolidate schema, publication, consumption, and sibling validation on this path. |
| Negative assertions | In-memory helpers are not canonical durable publication. Sibling supervisors exchange events and receipts; they do not write peer databases. |

### State / CAS authority

| Field | Value |
| --- | --- |
| Role | Mutable coordination state, leases/fences, and content-addressed artifacts |
| Disposition | **reuse** (Accelerate operational CAS + Kit durable bytes) |
| Primary paths | `runtime/quack_state_server.py`, `task_sources/intent_repository.py`, `task_sources/duckdb_state.py`, `runtime/runtime_cas.py`; Kit durable/CID/WAL/current-root CAS outside this package |
| Key symbols / interfaces | `QuackStateServer@1`, DuckDB intent/state rows, `ipfs_accelerate_py/agent-supervisor/runtime-cas@1` |
| Library vs DOEP wiring | Exclusive Quack owner is mandatory for parallel writers and cannot fall back silently. |
| Negative assertions | DuckLake is not state authority. Provider subprocesses do not receive the Quack mutation credential. Stale lease, fence, plan epoch, policy, or tree identity must not complete. |

### Cross-repository ownership (summary)

| Concern | Authority |
| --- | --- |
| Canonical semantic identity, schemas, ContextPack meaning | `ipfs_datasets_py` |
| Exact durable bytes, CIDs, WAL, recovery, current-root CAS | `ipfs_kit_py` |
| Operational admission, execution, validation, merge, recovery, terminalization | `ipfs_accelerate_py` |
| DuckLake history/analytics projection | Non-authoritative when enabled |
| Model / worker / advisor output | Never completion authority |

## Deferred hazards (record only)

These findings are recorded for later disposition. They do **not** expand the
sealed DOEP campaign. Direct-state-write and controller-bypass inventory is
owned by `DOEP-002`.

| Hazard | Observation | Disposition |
| --- | --- | --- |
| Legacy ContextPack candidates | Accelerator `semantic_state/context_pack.py` keeps `ContextPacker.V01_PRODUCTION_AUTHORITY = False`; production ContextPack identity remains datasets-owned. | Record; do not promote as DOEP authority. |
| Noncanonical in-memory event helpers | Compatibility helpers exist beside `DatabaseEventLog@1` / durable runtime event log. | Record; durable path remains canonical. |
| Duplicate adapters | Multiple CLI/MCP/facade adapters can project the same control operations. | Record; transports remain adapters over one control plane. |
| Retention / tombstone gaps | Event and artifact retention/tombstone policy is incomplete relative to long-running multi-lane campaigns. | Record; no silent weaken of fail-closed rules. |

## Alternatives considered

| Alternative | Decision |
| --- | --- |
| Fork a DOEP-specific planner / task / event stack | **Rejected.** Violates reuse-not-replace and creates a competing subsystem. |
| Treat Markdown boards, DuckLake, or model claims as authority | **Rejected.** Projections and analytics are non-authoritative; model claims are proposals. |
| Skip sealed inventory and proceed to contracts | **Rejected.** Bootstrap requires an exact authority map before `DOEP-G020+`. |
| Activate objective/codebase refill at bootstrap | **Rejected.** Board is sealed and bounded; refill activates only after `DOEP-050..056`. |

## Consequences

- Later DOEP tasks must extend the modules inventoried here rather than introduce
  parallel authorities.
- Candidate receipts remain non-authoritative (`completion_authoritative: false`)
  until independent fenced supervisor admission.
- Deferred hazards stay outside the sealed 85-task ceiling unless a later
  reviewed plan revision admits them.

## Evidence and verification

| Claim | Evidence |
| --- | --- |
| Objective modules landed | Paths under `objectives/` and `entrypoints/intent_service.py`, `entrypoints/plan_materializer.py` |
| Single planner family | `FormalPlanCompiler`, `FormalPlanValidator`, `FormalDeltaReplanner`, `AdaptivePlanner` present under `planning/` |
| Task + Quack interfaces | `DatabaseTaskSource@1`, `QuackStateServer@1` constants and classes |
| Event interface | `DatabaseEventLog@1` in `runtime/database_event_log.py` |
| No competing subsystem from this task | Declared outputs are ADR + independent test + output manifest + candidate receipt only |

Independent verification command (owner-relative under `external/ipfs_accelerate`):

```text
python3 -m pytest test/api/doep/test_doep_000_inventory_current_objective_planner_task_event_and_s.py -q
```

A worker or model assertion alone is insufficient for task completion.
