# ADR-0001: Separate durable objectives from regenerable task projections

- **Status:** Accepted
- **Date:** 2026-08-03
- **Last verified:** 2026-08-03
- **Deciders:** documentation-refresh trust-decisions track (DOC-015); agent supervisor control-plane maintainers
- **Scope:** Mutability and authority split between objective heaps (durable intent) and taskboards / queues (schedulable projections); rules that preserve intent during refinement, execution, and recovery.
- **Non-goals:** Model proposal vs evidence admission (ADR-0002); capability catalogs and routing (ADR-0003); worktree leases and fencing mechanics (ADR-0004); mutable coordination vs immutable replication formats (ADR-0005); domain package DAG layout (ADR-0006); proof trust tiers and formal verification policy.
- **Supersedes:** none
- **Superseded-by:** none
- **Related guides:**
  - `docs/architecture/agent_supervisor/CONTROL_PLANE.md` (§3 intent hierarchy)
  - `docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md` (pillar 1)
  - `docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md` (merge ≠ completion; recovery)
  - `docs/architecture/agent_supervisor/FOR_AGENTS.md` (protected paths; board headers)
  - `docs/architecture/agent_supervisor/packages/objectives.md`
  - `docs/architecture/agent_supervisor/packages/task_sources.md`
- **Source anchors:**
  - `ipfs_accelerate_py/agent_supervisor/objectives/` — `objective_tracker`, `objective_graph`, `goal_completion`, `backlog_refinery`
  - `ipfs_accelerate_py/agent_supervisor/task_sources/` — `task_identity`, `taskboard_store`, `markdown_task_source`, `persistent_task_queue`
  - `ipfs_accelerate_py/agent_supervisor/control/control_contracts.py` — `objective_id` / `objective_revision` on `OperationRequest`
  - `ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py` — completion gates separate from board status
  - `ipfs_accelerate_py/agent_supervisor/todo_daemon/task_execution_policy.py` — protected paths
  - `test/api/test_agent_supervisor_goal_completion.py`, `test_agent_supervisor_task_identity.py`, `test_agent_supervisor_taskboard_store.py`, `test_agent_supervisor_objective_graph.py`, `test_agent_supervisor_implementation_protected_paths.py`

## Status meanings (do not invent new values)

| Value | Use when |
| --- | --- |
| Proposed | Decision is under review; **not** yet evidenced current design |
| Accepted | Decision matches current code/tests/ops practice and is normative for Scope |
| Deprecated | Still historical; prefer another practice for new work |
| Superseded | Replaced by the ADR in Superseded-by |
| Rejected | Considered and not adopted; retained to document the negative choice |

Only **Accepted** records are current design authority. **Proposed** records
must not be treated as implemented system law.

## Context

The agent supervisor runs multi-lane, multi-writer work: backlog refinery emits
tasks, implementation daemons claim and drain boards, merge trains land code,
and recovery/rescue restarts after crashes. Humans and operators also edit
Markdown heaps and boards under `docs/architecture/`.

Two forces collide:

1. **Automation pressure.** Refill, reconcile, janitor, and projection rewrite
   boards often. Task status, shard filters, and ready queues must stay fluid so
   daemons can schedule work.
2. **Intent durability.** Goals, acceptance criteria, evidence expectations, and
   parent/child structure must survive refill, regeneration, and recovery. If the
   latest board rewrite is treated as truth, protected goals disappear, foreign
   work is marked complete, and audit trails become fiction.

A **single mutable todo list** (one board that is both “what we want” and “what
to run next”) fails under those forces:

| Failure mode | What goes wrong |
| --- | --- |
| Intent erasure | Aggressive refill or regeneration overwrites goals and acceptance prose |
| False completion | Board status `done` is treated as goal authority without fresh evidence |
| Identity thrash | Renumbering or rewriting headers breaks leases, deps, and receipts bound to task IDs |
| Authority collapse | Task projection layer starts defining control or proof trust |
| Recovery amnesia | After restart, replaying board text reconstructs schedule but not durable acceptance |

Who is affected: operators (launch and protected inputs), implementation agents
(edit policy and acceptance), maintainers of `objectives/` and `task_sources/`,
and any control transport that binds `objective_id` / `objective_revision`.

What fails if deferred: multi-lane automation cannot safely regenerate boards;
goal completion becomes model monologue or local green tests; rescue and
reconcile cannot tell durable intent from transient schedule state.

## Decision

**Separate durable intent from regenerable projections.** Objectives and
taskboards have **different mutability** and **different authority**.

### 1. Objectives are durable intent

An **objective heap** (`*.objectives.md` and the `objectives/` package) holds:

- goal identity and parent/child structure;
- evidence expectations and acceptance criteria;
- completion lifecycle authority for goals (`goal_completion`: provisional vs
  verified completion, reopen, blocked), not for arbitrary board rewrites.

Intent is **durable**: later scheduling, backlog refill, or taskboard
regeneration must not silently rewrite protected goals. Refinery and projection
may **emit** tasks that serve an objective; they do not **become** the objective.
Completing or advancing a goal is a **policy + evidence** decision, not a model
narrative or board status alone.

Control requests bind content identity via `objective_id` and
`objective_revision` so retries and alternate transports cannot retarget another
goal revision without detection.

### 2. Taskboards are schedulable projections

A **taskboard** (`*.todo.md`, DuckDB sources, queues under `task_sources/`) is a
**drainable projection**:

- stable machine identity (`## PREFIX-###` headers and `task_identity`);
- dependency, shard, and ready filters for daemons;
- status that records work progress for humans and agents.

Boards may be refined, refilled, reordered, and regenerated under policy.
Board status alone is **not** authoritative goal or task completion. The
`task_sources` layer is **forbidden** from re-defining control authority or
proof trust.

### 3. Rules that preserve intent

#### During refinement (objective_refine, backlog_refill, janitor)

| Rule | Intent preserved |
| --- | --- |
| Project, do not absorb | Refinery emits tasks from objectives; it does not replace the heap with the board |
| Protect operator inputs | Paths listed as protected / operator-owned heaps and sealed plans remain off-limits even when a task lists them in `Outputs`; authorized maintenance and refill use a separate trusted control protocol |
| Stable task identity | Keep `## PREFIX-###` headers; do not renumber foreign tasks; use `task_identity` content rules for semantic identity |
| Bounded generation | Durable ledgers / generation records dedupe refill so the same gap does not thrash the board into rewriting intent |
| Policy-gated mutations | Objective refine/reconcile and backlog refill are control **mutation** operations under authorization, effects, and (when configured) lease/fence—not free agent file edits |

#### During execution (claim, implement, validate, merge)

| Rule | Intent preserved |
| --- | --- |
| Identity-bound work | `OperationRequest` carries repository/tree and objective revision bindings; isolation uses worktree + protected paths |
| Proposal ≠ completion | Model patches and provider exit codes never mark goals complete |
| Merge ≠ acceptance | Landing code on a branch does not by itself complete a goal or authorize board “done” as goal truth |
| Authoritative completion separate | `authoritative_completion` recompute gates admit board/goal promotion; callers cannot self-assert completion authority on receipts |
| No foreign board rewrites | Implementation lanes do not edit protected heaps, sealed plans, or foreign task sections; declaring one as an output does not grant authority |

#### During recovery (restart, reconcile, rescue, quarantine)

| Rule | Intent preserved |
| --- | --- |
| Reconstruct schedule from durable sources | Recovery may re-read objectives and re-project tasks; it must not invent new goal acceptance from stale board prose alone |
| Immutable receipts | Recovery and restart publish typed receipts; they do not silently rewrite objective heaps to match a dirty board |
| Fence stale actors | Stale leases/fencing epochs cannot mutate state after recovery ownership moves |
| Operator-gated dirty reconcile | Unknown dirty checkout content is not auto-committed, stashed, or discarded to “fix” the board; reconciliation guardrails stay operator-visible |
| Explicit reopen on stale evidence | Recovery receipts alone neither complete nor reopen work. Stale post-merge evidence reopens acceptance only through the explicit `reopen_acceptance_for_stale_post_merge_validation` path |

### 4. Ownership summary

| Artifact | Mutability | Authority |
| --- | --- | --- |
| Objective heap | Durable; policy + evidence to change goals/completion | Intent and goal lifecycle |
| Taskboard / queue | Regenerable projection; high churn under refill | Scheduling and progress display only |
| Model plan / patch | Ephemeral proposal | None for mutation or completion |
| Receipt / audit / completion gate | Append/recompute under control | What actually ran and what may be admitted |

## Alternatives

### Alternative A: Single mutable todo list as sole source of truth

- **Summary:** Collapse goals and tasks into one board. Status, acceptance, and
  schedule all live in the same Markdown (or DB) document.
- **Expected benefits:** Simpler mental model; fewer files; fewer packages;
  agents only learn one format.
- **Why not chosen:** Refill and regeneration rewrite history. Multi-lane
  agents mark foreign work complete or erase acceptance criteria. There is no
  durable place for evidence expectations independent of drainable task status.
  Recovery after crash cannot distinguish “intent” from “last schedule
  snapshot.” This is the primary failure mode this ADR rejects.

### Alternative B: Immutable taskboards; only append new tasks

- **Summary:** Never rewrite boards; only append tasks and never regenerate.
  Status updates would be external event logs only.
- **Expected benefits:** Strong audit trail of every task line ever written;
  simpler “no regeneration” recovery story.
- **Why not chosen:** Boards would grow without bound and become unreadable.
  Ready/shard filters and dependency repair require in-place status and
  structural updates. Daemon drain semantics need mutable projection state.
  Immutability belongs to **receipts and content identities**, not to the
  schedulable view.

### Alternative C: Model chat transcript as durable intent

- **Summary:** Treat conversation history or agent monologue as the goal store;
  derive boards opportunistically from chat.
- **Expected benefits:** Natural for interactive agents; no separate heap
  format.
- **Why not chosen:** Transcripts are not content-addressed goal revisions,
  lack parent/child structure and typed acceptance, and cannot be
  policy-gated under `objective_revision`. Eloquence upgrades trust by accident.
  Control transports cannot bind operations to stable objective identity.

### Alternative D: Do nothing / leave mutability undocumented

- **Summary:** Keep the split only in informal philosophy and package READMEs
  without a numbered decision.
- **Expected benefits:** Less documentation work.
- **Why not chosen:** Agents and contributors repeatedly “simplify” dual
  artifacts into one board. Without a normative ADR, protected-path and
  projection rules look optional. DOC-015 and the control-plane guide require
  this boundary as recorded design authority.

## Consequences

### Positive

- **Protected intent survives automation.** Refill, janitor, and regeneration
  can be aggressive without erasing goals or acceptance criteria.
- **Clear authority ladder.** Intent → projection → proposal → validation →
  isolation → evidence → mutation; boards sit on the projection rung only.
- **Diagnosable completion.** Operators can reject “board says done” without
  gates; goal completion modules and authoritative completion gates remain
  the authority.
- **Stable multi-lane scheduling.** Task IDs and objective revisions bind
  leases, deps, and receipts across workers and restarts.
- **Safer recovery.** Restart reconstructs schedule from durable intent and
  projections without inventing goal success from dirty board text.

### Negative

- **Two artifacts to maintain.** Every program needs both `*.objectives.md`
  and `*.todo.md` (or equivalent store); authors must not dump acceptance only
  into task bodies.
- **Operational friction.** Mutations carry more bindings (objective revision,
  protected paths, effects, leases) than a casual shell edit of a single board.
- **Projection lag.** Boards can drift from objectives until refill/reconcile
  runs; operators must treat lag as expected, not as proof of missing goals.
- **Agent foot-guns remain.** Agents still try to renumber tasks, edit
  protected heaps “to help the daemon,” or treat merge as completion—policy
  must keep failing closed.
- **Harder to change later.** Collapsing the split would require rewriting
  control contracts, goal completion, task identity, and daemon admission—not
  a local doc change.

### Neutral / residual risks

- **Markdown dual-write races.** Concurrent human and daemon edits to boards
  still need leases, conflict policy, and protected-path discipline; the split
  reduces but does not eliminate coordination cost (see ADR-0004/0005 themes).
- **Incomplete heaps.** If authors put all acceptance only on the taskboard,
  the durable layer is empty and the benefits shrink—program conventions and
  review must keep goals in the heap.
- **Status vocabulary drift.** Board `done` vs goal `verified_complete` can
  confuse readers; guides and completion modules must keep the distinction
  explicit.
- **Out of scope here:** How model proposals are admitted (ADR-0002) and how
  leases fence writers (ADR-0004).

## Evidence

| Claim in Decision | Evidence (path, test, or operational check) | Notes |
| --- | --- | --- |
| Objectives hold durable intent and goal lifecycle | `objectives/goal_completion.py` (`GoalState`, provisional vs verified); `objectives/objective_tracker.py`; package README | Two-phase completion; board status is not goal authority |
| Taskboards are projections with stable IDs | `task_sources/task_identity.py`; `taskboard_store.py`; `markdown_task_source.py` | Headers and content identity; not control authority |
| Refill projects work without becoming the objective | `objectives/backlog_refinery.py`; control ops `backlog_refill` / `objective_refine` in `control_contracts.py` | Mutation ops under policy |
| Control binds objective revision | `control/control_contracts.py` — `OperationRequest` fields `objective_id`, `objective_revision` | Identity-bound control |
| Protected paths fail closed for implementation | `todo_daemon/task_execution_policy.py`; `test/api/test_agent_supervisor_implementation_protected_paths.py` | Operator heaps/boards protected |
| Merge ≠ authoritative completion | `todo_daemon/authoritative_completion.py`; `EXECUTION_AND_RECOVERY.md` | Gate recompute; no self-asserted completion |
| Product narrative matches code | `CONTROL_PLANE.md` §3; `AGENT_SUPERVISOR_PHILOSOPHY.md` pillar 1; `packages/task_sources.md` Forbidden row | Documented ownership |
| Identity and graph tests exist | `test/api/test_agent_supervisor_task_identity.py`, `test_agent_supervisor_goal_completion.py`, `test_agent_supervisor_objective_graph.py`, `test_agent_supervisor_taskboard_store.py` | Regression surface |

Evidence classes used: source layout and package exports; deterministic tests;
architecture guides aligned with live packages. Plans alone were not used to
mark this ADR Accepted.

## Verification

How a future reader confirms the decision still holds:

1. **Guides still separate the layers**

   ```text
   rg -n 'durable intent|schedulable projection|taskboards are projections' \
     docs/architecture/agent_supervisor/CONTROL_PLANE.md \
     docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md
   ```

2. **Packages still own distinct roles**

   ```text
   test -d ipfs_accelerate_py/agent_supervisor/objectives
   test -d ipfs_accelerate_py/agent_supervisor/task_sources
   rg -n 'Forbidden|control authority|proof trust' \
     docs/architecture/agent_supervisor/packages/task_sources.md
   ```

3. **Contracts still bind objective identity**

   ```text
   rg -n 'objective_id|objective_revision' \
     ipfs_accelerate_py/agent_supervisor/control/control_contracts.py
   ```

4. **Focused tests still pass (when the full suite is practical)**

   ```text
   python -m pytest \
     test/api/test_agent_supervisor_goal_completion.py \
     test/api/test_agent_supervisor_task_identity.py \
     test/api/test_agent_supervisor_taskboard_store.py \
     test/api/test_agent_supervisor_implementation_protected_paths.py -q
   ```

**Stale signals:** boards treated as completion authority in new APIs;
`task_sources` defining authorization or proof promotion; refill rewriting
protected objective heaps without policy; removal of `objective_revision` from
mutation requests; agents documented as free to renumber foreign tasks.

## Review triggers

- [ ] Source anchors no longer match the Decision statement
- [ ] A recorded negative consequence becomes unacceptable
- [ ] A rejected alternative becomes viable without those costs
- [ ] Security, isolation, lease/fence, or trust-tier changes touch this scope
- [ ] Related guide or package ownership is restructured
- [ ] Goal completion is merged into board status as sole authority
- [ ] Backlog refill is allowed to rewrite protected objective heaps by default
- [ ] Superseding design is Accepted under a new ADR number

When superseding: create a new ADR number; set this file to **Superseded** with
`Superseded-by`; set the successor’s `Supersedes`; do not delete this file.

## Notes (optional)

- Program reservation for this number: `docs/architecture/decisions/README.md`
  (ADR-0001 / DOC-015). Index status refresh is owned by a later closeout task;
  this file’s Status field is authoritative for the decision itself.
- Companion ADR-0002 covers model proposals and evidence admission; do not fold
  that trust-tier decision into this mutability split.
- Operational launch recipes (which paths to pass as
  `--implementation-protected-path`) live in program plans and operator guides,
  not in this ADR’s normative Decision.
