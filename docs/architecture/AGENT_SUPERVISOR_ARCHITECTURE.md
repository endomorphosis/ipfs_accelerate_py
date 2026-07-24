# Agent Supervisor Architecture

`ipfs_accelerate_py.agent_supervisor` is the control plane for objective-driven
agent work. It turns a durable objective into evidence-backed tasks, schedules
those tasks in isolated implementation lanes, validates the resulting changes,
and records enough state for a later run to resume, repair, or audit the work.

The package is intentionally broader than an LLM wrapper. Models propose plans
and edits, but deterministic parsers, policy checks, validation commands, Git
operations, leases, and evidence receipts decide whether work may advance.

## Current implementation status

The latest supervisor work (23-24 July 2026 UTC) has moved the
formal-planning and Leanstral goal-development designs into executable,
resumable components. The following formal-planning commits are now part of
the nested `ipfs_accelerate_py` repository:

| Capability | Implementation | Operational meaning |
| --- | --- | --- |
| Shared prover admission | `multi_prover_resources.py` (`REF-291`) | SMT, ATP, kernel, model-checker, protocol, hyperproperty, runtime, validation, model, and artifact-I/O work share one bounded top-level lease. Child processes inherit limits and release capacity on cancellation. |
| Adversarial evidence admission | `formal_planning_adversarial.py` (`REF-292`) | Plan identity, provider boundary evidence, cache freshness, conformance, public-output leakage, and property-specific assurance are checked together. Unknown, forged, stale, or insufficient evidence is rejected. |
| Plan conformance and completion | `formal_plan_conformance.py` (`REF-290`) | Canonical execution events are compared with the accepted plan; unauthorized, reordered, skipped, failed, overridden, or superseded transitions are retained as findings. Completion requires fresh evidence and can reopen a goal. |
| Counterexample-guided repair | `formal_replanner.py` (`REF-289`) | Typed, bounded repair rules produce content-addressed candidates and compact Codex packets. Retry, refinement, candidate, changed-record, and prompt budgets prevent unbounded replanning. |
| Proof-carrying execution | `proof_carrying_planner.py` (`REF-293`) | Compile, verify, implement, scope-check, merge, monitor, and repair nodes run as a durable DAG with paired JSON/DuckDB state. The workflow is replayable and only completes when required assurance is present. |
| Rollout measurement and gates | `formal_planning_metrics.py`, `formal_planning_rollout.py` (`REF-294`) | Cold/warm/parallel benchmark samples measure context reduction, defect detection, proof support, counterexample quality, cache reuse, queue latency, CPU, memory, and throughput before promotion. |

## Delivery update: 24 July 2026 UTC

The Leanstral-assisted goal-development board is complete: **10/10 tasks are
completed**. The five task commits delivered during the final two-hour
implementation window are:

| Task commit | Delivered capability | Operational boundary |
| --- | --- | --- |
| `c09e0008` (`LEAN-GOAL-006`) | Transactional objective and subgoal materialization | Preview-first; shadow mode cannot mutate the objective heap. |
| `963b13e2` (`LEAN-GOAL-007`) | Fresh code-conformance obligations | Plan evidence cannot be promoted to generated-code proof. |
| `117bb601` (`LEAN-GOAL-008`) | Route-aware capabilities, resource scheduling, cache separation, and metrics | Drafts and authoritative receipts use separate cache and trust paths. |
| `87af152f` (`LEAN-GOAL-009`) | End-to-end shadow lifecycle and restart recovery | Leanstral remains an untrusted proposal source with no completion authority. |
| `388be1d5` (`LEAN-GOAL-010`) | Paired benchmark reports and fail-closed rollout gates | `auto_safe` remains disabled unless explicitly authorized by policy. |

These changes were reconciled with the upstream formal-planning/prover tranche
in merge commit `c84802e1` and published together in `c764be51`. The board
completed on the first implementation attempt for each task, with no blocked
tasks or supervisor restarts during this tranche.

Validation recorded for the delivery:

- The required paired benchmark suite passed **3 tests**.
- The lifecycle and contract regression suite passed **37 tests**.
- The combined analysis, Leanstral, formal-planning, prover-resource,
  scheduler, and daemon suite passed **730 tests** in parallel; the one
  subprocess isolation-sensitive fallback test also passed when rerun serially.
- `compileall`, `git diff --check`, and the combined public-export import check
  passed.

The checked fixture benchmark reports four of five paired quality wins, zero
false completions, zero authority-boundary violations, and stable restart
recovery. These are fixture and policy-gate results, not a claim that arbitrary
Python execution is formally verified. Live Leanstral inference remains
optional, shadow is the default, and promotion still requires the independent
proof, scope, authority, validation, and completion gates described below.

These modules provide the execution surface for the design below; they do not
make arbitrary Python formally verified. Provider conformance, reviewed
obligation templates, exact tree and policy identities, and the configured
assurance threshold remain prerequisites for enforcement.

This document is an orientation guide to the implementation. The more detailed
objective scanner description is in
[`agent_supervisor_objective_graph.md`](../agent_supervisor_objective_graph.md),
and the formal-planning and prover design is in
[`AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md`](AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
and
[`AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md`](AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md).

## Design goals

The supervisor is designed around six constraints:

1. **Objectives are durable.** A Markdown objective heap is a human-readable
   source of intent; generated todos and derived graphs are projections, not a
   replacement for that intent.
2. **Work is reproducible.** Task identities, plans, commands, artifacts, and
   state transitions use canonical JSON, content identities, and versioned
   schemas where possible.
3. **Lanes are isolated.** Parallel bundles use separate state directories and
   worktrees. A worker should not accidentally share mutable task state or
   uncommitted files with another worker.
4. **Evidence is explicit.** A passing command, proof receipt, scan result, or
   merge event is recorded with provenance and freshness rather than inferred
   from a model's prose.
5. **Failure becomes work.** Repeated implementation, validation, or merge
   failures are bounded by policy and converted into repair or follow-up tasks.
6. **Assurance is scoped.** A bounded model check, runtime trace, solver
   candidate, and kernel-checked theorem are different kinds of evidence; none
   is silently promoted to another.

## System view

The main flow is:

```text
objective heap / operator request
              |
              v
   objective tracker + graph scanner
              |
              +--> AST, path, text, and vector evidence datasets
              +--> objective graph and bundle index
              v
     todo shards / task-queue payloads
              |
              v
 bundle supervisor -> implementation supervisor -> implementation daemon
                              |                         |
                              |                         +--> isolated worktree
                              |                         +--> LLM/Codex proposal
                              |                         +--> validation workspace
                              |                         +--> accepted/failed artifacts
                              v
                 proof, policy, lease, and resource gates
                              |
                              v
                 merge train / conflict repair / receipts
                              |
                              v
       completion decision, metrics, event log, and next scan
```

The flow is iterative. A completed task can produce new evidence for its goal;
a failed task can produce a bounded repair task; and a reopened goal can return
to the active queue. No component should assume that one pass proves the whole
repository or that an empty current backlog means the ultimate objective is
complete.

## Subsystem boundaries

### 1. Objective and goal lifecycle

`objective_tracker.py` owns the durable objective heap and compatibility with
older Markdown field spellings. `objective_graph.py` parses goals, computes
dependencies and priorities, scans repository evidence, and emits objective
findings and generated work proposals. `objective_daemon.py` is the CLI/runtime
bridge that performs a scan, writes artifacts, and optionally submits bundles
to the existing P2P task queue.

`goal_completion.py` is the lifecycle authority. Its normalized states are:

```text
active -> provisionally_complete -> verified_complete
   |             |                       |
   v             v                       v
blocked       reopened <----------------+
   ^             ^
   +-- analysis_inconclusive
```

The exact transition guards live in the module; the important concept is that
completion is evidence-sensitive and a completed goal may be reopened when
fresh evidence contradicts it or required proof becomes stale. `goal_coverage.py`
and the completion projection in `scheduler_metrics.py` reduce the state into
operator-facing coverage and health summaries.

### 2. Discovery, planning, and task identity

The scanner uses several evidence channels: tracked paths, exact text, parsed
symbols/AST records, and deterministic token-vector similarity. Large AST
payloads are stored as JSONL/dataset artifacts instead of being embedded in
todo prose. `todo_vector_index.py` records related tasks, merge keys, clusters,
and candidate relationships so a worker can select adjacent work without
reloading the entire repository.

`task_identity.py` supplies canonical task and bundle identities. These are
used for duplicate suppression, retry accounting, merge decisions, and
cross-process correlation. `conflict_graph.py` identifies file and semantic
overlap; `plan_evaluator.py` and `task_proposal_router.py` score or route
candidate plans while retaining deterministic fallback behavior.

### 3. Formal plan and policy gate

The formal planning family converts a goal/task into a typed work plan:

- `formal_planning_contracts.py` defines the plan vocabulary and assurance
  requirements.
- `formal_plan_context.py` builds bounded context from repository, task,
  resource, dependency, policy, and evidence records.
- `formal_plan_compiler.py` produces a canonical plan capsule.
- `formal_plan_validator.py` rejects malformed, contradictory, unauthorized,
  unsafe, or dependency-invalid plans.
- `logic_translation_validation.py` records what semantics were preserved or
  approximated while translating a plan into a logic representation.
- `authorization_logic.py`, `lease_coordination.py`, and
  `resource_scheduler.py` enforce authority, fencing, capacity, and admission
  rules at execution time.

The plan gate is not an assertion that arbitrary Python has been formally
verified. It is a bounded, typed check over reviewed obligations and declared
assumptions.

### 4. Prover and assurance layer

The package treats proof providers as capability-scoped adapters. The matrix in
`prover_matrix_registry.py` describes providers, identities, commands, fixtures,
and self-tests. `formal_verification_capabilities.py` probes what is actually
available on the host. `prover_conformance.py` quarantines providers whose
translation or fixture suite is not conformant.

`multi_prover_router.py` maps property kinds to prover lanes and records every
attempt. Supporting components include:

- `supervisor_state_model.py` for deterministic finite transition schemas and
  bounded TLC/Apalache model-check receipts;
- `kernel_verification.py` and `formal_verification_provider.py` for provider
  boundaries and trusted checking;
- `leanstral_proof_provider.py` for bounded, untrusted proof suggestions;
- `hyperproperty_verification.py` for non-interference-style properties;
- `proof_context.py`, `proof_scope_index.py`, and
  `code_proof_obligations.py` for scope and obligation derivation;
- `prover_evidence_store.py`, `formal_verification_cache.py`,
  `proof_attestation.py`, and `proof_metrics.py` for durable receipts,
  freshness, cache identity, and reporting.

The current planning rollout adds two gates around this layer. The adversarial
gate evaluates evidence at the property boundary and derives the authoritative
assurance from typed evidence rather than accepting a provider-declared level.
The rollout gate then compares a benchmark report with the reviewed policy in
one of three modes: `shadow` records diagnostics without changing dispatch,
`canary` allows a limited lane when thresholds pass, and `enforcement` requires
the configured minimum assurance. An expiring, content-addressed override may
waive one exact lane, but it is itself recorded as evidence and cannot weaken
the underlying trust rules.

The trust vocabulary is deliberately non-linear: `solver_candidate`,
`bounded_model_checked`, `runtime_checked`, `protocol_checked`,
`kernel_verified`, and `attested` describe different evidence types. A runtime
trace cannot substitute for a theorem, and a bounded model check must retain its
finite bounds.

### 4a. Leanstral goal-development lifecycle

`leanstral_goal_lifecycle.py` is the reviewed integration boundary for using
Leanstral to develop a frozen goal into candidate formal work. Construct it
with `build_configured_leanstral_goal_lifecycle_supervisor`; the factory
requires an explicit state directory and defaults to **shadow** mode when the
caller does not select a mode. This default is part of the safety contract, not
a deployment convention.

The lifecycle is deliberately staged:

```text
frozen root + bounded context
              |
              v
  one or more untrusted candidate drafts
              |
       schema/type/policy gate
              |
              v
   deterministic proposal receipt
              |
   independent refinement/proof portfolio
              |
              v
 objective preview or gated materialization
              |
   implementation conformance receipts
              |
              v
       goal-completion authority
```

The stages have separate trust boundaries:

- The model provider is an untrusted proposal source. It cannot issue proof,
  admission, implementation-conformance, or completion receipts. A malformed
  response, unavailable provider, wrong return type, or provider exception is
  converted into a stable fallback result rather than being treated as
  acceptance.
- `LeanstralGoalDevelopmentContextBuilder` supplies a bounded, redacted
  context. It includes only the configured number of templates, gap records,
  AST summaries, capability records, counterexamples, and reusable receipt
  summaries. Canonical source, proof bodies, secrets, and unrestricted
  repository data do not cross the model boundary. The goal root and formal
  assumptions are frozen inputs and remain frozen during counterexample repair.
- Contract validators and the deterministic proposal selector decide which
  draft can advance. Multiple candidates are isolated attempts; the stable
  selector prefers the greatest valid proposal coverage and then canonical
  draft identity. Candidate diversity never becomes voting-based proof.
- Refinement evidence is produced outside the drafting provider. A bounded
  counterexample may trigger a bounded repair round, but only an independent,
  authoritative verifier can accept the repaired formal plan. Bounded,
  assumed, unsupported, and inconclusive evidence retain those statuses.
- `objective_daemon.materialize_admitted_objective_work` is the only bridge
  from an admitted proposal to objective work. An admission receipt means that
  the policy gate allowed materialization; it is not a proof of the goal.
- `code_proof_obligations.py` binds implementation evidence to the exact goal
  root, plan, source tree, source paths, tests, and proof receipts.
  `goal_completion.py` remains the sole completion authority. Receipts bound to
  an older tree or root are stale and reopen completion instead of being
  silently reused.

#### Modes and mutation authority

| Mode | Provider and validation behavior | Permitted durable effects |
| --- | --- | --- |
| `off` | The route is disabled; no development invocation is valid. | None. |
| `shadow` (default) | Run bounded candidates, deterministic validation, independent checks supplied by the caller, and an objective materialization preview. | Audit journal, recoverable run state, and proof/operational metrics only. The objective heap, generation ledger, implementation tree, and completion state must remain byte-for-byte unchanged. |
| `assist` | Produce a reviewed proposal and admission decision for an operator. | Review/generation records allowed by objective policy; no automatic objective mutation or completion. |
| `repair_only` | Accept only a bounded repair of an existing plan, preserving the frozen goal root and assumptions. | Repair evidence and review records; no autonomous completion. |
| `auto_safe` | Apply the proposal only after deterministic validation, authoritative proof, freshness, capability, policy, and admission gates all pass. | Objective work may be materialized under the objective daemon's atomic update contract. Completion still requires separately bound implementation evidence. |

`auto_safe` must not be enabled merely because the model generated valid JSON.
It requires a fresh goal/tree binding, no undeclared assumptions or unsupported
semantics, healthy required capabilities, independent authoritative receipts,
and any configured lease/resource checks. Callers should promote through
shadow and assist using observed receipts before enabling it for a narrowly
scoped objective class.

#### Capabilities, resources, and controls

Model execution and proof execution use different capability and resource
classes. Startup should probe the configured model route, legal preprocessing
and codec support, independent verifier route, kernel checker, and any required
solver or model-checker. Effective context is the minimum safe budget after
route, server, and model reserves; a server-advertised maximum is not itself a
safe prompt budget. Provider time, output, context, concurrency, and network
limits remain explicit inputs to the bounded provider and scheduler.

The configured lifecycle bounds candidate count to one through eight and
records every candidate attempt, including fallbacks, in proof metrics. Its
state directory contains:

- an append-only, fsynced JSONL audit journal;
- an atomically replaced latest-run state document; and
- a proof metrics projection with attempt, validation, fallback, acceptance,
  latency, and availability observations.

Run records are schema-validated and content-addressed. Recovery first validates
the latest state and then scans the journal backward for the newest valid
record, so a torn or corrupt projection does not erase the audit trail. In
shadow mode every record also contains before/after digests and explicit
`objective_heap_unchanged` and `completion_state_unchanged` assertions. An
unexpected mutation is a failed run, not an advisory metric.

Receipt stores and caches retain the same separation as the live path:
proposal validation, bounded counterexamples, independent proof, admission,
implementation conformance, and completion are distinct schemas and cache
namespaces. Reuse requires exact schema, provider/version, goal-root, plan,
source-tree, policy, and capability bindings appropriate to that receipt.

### 5. Execution daemons and isolated lanes

The `todo_daemon` package contains the reusable implementation loop. Its
responsibilities are split so that policy and process supervision do not become
one opaque loop:

- `engine.py` parses tasks, materializes proposals, creates validation
  workspaces, runs commands, and promotes accepted files.
- `implementation_daemon.py` owns the task/pass state, selection, retries,
  active phases, heartbeat, LLM invocation, and validation handoff.
- `implementation_supervisor.py` and `supervisor.py` monitor heartbeats,
  process liveness, stuck phases, and restart/repair decisions.
- `lifecycle_wrapper.py`, `core.py`, and `cli.py` provide reusable process
  lifecycle and status interfaces.
- `worktrees.py`, `checkout_lock.py`, and `leased_lane.py` keep Git and lane
  mutations isolated and fenced.

`bundle_supervisor.py` converts bundle indexes into one lane per bundle;
`multi_supervisor_runner.py` manages several configured supervisor tracks. The
runner modules (`implementation_daemon_runner.py` and
`implementation_supervisor_runner.py`) are project-binding adapters: they add
defaults and hooks without duplicating the daemon engine.

### 6. Artifacts, events, and durable state

The supervisor has several intentionally different persistence surfaces:

| Surface | Purpose | Representative modules |
| --- | --- | --- |
| Objective heap | Human-readable intent and goal metadata | `objective_tracker.py` |
| Todo Markdown | Worker-facing executable queue | `taskboard_store.py`, `objective_graph.py` |
| JSON/JSONL artifacts | Bounded scan, proposal, validation, and completion records | `artifact_store.py`, `scan_receipts.py` |
| Event log | Append-only operational history | `event_log.py` |
| Task/daemon state | Resume, heartbeat, retry, active phase | `duckdb_state.py`, `todo_daemon/implementation_daemon.py` |
| DuckDB stores | Queryable task, evidence, and proof projections | `dataset_store.py`, `prover_evidence_store.py` |
| Vector/AST index | Related-work and bundle selection hints | `todo_vector_index.py` |
| Git/worktree state | Candidate changes and merge checkpoints | `merge_checkpoint.py`, `merge_queue.py` |

State files are operational projections and may be repaired or migrated. They
must not be treated as the sole source of truth for acceptance; accepted work
should have validation and provenance sidecars or receipts.

### 7. Merge, recovery, and maintenance

`merge_train.py`, `merge_queue.py`, `merge_resolver.py`,
`merge_conflict_repair.py`, and `merge_checkpoint.py` coordinate promotion of
independent lane results. `llm_merge_resolver_fallback.py` can provide a
bounded external suggestion, but Git checks and validation remain the gate.

`backlog_refinery.py` refills low or drained queues from objective gaps and
codebase findings. `objective_task_janitor.py` handles stale or inconsistent
objective work. `codex_failure_policy.py` classifies repeated failures, while
`submodule_degradation.py` records when nested repositories cannot provide their
normal validation surface. `git_gc.py` and cleanup helpers remove stale
worktrees or branches only under explicit, scoped policy.

`runtime_temporal_monitor.py`, `supervisor_watchdog.py`, `analyzer_health.py`,
and `scheduler_metrics.py` observe the live system. They report freshness,
capacity, deadlines, retries, proof invalidation, and terminal outcomes; they do
not silently mark work complete.

The proof-carrying planner composes these pieces into a durable workflow. Its
terminal result is `completed` only when plan compilation and bounded plan
validation pass, all implementation scopes and merges are accepted, the
runtime trace is accepted, and the required authoritative assurance is present.
Otherwise it returns a rejected, failed, or blocked result with replayable
decisions and evidence. A runtime counterexample is eligible for bounded repair
before finalization, and repaired counterexamples remain linked to the original
finding.

## Typical operating sequence

1. Create or update the objective heap and record acceptance criteria.
2. Run the objective daemon. Inspect `objective_graph.json`, discovery datasets,
   bundle indexes, and generated task identities.
3. Review or validate the proposed formal plan and required assurance. If a
   provider is unavailable, the result should be explicitly `unsupported` or a
   deterministic fallback—not an implicit pass.
4. Start a bundle supervisor in dry-run mode, then launch isolated lanes when
   the plan and resource budget are acceptable.
5. Let the implementation daemon select a ready task, obtain its lease, create
   a worktree, request a proposal, run validation, and persist success/failure
   sidecars.
6. Merge through the merge train. Resolve conflicts with bounded evidence and
   re-run affected validation; do not infer correctness from a clean merge.
7. Reconcile goal completion from fresh evidence. A goal with missing criteria,
   stale receipts, unhealthy analyzers, or unsatisfied exhaustion quorum remains
   open, blocked, or inconclusive.
8. Inspect event logs and metrics before starting another pass. A drained queue
   may trigger bounded objective/codebase refill, but refill must not create
   unbounded duplicate work.

## The theory behind the design

### The supervisor is a feedback controller

The useful mental model is a feedback-control loop, not an autonomous chatbot:

```text
desired state: acceptance criteria + policy + resource budget
        |
        v
  planner / scheduler ----> worker action ----> repository + runtime
        ^                                          |
        |                                          v
        +--------- observations: tests, receipts, events, health
```

The objective and its acceptance criteria describe the desired state. The
planner selects an action that should reduce the gap. The worker performs the
action in a controlled lane. Scanners, validators, proof providers, and
watchdogs observe the result. The next reconciliation cycle uses those
observations to decide whether to continue, repair, block, or reopen the goal.

This explains several otherwise surprising choices in the code:

- **Scans are repeatable.** A controller needs comparable observations across
  cycles, so scanners use deterministic identities, bounded outputs, and
  explicit analyzer health.
- **The objective is separate from the queue.** The desired state should not be
  rewritten merely because the current actuator queue is empty or a worker
  failed.
- **Completion is a decision, not a counter.** A zero open-task count is only
  one observation. Completion also requires coverage, fresh evidence, healthy
  analyzers, and the configured exhaustion quorum.
- **Repair is feedback.** A failure is useful when it becomes a typed signal
  that changes the next plan; blind retries simply repeat the same control
  input and can create an infinite loop.

### Three graphs, three questions

The package contains several graphs because one graph cannot answer every
question:

| Graph | Question | Main representation |
| --- | --- | --- |
| Objective graph | What outcomes and evidence are still needed? | goals, subgoals, evidence terms, dependencies |
| Task/conflict graph | Which executable items can run together? | task identities, prerequisites, file/symbol conflicts, bundle clusters |
| Evidence/proof graph | Why should a result be trusted? | source/tree identities, receipts, scopes, provider bindings, freshness |

Collapsing these graphs would create two common errors. First, a task that is
easy to schedule could be mistaken for a goal that is complete. Second, a proof
receipt for one source scope could be reused for a different candidate merely
because the task title is similar. The separate projections let each graph use
the invariants appropriate to its job while sharing canonical identities.

### State machines prevent ambiguous progress

The supervisor represents progress as transitions with guards and effects. In
the formal planning vocabulary, a transition is described by:

```text
event + actor + preconditions + effects + evidence requirements
```

For example, “accept implementation” is not equivalent to “the model returned
text.” It requires a current task lease, the expected worktree/fence, a valid
proposal, successful configured validation, and accepted artifacts. The
transition then records progress and releases or advances the relevant lease.

`formal_planning_contracts.py` makes this vocabulary explicit with actors,
goals, subgoals, plan tasks, events, fluents, preconditions, effects, norms,
temporal constraints, evidence requirements, and plan assurance. The
`FormalWorkPlan` validator checks referential integrity and acyclicity before a
plan reaches a provider. `supervisor_state_model.py` translates a reviewed
finite transition schema into TLA+ and records bounded checking results.

The default state model checks safety properties such as:

- `UniqueAcceptance`: one logical task cannot be accepted twice;
- `FencingSafety`: an old worker cannot commit after its lease/fence is stale;
- `DependencyOrder`: prerequisites are accepted before dependents;
- `IdempotentMerge`: replaying a merge decision does not duplicate its effect;
- `CapacitySafety`: admitted work stays within declared capacity; and
- `EvidenceGates`: terminal transitions cannot skip required evidence.

It also checks liveness properties such as bounded progress and terminal
outcomes. These are bounded experiments over a generated finite model, not a
claim that arbitrary Python execution has been proved correct.

### Leases, fencing, and worktrees are a distributed-systems protocol

Multiple daemons may observe the same queue, and processes can pause, crash, or
be restarted. A simple `locked = true` flag is not enough: a delayed worker may
resume after another worker has taken over. The supervisor therefore combines:

1. a task lease with an owner and expiry;
2. a monotonically changing fencing token/generation;
3. heartbeats that prove the owner is still live; and
4. validation of the token at terminal operations and merge boundaries.

The invariant is: **only the current owner with the current fence may perform
state-changing work**. Expiry makes capacity reclaimable; fencing makes stale
work harmless. `lease_coordination.py` persists grants, heartbeats, terminal
receipts, and conflicts; `leased_lane.py` adapts the same idea to a lane;
`checkout_lock.py` protects Git operations; and the daemon state records the
active phase so a watchdog can distinguish slow work from a dead process.

Worktrees provide filesystem isolation, but they are not the authority by
themselves. A worktree can be deleted, reused, or left dirty. The lease/fence
protocol and event/receipt history are what make a promotion decision
auditable.

### Scheduling is admission control, not just priority sorting

Priority answers “which eligible task is preferred?” Admission answers “may
this task run at all right now?” The resource scheduler performs both concepts
separately. It normalizes host capacity, provider concurrency, process slots,
resource classes, and proof-specific pools before issuing a reclaimable
`ResourceAdmissionLease`.

This separation matters when a high-priority proof task needs a scarce kernel
slot, while several ordinary implementation tasks need only CPU. Running the
highest-priority item without admission control can starve the proof lane or
overcommit the model provider. Conversely, refusing to schedule because a
provider is temporarily full should not turn into a permanent task failure;
the scheduler records a wait/admission reason and retries when capacity is
released.

The same principle applies to LLM routing. `resource_scheduler.py` normalizes
provider capacity and `formal_verification_capabilities.py` probes operation-
specific readiness. “The executable exists” is not the same as “this provider
can perform this translation under the requested isolation and timeout.”

### Evidence is a content-addressed claim

An evidence record should answer four questions:

1. **What** was observed or checked?
2. **Where** did it come from (repository/tree/path/provider)?
3. **When** was it valid, and is it still fresh?
4. **How** can another process reproduce or verify it?

That is why receipts contain schema versions, canonical payloads, source
identity, artifact paths or CIDs, and bounded projections. `scan_receipts.py`
computes a deterministic receipt identity from canonical content and rejects a
persisted artifact whose bytes no longer match its identity. `artifact_store.py`
keeps large payloads outside status files; `event_log.py` records compact
operational facts; and proof stores bind a receipt to its scope and provider.

The distinction between a full artifact and a compact projection is deliberate.
Status and heartbeat payloads must remain bounded and low-cardinality, while an
auditor must still be able to follow the content identity to the full scan,
validation output, or proof receipt.

### Why LLM output remains below the acceptance boundary

An LLM is valuable at proposing decomposition, implementation edits, repair
strategies, or proof candidates. It is not a stable authority for repository
state. Prompts can omit context, models can hallucinate files, and a plausible
answer can violate a lease, policy, or hidden dependency.

The supervisor therefore uses a two-stage architecture:

```text
model proposal -> schema/identity checks -> policy/plan checks
               -> isolated materialization -> deterministic validation
               -> evidence and merge gates
```

`todo_daemon.llm` and provider adapters are intentionally replaceable. The
proposal is bounded, normalized, and associated with the current task and
worktree. A deterministic fallback can keep the queue moving, but fallback
usage is recorded as inconclusive or lower-assurance evidence; it is not
silently presented as a model success.

### Retry budgets are a termination argument

Retries are useful for transient network, provider, or test failures. They are
dangerous when the underlying failure is deterministic. The implementation
supervisor tracks failure classes and budgets for implementation, validation,
and merge stages. Once a class crosses its budget, the source task is blocked
and a follow-up repair task carries the diagnostic evidence.

This gives the system a practical termination property: one failing input
cannot consume all worker time forever. It also preserves information that a
blind retry would destroy—the original command, output excerpt, phase, attempt
number, and suggested repair. `backlog_refinery.py` applies the same idea to
drained queues and recurring codebase findings.

### Completion is a quorum over independent observations

The completion gate treats “done” as a conjunction of independent dimensions,
not a single boolean:

```text
complete iff
  acceptance criteria covered
  AND required tasks terminal
  AND evidence is fresh and bound to the current tree
  AND analyzers are healthy
  AND exhaustion quorum is satisfied when required
  AND no contradiction/reopen condition is present
```

The quorum is especially important for negative claims such as “no matching
implementation remains.” One scanner run can miss files because of a parser
failure, ignored submodule, stale checkout, or unavailable analyzer. The
receipt model records distinct channels and rejects duplicate or mismatched
members, so repeated identical scans do not manufacture confidence.

## Reading the implementation as a set of contracts

The most efficient way to understand a new module is to identify which
contract it owns:

| Contract type | What to inspect | Failure if violated |
| --- | --- | --- |
| Identity | `task_identity.py`, canonical JSON helpers | duplicate or cross-scope work |
| Lifecycle | `goal_completion.py`, daemon state, state model | impossible or ambiguous transitions |
| Authority | `authorization_logic.py`, actor/role records | unauthorized work or merge |
| Exclusivity | `lease_coordination.py`, fences, worktrees | stale worker mutation |
| Capacity | `resource_scheduler.py` | overcommitment or starvation |
| Semantics | formal plan contracts and translation validation | proving the wrong statement |
| Evidence | artifacts, receipts, proof stores | unverifiable completion claim |
| Recovery | failure policy, refinery, merge repair | infinite retry or lost diagnostics |
| Observation | event log, watchdog, temporal monitor, metrics | undetected stuck or stale state |

When extending the package, start with the contract and its invariant, then
locate the projection that persists it, the event that records it, and the test
that exercises it. Avoid adding a shortcut directly to a daemon loop: it will
usually bypass identity, fencing, evidence, or migration behavior.

## Worked example: closing one objective gap

Suppose the objective says “serve model discovery through the MCP endpoint.”
The intended lifecycle is:

1. The objective scanner finds that the acceptance evidence is absent and emits
   a task with a goal ID, missing-evidence terms, validation commands, and a
   canonical task identity.
2. The plan compiler adds actors, affected paths, dependencies, resource
   requirements, and an evidence requirement for an endpoint-level test.
3. The plan validator rejects the plan if its dependency graph cycles, its
   actor lacks authority, or its required provider is unavailable.
4. The bundle supervisor admits the task to an isolated lane and the daemon
   obtains a lease/fence before asking the model for an edit.
5. The daemon materializes the proposal, runs the configured tests in a clean
   validation workspace, and persists the output and command receipt.
6. The merge train checks the current fence and target tree, applies the
   accepted change, and records a merge checkpoint.
7. The objective tracker reconciles the new endpoint test and source evidence.
   If all required dimensions are fresh, the goal can become provisionally or
   verified complete; otherwise it remains active, blocked, or inconclusive.

Notice that the model is only one participant in step 4. The evidence that
closes the objective comes from the validated repository and the recorded
receipt, not from the model's assertion that the feature exists.

## What the architecture does not promise

The supervisor is a reliability and assurance control plane, not a universal
program verifier or a guarantee of eventual success. In particular:

- arbitrary Python behavior is not formally verified merely because a plan
  passed a logic check;
- a provider capability report is not proof evidence;
- a bounded model check says nothing beyond its recorded bounds and assumptions;
- a clean Git merge does not imply tests or acceptance criteria passed;
- a healthy heartbeat does not prove useful progress; and
- a drained queue does not prove the objective is complete.

These limits are features of the design. Keeping claims narrower than the
available evidence is what allows the supervisor to combine probabilistic model
proposals with deterministic engineering controls without confusing the two.

## Extension points

New integrations should prefer these boundaries:

- Add an evidence-producing scanner or validator behind a versioned receipt.
- Add a prover through the capability registry and conformance fixtures rather
  than calling a CLI directly from scheduling code.
- Add a task source through `objective_graph`/`backlog_refinery` so identities,
  deduplication, and bundle metadata are preserved.
- Add an LLM through the existing router/provider boundary and keep its output
  in the proposal tier until deterministic checks accept it.
- Add a scheduler policy by extending typed resource/lease contracts and their
  metrics, not by mutating daemon state ad hoc.
- Add persistence through an artifact or projection store with a schema version,
  canonical identity, and migration behavior.

## Operational caveats

The package contains compatibility layers for older todo boards, wrappers, and
provider APIs. A module being importable does not mean every optional backend is
available or conformant. Capability probes and self-tests are therefore part of
normal startup for formal lanes. Similarly, model-generated plans and Leanstral
outputs are proposals; they become actionable only after schema, policy,
validation, and evidence gates pass.

For a compact implementation inventory, start with:

- `objective_tracker.py`, `objective_graph.py`, `objective_daemon.py`;
- `formal_plan_context.py`, `formal_plan_compiler.py`,
  `formal_plan_validator.py`;
- `todo_daemon/engine.py`, `todo_daemon/implementation_daemon.py`,
  `todo_daemon/implementation_supervisor.py`;
- `bundle_supervisor.py`, `resource_scheduler.py`, `lease_coordination.py`;
- `artifact_store.py`, `event_log.py`, `scan_receipts.py`;
- `merge_train.py`, `merge_queue.py`, `merge_resolver.py`; and
- `prover_matrix_registry.py`, `multi_prover_router.py`,
  `prover_evidence_store.py`.

## Appendix: complete module map

The following map is intended to make the package discoverable when reading the
source tree. The names are grouped by the contract they primarily implement;
many modules intentionally participate in more than one group.

**Objective, backlog, and identity:** `objective_tracker.py`,
`objective_graph.py`, `objective_daemon.py`, `objective_task_janitor.py`,
`backlog_refinery.py`, `goal_completion.py`, `goal_coverage.py`,
`task_identity.py`, `taskboard_store.py`, `persistent_task_queue.py`,
`todo_vector_index.py`, `dataset_store.py`, `code_evidence_graph.py`,
`conflict_graph.py`, `task_proposal_router.py`, `plan_evaluator.py`.

**Planning, logic, and policy:** `formal_planning_contracts.py`,
`formal_plan_context.py`, `formal_plan_compiler.py`,
`formal_plan_validator.py`, `formal_logic_vocabulary.py`,
`logic_translation_validation.py`, `authorization_logic.py`,
`interface_contract_codegen.py`, `validation_commands.py`,
`validation_scheduler.py`, `supervisor_state_model.py`.

**Proof scope, providers, and assurance:** `code_proof_obligations.py`,
`proof_obligation_templates.py`, `proof_context.py`, `proof_scope_index.py`,
`proof_scheduler.py`, `proof_fallbacks.py`, `proof_metrics.py`,
`formal_verification_contracts.py`, `formal_verification_policy.py`,
`formal_verification_provider.py`, `formal_verification_cache.py`,
`formal_verification_capabilities.py`, `prover_matrix_registry.py`,
`prover_conformance.py`, `multi_prover_router.py`,
`prover_evidence_store.py`, `proof_attestation.py`,
`kernel_verification.py`, `leanstral_proof_provider.py`,
`ipfs_datasets_logic_provider.py`, `hyperproperty_verification.py`.

**Execution, scheduling, and coordination:** `bundle_supervisor.py`,
`multi_supervisor_runner.py`, `implementation_daemon_runner.py`,
`implementation_supervisor_runner.py`, `resource_scheduler.py`,
`lease_coordination.py`, `leased_lane.py`, `checkout_lock.py`,
`wrapper_utils.py`, `analyzer_health.py`, `supervisor_watchdog.py`,
`runtime_temporal_monitor.py`, `scheduler_metrics.py`.

**Artifacts, events, and lifecycle state:** `artifact_store.py`,
`scan_receipts.py`, `event_log.py`, `duckdb_state.py`,
`proof_metrics.py`, `submodule_degradation.py`, `git_gc.py`.

**Merge and recovery:** `merge_train.py`, `merge_queue.py`,
`merge_checkpoint.py`, `merge_resolver.py`, `merge_conflict_repair.py`,
`llm_merge_resolver_fallback.py`, `codex_failure_policy.py`.

**The reusable todo-daemon runtime:** `todo_daemon/__init__.py` exposes the
public lifecycle/runtime helpers and `todo_daemon/__main__.py` provides module
execution. `todo_daemon/core.py` provides process
and state primitives; `engine.py` provides task/proposal/validation mechanics;
`implementation_daemon.py` and `implementation_supervisor.py` provide the
worker and watchdog loops; `supervisor.py`, `supervisor_loop.py`,
`supervisor_runtime.py`, `runner.py`, `app.py`, `cli.py`, and `wrapper.py`
provide lifecycle and command-line composition; `worktrees.py`, `git_utils.py`,
`file_replacement.py`, and `auto_commit.py` provide repository operations;
`artifacts.py`, `history.py`, `diagnostics.py`, `status.py`, and
`deterministic_fallback.py` provide observability and recovery; and
`context.py`, `llm.py`, `llm_defaults.py`, `plans.py`, `logic_port.py`,
`registry.py`, `specs.py`, `task_board.py`, `legal_parser.py`,
`legal_parser_daemon.py`, and `typescript.py` provide context, language-specific
adapters, registries, and task formats.

`__init__.py` is a public re-export surface, not the execution coordinator.
Importing a symbol from it should be understood as API convenience; the
invariants remain owned by the implementation module named above.
