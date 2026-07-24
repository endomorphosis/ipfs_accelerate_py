# Agent Supervisor Architecture

`ipfs_accelerate_py.agent_supervisor` is the control plane for objective-driven
agent work. It turns a durable objective into evidence-backed tasks, schedules
those tasks in isolated implementation lanes, validates the resulting changes,
and records enough state for a later run to resume, repair, or audit the work.

The package is intentionally broader than an LLM wrapper. Models propose plans
and edits, but deterministic parsers, policy checks, validation commands, Git
operations, leases, and evidence receipts decide whether work may advance.

## Current implementation status

The latest supervisor work (23 July 2026) has moved the formal-planning design
into executable, resumable components. The following commits are now part of
the nested `ipfs_accelerate_py` repository:

| Capability | Implementation | Operational meaning |
| --- | --- | --- |
| Shared prover admission | `multi_prover_resources.py` (`REF-291`) | SMT, ATP, kernel, model-checker, protocol, hyperproperty, runtime, validation, model, and artifact-I/O work share one bounded top-level lease. Child processes inherit limits and release capacity on cancellation. |
| Adversarial evidence admission | `formal_planning_adversarial.py` (`REF-292`) | Plan identity, provider boundary evidence, cache freshness, conformance, public-output leakage, and property-specific assurance are checked together. Unknown, forged, stale, or insufficient evidence is rejected. |
| Plan conformance and completion | `formal_plan_conformance.py` (`REF-290`) | Canonical execution events are compared with the accepted plan; unauthorized, reordered, skipped, failed, overridden, or superseded transitions are retained as findings. Completion requires fresh evidence and can reopen a goal. |
| Counterexample-guided repair | `formal_replanner.py` (`REF-289`) | Typed, bounded repair rules produce content-addressed candidates and compact Codex packets. Retry, refinement, candidate, changed-record, and prompt budgets prevent unbounded replanning. |
| Proof-carrying execution | `proof_carrying_planner.py` (`REF-293`) | Compile, verify, implement, scope-check, merge, monitor, and repair nodes run as a durable DAG with paired JSON/DuckDB state. The workflow is replayable and only completes when required assurance is present. |
| Rollout measurement and gates | `formal_planning_metrics.py`, `formal_planning_rollout.py` (`REF-294`) | Cold/warm/parallel benchmark samples measure context reduction, defect detection, proof support, counterexample quality, cache reuse, queue latency, CPU, memory, and throughput before promotion. |

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
