# Agent Supervisor Self-Improvement Plan

## Purpose

This program improves the `ipfs_accelerate_py.agent_supervisor` as a bounded
feedback controller. The target is not a larger autonomous prompt. The target
is a system that converts durable goals into compact evidence, validated plans,
conflict-aware work, accepted changes, and fresh completion receipts while
spending fewer model tokens and making better use of local software.

The executable work is in
[agent_supervisor_self_improvement.todo.md](agent_supervisor_self_improvement.todo.md).
The durable source of intent is
[agent_supervisor_self_improvement.objectives.md](agent_supervisor_self_improvement.objectives.md).
The objective heap remains authoritative if generated tasks are later
deduplicated, split, bundled, or refilled.

## Current baseline and principal gaps

The repository already contains strong pieces:

- objective, task, dependency, conflict, coverage, and proof-scope graphs;
- bounded analysis contracts, an incremental AST index, multi-signal retrieval,
  and a content-addressed analysis cache;
- formal plan compilation, plan validation, Leanstral goal development, Hammer
  and multi-prover routing, code-conformance obligations, and completion gates;
- durable task identities, leases, resource admission, worktree isolation,
  validation scheduling, merge recovery, watchdogs, and refill scanners; and
- standalone Python and console entry points for objective generation,
  implementation supervision, backlog refinement, and artifact queries.

The audit for this plan found several integration gaps:

1. `analysis_cache.py`, `analysis_ast_index.py`,
   `analysis_retrieval.py`, and `analysis_contracts.py` exist, but the main
   objective/refill/planning path does not compose them as one analysis
   pipeline.
2. Implementation context is partially bounded, but token estimation is still
   heuristic, the todo-vector budget is fixed, and some proposal paths admit up
   to 40,000 roadmap characters instead of compiling a stage-specific context.
3. Planning quality, token cost, cache reuse, validation cost, and accepted
   change quality are measured in separate places. There is no end-to-end cost
   per accepted task or evidence-gain-per-token metric.
4. The `ipfs_datasets_py` integration is strongest at the Hammer proof
   boundary. GraphRAG, dataset queries, premise selection, legal/logic analysis,
   and other reusable reasoning are not exposed through one bounded
   capability-negotiated supervisor provider.
5. Resource scheduling exists, but analysis, inference, proof, validation, and
   merge work do not yet share one adaptive stage model. Raising a worker count
   can still duplicate model state, cache misses, or conflicting work instead
   of increasing throughput.
6. Goal refinement and refill are capable but mostly threshold-driven. They
   need a closed-loop policy that reacts to counterexamples, stale evidence,
   repeated validation failures, queue shape, and measured planning quality.
7. The product MCP server and unified `ipfs-accelerate` CLI have no
   first-class agent-supervisor category. Operators must compose standalone
   scripts, and Python callers do not have one stable control service.
8. Objective evidence discovery can currently treat semantically similar
   planning or task-board prose as if it were implementation evidence. This
   makes an apparently complete goal possible without a qualifying test,
   proof, benchmark, or runtime receipt.
9. The implementation and objective paths currently overload task prefix as
   both a Markdown heading prefix and a task identifier prefix. Heading-style
   input can therefore produce malformed doubled headings during refill.

The first implementation tranche should integrate and benchmark existing
components before introducing replacements.

## Target control loop

```text
durable objective heap
        |
        v
coverage and freshness gap
        |
        v
content-addressed local analysis
  AST + dependency + retrieval + prior receipts
        |
        +----> optional ipfs_datasets_py analysis/proof candidates
        |
        v
token-budgeted context capsule
        |
        v
deterministic baseline + bounded plan branches
        |
        v
schema, authority, feasibility, and evidence validation
        |
        v
task sizing + dependency DAG + conflict-aware bundles
        |
        v
resource-admitted parallel lanes
        |
        v
patch validation + impact tests + semantic/proof checks + merge gate
        |
        v
fresh receipts and goal reconciliation
        |
        +----> complete, repair, reopen, or bounded refill
```

Every arrow produces a versioned receipt. LLM and `ipfs_datasets_py` reasoning
results are candidates. They do not become goal, merge, proof, or completion
authority merely because they are structured or confident.

## Design principles

### One source of intent

The objective heap owns desired outcomes and acceptance evidence. Todo
Markdown, bundle indexes, vector indexes, MCP responses, CLI output, and
DuckDB tables are projections. Rebuilding a projection must not create a
second goal identity or silently strengthen the objective.

### Software before model tokens

Use deterministic parsers, AST indexes, dependency graphs, caches, prior
receipts, test-impact maps, and theorem provers before asking an LLM to infer
the same facts. Model calls should address residual ambiguity, synthesis, or
repair, and receive evidence references rather than repository dumps.

### Shared service, multiple control surfaces

Python imports, CLI commands, and MCP tools must call the same typed control
service. They may render results differently, but must share authorization,
idempotency, dry-run, lifecycle, error, and status semantics.

### Parallelism follows independence

Concurrency is derived from dependencies, file and symbol conflicts, resource
classes, provider capacity, and merge pressure. A configured worker count is a
ceiling. It is not a reason to start duplicate or mutually blocking work.

### Completion remains evidence-sensitive

A passing model response, a generated task, an empty queue, a cache hit, or one
test command cannot complete a goal. Completion requires fresh evidence bound
to the repository tree, accepted plan, policy, and required validation.
Evidence requirements use stable identifiers and source policies: retrieval
may propose matches, but only an exact, typed, fresh receipt from an allowed
producer can discharge a requirement. Objective, plan, task-board, and
generated-discovery prose are never authoritative completion evidence.

## Workstream A: measurement and token efficiency

Add one supervisor efficiency receipt that joins stage timings, input/output
tokens, reused context, cache outcomes, queue delay, retries, validation cost,
changed scope, and final acceptance. Retain digests and artifact references,
not prompts or decoded model bodies.

Compile context in three tiers:

1. **Invariant core:** goal identity, acceptance criteria, policy, allowed
   scope, output contract, and current task.
2. **Selected evidence:** ranked AST symbols, dependency neighbors, prior
   receipts, proof gaps, and validation failures that fit the stage budget.
3. **On-demand expansion:** content-addressed chunks requested only when a
   planner or implementer identifies a missing reference.

Use the provider tokenizer when available. Otherwise use a calibrated estimator
with recorded error bounds. Reserve output and tool-call space before selecting
input evidence. Retries receive a delta capsule containing the prior decision,
new failure evidence, and changed files instead of the original full prompt.

Measure token cost per accepted task and evidence gain per thousand input
tokens. A cheap rejected call is not an optimization if it causes repeated
replanning.

## Workstream B: analysis and `ipfs_datasets_py` offload

Compose the existing analysis contracts, cache, AST index, and multi-signal
retrieval into one pipeline used by objective scanning, planning, task
generation, and implementation context.

Add an optional `ipfs_datasets_py` analysis provider with a strict capability
handshake. Initial operations should be:

- bounded GraphRAG retrieval over content-addressed repository evidence;
- dataset and provenance queries;
- premise and proof-candidate selection;
- legal/logic translation and consistency candidates; and
- batch analysis for related task or goal packets.

Requests carry repository/objective identities, compact queries, allowlisted
operation IDs, bounds, and artifact references. Responses carry normalized
evidence references, provenance, health, truncation, cost, and a non-authority
verdict. Unavailable operations degrade explicitly to local deterministic
analysis.

The boundary must not import the entire sibling project eagerly, expose
arbitrary execution, or copy large GraphRAG/model payloads into supervisor
state. Hammer and kernel proof authority remain governed by their existing
contracts.

## Workstream C: cache architecture

Use distinct cache namespaces for:

- source/AST analysis;
- retrieval and context capsules;
- plan candidates and deterministic evaluations;
- provider drafts;
- proof candidates and authoritative proof receipts;
- validation commands and impact selections; and
- merge/reconciliation classifications.

Every key includes all semantic invalidation dimensions that matter for its
namespace: repository tree or blob identities, objective and acceptance
revision, query, analyzer/compiler/provider versions, configuration, policy,
scope, and relevant environment capabilities.

Add cross-process single-flight leases so concurrent lanes share an expensive
miss. Negative, failed, timed-out, and inconclusive records receive short TTLs
and never count as completion evidence. Store compact receipts and artifact
references. Enforce entry, namespace, and total byte quotas with observable
eviction and corruption recovery.

## Workstream D: planning and responsive goals

Generate a deterministic baseline plan first. Optional LLM, Leanstral, and
GraphRAG branches compete against the same frozen goal. A deterministic
evaluator scores:

- acceptance and evidence coverage;
- unsupported assumptions or semantics;
- dependency validity and critical path;
- changed-scope and merge-conflict risk;
- validation and proof feasibility;
- expected token, runtime, memory, and artifact cost; and
- novelty relative to existing tasks and historical failures.

No weighted quality score may compensate for an authority, scope, safety, or
proof violation.

Goals should carry an explicit outcome, scope, assumptions, non-goals,
acceptance criteria, evidence producers, validation policy, freshness horizon,
resource envelope, and refinement budget. Track goal debt for ambiguity,
unsupported semantics, stale evidence, uncovered acceptance criteria, and
excessive breadth.

Refine or replan immediately when a typed signal changes the problem:
counterexample, stale receipt, repeated validation signature, unavailable
provider, scope conflict, changed interface, or resource infeasibility.
Unchanged failures use backoff and do not repeatedly consume planning tokens.
The root and assumptions remain frozen unless an operator admits a new
objective revision.

## Workstream E: output and implementation validation

Validation should be a fail-fast DAG:

1. strict output schema, canonical IDs, authority claims, and bounds;
2. patch parse, allowed paths, secret/binary/size policy, and non-empty semantic
   change;
3. AST/type/interface and dependency checks;
4. impact-selected unit, integration, and contract tests;
5. legal/logic, prover, protocol, and runtime checks when required;
6. merge-tree preflight and conflict repair;
7. fresh post-merge validation and goal-conformance receipts.

An LLM cannot choose which required gates to omit. It may propose extra tests
or explain a failure. Failed gates produce typed diagnostics suitable for one
bounded repair or a follow-up task. Repeated identical failures should reuse
the diagnostic receipt and escalate rather than replay the full model context.

Adversarial fixtures must cover no-op patches, test deletion, validation
weakening, out-of-scope edits, forged receipts, stale cache entries, symlink and
submodule escapes, prompt injection in repository text, oversized artifacts,
and model claims of proof or completion.

## Workstream F: task generation and bundling

Each task should represent one coherent state transition with:

- a goal and acceptance subset;
- predicted files, symbols, interfaces, and generated artifacts;
- preconditions, effects, dependencies, and conflict policy;
- validation/proof requirements;
- resource and token budget classes; and
- a canonical semantic identity.

Split tasks that exceed path, symbol, acceptance, context, or validation
budgets. Coalesce tiny tasks that share goal, context, outputs, validation, and
merge fate. Do not bundle merely because titles are similar.

Optimize bundles using dependency depth, shared context, conflict coloring,
resource class, provider batchability, validation reuse, and merge locality.
Represent packet aggregates explicitly so completing the aggregate can
propagate only to the exact covered sibling identities.

## Workstream G: adaptive parallelism

Model the following resource pools independently:

- deterministic analysis and indexing;
- LLM/Leanstral inference;
- ATP/SMT/kernel proof;
- validation subprocesses;
- Git/worktree and merge I/O; and
- artifact/database persistence.

The scheduler should adapt concurrency using sampled CPU, RAM, GPU memory,
provider capacity, disk pressure, validation queue depth, and merge queue age.
Use one shared model server or provider batch scheduler where possible instead
of loading model weights per lane. Batch compatible requests while preserving
per-request budgets, cancellation, and receipts.

Prioritize critical-path work without starving small independent tasks.
Backpressure task generation when ready work, merge debt, or artifact pressure
exceeds policy. Demonstrate throughput with independent fixture lanes and
correct serialization with conflicting lanes.

## Workstream H: Python, CLI, and MCP control

Create a typed `SupervisorControlService` and read-only `SupervisorClient`
facade. Core operations should include:

- capabilities, status, health, metrics, goals, tasks, bundles, lanes, events,
  receipts, and cache inspection;
- objective preview/refine/reconcile and backlog refill;
- plan, start, pause, resume, drain, stop, retry, cancel, and quarantine; and
- bounded artifact query and validation replay.

The unified CLI should expose these through `ipfs-accelerate agent ...`.
Machine output uses stable JSON schemas and meaningful exit codes. Destructive
or mutating commands support dry-run and explicit repository/state paths.

The canonical MCP server should add a lazily loaded `agent_supervisor`
category. Read tools may be broadly available under normal read policy.
Mutating tools require authorization, idempotency keys, repository allowlists,
lease/fencing checks, and audit receipts. MCP must call the control service,
not shell out to CLI strings.

## Workstream I: bounded self-refill

When the initial board drains, run a self-improvement epoch:

1. reconcile all goals against fresh receipts;
2. run the paired efficiency, quality, safety, and throughput benchmark;
3. classify regressions, uncovered criteria, stale evidence, persistent
   bottlenecks, and unsupported capabilities;
4. generate a bounded set of candidate successor goals;
5. deduplicate them against active, completed, rejected, and cooldown work;
6. validate goal quality and refinement obligations;
7. materialize only admitted goals transactionally; and
8. record either generated work or a healthy-exhaustion quorum.

An epoch is keyed by repository tree, objective revision, benchmark policy, and
capability snapshot. Replaying the same epoch is idempotent. Healthy exhaustion
does not immediately create another epoch; a cooldown, changed tree, stale
evidence, regression, operator objective revision, or scheduled observation
window is required.

This design lets the supervisor continue improving itself without manufacturing
busywork or treating an empty board as proof of success.

## Metrics and promotion gates

The paired benchmark compares the current baseline and candidate behavior on
the same repositories, goals, provider fixtures, and fault injections.

| Dimension | Required measurement | Initial promotion gate |
| --- | --- | --- |
| Safety | False completion and authority-boundary violations | Exactly zero |
| Token efficiency | Input, reused, and output tokens per accepted task | At least 35% lower median input tokens with no safety regression |
| Context quality | Selected evidence used by accepted plan/change | No lower evidence coverage than baseline |
| Planning | Valid plan rate, evidence coverage, duplicate/conflict rate, replans | At least 10 percentage points more coverage or 20% fewer invalid branches |
| Cache | Warm hit rate, stale-hit rate, bytes, lookup latency | At least 70% reuse on repeated fixtures and zero stale authoritative hits |
| Validation | Time to first useful failure, escaped defects, false rejection | Zero seeded escaped defects; lower median time to first failure |
| Parallelism | Accepted tasks/hour, queue delay, CPU/GPU/RAM/disk | At least 2x throughput on independent lanes without duplicate execution |
| Bundling | Context reuse, bundle completion, merge conflict rate | No merge-conflict regression and fewer model calls per accepted work item |
| Refill | Novel admitted goals, duplicates, churn, exhaustion behavior | No duplicate generation and idempotent healthy exhaustion |
| Control | Python/CLI/MCP schema and behavior parity | Contract tests pass for every shared operation |

These are promotion gates, not hard-coded production defaults. A gate failure
keeps the feature in shadow or assist mode and creates a bounded diagnostic
task. It must never be hidden by a composite score.

## Delivery order

The task board uses five dependency tranches:

1. **Measure and define:** efficiency receipts and shared contracts.
2. **Compile evidence:** integrate analysis, `ipfs_datasets_py`, context, cache,
   planning, and proposal validation.
3. **Execute efficiently:** improve validation, task quality, bundling,
   resource admission, batching, and merge throughput.
4. **Unify control:** add the shared Python service, CLI, MCP, and lifecycle
   controls.
5. **Close the loop:** add bounded self-refill, paired end-to-end gates,
   exports, and operator documentation.

Tasks that edit central registries or package exports occur late and depend on
standalone lane work. This keeps the early implementation width high without
creating avoidable merge conflicts.

## Rollout

All new planning, offload, context, and refill behavior starts in shadow mode.
The rollout sequence is:

```text
off -> shadow -> assist -> policy-approved automatic use
```

Shadow mode may write bounded metrics and candidate receipts, but may not
change canonical goals, tasks, implementation trees, or completion state.
Assist mode may present or queue operator-approved proposals. Automatic use
requires every non-negotiable gate, stable restart recovery, bounded artifacts,
and a paired improvement. Any false completion, authority violation, stale
authoritative cache hit, uncontrolled mutation, or idempotency failure rolls
the affected capability back to shadow.
