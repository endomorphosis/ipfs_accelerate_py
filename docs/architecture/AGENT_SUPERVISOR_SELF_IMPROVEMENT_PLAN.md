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

### Root completion is hierarchical and two-phase

ASI-G000 uses a closed completion adapter rather than a caller-assembled
generic summary. The adapter fixes the complete original producing-task
population (ASI-001 through ASI-024), the exact nine direct workstream goals
(ASI-G010 through ASI-G090), the four literal root acceptance clauses, and a
two-member exhaustive-receipt policy. A caller cannot lower or narrow those
sets. Every producer must be terminal-successful before root completion can be
requested.

Every submitted criterion receipt participates in the decision: a passing
receipt cannot mask a failed, stale, contradictory, malformed, or foreign-tree
sibling. Each literal criterion needs its own fresh passing receipt and an
exact current-tree coverage row naming both a concrete implementation and that
receipt identity. Analyzer health remains a separate authority input and must
explicitly be healthy and safe for completion reasoning with the repository,
tree, objective/revision, analyzer version, and configuration revision.
Operational pipeline output, provider health, or a discovery report cannot
stand in for analyzer health.

The exhaustion quorum must retain the configured count. Every counted member
is fresh, healthy, completion-safe, exhaustive, identically bound, and
independent by member ID, evidence channel, and receipt identity. Every direct
child must have a fresh passed current-tree completion gate, and recursive
descendant proof requirements remain proved, current, conclusive,
uncontradicted, and sufficient for their required assurance. An empty child
list, a state-only child summary, or a drained todo board is not proof.

Even a fully passing first evaluation can move an active root only to
`provisionally_complete`. A later, separate evaluation may verify it while all
bindings and proof remain fresh. Any later task regression, child reopening,
tree change, failed validation, analyzer degradation, or quorum loss reopens a
verified root. The ASI-082 discovery file is an audit index for this policy;
it never performs either lifecycle transition.

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

### Paired end-to-end rollout gate

ASI-023 closes the integration gate in
`agent_supervisor/self_improvement_rollout.py`. The gate consumes bounded
measurements from the existing analysis, cache, context, planning, validation,
resource, merge, control, and refill lanes; it does not rerun or replace those
lanes. Baseline and candidate measurements are paired by a frozen fixture ID,
fixture revision, input digest, and seeded-defect count. Reports contain
counts, scores, terminal classifications, and content identities only. Raw
prompts, model output, patches, proofs, cache values, and artifact bodies do
not cross this boundary.

The fixture population is closed and non-narrowable:

1. cold and warm execution;
2. a broad goal;
3. contradictory input;
4. malformed provider output;
5. a stale cache record;
6. an unavailable optional provider;
7. independent parallel lanes;
8. conflicting parallel lanes;
9. failed validation;
10. process restart; and
11. a drained board followed by refill/exhaustion reconciliation.

Each kind occurs exactly once in a report. A missing kind is a failed gate,
not a smaller benchmark, and duplicate IDs or kinds are malformed input.
Warm and restart fixtures form the repeated-fixture cache cohort. The
independent-parallel fixture supplies the paired throughput measurement. The
conflicting-parallel fixture supplies the merge-conflict comparison. The
restart fixture binds pre- and post-restart state digests, while the
drained-refill fixture records duplicate executions and its terminal
classification.

Promotion requires both gates below:

- **Non-negotiable gate:** the candidate has exactly zero false completions,
  authority violations, stale authoritative hits, escaped seeded defects,
  duplicate executions, and unauthorized mutations. Malformed and
  contradictory fixtures reject, provider unavailability remains degraded,
  fallback, or rejected, failed validation detects every seeded defect, the
  restart state digest is stable, and aggregate candidate artifacts remain at
  or below 256 records and 4 MiB.
- **Paired gate:** candidate terminal outcomes and accepted work do not regress;
  evidence coverage, quality, and defect detection do not decrease; false
  rejection and merge-conflict counts do not increase; median candidate input
  tokens are at least 35 percent below the paired baseline median; candidate
  cache reuse across repeated fixtures is at least 70 percent; and accepted
  work throughput on the independent fixture is at least twice baseline.
  Planning must also improve by either at least 1,000 basis points in median
  evidence coverage or at least 2,000 basis points in aggregate
  invalid-plan-branch reduction.

Threshold configuration may make these requirements stricter but cannot lower
the token, cache, or throughput minimums, raise artifact bounds, or narrow the
fixture population. There are no waivers for the non-negotiable gate.
Performance improvements cannot compensate for a safety, authority, quality,
validation, restart, merge, or population failure.

The report carries two stable runtime evidence terms. Term
`109590900757783560279417463762322084165` is the safety/shadow proof: the
complete seeded population has zero candidate false completions, while any
seeded false completion fails the non-negotiable gate and forces the effective
mode to `shadow`. Term
`146189916032404266364029134505159070240` is affirmative only when the token,
repeated-cache, planning, and independent-throughput gates all pass.
`report.evidence_for(requirement_id, repository_id=...,
repository_tree=...)` returns the bounded typed criterion projection for these
bindings and derives canonical ASI-G112/ASI-G113 internally. A failed
supported term returns a negative diagnostic witness; an unsupported term is
rejected. Serialized evidence is accepted only through
`PairedRolloutRequirementEvidence.from_dict(payload, report=report)`, which
re-derives the complete claim and rejects changed or detached content. The
projection is diagnostic rollout evidence, not completion, proof, merge, or
mutation authority.

Report version 2 carries the explicit invalid-plan-branch counter and component
gate projections. Version-1 reports remain readable and are re-derived for
audit, but cannot affirm the efficiency requirement because they lack the
planning measurement; operators must rerun the current shadow population.

The resulting report is canonical JSON with a stable SHA-256 identity that
excludes only its observation timestamp. Deserialization recomputes the full
decision from the embedded typed fixture evidence instead of trusting stored
summary fields. The append-only report store uses exclusive creation, file and
directory synchronization, symlink rejection, a 2 MiB hard report bound, and
idempotent replay. This makes a recovered decision stable across a supervisor
restart without turning the report into completion evidence.

The paired contracts, evidence identifiers, evaluator, and report store are
stable lazy exports from `ipfs_accelerate_py.agent_supervisor`. Import and
capability discovery remain provider-free and process-free; accessing these
exports loads only the rollout contract module. The deterministic smoke recipe
uses fixed bindings to seed every forced-shadow path and retain bounded reason
codes. The production recipe persists both criterion projections with the
report, profile, capability, repository tree, objective, and policy identities;
a stale or missing binding returns operation to shadow and requires a fresh
paired evaluation.

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
the affected capability back to shadow. The paired end-to-end gate also forces
the effective mode to `shadow` whenever its fixture population is incomplete
or either gate fails; a requested `assist` or `automatic` mode never survives
that decision.

## Operator adoption and operating profiles

ASI-024 publishes the reviewed control and rollout boundary for operators
without turning optional analysis, model, or proof providers into import-time
dependencies. Stable package exports cover the control service and client,
canonical request/result and authorization contracts, side-effect-free
capability and discovery checks, and the paired-rollout contracts. Importing
the package or inspecting capabilities must neither load an optional provider
nor start a process. Provider-specific implementations remain behind their
explicit capability handshakes and are loaded only after policy and resource
admission select them.

The Python, unified CLI, and MCP surfaces are adapters over the same
`SupervisorControlService` operation vocabulary and canonical contracts.
`ipfs-accelerate agent capabilities` is the first operator check; equivalent
Python and MCP discovery must report the same supported operation set, bounds,
authority class, dry-run support, and contract versions. The discovery report
also records whether optional providers were loaded or processes were started,
so discovery-side effects are observable contract failures rather than hidden
behavior. Operators must check support instead of inferring it from an import,
an installed extra, a tool name, or a provider configuration file.

The operating profiles below are normative configuration recipes, not new
global constructors or implicit environment selection. They use existing
`context_contracts.ContextBudget`, `analysis_cache.AnalysisCache`,
`ResourcePolicy`, control bounds, and rollout contracts. Library defaults
remain conservative: a one-lane, non-adaptive scheduler; explicit repository
and state allowlists; required provider telemetry; bounded control results; and
no automatic mutation merely because a provider is available.

| Setting | Production recipe | Deterministic smoke recipe |
| --- | --- | --- |
| Purpose | Sustained operation on a reviewed host after paired-gate promotion | Fast, repeatable contract, migration, and recovery checks |
| Rollout | Begin in `shadow`; request `assist` or `automatic` only from a passing, current paired report | `shadow` only |
| Context | `context_contracts.ContextBudget` defaults: 8,192 input tokens, 2,048 output reserve, 512 tool reserve, 128 items, and 256 KiB serialized | 2,048 input tokens, 512 output reserve, 128 tool reserve, 32 items, and 64 KiB serialized |
| Cache | 512 entries, 32 MiB total, 128 KiB per entry, 96 KiB per receipt, 5-minute negative TTL | 64 entries, 4 MiB total, 64 KiB per entry, 48 KiB per receipt, 60-second negative TTL |
| Resources | Four-lane ceiling, adaptive admission disabled initially, provider telemetry required; stage ceilings of four analysis, one inference, two proof, two validation, one merge, and one persistence lane. Enable adaptive admission only after its paired parallel gate passes | One lane, adaptive admission disabled, one process and one lane for every stage |
| Providers | Enable only individually discovered, policy-allowed providers with recorded quota, latency, token, memory, and GPU bounds; preserve deterministic local fallback | Do not load optional providers; capability discovery must still run and explicitly report them unavailable |
| State | Durable, access-controlled state/cache roots with independent artifact quotas and restart checks | Fresh temporary state/cache roots, frozen fixture IDs and inputs, fixed observation time, and no network/provider dependence |
| Refill | Enabled only after authorization and a current benchmark population; at most the policy-bounded admitted successor set | Evaluate replay and healthy exhaustion, but do not materialize successor work |

The production lane count is an upper bound, not a target. Admission still
reduces it for CPU, RAM, GPU memory, disk pressure, provider capacity, queue
shape, or merge pressure. Deployments with smaller measured capacity lower the
ceiling. The smoke profile deliberately fixes concurrency at one and disables
adaptive/provider variability; it does not weaken schemas, authorization,
evidence freshness, the closed paired fixture population, or any rollout
threshold.

### Promotion, authorization, and control operation

The rollout report never promotes itself. An operator or deployment policy
chooses a desired mode and binds it to a current report:

1. `shadow` evaluates the complete paired population and may persist bounded
   reports, metrics, diagnostics, and candidate receipts, but grants no
   canonical mutation authority.
2. `assist` exposes proposals for explicit approval. It is effective only when
   the same non-negotiable and paired gates required for automatic operation
   pass.
3. `automatic` permits only the operations already allowed by control policy
   and only while the complete gate passes. It does not bypass per-request
   authorization, leases, validation, or completion evidence.

Any missing fixture or failed gate makes `shadow` the effective mode regardless
of the desired mode. A passing report is restart-safe rollout evidence, not
goal, proof, merge, or completion evidence. Promotion also requires an
operator-reviewed profile revision and capability snapshot; a changed tree,
policy, provider capability, or profile makes the prior operational decision
stale and triggers reevaluation.

Read and proposal operations use their declared read/proposal authority. Every
real mutation, including objective refinement or reconciliation, backlog
refill, lifecycle changes, retry, cancel, quarantine, and validation replay,
requires all of the following:

- an exact permit decision bound to the operation, repository and state roots,
  repository/tree and objective revisions, policy revision, and caller;
- declared expected effects within the authorization and root allowlists;
- a caller-scoped idempotency key; and
- a live lease identity and fencing epoch checked before dispatch.

Dry-run mutation requests remain proposal-authority operations and never call a
mutating backend. MCP uses server-configured allowlists and the same service
boundary; it must not derive authority from tool arguments. CLI and Python
callers likewise cannot treat possession of a path, package import, rollout
mode, or capability report as authorization.

For adoption, operators first run capabilities, health, status, and metrics;
then use objective preview and plan; then exercise refine, reconcile, and
refill as dry runs. Only after reviewing expected effects should they submit an
authorized real mutation. The corresponding task-board views and lifecycle
commands use the same target binding and audit receipts. This replaces direct
composition of standalone objective, refinery, implementation-daemon, and
artifact-query scripts while leaving those scripts available as migration
references until the shared surface has equivalent capability.

### Metrics, recovery, and epoch operation

Production monitoring combines control-plane status, health, metrics, events,
and audit receipts with the paired rollout report. Alert and roll back on any
non-zero false completion, authority violation, stale authoritative cache hit,
escaped defect, duplicate execution, or unauthorized mutation; an unstable
restart; incomplete fixture coverage; an artifact bound violation; or a
regression in terminal outcome, accepted work, evidence coverage, quality,
defect detection, false rejection, or merge conflicts. Track the explicit
token-reduction, repeated-cache-reuse, and independent-lane-throughput basis
points rather than a composite score, and retain the report reason codes for
diagnosis.

Recovery is fail-closed and typed:

- an unavailable or unhealthy optional provider selects deterministic local
  fallback, degradation, or rejection and never claims provider authority;
- a stale tree, authorization, lease, or fencing epoch rejects before mutation
  and requires a fresh target binding rather than an in-place retry;
- an idempotent replay returns its durable result, while a key reused for
  different effects is a conflict;
- a stale or corrupt cache entry is invalidated and recomputed and cannot count
  as authoritative evidence;
- a validation, authority, population, or paired regression returns the
  affected behavior to shadow and produces a bounded diagnostic;
- restart recovery reloads canonical state and append-only rollout reports,
  recomputes report decisions from fixture evidence, and verifies stable state
  digests before resuming; and
- ambiguous or unrepaired work is paused, drained, or quarantined with durable
  events and receipts rather than silently retried.

A self-refill epoch begins only when the effective task board is drained. Its
content identity binds the repository and tree, objective and task-board
revisions, self-improvement policy, capability snapshot, observation window,
and operator revision. The ledger replay check happens before benchmark or
proposal callbacks, so the same binding is idempotent across retries and
restart. The benchmark population covers cache, control, efficiency, planning,
safety, throughput, and validation through at least two independent evidence
channels. Failed or partial analysis is an analyzer-health failure and cannot
authorize successor work.

Only fresh, complete, actionable observations can nominate proposals.
Proposals must pass the policy's confidence, novelty, quality, refinement,
depth, breadth, open-goal, and successor-count bounds before one transactional
objective/task-board materialization. If no proposal survives, the epoch
persists healthy-exhaustion evidence and waits. Another epoch requires a
meaningful trigger: a changed repository tree, capability snapshot or policy;
an operator objective revision; a regression or stale evidence; or a scheduled
observation window. Queue exhaustion alone never loops, manufactures work, or
proves the parent goal complete.
