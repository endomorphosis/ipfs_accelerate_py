# Agent Supervisor Self-Improvement Plan, Generation 2

## Purpose

The first self-improvement program established typed context, analysis,
planning, validation, scheduling, control, and refill components. Generation 2
turns those components into one causally measured, event-driven control system.
The objective is not to add more autonomous prompting. It is to reduce the
amount of model reasoning required per accepted change while increasing the
quality, throughput, auditability, and responsiveness of the whole supervisor.

The executable projection of this plan is the `ASI-092` through `ASI-123`
tranche in
[agent_supervisor_self_improvement.todo.md](agent_supervisor_self_improvement.todo.md).
The durable intent is the separate `ASI-G200` objective tree in
[agent_supervisor_self_improvement.objectives.md](agent_supervisor_self_improvement.objectives.md).
Generation 2 is deliberately outside the fixed `ASI-G000` child and producer
populations. It must not weaken or silently rewrite the completed generation-1
contracts.

## Why a second generation is needed

The repository now has mature first-generation components, including:

- a token-budgeted context compiler and end-to-end efficiency receipts;
- incremental AST, retrieval, cache, GraphRAG, logic, Leanstral, Hammer, and
  multi-prover adapters;
- deterministic and model-assisted planning, counterexample-driven
  refinement, strict proposal admission, impact-selected validation, and
  proof-aware completion;
- task identity, task quality, conflict graphs, bundle optimization, adaptive
  resources, provider batching, leases, worktrees, and merge trains;
- a typed Python control service with CLI and MCP adapters; and
- bounded benchmark-driven self-refill with replay and healthy-exhaustion
  evidence.

The remaining problem is integration quality rather than missing isolated
features. Current runtime and repository evidence exposes these residual gaps:

1. Token accounting is available, but model work is not always attributed to
   the exact stage, retry cause, cache decision, accepted criterion, and
   terminal outcome. It is therefore hard to distinguish a useful context
   reduction from cost shifted into retries or validation.
2. Context capsules are bounded, but common prefixes and immutable evidence
   are not systematically arranged for provider prompt/KV cache reuse.
   Evidence selection is relevance-oriented rather than explicitly
   value-of-information-oriented.
3. `ipfs_datasets_py` integrations are operation-specific. They need one
   asynchronous, capability-negotiated reasoning transport with uniform
   provenance, disagreement, timeout, cancellation, and fallback semantics.
4. Analysis, context, plan, proof, validation, and artifact caches have strong
   local implementations but do not yet behave as one dependency-aware,
   tiered content-addressed graph. Similar single-flight and invalidation logic
   is repeated across namespaces.
5. The planner evaluates candidates well, but it does not yet perform bounded
   AND/OR search over goal alternatives with hard-constraint pruning and a
   learned history of branch failures.
6. Goal refinement reacts to typed failures, but broad goal quality, residual
   uncertainty, information gain, and resource infeasibility are not one
   event-driven policy.
7. Validation is strict, but hermetic execution, differential and mutation
   checks, flaky classification, post-merge semantic assembly, and
   untrusted-repository defenses need one closed admission path.
8. Task splitting and bundling have formal contracts, but task granularity is
   not continuously calibrated against observed context cost, validation
   reuse, merge fate, and accepted criteria.
9. Resource admission is adaptive, but the runtime still needs cross-stage
   work stealing, shared inference batching, distributed lane fencing, and
   backpressure based on merge and persistence debt.
10. The Python, CLI, and MCP surfaces share contracts, but every operation
    should be generated or checked from one versioned capability catalog and
    exercised by the same conformance fixtures.
11. A drained board can still incur repeated polling, objective scanning, and
    large state/event projections. Completed-board idle operation should
    consume negligible CPU and perform no unchanged writes.
12. Generation-1 refill can prove idempotency, but successor selection still
    needs explicit Pareto evaluation, ablations, novelty distance, epoch
    cooldown, and hard per-epoch work budgets to resist metric gaming and
    evidence-churn loops.

## System boundary

`ipfs_accelerate_py` remains the authority-bearing orchestration layer. It owns
goal and task identity, policy, authorization, scheduling, leases, cache
coordination, validation, merge admission, lifecycle state, and control
surfaces.

`ipfs_datasets_py` is an optional software reasoning fabric. It may provide
bounded read-only operations such as:

- repository and AST retrieval;
- GraphRAG neighborhood and provenance queries;
- premise selection and contradiction search;
- legal/logic translation candidates;
- proof-candidate and counterexample analysis;
- semantic clustering and historical-result retrieval; and
- batch analysis over related goals or tasks.

An `ipfs_datasets_py` response remains proposal-tier evidence until an
`ipfs_accelerate_py` producer validates its schema, provenance, freshness,
scope, and authority. The provider cannot mutate the repository, satisfy a
completion criterion by confidence, choose which validation gates run, or
promote its own proof. The optional boundary must remain lazy and preserve a
deterministic local fallback.

## Target control system

```text
objective revision or typed runtime event
                 |
                 v
event cursor + dependency-aware content-addressed state
                 |
                 v
local deterministic analysis ----> optional ipfs_datasets_py operations
                 |                         |
                 +-----------+-------------+
                             v
value-of-information evidence selector
                             |
                             v
prefix-stable, token-budgeted stage capsule
                             |
                             v
deterministic baseline + bounded AND/OR candidate search
                             |
                             v
hard constraints + quality/cost/Pareto evaluation
                             |
                             v
goal delta or acceptance-bound task graph
                             |
                             v
conflict/resource/context-aware bundles
                             |
                             v
adaptive local/distributed stage scheduler
                             |
                             v
hermetic proposal, test, semantic, proof, and merge validation
                             |
                             v
fresh compact receipts + objective reconciliation
                             |
                 +-----------+-----------+
                 |                       |
              complete             residual event
                                         |
                                         v
                           bounded delta replan/refill
```

Each transition has a canonical input identity, a versioned policy, a bounded
result, and a receipt. Large bodies live in a content-addressed artifact store;
state, events, prompts, and receipts carry bounded references and summaries.

## Computer science foundations

Generation 2 uses the following theories as design constraints rather than
documentation decoration:

- **Feedback control and model-predictive control:** plan over a bounded
  horizon, observe typed outcomes, and replan only the affected suffix while
  preserving the frozen objective and safety invariants.
- **AND/OR search and constraint programming:** represent alternative ways to
  satisfy a criterion as OR nodes and jointly required evidence as AND nodes.
  Reject authority, scope, dependency, resource, and proof violations before
  applying soft scores.
- **Value of information and active learning:** spend tokens or expensive
  analysis only when evidence is expected to change a decision or reduce a
  material uncertainty.
- **Incremental computation and content-addressed memoization:** key derived
  facts by complete semantic dependencies, propagate invalidation through an
  artifact graph, and recompute only affected nodes.
- **Event sourcing and materialized projections:** append compact canonical
  transitions, maintain cursored projections, and checkpoint deltas. Markdown,
  status JSON, indexes, and dashboards are rebuildable views.
- **Conflict graph coloring, critical-path scheduling, and work stealing:**
  extract parallel width from dependency and conflict graphs, then fill idle
  capacity without duplicating or concurrently mutating the same scope.
- **Capability-based security and two-phase mutation:** discovery grants no
  authority. A mutation requires a target-bound permit, dry-run effects,
  idempotency key, lease, fencing epoch, and postcondition receipt.
- **Differential, metamorphic, and mutation testing:** validate behavior
  against independent transformations and seeded defects instead of trusting
  only tests produced by the same implementation path.
- **Multi-objective optimization:** compare candidate behavior as a Pareto
  vector. Safety, authority, freshness, and escaped-defect gates are
  non-compensable; a weighted average cannot hide their failure.

## Workstream 1: causal measurement and token efficiency

Build a generation-2 paired corpus that freezes goal, tree, provider,
capability, fault, and policy identities. Attribute every input, output,
cached, and speculative token to a stage, task, attempt, evidence selection,
provider request, validation result, and terminal accepted criterion.

Use provider-native tokenizers where available and calibrate fallbacks by model
and prompt envelope. Arrange prompts as a stable policy and objective prefix,
stable task core, and volatile evidence delta so providers with prompt or KV
caching can reuse the largest valid prefix. Record actual reuse when the
provider exposes it and a deterministic estimated reuse bound otherwise.

Select optional evidence by expected decision value divided by token, latency,
and invalidation cost. Retrieval relevance is an input, not the final policy.
Required authority and acceptance fields are never optional. Start with a
small capsule, expand a content-addressed reference only when the planner or
implementer identifies a named unresolved question, and charge the expansion
to the resulting decision.

## Workstream 2: software-first analysis and `ipfs_datasets_py`

Define one asynchronous operation transport with capability negotiation,
deadlines, cancellation, batching, progress, result bounds, and deterministic
fallback. The operation registry maps a typed question to a local producer, an
optional datasets producer, cache semantics, provenance requirements, and an
authority class.

The first registry population should cover AST/symbol impact, GraphRAG
retrieval, premise selection, contradiction search, legal/logic candidate
translation, and proof/counterexample candidate analysis. Equivalent local and
remote results normalize into one compact evidence reference shape.

When producers disagree, persist the disagreement and its provenance. Resolve
it with a deterministic policy, an independent validator, or an explicit
uncertainty record. Never silently choose the more confident model result.

## Workstream 3: planning and responsive goals

Add a typed goal grammar with outcome, scope, assumptions, non-goals,
acceptance criteria, evidence producers, validation policy, freshness,
resource envelope, uncertainty, and refinement budget. A goal-quality linter
rejects circular acceptance, unverifiable evidence, hidden authority,
unbounded scope, and dependencies with no satisfiable producer.

Compile the goal into a bounded AND/OR graph. Always include a deterministic
baseline. Optional LLM, Leanstral, and analysis-provider branches propose
alternatives under the same frozen context. Hard constraints prune branches;
soft objectives compare evidence coverage, uncertainty reduction, expected
cost, critical path, conflict risk, and historical failure likelihood.

Store only typed branch features and failure signatures, not model reasoning
transcripts. A fresh counterexample or capability change invalidates the
smallest dependent plan suffix. An unchanged failure reuses its diagnostic and
backs off. Goal refinement runs from meaningful events and information gain,
not every daemon poll.

## Workstream 4: output and semantic validation

Treat repository text and model output as untrusted. Strictly parse the output
envelope, canonical identities, authority claims, patch, normalized paths,
symlink/submodule boundaries, size limits, and expected effects before any
expensive command.

Run selected validation in a hermetic, resource-bounded environment. Combine
dependency impact with contract, differential, metamorphic, and mutation
tests. Record flaky outcomes separately; an intermittent pass cannot erase a
failure. Semantic, protocol, legal/logic, theorem, and proof checks remain
typed DAG nodes selected by declared task obligations.

After merge, rebuild the evidence graph from the actual merged tree and
assemble one authoritative receipt that binds proposal admission, executed
validation, semantic/proof results, merge identity, freshness, and covered
acceptance criteria. Pre-merge candidate evidence cannot substitute for this
receipt.

## Workstream 5: tiered caching and bounded persistence

Unify namespace coordination around a tiered content-addressed artifact graph:

1. process-local immutable object and prefix caches;
2. host-local durable receipts and artifacts;
3. optional shared/P2P immutable artifacts; and
4. authoritative current-tree projections.

Keep namespace schemas and authority distinct. The graph records dependencies,
producer and policy versions, freshness, capabilities, and invalidation edges.
Generalize the existing namespace-specific single-flight implementations into
a lease-aware coordinator that can collapse duplicate work across processes
and, when configured, hosts.

Persist summaries separately from payloads. A receipt is at most 256 KiB, a
routine projection is at most 1 MiB, and decoded model text, source bodies,
proof traces, and nested artifact graphs are referenced rather than embedded.
Use retention classes, incremental compaction, quotas, and observable
eviction. Failed, negative, timed-out, or inconclusive records have bounded
TTLs and never become completion evidence.

## Workstream 6: task generation, bundling, and parallelism

Calibrate task granularity with observed cost. Split work when one task exceeds
acceptance, context, path, symbol, validation, proof, or merge-risk bounds.
Coalesce only tasks with compatible dependencies, context, validation, file
ownership, resource class, and merge fate. Preserve an exact mapping from each
task to its acceptance subset.

Optimize bundles for dependency depth, conflict colors, shared immutable
context, provider batchability, validation reuse, artifact locality, and merge
pressure. Rebundle pending work after a typed change, but never change the
identity or completion scope of active work.

Schedule analysis, inference, proof, validation, merge, and persistence as
separate resource pools. Combine critical-path priority with fair work
stealing. Use one shared model service and batch compatible requests rather
than loading weights per worker. Apply backpressure when ready work, provider
queues, merge debt, memory, GPU memory, disk, or artifact persistence exceed
policy.

Distributed lanes are optional. They require immutable input artifacts,
lease/fencing epochs, capability and environment receipts, duplicate-work
suppression, and merge-train serialization. More configured workers are useful
only when the dependency/conflict graph exposes independent work.

## Workstream 7: unified Python, CLI, and MCP control

Define one versioned operation catalog. Each operation declares:

- request and result schemas;
- read, proposal, or mutation authority;
- target descriptor and allowed roots;
- bounds and pagination/event-cursor behavior;
- dry-run and idempotency support;
- required lease/fencing semantics;
- backend capability and degradation rules; and
- expected audit receipt.

Python, `ipfs-accelerate agent`, and the canonical MCP server must adapt this
catalog to their transport without changing behavior. Generate or
conformance-test all three surfaces from the catalog. Discovery and imports
remain lazy and side-effect-free.

All mutations support dry-run effects. Real mutations require policy
authorization, exact repository and state targets, idempotency, a live lease,
fencing, compare-and-swap state revision, and a durable result. Multi-step
operations expose transaction status and compensating repair, not silent
partial success.

## Workstream 8: event-driven reliability

Replace completed-board polling and full-state rewrites with event-triggered
wakeups plus a low-frequency safety timer. Sources include task-board revision,
objective revision, process exit, lease expiry, validation completion,
provider capacity, repository change, and scheduled observation windows.

Use cursors and delta materializations so an unchanged cycle reads bounded
metadata and performs no state write. Compact logs and checkpoints
incrementally. Fault-inject crashes, partial writes, stale leases, corrupt
caches, provider loss, disk pressure, merge interruption, and duplicate
events. Recovery must be deterministic, bounded, observable, and fail closed.

## Workstream 9: benchmark-gated bounded self-refill

When the initial v2 board drains:

1. reconcile current tasks and goals from authoritative receipts;
2. confirm analyzer health and a complete benchmark population;
3. evaluate the candidate against the frozen generation-2 baseline;
4. classify typed residuals, regressions, stale evidence, bottlenecks, and
   unsupported capabilities;
5. run ablations to identify which component caused each material change;
6. propose successor goals from residuals, never from generic exhortations;
7. reject duplicates and near-duplicates across active, completed, rejected,
   cooldown, and historical work;
8. enforce quality, novelty, depth, breadth, open-work, token, task, and epoch
   limits;
9. preview and transactionally materialize one bounded successor epoch; and
10. otherwise persist healthy exhaustion and wait for a meaningful event.

The same epoch binding is idempotent. It includes repository tree, objective
and board revisions, benchmark policy, capability set, operation catalog,
artifact-store policy, and observation window. A normal epoch may create at
most 8 goals and 24 tasks. It may not run again for 6 hours unless the tree,
objective, capability, policy, stale-evidence set, or measured regression
changes materially. The thresholds are initial conservative maxima and can be
lowered by policy.

Refill proposals remain in shadow until the generation-2 rollout gate passes.
A completed board with healthy exhaustion is a valid stable state, not a
reason to generate motivational busywork.

## Metrics and promotion gates

All gates compare baseline and candidate on the same frozen fixtures. Safety
gates are non-compensable.

| Dimension | Generation-2 promotion gate |
| --- | --- |
| Safety and authority | Zero false completion, unauthorized mutation, authority escalation, escaped seeded defect, path escape, or stale authoritative cache hit |
| Tokens | At least 40% lower median input tokens per accepted criterion and at least 60% fewer retry-input tokens, with unchanged required evidence coverage |
| Context reuse | At least 70% of eligible stable-prefix tokens reused on warm provider fixtures, with exact invalidation after a semantic dependency change |
| Planning | At least 15 percentage points higher first-valid-plan rate or at least 25% fewer invalid branches, with no hard-constraint violation |
| Analysis offload | At least 70% of eligible repeated analysis reused or offloaded; every fallback and disagreement is typed and no provider gains authority |
| Cache | At least 80% warm exact reuse, at least 60% duplicate-miss collapse, zero stale authoritative hits, and all byte quotas respected |
| Validation | Zero escaped seeded defects and at least 30% lower median time to first useful failure; flaky outcomes cannot produce authority |
| Task quality | Exact acceptance coverage with fewer model calls per accepted criterion and no duplicate semantic task identities |
| Parallelism | At least 3x accepted throughput over one lane on independent work, less than 5% duplicate compute, no conflict regression, and stable memory/disk use |
| Persistence | Receipts at or below 256 KiB, routine projections at or below 1 MiB, bounded aggregate growth, and no duplicated embedded payload graph |
| Idle reliability | On a drained board, less than 2% of one CPU core averaged over 10 minutes and zero unchanged state writes |
| Control | Python, CLI, and MCP pass the same operation-catalog fixtures; every mutation is target-bound, idempotent, authorized, leased, fenced, and audited |
| Refill | Exact epoch replay creates zero work; no duplicate successor; no epoch exceeds 8 goals or 24 tasks; healthy exhaustion waits for a meaningful trigger |

A composite score is diagnostic only. Failure of any safety, authority,
freshness, escaped-defect, artifact-bound, idempotency, or population gate
forces the candidate to shadow. Performance regressions cannot be hidden by
improvements in another dimension.

## Delivery graph

The board is intentionally wider after the shared contracts land:

1. `ASI-092` and `ASI-093` freeze the benchmark and shared contracts.
2. Token, analysis, cache, goal, validation, and control foundation lanes then
   run independently.
3. Planning, cache coordination, task quality, persistence, and control parity
   converge on those foundations.
4. Validation assembly, adaptive scheduling, event-driven reliability, and
   distributed execution converge only after their authority and fencing
   dependencies are complete.
5. Self-evaluation, bounded successor generation, refill transactions, paired
   rollout, and public integration close the loop in that order.

Tasks declare predicted files and conflict policy. Dependencies serialize
shared central files; unrelated lanes remain available to parallel workers.
The scheduler, not a fixed worker count, determines useful concurrency.

## Definition of done

Generation 2 is complete only when:

- every `ASI-092` through `ASI-123` task has terminal accepted implementation
  and fresh validation bound to the current repository tree;
- every `ASI-G210` through `ASI-G290` objective has complete criterion
  coverage, healthy analyzer evidence, and independent exhaustive evidence;
- the closed paired benchmark passes every non-compensable and quantitative
  gate;
- Python, CLI, and MCP operate from the same catalog;
- crash and drained-board fixtures recover with bounded artifacts and
  negligible idle work;
- an identical refill epoch is a no-op and a meaningful residual creates only
  the bounded novel work admitted by policy; and
- generation-1 contracts and compatibility fixtures continue to pass.

The first fully passing evaluation may only produce a provisional v2 rollout
decision. A later evaluation on the still-current tree and policy is required
for automatic operation. Any later safety, freshness, control-parity, restart,
or benchmark regression returns affected behavior to shadow.
