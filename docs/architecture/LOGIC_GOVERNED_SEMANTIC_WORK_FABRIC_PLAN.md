# Logic-Governed Semantic Work Fabric implementation and qualification plan

Plan identity: `LGSWF-PLAN-R1`  
Parent goal: `LGSWF-G000`  
Board namespace: `logic-governed-semantic-work-fabric-v1`  
Source baseline: `config/logic_governed_semantic_work_fabric_baseline.json`  
Executable board: `docs/architecture/logic_governed_semantic_work_fabric.todo.md`

## 1. Outcome and execution policy

This program extends the current `ipfs_accelerate_py` agent supervisor into a
`LogicGovernedSemanticWorkFabric`: one evidence-backed control loop that
consumes canonical datasets semantic facts and coordinates goals, immutable
plans, resources, claims, supervisors, daemons, verification, merge, semantic
refresh, refill, and fixed-point completion.

The board is the first deliverable, but not the terminal deliverable. The
configured supervisor must execute it in dependency order until all acceptance
criteria pass, qualification produces a documented no-go, or a typed external
terminal is reached. A worker or model cannot approve its own result.

The implementation rules are:

1. Preserve canonical ownership. Datasets owns semantic meaning; accelerator
   owns operational coordination.
2. Extend landed contracts and runtimes. Do not add another semantic index,
   capsule compiler, proof cache, plan store, objective tracker, daemon
   framework, model router, GUI, or MCP++ profile.
3. Bind every mutation to an isolated worktree, exact base tree, accepted plan,
   semantic root, scopes, resource reservation, lease, fence, validation, and
   compensation procedure.
4. Use deterministic code as final admission authority. LLM output is only a
   proposal or implementation candidate.
5. Preserve all attempts and accepted history. Plan changes are immutable
   revisions and stale workers return typed stale results.
6. Fail closed on unavailable, stale, inconsistent, or quarantined authority.

## 2. Revision reconciliation and implementation authority

The requested revisions were compared with every relevant checked-out head
before this plan was authored. The complete comparison commitments, including
ordered-log and diff hashes, are in the baseline JSON.

| Repository/copy | Observed HEAD | Tree | State | Decision |
|---|---|---|---|---|
| selected accelerator checkout | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` | `17fefd8b21566766ec7058044d128374b12f81cd` | clean, superproject-pinned | implementation authority |
| default accelerator checkout | `ea11293bb996f052d620eae989f5377a956764b1` | `ea6869d70e25c7bc8b80e6458c1a46b8c03f945f` | dirty, 1,245 commits older | legacy evidence only |
| newer LPC accelerator checkout | `c821d0b43877591bbb0fa3f328fbccff187b56e7` | `d0db9783b577526b65946352836300783d462720` | clean, 694 commits newer, submodules absent | unadopted evidence only |
| selected datasets checkout | `ac82107e246b30e35a2bbdcf75e01370d22350c6` | `2b3d892dd1c31fb6b8a3eebdb88616d411c49a47` | clean and exact nested gitlink | semantic authority |
| selected kit checkout | `6196017ca3df016c7159dce43af60f2a0d96a9ae` | `93070c709af29095fdff11f3e2698543449c08ef` | clean and exact nested gitlink | persistence dependency |
| selected MCP++ checkout | `dc3164653a48d059ae9812078359daeafb451c07` | `6560c3d0c926be12df860afb7d7c82043a1769ba` | clean and exact nested gitlink | integration dependency |

The selected source is itself an actual checked-out head, is clean, equals the
superproject gitlink, and contains the landed semantic-state/governor and
configured-board control planes. This makes it the only observed revision that
simultaneously satisfies implementation authority, reproducibility, and
fail-closed launch preflight. Remote-only post-pin functionality is not silently
adopted. In particular, datasets adversarial assurance is unavailable at the
selected head and remains an explicit qualification gap until an admitted
revision decision lands.

The baseline commits to all intervening changes with these exact ranges:

- accelerator legacy to selected: 1,245 commits;
- accelerator selected to newer unadopted: 694 commits;
- datasets embedded legacy to selected: 568 commits;
- datasets selected to observed remote: 96 commits.

`LGSWF-001` materializes the full ordered commit ledgers and verifies their
recorded hashes. A mismatch quarantines the baseline rather than changing the
source decision implicitly.

## 3. Current-state inventory and canonical authority map

### 3.1 Datasets semantic authority

The supervisor must consume these checked-out canonical surfaces:

- `logic/software_contracts/semantic_index/`: repository, AST symbol, symbol
  version, semantic relationship, diff, impact, and invalidation authority;
- `logic/software_contracts/semantic_state/`: semantic-state assembly and
  verification, symbol Merkle data, capsules, freshness, environment bindings,
  raw-source fallback, invalidation extension, and test/proof selection;
- `logic/software_contracts/contracts.py` and `registry.py`: callable contract
  and registry authority;
- `logic/software_verification/`: program contracts, verification conditions,
  proof graphs, proof plans, counterexamples, and proof evidence;
- `logic/software_contracts/semantic_governor/`: coverage, sufficiency,
  omission, expansion, calibration, and rule-proposal findings;
- `logic/verification_api.py`: `LogicVerificationAPI@1`, `VerificationAPI@2`,
  and `GoalTacticianAPI@1` consumer boundary;
- `logic/formalization/`, `logic/backends/`, and `logic/families/`: existing
  formalization and prover capability registries.

CLI/MCP wrappers and `logic/api.py` are projections or compatibility facades.
Lazy datasets imports that depend upward on accelerator adapters are unresolved
DAG inversions, not semantic authority.

Known checked-head gaps are first-class work:

- no persisted current datasets `SemanticStateRoot` was found;
- the root deliberately has no operational fields and has no current
  contract-root or proof-obligation-root reference;
- capsule compilation can preserve contract/proof references, but the Python
  scanner does not populate the corresponding metadata in production;
- `adversarial_assurance/` is absent at the selected datasets head.

Those gaps must be resolved through datasets-owned interfaces or documented as
qualification limitations. The accelerator must never reinterpret them.

### 3.2 Accelerator operational authority

The implementation reuses and improves these landed modules:

- semantic consumption: `agent_supervisor/semantic_state/`, including strict
  datasets adapters, ContextPack records, root manifests, durable CAS, and
  `SemanticWorkRequest`/`SemanticWorkResult`;
- planning and authority: `planning/plan_revision_contracts.py`,
  `task_sources/plan_revision_store.py`, formal planning contracts, objective
  graph, task projections, backlog refinery, and configured-board compiler;
- graphs: `core/conflict_graph.py`, `analysis/program_graph.py`, change
  propagation, and parallel-plan compilation;
- resources: `runtime/resource_scheduler.py`, provider batch/capacity
  schedulers, proof scheduler, formal verification cache, and worktree limits;
- coordination/execution: `runtime/multi_supervisor_runner.py`, daemon
  registry, implementation supervisor/daemon, leases, fencing, merge queue,
  merge train, event log, checkpoints, rescue, and unstall systems;
- closed-loop systems: validation, semantic governor, verification,
  objectives/refill, self-improvement, integrations, and highest-level
  entrypoints.

Compatibility shims remain import-compatible but receive no new authority.
`runtime/multi_supervisor_runner.py` contains duplicate runtime-effective
definitions and the current static package graph contains a 17-package strongly
connected component; A2 must inventory and repair these deliberately.

## 4. Frozen ownership boundary

`ipfs_datasets_py` is authoritative for semantic identity, symbols, dependency
facts, semantic roots, capsules, bindings, source evidence, contracts,
assumptions, proof and invalidation obligations, test/proof selection,
limitations, counterexamples, governor findings, and assurance findings.

`ipfs_accelerate_py` is authoritative for goals, tasks, immutable plan
revisions, accepted-plan pointers, claims, supervisor/daemon lifecycle,
resources, scheduling, worktrees, leases, fences, cancellation, retries,
checkpoints, model routing, execution attempts, validation orchestration, merge,
events, refill, rescue, and human escalation.

The datasets `SemanticStateRoot` remains semantic-only. Operational records
reference semantic CIDs; they never add claims, paths, workers, prompts,
provider payloads, credentials, or mutable state to that root.

## 5. Target architecture

```text
datasets semantic authorities             accelerator operational authorities
index/state/capsules/contracts/proofs      goals/plans/claims/resources/merge
                 \                         /
                  \ exact CID references /
                   SupervisorWorldSnapshot@1
                              |
                   SupervisorWorldView (read only)
                              |
       SemanticWorkBinding@1 + completion contracts
                              |
     SemanticWorkGraph@1 + dedicated ConflictGraph
                              |
          deterministic conflict-free frontier planner
                              |
 resource reservations -> supervisor partitions -> daemon packets
                              |
 provisional scan -> validation/proof -> merge -> canonical refresh
                              |
 completion evaluation -> immutable revision/refill -> fixed point
```

### 5.1 Operational world overlay

`SupervisorWorldSnapshot@1` is accelerator-owned and content addressed. It
references repository/tree identity; datasets repository, semantic, symbol,
capsule, environment, contract, and obligation roots; accepted plan/root and
revision; objectives, goal/subgoal/task populations; claims; resource and
capability snapshots; merge and completion roots; unresolved gaps and policy
roots; event cursor; coordination and fencing epochs.

Each component has one of `current`, `stale`, `unavailable`, `inconsistent`, or
`quarantined`. Construction obtains each dimension from its own verified
authority. Scheduling fails closed unless required authorities agree on the
repository, tree, plan, task population, semantic generation, and policy.
Raw source, prompt bodies, credentials, model responses, mutable paths, and
arbitrary provider payloads are referenced artifacts and never embedded.

`SupervisorWorldView` is a pure read model for goal/subgoal/task state,
bindings, readiness/blocking, conflicts, resources, claims, capsules,
contracts, obligations, completion evidence, and refill eligibility.

### 5.2 Semantic work binding and completion

`SemanticWorkBinding@1` references, without copying capsule contents, the
accepted plan, tree and semantic root; target symbols/artifacts/capsules; raw
source requirements; environment bindings; pre/post/exceptional conditions;
allowed/prohibited effects and scopes; tests/proofs; assumptions, limitations,
counterexamples and invalidation; completion rule, required authority, and
human review.

Goal completion requires current observable state, semantic properties,
accepted children, tests/proofs, resolved counterexamples and critical gaps,
the exact accepted tree/root, and any required human approval. Task lifecycle
separates worker completion, patch validation, proof verification, merge,
semantic refresh, and supervisor acceptance.

Worktree changes produce a provisional semantic root bound to task and attempt.
Only a post-merge rescan may produce the canonical root. Predicted and observed
deltas are compared and drive invalidation.

### 5.3 Composite work and conflict graphs

`SemanticWorkGraph@1` retains distinct edge classes: goal parent/dependency,
task, code, data, interface, schema, contract, proof, validation, policy, merge,
lifecycle, scope reads/writes/effects, invalidation, conflict, supersession,
generation, blocking, and unlocking. Every edge binds source/target, kind,
authority, evidence, certainty, source semantic root/plan, and invalidation
conditions.

Dependency and conflict remain separate. Shared reads do not serialize work.
Exact symbol writes are preferred; opaque analysis falls back to file or
repository serialization. The dedicated conflict graph includes predicted
paths, AST/interface/schema writes, effects, generated relationships, fixtures,
taskboard/database shards, merge order, external effects, and exclusive
resources. Unknown conflict information is conservative.

Durable priority inputs use integer or fixed-point representations: depth,
critical path, downstream unlocks, blocked goals, cost, uncertainty, merge
risk, resource bottleneck, and cache locality.

### 5.4 Deterministic parallel frontier

Readiness is the conjunction of active-plan membership, legal lifecycle,
satisfied mandatory predecessors, current binding, fresh capsules or admitted
raw source, resolvable contracts/obligations, admitted scope, no active
conflicting writer, reservable resources/provider capacity, known completion
policy, and absence of block/supersession/quarantine/human hold.

The planner computes the ready set and selects a deterministic conflict-free
antichain under hard resource constraints. Candidate ordering is stable by
fixed-point score and task ID. The score combines completion value, critical
path reduction, downstream unlock, priority, age/fairness and locality, then
subtracts resource/provider/proof cost, conflict/semantic uncertainty, retry
risk, and merge congestion. A bounded exact search is used below a configured
frontier size; a deterministic greedy plus bounded local-improvement algorithm
is used above it. The receipt records every component and rejection reason.

Existing `PlanDelta` operations split, coalesce, or rewire only future work.
Claimed through accepted specifications remain immutable and gain successors or
supersession edges. Speculation is read-only or isolated, bounded,
non-authoritative, and safely cancellable.

### 5.5 Resource scheduling and coordination

The existing resource scheduler is extended with integer vectors for CPU,
concurrency, RAM, GPU memory/class, disk capacity/bandwidth, network,
subprocesses, worktrees, model input/output tokens, provider quota/concurrency,
prover class/concurrency, license/key exclusivity, merge slots, and persistence.
Reservations precede dispatch and bind task, attempt, supervisor, daemon, lease,
and fence. Hard resources are never overcommitted.

Observed receipts remain distinct from estimates. Historical estimation,
cache affinity and single-flight reuse cover scans, semantic blocks, capsules,
contexts, provider sessions/prefixes, environments, tests, proofs,
dependencies, and worktrees. Independent backpressure prevents saturation in
one stage from stopping compatible work in another. Preemption targets stale or
low-priority speculative work and never abandons uncompensated effects.

Supervisors advertise observations, not authority. One fenced writer owns each
mutable coordination shard; peers read snapshots, submit proposals, claim
admitted work, and publish immutable evidence. Failover advances the epoch and
prevents an old coordinator from committing. Partitioning uses semantic/goal/
repository/resource/provider/worktree/merge locality while preserving explicit
cross-partition edges. Work stealing requires eligibility, a current checkpoint
where applicable, and a later fence. Execution may be at least once; logical
acceptance is exactly once for task, revision, base tree/root, and idempotency
key.

### 5.6 Daemon packets, refresh, refill, and convergence

The existing canonical packet is extended to bind the complete task, goal,
plan, source, semantic, ContextPack, scope/effect, resource/provider/model,
validation/proof/completion, lease/fence/attempt/idempotency, checkpoint,
cancellation, timeout, and output contract. The lifecycle is:

```text
offered -> admitted -> claimed -> running -> checkpointed -> settling
        -> worker_completed -> supervisor_verified -> accepted
```

Typed side states are rejected, blocked, cancelled, timed out, failed, partial
effect, compensation required, superseded, and quarantined. A checkpoint binds
attempt inputs, worktree tree, file/symbol delta, completed stage, consumed
resources/model calls/tests/proofs, outstanding obligations/effects, and resume
requirements. It is never completion.

The existing `PlanRevisionStore` and backlog refinery create evidence-backed,
deduplicated, bounded refill proposals. Bounds cover successor and revision
depth, tasks/subgoals, repeated semantic keys, retries/providers/tokens,
frequency, and no-progress epochs. Plan Doctor diagnoses cycles, unreachable or
orphaned work, missing bindings/completion/verification, unsafe or hidden
conflicts, over-serialization, resource infeasibility, bottlenecks, starvation,
retries, stale evidence, root mismatch, and inadequate parent coverage. It
proposes but never directly mutates.

Before execution the loop resolves a verified datasets view, freshness,
test/proof selection, and ContextPack. During execution it scans provisional
changes, computes delta/invalidation, rejects scope escapes, and replans
verification. Before merge it verifies effects/contracts/tests/proofs,
governor findings, assurance sampling, and an incremental seal. After merge it
rescans the accepted tree, builds the canonical datasets root, compares deltas,
updates the world snapshot, invalidates dependents, reevaluates completion, and
revises/refills until convergence.

Success requires accepted parent and required child goals; no mandatory ready,
blocked, or unresolved work; current completion evidence; no blocking
invalidation/proof/assurance gap; matching accepted tree/semantic root/current
plan; no active mutating claims; settled merge; and verified receipts/seals.
All other terminals are explicit non-success states.

## 6. Dependency-ordered implementation board

```text
A inventory/freeze
  -> B world overlay
  -> C bindings/completion
  -> D work/conflict graphs
  -> E frontier planner
  -> F resource scheduler
  -> G multi-supervisor coordination
  -> H daemon packet/checkpoint
  -> I plan revision/refill
  -> J semantic refresh
  -> K convergence
  -> L observability
  -> M fault qualification
  -> N benchmark
  -> O release qualification
```

The detailed task board declares exact ownership, scopes, roots, resource
vectors, contracts, leases, compensation, evidence, and validations for every
task. Within an epic, only file-disjoint tasks execute concurrently; each epic
has an integration join before the next begins.

| Wave | Ready set after predecessor | Purpose |
|---|---|---|
| W0 | `LGSWF-000` complete | sealed control plane |
| W1 | `001`, `002`, `003` | revision/inventory, DAG/interfaces, datasets producer readiness |
| W2 | `004`, then `005` | current semantic root and accepted R2 binding |
| W3 | `010`-`013` | operational world overlay |
| W4 | `020`-`023` | semantic bindings and completion gates |
| W5 | `030`-`033` | composite and conflict graph |
| W6 | `040`-`043` | deterministic frontier planner |
| W7 | `050`-`053` | resource scheduling, reuse, backpressure |
| W8 | `060`-`062` | supervisor fabric and failover |
| W9 | `070`-`072` | packets, checkpoints, stale behavior |
| W10 | `080`-`082` | bounded revision, refill, plan doctor |
| W11 | `090`-`092` | closed-loop semantic refresh |
| W12 | `100` | fixed-point convergence |
| W13 | `110` | receipts, metrics, entrypoint |
| W14 | `120`-`122` | deterministic and adversarial fault qualification |
| W15 | `130`-`131` | A/B/C/D benchmark and report |
| W16 | `140`-`141` | content-addressed release and go/no-go |

## 7. Qualification design

The deterministic fixture uses three supervisors and ten daemons, multiple
resource classes, independent and conflicting branches, a multi-level goal
graph, proof/validation tasks, merge pressure, and refill triggers. It exercises
all 26 required coordination/fault behaviors and all critical adversarial
inputs. Critical cases must fail closed.

The benchmark compares:

- A: one supervisor/one daemon/serial;
- B: one supervisor/multiple daemons/dependency-only;
- C: multiple supervisors/daemons/conflict-aware;
- D: the complete semantic fabric with reuse, resource awareness, adaptive
  plan operations, work stealing, incremental verification, and refill.

Workloads cover independent changes, fan-out/fan-in, a long critical path,
shared schemas, proof/model/merge pressure, bottlenecks, failures, semantic
invalidation, and refill. Every target is a hypothesis. Results report raw
wall time, throughput, efficiency, waits/utilization/overhead, duplicate work,
reuse, tokens, provider throttling, conflicts/failures/recovery, refill yield,
revision cost, and compute per accepted task. No unmet target is relabeled as a
success.

## 8. Safety, migration, and rollback

All mutation occurs in task worktrees. The configured branch is the only merge
target; no protected-branch merge is authorized. Every task provides a rollback
or compensation record. Stale claims are fenced, not hurried to completion.
Ambiguous or non-idempotent external effects require explicit compensation or
human review.

Migration is additive and versioned: introduce contracts, dual-read old
operational records where safe, write only the new canonical version, migrate
durable state with receipts, and retain compatibility projections. Rollback
stops dispatch, fences outstanding mutations, settles/repairs partial effects,
restores the prior accepted plan pointer with its existing CAS operation, and
continues read-only evidence preservation. Datasets semantic artifacts are
never rewritten by rollback.

## 9. Stop/go policy

The supervisor continues while safe work exists. It does not report success for
`blocked_external_dependency`, `resource_unavailable`, `provider_unavailable`,
`semantic_analysis_inconclusive`, `verification_inconclusive`,
`human_review_required`, `bounded_exhaustion`, `no_progress`, `policy_denied`,
`quarantined`, or `cancelled`.

The current release starts with two explicit limitations: no current semantic
root and no checked-head adversarial-assurance package. A tasks may use bounded
raw-source fallback to build/fix the producer and establish the root. Every
later task is marked `REBIND_REQUIRED_BY_LGSWF-005` and may dispatch only after
Plan Revision R2 binds exact semantic/capsule/contract/obligation identities.
If A cannot establish those identities, the qualification decision is no-go;
completion criteria may not be weakened.

## 10. Required final supervisor report

The release report must contain, in order: exact revisions; current-state
inventory; authority map; board/dependency graph; reused and changed modules;
world overlay; bindings; composite/conflict analysis; frontier algorithm;
resource policy; supervisor and daemon protocols; revision/refill; refresh and
fixed point; fault results; parallel/resource benchmarks; model/proof reuse;
scheduling overhead; security; limitations; qualification level; and an
explicit continuous-operation go/no-go.

Any positive conclusion is limited to this exact release, workload, policy,
provider, and environment. The permitted summary claim is:

> The agent supervisor composed canonical datasets semantic state, capsules,
> contracts, invalidations and proof obligations with accelerator-owned goals,
> plan revisions, resource state, claims, supervisors, daemons and merge
> authority. It selected conflict-free work frontiers, coordinated bounded
> parallel execution with leases and fencing, refreshed semantic state after
> accepted changes, and revised or refilled the active plan through immutable
> evidence-backed deltas. The reported parallelism and resource-efficiency
> results apply only to the exact release, workload, policies, providers and
> qualification environment identified in the release manifest.
