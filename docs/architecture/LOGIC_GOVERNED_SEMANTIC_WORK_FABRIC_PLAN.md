# Logic-Governed Semantic Work Fabric implementation and qualification plan

Plan identity: `LGSWF-PLAN-ACTUAL-R1-S1`

Predecessor evidence: run-actual-v2 plan root
`sha256:7b1b68177b09d71905c2e88790a68544b2c82ac64618d2c2689f3937359291ba`
is quarantined and immutable. Its unsupported completed `LGSWF-000` record is
preserved there; corrected successor `LGSWF-006` must be manually sealed before
the implementation frontier is launchable.
Parent goal: `LGSWF-G000`
Board namespace: `logic-governed-semantic-work-fabric-actual-v1`
Source baseline: `config/logic_governed_semantic_work_fabric_baseline.json`
Executable board: `docs/architecture/logic_governed_semantic_work_fabric.todo.md`

## 1. Outcome and execution policy

This program extends the current `ipfs_accelerate_py` agent supervisor into a
`LogicGovernedSemanticWorkFabric`: one evidence-backed control loop that
consumes canonical datasets semantic facts and coordinates goals, immutable
plans, resources, claims, supervisors, daemons, verification, merge, semantic
refresh, refill, and fixed-point completion. Accelerator-owned DuckDB and Quack
jointly form the operational orchestrator/control plane: DuckDB owns the
authoritative transactional records, schema, CAS, fencing, and outbox cursor;
Quack is the mandatory multi-reader/multi-writer transport and exclusive
state-owner service boundary. The existing nested datasets DuckLake package is
an optional, non-authoritative immutable analytics/history substrate downstream
of that control plane; it is never a scheduling prerequisite.

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

| Repository/copy | Observed HEAD/tree | State | Decision |
|---|---|---|---|
| actual accelerator checkout | `ea11293bb996f052d620eae989f5377a956764b1` / `ea6869d70e25c7bc8b80e6458c1a46b8c03f945f` | dirty: 38 tracked and 134 untracked entries | checked-out implementation authority; preserved untouched |
| pinned accelerator comparison | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` / `17fefd8b21566766ec7058044d128374b12f81cd` | 1,245 commits after actual HEAD | comparison evidence only; not silently substituted |
| isolated accelerator snapshot | `38cd50092d300b61327a9225e7f10cfe8acefb4f` / `005f885e270bbc7573710686a78fc2f740ee5b9f` | exact tracked patch plus 132 non-runtime untracked source files | reproducible source snapshot |
| integrated accelerator base | `b6dc155c3d779a4166a8ee92c0e0214e0157e2e2` / `1313cf18fecd969f654f0233f6678c2d851116e8` | snapshot plus existing DuckDB/Quack history and seven resolved conflicts | retained launch ancestor and lineage evidence |
| historical bootstrap contract | `3a07f2b9273161ce805feff98414ef3c66eae7cc` / `49df4554135a9fcd088f7b4d9c36a86ae4c508c9` | original Portal bridge/board contract before the quarantined unsupported completion | required ancestor only; never an execution base |
| C1 operational release baseline | `301d7318868bbc238abb6862d38c77253d5653b0` / `b08b08e17c4a10c8f0627daf4e1eb4348ef8ad3f` | frozen code/test checkpoint after launch-safety, seal, coordination, intent and expiry hardening | named C1 evidence base; C2 and any later integrated checkpoint remain distinct identities |
| corrected successor execution base | `ACCEPTED_LGSWF-006_SOURCE_HEAD` / materialized `repository_tree_id` | resolved only from the clean accepted R1-S1 source in DuckDB | exact accelerator task base; unresolved selector is nondispatchable |
| actual datasets checkout | `ac82107e246b30e35a2bbdcf75e01370d22350c6` / `2b3d892dd1c31fb6b8a3eebdb88616d411c49a47` | dirty | checked-out semantic authority; preserved untouched |
| isolated datasets snapshot | `0691203550c0f316852c74d293d8fc3c4ce130a6` / `35252228f51c1247a8f99939b36e74f0af36e411` | exact tracked patch plus 29 untracked source files | semantic task base and exact parent gitlink |
| DuckDB/Quack reuse source | `9e39c6c9edb0b756f99f9857a89e70642ef1321c` / `ea321ea749103ece6a175c4e984372e42ac204bd` | historical run stopped; no live server or store | implementation history only; freshly qualify |
| nested datasets DuckLake package | contained in the isolated datasets snapshot above | checked-in capability/catalog-owner/registry/ingest/snapshot/execution/security/recovery implementation; exact DuckDB 1.5.5 locks | optional typed dependency only; live projection disabled pending accepted capability/profile/release receipts |

The dirty checkouts are the implementation authority. They were not cleaned,
reset, or overwritten. Their tracked patches and bounded non-runtime untracked
files were copied into isolated commits; two accelerator lock/audit files and
the datasets nested `.tools/ipfs_kit_py` dirty work were explicitly excluded.
The exact patch and content-manifest roots are sealed in the baseline. The
accelerator tracked patch is
`e5ef6b048a0c14234229b6d4241ac4e5d60789ef`; its 132 included untracked
source paths have manifest root
`cf811d654180947e968a2076bdcceb22ab14fbabc8fc9a5acb60512f41bd5d9a`.
The datasets tracked patch excluding nested `.tools/ipfs_kit_py` is
`e5c96f2332d22ff35ba118ba75a416ba55a01995`; its 29 included untracked
paths have manifest root
`b3b6a01967e8d722714022048f873aff8af2204c067ac7b4d4ba8fa6e0a73281`.
The `2026-08-16T06:47:29Z` recheck reproduced both external HEAD/tree pairs,
patches, counts and roots without adopting a newer external revision.

The exact lineage from launch-safety base
`29136a4e1e6a846f0019a8601513ae9ccf7222d0` to C1, oldest first, is:
`2f38f681b79155bf9bb8cdc86575d6145c81a666` /
`98ded23b267e9a39e5f2b6c9da540a269c4f83bd`;
`07f0458da41b4c3667b2c2676b37dfb8988e7666` /
`15aee88357170da885515c80dcc29ebc9ed189d1`;
`b5bd8f5901e8162b9908613608a8ba92ec0ab73e` /
`8c6fd3bab1872c4c5c84bfca21469c408c5e95a4`;
`f2648dec2f6984e34382569ff5b60376892f2841` /
`2fd1a1c41326e6b864a09060ec7fbee130645bb6`;
`314da0c8f4940394b48000003adf9bb8110f1ef5` /
`35e25d9664b371559607c7fd9959da9d56cab5d2`;
`b9a4b0f5184f48a4bf23209c9424730297be80ce` /
`9e10c03b0ae27935e7fcd75df3dacdf1917f687b`; and C1
`301d7318868bbc238abb6862d38c77253d5653b0` /
`b08b08e17c4a10c8f0627daf4e1eb4348ef8ad3f`.

The prior LGSWF run at `f7c7d19a78c956e58d1f431b49e3b6ea0ef687c7`
completed tasks 000--003, then started LGSWF-004 attempt 1 at
`2026-08-16T02:00:31.023330+00:00`. Its configured-board wrapper received
signal 15 at `2026-08-16T02:01:12+00:00` during that attempt. Although lane 1
recorded `supervisor_shutdown_quiesced_attempt` at
`2026-08-16T02:01:13.333432+00:00`, lane 0 and lane 2 logs repeatedly recorded
`Supervisor maintenance hook failed; leaving child alive`; the later orphan
worker was terminated and quarantined. It used the pinned comparison revision
and legacy Markdown authority, so no descendant output, worktree, result or
completion from that run is resumed or accepted by this plan. The historical
receipt does not retain an exact orphan PID.

The existing DuckDB/Quack branch contributed 345 unique commits. Its core
database contract suite passes 75 tests on the integrated tree, but the current
host has DuckDB 1.5.2 without the Quack extension. Moreover, the production
database daemon opens the file directly instead of consuming
`QuackStateRepository`. A three-process launch would therefore violate the
single-writer safety contract. R1-S1 bootstraps with exactly one explicit embedded
DuckDB writer; its parallel waves remain planned but dispatch serially until a
tested Quack-backed claim path and capability report authorize an immutable
concurrency revision. This is a qualification limitation, not fabricated
parallelism.

The same host cannot activate the checked-in DuckLake profile: the runtime is
DuckDB 1.5.2, while `ipfs_datasets_py.ducklake.capabilities` pins exact 1.5.5
artifacts for its lake-enabled environment, including DuckLake and httpfs. No
accepted catalog profile or environment binding exists. Those lake pins are
not prerequisites for continuous DuckDB + Quack qualification. DuckLake stays
fail-closed and off for live execution while hermetic adapter, outage,
recovery, security, parity, and benchmark qualification proceeds. Only after
the separate DuckDB + Quack control-plane chain is admitted may live DuckLake
be promoted with its additional DuckLake/httpfs capability receipt, catalog
profile/binding, projection/security/recovery evidence, and release receipt.

A fresh read-only probe also rejects continuous Quack operation on this exact
host: DuckDB is 1.5.2; Quack reports `install-required`, unavailable, failed
health, not installed, not loaded, and `import_only_insufficient`. Network
installation is forbidden and only the generic accelerator DuckDB 1.5-minor
compatibility profile is present, not the exact sealed 1.5.5 profile. This
observation is no-go evidence, not yet a stable capability-receipt CID. The
current `DatabaseImplementationDaemon` Quack mode still fails closed/direct
DuckDB and is not a safe multi-writer path.

`LGSWF-001` persists the full revision ledgers and replays every baseline hash.
A mismatch quarantines the baseline instead of changing source authority.

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

The nested `ipfs_datasets_py.ducklake` package is also reused, but it is not a
semantic authority. Its reviewed public typed capability, catalog-owner,
registry, ingest, snapshot, execution/query, security, recovery, and release
contracts may store and query content-addressed projections. Accelerator code
must not issue DuckLake `ATTACH`, raw SQL, catalog-file access, credential
access, or unrestricted object-store requests.

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

- semantic consumption currently has no `agent_supervisor/semantic_state/`
  package on the actual source. Existing change-propagation, semantic
  dependency, capsule-reference and ContextPack consumers are reusable
  projections; B and C must add only the missing accelerator-owned reference
  contracts after A freezes their exact interfaces;
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
- operational persistence: in the accelerator DuckDB + Quack control plane,
  DuckDB remains the authoritative data model for goals, plans, tasks, claims,
  leases, resources, the projection outbox cursor, and accepted results, while
  Quack is mandatory for qualified multi-process access and exclusive
  state-owner serialization. The existing
  `DatabaseArtifactStore@1` export/projection receipt semantics are reusable,
  but its incompatible unsealed DDL and the incompatible database-event-log
  DDL are never opened against operational-v1.

Epic A freezes the actual `StateRepository`, `QuackStateRepository`,
`QuackStateClient`, `QuackStateServer`, `StateServerIdentity`,
`ControlPlaneStoreIdentity`, `StateCommand`, and `DatabaseProgramConfig`
surfaces plus server-side scoped authorization from datasets
`duckdb_control.quack_security`. Later integration must consolidate task,
intent, coordination, execution, attempt, provider, effect, validation, and CAS
access behind that canonical repository/transaction gateway. Seven labels or
receipts without one runtime-effective gateway are insufficient.

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

DuckLake owns none of those operational dimensions. It also owns no semantic
truth merely by storing a projection. Only canonical datasets semantic APIs can
assert semantic identity or meaning, and only the accelerator DuckDB + Quack
control plane can advance operational authority through DuckDB CAS/fences at
the Quack state-owner boundary.

The datasets `SemanticStateRoot` remains semantic-only. Operational records
reference semantic CIDs; they never add claims, paths, workers, prompts,
provider payloads, credentials, or mutable state to that root.

## 5. Target architecture

```text
datasets semantic authorities             accelerator operational authorities
index/state/capsules/contracts/proofs      DuckDB records/CAS + Quack transport
                                           goals/plans/claims/resources/merge
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

 DuckDB + Quack control-plane outbox (DuckDB-authoritative cursor/acceptance)
                              |
               bounded typed history projector
                              |
 datasets DuckLake events/artifacts/benchmarks (optional projection/query)
```

### 5.1 DuckDB + Quack-orchestrated DuckLake projection boundary

The projector reads verified, read-only operational-v1 snapshots of task and
phase state, provider/effect receipts, domain events, artifacts, and metrics.
It emits only the three existing logical datasets: `events`, `artifacts`, and
`benchmarks`. It reuses `DatabaseArtifactStore@1` export/projection receipt
semantics and the public typed datasets DuckLake contracts; it does not create
a second lake, artifact store, event store, plan store, or task authority.

DuckDB owns the outbox item, cursor, idempotency key, retry state, and
acceptance; in multi-process mode those transactions are accessible only
through the sealed Quack state-owner service. A batch is bounded to at most
5,000 rows, 16 MiB, and 10 seconds, and no lake or network call occurs inside a
DuckDB authority transaction.
DuckLake success produces a content-addressed projection receipt that DuckDB
records after the external operation. On outage, projection pressure is
isolated from analysis, scheduling, execution, merge, and acceptance; replay
resumes idempotently from the DuckDB cursor. DuckLake results are read-only
analytics and candidate evidence, never input authority for scheduling or goal
completion.

### 5.2 Operational world overlay

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

### 5.3 Semantic work binding and completion

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

### 5.4 Composite work and conflict graphs

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

### 5.5 Deterministic parallel frontier

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

### 5.6 Resource scheduling and coordination

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

Projection has its own queue, persistence, ingest, recovery, and query
backpressure. A DuckLake outage cannot consume the embedded DuckDB writer lease
or Quack state-owner lease and cannot stop CPU-only work. Qualification targets
are control-plane heartbeat p99 at or below 50
ms, commit-latency regression no greater than 5% at p95 and 10% at p99,
projection throughput at least twice observed peak production, and drainage of
a 30-minute backlog within 30 minutes. These are targets, not current results.

Supervisors advertise observations, not authority. One fenced writer owns each
mutable coordination shard; peers read snapshots, submit proposals, claim
admitted work, and publish immutable evidence. Failover advances the epoch and
prevents an old coordinator from committing. Partitioning uses semantic/goal/
repository/resource/provider/worktree/merge locality while preserving explicit
cross-partition edges. Work stealing requires eligibility, a current checkpoint
where applicable, and a later fence. Execution may be at least once; logical
acceptance is exactly once for task, revision, base tree/root, and idempotency
key.

In continuous multi-process mode, that writer is the fenced Quack state owner
over DuckDB transactions and peers are Quack clients. Quack loss fails closed;
it never silently falls back to direct shared DuckDB access. The separately
qualified embedded one-writer mode can resume only through an explicit profile
transition and fresh authority checks.

Configured-board and runner composition must propagate the live endpoint and
exact `StateServerIdentity`, own deterministic server start/stop, and pass an
independent remote readiness probe rather than treating local construction as
readiness. Server-side scoped authorization is enforced through a thin
high-level adapter to datasets `quack_security`; client declarations do not
authorize an operation. Qualification audits every production path for direct
file opens and rejects any separate coordination or execution authority.

### 5.7 Daemon packets, refresh, refill, and convergence

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
| W0 | trusted manual acceptance of successor `LGSWF-006` | corrected operational-only control seal; quarantined `LGSWF-000` is never reused |
| W1 | `001`, `002`, `003` | revision/inventory, DAG/interfaces, datasets producer readiness |
| W2 | `004`, then `005` | current semantic root and accepted R2 binding |
| W3 | `010`-`013` | operational world overlay |
| W4 | `020`-`023` | semantic bindings and completion gates |
| W5 | `030`-`033` | composite and conflict graph |
| W6 | `040`-`043` | deterministic frontier planner |
| W7 | `050`-`053` | resource scheduling, reuse, backpressure |
| W8 | `060`-`062` | supervisor fabric, Quack state-owner model and failover |
| W9 | `070`-`072` | packets, checkpoints, Quack repository integration, stale/steal fencing |
| W10 | `080`-`082` | bounded revision, refill, plan doctor |
| W11 | `090`-`092` | closed-loop semantic refresh |
| W12 | `100` | fixed-point convergence |
| W13 | `110` | receipts, metrics, entrypoint |
| W14 | `120`-`122` | deterministic and adversarial fault qualification |
| W15 | `130`-`131` | A/B/C/D benchmark and report |
| W16 | `140`-`141` | content-addressed release and go/no-go |

## 7. Qualification design

The C1 bootstrap population passed the five designated exact-byte-state
diagnostic suites with exit code 0, one pre-existing deprecation warning per
suite, and no skips or failures:

| Designated suite | Collected / passed | Skipped | Failed | Duration |
|---|---:|---:|---:|---:|
| operational schema | 14 / 14 | 0 | 0 | 4.09 s |
| intent repository (`test/api/test_agent_supervisor_intent_repository.py`) | 12 / 12 | 0 | 0 | 8.80 s |
| semantic/proof writer guards | 62 / 62 | 0 | 0 | 13.37 s |
| coordination, daemon, Portal and runner | 89 / 89 | 0 | 0 | 43.49 s |
| bootstrap materialization seal | 32 / 32 | 0 | 0 | 91.93 s |

That is 209/209 diagnostic tests. These results qualify the frozen C1
code/test population but do not self-accept LGSWF-006: the final clean C2 (or
later integrated) source must still pass the configured materializer and emit
its self-addressed qualification receipt. Continuous multi-supervisor
operation remains **NO-GO**.

DuckLake qualification is separate from that 209-test bootstrap evidence.
`LGSWF-110` must seal 19 direct plus 7 adjacent projection cases, including
typed API enforcement and exact outbox/projection receipts. Epics M and N then
exercise outage, replay, corruption, capability/profile forgery, recovery, and
DuckDB-only versus DuckDB-plus-DuckLake measurements. Required projection
outcomes are 100% parity, zero duplicate or missing logical rows, cold recovery
RPO 0 and RTO at most 300 seconds, plus the latency/throughput/backlog targets
above. Results are reported honestly when a target is missed.

The deterministic fixture uses three supervisors and ten daemons, multiple
resource classes, independent and conflicting branches, a multi-level goal
graph, proof/validation tasks, merge pressure, and refill triggers. Its
multi-process qualification lane requires a real exact-profile Quack service;
fakes are restricted to unit tests and cannot satisfy remote-readiness or
release gates. It exercises all 26 required coordination/fault behaviors and
all critical adversarial inputs. Critical cases must fail closed.

The benchmark compares:

- A: one supervisor/one daemon/serial;
- B: one supervisor/multiple daemons/dependency-only;
- C: multiple supervisors/daemons/conflict-aware;
- D: the complete semantic fabric with reuse, resource awareness, adaptive
  plan operations, work stealing, incremental verification, and refill.

Each feasible configuration also compares embedded one-writer DuckDB with the
sealed DuckDB + Quack multi-reader/multi-writer control plane, then runs the
qualified control-plane mode with DuckLake projection off and with a hermetic
DuckLake projection. This isolates Quack transport/state-owner cost and
projection cost, query value, replay behavior, and storage effects without
pretending the lake is operational authority. Unavailable Quack cells are
typed, not replaced by unsafe direct multi-process DuckDB. Live DuckLake is not
used until its separate promotion receipt is accepted.

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

DuckLake rollback disables projection dispatch, leaves the authoritative
DuckDB outbox/cursor intact, quarantines partial lake receipts, and later
replays from DuckDB. It never rewinds an accepted task, plan, lease, resource
reservation, or semantic root. Credentials and catalog locations remain behind
the datasets typed API boundary.

## 9. Stop/go policy

The supervisor continues while safe work exists. It does not report success for
`blocked_external_dependency`, `resource_unavailable`, `provider_unavailable`,
`semantic_analysis_inconclusive`, `verification_inconclusive`,
`human_review_required`, `bounded_exhaustion`, `no_progress`, `policy_denied`,
`quarantined`, or `cancelled`.

The current release starts with three explicit limitations: no current semantic
root, no checked-head adversarial-assurance package, and no admitted continuous
DuckDB + Quack control plane or live DuckLake extension. The former lacks its
exact pinned DuckDB/Quack profile and fully Quack-backed operational repository
set; the latter separately lacks DuckLake/httpfs pins, catalog/profile/binding,
projection/security/recovery evidence, and a release receipt. A tasks may use bounded
raw-source fallback to build/fix the producer and establish the root. Every
later task is marked `REBIND_REQUIRED_BY_LGSWF-005` and may dispatch only after
Plan Revision R2 binds exact semantic/capsule/contract/obligation identities.
If A cannot establish those identities, the qualification decision is no-go;
completion criteria may not be weakened.

Core embedded one-writer supervisor qualification is not blocked by an
optional DuckLake outage. Continuous DuckDB + Quack multi-supervisor operation
and live DuckLake activation receive separate go/no-go decisions. Continuous
mutation remains **NO-GO** until the exact 1.5.5 profile and Quack-backed task,
coordination, attempt, provider, effect, validation and CAS repositories pass
through one canonical gateway, remote readiness and server authorization pass,
and the direct-file-open audit is clean.
That decision does not depend on DuckLake or httpfs. The DuckLake decision
remains **NO-GO** until the control plane is admitted and the additional exact
DuckLake/httpfs pins, catalog profile/binding, projection/security/recovery
evidence, and release authorization are all accepted.

## 10. Required final supervisor report

The release report must contain, in order: exact revisions; current-state
inventory; authority map; board/dependency graph; reused and changed modules;
world overlay; bindings; composite/conflict analysis; frontier algorithm;
resource policy; supervisor and daemon protocols; revision/refill; refresh and
fixed point; fault results; parallel/resource benchmarks; model/proof reuse;
scheduling overhead; security; limitations; qualification level; and an
explicit continuous-operation go/no-go. It also reports embedded-DuckDB versus
DuckDB + Quack and control-plane-only versus DuckLake-projected measurements,
with separate continuous Quack and live DuckLake activation recommendations.

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
