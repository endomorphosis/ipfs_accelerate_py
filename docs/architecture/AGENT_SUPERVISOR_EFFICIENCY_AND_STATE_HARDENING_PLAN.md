# Agent Supervisor Efficiency and State Hardening v1

Program: `agent-supervisor-efficiency-and-state-hardening-v1`
Short ID: `ASEH`
Root objective: `ASEH-G000`
Execution authority: the existing `ipfs_accelerate_py.agent_supervisor`

Normative requirements:
`docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json`.
That closed matrix is a protected bootstrap input; the scheduler config binds
its exact digest and the board validator verifies the binding before
materialization. Prose in this plan summarizes the matrix but cannot weaken it.

## Decision and outcome

ASEH is one bounded consolidation, measurement, and efficiency program. It
extends the existing agent supervisor and does not introduce another
supervisor, planner family, meta-controller, daemon family, memory system,
federation layer, procedure compiler, task database, event system, dashboard,
provider abstraction, or transport profile.

The selected handoff is:

1. reviewed objective heap and taskboard are immutable bootstrap inputs;
2. `DatabaseTaskSource@1` / `IntentRepository@1` materializes their exact
   identities into one DuckDB control store;
3. one loopback `QuackStateServer@1` exclusively owns the file during live
   execution;
4. `configured_board_scheduler.py` launches the existing multi-lane
   implementation supervisor with `task_source_kind=duckdb`,
   `authority_mode=quack`, and `failover_policy=fail_closed`;
5. DuckLake is an append-only, rebuildable analytics/history projection and is
   never consulted for claims, readiness, leases, completion, promotion, or
   recovery authority;
6. admitted validators, current-tree merge receipts, and independent promotion
   admission remain authoritative. Process liveness and Markdown status are
   observations only.

The canonical product path being consolidated is the current domain layout:

- intent: `agent_supervisor/objectives/`;
- task projection and transactional state: `agent_supervisor/task_sources/`;
- operation and promotion authority: `agent_supervisor/control/`;
- semantic context and routing consumers: `agent_supervisor/semantic_state/`,
  `context/`, and `verification/`;
- analysis and planning: `agent_supervisor/analysis/` and `planning/`;
- execution, validation, merge, and recovery: `todo_daemon/`, `runtime/`,
  `validation/`, `merge/`, and `rescue/`.

### Bounded pre-authority repair: ASEH-BOOTSTRAP-001

Inspection found that the current configured-board path no longer had a usable
credential handoff after the Quack owner removed the legacy raw token file.
Before the ASEH DuckDB authority exists, the bootstrap engineer may therefore
perform exactly one pre-authority repair. It is not a forty-first campaign
task, supervisor, daemon, state store, or alternate mutation path.

- Stable task ID: `ASEH-BOOTSTRAP-001`
- Owning repository: `ipfs_accelerate_py`
- Dependencies: none; it runs before materialization
- Exact outputs: the existing typed-owner/client, Quack owner, configured
  scheduler/supervisor descriptor-propagation paths, the bounded ASEH
  operator, their focused test, the exact ASEH runtime ignore rule, and the
  bootstrap validation observation declared in scheduler config
- Validation: `python3 -m pytest -q test/api/test_agent_supervisor_configured_typed_grant_handoff.py`
- Risk: `R4_SECURITY_OR_PROTOCOL_SENSITIVE`
- Authority: repair only the existing PID-bound, fail-closed typed handoff;
  no general task CAS, new owner, new store, or provider/validator credential
- Terminal success: a real owner-to-configured-supervisor read/claim/transition
  path works without a legacy disk token, while untrusted children, stale
  identities, unscoped operations, and direct DuckDB opens remain denied
- Terminal non-success: any broad grant, bypass, credential leak, stale-token
  preference, descriptor loss, or failed focused test prevents materialization

The full machine-readable task contract is `bootstrap_handoff_repair` in the
scheduler config. Raw test exit status is an observation until the canonical
bootstrap validator admits it.

## Exact bootstrap baseline

The dirty or stale outer checkouts are inspection evidence, not launch
authority. The accelerator baseline is its fetched `origin/main` commit. The
sibling commits are deliberately selected authority snapshots required by the
current accelerator work, not claims about either sibling's moving `main`.
ASEH runs from an isolated branch rooted at these immutable trees:

| Repository | Commit | Tree | Authority |
| --- | --- | --- | --- |
| ipfs_accelerate_py | `755f45475cc2d13dacd8b330036c1d597afeddde` | `729da9f8293ecfa046a0136381a3d3808f9ed140` | supervisor state, execution, routing, validation, reconciliation, terminalization, promotion |
| ipfs_datasets_py | `209dbe2765593fbc6efe8e9281c34f2e8f6e37a6` | `95f54df34585d0b736706fd90c83f55954489ad9` | semantic identity, canonical IR, ContextPack construction, obligations, dependency meaning, proof relationships |
| ipfs_kit_py | `ba5508d940fb5b23a6d0d9b2084f5195cd26a671` | `7c71efa93c4e4124d12fa05515868df3a5344b2e` | canonical bytes, CID verification, immutable blocks, current-root CAS, WAL, recovery and retention |

The bootstrap baseline manifest records the exact selected remote refs, the
accelerator baseline's prior gitlinks, and typed pending fields for the final
bootstrap commit/tree. Materialization replaces those pending fields with the
committed superproject forest; no placeholder is accepted as runtime evidence.

The final bootstrap commit containing this plan, objectives, board, scheduler,
validator, and materializer is bound by the materialization receipt before any
task claim. Any later source change is a new tree and cannot reuse the
bootstrap seal as implementation or promotion evidence.

Observed local capability at bootstrap: Python 3.12.3, DuckDB 1.5.5, compatible
Quack `c154811` with fingerprint
`sha256:b77954ae50ecc06e10c6e20fc6fd421d73b5c31cf72bb60ae3f29b1f8a85f20b`,
and installed DuckLake `d8a1881e`. These observations establish attemptability,
not task success or production qualification.

## Cross-repository architecture decision

### Datasets authority

Extend the existing sole semantic builder
`ipfs_datasets_py.proof_context.context_pack.build_context_pack`; reconcile its
current `DatasetsContextPackAuthority@0.1` and the accelerator's existing
`ContextPack@1` consumer contract rather than minting
`SupervisorContextPack@1` as a competitor. Datasets owns semantic identity,
scope meaning, contracts, obligations, lineage, completeness witnesses,
counterexample minimization, CEGAR, unsat-core use, and qualified interpolation.
It never grants execution, persistence, proof-reuse admission, or promotion
authority.

### Kit authority

Extend the existing `ipfs_kit_py.proof_context.state_store`,
`incremental_seal_store`, and `verification_store` canonical-byte/CID,
state-root CAS, WAL, and recovery facilities. Candidate ContextPack storage is
separate from current-root publication. Exact stored bytes prove durability,
not semantic correctness or proof/test reuse admission. Package fixtures and
contract vectors are installed package data; cross-repository tests never
import a sibling source tree's `tests/` package.

### Accelerate authority

Accelerate verifies pack freshness, admits exact test/proof reuse, runs the
deterministic-first route, owns task claims/leases/fences/idempotency/retries,
reconciles unknown provider outcomes, validates and merges patches, terminalizes
once, and evaluates promotion. It consumes Datasets semantic identities and Kit
durable identities without reminting them.

## Non-negotiable invariants

Every implementation and qualification task preserves all of the following:

- simulated, estimated, attempted, observed, and verified are distinct typed
  states;
- missing telemetry is `unavailable`, never numeric zero;
- a stored proof is not admitted or reusable until the current-tree reuse gate
  accepts its repository, tree, objective revision, policy revision,
  interfaces, schemas, toolchain, and environment;
- stale leases or fences cannot complete, and provider-unknown effects enter
  reconciliation rather than blind retry;
- only the canonical transition service mutates production task state;
- merge is not completion and failed validation cannot become success through
  merge bookkeeping;
- a policy candidate, model, controller, or branch cannot authorize its own
  promotion;
- validation reserves cannot be consumed by provider calls;
- deterministic stages cannot be skipped because a larger model is available;
- hermetic evidence is never described as live or sufficient for promotion.

Promotion hard gates are exactly zero for false completions, unauthorized
mutations, simulated-as-live outcomes, stale authoritative admissions,
confirmation replay, path/scope escape, hidden validation reduction,
controlled critical omissions, escaped critical seeded defects,
self-authorized policy promotion, selected-test false negatives, double
terminalization, double idempotent effect execution, and stale-fence completion.

## Workstreams and bounded deliverables

### G010 — Authority inventory and sealed baseline

`ASEH-000` inventories every task/objective mutation, provider launch, test or
prover run, patch/merge, completion, authoritative receipt, policy pointer,
fallback, and recovery path. It classifies each as canonical, adapter,
deprecated, fixture, or removal candidate and publishes the required canonical
authority ADR. `ASEH-001` seals exact commits,
trees, policy/config identities, provider set, resource limits, task/acceptance
corpora, cost method, benchmark digest, and environment identity. These two
tasks may inspect in parallel; no candidate behavior may land until both are
accepted.

### G020 — Real paired efficiency measurement

`ASEH-010` through `ASEH-015` extend the existing benchmark causal-span and
provider ledgers with closed schemas for all requested identity, model, compute,
cost, work, validation, proof, retry, merge, human, and terminal fields.
Provider-reported usage is preferred; estimates are labeled with a price
snapshot and never overwrite unavailable measurements. The paired harness has
three arms (direct/minimal orchestration, sealed current supervisor, candidate),
identical inputs and gates, a 60+ hermetic development corpus, 20+ exact-tree
historical replays spanning all named outcomes, and a 10+ newly encountered
live shadow/canary cohort.

### G030 — Deterministic-first routing

`ASEH-020` through `ASEH-024` consolidate the existing semantic-state and
verification routers into one ordered ladder: exact authoritative receipt;
AST/symbol/dependency/impact; schema/type/static/lint/contract; selected tests;
incremental prover; local small specialist; local/remote medium; remote
frontier; human. A model call needs a closed unresolved-question record and is
rejected when its answer cannot change an admissible decision. Route receipts
record every run/skip, decisive evidence, escalation, executor, measured
resource use, and outcome.

### G040 — Canonical cross-repository ContextPack

`ASEH-030` through `ASEH-035` extend the existing ContextPack contract and its
Datasets builder, Kit `proof_context` state/incremental-seal/verification
stores, and Accelerate selector. The contract binds
all identity, scope, contracts, validation, history, questions, budgets,
freshness, invalidation, and parent/delta fields listed by the root objective.
Whole-repository expansion requires a typed completeness failure; incremental
expansion requests only named missing CIDs, symbols, contracts, tests,
counterexamples, or obligations. Benchmarks report reuse, before/after tokens,
expansion precision/recall, omissions, stale rejection, build/retrieval cost,
and net provider effect.

### G050 — Canonical control-plane state machine

`ASEH-040` through `ASEH-045` inventory and extend the existing supervisor
state model, provider-attempt/recovery surfaces, and a candidate transition
service over the transactional intent repository. The machine-readable transition
table covers the requested discovered-to-terminal states and events. Executable
invariants cover single owner, CAS, idempotency, lease/fence freshness,
unknown-outcome reconciliation, single terminalization, current validation
before merge, external-effect preservation, deterministic restart,
event/materialized-state reconciliation, receipt-required success, and
non-retroactive policy changes. Model/property/crash/race/replay tests use
Datasets formal capabilities where suitable. Production integration is the
owner-paused, exact-path `ASEH-061` cutover through the existing
IntentRepository and typed Quack owner; the candidate service is never a
second writable authority.

### G060 — Analysis, planning, and synthesis

`ASEH-050` through `ASEH-055` consolidate existing dependency graphs,
assume/guarantee contracts, formal replanning, counterexample/unsat-core/CEGAR
machinery, deterministic repair transforms, proposal validation, and merge
admission. Plans retain accepted prefixes and replan only invalidated suffixes.
Leaf work has deterministic acceptance. A typed PatchPlan binds base tree,
semantic intent, files/symbols, conditions, invariants, tests/proofs, scope, and
digest; empty, secret-bearing, generated-artifact, stale, or out-of-scope
patches fail closed.

### G070 — Consolidation and migration

`ASEH-060` through `ASEH-062` designate one owner for each mutable fact,
integrate supported callers into the existing IntentRepository-to-typed-Quack
path during drained, owner-paused staged maintenance, add bounded warnings,
document rollback, and qualify installed exact-tree contracts across all three
repositories. No public API
is removed before equivalent supported behavior, a replacement, caller
migration, and loss of independent write authority are proved. The legacy
inference coordinator is only touched if it contaminates measurement, and all
benchmarks consume real capability reports and real CIDs.

### G080 — Qualification and honest release

`ASEH-070` through `ASEH-075` attempt the sealed 60+ hermetic, 20+ replay, and
10+ live-shadow populations, then run a gated low-risk canary only when shadow
evidence admits it. Every cohort emits either a qualified receipt or a typed
insufficiency, safety/quality, or not-admitted receipt by its bounded deadline.
Paired results include median and mean
differences, per-task ratios, bootstrap confidence intervals, task-class
distributions, outliers, quality-adjusted cost, accepted-patch rate, and
time-to-terminal outcome. `ASEH-074` emits exactly one of
`promotion_eligible_operator_authorization_required`,
`non_promoted_unmeasured`, `non_promoted_safety_or_quality`, or
`non_promoted_efficiency`. It never mutates the policy pointer. `ASEH-075`
publishes the machine-readable and human reports with exact final trees,
limitations, missing evidence, residual risk, and marginal-return guidance.

## Parallel execution graph

Wave 0 starts `ASEH-000` and `ASEH-001` concurrently. Their join opens five
independent fronts: telemetry/schema, routing, ContextPack contract,
state-machine definition, and dependency analysis. Cross-repository semantic
and storage work uses isolated submodule-bound worktrees. Shared runtime,
contract, and merge files have explicit dependency serialization. Qualification
is a final fan-in and never runs early.

```text
000 + 001
  ├─ 010 → 011 → 012 → 013 → 014 → 015
  ├─ 020 → 021 → 022 → 023 → 024
  ├─ 030 → 031 + 032 → 033 → 034 → 035
  ├─ 040 → 041 → 042 → 043 → 044 → 045
  └─ 050 → 051 → 052/053/054 → 055
                joined implementations → 060 → 061 → 062
                all evidence → 070 → 071 → 072 → 073 → 074 → 075
```

Dependencies on the executable board, not this diagram, are scheduling
authority.

## Benchmark and promotion policy

All three paired arms use the same repository revision, objective, task input,
acceptance tests, available providers/models, price snapshot, resource limits,
maximum retries, and human-intervention policy. Direct/minimal orchestration
retains safe isolation and cannot mutate state directly. Current-supervisor
behavior is the exact pre-candidate policy. Candidate behavior starts in
shadow.

Live enrollment has a bounded window recorded in the cohort manifest. Fewer
than ten qualifying tasks at the deadline is a valid, truthful
`insufficient_live_population` receipt and leads to
`non_promoted_unmeasured`; it is not an indefinite wait and is not fabricated
live evidence. Cohort tasks succeed when they produce and validate an honest
measured, unavailable, or not-admitted receipt. They fail only when they cannot
produce a valid receipt, so expected negative evidence can still reach the
promotion-disposition and final-report tasks.

Promotion eligibility requires all hard gates at zero plus: median live input
tokens down at least 30%; weighted provider cost down at least 20%; frontier
calls down at least 40%; retry tokens down at least 25%; at least 60% of
low-risk tasks resolved deterministically or by a small model; eligible
ContextPack reuse at least 50%; manual recovery below 2%; no statistically
meaningful accepted-patch quality degradation; overall wall time no more than
10% worse; and positive savings after audit/verification overhead. Missing
measurements cannot be imputed.

Promotion itself remains a separate operator-authorized, CAS-protected action
after the campaign. Board drain proves bookkeeping only.

## Validation and evidence discipline

Every task has stable identity, one owning repository, exact outputs,
dependencies, validation commands, risk, authority, and explicit terminal
success/non-success criteria in the taskboard. Focused, schema/contract, and
negative tests are bound to the current task tree and policy. Raw logs are
non-authoritative. External effects require observation and admitted verifier
evidence, not exit status alone.

Cross-repository checks validate canonical bytes, real CID equality, closed
unknown-field behavior, numeric bounds, stale identity rejection, missing
dependency behavior, and installation without sibling source-tree tests.
Simulated providers and fixtures never satisfy a live gate.

## Launch and recovery contract

The reviewed sequence is:

```bash
python3 scripts/validate_agent_supervisor_efficiency_state_hardening_board.py \
  --check-all --json
python3 scripts/run_agent_supervisor_efficiency_state_hardening.py materialize
python3 scripts/run_agent_supervisor_efficiency_state_hardening.py preflight
python3 scripts/run_agent_supervisor_efficiency_state_hardening.py run --implement
python3 scripts/run_agent_supervisor_efficiency_state_hardening.py status \
  --require-ready
```

The combined operator is an adapter around the existing owner and configured
scheduler. Starting a standalone owner would omit the repaired typed-grant
handoff and is therefore not an ASEH launch path.

Health requires a ready owner identity, successful authenticated Quack query,
current store generation/schema, fresh coordinator/lane heartbeats, an exact
ready or claimed frontier, advancing events/receipts, and no active typed
blocker. A PID alone is insufficient. Diagnosis uses status/events/receipts,
the implementation supervisor's bounded recovery, and
`duckdb_quack_doctor.py`; state is never deleted or directly edited to make a
board appear healthy.

## Deferred, non-executing backlog

Ideas outside the forty sealed tasks are recorded only in the final residual
report. They do not create tasks, refill epochs, new planners, or follow-on
programs automatically.
