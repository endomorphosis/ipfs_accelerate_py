# Agent Supervisor Efficiency and State Hardening objective heap

Program: `agent-supervisor-efficiency-and-state-hardening-v1`. The records
below are sealed bootstrap inputs for the existing supervisor. DuckDB becomes
the mutable goal authority after materialization; Markdown never proves
completion.

```text
ASEH-G000  Increase measured efficiency and harden authoritative state
|-- ASEH-G010  Inventory authorities and seal the baseline
|-- ASEH-G020  Measure paired token, compute, cost, work, and quality
|-- ASEH-G030  Consolidate deterministic-first routing
|-- ASEH-G040  Qualify the canonical cross-repository ContextPack
|-- ASEH-G050  Enforce one control-plane state machine
|-- ASEH-G060  Improve canonical analysis, planning, and synthesis
|-- ASEH-G070  Consolidate and migrate duplicate paths
`-- ASEH-G080  Qualify honestly and issue a promotion disposition
```

## ASEH-G000 Increase measured efficiency and harden authoritative state

- Status: active
- Parent:
- Depends on:
- Priority: P0
- Track: aseh-root
- Goal: Increase measured token and compute efficiency, state-management reliability, and planning/synthesis quality without weakening safety, authority, validation, or promotion gates.
- Completion contract: ASEH-000 through ASEH-075 have authoritative current-tree terminal receipts; all required paired and live evidence exists or the release is honestly non-promoted; hard safety gates remain zero; the control plane drains without direct database repair.
- Evidence: aseh/release-report@1, aseh/promotion-decision@1, admitted current-tree validation receipts
- Acceptance criteria: canonical-authorities; paired-measurement; live-cohort; zero-hard-gate-violations; settled-control-plane; truthful-disposition
- Outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_FINAL_REPORT.md
- Validation: python3 scripts/validate_agent_supervisor_efficiency_state_hardening_board.py --check-all
- Acceptance: Board drain is bookkeeping only; production promotion remains an independent operator-authorized CAS action.
- Gap task: ASEH-000 through ASEH-075

## ASEH-G010 Inventory authorities and seal the baseline

- Status: active
- Parent: ASEH-G000
- Depends on:
- Priority: P0
- Track: authority-baseline
- Goal: Inventory every state, execution, validation, receipt, recovery, merge, and policy path and seal exact repository, policy, environment, corpus, and accounting identities before candidate changes.
- Completion contract: ASEH-000 and ASEH-001 are accepted; every discovered path has an owner, status, entry point, store, bypass/fallback/metric classification, and migration disposition; the architecture decision record binds the canonical supervisor, cross-repository, state, routing, and ContextPack authorities.
- Evidence: aseh/authority-inventory@1, aseh/sealed-baseline@1
- Acceptance criteria: exact-trees; exhaustive-scoped-paths; typed-unknowns; no-candidate-before-seal
- Outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json, docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md
- Validation: focused inventory and baseline schema tests
- Acceptance: Plans and historical prose cannot substitute for current source or observed capability evidence.
- Gap task: ASEH-000, ASEH-001

## ASEH-G020 Measure paired token, compute, cost, work, and quality

- Status: active
- Parent: ASEH-G000
- Depends on: ASEH-G010
- Priority: P0
- Track: measurement
- Goal: Produce reusable closed telemetry and paired benchmark evidence for direct, sealed-current, and candidate cohorts with missing values represented as unavailable.
- Completion contract: ASEH-010 through ASEH-015 are accepted; the 60+ hermetic inputs and historical/live cohort contracts are sealed, distinguish measured, estimated, unavailable, simulated, observed, and verified facts, and emit typed bounded insufficiency dispositions when 20+ replay or 10+ live enrollment evidence is unavailable by its sealed deadline.
- Evidence: aseh/task-efficiency-receipt@1, aseh/paired-benchmark-manifest@1, aseh/cohort-manifest@1
- Acceptance criteria: causal-identity; actual-usage-when-available; labeled-estimates; paired-input-equality; audit-overhead-included
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/efficiency_receipts.py, benchmarks/agent_supervisor/efficiency_state_hardening/manifest.json
- Validation: focused telemetry, schema, pairing, and negative truth-state tests
- Acceptance: Hermetic or replay evidence never satisfies the live promotion gate.
- Gap task: ASEH-010 through ASEH-015

## ASEH-G030 Consolidate deterministic-first routing

- Status: active
- Parent: ASEH-G000
- Depends on: ASEH-G010
- Priority: P0
- Track: deterministic-routing
- Goal: Make one existing router execute the exact receipt-to-human ladder and require a decision-relevant unresolved-question record before every model escalation.
- Completion contract: ASEH-020 through ASEH-024 are accepted; every route has a closed receipt; non-decision-changing model calls fail closed; stage skips and actual resource use are explicit.
- Evidence: aseh/unresolved-question@1, aseh/routing-decision@1, aseh/route-receipt@1
- Acceptance criteria: ordered-ladder; deterministic-share; smallest-adequate-executor; typed-escalation; no-availability-escalation
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/verification/model_route.py
- Validation: focused routing, escalation, cache-freshness, and negative bypass tests
- Acceptance: A larger available model is never a reason to skip a prior stage.
- Gap task: ASEH-020 through ASEH-024

## ASEH-G040 Qualify the canonical cross-repository ContextPack

- Status: active
- Parent: ASEH-G000
- Depends on: ASEH-G010
- Priority: P0
- Track: context-pack
- Goal: Extend the current Datasets semantic builder, Kit durable store/root CAS, and Accelerate freshness selector into one versioned ContextPack contract with incremental expansion.
- Completion contract: ASEH-030 through ASEH-035 are accepted; canonical bytes/CIDs round-trip across installed packages; stale and incomplete identities reject; fixture packs cannot masquerade as live; no repository remints another's identity.
- Evidence: datasets/context-pack@qualified, kit/context-pack-root@1, aseh/context-pack-admission@1
- Acceptance criteria: sole-semantic-builder; real-cid; immutable-store; separate-candidate-current-root; exact-freshness; delta-expansion
- Outputs: installed cross-repository contract modules, schemas, vectors, tests, and context efficiency report
- Validation: current-head installed-package contract, CID, stale-identity, omission, recovery, and benchmark tests
- Acceptance: Stored bytes prove durability only; Accelerate independently admits reuse and execution.
- Gap task: ASEH-030 through ASEH-035

## ASEH-G050 Enforce one control-plane state machine

- Status: active
- Parent: ASEH-G000
- Depends on: ASEH-G010
- Priority: P0
- Track: state-machine
- Goal: Formalize and enforce one transactional transition authority with CAS, leases, fencing, idempotency, reconciliation, restart recovery, and single terminalization.
- Completion contract: ASEH-040 through ASEH-045 are accepted; the common transition contract and existing recovery surfaces pass transition, reconciliation, crash, race, duplicate, and unknown-outcome properties; the exact production integration obligation is admitted to staged ASEH-061 rather than creating a second writer in this goal.
- Evidence: aseh/state-transition@1, aseh/recovery-decision@1, admitted invariant and recovery receipts
- Acceptance criteria: one-writer-authority; cas-every-mutation; fresh-fence; unknown-outcome-reconcile; receipt-required-success; deterministic-restart
- Outputs: machine-readable transition table, transition service, invariant validator, recovery/reconciliation implementation, model/property/crash tests
- Validation: focused state model, owner restart, fence race, duplicate delivery, idempotency, and reconciliation tests
- Acceptance: No controller or compatibility path may update authoritative task tables directly.
- Gap task: ASEH-040 through ASEH-045

## ASEH-G060 Improve canonical analysis, planning, and synthesis

- Status: active
- Parent: ASEH-G000
- Depends on: ASEH-G010
- Priority: P1
- Track: planning-synthesis
- Goal: Extend existing dependency, contract, formal replanning, counterexample refinement, deterministic repair, PatchPlan, validation, and merge paths without creating a planner family.
- Completion contract: ASEH-050 through ASEH-055 are accepted; AST/dependency analysis precedes model use; accepted plan prefixes survive suffix invalidation; leaves have deterministic acceptance; patches are exact-tree, scoped, nonempty, and currently validated.
- Evidence: aseh/task-contract@1, aseh/completeness-witness@1, aseh/patch-plan@1, admitted merge receipt
- Acceptance criteria: dependency-cones; assumption-guarantee; affected-suffix; counterexample-minimization; bounded-transform; current-validation
- Outputs: canonical dependency/planning/synthesis modules, PatchPlan schema, focused and negative tests
- Validation: focused dependency, plan-delta, CEGAR, deterministic-repair, PatchPlan, scope, secret, and merge-admission tests
- Acceptance: Model statements cannot substitute for observed tests, proofs, or effects.
- Gap task: ASEH-050 through ASEH-055

## ASEH-G070 Consolidate and migrate duplicate paths

- Status: active
- Parent: ASEH-G000
- Depends on: ASEH-G020, ASEH-G030, ASEH-G040, ASEH-G050, ASEH-G060
- Priority: P1
- Track: consolidation-migration
- Goal: Select one owner per mutable concern, demote or adapt duplicate paths, document migration/rollback, and qualify the installed cross-repository surface at current heads.
- Completion contract: ASEH-060 through ASEH-062 are accepted; after owner-paused staged integration, every production mutation traverses the existing IntentRepository and typed Quack owner; there are no two writable authorities; compatibility paths warn and cannot independently claim authority; all three installed-package contract suites pass.
- Evidence: aseh/authority-disposition@1, aseh/migration-report@1, aseh/cross-repository-qualification@1
- Acceptance criteria: one-owner-per-fact; no-direct-write; adapter-equivalence; documented-rollback; current-head-installation
- Outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/migration_matrix.json, migration/rollback documentation, qualification receipts
- Validation: duplicate-authority, legacy-warning, adapter-equivalence, package-installation, and cross-repository contract tests
- Acceptance: Public compatibility APIs remain until their replacement, caller migration, equivalence, and loss of independent authority are proved.
- Gap task: ASEH-060 through ASEH-062

## ASEH-G080 Qualify honestly and issue a promotion disposition

- Status: active
- Parent: ASEH-G000
- Depends on: ASEH-G070
- Priority: P0
- Track: qualification
- Goal: Run sealed hermetic, replay, live-shadow, and gated canary cohorts and publish measured paired results, safety/quality gates, limitations, residual risk, and an honest promotion disposition.
- Completion contract: ASEH-070 through ASEH-075 are accepted; every qualification task emits a current schema-valid paired, insufficient-evidence, safety-or-quality, or not-admitted receipt by its bounded deadline; qualified populations meet their minimum sizes, honest shortfalls flow to non-promotion, the disposition vocabulary is closed, and policy promotion itself is not executed.
- Evidence: aseh/paired-results@1, aseh/safety-report@1, aseh/promotion-decision@1, aseh/release-report@1
- Acceptance criteria: sixty-hermetic-or-typed-insufficiency; twenty-replay-or-typed-insufficiency; ten-live-or-bounded-deadline-insufficiency; shadow-before-canary; not-admitted-canary-is-nonmutating; hard-gates-zero-for-promotion; confidence-intervals-when-measurable; honest-nonpromotion
- Outputs: benchmark receipts, final machine-readable release report, final human-readable report
- Validation: held-out replay, live shadow, low-risk canary, statistics, safety, quality, and promotion-decision validation
- Acceptance: Missing evidence yields non_promoted_unmeasured; promotion remains operator-authorized and CAS-protected.
- Gap task: ASEH-070 through ASEH-075
