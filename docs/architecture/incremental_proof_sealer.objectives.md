# IncrementalProofSealer objective heap (IPS)

Machine-ingestible goal hierarchy for
`INCREMENTAL_PROOF_SEALER_PLAN.md`. The executable fixed-board projection is
`incremental_proof_sealer.todo.md` with task prefix `IPS-`.

## North star

Safely reuse independently verifiable repository proof units only when their
complete statement, dependency, environment, tool, circuit, key, selector, and
policy context is unchanged; re-prove invalidated units, update affected Merkle
branches, and publish a parent-bound seal without treating receipts, hashes, or
simulations as direct proof of execution.

## Goal tree

```text
IPS-G000  IncrementalProofSealer release
├── IPS-G010  Executable inventory and trust baseline
├── IPS-G020  Canonical proof-unit, manifest, identity, and cache-key contracts
├── IPS-G030  Dependency graph, selection, diff, and invalidation authority
├── IPS-G040  Immutable proof store, cache index, and proof forest
├── IPS-G050  Seal WAL, current-root CAS, recovery, and concurrency
├── IPS-G060  Proof-class admission, backend capabilities, keys, and provers
├── IPS-G070  Incremental planning, scheduling, execution, aggregation, and cost
├── IPS-G080  Full checkpoints, delta seals, verification, and compaction
├── IPS-G090  Public API, CLI, packaging, and migration
├── IPS-G100  Deterministic fixtures and positive lifecycle conformance
├── IPS-G110  Tamper, trust, crash-recovery, and adversarial conformance
├── IPS-G120  Forty-transition benchmark and performance evidence
└── IPS-G130  Trust documentation, final report, and release fan-in
```

## IPS-G000 IncrementalProofSealer release

- Status: active
- Parent:
- Depends on:
- Fib priority: 100
- Priority: P0
- Track: incremental-proof-sealing
- Bundle: incremental-proof-sealer/root
- Parallel lane: release
- Resource class: cpu-large
- Goal: Deliver the focused three-repository IncrementalProofSealer with exact reuse, invalidation, deterministic forest roots, accepted-parent transitions, fail-closed storage and scheduling, honest aggregation semantics, and measured savings.
- Evidence: ips/final-report@1, ips/release-conformance@1, ips/benchmark@1
- Acceptance criteria: all child goals terminal; zero stale/mismatched/simulated production acceptance; deterministic roots; all crash and tamper cases pass; actual and estimated measurements distinguished
- Outputs: docs/architecture/INCREMENTAL_PROOF_SEALER_REPORT.md, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing
- Validation: python -m pytest -q test/api/incremental_sealing ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing ipfs_kit_py/tests/proof_seal_store
- Acceptance: The required APIs and CLI operate over the canonical datasets and kit authorities; release evidence is current-tree, no required non-pass is accepted, and the report uses only defensible claim language.
- Gap task: IPS-056
- Refinement: Children own disjoint semantic, storage, execution, evidence, and release surfaces joined only after focused conformance gates.
- Conflict policy: Do not build a generic agent framework, GUI, MCP profile, payment path, network service, or universal zkVM.

## IPS-G010 Executable inventory and trust baseline

- Status: active
- Parent: IPS-G000
- Depends on:
- Fib priority: 100
- Priority: P0
- Track: reconnaissance
- Bundle: incremental-proof-sealer/baseline
- Parallel lane: baseline
- Resource class: cpu-medium
- Goal: Inspect exact bound commits, executable proof/storage/scheduler code and focused tests in all three repositories before implementation, and classify every proof/receipt path by what it actually establishes.
- Evidence: ips/accelerate-inventory@1, ips/datasets-inventory@1, ips/kit-inventory@1, ips/trust-matrix@1
- Acceptance criteria: exact revisions; real/simulated/mock/structural classification; trusted setup/key assumptions; baseline commands/results; direct-execution versus receipt-consistency distinction; existing authority map
- Outputs: docs/architecture/incremental_proof_sealer_inventory, docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_BASELINE.md
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-all
- Acceptance: No implementation task is ready before the three inventories and synthesis are complete; pre-existing failures remain explicit.
- Gap task: IPS-001, IPS-002, IPS-003, IPS-004
- Refinement: Run repository inventories in parallel, then synthesize one reviewed cross-repository boundary/trust record.
- Conflict policy: Read executable code and run existing focused tests; do not install dependencies, build/download keys, or rely on documentation claims alone.

## IPS-G020 Canonical proof-unit, manifest, identity, and cache-key contracts

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G010
- Fib priority: 100
- Priority: P0
- Track: datasets-contracts
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets
- Resource class: cpu-medium
- Goal: Make datasets the single authority for closed proof classes, ProofUnit and manifest schemas, canonical statements and identities, exact cache keys, and forest commitment codecs.
- Evidence: ips/proof-evidence-classes@1, ips/proof-unit@1, ips/manifest@1, ips/cache-key-vectors@1, ips/forest-codec-vectors@1
- Acceptance criteria: every required field and enum is closed; exact canonical round trips; secrets excluded; every cache-key mutation invalidates; duplicates/order rejected
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing
- Acceptance: Models are deterministic, versioned, immutable, complete, and imported without optional capability or environment side effects.
- Gap task: IPS-005, IPS-006, IPS-007, IPS-008, IPS-009, IPS-010, IPS-011, IPS-012
- Refinement: Separate proof semantics, unit schema, identities, key, manifest, statements, and Merkle codec so independently reviewable tasks can proceed with disjoint files.
- Conflict policy: Extend existing canonical CID/ZK authorities; do not copy models into kit or accelerate.

## IPS-G030 Dependency graph, selection, diff, and invalidation authority

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G020
- Fib priority: 90
- Priority: P0
- Track: datasets-invalidation
- Bundle: incremental-proof-sealer/datasets-invalidation
- Parallel lane: datasets
- Resource class: cpu-medium
- Goal: Build the reason-labeled proof dependency graph, deterministic repository diff and requirement discovery, exact invalidation closure, full-fallback classification, and human-readable explanations.
- Evidence: ips/dependency-graph@1, ips/requirement-discovery@1, ips/diff-classification@1, ips/invalidation-matrix@1
- Acceptance criteria: all required edge/change types; transitive roots; uncertain closure broadens; test additions/deletions explicit; documentation-only preservation; full fallback rules
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/dependency_graph.py, ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/invalidation.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_dependency_graph.py ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_invalidation.py
- Acceptance: No unit is reused after any relevant transitive dependency or trust-context change; unrelated units remain reusable.
- Gap task: IPS-013, IPS-014, IPS-015, IPS-016, IPS-017
- Refinement: Graph schema, discovery, diff, invalidation, and conformance are independently testable pure layers.
- Conflict policy: Unknown semantic coverage is never interpreted as proof of independence.

## IPS-G040 Immutable proof store, cache index, and proof forest

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G020
- Fib priority: 90
- Priority: P0
- Track: kit-storage
- Bundle: incremental-proof-sealer/kit-storage
- Parallel lane: storage
- Resource class: io-medium
- Goal: Make kit the single narrow authority for immutable proof-seal artifacts, candidate cache indexing, corruption detection, optional IPFS transport, and affected-branch forest persistence.
- Evidence: ips/store-protocol@1, ips/local-store@1, ips/ipfs-adapter@1, ips/cache-index@1, ips/forest-store@1
- Acceptance criteria: exact-byte rehash; closed kinds; candidate-only lookup; verification-gated admission; duplicate/order/lost-leaf rejection; hermetic default
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store, ipfs_kit_py/tests/proof_seal_store
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store
- Acceptance: Local tests require no daemon; optional transport corruption fails closed; storage never upgrades proof authority.
- Gap task: IPS-018, IPS-019, IPS-020, IPS-021, IPS-022
- Refinement: Store protocol unlocks local and optional transport in parallel; cache and forest consume only canonical datasets records.
- Conflict policy: Do not use pseudo-CIDs, Event-DAG Merkle helpers, merkle_clock, or mutable indexes as proof authority.

## IPS-G050 Seal WAL, current-root CAS, recovery, and concurrency

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G040
- Fib priority: 100
- Priority: P0
- Track: kit-durability
- Bundle: incremental-proof-sealer/kit-durability
- Parallel lane: storage
- Resource class: io-large
- Goal: Implement repository/branch-namespaced compare-and-swap, durable seal transitions, deterministic recovery, concurrent-writer rejection, and corruption/tamper conformance.
- Evidence: ips/seal-pointer-cas@1, ips/transition-wal@1, ips/recovery-matrix@1, ips/storage-adversarial@1
- Acceptance criteria: seven crash points; committed-only replay; idempotent restart; stale writer rejected; ambiguous prover outcome never guessed; corrupt tail preserves valid prefix
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/wal.py, ipfs_kit_py/ipfs_kit_py/proof_seal_store/recovery.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_wal.py ipfs_kit_py/tests/proof_seal_store/test_recovery.py
- Acceptance: A seal becomes current only through expected-parent CAS after verified persistence; restart yields an explicit resume/replay/verify/discard/repair/full-reproof disposition.
- Gap task: IPS-023, IPS-024, IPS-025, IPS-026, IPS-027
- Refinement: CAS, WAL, and recovery remain separate tasks before adversarial fan-in and migration.
- Conflict policy: Build on modern core/wal semantics; do not retrofit legacy WAL or silently absorb concurrent writes.

## IPS-G060 Proof-class admission, backend capabilities, keys, and provers

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G010, IPS-G020
- Fib priority: 100
- Priority: P0
- Track: accelerate-proof-trust
- Bundle: incremental-proof-sealer/accelerate-trust
- Parallel lane: proving
- Resource class: cpu-proof
- Goal: Classify and verify integrity, signed receipt, receipt aggregation, theorem certificate, direct execution, and seal evidence with allowlisted keys and operational backend capabilities.
- Evidence: ips/evidence-admission@1, ips/backend-capability-matrix@1, ips/key-registry@1, ips/prover-adapters@1
- Acceptance criteria: proof before cache admission; signatures verified; unknown backend rejected; recursion only after safe probe; no production key generation; witness secrecy
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/admission.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/backends.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_admission.py test/api/incremental_sealing/test_backends.py
- Acceptance: Each evidence result states exactly what is and is not proven; no receipt/hash/simulation is promoted to direct execution.
- Gap task: IPS-028, IPS-029, IPS-030, IPS-031
- Refinement: Admission, backend probing, key policy, and invocation are independently reviewed before orchestration.
- Conflict policy: No caller-selected key, arbitrary executable/circuit path, automatic setup, or mock proof success.

## IPS-G070 Incremental planning, scheduling, execution, aggregation, and cost

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G030, IPS-G040, IPS-G050, IPS-G060
- Fib priority: 100
- Priority: P0
- Track: accelerate-orchestration
- Bundle: incremental-proof-sealer/accelerate-orchestration
- Parallel lane: orchestration
- Resource class: cpu-large
- Goal: Plan full versus delta work, schedule changed units with bounded resources and cancellation, revalidate/admit proofs, aggregate safely, and measure expected/actual savings.
- Evidence: ips/incremental-plan@1, ips/proof-schedule@1, ips/process-fencing@1, ips/execution@1, ips/aggregation@1, ips/cost@1
- Acceptance criteria: all plan fields; priority order; process-tree termination; affected branches only; honest recursion label; fallback reasons; CPU/GPU/memory/time/size metrics
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/planner.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/executor.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_planner.py test/api/incremental_sealing/test_executor.py test/api/incremental_sealing/test_aggregation.py
- Acceptance: Exact candidates are freshly verified, invalidated/added units are proven, all non-pass states fence publication, and full fallback occurs whenever reuse cannot be justified.
- Gap task: IPS-032, IPS-033, IPS-034, IPS-035, IPS-036, IPS-037
- Refinement: Pure planning, scheduling, process fencing, proof lifecycle, aggregation, and telemetry are separate tasks with explicit joins.
- Conflict policy: Reuse modern scheduler/resource primitives; do not introduce a competing scheduler or simulate hardware/proof success.

## IPS-G080 Full checkpoints, delta seals, verification, and compaction

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G050, IPS-G070
- Fib priority: 100
- Priority: P0
- Track: seal-lifecycle
- Bundle: incremental-proof-sealer/seals
- Parallel lane: integration
- Resource class: cpu-large
- Goal: Implement full checkpoint, all fourteen delta invariants, atomic end-to-end publication, verification/explanation APIs, periodic policy, and chain compaction.
- Evidence: ips/full-seal@1, ips/delta-seal@1, ips/atomic-transition@1, ips/seal-verification@1, ips/compaction@1
- Acceptance criteria: exact parent/branch/revision binding; complete required set; added/removed/reused/replaced correctness; CAS publication; chain retention
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/sealer.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/verification.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_seals.py test/api/incremental_sealing/test_compaction.py
- Acceptance: Wrong parent/replay, missing invalidation, stale aggregate, and lost unaffected leaf fail; compaction verifies complete history and preserves retained evidence.
- Gap task: IPS-038, IPS-039, IPS-040, IPS-041, IPS-042
- Refinement: Full and delta construction land separately before atomic workflow, public verification/explanation, and compaction.
- Conflict policy: No pointer publication before all evidence and transition checks pass.

## IPS-G090 Public API, CLI, packaging, and migration

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G080
- Fib priority: 70
- Priority: P1
- Track: public-surface
- Bundle: incremental-proof-sealer/public
- Parallel lane: public
- Resource class: cpu-medium
- Goal: Expose the required lazy public APIs and narrowly scoped zk-seal CLI, and migrate existing ZK tests/receipts without changing their truth semantics.
- Evidence: ips/public-api@1, ips/cli@1, ips/import-hermeticity@1, ips/migration@1
- Acceptance criteria: all requested commands/APIs; JSON errors/statuses; no import-time network/process/state/key effects; old receipts explicitly adapted or rejected
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/__init__.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/cli.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_public_api.py test/api/incremental_sealing/test_cli.py
- Acceptance: CLI and API semantics match; absence of optional kit/datasets/backend is typed; package import is cold and side-effect free.
- Gap task: IPS-043, IPS-044
- Refinement: CLI/API and packaging/migration are separate review surfaces.
- Conflict policy: Do not add services, GUI, broad package refactors, or auto-install paths.

## IPS-G100 Deterministic fixtures and positive lifecycle conformance

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G030, IPS-G080
- Fib priority: 80
- Priority: P0
- Track: positive-conformance
- Bundle: incremental-proof-sealer/fixtures-positive
- Parallel lane: tests
- Resource class: cpu-medium
- Goal: Build tiny deterministic repository histories/graphs and prove all required positive invalidation, reuse, branch, merge, rollback, seal, checkpoint, and compaction behavior.
- Evidence: ips/fixture-corpus@1, ips/invalidation-positive@1, ips/seal-lifecycle-positive@1
- Acceptance criteria: every requested change fixture; unrelated reuse; deterministic repeated roots; explicit test add/delete; merge/rollback behavior
- Outputs: test/fixtures/incremental_proof_sealer, test/api/incremental_sealing/test_positive_matrix.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_positive_matrix.py
- Acceptance: Controlled localized changes avoid unrelated invalidation; all required additions and authorized removals appear in manifests and seals.
- Gap task: IPS-045, IPS-046, IPS-047
- Refinement: Generator lands before invalidation and full lifecycle matrices.
- Conflict policy: Fixture proof modes are labeled; simulated fixtures test plumbing/rejection, not production acceptance.

## IPS-G110 Tamper, trust, crash-recovery, and adversarial conformance

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G080, IPS-G100
- Fib priority: 100
- Priority: P0
- Track: negative-conformance
- Bundle: incremental-proof-sealer/adversarial
- Parallel lane: security-tests
- Resource class: cpu-large
- Goal: Reject every requested cache-key, cryptographic, signature, manifest, parent, status, forest, concurrent-writer, privacy, and crash-recovery attack.
- Evidence: ips/cache-tamper@1, ips/crypto-trust-negative@1, ips/crash-matrix@1, ips/e2e-adversarial@1
- Acceptance criteria: all normative negative cases; seven crash boundaries; zero stale/simulated/unknown acceptance; no lost unaffected leaf or stale CAS win
- Outputs: test/api/incremental_sealing/test_tamper_matrix.py, ipfs_kit_py/tests/proof_seal_store/test_crash_matrix.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_tamper_matrix.py test/api/incremental_sealing/test_adversarial_e2e.py ipfs_kit_py/tests/proof_seal_store/test_crash_matrix.py
- Acceptance: Every single-field and cryptographic mutation fails closed with a typed reason, and recovery never guesses external success.
- Gap task: IPS-048, IPS-049, IPS-050, IPS-051
- Refinement: Cache/forest, crypto/trust, recovery/concurrency, and joined e2e matrices remain independently diagnosable.
- Conflict policy: Never weaken an assertion to accommodate a backend absence; classify absence explicitly.

## IPS-G120 Forty-transition benchmark and performance evidence

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G080, IPS-G100, IPS-G110
- Fib priority: 60
- Priority: P1
- Track: benchmark
- Bundle: incremental-proof-sealer/benchmark
- Parallel lane: benchmark
- Resource class: cpu-large
- Goal: Execute or explicitly estimate full and incremental proof cost over the reviewed 40-transition history and report reuse, compute, latency, size, memory, storage, and fallback evidence honestly.
- Evidence: ips/benchmark-workload@1, ips/benchmark-results@1, ips/performance-analysis@1
- Acceptance criteria: 40 transitions; full/incremental per accepted state; all requested metrics; source of measurement; target versus actual separation; best/worst/fallback cases
- Outputs: benchmarks/agent_supervisor/incremental_proof_sealer.py, artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json
- Validation: python -m pytest -q test/benchmarks/test_incremental_proof_sealer_benchmark.py
- Acceptance: Results are deterministic for controlled inputs, exclude simulated work from production proving claims, and report unmet goals rather than manufacturing savings.
- Gap task: IPS-052, IPS-053, IPS-054
- Refinement: Workload, execution artifact, and performance interpretation are separate tasks.
- Conflict policy: Benchmark code cannot relax correctness or fallback policy.

## IPS-G130 Trust documentation, final report, and release fan-in

- Status: active
- Parent: IPS-G000
- Depends on: IPS-G090, IPS-G110, IPS-G120
- Fib priority: 100
- Priority: P0
- Track: release
- Bundle: incremental-proof-sealer/release
- Parallel lane: release
- Resource class: cpu-large
- Goal: Document exact trust/setup/privacy/claim semantics and migration, then run current-tree cross-repository release validation and publish the required evidence-backed report.
- Evidence: ips/trust-model@1, ips/migration-guide@1, ips/final-report@1, ips/final-validation@1
- Acceptance criteria: exact commits/systems/test classifications/modules/rules/results; direct versus signed versus integrity claims; remaining production gaps; clean nested commits and gitlinks
- Outputs: docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_MODEL.md, docs/architecture/INCREMENTAL_PROOF_SEALER_REPORT.md
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-all && python -m pytest -q test/api/incremental_sealing ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing ipfs_kit_py/tests/proof_seal_store
- Acceptance: All focused tests and tamper/recovery gates pass, benchmark/report evidence is current, and no final sentence overstates what the selected proof systems establish.
- Gap task: IPS-055, IPS-056
- Refinement: Documentation follows measured behavior; terminal fan-in may only repair demonstrated integration/regression gaps.
- Conflict policy: Do not claim repository correctness, proven pytest execution, or direct computation from receipt aggregation or integrity commitments.
