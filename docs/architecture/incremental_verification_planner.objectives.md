# Incremental Verification Planner Objective Heap (IVP)

Machine-ingestible goal hierarchy for
`INCREMENTAL_VERIFICATION_PLANNER_PLAN.md`. The executable projection is
`incremental_verification_planner.todo.md` with prefix `IVP-`.

## North star

Select and execute the minimum defensible verification work for a proposed
patch, reuse only exact production-admissible receipts, preserve uncertainty,
and provide a compact counterexample and provider-neutral next-repair route.

## Goal tree

```text
IVP-G000  Trustworthy incremental verification and repair routing
├── IVP-G010  Canonical contracts and exact receipt identity
├── IVP-G020  Narrow datasets, storage, and execution boundaries
├── IVP-G030  Durable exact verification receipt cache
├── IVP-G040  Reproducible check and prover adapters
├── IVP-G050  Semantic selection and incremental plan construction
├── IVP-G060  Counterexamples, bundles, summaries, and commitments
├── IVP-G070  Provider-neutral repair model routing
├── IVP-G080  Plan execution and acceptance orchestration
├── IVP-G090  Differential conformance and benchmark evidence
└── IVP-G100  Public surface, final report, and release gate
```

## IVP-G000 Trustworthy incremental verification and repair routing

- Status: active
- Parent:
- Depends on:
- Fib priority: 100
- Track: incremental-verification
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/root
- Parallel lane: release
- Resource class: cpu-medium
- Goal: Deliver one focused IncrementalVerificationPlanner subsystem that selects affected checks, reuses exact valid receipts, executes missing work reproducibly, minimizes failures, and emits a safe next-repair route.
- Evidence: ivp/release-report@1, ivp/conformance-matrix@1, ivp/benchmark@1
- Evidence requirements JSON: ["ivp/release-report@1","ivp/conformance-matrix@1","ivp/benchmark@1"]
- Evidence criteria: {"all_required_children_authoritatively_completed":true,"stale_accepted":0,"simulated_production_accepted":0,"fixture_false_negatives":0,"deterministic_commitments":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification, test/api/test_agent_supervisor_incremental_verification_conformance.py, benchmarks/agent_supervisor/incremental_verification.py, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md
- Interfaces: create_verification_plan, choose_model_route, build_verification_commitment
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_incremental_verification_conformance.py
- Acceptance: Every required receipt mutation fails closed; controlled selection has zero false negatives; unavailable, timeout, stale, and simulated results remain non-passes; provider selection remains separate; the benchmark and limitations are honest.
- Gap task: IVP-019
- Refinement: Children own disjoint contracts, adapters, cache, selection, routing, orchestration, and evidence before the terminal fan-in.
- Embedding query: incremental verification exact receipt cache semantic test selection reproducible checks counterexample model route commitment
- AST query: Locate proof receipt contracts, validation DAG selection, process-tree fencing, resource admission, semantic impact graphs, and cache authority reused by the focused planner.
- Conflict policy: Extend existing trust and runtime primitives; do not fork assurance lattices or build a generic agent platform.

## IVP-G010 Canonical contracts and exact receipt identity

- Status: active
- Parent: IVP-G000
- Depends on:
- Fib priority: 100
- Track: contracts
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/contracts
- Parallel lane: contracts
- Resource class: cpu-small
- Goal: Define bounded canonical statuses, receipt types, decisions, plans, bundles, summaries, commitments, and an exact VerificationReceiptKey binding every authority-relevant input.
- Evidence: ivp/contracts@1, ivp/cache-key-vectors@1
- Evidence requirements JSON: ["ivp/contracts@1","ivp/cache-key-vectors@1"]
- Evidence criteria: {"closed_statuses":12,"required_key_components":13,"canonical_roundtrip":true,"private_fields_rejected":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, test/api/test_agent_supervisor_verification_contracts.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, ipfs_accelerate_py/agent_supervisor/verification/__init__.py, test/api/test_agent_supervisor_verification_contracts.py
- Interfaces: VerificationReceiptKey, VerificationPlan, VerificationBundle, VerificationSummary, CacheReuseDecision, ModelRouteDecision, VerificationCommitment
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_contracts.py
- Acceptance: Canonical immutable round-trips; exact key mutation vectors; terminal statuses cannot be widened; timeout, unavailable, and simulated are never accepting; no secrets or witnesses serialize.
- Gap task: IVP-001
- Refinement: Reuse canonical identity helpers and explicitly bridge existing test/proof receipts without making a second trust root.
- Embedding query: canonical verification receipt types exact key terminal status bundle summary decision
- AST query: Find existing canonical contract mixins, content identity functions, TestPassReceipt, ProofReceipt, and assurance projection code.
- Conflict policy: Preserve existing receipt wire contracts; use typed adapters rather than rewriting historical schemas.

## IVP-G020 Narrow datasets, storage, and execution boundaries

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G010
- Fib priority: 90
- Track: adapters
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/boundaries
- Parallel lane: boundaries
- Resource class: cpu-medium
- Goal: Add lazy fail-closed adapters for datasets semantic inputs, ipfs-kit immutable storage/CAS, and explicit-argv bounded execution using existing resource/cancellation infrastructure.
- Evidence: ivp/datasets-adapter@1, ivp/store-protocol@1, ivp/process-runner@1
- Evidence requirements JSON: ["ivp/datasets-adapter@1","ivp/store-protocol@1","ivp/process-runner@1"]
- Evidence criteria: {"no_eager_optional_import":true,"no_auto_install":true,"shell_false":true,"process_tree_termination":true,"bounded_output":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/datasets_adapter.py, ipfs_accelerate_py/agent_supervisor/verification/receipt_store.py, ipfs_accelerate_py/agent_supervisor/verification/process_runner.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/datasets_adapter.py, ipfs_accelerate_py/agent_supervisor/verification/receipt_store.py, ipfs_accelerate_py/agent_supervisor/verification/process_runner.py
- Interfaces: DatasetsVerificationInputAdapter, VerificationReceiptStore, IpfsKitVerificationReceiptStore, HermeticVerificationReceiptStore, VerificationProcessRunner
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_datasets_adapter.py test/api/test_agent_supervisor_verification_receipt_store.py test/api/test_agent_supervisor_verification_process_runner.py
- Acceptance: Missing upstream schemas/capabilities are typed unavailable; immutable bytes are CID-verified; CAS prevents lost updates; commands are argv-only and cancellation kills descendants.
- Gap task: IVP-002, IVP-003, IVP-004
- Refinement: The four requested datasets classes are not present at planning revision, so structural adaptation is narrow and must fail closed until upstream canonical types land.
- Embedding query: ipfs datasets semantic impact adapter ipfs kit durable coordination store CAS bounded process runner
- AST query: Locate CodeEvidenceCorpusAdapter, DurableCoordinationStore, BoundedToolRunner, SubprocessProofProvider, ResourceScheduler, and cancellation tokens.
- Conflict policy: No copied datasets models, private storage internals, mock IPFS fallback, shell strings, or dependency installation.

## IVP-G030 Durable exact verification receipt cache

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G010, IVP-G020
- Fib priority: 100
- Track: cache
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/cache
- Parallel lane: storage
- Resource class: io-medium
- Goal: Store immutable receipts and maintain an exact-key verification index with CAS, replay, corruption detection, tombstones, GC metadata, and fail-closed production admission.
- Evidence: ivp/receipt-cache@1, ivp/concurrent-writer@1, ivp/replay-corruption@1
- Evidence requirements JSON: ["ivp/receipt-cache@1","ivp/concurrent-writer@1","ivp/replay-corruption@1"]
- Evidence criteria: {"exact_lookup":true,"lost_updates":0,"corruption_accepted":0,"stale_accepted":0,"simulated_production_accepted":0}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py, test/api/test_agent_supervisor_verification_receipt_cache.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py, test/api/test_agent_supervisor_verification_receipt_cache.py
- Interfaces: VerificationReceiptCache, lookup, admit, tombstone, replay, collect_gc_metadata
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_receipt_cache.py
- Acceptance: Exact key and embedded CID are revalidated on every hit; unrelated edits preserve old immutable history but a changed full-tree key rejects cross-tree reuse; every identity change invalidates; CAS conflicts merge/retry without overwriting peers.
- Gap task: IVP-008
- Refinement: Build on the stable store protocol after canonical key and store tasks complete.
- Embedding query: verification receipt cache exact key CAS replay corruption stale simulated tombstone garbage collection
- AST query: Find test_proof_cache and formal_verification_cache trust checks and reuse only their compatible admission semantics.
- Conflict policy: Cache presence is never authority; do not weaken or replace existing proof caches.

## IVP-G040 Reproducible check and prover adapters

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G010, IVP-G020
- Fib priority: 90
- Track: execution-adapters
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/checks
- Parallel lane: adapters
- Resource class: cpu-medium
- Goal: Provide narrow pytest, mypy, Z3, and existing proof-assistant adapters over the shared admitted process runner.
- Evidence: ivp/pytest-adapter@1, ivp/mypy-adapter@1, ivp/prover-adapter@1
- Evidence requirements JSON: ["ivp/pytest-adapter@1","ivp/mypy-adapter@1","ivp/prover-adapter@1"]
- Evidence criteria: {"explicit_argv":true,"typed_unavailable":true,"timeout_is_timeout":true,"cancel_fences_publication":true,"no_new_prover":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/adapters
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/adapters, test/api/test_agent_supervisor_verification_adapters.py
- Interfaces: PytestVerificationAdapter, MypyVerificationAdapter, Z3VerificationAdapter, ExistingProofAssistantAdapter
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_pytest_verification_adapter.py test/api/test_agent_supervisor_mypy_verification_adapter.py test/api/test_agent_supervisor_prover_verification_adapters.py
- Acceptance: All adapters persist bounded artifacts and canonical receipts; missing tools are unavailable; Z3 unknown stays unknown; only registry-admitted operational kernels may prove.
- Gap task: IVP-005, IVP-006, IVP-007
- Refinement: Adapter tasks are independent after the shared process runner exists.
- Embedding query: pytest mypy z3 lean adapter deterministic environment timeout cancellation receipt
- AST query: Locate BoundedToolRunner and current Z3/Lean kernel adapters; identify unsafe raw subprocess paths that must not be reused.
- Conflict policy: No shell interpolation, auto-install, mock execution, or new theorem prover.

## IVP-G050 Semantic selection and incremental plan construction

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G010, IVP-G020, IVP-G030
- Fib priority: 100
- Track: planning
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/planner
- Parallel lane: selection
- Resource class: cpu-medium
- Goal: Select affected tests/checks/proofs from semantic edges and build a deterministic VerificationPlan with cache decisions, fallback, resources, deadlines, human-review flags, and acceptance criteria.
- Evidence: ivp/test-selection@1, ivp/verification-plan@1
- Evidence requirements JSON: ["ivp/test-selection@1","ivp/verification-plan@1"]
- Evidence criteria: {"semantic_edges_used":true,"uncertainty_broadens":true,"exact_cache_decisions":true,"bounded_resources":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/selection.py, ipfs_accelerate_py/agent_supervisor/verification/planner.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/selection.py, ipfs_accelerate_py/agent_supervisor/verification/planner.py, test/api/test_agent_supervisor_incremental_verification_planner.py
- Interfaces: select_affected_verification, create_verification_plan, IncrementalVerificationPlanner
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_selection.py test/api/test_agent_supervisor_incremental_verification_planner.py
- Acceptance: Relevant changes invalidate and select; unrelated changes avoid semantic over-selection but changed full-tree identity still rejects old-receipt admission; environment/lock/tool changes invalidate; opaque or uncovered impact requires broader/full suite; non-reproducibility requires review.
- Gap task: IVP-009, IVP-010
- Refinement: Selection remains pure/deterministic; plan creation joins it with exact cache decisions and policy.
- Embedding query: semantic dependency edges affected tests invalidation plan verification plan full suite fallback acceptance criteria
- AST query: Locate CodeImpactIndex, impact_query, validation DAG affected_paths, proof obligation dependencies, and semantic frontier fields.
- Conflict policy: Never infer missing semantic edges as exact or suppress uncertainty to improve selection size.

## IVP-G060 Counterexamples, bundles, summaries, and commitments

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G010, IVP-G020, IVP-G040
- Fib priority: 80
- Track: diagnostics
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/diagnostics
- Parallel lane: diagnostics
- Resource class: cpu-small
- Goal: Minimize failures, assemble accepted/unresolved receipts, produce a compact ContextPack summary, and commit the admitted set with a deterministic Merkle root.
- Evidence: ivp/counterexample@1, ivp/verification-summary@1, ivp/verification-commitment@1
- Evidence requirements JSON: ["ivp/counterexample@1","ivp/verification-summary@1","ivp/verification-commitment@1"]
- Evidence criteria: {"bounded_counterexample":true,"raw_logs_excluded_by_default":true,"merkle_deterministic":true,"zk_non_claims_explicit":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/counterexamples.py, ipfs_accelerate_py/agent_supervisor/verification/bundle.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/counterexamples.py, ipfs_accelerate_py/agent_supervisor/verification/bundle.py
- Interfaces: minimize_counterexample, build_verification_summary, build_verification_commitment
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_counterexamples.py test/api/test_agent_supervisor_verification_bundle.py
- Acceptance: Failed selected tests yield lease-rerun compact reproductions; commitment changes with any required membership/content change while input permutation is invariant; aggregation cannot upgrade a leaf; non-ZK and signature trust caveats are serialized.
- Gap task: IVP-011, IVP-013
- Refinement: Counterexample and bundle tasks may proceed independently against canonical contracts.
- Embedding query: minimized traceback assertion input source span verification summary context pack merkle receipt commitment
- AST query: Locate formal_counterexamples, proof_context compact failures, artifact store, canonical hashing, and Merkle helpers.
- Conflict policy: Do not send full logs by default, hide unresolved obligations, or claim cryptographic execution proof.

## IVP-G070 Provider-neutral repair model routing

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G010
- Fib priority: 80
- Track: routing
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/routing
- Parallel lane: routing
- Resource class: cpu-small
- Goal: Choose deterministic, small, medium, frontier, or human-review capability class from verification/context facts without selecting a vendor.
- Evidence: ivp/model-route@1
- Evidence requirements JSON: ["ivp/model-route@1"]
- Evidence criteria: {"provider_neutral":true,"human_review_precedence":true,"small_localized":true,"frontier_broad_opaque":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/model_route.py, test/api/test_agent_supervisor_verification_model_route.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/model_route.py, test/api/test_agent_supervisor_verification_model_route.py
- Interfaces: ModelRoutePlanner, choose_model_route
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_model_route.py
- Acceptance: Exact localized work selects small when not mechanical; broad/opaque or failed-smaller work selects frontier; provider-neutral availability cannot downgrade the required tier; pending full-suite, unavailable required tier, or unresolved high-risk authority/reproducibility selects human review; no provider identifier appears.
- Gap task: IVP-012
- Refinement: Decision precedence is table-driven, bounded, deterministic, and independently testable.
- Embedding query: deterministic small medium frontier human review model route context tokens opaque dependencies risk
- AST query: Locate existing provider routers only to keep provider resolution downstream and separate.
- Conflict policy: Route capability only; do not invoke models or hardcode a vendor.

## IVP-G080 Plan execution and acceptance orchestration

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G030, IVP-G040, IVP-G050, IVP-G060, IVP-G070
- Fib priority: 90
- Track: orchestration
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/execution
- Parallel lane: orchestration
- Resource class: cpu-large
- Goal: Execute VerificationPlan DAGs under resource leases, reuse admitted hits, persist receipts/artifacts, minimize failures, bundle results, and evaluate acceptance without status upgrades.
- Evidence: ivp/execution-bundle@1
- Evidence requirements JSON: ["ivp/execution-bundle@1"]
- Evidence criteria: {"resource_admission":true,"cancellation_fenced":true,"receipt_persistence":true,"acceptance_recomputed":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/executor.py, test/api/test_agent_supervisor_verification_executor.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/executor.py, test/api/test_agent_supervisor_verification_executor.py
- Interfaces: VerificationExecutor, execute_verification_plan
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_executor.py
- Acceptance: Reused and executed receipts are distinguishable; dependencies and deadlines hold; timeout/unavailable/cancelled/simulated cannot pass; failures carry minimized counterexamples; acceptance exactly matches current required leaves.
- Gap task: IVP-014
- Refinement: Single integration task follows all collaborating surfaces to avoid parallel cross-file rewrites.
- Embedding query: execute verification DAG resource admission receipt persistence acceptance bundle cancellation
- AST query: Locate proof scheduler/resource scheduler/validation scheduler patterns and authoritative completion checks.
- Conflict policy: Execution cannot alter plan scope, install dependencies, widen network policy, or self-assert acceptance.

## IVP-G090 Differential conformance and benchmark evidence

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G080
- Fib priority: 100
- Track: evaluation
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/evaluation
- Parallel lane: evaluation
- Resource class: cpu-large
- Goal: Compare selected tests with full-suite oracle on controlled semantic fixtures, prove the required mutation/concurrency/cancellation/routing matrix, and report performance honestly.
- Evidence: ivp/test-selection-evaluation@1, ivp/conformance-matrix@1, ivp/benchmark@1
- Evidence requirements JSON: ["ivp/test-selection-evaluation@1","ivp/conformance-matrix@1","ivp/benchmark@1"]
- Evidence criteria: {"controlled_false_negatives":0,"required_cases":18,"metrics_complete":true,"unavailable_explicit":true,"missing_corpus_not_measured":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/evaluation.py, test/api/test_agent_supervisor_incremental_verification_conformance.py, benchmarks/agent_supervisor/incremental_verification.py, artifacts/agent_supervisor/incremental_verification/benchmark.json
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/evaluation.py, test/fixtures/incremental_verification, test/api/test_agent_supervisor_incremental_verification_conformance.py, benchmarks/agent_supervisor/incremental_verification.py, artifacts/agent_supervisor/incremental_verification/benchmark.json
- Interfaces: compare_selected_with_full_suite, run_incremental_verification_benchmark
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_selection_evaluation.py test/api/test_agent_supervisor_incremental_verification_conformance.py test/benchmarks/test_incremental_verification_planner_benchmark.py
- Acceptance: Ground-truth controlled false negatives and false positives plus full-suite outcome discrepancies are measured separately; the hard conformance gate requires zero controlled false negatives and all 18 behaviors, while benchmark/report evidence still lands when red; timing is sampled rather than deterministic; missing upstream semantic-capsule fixtures or real provers are typed not_measured/unavailable, never fabricated.
- Gap task: IVP-015, IVP-016, IVP-017
- Refinement: Differential methodology precedes two independent children: the consolidated hard adversarial suite and an always-reporting benchmark evidence path.
- Embedding query: selected tests full suite oracle false negative cache hit benchmark routing token savings counterexample context
- AST query: Locate code-evidence tiny fixtures and existing semantic/proof benchmark conventions.
- Conflict policy: Performance cannot weaken correctness; skips and unavailable capabilities are not passes or measured wins.

## IVP-G100 Public surface, final report, and release gate

- Status: active
- Parent: IVP-G000
- Depends on: IVP-G090
- Fib priority: 100
- Track: release
- Priority: P0
- Bundle: agent-supervisor/incremental-verification/release
- Parallel lane: release
- Resource class: cpu-large
- Goal: Document the subsystem, freeze public exports, run focused/regression suites and static checks, and publish the requested final report including the exact ZK-aggregation next step.
- Evidence: ivp/public-api@1, ivp/final-validation@1, ivp/release-report@1
- Evidence requirements JSON: ["ivp/public-api@1","ivp/final-validation@1","ivp/release-report@1"]
- Evidence criteria: {"public_apis_importable":true,"focused_suite_green":true,"regression_suite_green":true,"limitations_listed":true,"zk_next_step_exact":true}
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/README.md, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md, ipfs_accelerate_py/agent_supervisor/verification/__init__.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/README.md, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md, test/api/test_agent_supervisor_incremental_verification_report.py, ipfs_accelerate_py/agent_supervisor/verification/__init__.py
- Interfaces: package public exports, final report
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_contracts.py test/api/test_agent_supervisor_verification_receipt_cache.py test/api/test_agent_supervisor_incremental_verification_planner.py test/api/test_agent_supervisor_verification_executor.py test/api/test_agent_supervisor_incremental_verification_conformance.py
- Acceptance: Clean target branch; required APIs and types exported; full report contains modules/adapters/schemas/key/invalidation/tests/proofs/hits/routes/counterexamples/commitment/limitations; ZK work remains external and subsequent to trusted ordinary receipts.
- Gap task: IVP-018, IVP-019
- Refinement: Documentation may draft after benchmark; terminal fan-in owns exports and integrated corrections.
- Embedding query: incremental verification public API final report receipt cache model route ZK aggregator next step
- AST query: Verify all new imports, package exports, focused tests, and regressions against the final tree.
- Conflict policy: Do not mark targets met without current evidence or expand scope during release cleanup.
