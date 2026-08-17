# Semantic Compression Governor objective graph

Durable goal heap for board namespace `semantic-compression-governor-v1`.
Task completion is implementation progress; only evidence-backed goal
reconciliation may mark a goal verified complete.

## SCG-G000 Deliver a safe empirical semantic-compression governance loop

- Status: active
- Parent: none
- Parent goal IDs JSON: []
- Depends on:
- Dependencies JSON: []
- Fib priority: 1000
- Track: program
- Priority: P0
- Bundle: semantic-governor/program
- Parallel lane: integration
- Resource class: cpu-large
- Goal: Implement the closed audit, shadow, omission, expansion, calibration, held-out evaluation, authorized promotion, and evidence-reporting loop without allowing compression economics to weaken correctness, privacy, or assurance.
- Producing tasks: SCG-001 through SCG-048
- Evidence: scg/current-tree-conformance@1, scg/controlled-benchmark@1, scg/promotion-qualification@1, scg/final-report@1
- Evidence requirements JSON: ["scg/current-tree-conformance@1","scg/controlled-benchmark@1","scg/promotion-qualification@1","scg/final-report@1"]
- Evidence criteria: {"critical_intentional_omissions_accepted":0,"stale_artifacts_accepted":0,"heuristic_treated_as_exact":0,"held_out_required":true,"authorized_cas_required":true,"results_honest":true}
- Outputs: datasets semantic-governor analysis, accelerate semantic-governor runtime, kit governor storage, benchmark evidence, trust documentation, final report
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor, ipfs_accelerate_py/agent_supervisor/semantic_governor, ipfs_kit_py/ipfs_kit_py/semantic_governor_store, test/api/semantic_governor, test/fixtures/semantic_governor, benchmarks/agent_supervisor/semantic_compression_governor.py, artifacts/agent_supervisor/semantic_compression_governor, docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_REPORT.md
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor","ipfs_accelerate_py/agent_supervisor/semantic_governor","ipfs_kit_py/ipfs_kit_py/semantic_governor_store","test/api/semantic_governor","test/fixtures/semantic_governor","benchmarks/agent_supervisor/semantic_compression_governor.py","artifacts/agent_supervisor/semantic_compression_governor","docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_REPORT.md"]
- Interfaces: SemanticCompressionGovernor@1, semantic-governor CLI
- Validation: python3 -m pytest -q test/api/semantic_governor
- Acceptance: Every required API and safety invariant is current-tree tested; actual benchmark results and seal scope are reported without upgrading simulations or commitments into proof.
- Gap task: SCG-048
- Refinement: Child goals own exact authorities, canonical contracts, analysis, durability, execution, policy, interfaces, benchmarks, and final qualification.
- Embedding query: semantic compression context sufficiency shadow execution omission expansion calibration safe policy promotion
- AST query: Locate semantic-state harness, verification planner/cache/routes/counterexamples, datasets semantic state, kit durable roots, and incremental sealing.
- Conflict policy: Extend completed authorities; never create a second semantic index, capsule compiler, proof cache, CID implementation, receipt envelope, ZK system, provider, GUI, or MCP++ profile.

## SCG-G010 Freeze exact authorities and executable baselines

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 144
- Track: authority
- Priority: P0
- Bundle: semantic-governor/authority
- Parallel lane: inventory
- Resource class: cpu-medium
- Goal: Record exact repository commits, public interfaces, schemas, status vocabularies, context/routing rules, benchmark evidence, metrics, and known failures before implementation.
- Producing tasks: SCG-001, SCG-002, SCG-003, SCG-004, SCG-005
- Evidence: scg/accelerate-inventory@1, scg/datasets-inventory@1, scg/kit-inventory@1, scg/mcplusplus-boundary@1, scg/authority-matrix@1
- Evidence requirements JSON: ["scg/accelerate-inventory@1","scg/datasets-inventory@1","scg/kit-inventory@1","scg/mcplusplus-boundary@1","scg/authority-matrix@1"]
- Evidence criteria: {"exact_commits":4,"focused_baselines":4,"unknown_interfaces_fail_closed":true,"incremental_sealer_status_explicit":true}
- Outputs: docs/architecture/semantic_compression_governor_inventory
- Predicted files: docs/architecture/semantic_compression_governor_inventory
- Predicted files JSON: ["docs/architecture/semantic_compression_governor_inventory"]
- Interfaces: SemanticGovernorAuthorityMatrix@1
- Validation: python3 scripts/validate_semantic_compression_governor_board.py --check-all
- Acceptance: Inventories are generated from clean pinned trees; existing RED/non-authoritative evidence and unavailable proof-sealer capability remain visible.
- Gap task: SCG-005
- Refinement: Four repository inventories run in parallel and join into one reviewed consumption and non-reimplementation matrix.
- Embedding query: exact commits interfaces schemas statuses context routing benchmarks metrics failures
- AST query: Enumerate public exports and tests for semantic index/state, context harness, verification, durable roots, and MCP++ profiles.
- Conflict policy: Inventory tasks do not change production code or infer a missing interface from documentation alone.

## SCG-G020 Define one closed canonical artifact vocabulary

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on: SCG-G010
- Dependencies JSON: ["SCG-G010"]
- Fib priority: 233
- Track: contracts
- Priority: P0
- Bundle: semantic-governor/contracts
- Parallel lane: contracts
- Resource class: cpu-small
- Goal: Define all required versioned artifacts, shared headers, enums, assumptions, provenance, identities, execution records, calibration profiles, policies, and storage protocols without duplicating canonical identity or receipt envelopes.
- Producing tasks: SCG-006, SCG-007, SCG-008, SCG-009, SCG-010
- Evidence: scg/artifact-base@1, scg/audit-contracts@1, scg/execution-contracts@1, scg/policy-contracts@1, scg/storage-contracts@1
- Evidence requirements JSON: ["scg/artifact-base@1","scg/audit-contracts@1","scg/execution-contracts@1","scg/policy-contracts@1","scg/storage-contracts@1"]
- Evidence criteria: {"required_models":23,"closed_sufficiency_states":9,"closed_outcomes":10,"unknown_fields_rejected":true,"deterministic_identity":true}
- Outputs: canonical datasets models and schemas, accelerate execution contracts, kit storage protocols
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/schemas, ipfs_accelerate_py/agent_supervisor/semantic_governor/contracts.py, ipfs_kit_py/ipfs_kit_py/semantic_governor_store/contracts.py
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/contracts.py","ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor/schemas","ipfs_accelerate_py/agent_supervisor/semantic_governor/contracts.py","ipfs_kit_py/ipfs_kit_py/semantic_governor_store/contracts.py"]
- Interfaces: SemanticGovernorArtifacts@1, SemanticGovernorExecution@1, SemanticGovernorStore@1
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor ipfs_kit_py/tests/semantic_governor_store test/api/semantic_governor/test_contracts.py
- Acceptance: Every artifact rederives its canonical CID and binds state/context/verification identities; nonfinite, private, executable, versionless, forged, or overlarge inputs fail closed.
- Gap task: SCG-006, SCG-007, SCG-008, SCG-009, SCG-010
- Refinement: Neutral semantic artifacts live in datasets; accelerate owns only execution projections; kit owns only storage state and receipts.
- Embedding query: closed versioned governor models canonical identity provenance assumptions terminal status
- AST query: Find datasets content identity, verification contracts, semantic-state wire records, and kit root CAS protocols.
- Conflict policy: No generic envelope fork, pseudo-CID, model reasoning as evidence, or executable rule payload.

## SCG-G030 Audit sufficiency, omissions, expansion, and calibration

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on: SCG-G020
- Dependencies JSON: ["SCG-G020"]
- Fib priority: 377
- Track: semantic-analysis
- Priority: P0
- Bundle: semantic-governor/analysis
- Parallel lane: datasets
- Resource class: cpu-medium
- Goal: Build complete exclusion-aware coverage manifests, conservative pre-execution sufficiency decisions, prompt-injection evidence, omission attribution, bounded expansion plans, calibration, and declarative rule proposals over canonical semantic-state views.
- Producing tasks: SCG-011, SCG-012, SCG-013, SCG-014, SCG-015, SCG-016, SCG-017, SCG-018
- Evidence: scg/coverage@1, scg/sufficiency@1, scg/omission@1, scg/expansion-plan@1, scg/calibration@1, scg/rule-dsl@1
- Evidence requirements JSON: ["scg/coverage@1","scg/sufficiency@1","scg/omission@1","scg/expansion-plan@1","scg/calibration@1","scg/rule-dsl@1"]
- Evidence criteria: {"explicit_exclusion_reason":true,"opaque_critical_expands":true,"stale_requires_raw_or_regeneration":true,"bounded_expansion":true,"empirical_exactness_upgrade":false}
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor, ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor","ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor"]
- Interfaces: build_context_coverage_manifest, evaluate_context_sufficiency, diagnose_omission, plan_context_expansion, update_calibration, propose_rule_change
- Validation: python3 -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_governor
- Acceptance: Weak tests cannot independently imply sufficiency; omission and reasoning failure remain distinguishable; every expansion and rule proposal is bounded and explainable.
- Gap task: SCG-018
- Refinement: Independent pure modules converge at the public datasets export and conformance task.
- Embedding query: coverage manifest exclusion assumption sufficiency omission graph counterexample expansion calibration rule DSL
- AST query: Traverse semantic-state facts, edges, capsules, freshness, invalidation obligations, source evidence, and selection policy.
- Conflict policy: Consume verified semantic views; never rescan, invent graph edges, raise confidence, or execute proposed rules.

## SCG-G040 Persist immutable audits, histories, and policy CAS

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on: SCG-G020
- Dependencies JSON: ["SCG-G020"]
- Fib priority: 233
- Track: durability
- Priority: P0
- Bundle: semantic-governor/storage
- Parallel lane: kit
- Resource class: io-medium
- Goal: Persist immutable audit, calibration, benchmark, policy, evaluation, and promotion artifacts with verified CIDs, generation CAS, operation idempotency, recovery, and concurrency safety.
- Producing tasks: SCG-019, SCG-020, SCG-021, SCG-022
- Evidence: scg/audit-store@1, scg/history-store@1, scg/policy-cas@1, scg/store-recovery@1
- Evidence requirements JSON: ["scg/audit-store@1","scg/history-store@1","scg/policy-cas@1","scg/store-recovery@1"]
- Evidence criteria: {"immutable_bytes_verified":true,"lost_updates":0,"stale_candidate_overwrites":0,"interrupted_recovery":true,"raw_private_public_reports":0}
- Outputs: ipfs_kit_py/ipfs_kit_py/semantic_governor_store
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_governor_store, ipfs_kit_py/tests/semantic_governor_store
- Predicted files JSON: ["ipfs_kit_py/ipfs_kit_py/semantic_governor_store","ipfs_kit_py/tests/semantic_governor_store"]
- Interfaces: SemanticGovernorStore, AuditHistoryStore, CompressionPolicyRepository, PromotionStateRepository
- Validation: python3 -m pytest -q ipfs_kit_py/tests/semantic_governor_store
- Acceptance: The implementation is a thin typed layer over DurableCoordinationStore and root CAS; corruption, ABA, stale generation, and concurrent writers fail closed without losing immutable history.
- Gap task: SCG-022
- Refinement: Immutable blocks, histories, policy pointers, and recovery are split by file and joined by adversarial storage tests.
- Embedding query: immutable audit calibration benchmark policy history compare swap recovery concurrency receipt
- AST query: Locate DurableCoordinationStore, DurableStateRootAdapter, current roots, root transitions, recovery, and canonical validators.
- Conflict policy: No second CAS, WAL, local object database, network daemon, or raw private source in public artifacts.

## SCG-G050 Execute bounded shadow audits and repair loops

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on: SCG-G030, SCG-G040
- Dependencies JSON: ["SCG-G030","SCG-G040"]
- Fib priority: 610
- Track: execution
- Priority: P0
- Bundle: semantic-governor/runtime
- Parallel lane: accelerate
- Resource class: cpu-large
- Goal: Select high-information audits, enforce disclosure and resource policy, execute paired isolated contexts, compare semantic outcomes, integrate verification and counterexamples, expand context before model escalation, and measure route/cost behavior.
- Producing tasks: SCG-023, SCG-024, SCG-025, SCG-026, SCG-027, SCG-028, SCG-029, SCG-030, SCG-031, SCG-032
- Evidence: scg/runtime-adapters@1, scg/privacy-gate@1, scg/shadow-plan@1, scg/shadow-run@1, scg/differential@1, scg/expansion-loop@1, scg/route-calibration@1, scg/active-scheduler@1
- Evidence requirements JSON: ["scg/runtime-adapters@1","scg/privacy-gate@1","scg/shadow-plan@1","scg/shadow-run@1","scg/differential@1","scg/expansion-loop@1","scg/route-calibration@1","scg/active-scheduler@1"]
- Evidence criteria: {"isolated_worktrees":true,"external_private_expansion_without_authorization":0,"bounded_retries":true,"context_before_model_escalation":true,"text_difference_is_failure":false}
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_governor runtime modules
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/adapters.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/privacy.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/shadow.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/differential.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/expansion_loop.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/routes.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/scheduler.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/semantic_governor/adapters.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/privacy.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/shadow.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/differential.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/expansion_loop.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/routes.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/scheduler.py"]
- Interfaces: create_shadow_plan, compare_shadow_results, execute_expansion_loop, ActiveAuditScheduler
- Validation: python3 -m pytest -q test/api/semantic_governor/test_shadow.py test/api/semantic_governor/test_expansion_loop.py test/api/semantic_governor/test_scheduler.py
- Acceptance: Expanded results remain evaluation candidates; all model calls are provider-policy admitted, redacted, budgeted, cancellable, and costed; verification failures remain visible.
- Gap task: SCG-032
- Refinement: Adapters and privacy precede planning; planning precedes execution; verification and comparison precede the bounded loop; calibration precedes active scheduling.
- Embedding query: shadow sample resource admission privacy isolated worktree differential patch verification counterexample expansion route cost
- AST query: Locate SemanticCompressionHarness, ProviderExecutionGateway, ResourceScheduler, worktree lifecycle, VerificationExecutor, CounterexampleMinimizer, and ModelRoutePlanner.
- Conflict policy: Do not use rollout shadow mode as paired semantic evaluation, accept an expanded patch automatically, or bypass verification.

## SCG-G060 Evaluate and promote policy safely

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on: SCG-G050
- Dependencies JSON: ["SCG-G050"]
- Fib priority: 377
- Track: policy
- Priority: P0
- Bundle: semantic-governor/policy
- Parallel lane: policy
- Resource class: security-review
- Goal: Evaluate declarative candidates on disjoint held-out evidence, enforce safety/non-regression thresholds, require an incremental seal or separately authorized release qualification before promotion, require promotion authorization, publish with expected-version CAS, support rollback, and bind qualification artifacts to the released sealer when available.
- Producing tasks: SCG-033, SCG-034, SCG-035
- Evidence: scg/held-out-evaluation@1, scg/authorized-promotion@1, scg/seal-binding@1
- Evidence requirements JSON: ["scg/held-out-evaluation@1","scg/authorized-promotion@1","scg/seal-binding@1"]
- Evidence criteria: {"self_promotion":false,"held_out_disjoint":true,"high_risk_threshold_reduction_without_authorization":0,"expected_version_cas":true,"zk_nonclaims":true}
- Outputs: policy evaluator, promotion workflow, seal adapter
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/policy_evaluation.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/promotion.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/sealing.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/semantic_governor/policy_evaluation.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/promotion.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/sealing.py"]
- Interfaces: evaluate_rule_candidate, promote_compression_policy, SemanticGovernorSealAdapter
- Validation: python3 -m pytest -q test/api/semantic_governor/test_policy_evaluation.py test/api/semantic_governor/test_promotion.py test/api/semantic_governor/test_sealing.py
- Acceptance: A stale or self-authorizing candidate cannot promote; missing release sealer is typed unavailable and requires a separately authorized VerificationBundle-backed release qualification; a non-ZK commitment is never represented as a semantic or execution proof.
- Gap task: SCG-035
- Refinement: Evaluation produces no mutation; qualification precedes promotion; separate authorization plus current evaluation and qualification enables CAS; post-decision sealing binds but never upgrades evidence semantics.
- Embedding query: held out policy evaluation safety regression authorization compare swap rollback incremental seal
- AST query: Locate verification commitments, proof-sealer public surface at release time, policy root CAS, and authorization contracts.
- Conflict policy: No automatic assurance reduction, executable model rules, trusted-key changes, or promotion from calibration cases alone.

## SCG-G070 Publish narrow orchestration, CLI, metrics, and reports

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on: SCG-G050, SCG-G060
- Dependencies JSON: ["SCG-G050","SCG-G060"]
- Fib priority: 144
- Track: interfaces
- Priority: P1
- Bundle: semantic-governor/interfaces
- Parallel lane: interfaces
- Resource class: cpu-medium
- Goal: Compose the ten required APIs, ten closed CLI commands, complete metrics, and machine-readable reporting without a GUI or public server.
- Producing tasks: SCG-036, SCG-037, SCG-038, SCG-039
- Evidence: scg/public-api@1, scg/cli@1, scg/metrics@1, scg/dashboard-data@1
- Evidence requirements JSON: ["scg/public-api@1","scg/cli@1","scg/metrics@1","scg/dashboard-data@1"]
- Evidence criteria: {"required_apis":10,"required_commands":10,"default_json":true,"public_server":false,"arbitrary_paths_exposed":false}
- Outputs: public package exports, semantic-governor console entry, metrics and reports
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_governor/__init__.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/cli.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/metrics.py, ipfs_accelerate_py/agent_supervisor/semantic_governor/report.py, setup.py, pyproject.toml
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/semantic_governor/__init__.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/cli.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/metrics.py","ipfs_accelerate_py/agent_supervisor/semantic_governor/report.py","setup.py","pyproject.toml"]
- Interfaces: SemanticCompressionGovernor, semantic-governor
- Validation: python3 -m pytest -q test/api/semantic_governor/test_public_api.py test/api/semantic_governor/test_cli.py test/api/semantic_governor/test_metrics.py
- Acceptance: Imports perform no I/O; outputs are bounded, deterministic, privacy-filtered, and distinguish simulated from live evidence.
- Gap task: SCG-039
- Refinement: Orchestration stabilizes before the two CLI surfaces and reporting proceed in parallel.
- Embedding query: semantic governor public API CLI audit shadow diagnose expand calibrate policy report dashboard metrics
- AST query: Mirror semantic-state CLI conventions and agent-supervisor lazy export patterns.
- Conflict policy: No default listener, GUI, arbitrary path input, new provider, or hidden side effect on import.

## SCG-G080 Build adversarial held-out benchmark evidence

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on: SCG-G030, SCG-G050, SCG-G070
- Dependencies JSON: ["SCG-G030","SCG-G050","SCG-G070"]
- Fib priority: 610
- Track: benchmark
- Priority: P0
- Bundle: semantic-governor/benchmark
- Parallel lane: evaluation
- Resource class: cpu-large
- Goal: Build deterministic calibration/development/held-out fixtures and prove static, dynamic, security, privacy, model-capability, recovery, and concurrency cases before producing actual measured results.
- Producing tasks: SCG-040, SCG-041, SCG-042, SCG-043, SCG-044, SCG-045
- Evidence: scg/partitioned-corpus@1, scg/adversarial-omissions@1, scg/capability-differential@1, scg/e2e@1, scg/benchmark-results@1
- Evidence requirements JSON: ["scg/partitioned-corpus@1","scg/adversarial-omissions@1","scg/capability-differential@1","scg/e2e@1","scg/benchmark-results@1"]
- Evidence criteria: {"partitions_disjoint":true,"required_adversarial_cases":18,"intentional_critical_accepted":0,"simulated_live_claims":0,"actual_results_persisted":true}
- Outputs: controlled fixture repository, adversarial tests, benchmark runner and artifacts
- Predicted files: test/fixtures/semantic_governor, test/api/semantic_governor/test_adversarial_static.py, test/api/semantic_governor/test_adversarial_dynamic.py, test/api/semantic_governor/test_resilience_privacy.py, test/api/semantic_governor/test_end_to_end.py, benchmarks/agent_supervisor/semantic_compression_governor.py, artifacts/agent_supervisor/semantic_compression_governor
- Predicted files JSON: ["test/fixtures/semantic_governor","test/api/semantic_governor/test_adversarial_static.py","test/api/semantic_governor/test_adversarial_dynamic.py","test/api/semantic_governor/test_resilience_privacy.py","test/api/semantic_governor/test_end_to_end.py","benchmarks/agent_supervisor/semantic_compression_governor.py","artifacts/agent_supervisor/semantic_compression_governor"]
- Interfaces: SemanticGovernorBenchmark@1
- Validation: python3 -m pytest -q test/api/semantic_governor test/benchmarks/test_semantic_compression_governor_benchmark.py
- Acceptance: Held-out tasks are never used to generate their candidate; all specified adversarial omissions and model-insufficiency controls are classified honestly; actual metrics include unavailable/inconclusive cases.
- Gap task: SCG-045
- Refinement: Corpus generation precedes three disjoint adversarial suites; API/CLI fan-in precedes e2e and measured benchmark execution.
- Embedding query: held out adversarial omission fixture dynamic import prompt injection model insufficiency benchmark
- AST query: Reuse semantic-state controlled fixture and incremental-verification selected/full evaluation patterns.
- Conflict policy: No full repository disclosure, fabricated model output, benchmark-to-policy leakage, or result target hard-coding.

## SCG-G090 Qualify trust, rollback, sealing, and final claims

- Status: active
- Parent: SCG-G000
- Parent goal IDs JSON: ["SCG-G000"]
- Depends on: SCG-G060, SCG-G070, SCG-G080
- Dependencies JSON: ["SCG-G060","SCG-G070","SCG-G080"]
- Fib priority: 987
- Track: release
- Priority: P0
- Bundle: semantic-governor/release
- Parallel lane: release
- Resource class: security-review
- Goal: Publish trust/privacy/promotion operations, bind the released incremental sealer, run current-tree release and rollback qualification, and produce the required evidence-rich final report and promotion recommendation.
- Producing tasks: SCG-046, SCG-047, SCG-048
- Evidence: scg/trust-docs@1, scg/incremental-seal-qualification@1, scg/release@1, scg/rollback@1, scg/final-report@1
- Evidence requirements JSON: ["scg/trust-docs@1","scg/incremental-seal-qualification@1","scg/release@1","scg/rollback@1","scg/final-report@1"]
- Evidence criteria: {"current_tree":true,"rollback_tested":true,"proof_scope_precise":true,"promotion_requires_authorization":true,"remaining_risks_reported":true}
- Outputs: trust/privacy/promotion documentation, release evidence, final report
- Predicted files: docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md, docs/guides/SEMANTIC_COMPRESSION_GOVERNOR.md, docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_REPORT.md, artifacts/agent_supervisor/semantic_compression_governor/release.json
- Predicted files JSON: ["docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md","docs/guides/SEMANTIC_COMPRESSION_GOVERNOR.md","docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_REPORT.md","artifacts/agent_supervisor/semantic_compression_governor/release.json"]
- Interfaces: SemanticGovernorReleaseQualification@1
- Validation: python3 -m pytest -q test/api/semantic_governor && python3 scripts/validate_semantic_compression_governor_board.py --check-all
- Acceptance: Final evidence names exact commits and all required outcome/cost/risk fields; no policy is promoted without separate authorization; proof claims stop at properties actually encoded and verified.
- Gap task: SCG-048
- Refinement: Documentation and seal qualification may proceed in parallel after benchmark evidence, then terminal qualification joins them.
- Embedding query: semantic governor trust privacy promotion rollback seal scope final report production risk
- AST query: Verify public exports, installed CLI, artifact bindings, CAS history, release sealer, and report generation on the exact current tree.
- Conflict policy: Release cannot infer unavailable evidence, rewrite history, self-authorize promotion, or claim universal semantic completeness.
