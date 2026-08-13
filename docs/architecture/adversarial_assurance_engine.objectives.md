# Adversarial Assurance Engine objective graph

Durable goal heap for board namespace `adversarial-assurance-engine-v1`. Task completion records implementation progress; only evidence-backed reconciliation may mark a goal verified complete. `AAE-006` is an operator-controlled upstream-release gate and cannot be completed by workers.

## AAE-G000 Deliver a safe, reproducible counterfactual semantic-mutation system that finds false assurance, diagnoses the smallest assurance gap, qualifies held-out remediations, and permits only authorized CAS policy promotion

- Status: active
- Parent: none
- Parent goal IDs JSON: []
- Depends on:
- Dependencies JSON: []
- Fib priority: 1000
- Track: program
- Priority: P0
- Bundle: adversarial-assurance/program
- Parallel lane: integration
- Resource class: cpu-large
- Goal: Deliver a safe, reproducible counterfactual semantic-mutation system that finds false assurance, diagnoses the smallest assurance gap, qualifies held-out remediations, and permits only authorized CAS policy promotion.
- Producing tasks: AAE-001 through AAE-063
- Evidence: aae/current-tree-conformance@1, aae/controlled-campaigns@1, aae/promotion-qualification@1, aae/final-report@1
- Evidence requirements JSON: ["aae/current-tree-conformance@1","aae/controlled-campaigns@1","aae/promotion-qualification@1","aae/final-report@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":true,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: all four repository-owned AAE packages, controlled campaigns, sealed evidence, benchmark and final report
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance, ipfs_accelerate_py/agent_supervisor/adversarial_assurance, ipfs_accelerate_py/cli.py, ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store, ipfs_accelerate_py/mcplusplus/schemas, ipfs_accelerate_py/mcplusplus/conformance/vectors, test/api/adversarial_assurance, docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance","ipfs_accelerate_py/agent_supervisor/adversarial_assurance","ipfs_accelerate_py/cli.py","ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store","ipfs_accelerate_py/mcplusplus/schemas","ipfs_accelerate_py/mcplusplus/conformance/vectors","test/api/adversarial_assurance","docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md"]
- Interfaces: AdversarialAssuranceEngine@1, assurance CLI
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: All required APIs and campaigns are current-tree tested; high-risk survivors are explicit gaps; results and proof scope are reported honestly.
- Gap task: AAE-063
- Refinement: Child goals own exact authorities, contracts, semantic operators and analyses, persistence, isolated execution, campaigns, surfaces, qualification, and closeout.
- Embedding query: counterfactual semantic mutation false assurance vacuity detection remediation held out promotion
- AST query: Locate semantic index/state, context packer, verification planner/cache, worktree/resource schedulers, durable CAS, policy/state checking, and proof sealing.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G010 Freeze exact commits, interfaces, status taxonomies, manifests, mutation/fuzz/vacuity assets, blind spots, and focused baselines

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 987
- Track: authority
- Priority: P0
- Bundle: adversarial-assurance/authority
- Parallel lane: inventory
- Resource class: cpu-medium
- Goal: Freeze exact commits, interfaces, status taxonomies, manifests, mutation/fuzz/vacuity assets, blind spots, and focused baselines.
- Producing tasks: AAE-001, AAE-002, AAE-003, AAE-004, AAE-005
- Evidence: aae/accelerate-inventory@1, aae/datasets-inventory@1, aae/kit-inventory@1, aae/interop-inventory@1, aae/authority-matrix@1
- Evidence requirements JSON: ["aae/accelerate-inventory@1","aae/datasets-inventory@1","aae/kit-inventory@1","aae/interop-inventory@1","aae/authority-matrix@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":false,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: docs/architecture/adversarial_assurance_inventory and durable focused baseline receipts
- Predicted files: docs/architecture/adversarial_assurance_inventory
- Predicted files JSON: ["docs/architecture/adversarial_assurance_inventory"]
- Interfaces: AssuranceAuthorityMatrix@1, FocusedBaselineReceipt@1
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: Every reuse claim binds exact source and test evidence; stale/red/unavailable authorities remain visible and release gating is fail closed.
- Gap task: AAE-005
- Refinement: Four inventories run in parallel and join into a current-tree matrix with durable baseline receipts; the separate runtime goal owns the operator release gate.
- Embedding query: exact commits interfaces statuses manifests tests mutation fuzz vacuity sealer availability
- AST query: Enumerate exports and tests for all named reused systems and existing mutation, property, proof, policy, state, ZK, storage, and scheduler surfaces.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G020 Define one closed, versioned canonical artifact vocabulary with deterministic identity and no duplicate content, receipt, proof, or storage authority

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on: AAE-G010
- Dependencies JSON: ["AAE-G010"]
- Fib priority: 610
- Track: contracts
- Priority: P0
- Bundle: adversarial-assurance/contracts
- Parallel lane: datasets
- Resource class: cpu-medium
- Goal: Define one closed, versioned canonical artifact vocabulary with deterministic identity and no duplicate content, receipt, proof, or storage authority.
- Producing tasks: AAE-007, AAE-008, AAE-009, AAE-010, AAE-011, AAE-012, AAE-013
- Evidence: aae/common-contracts@1, aae/mutation-contracts@1, aae/execution-contracts@1, aae/assurance-contracts@1, aae/remediation-contracts@1, aae/receipt-contracts@1, aae/conformance-decision@1
- Evidence requirements JSON: ["aae/common-contracts@1","aae/mutation-contracts@1","aae/execution-contracts@1","aae/assurance-contracts@1","aae/remediation-contracts@1","aae/receipt-contracts@1","aae/conformance-decision@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":false,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: datasets contracts, schemas, signed receipt models, and package exports plus MCP++ shared schemas and flat canonical vectors
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance, ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance, ipfs_accelerate_py/mcplusplus/schemas, ipfs_accelerate_py/mcplusplus/conformance/vectors
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance","ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance","ipfs_accelerate_py/mcplusplus/schemas","ipfs_accelerate_py/mcplusplus/conformance/vectors"]
- Interfaces: AdversarialAssuranceArtifacts@1, AssuranceCampaignReceipt, AssurancePolicyPromotionReceipt
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: Every required model, binding, enum, identity, provenance field, and fail-closed decoder is tested with canonical round trips and negative vectors.
- Gap task: AAE-013
- Refinement: Datasets owns neutral semantic artifacts; accelerate and kit use narrow typed projections; MCP++ changes only for demonstrated cross-language needs.
- Embedding query: closed versioned mutation campaign detection outcome gap remediation receipt canonical identity
- AST query: Reuse datasets content identity, semantic-state wire contracts, verification receipts, and kit storage conventions.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G030 Implement deterministic bounded semantic operators, risk-target selection, generation, expected detectors, and fail-closed mutant admission

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on: AAE-G020
- Dependencies JSON: ["AAE-G020"]
- Fib priority: 377
- Track: semantic-mutation
- Priority: P0
- Bundle: adversarial-assurance/mutation
- Parallel lane: datasets
- Resource class: cpu-medium
- Goal: Implement deterministic bounded semantic operators, risk-target selection, generation, expected detectors, and fail-closed mutant admission.
- Producing tasks: AAE-014, AAE-015, AAE-016, AAE-017, AAE-018, AAE-019, AAE-020, AAE-021, AAE-022, AAE-023, AAE-024
- Evidence: aae/operator-registry@1, aae/operator-classes@1, aae/target-selection@1, aae/generator@1, aae/detection-prediction@1, aae/admission@1
- Evidence requirements JSON: ["aae/operator-registry@1","aae/operator-classes@1","aae/target-selection@1","aae/generator@1","aae/detection-prediction@1","aae/admission@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":false,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: datasets mutation registry/generator/detection logic and accelerate admission guardrails
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/generator.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/detection.py, ipfs_accelerate_py/agent_supervisor/adversarial_assurance/admission.py
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/operators","ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/generator.py","ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/detection.py","ipfs_accelerate_py/agent_supervisor/adversarial_assurance/admission.py"]
- Interfaces: MutationOperatorRegistry@1, generate_mutation_candidates, predict_detection_set, admit_mutation
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: All eleven operator classes are bounded, deterministic, rollback-capable, and unable to escape declared targets or mutate assurance authority implicitly.
- Gap task: AAE-024
- Refinement: Operators are split by semantic class for parallel ownership, then join through risk selection, deterministic generation, detection prediction, and admission.
- Embedding query: semantic mutation operator deterministic bounded expected detector isolated admission
- AST query: Extend existing AST, schema, semantic graph, MutationLedger, and worktree scan surfaces without cloning them.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G040 Analyze bounded equivalence and vacuity, compare expected versus observed detection, classify gaps and adequacy, minimize survivors, and specify non-overfit remediations and held-out evaluation

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on: AAE-G020, AAE-G030
- Dependencies JSON: ["AAE-G020","AAE-G030"]
- Fib priority: 233
- Track: assurance-analysis
- Priority: P0
- Bundle: adversarial-assurance/analysis
- Parallel lane: datasets
- Resource class: cpu-medium
- Goal: Analyze bounded equivalence and vacuity, compare expected versus observed detection, classify gaps and adequacy, minimize survivors, and specify non-overfit remediations and held-out evaluation.
- Producing tasks: AAE-025, AAE-026, AAE-027, AAE-028, AAE-029, AAE-030, AAE-031, AAE-032, AAE-033
- Evidence: aae/equivalence@1, aae/vacuity@1, aae/gap-diagnosis@1, aae/adequacy@1, aae/minimization@1, aae/remediation-spec@1, aae/held-out-policy@1
- Evidence requirements JSON: ["aae/equivalence@1","aae/vacuity@1","aae/gap-diagnosis@1","aae/adequacy@1","aae/minimization@1","aae/remediation-spec@1","aae/held-out-policy@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":true,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: datasets analyzers and specifications plus minimized survivor contracts
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/equivalence.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/vacuity.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/gaps.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/remediation.py
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/equivalence.py","ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/vacuity.py","ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/gaps.py","ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance/remediation.py"]
- Interfaces: analyze_vacuity, diagnose_assurance_gap, propose_gap_remediation
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: No hard-to-kill mutant is called equivalent by default; each vacuity finding states what remains proven; candidate intent and held-out partitions have deterministic provenance.
- Gap task: AAE-033
- Refinement: Equivalence, formal/policy/test/ZK vacuity, detection failure, adequacy, diagnosis, minimization, remediation, and held-out policy are separately testable.
- Embedding query: equivalent mutant vacuous proof policy test ZK assurance gap adequacy remediation held out
- AST query: Reuse symbolic, SMT, state-machine, policy, proof candidate, dependency, context, and counterexample surfaces where available.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G050 Persist immutable campaign artifacts, histories, Merkle roots, promotion revisions, and crash recovery using the existing durable coordination and CAS authority

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on: AAE-G020
- Dependencies JSON: ["AAE-G020"]
- Fib priority: 144
- Track: durability
- Priority: P0
- Bundle: adversarial-assurance/storage
- Parallel lane: kit
- Resource class: io-medium
- Goal: Persist immutable campaign artifacts, histories, Merkle roots, promotion revisions, and crash recovery using the existing durable coordination and CAS authority.
- Producing tasks: AAE-034, AAE-035, AAE-036, AAE-037, AAE-038
- Evidence: aae/artifact-store@1, aae/campaign-store@1, aae/merkle-manifest@1, aae/policy-cas@1, aae/recovery@1
- Evidence requirements JSON: ["aae/artifact-store@1","aae/campaign-store@1","aae/merkle-manifest@1","aae/policy-cas@1","aae/recovery@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":false,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: ipfs_kit_py.adversarial_assurance_store
- Predicted files: ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store, ipfs_kit_py/tests/adversarial_assurance_store
- Predicted files JSON: ["ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store","ipfs_kit_py/tests/adversarial_assurance_store"]
- Interfaces: AdversarialAssuranceStore@1, AssurancePolicyRepository@1
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: Artifacts and signed receipts are immutable, content-addressed, and signature-verified through existing authorities; histories replay deterministically; stale writers fail; partial campaigns never promote.
- Gap task: AAE-038
- Refinement: Separate immutable blocks, campaign history, Merkle/seal manifests, policy CAS, and recovery/concurrency for bounded conflict domains.
- Embedding query: immutable mutant campaign receipt assurance gap benchmark Merkle CAS promotion recovery
- AST query: Follow DurableCoordinationStore, DurableStateRootAdapter, semantic_governor_store, and proof_seal_store patterns.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G060 Compose released authorities into isolated resource-admitted mutation campaigns, incremental verification, minimization, remediation evaluation, and authorized promotion

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on: AAE-G010, AAE-G030, AAE-G040, AAE-G050
- Dependencies JSON: ["AAE-G010","AAE-G030","AAE-G040","AAE-G050"]
- Fib priority: 89
- Track: execution
- Priority: P0
- Bundle: adversarial-assurance/runtime
- Parallel lane: accelerate
- Resource class: cpu-large
- Goal: Compose released authorities into isolated resource-admitted mutation campaigns, incremental verification, minimization, remediation evaluation, and authorized promotion.
- Producing tasks: AAE-006, AAE-039, AAE-040, AAE-041, AAE-042, AAE-043, AAE-044, AAE-045, AAE-046, AAE-047, AAE-048
- Evidence: aae/prerequisite-release@1, aae/runtime-adapters@1, aae/campaign-plan@1, aae/isolated-executor@1, aae/incremental-execution@1, aae/outcome@1, aae/diagnosis-run@1, aae/remediation-evaluation@1, aae/promotion@1, aae/public-api@1
- Evidence requirements JSON: ["aae/prerequisite-release@1","aae/runtime-adapters@1","aae/campaign-plan@1","aae/isolated-executor@1","aae/incremental-execution@1","aae/outcome@1","aae/diagnosis-run@1","aae/remediation-evaluation@1","aae/promotion@1","aae/public-api@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":true,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: ipfs_accelerate_py.agent_supervisor.adversarial_assurance runtime
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance, test/api/adversarial_assurance
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/adversarial_assurance","test/api/adversarial_assurance"]
- Interfaces: create_assurance_manifest, plan_mutation_campaign, execute_mutation_campaign and all required APIs
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: Only disposable worktrees execute; network and credentials are absent; verification is incremental and fully keyed; the canonical seal/policy remains untouched without authority.
- Gap task: AAE-048
- Refinement: A manual prerequisite gate precedes adapters; planning, worktrees, workers, incremental verification, classification, diagnosis, remediation, promotion, and API composition remain bounded.
- Embedding query: isolated mutation worktree resource worker incremental verification proof cache temporary forest remediation promotion
- AST query: Reuse WorktreeLifecycleStore, ResourceScheduler, ValidationScheduler, ProofScheduler, VerificationExecutor/cache, CounterexampleMinimizer, and released sealer.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G070 Build deterministic diagnosis/development/held-out fixture campaigns for security, semantic compression, ZK/seals, distributed durability, vacuity, and conditional GUI action binding

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on: AAE-G030, AAE-G040, AAE-G050, AAE-G060
- Dependencies JSON: ["AAE-G030","AAE-G040","AAE-G050","AAE-G060"]
- Fib priority: 55
- Track: campaigns
- Priority: P0
- Bundle: adversarial-assurance/campaigns
- Parallel lane: fixtures
- Resource class: cpu-large
- Goal: Build deterministic diagnosis/development/held-out fixture campaigns for security, semantic compression, ZK/seals, distributed durability, vacuity, and conditional GUI action binding.
- Producing tasks: AAE-049, AAE-050, AAE-051, AAE-052, AAE-053, AAE-054, AAE-055
- Evidence: aae/fixture-partition@1, aae/security-campaign@1, aae/compression-campaign@1, aae/seal-campaign@1, aae/distributed-crash-campaign@1, aae/vacuity-gui-campaign@1
- Evidence requirements JSON: ["aae/fixture-partition@1","aae/security-campaign@1","aae/compression-campaign@1","aae/seal-campaign@1","aae/distributed-crash-campaign@1","aae/vacuity-gui-campaign@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":true,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: test/fixtures/adversarial_assurance and controlled campaign tests
- Predicted files: test/fixtures/adversarial_assurance, test/api/adversarial_assurance/test_campaigns.py
- Predicted files JSON: ["test/fixtures/adversarial_assurance","test/api/adversarial_assurance/test_campaigns.py"]
- Interfaces: AssuranceFixtureCorpus@1
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: Every mandated fixture is present with requirement provenance, expected detector, bounded oracle, risk, and held-out partition; critical seal fixtures fail closed.
- Gap task: AAE-055
- Refinement: Corpus authority and partitions land first; security halves, compression, ZK/seal, distributed/crash, and vacuity/GUI cases proceed in parallel.
- Embedding query: security authorization storage semantic compression ZK seal distributed crash vacuity GUI mutation fixtures
- AST query: Reuse existing semantic-state, SCG, proof reuse, policy, state-machine, CID, receipt, and GUI optimizer fixtures.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G080 Expose the CLI, metrics, reports, harden security/crash behavior, qualify held-out promotion, seal campaigns, benchmark economics, and feed non-authoritative SCG calibration evidence

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on: AAE-G050, AAE-G060, AAE-G070
- Dependencies JSON: ["AAE-G050","AAE-G060","AAE-G070"]
- Fib priority: 34
- Track: qualification
- Priority: P0
- Bundle: adversarial-assurance/qualification
- Parallel lane: integration
- Resource class: cpu-large
- Goal: Expose the CLI, metrics, reports, harden security/crash behavior, qualify held-out promotion, seal campaigns, benchmark economics, and feed non-authoritative SCG calibration evidence.
- Producing tasks: AAE-056, AAE-057, AAE-058, AAE-059, AAE-060, AAE-061, AAE-062
- Evidence: aae/cli@1, aae/metrics@1, aae/security-e2e@1, aae/crash-e2e@1, aae/promotion-e2e@1, aae/seal-benchmark@1
- Evidence requirements JSON: ["aae/cli@1","aae/metrics@1","aae/security-e2e@1","aae/crash-e2e@1","aae/promotion-e2e@1","aae/seal-benchmark@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":true,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: CLI, metrics and reports, security/crash/promotion E2E, benchmark and seal artifacts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli.py, ipfs_accelerate_py/cli.py, benchmarks/agent_supervisor/adversarial_assurance.py, artifacts/agent_supervisor/adversarial_assurance, test/api/adversarial_assurance
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/adversarial_assurance/cli.py","ipfs_accelerate_py/cli.py","benchmarks/agent_supervisor/adversarial_assurance.py","artifacts/agent_supervisor/adversarial_assurance","test/api/adversarial_assurance"]
- Interfaces: assurance CLI, AssuranceMetrics@1, AssuranceBenchmarkReport@1
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: All surface commands are hermetic; metrics are disjoint and reproducible; targets are reported honestly; no candidate, simulation, or seal overclaims assurance.
- Gap task: AAE-062
- Refinement: Two CLI slices, metrics, security and crash E2E, held-out promotion qualification, and sealing/benchmark integration converge before closeout.
- Embedding query: assurance CLI metrics economics security crash held out promotion campaign seal benchmark calibration
- AST query: Integrate current CLI host, lifecycle cancellation, durable recovery, verifier metrics, SCG calibration store, and released proof sealer.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.

## AAE-G090 Audit the current recursive tree and publish the trust model, limitation record, benchmark evidence, campaign results, and exact production qualification next steps

- Status: active
- Parent: AAE-G000
- Parent goal IDs JSON: ["AAE-G000"]
- Depends on: AAE-G080
- Dependencies JSON: ["AAE-G080"]
- Fib priority: 21
- Track: closeout
- Priority: P0
- Bundle: adversarial-assurance/closeout
- Parallel lane: integration
- Resource class: cpu-large
- Goal: Audit the current recursive tree and publish the trust model, limitation record, benchmark evidence, campaign results, and exact production qualification next steps.
- Producing tasks: AAE-063
- Evidence: aae/final-report@1, aae/current-tree-conformance@1
- Evidence requirements JSON: ["aae/final-report@1","aae/current-tree-conformance@1"]
- Evidence criteria: {"results_honest":true,"canonical_identity_required":true,"held_out_required":true,"unauthorized_policy_changes":0,"proof_scope_bounded":true}
- Outputs: docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md and trust/limitations documentation
- Predicted files: docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md, docs/guides/adversarial_assurance_engine.md
- Predicted files JSON: ["docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md","docs/guides/adversarial_assurance_engine.md"]
- Interfaces: AdversarialAssuranceEngine qualification report
- Validation: python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
- Acceptance: The final report contains every required count/rate/commit/reuse/gap/remediation/seal/cost/nonclaim and only the prescribed bounded concluding claim.
- Gap task: AAE-063
- Refinement: One terminal task reruns the exact current-tree matrix and fans in every campaign, metric, promotion, benchmark, seal, and limitation artifact.
- Embedding query: final assurance report commits operators mutants detection gaps remediation promotion seal limits next steps
- AST query: Audit exported APIs, CLI commands, tests, fixtures, receipts, Merkle roots, policy revisions, benchmark results, and documentation.
- Conflict policy: Reuse canonical identity, indexing, context, verification, scheduling, persistence, proof, and receipt authorities; fail closed on missing capability; never create a new MCP++ profile or lower assurance.
