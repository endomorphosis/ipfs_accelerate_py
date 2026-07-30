# Formal Verification Tactician Readiness Objective Heap

Ultimate goal: turn the completed LFV architecture into a production-qualified
formal software-verification platform and add a sound, goal-directed proof
tactician that can formalize an explained end state, discover the missing
proof obligations needed to reach it, validate candidate lemmas and
assumptions, and return safe replayable counterexamples when the goal does not
hold.

Program invariants:

- Existing `LFV-G000` artifacts are the implementation baseline, not evidence
  that every optional external tool is installed or production-qualified.
- `ipfs_datasets_py.logic` owns canonical semantics, goal/proof-hole
  contracts, provider-facing compilation, counterexample semantics, and the
  stable public verification API.
- `ipfs_accelerate_py.agent_supervisor` owns orchestration, resource leases,
  isolation, durable proof-plan execution, replanning, and operational
  monitoring.
- Imports, declarations, inventory, and probes never install, download,
  access the network, spawn an unbounded process, or mutate a checkout.
- An advisor, model, autoencoder, Leanstral, SymAI, retrieval result, cache
  hit, test, monitor, bounded model, or ZKP attestation is never silently
  promoted to theorem or source-translation authority.
- New assumptions are visible proof obligations with explicit cost and review
  state; a favorable assumption cannot be inserted merely because it entails
  the target.
- A structural repair never closes a counterexample. Only a fresh matching
  verifier receipt can close it.
- Raw source, stdout, credentials, tokens, private witnesses, and hidden
  channels never enter a public API response or model context.
- Every translation, check, proof, replay, and minimization result binds the
  exact source tree, property, assumptions, provider/version, policy, and
  bounds.
- Minimized counterexamples state their actual guarantee: none, normalized,
  bounded, locally minimal, or globally minimal.
- Tool absence, malformed input, timeout, disagreement, unsupported
  semantics, budget exhaustion, and ambiguity remain explicit non-success
  states.
- Existing legal tactician, logic API, CLI, MCP, DCEC/TDFOL/CEC, Hammer,
  cache, corpus, and provider behavior remains available through reviewed
  compatibility adapters.

## FVT-G000 Production-ready formal verification and goal-directed proof tactician

- Status: provisionally_complete
- Parent:
- Depends on: FVT-G090
- Fib priority: 24157817
- Priority: P0
- Track: integration
- Bundle: formal-verification-tactician/release
- Goal: Complete and attest the trust-boundary, production-readiness, end-goal formalization, missing-proof search, semantic counterexample, public-surface, supervisor, and rollout program.
- Evidence: docs/architecture/formal_verification_tactician_readiness_completion_receipt.json
- Outputs: docs/architecture/formal_verification_tactician_readiness_completion_receipt.json
- Validation: python -m pytest test/api/test_formal_verification_tactician_readiness_completion.py -q
- Acceptance: A current-tree receipt binds every executable child receipt, parent and datasets identities, live toolchain report, corpus and benchmark results, public-operation conformance, rollout policy, zero false-proof/false-closure/leakage/authority violations, and all disclosed unsupported or unavailable capabilities.
- Conflict policy: Tracking-only root; reconcile only after FVT-G090 and every executable child goal are complete.
- Interfaces: FormalVerificationTacticianRelease@1
- Resource class: cpu-validation
- Goal completion schema version: 1
- Completion confidence: 0.083333
- Uncovered criteria: ["A current-tree receipt binds every executable child receipt, parent and datasets identities, live toolchain report, corpus and benchmark results, public-operation conformance, rollout policy, zero false-proof/false-closure/leakage/authority violations, and all disclosed unsupported or unavailable capabilities."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: ["missing_criterion_evidence","coverage_missing","validation_evidence_incomplete","analyzer_health_missing","exhaustion_quorum_missing","child_unverified","tasks_incomplete"]
- State transitioned at: 2026-07-30T19:21:09.208124+00:00
- State transition reason: Produce completion evidence for: A current-tree receipt binds every executable child receipt, parent and datasets identities, live toolchain report, corpus and benchmark results, public-operation conformance, rollout policy, zero false-proof/false-closure/leakage/authority violations, and all disclosed unsupported or unavailable capabilities.; Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree.; Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one.; Require an explicitly healthy analyzer that is safe for completion reasoning.; Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.; Every descendant must remain verified with all proof requirements fresh, conclusive, uncontradicted, and satisfied.; Complete the remaining producing tasks before requesting verification.

## FVT-G005 Establish an executable production-readiness baseline

- Status: active
- Parent: FVT-G000
- Depends on:
- Fib priority: 1
- Priority: P0
- Track: readiness-baseline
- Bundle: formal-verification-tactician/readiness
- Goal: Turn the current audit into a current-tree, machine-specific readiness ledger that distinguishes implemented, fixture-tested, live-tested, installed, usable, production-certified, unsupported, and unavailable capabilities.
- Evidence: docs/architecture/formal_verification_readiness_baseline.json, test/api/test_formal_verification_readiness_baseline.py
- Outputs: docs/architecture/formal_verification_readiness_baseline.json, test/api/test_formal_verification_readiness_baseline.py
- Validation: python -m pytest test/api/test_formal_verification_readiness_baseline.py -q
- Acceptance: The ledger is derived from bounded checks, records parent/gitlink/origin alignment separately, reports exact executable and package identities, catches the observed Lean shim/toolchain mismatch, labels synthetic/offline evidence, and never infers usability from source or PATH presence.
- Conflict policy: Own the new baseline artifact and test; inspect existing receipts and probes read-only and do not install, fetch, publish, rewrite gitlinks, or edit provider behavior.
- Interfaces: FormalVerificationReadinessBaseline@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-small
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["The ledger is derived from bounded checks, records parent/gitlink/origin alignment separately, reports exact executable and package identities, catches the observed Lean shim/toolchain mismatch, labels synthetic/offline evidence, and never infers usability from source or PATH presence."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G006 Make receipt verification and attestation fail closed

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G005
- Fib priority: 2
- Priority: P0
- Track: receipt-authority
- Bundle: formal-verification-tactician/trust-boundary
- Goal: Replace permissive structural receipt handling with closed schema dispatch and exact validation of content identity, source/property/assumption/bound/tool bindings, freshness, authority, proof artifacts, and independent checker evidence.
- Evidence: ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py, test/api/test_logic_receipt_authority_boundary.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py, test/api/test_logic_receipt_authority_boundary.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py test/api/test_logic_receipt_authority_boundary.py -q
- Acceptance: Empty, unknown, forged-kernel, stale, wrong-tree, wrong-property, wrong-assumption, wrong-bound, wrong-tool, and cross-authority inputs are rejected; a prepared/simulated attestation cannot report proof success; valid typed receipts round trip without authority loss.
- Conflict policy: Own stable receipt and attestation dispatch plus adversarial tests; preserve existing typed receipt schemas and do not weaken them to accommodate legacy mappings.
- Interfaces: VerifiedReceiptDispatch@2, AttestationAuthorityBoundary@2
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-type-check
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Empty, unknown, forged-kernel, stale, wrong-tree, wrong-property, wrong-assumption, wrong-bound, wrong-tool, and cross-authority inputs are rejected","a prepared/simulated attestation cannot report proof success","valid typed receipts round trip without authority loss."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G007 Unify the secret-safe public counterexample boundary

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G005
- Fib priority: 2
- Priority: P0
- Track: counterexample-boundary
- Bundle: formal-verification-tactician/trust-boundary
- Goal: Route datasets Python/CLI/MCP and supervisor/model projections through one closed, bounded, content-addressed counterexample envelope and eliminate raw payload exposure.
- Evidence: ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py, test/api/test_counterexample_cross_repository_contract.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py, test/api/test_counterexample_cross_repository_contract.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py test/api/test_counterexample_cross_repository_contract.py -q
- Acceptance: Unknown fields and forged identities fail closed; hidden_witness, token, credential, raw source, stdout, and private channels never appear publicly; raw artifacts are referenced only by private digest/retention metadata; all projections preserve kind, property, source-map, tool, assumptions, bounds, and authority.
- Conflict policy: Own the new datasets wire contract, verification API delegation, and cross-repository adapter tests; extend the mature supervisor normalizer without creating a second semantic identity.
- Interfaces: CounterexampleEnvelope@2, PublicCounterexampleBoundary@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-sanitize
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Unknown fields and forged identities fail closed","hidden_witness, token, credential, raw source, stdout, and private channels never appear publicly","raw artifacts are referenced only by private digest/retention metadata","all projections preserve kind, property, source-map, tool, assumptions, bounds, and authority."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G008 Require verifier-backed counterexample closure

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G006, FVT-G007
- Fib priority: 3
- Priority: P0
- Track: replan-soundness
- Bundle: formal-verification-tactician/trust-boundary
- Goal: Make formal replanning distinguish a structurally admissible repair from a verifier-confirmed repair and keep every witness open until exact fresh re-verification succeeds.
- Evidence: test/api/test_agent_supervisor_formal_replanner_verifier_closure.py
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py, test/api/test_agent_supervisor_formal_replanner_verifier_closure.py
- Validation: python -m pytest test/api/test_agent_supervisor_formal_replanner.py test/api/test_agent_supervisor_formal_replanner_verifier_closure.py -q
- Acceptance: `_addresses_counterexample` alone never changes open count from one to zero; no verifier, unavailable verifier, stale receipt, changed tree/property/assumption/bound, timeout, and disagreement leave the witness open or unknown; closure names a fresh matching verifier receipt.
- Conflict policy: Own formal-replanner closure semantics and focused tests; preserve bounded repair proposal generation and do not treat a successful compile or plan consistency check as semantic verification.
- Interfaces: VerifierBackedRepairClosure@1
- Resource class: cpu-proof-check
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["`_addresses_counterexample` alone never changes open count from one to zero","no verifier, unavailable verifier, stale receipt, changed tree/property/assumption/bound, timeout, and disagreement leave the witness open or unknown","closure names a fresh matching verifier receipt."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G009 Route every external tool through one bounded lifecycle

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G005
- Fib priority: 2
- Priority: P0
- Track: tool-runtime
- Bundle: formal-verification-tactician/runtime
- Goal: Remove direct unbounded subprocess execution from concrete backends and version probes and enforce one injected process lifecycle across native, JVM, OCaml/opam, kernel, and WASM tools.
- Evidence: ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/process.py, ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_process_lifecycle.py ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py -q
- Acceptance: SMT/differential and every other adapter and probe use argument arrays, private workspaces, process-tree termination, wall/memory/CPU/output bounds, cancellation, redaction, and cleanup; adversarial fake tools cannot escape paths, leave children, flood output, or trigger installation/network access.
- Conflict policy: Own shared process lifecycle integration and isolation tests; change backend invocation mechanics without changing their formula semantics or result authority.
- Interfaces: UniversalBoundedToolLifecycle@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["SMT/differential and every other adapter and probe use argument arrays, private workspaces, process-tree termination, wall/memory/CPU/output bounds, cancellation, redaction, and cleanup","adversarial fake tools cannot escape paths, leave children, flood output, or trigger installation/network access."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G010 Qualify clean packages and hermetic offline toolchains

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G005, FVT-G009
- Fib priority: 3
- Priority: P0
- Track: packaging-toolchains
- Bundle: formal-verification-tactician/runtime
- Goal: Prove that Python wheel/sdist and TypeScript package artifacts contain every stable verification module and run against exact pinned offline external-tool identities without opportunistic downloads.
- Evidence: config/formal_verification_toolchains.lock.json, test/packaging/test_logic_verification_clean_install.py, ipfs_datasets_py/tests/packaging/test_logic_verification_npm_layout.py
- Outputs: config/formal_verification_toolchains.lock.json, test/packaging/test_logic_verification_clean_install.py, ipfs_datasets_py/tests/packaging/test_logic_verification_npm_layout.py
- Validation: python -m pytest test/packaging/test_logic_verification_clean_install.py ipfs_datasets_py/tests/packaging/test_logic_verification_npm_layout.py -q
- Acceptance: Empty-environment installs import and exercise all stable Python operations; npm declared and built entrypoints agree; namespace/package discovery includes new modules; exact toolchain probes detect shims and version mismatch; offline verification performs no install, download, or network access.
- Conflict policy: Own package manifests, toolchain lock, and clean-artifact tests; do not vendor unreviewed binaries or alter solver semantics.
- Interfaces: FormalVerificationPackagingGate@1, OfflineToolchainLock@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-install-test
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Empty-environment installs import and exercise all stable Python operations","npm declared and built entrypoints agree","namespace/package discovery includes new modules","exact toolchain probes detect shims and version mismatch","offline verification performs no install, download, or network access."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G011 Build the source-to-VC-to-solver vertical slice

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G006, FVT-G007, FVT-G009
- Fib priority: 5
- Priority: P0
- Track: executable-pipeline
- Bundle: formal-verification-tactician/vertical-slice
- Goal: Connect a source snapshot through typed program/contracts, verification-condition generation, backend-neutral SMT obligations, Z3/CVC5 execution, and source-bound proof/counterexample receipts.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/pipeline.py, ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/pipeline.py, ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py -q
- Acceptance: Checked-in buggy/fixed programs generate their own VCs and witnesses; Z3 and CVC5 agree or disagreement is quarantined; every result binds source spans/tree/property/assumptions/tool/bounds/translation; unsupported constructs fail explicitly rather than being erased.
- Conflict policy: Own the new pipeline composition and integration test; reuse existing source, ProgramIR, VC, SMT compiler, runner, and receipt modules without inventing parallel semantic contracts.
- Interfaces: SourceToVerificationPipeline@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-smt
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Checked-in buggy/fixed programs generate their own VCs and witnesses","Z3 and CVC5 agree or disagreement is quarantined","every result binds source spans/tree/property/assumptions/tool/bounds/translation","unsupported constructs fail explicitly rather than being erased."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G012 Execute the full lazy provider matrix through stable surfaces

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G006, FVT-G009, FVT-G011
- Fib priority: 8
- Priority: P0
- Track: provider-execution
- Bundle: formal-verification-tactician/provider-surface
- Goal: Register every LFV provider lazily behind the shared protocol, make portfolio execution real rather than plan-only, and expose equivalent availability and execution semantics through Python, datasets MCP, and parent MCP.
- Evidence: ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py, test/api/test_root_mcp_formal_verification_parity.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/registry.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py, test/api/test_root_mcp_formal_verification_parity.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py test/api/test_root_mcp_formal_verification_parity.py -q
- Acceptance: SMT, state-model, runtime, authorization, protocol, hyperproperty, ATP, Hammer, and kernel providers are discoverable without import side effects; available lanes execute; absent lanes report unavailable; portfolios preserve typed authority and quarantine contradiction; both MCP roots match the stable schema.
- Conflict policy: Own registry/public execution wiring and parity tests; do not install providers during discovery or weaken property-specific routing and authority policy.
- Interfaces: ExecutableProviderMatrix@1, FormalVerificationMCPParity@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-portfolio
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["SMT, state-model, runtime, authorization, protocol, hyperproperty, ATP, Hammer, and kernel providers are discoverable without import side effects","available lanes execute","absent lanes report unavailable","portfolios preserve typed authority and quarantine contradiction","both MCP roots match the stable schema."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G013 Replace manifest-only examples and synthetic readiness claims

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G011, FVT-G012
- Fib priority: 13
- Priority: P1
- Track: runnable-examples
- Bundle: formal-verification-tactician/readiness
- Goal: Check in the referenced example sources and mutations, run them through production entrypoints, and derive outcome/security/readiness reports from actual receipts rather than manually injected witnesses or hardcoded distributions.
- Evidence: ipfs_datasets_py/examples/logic/software_verification/README.md, ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py
- Outputs: ipfs_datasets_py/examples/logic/software_verification/README.md, ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py, docs/architecture/formal_verification_live_example_report.json
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py -q
- Acceptance: Every manifest source exists and runs; negative variants generate rather than inject counterexamples; positive variants generate current receipts; reports cite run identities and clearly separate fixture, simulated, live, skipped, unsupported, and unavailable results.
- Conflict policy: Own example sources, runnable integration test, and live report; retain small deterministic fixtures but remove them from production-readiness claims.
- Interfaces: RunnableVerificationExamples@1, LiveReadinessReport@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Every manifest source exists and runs","negative variants generate rather than inject counterexamples","positive variants generate current receipts","reports cite run identities and clearly separate fixture, simulated, live, skipped, unsupported, and unavailable results."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G014 Expand source frontends with explicit semantic profiles

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G011
- Fib priority: 8
- Priority: P1
- Track: source-frontends
- Bundle: formal-verification-tactician/vertical-slice
- Goal: Harden Python and JavaScript/TypeScript frontends and add staged typed frontends for Rust, Go, Java, C/C++, and WASM with source spans, language semantics, and fail-closed supported-fragment coverage.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/frontends/registry.py, ipfs_datasets_py/tests/integration/logic/software_verification/test_frontend_semantic_profiles.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/frontends/registry.py, ipfs_datasets_py/tests/integration/logic/software_verification/test_frontend_semantic_profiles.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/test_frontend_semantic_profiles.py -q
- Acceptance: Each language declares parsed constructs, numeric/memory/concurrency/exception behavior, undefined or implementation-defined semantics, unsupported features, and coverage; opaque bodies and regex approximations cannot receive translation authority; source mapping survives the pipeline.
- Conflict policy: Own frontend registry, semantic profiles, and coverage tests; implement languages incrementally and never claim whole-language support from a partial parser.
- Interfaces: SourceFrontendSemanticProfile@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-parser
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Each language declares parsed constructs, numeric/memory/concurrency/exception behavior, undefined or implementation-defined semantics, unsupported features, and coverage","opaque bodies and regex approximations cannot receive translation authority","source mapping survives the pipeline."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G020 Establish the goal/proof-gap/counterexample golden corpus

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G005
- Fib priority: 2
- Priority: P0
- Track: tactician-corpus
- Bundle: formal-verification-tactician/corpus
- Goal: Define solvable, mutated, impossible, ambiguous, unsupported, and unavailable cases that measure end-goal formalization, proof-gap recovery, proof-chain authority, counterexample replay/minimization, and honest failure.
- Evidence: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py
- Outputs: ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json, test/api/test_formal_verification_tactician_corpus_contract.py
- Validation: python -m pytest test/api/test_formal_verification_tactician_corpus_contract.py -q
- Acceptance: The corpus covers missing loop invariant, callee contract/frame, lease safety, fairness ambiguity, impossible target/core, SMT model, runtime trace, protocol attack, hypertrace, kernel rejection, bridge lemma, and legal evidence routing; fixtures bind licenses/provenance and expected authority without embedding private witnesses.
- Conflict policy: Own new corpus contracts and fixtures; do not tune production behavior to fixture strings or label injected expected results as live verification.
- Interfaces: ProofTacticianCorpus@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-small
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["The corpus covers missing loop invariant, callee contract/frame, lease safety, fairness ambiguity, impossible target/core, SMT model, runtime trace, protocol attack, hypertrace, kernel rejection, bridge lemma, and legal evidence routing","fixtures bind licenses/provenance and expected authority without embedding private witnesses."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G021 Define closed end-goal, proof-hole, graph, and plan contracts

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G006, FVT-G007, FVT-G020
- Fib priority: 3
- Priority: P0
- Track: tactician-contracts
- Bundle: formal-verification-tactician/goal-contracts
- Goal: Define content-addressed EndGoalSpec, interpretation, FormalGoal, ProofHole, ProofGraphNode/Edge, CandidateProofStep, ProofPlan, validation, and completion contracts shared across datasets and supervisor.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/contracts.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_contracts.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/contracts.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_contracts.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_contracts.py -q
- Acceptance: Closed schemas bind tree/source spans/current/target state/property/quantifiers/environment/assumptions by class/logic/providers/bounds/ambiguity/provenance/authority/status; identities change under every semantic binding; proposals cannot claim proof or completion.
- Conflict policy: Own new canonical contracts and tests; adapt existing GoalDevelopment and formal-planning contracts by explicit conversion without introducing a competing root-goal identity.
- Interfaces: EndGoalSpec@1, ProofHole@1, ProofObligationGraph@1, GoalDirectedProofPlan@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-type-check
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Closed schemas bind tree/source spans/current/target state/property/quantifiers/environment/assumptions by class/logic/providers/bounds/ambiguity/provenance/authority/status","identities change under every semantic binding","proposals cannot claim proof or completion."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G022 Formalize prose end goals with source-grounded alternatives

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G021
- Fib priority: 5
- Priority: P0
- Track: goal-formalization
- Bundle: formal-verification-tactician/goal-contracts
- Goal: Extend the prompt/Intent IR path to extract bounded end-goal candidates with actors, state, transitions, environment, quantifiers, property class, assumptions, bounds, assurance, acceptance evidence, and phrase-to-clause provenance.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/end_goal_formalizer.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_end_goal_formalizer.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/end_goal_formalizer.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_end_goal_formalizer.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_end_goal_formalizer.py -q
- Acceptance: Deterministic controlled-language cases round trip; learned parsing is candidate-only; every clause maps to prompt/repository spans; hidden assumptions and identifiers are rejected; unsupported or underspecified semantics remain explicit.
- Conflict policy: Own end-goal extraction and tests; reuse Intent IR/formalization advisor contracts and do not mutate the frozen caller request or admit a candidate.
- Interfaces: EndGoalFormalizer@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Deterministic controlled-language cases round trip","learned parsing is candidate-only","every clause maps to prompt/repository spans","hidden assumptions and identifiers are rejected","unsupported or underspecified semantics remain explicit."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G023 Expose ambiguity and require material interpretation selection

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G022
- Fib priority: 8
- Priority: P0
- Track: goal-ambiguity
- Bundle: formal-verification-tactician/goal-contracts
- Goal: Generate bounded alternative interpretations, controlled-English renderings, semantic diffs, unresolved fields, and confirmation requirements for materially different end-goal meanings.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/ambiguity.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_ambiguity.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/ambiguity.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_ambiguity.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_ambiguity.py -q
- Acceptance: Existential reachability, universal reachability, eventual inevitability, invariance, termination, and refinement cannot collapse; ambiguous corpus prompts return at least two visibly different candidates; no material ambiguity is silently selected.
- Conflict policy: Own interpretation comparison and confirmation policy; do not call external provers or models during deterministic semantic diff.
- Interfaces: GoalInterpretationSet@1, GoalAmbiguityGate@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-translate
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Existential reachability, universal reachability, eventual inevitability, invariance, termination, and refinement cannot collapse","ambiguous corpus prompts return at least two visibly different candidates","no material ambiguity is silently selected."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G024 Compile confirmed goals into shared verification semantics

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G011, FVT-G023
- Fib priority: 13
- Priority: P0
- Track: goal-compilation
- Bundle: formal-verification-tactician/goal-contracts
- Goal: Compile a confirmed EndGoalSpec into SoftwareVerificationIR properties, contracts, transition/environment models, and backend-neutral root obligations with a loss-aware translation receipt.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/goal_compiler.py, ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/goal_compiler.py, ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py -q
- Acceptance: Exact targets and bounds reproduce from content identities; source spans and assumption classes survive; material translation loss or ambiguity fails closed; backend choice cannot raise assurance above the translation ceiling.
- Conflict policy: Own goal-to-shared-IR composition and integration test; reuse LFV semantics and translation receipts rather than embedding provider syntax in EndGoalSpec.
- Interfaces: FormalGoalCompiler@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-translate
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Exact targets and bounds reproduce from content identities","source spans and assumption classes survive","material translation loss or ambiguity fails closed","backend choice cannot raise assurance above the translation ceiling."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G025 Invoke Leanstral goal development only after formalization

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G024
- Fib priority: 21
- Priority: P1
- Track: advisor-routing
- Bundle: formal-verification-tactician/goal-contracts
- Goal: Route prompt workflows through confirmed formalization before Leanstral goal development and expose only immutable selected goal, formula, assumption, vocabulary, and template identifiers to the untrusted provider.
- Evidence: test/api/test_leanstral_end_goal_formalization_route.py
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/end_goal_development.py, test/api/test_leanstral_end_goal_formalization_route.py
- Validation: python -m pytest test/api/test_agent_supervisor_leanstral_goal_development.py test/api/test_leanstral_end_goal_formalization_route.py -q
- Acceptance: Prose cannot bypass formalization; Leanstral cannot create/mutate formulas, source, assumptions, proof, commands, admission, or completion; timeout/unavailable/malformed responses fall back deterministically without stalling the supervisor.
- Conflict policy: Own the post-formalization supervisor adapter and tests; preserve the existing Leanstral capability-isolation boundary and provider modes.
- Interfaces: FormalizedGoalDevelopmentRoute@1
- Resource class: gpu-advisor-optional
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Prose cannot bypass formalization","Leanstral cannot create/mutate formulas, source, assumptions, proof, commands, admission, or completion","timeout/unavailable/malformed responses fall back deterministically without stalling the supervisor."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G030 Emit actionable typed proof holes

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G011, FVT-G021
- Fib priority: 5
- Priority: P0
- Track: proof-holes
- Bundle: formal-verification-tactician/proof-search
- Goal: Make VC and model compilation return source-bound typed holes for missing invariants, variants, contracts, frames, summaries, concurrency/temporal/refinement premises, bridge lemmas, evidence, semantics, tools, and necessary implementation changes.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_holes.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_holes.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py -q
- Acceptance: Removing a loop invariant, callee contract/frame, fairness premise, or bridge lemma yields the matching typed hole with source span, rationale, dependencies, expected authority, and validation recipe; unsupported semantics remain different from missing proof.
- Conflict policy: Own proof-hole contracts/adapters and focused VC behavior; retain fail-closed compilation and do not invent default invariants or contracts.
- Interfaces: TypedProofHoleEmitter@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-compile
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Removing a loop invariant, callee contract/frame, fairness premise, or bridge lemma yields the matching typed hole with source span, rationale, dependencies, expected authority, and validation recipe","unsupported semantics remain different from missing proof."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G031 Construct a bounded backward AND/OR obligation graph

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G024, FVT-G030
- Fib priority: 13
- Priority: P0
- Track: backward-proof-search
- Bundle: formal-verification-tactician/proof-search
- Goal: Regress formal targets through programs and transition systems using weakest preconditions, preimages, temporal regression, typed rule inversion/unification, subsumption, cycle control, and reconstructable AND/OR proof rules.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_graph.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_graph.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py -q
- Acceptance: Every edge names a checked inference/reconstruction rule; AND/OR meanings are distinct; finite budgets, SCC/cycle and subsumption controls terminate; solved leaves cite adequate evidence; legacy string-equality or forward-only “backward” paths cannot receive trusted status.
- Conflict policy: Own the new general proof graph and tests; wrap legacy CEC/TDFOL strategies as experimental candidates unless they reconstruct through the typed rules.
- Interfaces: BackwardProofObligationGraph@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-search
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Every edge names a checked inference/reconstruction rule","AND/OR meanings are distinct","finite budgets, SCC/cycle and subsumption controls terminate","solved leaves cite adequate evidence","legacy string-equality or forward-only \u201cbackward\u201d paths cannot receive trusted status."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G032 Find weakest admissible missing premises by bounded abduction

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G031
- Fib priority: 21
- Priority: P0
- Track: proof-abduction
- Bundle: formal-verification-tactician/proof-search
- Goal: Implement bounded abductive search that classifies facts-to-prove, reviewable environment assumptions, invariants/contracts/lemmas to synthesize, unsupported semantics, unavailable authority, and implementation changes.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/abduction.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_abduction.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/abduction.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_abduction.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_abduction.py -q
- Acceptance: Candidates are relevant, consistent, source/scoped, non-circular, non-vacuous, and weak under the declared finite theory/budget; arbitrary goal-entailing assumptions and contradictions are rejected; impossible targets return a core/witness or honest unknown.
- Conflict policy: Own bounded abduction and tests; never insert a generated premise into the trusted assumption set without separate validation and policy admission.
- Interfaces: MissingProofAbduction@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-search
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Candidates are relevant, consistent, source/scoped, non-circular, non-vacuous, and weak under the declared finite theory/budget","arbitrary goal-entailing assumptions and contradictions are rejected","impossible targets return a core/witness or honest unknown."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G033 Build the candidate lemma, invariant, contract, and evidence portfolio

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G030
- Fib priority: 8
- Priority: P1
- Track: candidate-synthesis
- Bundle: formal-verification-tactician/proof-search
- Goal: Combine exact corpus/cache/Hammer retrieval, reviewed templates, Houdini elimination, SMT cores/interpolation, CHC/PDR/IC3, SyGuS, legal evidence routing, and learned proposal/ranking providers into typed candidate sources.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_synthesis.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_synthesis.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py -q
- Acceptance: Every candidate records source/provider/provenance/trust/budget and targeted holes; autoencoder, Leanstral, SymAI, embeddings, and model output remain proposal-only; legal obligations delegate evidence routing to the existing legal tactician compatibility adapter.
- Conflict policy: Own candidate-source composition and tests; reuse existing utilities through adapters and do not create independent caches, provider registries, or proof authority.
- Interfaces: ProofCandidatePortfolio@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-portfolio
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Every candidate records source/provider/provenance/trust/budget and targeted holes","autoencoder, Leanstral, SymAI, embeddings, and model output remain proposal-only","legal obligations delegate evidence routing to the existing legal tactician compatibility adapter."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G034 Independently validate proof-gap candidates

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G012, FVT-G032, FVT-G033
- Fib priority: 34
- Priority: P0
- Track: candidate-validation
- Bundle: formal-verification-tactician/proof-search
- Goal: Validate each candidate and candidate set with parse/type checks, exact bindings, consistency/non-vacuity/non-circularity, solver/model-checker/kernel replay, deletion/core minimality, and truthful authority/unknown results.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_validation.py, ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_candidate_validation.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_validation.py, ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_candidate_validation.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_candidate_validation.py -q
- Acceptance: No unvalidated or stale candidate discharges a graph node; exact tree/goal/assumptions/tool/policy/bounds are bound; deletion of a selected premise breaks the proof for small minimal cases or the receipt explicitly limits its guarantee; disagreement is quarantined.
- Conflict policy: Own candidate admission/validation and tests; providers may propose evidence but the deterministic validator alone sets validation status.
- Interfaces: ProofCandidateValidator@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-check
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["No unvalidated or stale candidate discharges a graph node","exact tree/goal/assumptions/tool/policy/bounds are bound","deletion of a selected premise breaks the proof for small minimal cases or the receipt explicitly limits its guarantee","disagreement is quarantined."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G035 Rank complete missing-proof plans by authority and utility

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G034
- Fib priority: 55
- Priority: P1
- Track: proof-plan-ranking
- Bundle: formal-verification-tactician/proof-search
- Goal: Construct complete alternatives for the existing AND/OR evaluator, hard-reject invalid branches, and rank plans by discharged coverage, downstream unlock, critical path, authority, assumption cost/risk, proof cost, cache value, and fallback quality.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_plan.py, test/api/test_goal_directed_proof_plan_ranking.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_plan.py, test/api/test_goal_directed_proof_plan_ranking.py
- Validation: python -m pytest test/api/test_goal_directed_proof_plan_ranking.py -q
- Acceptance: Rankings are deterministic and explainable; incomplete/invalid/insufficient-authority branches are hard-pruned; each step names dependencies, expected receipts, validation, fallback, resources, and completion conditions; assumption-heavy plans pay explicit cost.
- Conflict policy: Own proof-plan construction and evaluator adapter tests; reuse existing plan-evaluator scoring primitives without changing unrelated implementation-task routing.
- Interfaces: GoalDirectedProofPlanRanker@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-plan
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Rankings are deterministic and explainable","incomplete/invalid/insufficient-authority branches are hard-pruned","each step names dependencies, expected receipts, validation, fallback, resources, and completion conditions","assumption-heavy plans pay explicit cost."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G036 Integrate the goal-directed tactician with existing utilities

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G025, FVT-G035
- Fib priority: 89
- Priority: P0
- Track: tactician-integration
- Bundle: formal-verification-tactician/proof-search
- Goal: Compose formalization, retrieval, proof scheduler, proof-carrying planner, Hammer/kernels, Leanstral, SymAI, autoencoder, legal evidence adapter, caches, corpus, ZKP receipt binding, and supervisor admission behind one restartable tactician.
- Evidence: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, test/api/test_goal_directed_tactician_integration.py
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, test/api/test_goal_directed_tactician_integration.py
- Validation: python -m pytest test/api/test_goal_directed_tactician_integration.py -q
- Acceptance: Exact cache keys include tree/target/assumptions/provider/version/policy/bounds; model and cache evidence cannot bypass validation; proof-carrying execution is resumable; ZKP binds an existing trusted receipt without increasing its assurance; legal compatibility remains intact.
- Conflict policy: Own the parent orchestration facade and integration test; import canonical datasets contracts through the existing provider boundary and do not duplicate semantics in the supervisor.
- Interfaces: GoalDirectedProofTactician@1
- Resource class: cpu-proof-orchestrate
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Exact cache keys include tree/target/assumptions/provider/version/policy/bounds","model and cache evidence cannot bypass validation","proof-carrying execution is resumable","ZKP binds an existing trusted receipt without increasing its assurance","legal compatibility remains intact."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G040 Implement oracle-preserving semantic counterexample minimizers

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G007, FVT-G020
- Fib priority: 3
- Priority: P0
- Track: counterexample-minimization
- Bundle: formal-verification-tactician/counterexamples
- Goal: Replace unconditional Boolean minimization with backend-specific, budgeted, oracle-preserving model/core/trace/attack/hypertrace/kernel reducers and truthful reduction guarantees.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/minimization.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/minimization.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py -q
- Acceptance: SMT projection/don't-cares and subset cores, shortest prefix/lasso/event slice, protocol dependency slice, and earliest hypertrace divergence recheck the violation after every accepted removal; receipts record oracle, algorithm/version, budget, reduction log, and actual guarantee including exhaustion.
- Conflict policy: Own semantic reducer protocols/implementations and tests; retain normalization/bounding as a distinct lower guarantee and never stamp `minimized` merely because output is short.
- Interfaces: SemanticCounterexampleMinimizer@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-minimize
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["SMT projection/don't-cares and subset cores, shortest prefix/lasso/event slice, protocol dependency slice, and earliest hypertrace divergence recheck the violation after every accepted removal","receipts record oracle, algorithm/version, budget, reduction log, and actual guarantee including exhaustion."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G041 Make every counterexample exactly replayable

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G011, FVT-G040
- Fib priority: 8
- Priority: P0
- Track: counterexample-replay
- Bundle: formal-verification-tactician/counterexamples
- Goal: Define safe replay recipes and receipts that reconstruct the exact property violation from immutable source/model/tool/policy/bound identities without exposing private material.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/replay.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/replay.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py -q
- Acceptance: Corpus witnesses replay under their exact identities and fail binding on changed tree/property/assumption/tool/bound; unavailable tools return unavailable rather than success; raw private artifacts remain out of public recipes; replay result is content addressed.
- Conflict policy: Own replay contracts/runtime and tests; use the universal bounded runner and do not reinterpret provider syntax outside its adapter.
- Interfaces: CounterexampleReplay@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-replay
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Corpus witnesses replay under their exact identities and fail binding on changed tree/property/assumption/tool/bound","unavailable tools return unavailable rather than success","raw private artifacts remain out of public recipes","replay result is content addressed."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G042 Explain first divergence, causal slice, and missing proof

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G030, FVT-G041
- Fib priority: 13
- Priority: P0
- Track: counterexample-explanation
- Bundle: formal-verification-tactician/counterexamples
- Goal: Produce deterministic source-aware explanations with decoded values, expected/actual deltas, first violated condition or observation divergence, causal chain, assumptions/bounds, affected proof holes, and separately labeled repair hypotheses.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/explanation.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_explanation.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/explanation.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_explanation.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_explanation.py -q
- Acceptance: First divergence/source spans are stable; explanations cite only replay-verified facts; repair hypotheses never claim proof; redaction holds after decoding; unsupported mappings remain explicit; the stable API returns no raw payload.
- Conflict policy: Own deterministic explanation and tests; model prose may summarize the verified fact set but cannot add causes, source spans, assumptions, or authority.
- Interfaces: CounterexampleExplanation@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-explain
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["First divergence/source spans are stable","explanations cite only replay-verified facts","repair hypotheses never claim proof","redaction holds after decoding","unsupported mappings remain explicit","the stable API returns no raw payload."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G043 Deduplicate semantic witnesses and quarantine disagreement

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G012, FVT-G041
- Fib priority: 13
- Priority: P1
- Track: counterexample-equivalence
- Bundle: formal-verification-tactician/counterexamples
- Goal: Define property-specific semantic witness equivalence, diversity/coverage selection, cross-provider differential replay, and explicit disagreement quarantine.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/equivalence.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/equivalence.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py -q
- Acceptance: Syntactic variants of one witness deduplicate only under a reviewed semantic relation; materially different causal paths remain diverse; cross-provider disagreement is retained with both receipts and cannot raise authority or be reported as consensus.
- Conflict policy: Own equivalence/diversity/differential tests; do not use hashes alone as semantic equivalence or discard contradictory evidence.
- Interfaces: CounterexampleSemanticEquivalence@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-differential
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Syntactic variants of one witness deduplicate only under a reviewed semantic relation","materially different causal paths remain diverse","cross-provider disagreement is retained with both receipts and cannot raise authority or be reported as consensus."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G044 Close the loop with verifier-backed counterexample-guided synthesis

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G008, FVT-G034, FVT-G042, FVT-G043
- Fib priority: 144
- Priority: P0
- Track: cegis
- Bundle: formal-verification-tactician/counterexamples
- Goal: Implement a bounded CEGIS/CEGAR loop from verified counterexample through proof-graph/candidate refinement to exact originating-verifier rerun and auditable closure or honest continued failure.
- Evidence: ipfs_accelerate_py/agent_supervisor/proof/counterexample_guided_tactician.py, test/api/test_counterexample_guided_tactician.py
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/counterexample_guided_tactician.py, test/api/test_counterexample_guided_tactician.py
- Validation: python -m pytest test/api/test_counterexample_guided_tactician.py -q
- Acceptance: Each iteration binds prior witness, candidate, repaired tree/goal, exact verifier, budget, and result; only fresh success closes; unchanged witnesses back off; repeated failure terminates under policy; disagreement/timeout/unavailable/bound change remains open or unknown.
- Conflict policy: Own the supervisor refinement loop and tests; call canonical datasets validation/replay through providers and preserve existing formal-replanner retry/fencing semantics.
- Interfaces: CounterexampleGuidedProofDevelopment@1
- Resource class: cpu-proof-orchestrate
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Each iteration binds prior witness, candidate, repaired tree/goal, exact verifier, budget, and result","only fresh success closes","unchanged witnesses back off","repeated failure terminates under policy","disagreement/timeout/unavailable/bound change remains open or unknown."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G050 Expose stable goal-directed verification operations everywhere

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G036, FVT-G042, FVT-G044
- Fib priority: 233
- Priority: P0
- Track: public-api
- Bundle: formal-verification-tactician/provider-surface
- Goal: Add schema-equivalent Python, CLI, datasets MCP, and parent MCP operations for goal formalization, interpretation comparison, missing-proof discovery, proof planning/validation/execution/status, counterexample minimization/explanation/replay.
- Evidence: ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py, test/api/test_goal_tactician_cli_mcp_parity.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py, test/api/test_goal_tactician_cli_mcp_parity.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py test/api/test_goal_tactician_cli_mcp_parity.py -q
- Acceptance: All channels share closed requests/responses, identities, status, authority, diagnostics, redaction, bounds, cancellation, and availability; imports are side-effect free; legacy operations remain compatible; transport success never implies proof success.
- Conflict policy: Own stable public wiring and conformance tests; version additive operations and do not expose supervisor-only mutation controls through datasets APIs.
- Interfaces: GoalTacticianAPI@1, GoalTacticianCLIMCP@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-api
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["All channels share closed requests/responses, identities, status, authority, diagnostics, redaction, bounds, cancellation, and availability","imports are side-effect free","legacy operations remain compatible","transport success never implies proof success."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G051 Make proof-tactician supervisor execution restartable and fenced

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G044, FVT-G050
- Fib priority: 377
- Priority: P0
- Track: supervisor-integration
- Bundle: formal-verification-tactician/supervisor
- Goal: Persist end-goal, proof graph, candidate, verification, counterexample, closure, and completion transitions under content identities, leases, resource policy, retry bounds, exact cache keys, and restart-safe reconciliation.
- Evidence: test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py, test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Validation: python -m pytest test/api/test_goal_tactician_supervisor_lifecycle.py test/api/test_goal_tactician_supervisor_restart.py -q
- Acceptance: Restart replays identical authoritative state; stale workers/receipts cannot close or mutate a plan; cancellation/timeout/backpressure are durable; changed trees invalidate scoped work; completion requires all selected graph leaves and counterexamples to have adequate fresh receipts.
- Conflict policy: Own tactician lifecycle/restart integration and tests; reuse scheduler, proof-carrying planner, event store, leases, resources, cache, and completion authority rather than adding parallel persistence.
- Interfaces: GoalTacticianSupervisorLifecycle@1
- Resource class: cpu-supervisor
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Restart replays identical authoritative state","stale workers/receipts cannot close or mutate a plan","cancellation/timeout/backpressure are durable","changed trees invalidate scoped work","completion requires all selected graph leaves and counterexamples to have adequate fresh receipts."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G060 Certify the real multi-prover matrix in hermetic lanes

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G010, FVT-G012, FVT-G013, FVT-G043
- Fib priority: 233
- Priority: P1
- Track: real-tool-certification
- Bundle: formal-verification-tactician/real-tool-quality
- Goal: Run property-specific offline-pinned live lanes for SMT, TLA/TLC/Apalache, Datalog/SecPAL, Tamarin/ProVerif, HyperLTL tools, ATP, Hammer, Lean/Rocq/Isabelle, runtime MTL, and attestation verification.
- Evidence: tools/logic/certify_formal_verification_toolchains.py, test/integration/test_formal_verification_real_tool_matrix.py
- Outputs: tools/logic/certify_formal_verification_toolchains.py, test/integration/test_formal_verification_real_tool_matrix.py, docs/architecture/formal_verification_toolchain_certificate.json
- Validation: python -m pytest test/integration/test_formal_verification_real_tool_matrix.py -q
- Acceptance: Available tools pass live positive/negative/mutation/replay checks with exact identities; absent/mismatched lanes are explicit skips/unavailable and block only their promotion; PATH shims are not usability; certification performs no download/network/install and quarantines disagreement.
- Conflict policy: Own certification runner, live matrix test, and certificate; do not make every optional tool mandatory for unrelated properties or conceal unavailable lanes.
- Interfaces: FormalVerificationToolchainCertificate@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-toolchain
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Available tools pass live positive/negative/mutation/replay checks with exact identities","absent/mismatched lanes are explicit skips/unavailable and block only their promotion","PATH shims are not usability","certification performs no download/network/install and quarantines disagreement."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G062 Prove soundness, privacy, and robustness adversarially

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G014, FVT-G051, FVT-G060
- Fib priority: 610
- Priority: P0
- Track: adversarial-quality
- Bundle: formal-verification-tactician/real-tool-quality
- Goal: Add property-based, fuzz, metamorphic, mutation, differential, packaging, cancellation, resource, injection, forged-identity, stale-cache, leakage, vacuity, circularity, disagreement, and restart tests across the complete workflow.
- Evidence: test/security/test_formal_verification_tactician_adversarial.py, ipfs_datasets_py/tests/security/logic/test_goal_tactician_adversarial.py
- Outputs: test/security/test_formal_verification_tactician_adversarial.py, ipfs_datasets_py/tests/security/logic/test_goal_tactician_adversarial.py
- Validation: python -m pytest test/security/test_formal_verification_tactician_adversarial.py ipfs_datasets_py/tests/security/logic/test_goal_tactician_adversarial.py -q
- Acceptance: False proof, false closure, authority escalation, hidden assumption, vacuous proof, circular lemma, forged receipt, stale identity, secret/private-witness leak, unbounded process, and unresolved disagreement reported as success are hard-zero failures; fuzz inputs remain bounded and fail closed.
- Conflict policy: Own new cross-layer adversarial suites; fix production defects in the owning leaf module and preserve unrelated user changes.
- Interfaces: FormalVerificationTacticianAdversarialGate@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-security
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["False proof, false closure, authority escalation, hidden assumption, vacuous proof, circular lemma, forged receipt, stale identity, secret/private-witness leak, unbounded process, and unresolved disagreement reported as success are hard-zero failures","fuzz inputs remain bounded and fail closed."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G063 Benchmark quality, resources, cache behavior, and observability

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G062
- Fib priority: 987
- Priority: P1
- Track: metrics-benchmark
- Bundle: formal-verification-tactician/real-tool-quality
- Goal: Measure actual formalization, proof-gap recall/precision, plan solvability, proof authority, counterexample replay/reduction/explanation, provider agreement, resources, cancellation, cache correctness, and supervisor progress from run receipts.
- Evidence: test/benchmarks/test_formal_verification_tactician_benchmark.py, docs/architecture/formal_verification_tactician_benchmark.json
- Outputs: test/benchmarks/test_formal_verification_tactician_benchmark.py, docs/architecture/formal_verification_tactician_benchmark.json, ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_metrics.py
- Validation: python -m pytest test/benchmarks/test_formal_verification_tactician_benchmark.py -q
- Acceptance: Metrics are derived from actual cohort receipts, not synthetic distributions; hard correctness/privacy/authority gates are 100 percent; timing is observational unless calibrated; cache hits preserve authority and exact identity; progress exposes unresolved holes, witnesses, critical path, budgets, and next actions.
- Conflict policy: Own benchmark, report, and tactician metrics; do not turn unstable timing ratios or tool availability into correctness gates.
- Interfaces: GoalTacticianBenchmark@1, GoalTacticianMetrics@1
- Resource class: cpu-benchmark
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Metrics are derived from actual cohort receipts, not synthetic distributions","hard correctness/privacy/authority gates are 100 percent","timing is observational unless calibrated","cache hits preserve authority and exact identity","progress exposes unresolved holes, witnesses, critical path, budgets, and next actions."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G070 Document operation, migration, evidence, and failure handling

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G050, FVT-G051, FVT-G063
- Fib priority: 1597
- Priority: P1
- Track: documentation
- Bundle: formal-verification-tactician/release
- Goal: Publish architecture, API/CLI/MCP examples, proof-authority interpretation, end-goal authoring, missing-proof review, counterexample replay, provider/toolchain setup, supervisor operations, incident response, and migration guidance.
- Evidence: docs/formal_verification_tactician.md, docs/operations/formal_verification_tactician_runbook.md
- Outputs: docs/formal_verification_tactician.md, docs/operations/formal_verification_tactician_runbook.md, ipfs_datasets_py/docs/logic/proof_tactician_migration.md
- Validation: python scripts/docs/check_agent_supervisor_docs.py && python -m pytest test/api/test_formal_verification_tactician_docs.py -q
- Acceptance: Docs clearly distinguish legal evidence routing from formal proof planning, proposals from proofs, bounded checks from theorem proof, implementation completeness from deployment certification, assumptions from obligations, and every failure/rollback state; examples are executable.
- Conflict policy: Own new tactician/readiness docs and documentation tests; preserve legacy public names through documented compatibility aliases and do not promise unsupported languages/tools.
- Interfaces: FormalVerificationTacticianDocumentation@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-docs
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Docs clearly distinguish legal evidence routing from formal proof planning, proposals from proofs, bounded checks from theorem proof, implementation completeness from deployment certification, assumptions from obligations, and every failure/rollback state","examples are executable."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G080 Define property-specific rollout, promotion, and rollback

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G063, FVT-G070
- Fib priority: 2584
- Priority: P0
- Track: rollout
- Bundle: formal-verification-tactician/release
- Goal: Promote goal-directed formalization, proof-gap proposals, validated plans, and counterexample-guided repair through off, shadow, assist, auto-safe, and property/provider-specific enforcement with automatic quarantine and rollback.
- Evidence: docs/architecture/formal_verification_tactician_rollout.md, test/api/test_formal_verification_tactician_rollout.py
- Outputs: docs/architecture/formal_verification_tactician_rollout.md, test/api/test_formal_verification_tactician_rollout.py
- Validation: python -m pytest test/api/test_formal_verification_tactician_rollout.py -q
- Acceptance: Gates consume actual conformance/benchmark/toolchain receipts; auto-safe admits only allowlisted independently validated steps; false proof/closure, leakage, binding mismatch, authority escalation, or unresolved disagreement triggers quarantine and rollback; unsupported/unavailable lanes remain disclosed.
- Conflict policy: Own tactician rollout policy and tests; do not globally enforce a provider or property based on aggregate success.
- Interfaces: FormalVerificationTacticianRollout@1
- Resource class: cpu-policy
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Gates consume actual conformance/benchmark/toolchain receipts","auto-safe admits only allowlisted independently validated steps","false proof/closure, leakage, binding mismatch, authority escalation, or unresolved disagreement triggers quarantine and rollback","unsupported/unavailable lanes remain disclosed."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## FVT-G090 Issue the final implementation and deployment-readiness receipts

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G080
- Fib priority: 4181
- Priority: P0
- Track: completion
- Bundle: formal-verification-tactician/release
- Goal: Recompute current-tree implementation completion and machine-specific deployment certification, bind every child artifact and receipt, and disclose all remaining bounds, unsupported semantics, unavailable tools, publication gates, and assurance ceilings.
- Evidence: tools/logic/build_formal_verification_tactician_receipt.py, test/api/test_formal_verification_tactician_readiness_completion.py
- Outputs: tools/logic/build_formal_verification_tactician_receipt.py, test/api/test_formal_verification_tactician_readiness_completion.py, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json
- Validation: python -m pytest test/api/test_formal_verification_tactician_readiness_completion.py -q
- Acceptance: Separate implementation and deployment sections bind parent tree, datasets gitlink and publication alignment, schemas, corpus, live/simulated/skipped tests, exact tools, public operations, metrics, rollout, all child receipts, and hard-zero false-proof/false-closure/leakage/authority/disagreement gates; no hardcoded success counters.
- Conflict policy: Own receipt builder, completion test, and generated receipt; generate only from a clean current tree and immutable evidence, never edit source evidence to make the gate pass.
- Interfaces: FormalVerificationTacticianCompletionReceipt@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Separate implementation and deployment sections bind parent tree, datasets gitlink and publication alignment, schemas, corpus, live/simulated/skipped tests, exact tools, public operations, metrics, rollout, all child receipts, and hard-zero false-proof/false-closure/leakage/authority/disagreement gates","no hardcoded success counters."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []
