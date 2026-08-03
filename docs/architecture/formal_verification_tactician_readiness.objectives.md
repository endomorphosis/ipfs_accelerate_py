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
- Lean, the in-process Runtime MTL monitor, and the in-process Datalog/SecPAL
  authorization engines are usable today, but remain below production
  semantic certification until real positive, negative, mutation, and replay
  evidence is complete and bound to their exact identities.
- Every remaining unavailable external prover or checker in the declared
  matrix is open capability-expansion work. Its lane must gain a reviewed,
  explicitly invoked installation path and semantic certification; mere PATH,
  package, source, support-runtime, or advisor presence cannot close the work.
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

- Status: active
- Parent:
- Depends on: FVT-G200
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
- Resource class: cpu-medium
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

## FVT-G100 Define role-aware toolchain authority and promotion

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G060
- Fib priority: 6765
- Priority: P0
- Track: toolchain-governance
- Bundle: formal-verification-tactician/toolchain-governance
- Goal: Replace availability-shaped promotion with a closed per-tool role model and split the monolithic certificate runner into independently owned semantic lanes.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, test/api/test_formal_verification_toolchain_roles.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certification/roles.py, test/api/test_formal_verification_toolchain_roles.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_verification_toolchain_security.py test/api/test_formal_verification_toolchain_roles.py test/integration/test_formal_verification_real_tool_matrix.py -q
- Acceptance: Every matrix entry has exactly one closed role and authority ceiling; Java, Maude, and OPAM are support only; Leanstral, autoencoder, SymAI, ErgoAI, and Hammer are advisor/candidate only until independent reconstruction; external Souffle/SecPAL are shadow checkers; in-process Datalog/SecPAL have authorization-only authority; Runtime MTL has finite-trace authority; state and hyperproperty tools have bounded authority; Lean/Rocq/Isabelle have kernel authority; ZKP has attestation authority only; support, advisor, or shadow presence alone can never satisfy a certified-authority requirement.
- Conflict policy: Own the canonical role schema, lane registration, and authority-boundary tests; pre-register per-lane handlers so later tasks do not concurrently edit the central certifier or generated certificate.
- Interfaces: FormalVerificationToolRole@1, RoleAwarePromotionPolicy@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-policy

## FVT-G101 Semantically certify the pinned Lean kernel

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G100
- Fib priority: 10946
- Priority: P0
- Track: semantic-reference
- Bundle: formal-verification-tactician/lean-certification
- Goal: Promote the already usable locked Lean toolchain only after real kernel semantics, rejection behavior, and deterministic replay are certified.
- Evidence: tools/logic/certification/lean.py, test/integration/toolchains/test_lean_semantic_certification.py
- Outputs: tools/logic/certification/lean.py, test/fixtures/formal_verification/toolchains/lean/manifest.json, test/integration/toolchains/test_lean_semantic_certification.py
- Validation: ELAN_TOOLCHAIN=leanprover/lean4:v4.31.0 ELAN_NO_AUTO_INSTALL=1 python -m pytest test/integration/toolchains/test_lean_semantic_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k lean -q
- Acceptance: Exact Lean v4.31.0 compiles a true theorem, rejects false and malformed proofs, rejects hypothesis/conclusion mutations, and replays deterministically; imports, source tree, theorem, assumptions, toolchain, and output are bound; sorry, admit, unsafe escape, shim mismatch, install, download, and network use fail closed; the resulting authority is kernel proof checking only.
- Conflict policy: Own the Lean lane handler, corpus, and focused test; do not edit the central certificate or select/download a different Elan toolchain.
- Interfaces: LeanSemanticCertification@1
- Resource class: cpu-proof-type-check

## FVT-G102 Semantically certify reference Datalog and SecPAL authorization

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G100
- Fib priority: 10946
- Priority: P0
- Track: semantic-reference
- Bundle: formal-verification-tactician/authorization-certification
- Goal: Promote the already usable in-process Datalog and SecPAL-style engines only after full authorization semantics are certified.
- Evidence: tools/logic/certification/authorization.py, test/integration/toolchains/test_authorization_semantic_certification.py
- Outputs: tools/logic/certification/authorization.py, test/fixtures/formal_verification/toolchains/authorization/manifest.json, test/integration/toolchains/test_authorization_semantic_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_authorization_backends.py test/integration/toolchains/test_authorization_semantic_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'authorization or datalog or secpal' -q
- Acceptance: Both in-process engines exercise allow, deny, unknown, conflict, scoped delegation, revocation, negative and malformed inputs; rule, principal, scope, and delegation mutations change or quarantine the verdict; counterexamples replay deterministically; receipts bind the exact policy and engine; certification grants authorization-decision authority, never theorem authority.
- Conflict policy: Own the reference authorization lane and corpus; do not install external shadows or edit the central certificate.
- Interfaces: AuthorizationSemanticCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G103 Semantically certify finite-trace Runtime MTL

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G100
- Fib priority: 10946
- Priority: P0
- Track: semantic-reference
- Bundle: formal-verification-tactician/runtime-mtl-certification
- Goal: Promote the already usable in-process Runtime MTL monitor only after interval, event, violation, and replay semantics are certified across supported surfaces.
- Evidence: tools/logic/certification/runtime_mtl.py, test/integration/toolchains/test_runtime_mtl_semantic_certification.py
- Outputs: tools/logic/certification/runtime_mtl.py, test/fixtures/formal_verification/toolchains/runtime_mtl/manifest.json, test/integration/toolchains/test_runtime_mtl_semantic_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py test/integration/toolchains/test_runtime_mtl_semantic_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'runtime_mtl or mtl' -q
- Acceptance: Live satisfied and violated traces, interval and event mutations, shortest violating-prefix replay, timestamp boundaries, malformed traces, and Python/TypeScript golden parity pass; receipts bind formula, trace, clock policy, bounds, implementation, and source tree; a clean finite prefix never becomes an unbounded theorem.
- Conflict policy: Own the in-process Runtime MTL lane, golden corpus, and focused test; do not install the external parity checker or edit the central certificate.
- Interfaces: RuntimeMTLSemanticCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G110 Replace declared external-tool gaps with reviewed deployment locks

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G100
- Fib priority: 17711
- Priority: P0
- Track: toolchain-governance
- Bundle: formal-verification-tactician/toolchain-locks
- Goal: Turn every remaining declared installation gap or incomplete managed pin into a reviewed, licensed, per-platform, explicitly invoked deployment contract.
- Evidence: config/formal_verification_toolchains.lock.json, test/packaging/test_formal_verification_external_tool_locks.py
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, test/packaging/test_formal_verification_external_tool_locks.py
- Validation: python -m pytest test/packaging/test_formal_verification_external_tool_locks.py ipfs_datasets_py/tests/integration/logic/test_verification_toolchain_security.py -k 'pin or checksum or gap or install' -q
- Acceptance: TLC, HyperLTL/AutoHyper/MCHyper, Souffle/SecPAL, external Runtime MTL, Vampire, Lean, Rocq, Isabelle, OPAM, SymbolicAI, and ErgoAI have reviewed version/license/platform/source/checksum or immutable package-lock identities and installer entries; ZKP has a secret-safe deployment-artifact schema; unsupported platforms fail explicitly; installs are user-local and require explicit opt-in; imports, discovery, tests, and offline certification never install, download, access the network, or mutate a system package manager.
- Conflict policy: Sole owner for the shared lock and installer registry; add per-family installer plugins for downstream tasks and do not install tools as part of this metadata task.
- Interfaces: FormalVerificationDeploymentLock@2
- Submodules: ipfs_datasets_py
- Resource class: cpu-install-test

## FVT-G120 Install and certify TLC and Apalache state-model checking

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/state-model-toolchains
- Goal: Complete the pinned user-local installation and semantic certification of TLC and Apalache for distributed workflows and state machines.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py test/integration/toolchains/test_state_model_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'tlc or apalache or state_model' -q
- Acceptance: Explicit strict installation makes both exact tools usable; invariant-holds, violation trace, mutated Next/invariant, replay, malformed model, timeout, and bound behavior pass; model/config/constants/bounds/tool identities are bound; Java remains support only and bounded model-checking never promotes to theorem authority.
- Conflict policy: Own the state-model installer plugin, handler, and test; consume the shared lock without editing it or the central certificate.
- Interfaces: StateModelToolchainCertification@1
- Submodules: ipfs_datasets_py
- Resource class: jvm-proof-solver

## FVT-G130 Install and certify Tamarin with Maude

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/tamarin-toolchain
- Goal: Complete the exact Tamarin and compatible Maude installation and certify cryptographic-protocol claims and attacks.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, tools/logic/certification/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py test/integration/toolchains/test_tamarin_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'tamarin or maude' -q
- Acceptance: Explicit strict installation selects Tamarin 1.12.0 and Maude 3.5.1; secure, attack, mutated claim/rule, replay, malformed output, timeout, and version mismatch cases pass; theory, claims, bounds, and exact binaries are bound; Maude is support only and cannot promote a property lane by itself.
- Conflict policy: Own the Tamarin/Maude installer plugin, handler, and test; do not edit the ProVerif lane, shared lock, or central certificate.
- Interfaces: TamarinToolchainCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## FVT-G131 Install and certify ProVerif in isolated OPAM

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/proverif-toolchain
- Goal: Complete an isolated pinned OPAM/ProVerif deployment and semantic protocol certification without mutating global switches.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, tools/logic/certification/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py test/integration/toolchains/test_proverif_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'proverif or opam' -q
- Acceptance: Explicit strict installation selects OPAM 2.5.2 support and ProVerif 2.05 in a repository-local isolated root; secure, attack, mutation, replay, malformed output, cancellation, and mismatch checks pass; model and claim identities bind receipts; OPAM alone has no semantic authority.
- Conflict policy: Own the ProVerif installer plugin, handler, isolated root contract, and test; serialize OPAM resource use with Rocq and never modify a global OPAM switch.
- Interfaces: ProVerifToolchainCertification@1
- Submodules: ipfs_datasets_py
- Resource class: exclusive-opam-toolchain

## FVT-G140 Install and certify Vampire and E ATP

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P1
- Track: external-capability
- Bundle: formal-verification-tactician/atp-toolchains
- Goal: Complete exact Vampire and E prover installation and certify theorem/non-theorem behavior for premise and proof search.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, tools/logic/certification/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/unit_tests/logic/CEC/provers/test_vampire_eprover.py test/integration/toolchains/test_atp_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'vampire or eprover or atp' -q
- Acceptance: Explicit strict installation selects Vampire 5.0.1 and E 3.2.5; theorem, non-theorem, premise/conclusion mutation, proof-output binding, replay, malformed output, and timeout checks pass; ATP results remain candidates unless an allowed independent kernel reconstruction validates them.
- Conflict policy: Own ATP installer plugins, handler, and test; do not edit CEC semantics, shared lock, or central certificate.
- Interfaces: ATPToolchainCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## FVT-G150 Install and semantically certify Rocq

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/rocq-toolchain
- Goal: Complete isolated installation and real kernel certification for the locked Rocq/Coq provider.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, tools/logic/certification/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_kernel_backends.py test/integration/toolchains/test_rocq_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'rocq or coq' -q
- Acceptance: Explicit strict installation selects Rocq 9.1.1 in an isolated pinned OPAM root; true proof, false proof, hypothesis/conclusion mutation, deterministic replay, forbidden admits/axiom escapes, malformed input, and mismatch checks pass; receipts bind imports, source, theorem, assumptions, and exact kernel identity.
- Conflict policy: Own the Rocq installer plugin, handler, and test; serialize OPAM resource use with ProVerif and never modify a global switch.
- Interfaces: RocqToolchainCertification@1
- Submodules: ipfs_datasets_py
- Resource class: exclusive-opam-toolchain

## FVT-G151 Install and semantically certify Isabelle

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P1
- Track: external-capability
- Bundle: formal-verification-tactician/isabelle-toolchain
- Goal: Complete the pinned Isabelle installation and real session/kernel certification used for reconstruction and Hammer validation.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_kernel_backends.py test/integration/toolchains/test_isabelle_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k isabelle -q
- Acceptance: Explicit strict installation selects Isabelle2025-2; a checked theory/session passes while bad proof, mutated assumptions/conclusion, replay mismatch, malformed output, timeout, and wrong installation fail; theory heap, session, imports, source, property, and exact tool identity are bound; Hammer remains proposal-only until kernel reconstruction.
- Conflict policy: Own the Isabelle installer plugin, handler, and test; observe an explicit large-download/storage budget and do not edit the shared lock or central certificate.
- Interfaces: IsabelleToolchainCertification@1
- Submodules: ipfs_datasets_py
- Resource class: large-kernel-toolchain

## FVT-G160 Install and role-certify SymAI, ErgoAI, Leanstral, autoencoder, and Hammer advisors

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P1
- Track: advisor-capability
- Bundle: formal-verification-tactician/advisor-toolchains
- Goal: Complete missing SymAI and ErgoAI deployment support and certify every existing advisor utility as bounded candidate generation rather than semantic proof authority.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py test/integration/toolchains/test_advisor_role_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'advisor or symbolicai or ergoai or leanstral or hammer or autoencoder' -q
- Acceptance: Explicit strict installation selects locked SymAI and ErgoAI identities where supported; SymAI, ErgoAI, Leanstral, autoencoder, and Hammer proposals are bounded, sanitized, source-bound, deterministic or replay-bound, cache-safe, and failure-explicit; no confidence, similarity, generated text, or advisor availability becomes proof without deterministic compilation and independent solver/kernel validation.
- Conflict policy: Own advisor installer plugins, role handler, and test; reuse existing adapters and caches without changing model runtimes or central certificate generation.
- Interfaces: AdvisorRoleCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## FVT-G170 Install and certify HyperLTL, AutoHyper, and MCHyper

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/hyperproperty-toolchains
- Goal: Replace the hyperproperty declared gap with pinned external engines and bounded information-flow semantic certification.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, tools/logic/certification/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_hyperproperty_backends.py test/integration/toolchains/test_hyperproperty_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'hyperltl or autohyper or mchyper or hyperproperty' -q
- Acceptance: Explicit strict installation selects reviewed HyperLTL, AutoHyper, and MCHyper artifacts; quantifiers and observation projections are preserved; satisfaction, violating trace tuples, semantic mutations, replay, malformed output, disagreement, timeout, and exact bounds pass; results retain their declared bounded hyperproperty authority and cannot make universal claims beyond bounds.
- Conflict policy: Own hyperproperty installer plugins, handler, fixtures, and test; do not edit shared lock or central certificate.
- Interfaces: HyperpropertyToolchainCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## FVT-G180 Install external Datalog and SecPAL differential shadows

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G102, FVT-G110
- Fib priority: 46368
- Priority: P1
- Track: external-capability
- Bundle: formal-verification-tactician/authorization-toolchains
- Goal: Replace the external authorization gap with pinned Souffle/SecPAL-compatible shadows and differential disagreement handling.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_authorization_backends.py test/integration/toolchains/test_external_authorization_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'souffle or secpal or authorization' -q
- Acceptance: Explicit strict installation selects exact external engines; the allow/deny/unknown/conflict/delegation corpus, rule/scope mutation, replay, malformed output, timeout, and differential comparison pass; any disagreement quarantines promotion; external engines remain shadows while the certified in-process references retain authorization authority.
- Conflict policy: Own external authorization installer plugins, differential handler, and test; do not weaken or edit the in-process reference semantics.
- Interfaces: ExternalAuthorizationShadowCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G181 Install and certify external Runtime MTL parity

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G103, FVT-G110
- Fib priority: 46368
- Priority: P1
- Track: external-capability
- Bundle: formal-verification-tactician/runtime-monitor-toolchains
- Goal: Replace the external Runtime MTL gap with a pinned parity engine and cross-runtime semantic disagreement checks.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py test/integration/toolchains/test_external_runtime_mtl_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'runtime_mtl or mtl' -q
- Acceptance: Explicit strict installation selects an exact external monitor; Python, TypeScript, and external implementations agree on satisfied/violated golden traces, boundary intervals, mutations, shortest-prefix replay, malformed input, and bounds or quarantine disagreement; finite-trace authority is preserved and no global correctness claim is inferred.
- Conflict policy: Own the external monitor installer plugin, parity handler, and test; do not edit the in-process semantic reference lane.
- Interfaces: ExternalRuntimeMTLCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G190 Bind and certify a production ZKP circuit deployment

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110
- Fib priority: 28657
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/zkp-attestation-toolchain
- Goal: Replace the production-circuit gap with a reviewed, secret-safe deployment binding and live verifier attestation certification.
- Evidence: config/formal_verification_zkp_deployment.lock.json, test/integration/toolchains/test_zkp_deployment_certification.py
- Outputs: config/formal_verification_zkp_deployment.lock.json, tools/logic/certification/zkp.py, test/integration/toolchains/test_zkp_deployment_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/api/test_agent_supervisor_program_analysis_zkp_conformance.py ipfs_datasets_py/tests/integration/logic/test_proof_receipt_attestation.py test/integration/toolchains/test_zkp_deployment_certification.py -q
- Acceptance: Circuit, ceremony, proving-key and verification-key digests, public-input schema, backend, expiry, freshness, and revocation are exact and reviewable; live positive verification and corrupted proof/key/public-input, circuit mismatch, mutation, replay, stale, and revoked cases pass; private witnesses and secrets never enter Git, logs, caches, public receipts, or model context; ZKP authority attests an underlying receipt and never replaces semantic theorem authority.
- Conflict policy: Own the deployment binding, ZKP handler, and test; reference private artifacts only by digest and configured secret-safe location.
- Interfaces: ZKPDeploymentCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## FVT-G201 Derive exact host support and platform exceptions from the lock

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G010, FVT-G110
- Fib priority: 121393
- Priority: P0
- Track: deployment-integrity
- Bundle: formal-verification-tactician/platform-support-classifier
- Goal: Give every locked tool an auditable host-platform classification so missing supported capabilities can never be relabeled as exceptions.
- Evidence: test/integration/toolchains/test_formal_verification_platform_support.py
- Outputs: tools/logic/certification/platform_support.py, test/integration/toolchains/test_formal_verification_platform_support.py
- Validation: python -m pytest test/integration/toolchains/test_formal_verification_platform_support.py -q
- Acceptance: The normalized host key is derived from the running OS and architecture; each tool reports supported_here, unsupported_here, or ambiguous from its own pins and deployment contract; `any` support is honored; absent, contradictory, or ambiguous metadata is a blocker; only an explicit host exclusion can produce a narrow platform exception; linux-aarch64 classifies HyperLTL, AutoHyper, MCHyper, Souffle, and external Runtime MTL as supported under the current lock, external SecPAL as unsupported, and ZKP as a platform-independent deployment binding; a lock mutation that adds or removes linux-aarch64 changes the classification and final digest.
- Conflict policy: Own platform normalization and classification only; never probe or install tools, infer support from PATH, or convert unavailability into unsupported status.
- Interfaces: FormalVerificationPlatformSupport@1
- Resource class: cpu-validation

## FVT-G202 Repair exact probes and managed artifact identities

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G110, FVT-G120, FVT-G131, FVT-G151
- Fib priority: 121393
- Priority: P0
- Track: dependency-integrity
- Bundle: formal-verification-tactician/toolchain-probe-integrity
- Goal: Make generic and state-model identity probing command-correct, return-code-aware, digest-bound, hostile-environment-safe, and atomic.
- Evidence: test/integration/toolchains/test_formal_verification_probe_integrity.py
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_probe_integrity.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_formal_verification_probe_integrity.py test/integration/toolchains/test_state_model_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -q
- Acceptance: Java identity is parsed only from the quoted java/openjdk version banner after hostile Java option variables are neutralized; bare names resolve only through PATH and dry-run executes nothing; Apalache uses `version`, Isabelle uses `version`, ProVerif uses a valid identity command, and nonzero error banners cannot prove usability; TLC 1.8.0 binds SHA-256 `e22f8ffb4bacdea0a871f444dd94fe5fb0d8013b3388ae39e82e26f852c735d5` plus manifest tag `v1.8.0` and revision `30cc360`; genuine TLC help is recognized despite exit 1 only with required markers; returned launchers execute through the validated Java 17+ runtime; TLC and Apalache artifact plus launcher repair is staged, atomic, and rollback-safe; failed repair preserves a prior good install.
- Conflict policy: Own exact probe commands, reviewed identities, and atomic publication; never mutate system Java, accept an unbound artifact, trust arbitrary nonempty output, or validate a different path than the public launcher.
- Interfaces: FormalVerificationProbeIntegrity@1
- Submodules: ipfs_datasets_py
- Resource class: exclusive-jvm-toolchain

## FVT-G203 Aggregate full specialized receipts with composite lane handlers

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G101, FVT-G102, FVT-G103, FVT-G120, FVT-G130, FVT-G131, FVT-G140, FVT-G150, FVT-G151, FVT-G160, FVT-G170, FVT-G180, FVT-G181, FVT-G190, FVT-G201, FVT-G202
- Fib priority: 196418
- Priority: P0
- Track: certification-integrity
- Bundle: formal-verification-tactician/semantic-receipt-aggregation
- Goal: Replace first-check and one-handler-per-lane fan-in with lossless, per-tool specialized evidence aggregation.
- Evidence: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py test/integration/test_formal_verification_real_tool_matrix.py -q
- Acceptance: Handlers are keyed by `(lane_id, tool_id)` or a composite lane returns distinct per-tool receipts; kernel retains Lean, Rocq, and Isabelle evidence and protocol retains Tamarin and ProVerif evidence; state, protocol, kernel, ATP, hyperproperty, advisor, in-process and external authorization, in-process and external Runtime MTL, and ZKP certifiers are all represented; every check, case, binding, executable, artifact, dependency, source, authority ceiling, and raw receipt digest participates in the top-level digest; a second failed check of an already-present kind blocks promotion; mutating any retained check or identity changes the certificate digest.
- Conflict policy: Own role registration and lossless aggregation; do not run installers, collapse by check kind, discard raw receipt identity, or let one tool overwrite a sibling handler.
- Interfaces: FormalVerificationSpecializedReceiptAggregation@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G204 Execute real TLC and Apalache state-model semantics

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G120, FVT-G201, FVT-G202
- Fib priority: 75025
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/state-model-live-semantics
- Goal: Replace classifier-backed state-model promotion with real TLC and Apalache execution against positive and adversarial models.
- Evidence: test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- Outputs: tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_state_model_live_semantic_certification.py test/integration/toolchains/test_state_model_toolchain_certification.py -q
- Acceptance: The pinned TLC jar and Apalache executable each run a valid invariant model, a violating model with concrete counterexample, specification and invariant mutations, deterministic replay, malformed input, timeout, and bounded-state/resource cases; source model, property, bound, JVM, executable, jar/archive, and output digests are exact; canned text and parser classification remain `hermetic_parser` and cannot satisfy live external semantics.
- Conflict policy: Own live state-model cases and receipts; use the installed toolchain without downloading during certification and never promote from identity or output parsing alone.
- Interfaces: StateModelLiveSemanticCertification@1
- Resource class: exclusive-jvm-toolchain

## FVT-G205 Execute real Tamarin and ProVerif protocol semantics

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G130, FVT-G131, FVT-G201, FVT-G202
- Fib priority: 75025
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/protocol-live-semantics
- Goal: Run the protocol corpus through the pinned Tamarin and ProVerif binaries and retain per-tool cryptographic-protocol evidence.
- Evidence: test/integration/toolchains/test_protocol_live_semantic_certification.py, docs/architecture/formal_verification_protocol_live_certificate.json
- Outputs: tools/logic/certification/tamarin.py, tools/logic/certification/proverif.py, test/integration/toolchains/test_protocol_live_semantic_certification.py, docs/architecture/formal_verification_protocol_live_certificate.json
- Validation: python -m pytest test/integration/toolchains/test_protocol_live_semantic_certification.py test/integration/toolchains/test_tamarin_toolchain_certification.py test/integration/toolchains/test_proverif_toolchain_certification.py -q
- Acceptance: Both binaries execute valid secrecy/authentication protocols, concrete attacks, premise/conclusion and protocol mutations, replay, malformed models, timeout, disagreement, and bounded-search cases; receipts bind tool and dependency identities, source, query, assumptions, bound, witnesses/traces, and raw output; parser fixtures remain non-production.
- Conflict policy: Own live protocol execution and per-tool receipts; never turn parser-recognized canned output into semantic proof or allow one protocol engine to stand in for the other.
- Interfaces: ProtocolLiveSemanticCertification@1
- Resource class: cpu-proof-solver

## FVT-G206 Execute and bind Lean, Rocq, and Isabelle kernel semantics

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G101, FVT-G150, FVT-G151, FVT-G201, FVT-G202
- Fib priority: 75025
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/kernel-live-semantics
- Goal: Require each installed proof kernel to check its own generated source and retain all assumptions, imports, theorem, and mutation evidence.
- Evidence: test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- Outputs: tools/logic/certification/lean.py, tools/logic/certification/rocq.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- Validation: python -m pytest test/integration/toolchains/test_kernel_live_semantic_fanin.py test/integration/toolchains/test_lean_semantic_certification.py test/integration/toolchains/test_rocq_toolchain_certification.py test/integration/toolchains/test_isabelle_toolchain_certification.py -q
- Acceptance: Lean, Rocq, and Isabelle independently execute a valid theorem, false theorem, hypothesis/conclusion mutation, deterministic replay, malformed source, timeout, and forbidden admit/axiom-oracle checks; Isabelle's live source/session helper is exercised rather than only offline fixtures; receipts bind exact kernel, dependency, source, imports/session, assumptions, theorem, and output digests; no advisor or sibling kernel substitutes for the selected kernel.
- Conflict policy: Own kernel fan-in and live source checks; serialize expensive OPAM/Isabelle resources and preserve each kernel's separate authority.
- Interfaces: KernelLiveSemanticFanIn@1
- Resource class: large-kernel-toolchain

## FVT-G207 Execute real Vampire and E prover semantics

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G140, FVT-G201, FVT-G202
- Fib priority: 46368
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/atp-live-semantics
- Goal: Replace SZS parser fixtures with real pinned ATP runs while preserving reconstruction and kernel-checking ceilings.
- Evidence: test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- Outputs: tools/logic/certification/atp.py, test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- Validation: python -m pytest test/integration/toolchains/test_atp_live_semantic_certification.py test/integration/toolchains/test_atp_toolchain_certification.py -q
- Acceptance: Vampire and E each execute theorem and counter-satisfiable problems, premise/conclusion mutations, replay, malformed TPTP, timeout/resource bounds, disagreement, and proof-object/reconstruction cases; receipts bind exact binary and artifact digests, TPTP source, assumptions, conclusion, limits, raw SZS output, and reconstruction status; an ATP result cannot exceed candidate/reconstruction authority until checked by a trusted kernel.
- Conflict policy: Own real ATP execution and receipts; keep SZS parsing as adapter evidence and never grant kernel authority to an unreconstructed ATP result.
- Interfaces: ATPLiveSemanticCertification@1
- Resource class: cpu-proof-solver

## FVT-G208 Install and live-certify supported hyperproperty engines

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G170, FVT-G201, FVT-G202
- Fib priority: 121393
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/hyperproperty-vendor-toolchains
- Goal: Correct the upstream identities and deploy real HyperLTL satisfiability, AutoHyper, and MCHyper toolchains on every declared supported host.
- Evidence: test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py, docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, tools/logic/certification/hyperproperty.py, test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py, docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py test/integration/toolchains/test_hyperproperty_toolchain_certification.py -q
- Acceptance: AutoHyper binds its official revision, .NET runtime, Spot tools, build inputs, executable digest, and live semantic cases; MCHyper binds its official revision, ABC/AIGER dependencies, executable digest, supported fragment, and live witness/counterexample cases; the selected HyperLTL satisfiability engine has its own correct upstream identity and decidable-fragment ceiling; satisfaction, violation, observation/quantifier mutation, replay, malformed output, timeout, disagreement, and exact bounds execute through real binaries; linux-aarch64 remains supported only if that complete chain is real; case-oracle, hermetic shim, fixture, parser, or canned output cannot satisfy this goal.
- Conflict policy: Own vendor acquisition, correct per-product pins, dependencies, and live adapters; preserve bounded authority and never relabel the existing hermetic engines.
- Interfaces: HyperpropertyVendorToolchainCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## FVT-G209 Install Souffle and derive the SecPAL platform exception

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G102, FVT-G180, FVT-G201, FVT-G202
- Fib priority: 75025
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/authorization-vendor-toolchains
- Goal: Replace the Souffle case-oracle shadow with a checksummed vendor build and keep external SecPAL support classification lock-derived.
- Evidence: test/integration/toolchains/test_external_authorization_vendor_certification.py, docs/architecture/formal_verification_authorization_vendor_install_receipt.json
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_external_authorization_vendor_certification.py, docs/architecture/formal_verification_authorization_vendor_install_receipt.json
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_external_authorization_vendor_certification.py test/integration/toolchains/test_external_authorization_toolchain_certification.py -q
- Acceptance: Souffle 2.4.1 source/archive and build dependencies are immutable and checksummed, the user-local executable and artifact digest are exact, and real allow/deny/unknown/conflict/delegation plus rule/scope mutation, replay, malformed, timeout, and disagreement cases execute through it; linux-aarch64 is supported for Souffle; external SecPAL is a narrow unsupported-platform exception on linux-aarch64 under the current contract and never counts as installed, complete, authoritative, or production-certified; hermetic shadows remain differential-only.
- Conflict policy: Own vendor Souffle installation and external authorization production evidence; never mutate the system package manager, promote a shadow, or excuse missing supported capability.
- Interfaces: ExternalAuthorizationVendorCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G210 Build and certify an independent external Runtime MTL engine

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G103, FVT-G181, FVT-G201, FVT-G202
- Fib priority: 75025
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/runtime-mtl-external-runtime
- Goal: Replace the Python-backed parity wrapper with a reproducibly built TypeScript/Node monitor, enforce the install-versus-offline-certification boundary, and produce honest cross-runtime evidence.
- Evidence: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Outputs: ipfs_datasets_py/typescript/logic-runtime-mtl, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py test/integration/toolchains/test_external_runtime_mtl_certification.py test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py -q
- Acceptance: A locked TypeScript dependency graph builds an independent Node package/executable without importing or dispatching to the Python reference; the explicit opt-in user-local installation phase may run the locked build, but every offline semantic-certification path, including the in-process Runtime MTL parity helper, consumes only a preinstalled digest-verified artifact and never runs `npm install`, `npm ci`, `npm run build`, downloads, or network access; a missing or stale prebuilt artifact blocks certification instead of rebuilding; the authoritative private-HOME validation environment receives an explicit approved immutable deployment root rather than discovering mutable user paths; package, source, lockfile, runtime, launcher, launcher target, executable, and artifact digests are bound; positive, negative, interval/event mutation, timestamp boundary, shortest-prefix replay, malformed input, timeout, bounds, and disagreement cases execute out of process; finite-trace authority and inconclusive-prefix semantics are preserved; generated Python parity wrappers remain non-production shadow evidence.
- Conflict policy: Own TypeScript monitor, reproducible package build, installer, offline install boundary, and cross-runtime certifier; do not silently build during certification, change the Python reference semantics, or infer global proof from finite traces.
- Interfaces: ExternalRuntimeMTLVendorCertification@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G211 Integrate the live secret-safe ZKP deployment certificate

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G190, FVT-G201, FVT-G203
- Fib priority: 75025
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/zkp-production-runtime
- Goal: Distinguish schema-valid sample bindings from a live verifier and bind the complete ZKP deployment receipt into the semantic fan-in.
- Evidence: test/integration/toolchains/test_zkp_live_verifier_deployment.py, docs/architecture/formal_verification_zkp_live_deployment_receipt.json
- Outputs: config/formal_verification_zkp_deployment.lock.json, tools/logic/certification/zkp.py, test/integration/toolchains/test_zkp_live_verifier_deployment.py, docs/architecture/formal_verification_zkp_live_deployment_receipt.json
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_zkp_live_verifier_deployment.py test/integration/toolchains/test_zkp_deployment_certification.py -q
- Acceptance: The configured backend performs live verification against exact circuit, ceremony, proving-key, verification-key, public-parameter, public-input-schema, version, expiry, freshness, and revocation identities; positive and corrupted proof/key/public-input, circuit mismatch, mutation, replay, stale, and revoked cases run against it; no private witness, proving-key bytes, trapdoor, secret path, or secret value enters Git, logs, caches, public receipts, or model context; absent operator-bound public artifacts remain deployment blockers, not platform exceptions; ZKP attests and never replaces underlying semantic authority.
- Conflict policy: Own live verifier binding and secret-safe aggregation only; never generate/expose private material or manufacture deployment evidence.
- Interfaces: ZKPLiveVerifierDeployment@1
- Resource class: cpu-proof-solver

## FVT-G212 Bind durable supervisor evidence and enforce expected outputs

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G080, FVT-G201
- Fib priority: 121393
- Priority: P0
- Track: supervisor-integrity
- Bundle: formal-verification-tactician/supervisor-release-evidence
- Goal: Export a read-only, content-addressed execution snapshot and reject proposals whose declared outputs are ignored, absent, or unstaged.
- Evidence: test/api/test_agent_supervisor_release_evidence_binding.py
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py, ipfs_accelerate_py/agent_supervisor/release_evidence.py, test/api/test_agent_supervisor_release_evidence_binding.py
- Validation: python -m pytest test/api/test_agent_supervisor_release_evidence_binding.py test/api/test_agent_supervisor_todo_daemon_port.py -k 'expected_output or completion_receipt or release_evidence' -q
- Acceptance: Declared outputs are compared with filesystem, proposed paths, staged paths, and ignore rules; an exact allowed ignored output is force-added only by its explicit path or the proposal fails `expected_output_ignored_or_unstaged`; a regression proves an ignored JSON and tracked source both enter the commit; the exporter reads committed bundle/task metadata, lane manifest, scheduler snapshot, task state, event manifest/JSONL, and durable member_completion receipts once and hashes raw bytes; output binds canonical task CID/key, dependency CIDs, baseline and merged trees/gitlinks, attempt/phase, continuous event sequence, validation and merge outcomes, freshness, authority, and publication state; it never edits live state and cannot treat metrics-module presence as completion.
- Conflict policy: Own proposal-output enforcement and read-only release evidence; preserve path fences, never broadly force-add ignored files, and never synthesize a missing terminal receipt.
- Interfaces: AgentSupervisorReleaseEvidence@1
- Resource class: cpu-validation

## FVT-G213 Build the fail-closed role-aware release candidate

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G203, FVT-G204, FVT-G205, FVT-G206, FVT-G207, FVT-G208, FVT-G209, FVT-G210, FVT-G211, FVT-G212
- Fib priority: 317811
- Priority: P0
- Track: completion
- Bundle: formal-verification-tactician/toolchain-release-candidate
- Goal: Generate a role-aware release candidate from the complete supported matrix, close the exact production-semantic elevation fan-in, and do so without claiming its own future merge or deployment.
- Evidence: test/integration/test_formal_verification_role_aware_release_candidate.py, test/integration/toolchains/test_formal_verification_production_elevation_fanin.py, docs/architecture/formal_verification_role_aware_release_candidate.json, docs/architecture/formal_verification_production_elevation_fanin_receipt.json
- Outputs: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, test/integration/test_formal_verification_role_aware_release_candidate.py, test/integration/toolchains/test_formal_verification_production_elevation_fanin.py, docs/architecture/formal_verification_role_aware_release_candidate.json, docs/architecture/formal_verification_production_elevation_fanin_receipt.json, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_formal_verification_production_elevation_fanin.py test/integration/test_formal_verification_role_aware_release_candidate.py test/integration/test_formal_verification_real_tool_matrix.py -q
- Acceptance: The candidate derives host support, roles, ceilings, evidence classes, platform exceptions, blockers, offline policy, quarantine state, public surfaces, and every success boolean from bound evidence; `lean`, `runtime-mtl`, `datalog-authorization`, `secpal-authorization`, `coq`, and `isabelle` each have exact independently reconstructed positive, negative, mutation, and replay evidence before their corresponding production elevation is present; the independent TypeScript/Node Runtime MTL vendor lane is consumed through `ExternalRuntimeMTLVendorCertification@1`, binds its package, source, lockfile, Node runtime, launcher, target, executable, and artifact digests, and gates both external finite-trace authority and in-process Runtime MTL elevation on exact cross-runtime parity and disagreement handling; hermetic Runtime MTL wrappers never promote; real supported HyperLTL, AutoHyper, MCHyper, and Souffle vendor binaries and their native/runtime dependencies are distinguished from identity manifests, adapters, case oracles, fixtures, and shims; offline certification resolves only an approved immutable preinstalled deployment root under the authoritative private-HOME validation environment and never installs, builds, downloads, or accesses the network; all supported managed capabilities have their required installed and specialized semantic evidence; every raw receipt/check/case/binding and executable/artifact/dependency digest affects the canonical digest; external SecPAL is an exception only when the current lock explicitly excludes the running host; missing supported, ambiguous, stale, parser-only, fixture, canned, hermetic, advisor, shadow, identity-only, dependency-only, or incomplete evidence blocks readiness at its correct ceiling; the checked-in candidate binds an explicit certified source commit/tree and cannot exceed `release_candidate` before its merge event exists.
- Generated artifacts: docs/architecture/formal_verification_tactician_readiness_completion_receipt.json
- Allowed paths: docs/architecture/formal_verification_tactician_readiness_completion_receipt.json, tools/logic/certification/public_evidence.py
- Conflict policy: Sole owner of the central candidate and production-elevation fan-in after its semantic/vendor dependencies merge; never install during offline certification, treat an adapter or identity manifest as a vendor binary, collapse checks, hardcode success, conceal blockers, or make a self-referential current-tree claim.
- Interfaces: RoleAwareFormalVerificationReleaseCandidate@1, ProductionSemanticElevationFanIn@1
- Resource class: cpu-validation

## FVT-G214 Publish the post-merge deployment attestation

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G213
- Fib priority: 514229
- Priority: P0
- Track: completion
- Bundle: formal-verification-tactician/toolchain-release-finalizer
- Goal: After the release-candidate merge, bind its durable terminal supervisor receipt and publish the final deployment attestation without circular tree identity.
- Evidence: test/integration/test_formal_verification_role_aware_post_merge_attestation.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Outputs: tools/logic/finalize_formal_verification_deployment.py, test/integration/test_formal_verification_role_aware_post_merge_attestation.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json
- Validation: python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py test/api/test_formal_verification_tactician_readiness_completion.py -q
- Acceptance: The finalizer runs only after FVT-G213 has a successful, durable, canonical member completion receipt and reachable merged commit; it verifies event-chain continuity, expected outputs, validation result, source tree, merged tree, datasets gitlink, origin publication, candidate digest, supported-capability closure, hard-zero gates, authority boundaries, quarantines, and public surfaces; it publishes either a receipt commit whose parent is the certified release commit with a strictly limited generated-artifact diff or an external content-addressed attestation; mutating any event, tree, artifact, check, binding, or publication fact invalidates the receipt; absent or stale terminal evidence remains partial and can never be called deployment-ready.
- Conflict policy: Sole post-merge finalizer; read live state without mutation, never attest the current task's future event, and never weaken a missing terminal receipt or publication gate.
- Interfaces: RoleAwareFormalVerificationRelease@1
- Resource class: cpu-validation

## FVT-G200 Reissue full role-aware deployment certification

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G090, FVT-G101, FVT-G102, FVT-G103, FVT-G120, FVT-G130, FVT-G131, FVT-G140, FVT-G150, FVT-G151, FVT-G160, FVT-G170, FVT-G180, FVT-G181, FVT-G190, FVT-G214
- Fib priority: 75025
- Priority: P0
- Track: completion
- Bundle: formal-verification-tactician/toolchain-release
- Goal: Run the complete role-aware matrix after the explicit installation phase and reissue current-tree implementation and deployment-readiness receipts.
- Evidence: test/integration/test_formal_verification_role_aware_completion.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Outputs: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_toolchain_certificate.json, docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json, test/integration/test_formal_verification_role_aware_completion.py
- Validation: python -m pytest test/integration/test_formal_verification_real_tool_matrix.py test/integration/test_formal_verification_role_aware_completion.py test/api/test_formal_verification_tactician_readiness_completion.py -q
- Acceptance: A fresh offline certificate and completion receipt bind the current parent tree, datasets gitlink, exact tool and artifact identities, every required positive/negative/mutation/replay result, authority roles and ceilings, disagreement quarantines, public surfaces, and supervisor evidence; Lean, Runtime MTL, and Datalog/SecPAL are no longer merely usable; every supported managed external capability is installed and semantically certified; any genuinely unsupported platform exception is explicit, narrowly scoped, and cannot be counted as complete or production-certified.
- Conflict policy: Sole owner for central certificate and completion-receipt regeneration after every dependency merges; never manufacture success, weaken skips, install during offline certification, or conceal an unavailable lane.
- Interfaces: RoleAwareFormalVerificationRelease@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G215 Close formal-verification packaging and distribution coverage

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G090, FVT-G201
- Fib priority: 121393
- Priority: P0
- Track: dependency-integrity
- Bundle: formal-verification-tactician/packaging-distribution
- Goal: Make every public formal-verification module, reviewed installer plugin, runtime asset, and optional dependency declaration survive both source and clean wheel installations.
- Evidence: test/packaging/test_logic_verification_clean_install.py, test/packaging/test_formal_verification_distribution_contract.py
- Outputs: setup.py, pyproject.toml, requirements.txt, ipfs_datasets_py/setup.py, ipfs_datasets_py/pyproject.toml, ipfs_datasets_py/requirements.txt, ipfs_datasets_py/requirements-lazy.txt, ipfs_datasets_py/requirements-theorem-provers.txt, test/packaging/test_formal_verification_distribution_contract.py
- Validation: python -m pytest test/packaging/test_logic_verification_clean_install.py test/packaging/test_formal_verification_distribution_contract.py test/test_pip_install_simulation.py -q
- Acceptance: Root and datasets setup.py, pyproject metadata, requirements files, and extras have one machine-checked dependency inventory; namespace-package discovery includes logic backends, software verification, installer plugins, and runtime assets in built wheels; every declared plugin module exists; a clean isolated wheel install imports and inventories the Logic API without network access, downloads, builds, user-site leakage, editable-source leakage, or installation side effects; optional native and external provers remain optional and are surfaced as unavailable rather than breaking base installation.
- Conflict policy: Own packaging metadata and clean-install gates; do not make heavyweight prover binaries mandatory Python dependencies, hide missing wheel content with PYTHONPATH, or install anything during import and inventory.
- Interfaces: FormalVerificationDistributionContract@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G216 Bind the public Logic API to transactional lazy installers

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G215
- Fib priority: 196418
- Priority: P0
- Track: dependency-integrity
- Bundle: formal-verification-tactician/lazy-installer-facade
- Goal: Replace placeholder and stale installer dispatch with one explicit, platform-aware, transactional lazy-install lifecycle for every reviewed prover family.
- Evidence: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py, ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py ipfs_datasets_py/tests/unit/test_lazy_dependency_installation.py ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py -q
- Acceptance: LogicVerificationAPI.install_provider resolves reviewed family plugins for SMT, kernels, state models, authorization, protocols, ATP, hyperproperties, Runtime MTL, advisors, and ZKP; probe, inventory, import, dry-run, and offline certification execute no installer command and perform no network access; installation requires an explicit allow flag and produces a bounded plan before mutation; platform, dependency, license, checksum, artifact, executable, rollback, and post-install semantic-probe results are returned as structured evidence; interrupted or failed publication preserves the previous good installation and cannot promote capability or semantic authority.
- Conflict policy: Own the public install facade, registry, and lifecycle; never infer permission from a probe, dispatch an unreviewed shell command, silently fall back to a shim, or let installation occur inside certification.
- Interfaces: LogicVerificationLazyInstaller@1
- Submodules: ipfs_datasets_py
- Resource class: io-artifact

## FVT-G217 Implement the genuine SecPAL external-toolchain path

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G209, FVT-G216
- Fib priority: 121393
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/secpal-live-toolchain
- Goal: Replace ambiguous SecPAL acquisition and adapter behavior with an official-artifact, license-aware, host-specific lazy installer and live semantic runner.
- Evidence: test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_secpal_live_toolchain_contract.py test/integration/toolchains/test_external_authorization_vendor_certification.py test/integration/toolchains/test_external_authorization_toolchain_certification.py -q
- Acceptance: Every supported SecPAL target binds an official publisher URL or operator-supplied reviewed artifact, immutable version and digest, redistribution and execution terms, architecture and OS, runtime dependencies, install plan, executable identity, and rollback behavior; unsupported hosts are derived from the reviewed lock and cannot install, certify, or count as complete; real allow, deny, unknown, delegation, conflict, rule/scope mutation, replay, malformed, timeout, and disagreement cases execute through the selected external engine; the in-process Datalog/SecPAL family and any hermetic adapter remain separately named and cannot impersonate the vendor tool.
- Conflict policy: Own SecPAL artifact provenance, platform matrix, installer, and external semantics; never invent an upstream release, accept an unreviewed mirror, bypass license terms, or label the in-process engine as the external vendor binary.
- Interfaces: SecPALLiveToolchainContract@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G218 Implement the genuine ErgoAI advisor-toolchain path

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G201, FVT-G216
- Fib priority: 121393
- Priority: P0
- Track: external-capability
- Bundle: formal-verification-tactician/ergoai-live-toolchain
- Goal: Replace ErgoAI wrapper and proposal-only assumptions with a locked official distribution, dependency-complete lazy installer, and bounded live semantic adapter while preserving advisor authority ceilings.
- Evidence: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_ergoai_live_toolchain_contract.py test/integration/toolchains/test_advisor_role_certification.py -q
- Acceptance: The lock binds the official ErgoAI distribution or reviewed source revision, license and acquisition conditions, archive/source digests, XSB and every runtime/build dependency, supported OS/architecture matrix, entry point, and exact identity probe; explicit lazy installation is staged, checksum-verified, atomic, relocatable, and offline after acquisition; live entailment, non-entailment, contradiction, rule/query mutation, deterministic replay, malformed input, timeout, and resource-bound cases execute through ErgoAI; results remain proposal or candidate evidence until reconstructed or checked by an independent proof authority.
- Conflict policy: Own ErgoAI provenance, dependencies, lazy installer, wrapper, and bounded semantics; never scrape an unauthoritative artifact, download during certification, treat wrapper fixtures as live execution, or elevate an advisor verdict to theorem authority.
- Interfaces: ErgoAILiveToolchainContract@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## FVT-G219 Acquire authoritative SecPAL and ErgoAI live evidence

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G217, FVT-G218
- Fib priority: 196418
- Priority: P0
- Track: external-authority
- Bundle: formal-verification-tactician/vendor-live-authority
- Goal: Execute SecPAL and ErgoAI on genuinely supported hosts with legally acquired official artifacts and publish replayable, content-addressed live evidence.
- Evidence: docs/architecture/formal_verification_secpal_ergoai_live_receipt.json, test/integration/toolchains/test_secpal_ergoai_authoritative_live_evidence.py
- Outputs: docs/architecture/formal_verification_secpal_ergoai_live_receipt.json, test/integration/toolchains/test_secpal_ergoai_authoritative_live_evidence.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_secpal_ergoai_authoritative_live_evidence.py -q
- Acceptance: Independent positive, negative, mutation, replay, malformed, timeout, bound, and cross-engine disagreement cases run against exact official SecPAL and ErgoAI executables on hosts covered by their reviewed platform locks; receipts bind acquisition authority, license disposition without publishing restricted bytes, host OS and architecture, every runtime dependency, executable and artifact digests, sources, queries, bounds, raw-output digests, parser decisions, witnesses, timestamps, freshness, and deterministic replay; fixture, shim, parser-only, wrapper-only, proposed, identity-only, or unsupported-host results cannot satisfy this goal.
- Conflict policy: Own only the externally executed evidence and its public-safe envelope; never commit restricted artifacts, credentials, private paths, or raw secrets and never convert missing external authority into local completion.
- Interfaces: SecPALErgoAIAuthoritativeLiveEvidence@1
- Completion authority: external
- External authority blockers: legally acquired official SecPAL and ErgoAI artifacts, accepted license terms where required, and suitable supported execution hosts
- Resource class: cpu-proof-solver

## FVT-G220 Audit every deployment axis end to end

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G215, FVT-G216, FVT-G217, FVT-G218
- Fib priority: 317811
- Priority: P0
- Track: certification-integrity
- Bundle: formal-verification-tactician/end-to-end-assurance
- Goal: Make dependency, capability, semantic, platform-binding, authority, packaging, installer-boundary, and public-surface readiness independently visible and jointly fail closed.
- Evidence: test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json
- Outputs: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json, test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py test/integration/test_formal_verification_real_tool_matrix.py test/packaging/test_logic_verification_clean_install.py -q
- Acceptance: Each provider and host tuple reports separate dependency, packaging, installer, capability, semantic, platform, authority, freshness, and public-surface states with exact evidence references and reason codes; no axis inherits success from another; supported missing dependencies, missing wheel files, placeholder dispatch, stale locks, wrong-architecture artifacts, parser fixtures, advisor-only evidence, and unsupported hosts are distinguishable; SecPAL in-process and external identities and ErgoAI advisor and independent proof authority remain distinct; an adversarial test mutates every axis and proves that the joint readiness claim fails closed.
- Conflict policy: Own the cross-axis matrix and aggregation policy; do not hardcode green states, collapse platform exceptions into success, or let one provider stand in for another.
- Interfaces: FormalVerificationEndToEndAssuranceMatrix@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## FVT-G221 Reissue deployment certification with authoritative vendor evidence

- Status: active
- Parent: FVT-G000
- Depends on: FVT-G214, FVT-G219, FVT-G220
- Fib priority: 514229
- Priority: P0
- Track: completion
- Bundle: formal-verification-tactician/authoritative-vendor-release
- Goal: Reissue the role-aware release and post-merge attestation only after packaging, lazy installers, every readiness axis, and genuine SecPAL and ErgoAI live evidence are closed.
- Evidence: docs/architecture/formal_verification_authoritative_vendor_release.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- Outputs: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_authoritative_vendor_release.json, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/test_formal_verification_authoritative_vendor_release.py test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py test/integration/toolchains/test_secpal_ergoai_authoritative_live_evidence.py -q
- Acceptance: The release binds clean-wheel evidence, explicit lazy-install receipts, exact dependency and platform identities, complete specialized semantic cases, SecPAL and ErgoAI authoritative live receipts, authority ceilings, disagreement quarantines, public-safe envelopes, durable supervisor completion, source and merged trees, recursive gitlinks, and origin publication; every dependency is reachable and fresh; fixture, shim, unsupported, proposal-only, or externally blocked lanes remain disclosed and prevent deployment-ready status.
- Conflict policy: Sole owner of the authoritative vendor release after every dependency closes; never manufacture external evidence, weaken a platform or authority gate, or attest the current task's future merge.
- Interfaces: FormalVerificationAuthoritativeVendorRelease@1
- Resource class: cpu-validation
