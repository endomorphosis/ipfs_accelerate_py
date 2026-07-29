# Agent Supervisor Proof-Gated Contract Repair Taskboard

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix `RPR-`.

Companion artifacts:

- objective heap:
  `docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md`
- architecture:
  `docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md`
- scheduler:
  `config/agent_supervisor_proof_gated_contract_repair_scheduler.json`

Normative execution order:

```text
trace -> sender requirement -> code candidate nomination
      -> ipfs_datasets_py.logic proof/reconstruction
      -> hard eligibility -> deterministic ranking
      -> exact target decision -> repair packet -> patch-bound re-proof
```

Vector similarity never grants semantic or mutation authority. Cross-program
VFS and datasets features are capability-probed preconditions, not unknown
task IDs on this board.

## RPR-000 Seal proof-gated contract-repair control plane

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: control
- Depends on:
- Goal id: RPR-G000
- Outputs: docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md, docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md, docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md, config/agent_supervisor_proof_gated_contract_repair_scheduler.json, scripts/proof_gated_contract_repair_supervisor.sh
- Validation: test -f docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md && test -f docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md && test -f docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md && test -f config/agent_supervisor_proof_gated_contract_repair_scheduler.json && test -x scripts/proof_gated_contract_repair_supervisor.sh
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/control
- Parallel lane: control
- Resource class: cpu-small
- Resource stage: analysis
- Token class: small
- Estimated tokens: 4000
- Implementation timeout seconds: 1800
- Predicted files: docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md, docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md, docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md, config/agent_supervisor_proof_gated_contract_repair_scheduler.json, scripts/proof_gated_contract_repair_supervisor.sh
- AST symbols: BrokenContractTrace, CallRequirementContract, RepairCandidate, RepairTargetDecision
- Interfaces: ProofGatedContractRepairPlan@1
- Allow concurrent with:
- Conflict policy: Planning/control artifacts are protected after commit; workers must not edit them.
- Preconditions: Dedicated branch agent/proof-gated-contract-repair and exact ipfs_datasets_py gitlink are present.
- Effects: A committed, parseable, parallel implementation program can be launched reproducibly.
- Evidence subset: plan, objective heap, task dependency DAG, scheduler profile, launcher doctor
- Acceptance: All five artifacts exist; board has four file-disjoint ready tasks after this completed seal; no local dependency references an unknown task; launcher binds imports to the exact accelerator and datasets worktree and starts strict shards without objective/codebase refill.
- Embedding query: proof gated contract repair supervisor taskboard vector candidate logic target decision

## RPR-001 Define bounded contract-repair records

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts
- Depends on: RPR-000
- Goal id: RPR-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_contracts.py, test/api/test_agent_supervisor_contract_repair_contracts.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_contracts.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/contracts
- Parallel lane: rpr-contracts
- Resource class: cpu-small
- Resource stage: analysis
- Token class: medium
- Estimated tokens: 12000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_contracts.py, test/api/test_agent_supervisor_contract_repair_contracts.py
- AST symbols: TraceDisposition, RepairStrategy, BrokenContractTrace, CallRequirementContract, MemorySafetyFacet, RepairCandidate, RepairTargetDecision
- Interfaces: ProgramContract@1, CodeClaimRecord@1, ProofReceipt, RetrievalSnapshotBinding
- Allow concurrent with: RPR-002, RPR-003, RPR-004
- Conflict policy: Own only the new RPR contract module and its test; import existing content identity, assurance, retrieval, and ProgramContract types without editing them.
- Preconditions: Existing ProgramContract@1, proof receipt, and canonical identity helpers are importable.
- Effects: Every later stage exchanges immutable, bounded, content-addressed records with exact authority and invalidation roots.
- Evidence subset: canonical serialization, bounds, state machine, identity, forged/stale rejection
- Acceptance: Define closed dispositions and strategies; bind repository/forest/tree, graph/index/model/config, translator/toolchain/policy, caller/target spans, evidence refs, proof refs, and exact read/write authority; reject source bodies, non-finite/unbounded values, forged ids, invalid disposition combinations, decisions without full candidate-set identity, and write paths not derived by a decision; MemorySafetyFacet distinguishes unsupported, empirical, and proved evidence and cannot be inferred from max_memory_bytes.
- Embedding query: typed broken trace sender receiver candidate target decision memory safety exact roots

## RPR-002 Bind exact datasets-logic and VFS capabilities

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: capabilities
- Depends on: RPR-000
- Goal id: RPR-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/integrations/contract_repair_capabilities.py, test/api/test_agent_supervisor_contract_repair_capabilities.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_capabilities.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/capabilities
- Parallel lane: rpr-capabilities
- Resource class: cpu-small
- Resource stage: analysis
- Token class: medium
- Estimated tokens: 10000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/integrations/contract_repair_capabilities.py, test/api/test_agent_supervisor_contract_repair_capabilities.py
- AST symbols: ContractRepairCapability, ContractRepairCapabilityReport, probe_contract_repair_capabilities
- Interfaces: IPFSDatasetsLogicProvider, BackendCapability, ProgramGraph, ProgramCallResolver, ProgramContract
- Allow concurrent with: RPR-001, RPR-003, RPR-004
- Conflict policy: Own only the new capability adapter/test; do not edit VFS-owned graph/resolver/contract/prover files or ipfs_datasets_py.
- Preconditions: The launcher pins PYTHONPATH to this checkout and exact ipfs_datasets_py gitlink.
- Effects: Runtime can prove which exact upstream interfaces and proof toolchains are usable and which semantics must remain unsupported.
- Evidence subset: module file paths, git revisions, schema versions, executable versions, supported semantics, reconstruction support
- Acceptance: Lazy cold-import probe records exact accelerator/datasets module paths, gitlink revision, interface/schema versions, IR/TDFOL/CEC/SMT/Hammer capabilities, cvc5/Z3 availability, Python/Node/TypeScript/mypy versions, and VFS graph/resolver/extractor/checker/prover/repair interfaces; package presence alone never means available; missing/incompatible/partial/timeouts return typed diagnostics; no auto-install/network; solver candidates remain non-authoritative; current environment records cvc5 while tolerating absent Z3/mypy/pinned TypeScript.
- Embedding query: capability probe ipfs datasets logic cvc5 z3 hammer resolver graph contract exact module gitlink

## RPR-003 Implement a snapshot-bound code-symbol vector index

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: code-index
- Depends on: RPR-000
- Goal id: RPR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/code_symbol_vector_index.py, test/api/test_agent_supervisor_code_symbol_vector_index.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_code_symbol_vector_index.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/code-index
- Parallel lane: rpr-code-index
- Resource class: cpu-medium
- Resource stage: retrieval
- Token class: large
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/code_symbol_vector_index.py, test/api/test_agent_supervisor_code_symbol_vector_index.py
- AST symbols: CodeSymbolIndexRow, CodeVectorIndexSnapshot, CodeVectorQuery, CodeVectorHit
- Interfaces: RepositoryIndexSnapshot, AnalysisASTIndex, RetrievalSnapshotBinding, VectorSearchProvider
- Allow concurrent with: RPR-001, RPR-002, RPR-004
- Conflict policy: Own only the new code-symbol index/test; do not modify RepositoryIndexer, AnalysisASTIndex, program_ast_adapters, analysis_retrieval, or the task/objective vector index.
- Preconditions: Existing compact repository and AST indexes can supply body-free source/AST references; deterministic fixture vectors may stand in for an unavailable embedding backend.
- Effects: Repair retrieval gains bounded per-symbol vector nomination with exact provenance and invalidation.
- Evidence subset: symbol chunks, signatures, imports/exports, calls, errors/effects, docs/tests, ownership, Git lineage, tombstones
- Acceptance: Index root binds exact forest/tree, coverage, producer, chunking/normalization, model/revision/dimensions/metric/config, included/excluded paths, rich AST sidecar refs, and tombstones; rows contain bounded metadata and references rather than bodies; incremental update equals clean rebuild on fixtures; moved blobs preserve reviewed lineage without inventing semantic rename; stale/cross-tree/poisoned/forged/dimension mismatch/incomplete results fail; every hit has semantic_authority=false.
- Embedding query: code symbol vector index signature call effect documentation test git lineage snapshot tombstone

## RPR-004 Build the adversarial broken-contract fixture corpus

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: fixtures
- Depends on: RPR-000
- Goal id: RPR-G090
- Outputs: test/fixtures/agent_supervisor/contract_repair, test/api/test_agent_supervisor_contract_repair_fixtures.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_fixtures.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/fixtures
- Parallel lane: rpr-fixtures
- Resource class: cpu-small
- Resource stage: analysis
- Token class: large
- Estimated tokens: 16000
- Implementation timeout seconds: 7200
- Predicted files: test/fixtures/agent_supervisor/contract_repair, test/api/test_agent_supervisor_contract_repair_fixtures.py
- AST symbols: ContractRepairFixture, ContractRepairFixtureManifest
- Interfaces: fixture manifest schema only
- Allow concurrent with: RPR-001, RPR-002, RPR-003
- Conflict policy: Own only the new fixture directory and fixture validation test; do not implement production repair logic in fixtures.
- Preconditions: Fixtures are hermetic, tiny, deterministic, and contain no credentials/network calls.
- Effects: Every later component has shared positive, negative, ambiguous, stale, and poisoned cases with expected dispositions.
- Evidence subset: rename, move, re-export, same-name decoy, poisoned vector, adapter, missing/new implementation, async/error/effect/auth/resource/lifetime drift
- Acceptance: Manifest content-identifies source/spec/test/history/index/proof expectations for pure rename, module move, alias/re-export/registration, signature drift, vector-nearest incompatible decoy, adapter-required, declaration without implementation, unique new site, multiple-site abstention, dynamic/reflection/FFI, ownership/lifetime unsupported, stale roots, read-only target, dependency cycle, and tombstone cases; test validates no fixture expectation treats vector score or implementation observation as authority.
- Embedding query: adversarial contract repair fixture rename move decoy poison adapter ambiguity stale FFI

## RPR-005 Classify a broken call into a bounded trace

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: trace-analysis
- Depends on: RPR-001, RPR-002, RPR-004
- Goal id: RPR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/broken_contract_trace.py, test/api/test_agent_supervisor_broken_contract_trace.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_broken_contract_trace.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/trace
- Parallel lane: rpr-trace
- Resource class: cpu-medium
- Resource stage: analysis
- Token class: large
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/broken_contract_trace.py, test/api/test_agent_supervisor_broken_contract_trace.py
- AST symbols: BrokenContractTraceBuilder, BrokenTraceClassifier, TraceDisposition
- Interfaces: BrokenContractTrace@1, ProgramGraph protocol, ProgramCallResolver protocol, ProgramEvidenceFact
- Allow concurrent with: RPR-006, RPR-007
- Conflict policy: Own only the new trace adapter/test; consume compatible upstream interfaces but never edit or manufacture graph/resolver evidence.
- Preconditions: RPR contracts and capability report available; fixture corpus supplies a protocol-compatible graph when VFS graph/resolver is unavailable.
- Effects: Findings gain exact caller facts and a conservative reasoned frontier before candidate search.
- Evidence subset: caller span, call form, actual args, awaitedness, return uses, caught errors, effects/capabilities, graph completeness
- Acceptance: Emit resolved_mismatch, missing_local, likely_refactor, adapter_required, external, dynamic, ambiguous, or unsupported only from bounded evidence; preserve actual arg count/names/types/ranges, awaitedness, result uses, handled errors, policy context, graph/index/toolchain roots, exclusions, and unknown frontier; exact same-name or vector evidence cannot resolve a call; missing/incompatible resolver yields unsupported and never blocks unrelated lanes.
- Embedding query: broken call trace classifier likely refactor missing local external dynamic ambiguous

## RPR-006 Synthesize sender requirements and receiver guarantees

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contract-synthesis
- Depends on: RPR-001, RPR-002, RPR-004
- Goal id: RPR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/sender_receiver_contracts.py, test/api/test_agent_supervisor_sender_receiver_contracts.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_sender_receiver_contracts.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/contract-synthesis
- Parallel lane: rpr-contract-synthesis
- Resource class: cpu-medium
- Resource stage: analysis
- Token class: large
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/sender_receiver_contracts.py, test/api/test_agent_supervisor_sender_receiver_contracts.py
- AST symbols: SenderRequirementCompiler, ReceiverGuaranteeCompiler, CallRequirementContract
- Interfaces: CallRequirementContract@1, ExpectedProgramContract, ObservedProgramContract, ProgramContractComparison
- Allow concurrent with: RPR-005, RPR-007
- Conflict policy: Own only the new synthesis adapter/test; do not edit ProgramContract@1, contract extractor/checker, schemas, or VFS files.
- Preconditions: Existing ProgramContract@1 is reused; capability report may select fixture protocol adapters when upstream extraction/checking is unavailable.
- Effects: Every candidate can be compared against a precise independently sourced caller/consumer contract.
- Evidence subset: source precedence, variance, arity/defaults, nullable/schema, async/errors/effects/capabilities/auth, temporal/resource
- Acceptance: Build expectation only from reviewed IDL/schema, public signatures/stubs, conformance tests, normative specs, and manifests under explicit precedence; implementation observations cannot validate themselves; encode caller-provided input domain, receiver accepted domain, receiver guarantees, consumer-required output, handled errors, permitted effects/capabilities/auth, lifecycle/cancellation/state, ordering/atomicity/consistency/resource/fallback; inputs are contravariant and outputs covariant; conflicts and unsupported clauses remain explicit.
- Embedding query: sender receiver requirement guarantee source precedence variance async error effect resource

## RPR-007 Capture memory-safety and native-boundary evidence

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: memory-safety
- Depends on: RPR-001, RPR-002, RPR-004
- Goal id: RPR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/memory_safety_facets.py, test/api/test_agent_supervisor_memory_safety_facets.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_memory_safety_facets.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/memory-safety
- Parallel lane: rpr-memory-safety
- Resource class: cpu-medium
- Resource stage: proof
- Token class: medium
- Estimated tokens: 14000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/memory_safety_facets.py, test/api/test_agent_supervisor_memory_safety_facets.py
- AST symbols: MemorySafetyEvidenceCollector, MemorySafetyFacet, NativeBoundary
- Interfaces: MemorySafetyFacet@1, ResourceBounds, ProofEvidence
- Allow concurrent with: RPR-005, RPR-006
- Conflict policy: Own only the new memory facet adapter/test; do not extend ProgramContract@1 or claim tool authority beyond existing proof evidence contracts.
- Preconditions: Tool availability comes only from RPR-020; tests use deterministic receipts and do not require installing native tools.
- Effects: Repair policy can distinguish resource bounds from ownership/lifetime/unsafe safety and require language-specific evidence.
- Evidence subset: ownership, mutation regions, aliasing, borrow/lifetime, bounds/nullability, FFI, unsafe, allocator, compiler/analyzer/sanitizer
- Acceptance: Model supported, empirical, proved, unsupported, stale, and error states; bind language/runtime/toolchain/tree/scope; recognize Python/TypeScript reflection/native extension/FFI limits; native proof policy can require borrow checker, Miri, ASan/UBSan or equivalent receipts; max_memory_bytes and a passing unit test cannot independently produce memory_safe; missing required evidence fails closed.
- Embedding query: memory safety ownership borrow lifetime alias unsafe FFI allocator sanitizer resource bound

## RPR-008 Nominate refactored receivers and implementation sites

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: candidate-retrieval
- Depends on: RPR-003, RPR-005, RPR-006, RPR-007
- Goal id: RPR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_candidate_retrieval.py, test/api/test_agent_supervisor_contract_repair_candidate_retrieval.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_candidate_retrieval.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/retrieval
- Parallel lane: rpr-retrieval
- Resource class: cpu-medium
- Resource stage: retrieval
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_candidate_retrieval.py, test/api/test_agent_supervisor_contract_repair_candidate_retrieval.py
- AST symbols: ContractRepairCandidateRetriever, CandidateNominationReceipt
- Interfaces: RepairCandidate@1, CodeVectorQuery, BoundRetrievalCandidate, CallRequirementContract@1
- Allow concurrent with:
- Conflict policy: Own only the new retrieval adapter/test; existing analysis_retrieval and VFS graph/query modules stay unchanged and non-authoritative.
- Preconditions: Trace, sender requirement, memory facet, and code index are available under one exact tree binding.
- Effects: High-recall candidate sets include moved receivers, adapters, existing declarations, and architecture-valid site anchors without choosing a winner.
- Evidence subset: exact lineage, structural fingerprint, aliases/reexports/registrations, call/dependency distance, ownership, AST, specs/tests, lexical/vector
- Acceptance: Deterministically union and deduplicate all signal families; bind complete candidate set, per-signal evidence refs, bounds, tree/graph/index/model/config roots; preserve semantic_authority=false; classify proposed strategies without admission; same-name incompatible, poisoned vector, stale/cross-tree, read-only, generated/vendor/archive, forbidden-layer, partial, and forged-history candidates receive stable rejection/diagnostic reasons; retrieval never emits write authority.
- Embedding query: nominate repair candidate moved function refactor adapter implementation site vector graph history ownership

## RPR-009 Compile substitution, equivalence, adapter, and placement obligations

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: proof-obligations
- Depends on: RPR-006, RPR-007, RPR-008
- Goal id: RPR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_obligations.py, test/api/test_agent_supervisor_contract_repair_obligations.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_obligations.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/obligations
- Parallel lane: rpr-obligations
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_obligations.py, test/api/test_agent_supervisor_contract_repair_obligations.py
- AST symbols: ContractRepairObligationCompiler, SubstitutionObligation, PlacementObligation
- Interfaces: CodeProofObligation, IRClaim, ProofObligation, CallRequirementContract@1, RepairCandidate@1
- Allow concurrent with:
- Conflict policy: Own only RPR obligation adapter/test; reuse existing claim/proof/cache/provider contracts and do not edit VFS code_contract_logic or datasets software_contracts.
- Preconditions: RPR capability report identifies exact supported LogicIR/solver semantics; unsupported clauses are retained rather than approximated.
- Effects: Each candidate receives independent, source-bound obligations before ranking.
- Evidence subset: input implication, output implication, errors, effects, capabilities, lifecycle, resources, memory facet, lineage, reachability, placement
- Acceptance: Compile sender facts imply receiver preconditions, receiver guarantees imply caller requirements, error/effect/capability/auth/resource/lifecycle/memory compatibility; pure rename additionally requires bidirectional refinement plus identity/history and route wiring; adapters require total finite mappings; new/existing sites require ownership, no omitted reachable compatible implementation, dependency DAG, visibility/registration, and exact stub contract; every obligation binds premise/source/assumption/tree/translator/toolchain/policy ids; invented axioms, partial slices as closed, and silent approximation are rejected.
- Embedding query: compile contract repair logic obligations substitution rename equivalence adapter placement

## RPR-010 Prove, refute, and reconstruct candidate obligations

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: proof
- Depends on: RPR-002, RPR-009
- Goal id: RPR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_prover.py, test/api/test_agent_supervisor_contract_repair_prover.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_prover.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/prover
- Parallel lane: rpr-prover
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_prover.py, test/api/test_agent_supervisor_contract_repair_prover.py
- AST symbols: ContractRepairProver, CandidateProofBundle, reconstruct_contract_repair_proof
- Interfaces: IPFSDatasetsLogicProvider, TrustAwareProofCache, ProofReceipt, ProofEvidence
- Allow concurrent with:
- Conflict policy: Own only the new prover adapter/test; do not alter datasets provider authority, proof cache trust, or VFS prover/context modules.
- Preconditions: cvc5 may be used when capability-probed; absence of Z3 or another backend is a non-conclusive capability fact, not permission to degrade.
- Effects: Candidates carry current authoritative proof/refutation receipts or explicit non-conclusive dispositions.
- Evidence subset: backend request, candidate result, independent reconstruction, counterexample, cache identity, toolchain and capability snapshot
- Acceptance: Route supported obligations through exact ipfs_datasets_py logic backends; candidate_authoritative remains false until policy-approved deterministic check or reconstruction; independently verify theorem/model/counterexample as supported; cache key binds full semantics and re-derives assurance; timeout/unknown/malformed/wrong theorem/stale toolchain/changed tree/incomplete slice/missing backend/unsupported semantics remain non-conclusive; produce minimal counterexample refs where available.
- Embedding query: prove reconstruct code contract candidate cvc5 hammer cache counterexample authority

## RPR-011 Prove implementation-site admissibility

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: placement
- Depends on: RPR-008, RPR-009, RPR-010
- Goal id: RPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/implementation_site_admissibility.py, test/api/test_agent_supervisor_implementation_site_admissibility.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_implementation_site_admissibility.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/placement
- Parallel lane: rpr-placement
- Resource class: cpu-medium
- Resource stage: planning
- Token class: large
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/implementation_site_admissibility.py, test/api/test_agent_supervisor_implementation_site_admissibility.py
- AST symbols: ImplementationSiteAdmissibility, PlacementDecision
- Interfaces: RepairCandidate@1, PlacementObligation, ProofReceipt, RepositoryAuthority
- Allow concurrent with:
- Conflict policy: Own only the new placement module/test; no source insertion or shared-file edit occurs in this task.
- Preconditions: Placement obligations and proof bundles are current under the same candidate set/tree.
- Effects: Missing implementations can identify one architecture-valid exact site or abstain.
- Evidence subset: declaration/interface anchor, ownership, mutation authority, no duplicate, dependency layering, visibility, registration, effects/capabilities, memory facet
- Acceptance: Admit a site only if target module/interface ownership and write authority are exact, no compatible reachable implementation was omitted, dependency DAG remains legal, visibility/export/registration route is satisfiable, required effects/capabilities/memory policy are supportable, and generated stub contract exactly matches sender requirements; exclude external read-only, generated/vendor/archive, forbidden layer/cycle, ambiguous owner, and multiple equal sites; ambiguity yields no write path.
- Embedding query: prove new implementation site ownership architecture dependency cycle no duplicate exact stub

## RPR-012 Hard-gate and rerank eligible candidates

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: reranking
- Depends on: RPR-008, RPR-010, RPR-011
- Goal id: RPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_reranker.py, test/api/test_agent_supervisor_contract_repair_reranker.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_reranker.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/reranker
- Parallel lane: rpr-reranker
- Resource class: cpu-medium
- Resource stage: planning
- Token class: large
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_reranker.py, test/api/test_agent_supervisor_contract_repair_reranker.py
- AST symbols: ContractRepairReranker, CandidateEligibility, CandidateRank
- Interfaces: RepairCandidate@1, CandidateProofBundle, PlacementDecision
- Allow concurrent with:
- Conflict policy: Own only the new reranker/test; do not issue write authority or edit retrieval/proof modules.
- Preconditions: Complete bounded candidate set, proof bundles, and placement results share exact roots.
- Effects: Ineligible candidates are removed before a deterministic soft ordering can be computed.
- Evidence subset: freshness, authority, semantic completeness, proof reconstruction, contract coverage, lineage, graph/ownership, specs/tests, AST, lexical/vector
- Acceptance: Hard gates require exact fresh roots, target/site validity, write authority, independent expectation, complete supported slice, all mandatory reconstructed proofs, and no counterexample; a failed obligation rejects regardless of vector/lexical score; eligible ordering is proof/coverage then lineage then graph/ownership then authoritative spec/test then AST then lexical then vector; missing signals do not silently inflate others; fixed weights/tie breakers are receipt-bound; tie or insufficient margin yields ambiguous.
- Embedding query: hard gate candidate rerank proof coverage lineage graph ownership vector cannot override

## RPR-013 Admit one exact repair target or abstain

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: target-admission
- Depends on: RPR-001, RPR-012
- Goal id: RPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/repair_target_admission.py, test/api/test_agent_supervisor_repair_target_admission.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_repair_target_admission.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/target-admission
- Parallel lane: rpr-target-admission
- Resource class: cpu-small
- Resource stage: planning
- Token class: medium
- Estimated tokens: 14000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/repair_target_admission.py, test/api/test_agent_supervisor_repair_target_admission.py
- AST symbols: RepairTargetAdmission, RepairTargetDecisionValidator
- Interfaces: RepairTargetDecision@1, CandidateRank, RepositoryAuthority
- Allow concurrent with:
- Conflict policy: Own only the new target-admission module/test; no packet, finding, taskboard, or provider changes.
- Preconditions: Reranker result is deterministic, current, and has one policy-margin winner or explicit abstention.
- Effects: A content-addressed expiring decision becomes the sole source of exact repair read/write authority.
- Evidence subset: complete candidate set, eligibility/rejection reasons, selected strategy/target, roots, proof receipts, allowlists, invalidators
- Acceptance: Validate and content-identify complete candidate ordering and rejection reasons; issue rename_substitution, adapter, implement_existing_declaration, new_implementation, reject, or ambiguous; exact read/write paths/spans derive from selected target and repository authority only; bind tree/forest/graph/index/model/config/translator/toolchain/policy/proof roots and expiry; changed root, missing target, read-only path, proof downgrade, candidate-set mutation, tie, or low margin invalidates; abstention has no write paths.
- Embedding query: admit exact repair target decision write allowlist invalidation abstain

## RPR-014 Materialize a target-decision-bound edit packet

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: repair-packet
- Depends on: RPR-013
- Goal id: RPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_edit_packet.py, test/api/test_agent_supervisor_contract_repair_edit_packet.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_edit_packet.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/edit-packet
- Parallel lane: rpr-edit-packet
- Resource class: cpu-medium
- Resource stage: materialize
- Token class: large
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_edit_packet.py, test/api/test_agent_supervisor_contract_repair_edit_packet.py
- AST symbols: ContractRepairEditPacket, materialize_contract_repair_edit_packet
- Interfaces: RepairTargetDecision@1, CodeEditPacket@1, ContextCapsule
- Allow concurrent with:
- Conflict policy: Own only the new @2 packet module/test; do not modify mcp_contract_edit_packet.py or weaken @1 exact affected-path behavior.
- Preconditions: A current admitted target decision supplies all mutation authority.
- Effects: The implementation agent receives one exact target, precise contract, proof refs, postconditions, and bounded expansion handles.
- Evidence subset: broken trace, sender/receiver table, target rationale, counterexample, proof/index roots, allowlists, validation commands
- Acceptance: Materialize only from a current admitted non-abstaining decision; write paths exactly equal decision authority; compact packet binds selected strategy/path/span, expected/observed clauses, unsupported limits, proof/counterexample/index refs, post-edit obligations, focused commands, and bounded handles; source/proof bodies and non-selected alternatives cannot expand scope; stale/forged/ambiguous decisions fail.
- Embedding query: contract repair edit packet exact target contract proof write paths bounded context

## RPR-015 Project admitted packets into precise supervisor tasks

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: task-source
- Depends on: RPR-014
- Goal id: RPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/objectives/contract_repair_task_source.py, test/api/test_agent_supervisor_contract_repair_task_source.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_task_source.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/task-source
- Parallel lane: rpr-task-source
- Resource class: cpu-small
- Resource stage: materialize
- Token class: medium
- Estimated tokens: 14000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/objectives/contract_repair_task_source.py, test/api/test_agent_supervisor_contract_repair_task_source.py
- AST symbols: ContractRepairTaskSource, ContractRepairTaskProjection
- Interfaces: ContractRepairEditPacket@2, ObjectiveTaskRecord
- Allow concurrent with: RPR-016
- Conflict policy: Own only the new task source/test; do not modify contract_mismatch_refinery.py or taskboard control artifacts.
- Preconditions: @2 edit packet schema and canonical identity are stable.
- Effects: Repair packets become deduplicated bounded tasks with exact file ownership and validation.
- Evidence subset: decision id, strategy, target, contract table, proof refs, predicted files, interfaces, effects, postconditions
- Acceptance: Projection is deterministic/idempotent and binds packet/decision/tree ids; task predicted files and write scope exactly equal the packet; prompt states precise sender/receiver contract, selected strategy, target reason, unsupported limits, and validation/re-proof; rejected/ambiguous/stale packets produce no implementation task; provider cannot widen outputs; duplicate finding/decision does not duplicate task.
- Embedding query: project repair packet supervisor task exact files contract target proof

## RPR-016 Reject stale or unproved targets before provider invocation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-gate
- Depends on: RPR-014
- Goal id: RPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/contract_repair_pre_provider_gate.py, test/api/test_agent_supervisor_contract_repair_pre_provider_gate.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_pre_provider_gate.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/provider-gate
- Parallel lane: rpr-provider-gate
- Resource class: cpu-small
- Resource stage: validation
- Token class: medium
- Estimated tokens: 14000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/contract_repair_pre_provider_gate.py, test/api/test_agent_supervisor_contract_repair_pre_provider_gate.py
- AST symbols: ContractRepairPreProviderGate, PreProviderGateReceipt
- Interfaces: ContractRepairEditPacket@2, RepairTargetDecision@1, RepositorySnapshot
- Allow concurrent with: RPR-015
- Conflict policy: Own only the new pre-provider gate/test; do not alter provider implementations or daemon dispatch yet.
- Preconditions: Packet and decision validators are available.
- Effects: Exact repository state and proof authority are checked immediately before any LLM sees the task.
- Evidence subset: current tree, target span/hash, index/model/config, translator/toolchain/policy, proof reconstruction, write authority
- Acceptance: Reject changed tree/overlay, target moved/removed, index/model/config drift, changed translator/toolchain/policy, expired/downgraded proof, incomplete capability, packet/decision mismatch, read-only/escaped write path, ambiguity, and abstention before provider invocation; success emits a bounded receipt and cannot broaden paths; checks do not execute untrusted source.
- Embedding query: pre provider gate stale target proof tree index policy write authority

## RPR-017 Integrate the @2 decision path into the existing repair flow

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: integration-cutover
- Depends on: RPR-015, RPR-016
- Goal id: RPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_edit_packet.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_mismatch_refinery.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_pipeline.py, test/api/test_agent_supervisor_contract_repair_integration.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_integration.py test/api/test_agent_supervisor_mcp_contract_edit_packet.py test/api/test_agent_supervisor_contract_mismatch_refinery.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/integration
- Parallel lane: rpr-integration-serialized
- Resource class: cpu-medium
- Resource stage: integration
- Token class: large
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_edit_packet.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_mismatch_refinery.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_pipeline.py, test/api/test_agent_supervisor_contract_repair_integration.py
- AST symbols: materialize_contract_edit_packet, ContractMismatchRefinery, AnalysisPipeline
- Interfaces: ContractRepairEditPacket@2, ContractRepairTaskSource, ContractRepairPreProviderGate
- Allow concurrent with:
- Conflict policy: Sole serialized owner of these existing shared files for this board; inspect active VFS/datasets work before editing; preserve legacy @1 exact affected-path semantics and tests; abort/rebase on overlapping active ownership.
- Preconditions: New-file @2 packet, task source, and provider gate are independently tested; active shared-file ownership is clear.
- Effects: Analysis findings can opt into proof-gated target selection before packet write scope is fixed, while legacy callers remain compatible.
- Evidence subset: feature-gated pipeline route, @1 regression, @2 decision authority, task projection, pre-provider rejection
- Acceptance: Add an explicit feature-gated @2 route that inserts candidate retrieval/proof/admission before packet materialization; @2 write paths derive from RepairTargetDecision rather than finding.affected_paths; @1 continues requiring affected_paths equality; refinery accepts @2 only after decision validation and never lets provider expand scope; analysis pipeline preserves no-provider-before-admission; all legacy packet/refinery tests pass plus rename-to-moved-file, new-site, stale, ambiguous, read-only, and incompatible-decoy integrations.
- Embedding query: integrate proof gated target decision existing edit packet refinery analysis pipeline backward compatible

## RPR-018 Re-index, re-resolve, and re-prove candidate patches

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: post-edit-validation
- Depends on: RPR-010, RPR-017
- Goal id: RPR-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/contract_repair_validation.py, test/api/test_agent_supervisor_contract_repair_validation.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_validation.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/post-edit
- Parallel lane: rpr-post-edit
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/contract_repair_validation.py, test/api/test_agent_supervisor_contract_repair_validation.py
- AST symbols: ContractRepairValidator, ContractRepairCompletionReceipt
- Interfaces: ContractRepairEditPacket@2, ContractRepairProver, CompletionEvidence
- Allow concurrent with:
- Conflict policy: Own only the new validation module/test; invoke existing index/resolver/proof/type/test adapters without weakening their policies.
- Preconditions: Candidate patch/tree and original packet/decision/proof roots are available.
- Effects: Completion is bound to actual repaired code and the affected dependency closure.
- Evidence subset: rebuilt indexes, original edge resolution, re-extracted contracts, re-proved obligations, focused/impacted tests, type/effect/resource/memory checks
- Acceptance: Rebuild affected source/AST/vector rows and tombstones; re-resolve original edge; re-extract contracts; re-run original and introduced obligations; enforce policy-selected type/schema/error/effect/capability/lifecycle/resource/memory tools; run focused and dependency-complete impacted tests; detect deleted/weakened contracts/tests/checkers, suppressed findings, omitted dependants, stale candidate tree, and skipped required tool; only current complete receipts close the original finding.
- Embedding query: validate repaired patch reindex resolve reextract reprove impacted tests completion receipt

## RPR-019 Measure proof-gated retrieval and repair safety

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: benchmark
- Depends on: RPR-004, RPR-012, RPR-018
- Goal id: RPR-G090
- Outputs: test/api/test_agent_supervisor_contract_repair_benchmark.py, scripts/benchmark_contract_repair.py, data/agent_supervisor/proof_gated_contract_repair/benchmark/.gitkeep
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_benchmark.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/benchmark
- Parallel lane: rpr-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: test/api/test_agent_supervisor_contract_repair_benchmark.py, scripts/benchmark_contract_repair.py, data/agent_supervisor/proof_gated_contract_repair/benchmark/.gitkeep
- AST symbols: ContractRepairBenchmark, BenchmarkMetrics
- Interfaces: fixture manifest, CandidateNominationReceipt, RepairTargetDecision@1, ContractRepairCompletionReceipt
- Allow concurrent with:
- Conflict policy: Own benchmark runner/test/output placeholder; benchmark result files remain runtime artifacts unless explicitly reviewed.
- Preconditions: Fixture corpus and end-to-end validation path are stable.
- Effects: Release decisions use reproducible quality, safety, latency, cache, token, and context metrics.
- Evidence subset: recall@K, proof-eligible recall, target precision, wrong-path rate, false admission, abstention, rename precision, repair success, stale/poison rejection, latency/cache/tokens
- Acceptance: Deterministic benchmark runs all fixture families and records exact code/index/model/translator/toolchain/policy roots; safety floors require wrong-path automated mutation, failed-obligation override, stale/forged/poison authoritative admission, and unsupported memory-safe promotion each equal zero; report distinguishes nomination failure, proof abstention, target error, implementation error, and validation failure; repeated clean runs produce equivalent metrics/identities.
- Embedding query: benchmark contract repair recall precision wrong path false admission abstention latency tokens

## RPR-020 Add operations, metrics, feature flags, and rollback

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: rollout
- Depends on: RPR-018, RPR-019
- Goal id: RPR-G100
- Outputs: scripts/validate_proof_gated_contract_repair.py, docs/guides/PROOF_GATED_CONTRACT_REPAIR_GUIDE.md, test/api/test_agent_supervisor_contract_repair_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_rollout.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/rollout
- Parallel lane: rpr-rollout
- Resource class: cpu-small
- Resource stage: rollout
- Token class: medium
- Estimated tokens: 16000
- Implementation timeout seconds: 7200
- Predicted files: scripts/validate_proof_gated_contract_repair.py, docs/guides/PROOF_GATED_CONTRACT_REPAIR_GUIDE.md, test/api/test_agent_supervisor_contract_repair_rollout.py
- AST symbols: ContractRepairRolloutPolicy, ContractRepairMetrics
- Interfaces: RepairTargetDecision@1, ContractRepairCompletionReceipt, BenchmarkMetrics
- Allow concurrent with:
- Conflict policy: Own only RPR validation CLI, guide, and rollout tests; do not default-enable automated mutation.
- Preconditions: Post-edit validation and safety benchmark meet reviewed floors.
- Effects: Operators can doctor, inspect, replay, shadow, assist, narrowly automate, and roll back proof-gated repair.
- Evidence subset: capability health, decision dispositions, reason codes, cache/proof/retrieval metrics, feature flag, release/rollback receipts
- Acceptance: Validation CLI checks plan/objective/task DAG, exact source bindings, capability health, supervisor/process state, and benchmark floors; shadow is default; assist and narrow-auto require explicit scoped policy; auto is initially limited to unique reconstructed supported substitutions/renames; capability regression, stale root, reconstruction failure, or metric breach rolls back; guide states model boundaries and that vector/test/type/resource evidence does not prove memory safety.
- Embedding query: operations doctor status replay shadow assist auto rollback contract repair metrics
