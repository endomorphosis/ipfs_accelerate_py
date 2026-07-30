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

Intentional contract changes use the same proof boundary:

```text
base/candidate delta -> program graph/index nomination
  -> reverse transitive impact closure plus unknown frontier
  -> one migration obligation per consumer
  -> value/behavior nomination
  -> ipfs_datasets_py.logic proof/refutation/reconstruction
  -> deterministic analytical transform
  -> exact atomic propagation plan
  -> bounded llm_router fallback for admitted unresolved steps
  -> transactional implementation -> fixed-point validation
```

Knowledge graphs, vectors, history, runtime witnesses, and LLM output remain
non-authoritative. No automatic plan is complete while a required impacted
consumer or frontier is unresolved.

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

- Status: completed
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

- Status: completed
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

- Status: completed
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

- Status: completed
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

- Status: completed
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

- Status: completed
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

- Status: completed
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

## RPR-021 Seal the transitive change-propagation control plane

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: control
- Depends on: RPR-020
- Goal id: RPR-G110
- Outputs: docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md, docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md, docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md, config/agent_supervisor_proof_gated_contract_repair_scheduler.json, scripts/proof_gated_contract_repair_supervisor.sh
- Validation: python scripts/validate_proof_gated_contract_repair.py check-dag
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-control
- Parallel lane: control
- Resource class: cpu-small
- Resource stage: analysis
- Token class: small
- Estimated tokens: 6000
- Implementation timeout seconds: 1800
- Predicted files: docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md, docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md, docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md, config/agent_supervisor_proof_gated_contract_repair_scheduler.json, scripts/proof_gated_contract_repair_supervisor.sh
- AST symbols: ProgramContractDelta, ImpactClosureReceipt, MissingInputRequirement, RequiredBehaviorContract, AtomicPropagationPlan
- Interfaces: ProofGatedChangePropagationPlan@1
- Allow concurrent with:
- Conflict policy: Planning/control artifacts are protected after this committed seal; no pending worker may edit them.
- Preconditions: RPR-000 through RPR-020 are complete and the exact accelerator/datasets checkout is healthy.
- Effects: A committed parseable extension with four file-disjoint strict-shard entry tasks can launch reproducibly.
- Evidence subset: architecture, objective heap, task dependency DAG, scheduler propagation policy, launcher doctor
- Acceptance: Extension goals RPR-G110 through RPR-G220 and tasks RPR-022 through RPR-047 are present; all dependencies and goal references resolve; RPR-022, RPR-023, RPR-024, and RPR-025 are the only initially ready tasks and map to four distinct numeric shards; vector/graph/LLM authority is false and analytical/proof/atomic/fixed-point gates are launch-enforced.
- Embedding query: seal transitive change propagation taskboard four shards proof atomic fixed point

## RPR-022 Define bounded change-propagation records

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-contracts
- Depends on: RPR-021
- Goal id: RPR-G120
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_contracts.py, test/api/test_agent_supervisor_change_propagation_contracts.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_contracts.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-contracts
- Parallel lane: rpr-propagation-contracts
- Resource class: cpu-small
- Resource stage: analysis
- Token class: large
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_contracts.py, test/api/test_agent_supervisor_change_propagation_contracts.py
- AST symbols: ProgramChangeSet, ProgramContractDelta, ImpactClosureReceipt, ConsumerMigrationObligation, MissingInputRequirement, RequiredBehaviorContract, AtomicPropagationPlan, PropagationCompletionReceipt
- Interfaces: RepositorySnapshot, ProgramContract@1, MemorySafetyFacet@1, ProofReceipt, RetrievalSnapshotBinding
- Allow concurrent with: RPR-023, RPR-024, RPR-025
- Conflict policy: Own only the new propagation contract module/test; import existing canonical identity, bounds, ProgramContract, assurance, and receipt types without editing them.
- Preconditions: Existing repository identity, ProgramContract, MemorySafetyFacet, proof, vector binding, and canonical serialization helpers are importable.
- Effects: Every propagation stage exchanges immutable bounded records with exact state, authority, completeness, and invalidation.
- Evidence subset: canonical identities, base/candidate roots, closed dispositions, bounds, authority, state machines, forged/stale rejection
- Acceptance: Define finite validated schemas for change set, semantic delta, graph node/edge refs, impact closure/frontier, per-consumer obligation, missing input, value candidate, behavior contract, analytical transform, plan/step/SCC group, transaction, and completion; bind base/candidate forest/tree/overlay plus graph/index/model/config/translator/toolchain/policy roots; forbid source bodies, unbounded values, unsafe paths, forged ids, authority promotion, invalid state combinations, plans without complete consumer dispositions, and completion without a fixed-point receipt.
- Embedding query: typed change propagation contracts delta impact closure missing input behavior atomic plan completion

## RPR-023 Bind exact graph, dataflow, logic, vector, and llm_router capabilities

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-capabilities
- Depends on: RPR-021
- Goal id: RPR-G120
- Outputs: ipfs_accelerate_py/agent_supervisor/integrations/change_propagation_capabilities.py, test/api/test_agent_supervisor_change_propagation_capabilities.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_capabilities.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-capabilities
- Parallel lane: rpr-propagation-capabilities
- Resource class: cpu-small
- Resource stage: analysis
- Token class: large
- Estimated tokens: 16000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/integrations/change_propagation_capabilities.py, test/api/test_agent_supervisor_change_propagation_capabilities.py
- AST symbols: ChangePropagationCapability, ChangePropagationCapabilityReport, probe_change_propagation_capabilities
- Interfaces: ProgramGraph, ProgramCallResolver, CodeImpactIndex, SemanticDependencyGraph, CodeSymbolVectorIndex, IPFSDatasetsAnalysisProvider, IPFSDatasetsLogicProvider, LLMRouter
- Allow concurrent with: RPR-022, RPR-024, RPR-025
- Conflict policy: Own the new capability adapter/test; do not edit accelerator or datasets provider foundations and never install tools during probing.
- Preconditions: Launcher pins imports to this checkout and the exact initialized ipfs_datasets_py gitlink.
- Effects: Workers know exactly which graph, dataflow, proof, reconstruction, vector, provider-router, type, and runtime semantics are available.
- Evidence subset: module paths, git revisions, schema versions, operations, executable versions, supported languages and semantics, timeouts
- Acceptance: Lazy cold probes bind concrete module file, revision, interface/schema, producer/toolchain, operation, and supported-semantics identities for repository/AST indexes, program graph/call resolver, impact graph, value provenance, vector search, datasets GraphRAG/premise selection/LogicIR/TDFOL/CEC/SMT/Hammer/reconstruction, cvc5/Z3, Python/Node/TypeScript/mypy, and canonical llm_router/provider receipt APIs; missing, partial, incompatible, or timed-out capabilities yield typed unavailable diagnostics; package presence and solver/model candidates grant no authority; no network or auto-install.
- Embedding query: exact capability program graph dataflow logic theorem vector llm router lazy probe

## RPR-024 Build the adversarial transitive-change fixture corpus

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-fixtures
- Depends on: RPR-021
- Goal id: RPR-G220
- Outputs: test/fixtures/agent_supervisor/change_propagation, test/api/test_agent_supervisor_change_propagation_fixtures.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_fixtures.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-fixtures
- Parallel lane: rpr-propagation-fixtures
- Resource class: cpu-small
- Resource stage: analysis
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: test/fixtures/agent_supervisor/change_propagation, test/api/test_agent_supervisor_change_propagation_fixtures.py
- AST symbols: ChangePropagationFixture, ChangePropagationFixtureManifest
- Interfaces: fixture manifest schema only
- Allow concurrent with: RPR-022, RPR-023, RPR-025
- Conflict policy: Own only the new fixture directory and validator test; fixtures contain no production implementation or network/credential behavior.
- Preconditions: Fixtures are tiny, deterministic, hermetic, source-rooted, and state expected closure, proof, plan, and completion dispositions.
- Effects: Every later task shares positive, negative, ambiguous, stale, poisoned, dynamic, partial, and second-order change cases.
- Evidence subset: added argument, threading, provenance, schemas, constructor, state machine, SCC, reflection, FFI, poison, rollback, fixed point
- Acceptance: Manifest covers two-to-three argument change across direct/aliased/wrapped/method callers; unique in-scope value; same-typed wrong information; branch-local and nullable values; parameter threading; config/DI/factory construction; schema/serializer/generated client changes; new class/method/data structure and stateful service; async/error/effect/auth/resource/lifetime drift; dependency cycle/SCC; reflection/plugin/registry/FFI frontier; stale graph/vector/proof; poisoned retrieval; read-only/cross-repository paths; partial transaction; LLM scope escape; weakened test; and a repair that creates a second-order breaking delta; expectations never grant vector/KG/LLM semantic authority.
- Embedding query: adversarial change propagation missing third argument wrong value stateful class SCC dynamic poison fixed point

## RPR-025 Implement a snapshot-bound typed program dependency graph

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: program-graph
- Depends on: RPR-021
- Goal id: RPR-G130
- Outputs: ipfs_accelerate_py/agent_supervisor/program_graph.py, ipfs_accelerate_py/agent_supervisor/program_call_resolver.py, ipfs_accelerate_py/agent_supervisor/analysis/program_dependency_graph.py, test/api/test_agent_supervisor_program_dependency_graph.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_dependency_graph.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/program-dependency-graph
- Parallel lane: rpr-program-graph
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_graph.py, ipfs_accelerate_py/agent_supervisor/program_call_resolver.py, ipfs_accelerate_py/agent_supervisor/analysis/program_dependency_graph.py, test/api/test_agent_supervisor_program_dependency_graph.py
- AST symbols: ProgramGraph, ProgramCallResolver, ProgramDependencyGraph, ProgramNode, ProgramEdge, ProgramGraphSnapshot
- Interfaces: RepositoryIndexer, AnalysisASTIndex, CodeImpactIndex, SemanticDependencyGraph, ProgramGraph, ProgramCallResolver
- Allow concurrent with: RPR-022, RPR-023, RPR-024
- Conflict policy: Own concrete graph/resolver façades, the new adapter, and its test; consume existing repository/AST/evidence/dependency graphs without editing or forking them.
- Preconditions: Existing repository/AST indexes provide exact source facts and existing graph types preserve trusted versus nominated provenance.
- Effects: The existing capability probes gain concrete root-bound whole-program call/dependency interfaces with explicit incomplete frontiers.
- Evidence subset: definitions, calls, overrides, factories, DI, registries, imports, data/state flow, schemas, public surfaces, tests, ownership, dynamic frontier
- Acceptance: Graph identity binds forest/tree/overlay, coverage, included/excluded/generated/native roots, extractor/config/toolchain, and tombstones; typed nodes/edges cover supported declarations, calls, override/implementation, constructors/factories/builders, DI/registries/callbacks/decorators, import/export/alias, parameters/returns/fields/data/state flow, schemas/serializers/migrations, API/RPC/CLI/config, tests/mocks/fixtures/docs, build/generated/native boundaries, ownership and validation; resolver returns resolved/ambiguous/dynamic/external/unsupported with bounded frontier; GraphRAG/runtime/vector edges remain nominated; incremental rebuild equals clean rebuild on fixtures.
- Embedding query: snapshot program dependency graph call resolver data flow schema factory registry impact

## RPR-026 Compute exact before-and-after semantic contract deltas

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contract-delta
- Depends on: RPR-022, RPR-023, RPR-024, RPR-025
- Goal id: RPR-G150
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_contract_delta.py, test/api/test_agent_supervisor_program_contract_delta.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_contract_delta.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/program-contract-delta
- Parallel lane: rpr-contract-delta
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/program_contract_delta.py, test/api/test_agent_supervisor_program_contract_delta.py
- AST symbols: ProgramContractDeltaAnalyzer, ProgramContractDelta, ContractClauseDelta, DeltaDisposition
- Interfaces: ProgramChangeSet@1, ProgramContract@1, ExpectedProgramContract, ObservedProgramContract, ProgramGraph
- Allow concurrent with: RPR-028, RPR-029, RPR-031
- Conflict policy: Own the new semantic-delta analyzer/test; import and return the canonical RPR-022 ProgramContractDelta@1 and reuse ProgramContract@1 without redefining either record or editing expected/observed extraction.
- Preconditions: Exact base/candidate snapshots, change records, capability report, program graph, and fixture expectations are available.
- Effects: Textual edits become typed consumer-domain contract changes rather than unstructured diff prompts.
- Evidence subset: parameter and result clauses, schemas, behavior, visibility, lineage, compatibility, unknown semantics, exact roots
- Acceptance: Normalize moves, renames, formatting, comments, and generated churn separately from semantic changes; detect parameter add/remove/rename/reorder/default/keyword/variance, result/generic/nullability/schema/serialization/protocol, sync/async/cancellation, error/effect/capability/auth, lifecycle/state/consistency/resource/memory, visibility/registration and class/method/field/factory changes; classify each clause compatible/breaking/behavioral/unknown/unsupported for an explicit consumer domain; expected behavior follows reviewed evidence precedence and cannot be self-authored by the candidate implementation; stale, incomplete, cross-root, and unsupported comparisons fail closed.
- Embedding query: semantic before after ProgramContract delta added parameter schema effect behavior

## RPR-027 Index value and behavior candidates without granting authority

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: value-index
- Depends on: RPR-022, RPR-024, RPR-025, RPR-026
- Goal id: RPR-G130
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/change_value_vector_index.py, test/api/test_agent_supervisor_change_value_vector_index.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_value_vector_index.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-value-index
- Parallel lane: rpr-value-index
- Resource class: cpu-medium
- Resource stage: retrieval
- Token class: large
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/change_value_vector_index.py, test/api/test_agent_supervisor_change_value_vector_index.py
- AST symbols: ChangeValueIndexRow, ChangeValueIndexSnapshot, ChangeValueQuery, ChangeValueHit
- Interfaces: CodeSymbolVectorIndex, ProgramDependencyGraph, ProgramContractDelta, RetrievalSnapshotBinding
- Allow concurrent with: RPR-028, RPR-029, RPR-030, RPR-031
- Conflict policy: Own only the new value/behavior vector adapter/test; consume the code-symbol index and graph by reference and never alter their semantic_authority=false rule.
- Preconditions: Typed program graph, contract delta schema, fixture corpus, and existing snapshot-bound vector primitives are available.
- Effects: Missing-input and behavior searches gain bounded high-recall nomination across values, constructors, factories, schemas, tests, documentation, and history.
- Evidence subset: qualified value, type/schema, definition/use context, scope, constructor/factory, effects, ownership, docs/tests, lineage, model roots
- Acceptance: Rows contain bounded metadata and source/AST/graph refs rather than bodies; identity binds forest/tree, coverage, chunk/normalization/model/revision/dimensions/metric/config and tombstones; queries include exact missing contract and consumer context; hits retain signal provenance and semantic_authority=false; incremental update equals clean rebuild; stale/cross-tree/poisoned/forged/dimension mismatch and incomplete results fail; same-typed or semantically similar values receive no compatibility claim.
- Embedding query: vector index missing argument value factory behavior schema provenance non authoritative

## RPR-028 Compute reverse transitive impact closure and SCCs

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: impact-closure
- Depends on: RPR-022, RPR-024, RPR-025
- Goal id: RPR-G140
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py, test/api/test_agent_supervisor_contract_change_impact.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_change_impact.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/contract-change-impact
- Parallel lane: rpr-impact-closure
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py, test/api/test_agent_supervisor_contract_change_impact.py
- AST symbols: ContractChangeImpactAnalyzer, ImpactClosureReceipt, ImpactConsumer, ImpactSCC
- Interfaces: ProgramContractDelta, ProgramDependencyGraph, CodeImpactIndex, SemanticDependencyGraph
- Allow concurrent with: RPR-026, RPR-029, RPR-031
- Conflict policy: Own the new impact analyzer/test; import and return the canonical RPR-022 ImpactClosureReceipt@1 and reuse graph traversal/CodeImpactIndex helpers without redefining the record or treating nominated edges as trusted.
- Preconditions: Propagation contracts, exact program graph, and fixture graph/delta records are available; tests may construct typed deltas directly before RPR-026 integration.
- Effects: Each semantic delta has a deterministic bounded reverse dependency worklist, cycle groups, validations, and coverage receipt.
- Evidence subset: seed clauses, reverse edges, consumer paths, SCCs, exclusions, bounds, graph trust, required validation, coverage
- Acceptance: Traverse supported calls, wrappers, overrides, implementations, constructors/factories, data/state flow, schemas/serializers/migrations, imports/exports/registries, API/RPC/CLI/config, generated bindings, tests/mocks/fixtures/docs and ownership/build dependencies to a fixed point; deduplicate consumers while retaining all paths; compute deterministic SCCs and topological condensation; record exclusions, bounds, required validations, graph roots and an unknown frontier; no nominated GraphRAG/vector/runtime edge closes coverage; truncation, stale graph, unresolved required route, and forged completeness cannot yield complete.
- Embedding query: reverse transitive code impact closure dependency consumer SCC complete coverage

## RPR-029 Inventory compatibility at every affected call site

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: callsite-inventory
- Depends on: RPR-022, RPR-024, RPR-025
- Goal id: RPR-G150
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/change_consumer_inventory.py, test/api/test_agent_supervisor_change_consumer_inventory.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_consumer_inventory.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-consumer-inventory
- Parallel lane: rpr-consumer-inventory
- Resource class: cpu-medium
- Resource stage: analysis
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/change_consumer_inventory.py, test/api/test_agent_supervisor_change_consumer_inventory.py
- AST symbols: ChangeConsumerInventory, ConsumerCompatibilityLedger, ConsumerMigrationObligation
- Interfaces: ProgramGraph, ProgramCallResolver, ProgramContractDelta, CallRequirementContract
- Allow concurrent with: RPR-026, RPR-028, RPR-031
- Conflict policy: Own the new callsite inventory/builder/test; import and return canonical RPR-022 ConsumerMigrationObligation@1 records and do not redefine them or change graph/sender-receiver foundations.
- Preconditions: Propagation contract types, program graph/call resolver, and callsite fixtures are available; semantic deltas may be supplied as typed fixture records.
- Effects: A changed callable produces one explicit compatibility disposition and migration obligation per resolved caller and route.
- Evidence subset: caller span, alias/dispatch route, actual args, defaults, awaitedness, result uses, errors/effects, path condition, consumer contract
- Acceptance: Enumerate direct, aliased, re-exported, wrapped, decorated, callback, overload, method/override, factory, test/mock and generated-client calls; record actual positional/keyword/splat arguments, defaults, receiver state, path condition, awaitedness, result uses, handled errors/effects/capabilities and exact route; a two-to-three required-argument change flags every still-two-argument caller independently; one compatible caller or callee default cannot discharge others; ambiguous/dynamic calls remain frontier records; duplicate paths do not duplicate obligations.
- Embedding query: every call site compatibility added third argument caller inventory obligation

## RPR-030 Analyze schema, constructor, serialization, and protocol impacts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: schema-impact
- Depends on: RPR-022, RPR-024, RPR-025, RPR-026
- Goal id: RPR-G150
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py, test/api/test_agent_supervisor_schema_protocol_change_impact.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_schema_protocol_change_impact.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/schema-protocol-impact
- Parallel lane: rpr-schema-impact
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py, test/api/test_agent_supervisor_schema_protocol_change_impact.py
- AST symbols: SchemaProtocolChangeAnalyzer, ConstructorImpact, SerializationImpact, ProtocolImpact
- Interfaces: ProgramContractDelta, ProgramDependencyGraph, ConsumerMigrationObligation
- Allow concurrent with: RPR-027, RPR-028, RPR-029, RPR-031
- Conflict policy: Own the new schema/protocol analyzer/test; consume graph and delta contracts without editing generated artifacts or external schemas.
- Preconditions: Typed semantic deltas, program graph, propagation records, and schema/protocol fixtures exist.
- Effects: Data-shape and support-type changes account for constructors, serializers, storage/messages, public protocols, generated bindings, migrations, and compatibility.
- Evidence subset: type fields, constructors, factories, defaults, codecs, persistence, versioning, IDL, generated roots, migrations
- Acceptance: Detect added/removed/renamed/retyped fields and variants; constructor/factory/builder changes; JSON/protobuf/IDL/database/message/RPC/HTTP/CLI schemas; serializers/deserializers, persistence, cache keys, equality/hash, version negotiation, migration and generated clients; distinguish backward/forward/full/incompatible/unknown per consumer; required defaults and migrations need independent authority; generated/read-only roots produce regeneration or external obligations rather than direct writes; missing or dynamic codecs remain frontiers.
- Embedding query: schema constructor serialization protocol data structure field change migration impact

## RPR-031 Preserve dynamic, reflection, registry, generated, and FFI frontiers

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: dynamic-frontier
- Depends on: RPR-022, RPR-023, RPR-024, RPR-025
- Goal id: RPR-G140
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/dynamic_impact_frontier.py, test/api/test_agent_supervisor_dynamic_impact_frontier.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_dynamic_impact_frontier.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/dynamic-impact-frontier
- Parallel lane: rpr-dynamic-frontier
- Resource class: cpu-medium
- Resource stage: analysis
- Token class: large
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/dynamic_impact_frontier.py, test/api/test_agent_supervisor_dynamic_impact_frontier.py
- AST symbols: DynamicImpactFrontierAnalyzer, ImpactFrontierEntry, FrontierDisposition
- Interfaces: ProgramGraph, ProgramCallResolver, ChangePropagationCapabilityReport, ImpactClosureReceipt
- Allow concurrent with: RPR-026, RPR-028, RPR-029
- Conflict policy: Own the new frontier adapter/test; runtime witnesses, manifests, GraphRAG and model annotations remain proposals until admitted by exact policy.
- Preconditions: Concrete graph/resolver, capability report, and adversarial dynamic/generated/native fixtures exist.
- Effects: Coverage reports cannot silently lose consumers hidden behind unsupported or runtime-only mechanisms.
- Evidence subset: reflection, string dispatch, monkey patch, plugins, registries, callbacks, code generation, remote service, FFI, exclusions, runtime witness
- Acceptance: Emit bounded entries for reflection/introspection, getattr/eval/import strings, monkey patches, plugins/entry points, runtime DI/registries, callbacks, generated code, native extensions/FFI, remote services, vendored/read-only/excluded roots and unbounded resource limits; record route, affected contract, evidence, supported closure mechanisms and reason; reviewed manifests or root-bound runtime witnesses may close only the observed route under policy; vector/KG/LLM claims cannot; absent evidence and timeout remain unknown; complete impact is impossible while a required entry is open.
- Embedding query: dynamic reflection registry plugin generated FFI unknown impact frontier fail closed

## RPR-032 Nominate missing-input and construction routes

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: missing-input-retrieval
- Depends on: RPR-027, RPR-028, RPR-029, RPR-030, RPR-031
- Goal id: RPR-G160
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/missing_input_candidate_retrieval.py, test/api/test_agent_supervisor_missing_input_candidate_retrieval.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_missing_input_candidate_retrieval.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/missing-input-retrieval
- Parallel lane: rpr-missing-input-retrieval
- Resource class: cpu-medium
- Resource stage: retrieval
- Token class: large
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/missing_input_candidate_retrieval.py, test/api/test_agent_supervisor_missing_input_candidate_retrieval.py
- AST symbols: MissingInputCandidateRetriever, MissingInputQuery, ValueProvenanceCandidate, ConstructionRouteCandidate
- Interfaces: MissingInputRequirement@1, ProgramDependencyGraph, ChangeValueVectorIndex, IPFSDatasetsAnalysisProvider
- Allow concurrent with: RPR-033
- Conflict policy: Own the new retrieval adapter/test; graph, vector, lexical, history, GraphRAG, runtime, spec and test signals nominate only and no winner or code path is chosen.
- Preconditions: Impact closure, callsite/schema ledgers, dynamic frontiers, and value/behavior index snapshots bind the same repository roots.
- Effects: Each missing argument or support dependency has a bounded union of existing, threaded, configured, injected, derived, factory, constructor, adapter, or new-behavior candidates.
- Evidence subset: lexical, AST, scope, graph, history, vector, specs, tests, config, DI, factory, constructor, route provenance
- Acceptance: Union exact in-scope symbols, receiver state, caller parameters, constants/defaults, request/session context, reaching-definition hints, reviewed config/env providers, DI/registry providers, factories/builders/constructors, schemas, lineage, authoritative specs/tests, lexical/BM25, graph and vector hits; bind complete query/candidate set identity and per-signal refs; cap results and redact bodies/secrets; distinguish reuse/thread/convert/construct/new-behavior routes; stale/poisoned/cross-root/forged results reject; every candidate has semantic_authority=false and retrieval cannot assert compatibility, placement, or write scope.
- Embedding query: missing argument candidate retrieval value constructor factory dependency injection vector graph

## RPR-033 Compile reaching definitions, dominance, path conditions, and value provenance

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: value-provenance
- Depends on: RPR-028, RPR-029, RPR-030, RPR-031
- Goal id: RPR-G160
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/value_provenance_graph.py, test/api/test_agent_supervisor_value_provenance_graph.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_value_provenance_graph.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/value-provenance
- Parallel lane: rpr-value-provenance
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 28000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/value_provenance_graph.py, test/api/test_agent_supervisor_value_provenance_graph.py
- AST symbols: ValueProvenanceGraph, ReachingDefinition, DominanceFact, PathCondition, InformationProvenance
- Interfaces: ProgramDependencyGraph, ProgramCallResolver, MissingInputRequirement@1, MemorySafetyFacet@1
- Allow concurrent with: RPR-032
- Conflict policy: Own the new provenance compiler/test; extend graph evidence by reference and preserve alias/dynamic/unsupported uncertainty instead of guessing.
- Preconditions: Exact program graph, impact closure, consumer ledgers, schema impacts, and frontier records exist.
- Effects: Logic obligations can distinguish a merely assignable name from a value that exists on every path and contains the required information safely.
- Evidence subset: definitions, uses, SSA-like versions, aliases, dominance, branches, guards, type refinements, field flow, interprocedural threading, ownership
- Acceptance: Compile bounded intraprocedural reaching definitions, def-use chains, dominance/post-dominance, path conditions, guards and type refinements; track parameters, returns, fields, aliases, constructors, conversions, config/DI sources and interprocedural threading with explicit completeness; attach type/schema/range/nullability, information-origin labels, effects/capabilities/auth, ownership/lifetime/mutation/concurrency and dependency direction; prove only supported AST/control-flow shapes; branch-local absence, alias ambiguity, loops beyond bounds, exceptions, concurrency, reflection/native calls and incomplete interprocedural routes remain unknown; exact root and producer identities prevent stale reuse.
- Embedding query: reaching definitions dominance path condition data flow value information provenance missing input

## RPR-034 Synthesize required behavior for new support types

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: behavior-synthesis
- Depends on: RPR-026, RPR-030, RPR-032, RPR-033
- Goal id: RPR-G170
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/required_behavior_synthesis.py, test/api/test_agent_supervisor_required_behavior_synthesis.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_required_behavior_synthesis.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/required-behavior
- Parallel lane: rpr-behavior-synthesis
- Resource class: cpu-proof-solver
- Resource stage: analysis
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/required_behavior_synthesis.py, test/api/test_agent_supervisor_required_behavior_synthesis.py
- AST symbols: RequiredBehaviorSynthesizer, RequiredBehaviorContract, BehaviorEvidencePrecedence, BehaviorGap
- Interfaces: ProgramContractDelta, MissingInputRequirement@1, ValueProvenanceGraph, ProgramContract@1, MemorySafetyFacet@1
- Allow concurrent with:
- Conflict policy: Own the new behavior synthesizer/test; import and return the canonical RPR-022 RequiredBehaviorContract@1 and do not redefine it; expected behavior cannot come from candidate implementation or LLM opinion.
- Preconditions: Exact semantic delta, schema/protocol impacts, missing-input candidates, and value provenance are available.
- Effects: A needed class, method, data structure, provider, factory, or schema has a precise reviewable contract before placement or implementation.
- Evidence subset: IDL/schema/stubs, normative specs, conformance tests, caller and callee clauses, invariants, state, lifecycle, ownership, migrations
- Acceptance: Apply explicit precedence across reviewed IDL/schema/public stubs, normative specs/conformance tests, caller postconditions/callee preconditions, data invariants/migration manifests/architecture ownership/history, and non-authoritative observations; define fields/variants/generics/invariants/defaults, constructors/factories/totality, methods/state machine/transitions/idempotence, ownership/lifetime/mutation/concurrency/cache/disposal, serialization/persistence/versioning/migrations/equality/hash, errors/cancellation/effects/capabilities/auth/trust/privacy/resources/degradation, compatibility/tests/telemetry; state assumptions and unsupported clauses; conflicting or insufficient evidence yields a typed behavior gap and no implementation request.
- Embedding query: required behavior contract new class method data structure invariant state lifecycle

## RPR-035 Compile change-propagation LogicIR obligations

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-obligations
- Depends on: RPR-028, RPR-029, RPR-030, RPR-031, RPR-033, RPR-034
- Goal id: RPR-G180
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/change_propagation_obligations.py, test/api/test_agent_supervisor_change_propagation_obligations.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_obligations.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-obligations
- Parallel lane: rpr-propagation-obligations
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/change_propagation_obligations.py, test/api/test_agent_supervisor_change_propagation_obligations.py
- AST symbols: ChangePropagationObligationCompiler, ChangePropagationObligation, ValueMappingClaim, BehaviorRefinementClaim
- Interfaces: ProgramContractDelta, ConsumerMigrationObligation@1, ValueProvenanceGraph, RequiredBehaviorContract@1, ProofObligation
- Allow concurrent with:
- Conflict policy: Own the propagation-specific compiler/test; use existing proof contracts and datasets translation adapters without editing generic VFS or datasets logic foundations.
- Preconditions: Impact/consumer/frontier ledgers, provenance facts, schema impacts, and required behavior contracts bind one source snapshot.
- Effects: Every automated migration choice is expressed as explicit finite premises, conclusions, assumptions, unsupported semantics, and counterexample targets.
- Evidence subset: scope, control flow, type/schema, information, constructor, error/effect/capability/auth, ownership/lifetime, cycle, behavior, placement
- Acceptance: Lower separate obligations for closure coverage, consumer compatibility, source scope and path availability, type/schema/range/nullability, information sufficiency, conversion/constructor totality, error/effect/capability/auth/trust/resource, ownership/lifetime/mutation/concurrency, dependency cycles, parameter threading, behavior invariants/state transitions, serialization/migration and placement; bind source/premise/assumption/tree/graph/translator/toolchain/policy ids; unsupported dynamic or native semantics stay explicit; no retrieved/model statement becomes an axiom; obligations are bounded and deterministic.
- Embedding query: change propagation LogicIR obligation value mapping behavior refinement theorem

## RPR-036 Prove, refute, and reconstruct missing-value and behavior mappings

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-proof
- Depends on: RPR-023, RPR-035
- Goal id: RPR-G180
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/missing_input_synthesis.py, test/api/test_agent_supervisor_missing_input_synthesis.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_missing_input_synthesis.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/missing-input-synthesis
- Parallel lane: rpr-propagation-proof
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 32000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/missing_input_synthesis.py, test/api/test_agent_supervisor_missing_input_synthesis.py
- AST symbols: MissingInputSynthesizer, SynthesisDisposition, ValueMappingProof, BehaviorProofSet
- Interfaces: ChangePropagationObligationCompiler, IPFSDatasetsLogicProvider, MultiProverRouter, ProofReconstructor, ProofReceipt
- Allow concurrent with:
- Conflict policy: Own the propagation synthesis/prover adapter/test; solver, premise-selection, vector, graph and LLM results are candidates until independent kernel reconstruction.
- Preconditions: Exact capability admission and compiled bounded obligations are available under one translator/toolchain/policy identity.
- Effects: Each missing value or behavior clause has a unique-proved, refuted, ambiguous, unknown, timeout, or unsupported analytical disposition with current receipts.
- Evidence subset: deterministic premises, solver attempts, counterexamples, reconstruction, cache identity, competing candidates, unsupported clauses
- Acceptance: Route finite obligations through admitted ipfs_datasets_py LogicIR/TDFOL/CEC/SMT/Hammer capabilities; reconstruct successful candidates under exact premises and kernel/toolchain roots; validate cached receipts against all invalidators; prove no candidate, exactly one candidate, or multiple candidates without turning search order into uniqueness; preserve minimal counterexamples and unsatisfied clauses; one proved value can thread a new upstream requirement only with its origin; unknown/timeout/missing backend/incomplete slice/unsupported/stale/failed reconstruction remain non-conclusive and produce no code authority.
- Embedding query: theorem prove refute reconstruct missing third argument behavior mapping ipfs datasets logic

## RPR-037 Implement deterministic analytical change transforms

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: analytical-transforms
- Depends on: RPR-032, RPR-033, RPR-036
- Goal id: RPR-G180
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/analytical_change_transforms.py, test/api/test_agent_supervisor_analytical_change_transforms.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_analytical_change_transforms.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/analytical-change-transforms
- Parallel lane: rpr-analytical-transforms
- Resource class: cpu-large
- Resource stage: planning
- Token class: large
- Estimated tokens: 28000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/analytical_change_transforms.py, test/api/test_agent_supervisor_analytical_change_transforms.py
- AST symbols: AnalyticalChangeTransformer, AnalyticalTransform, TransformDisposition
- Interfaces: ValueMappingProof, ConsumerMigrationObligation@1, ProgramASTAdapter, RepairTargetDecision@1
- Allow concurrent with: RPR-038
- Conflict policy: Own the new deterministic transform builder/test; import and return the canonical RPR-022 AnalyticalTransform@1 rather than redefining it; emit plans but do not execute them or edit source under test.
- Preconditions: Current reconstructed synthesis receipts, exact AST/source spans, and bounded candidate routes are available.
- Effects: Unique supported repairs are rendered mechanically before any model escalation.
- Evidence subset: proof id, transform kind, exact span, before hash, replacement, imports, postconditions, replay identity
- Acceptance: Support only closed replayable Python shapes for add/rename/reorder argument from a unique proved expression, thread a parameter through admitted routes, add an allowed import/export/registration, finite adapter/conversion, typed constructor/factory update, and authorized schema/fixture mapping; preserve formatting and idempotency; bind exact before hashes, roots, proof and expected after identity; reject dynamic splats, ambiguous overloads, unsupported syntax, stale spans, non-total mappings, new dependencies, scope escape and transforms requiring invented behavior; repeated rendering is byte-equivalent.
- Embedding query: deterministic analytical codemod add argument thread parameter exact proof

## RPR-038 Prove placement for new classes, methods, and data structures

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: support-placement
- Depends on: RPR-011, RPR-034, RPR-036
- Goal id: RPR-G170
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/support_behavior_placement.py, test/api/test_agent_supervisor_support_behavior_placement.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_support_behavior_placement.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/support-behavior-placement
- Parallel lane: rpr-support-placement
- Resource class: cpu-proof-solver
- Resource stage: planning
- Token class: large
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/support_behavior_placement.py, test/api/test_agent_supervisor_support_behavior_placement.py
- AST symbols: SupportBehaviorPlacement, SupportPlacementCandidate, SupportPlacementDecision
- Interfaces: RequiredBehaviorContract@1, ImplementationSiteAdmissibility, ProgramDependencyGraph, ProofReceipt
- Allow concurrent with: RPR-037
- Conflict policy: Own the new placement adapter/test; reuse RPR implementation-site admissibility and no target path is authorized by vector/KG/LLM nomination.
- Preconditions: Independently sourced required behavior, reconstructed proof clauses, program ownership graph, and exact write policy are available.
- Effects: Complex new support behavior has one proved owner/wiring site or an explicit ambiguous/unsupported/review-only disposition.
- Evidence subset: declaration/architecture anchor, ownership, visibility, layering, dependency cycles, registration/export, generated/read-only, capability support
- Acceptance: Enumerate bounded candidates from declarations, interfaces, architecture ownership, factories/providers, schemas and existing placement anchors; prove language/runtime, module owner, visibility, dependency direction/acyclicity, registration/export/DI wiring, capability/effect/resource/memory support, generated/vendor/read-only exclusion, exact mutation authority and behavior-contract fit; reuse an existing admissible implementation before creating a duplicate; unique eligible candidate and margin are required; ties, missing owner, cross-root writes, cycles, unsupported lifecycle/native semantics, or unproved behavior yield abstention; selection alone defines exact placement paths.
- Embedding query: new class method data structure placement ownership dependency cycle proof

## RPR-039 Admit one complete atomic transitive repair plan

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-plan
- Depends on: RPR-028, RPR-036, RPR-037, RPR-038
- Goal id: RPR-G190
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/change_propagation_plan.py, test/api/test_agent_supervisor_change_propagation_plan.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_plan.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-plan
- Parallel lane: rpr-propagation-plan
- Resource class: cpu-large
- Resource stage: planning
- Token class: large
- Estimated tokens: 32000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/change_propagation_plan.py, test/api/test_agent_supervisor_change_propagation_plan.py
- AST symbols: ChangePropagationPlanner, AtomicPropagationPlan, PropagationPlanStep, PropagationPlanAdmission
- Interfaces: ImpactClosureReceipt@1, ValueMappingProof, AnalyticalTransform@1, SupportPlacementDecision, RepairTargetDecision@1
- Allow concurrent with:
- Conflict policy: Own the new plan builder/admission/test; import and return the canonical RPR-022 AtomicPropagationPlan@1 rather than redefining it; construction cannot edit code, execute providers, or expand paths beyond admitted evidence.
- Preconditions: Complete current impact closure, proof set, analytical transform set, and complex behavior placement decisions are available.
- Effects: Every affected consumer and support behavior is covered by one deterministic dependency-ordered transaction plan or explicit abstention.
- Evidence subset: complete candidate set, consumer dispositions, exact paths, steps, DAG, SCC groups, proofs, checkpoints, rollback, validation, invalidators
- Acceptance: Plan identity binds source/candidate roots, complete delta/closure/frontier/consumer/candidate/proof/transform/placement sets, graph/index/model/translator/toolchain/policy, exact read/write spans, step DAG, SCC transaction groups, pre/postconditions, validation commands, resource bounds, checkpoint/rollback and fixed-point obligation; each mandatory consumer has exactly one disposition and each write derives from authority; deterministic ordering and replay are stable; omissions, duplicates, competing mappings/sites, failed proofs, unresolved required frontier, stale roots, forbidden/cross-root paths, cycle outside an SCC group, invalid validation, or two equally valid plans yield abstention.
- Embedding query: admit atomic multi file transitive repair plan every consumer SCC rollback

## RPR-040 Materialize plan-bound multi-edit packets

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-packet
- Depends on: RPR-039
- Goal id: RPR-G200
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/change_propagation_edit_packet.py, test/api/test_agent_supervisor_change_propagation_edit_packet.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_edit_packet.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-packet
- Parallel lane: rpr-propagation-packet
- Resource class: cpu-medium
- Resource stage: materialize
- Token class: large
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/change_propagation_edit_packet.py, test/api/test_agent_supervisor_change_propagation_edit_packet.py
- AST symbols: ChangePropagationEditPacket, PropagationEditStep, PropagationExpansionHandle
- Interfaces: AtomicPropagationPlan@1, AnalyticalTransform@1, ContractRepairEditPacket@2, ProofReceipt
- Allow concurrent with:
- Conflict policy: Own the new multi-edit packet/test; do not weaken existing repair packet versions or allow packet content to expand admitted plan scope.
- Preconditions: A current admitted non-abstaining atomic propagation plan and all referenced proof/transform receipts are available.
- Effects: Analytical and model-required steps receive compact exact behavior, authority, dependency, and validation context.
- Evidence subset: plan/step ids, exact paths/spans, before hashes, contract delta, value mappings, behavior, counterexamples, proof refs, postconditions
- Acceptance: Materialize only from one current admitted plan; partition deterministic analytical steps from behavior-complete model-required steps; packet binds all roots, plan/SCC/dependency order, exact read/write allowlists, before hashes, selected values/sources, required behavior, minimal counterexamples, proof/index/graph refs, unsupported limits, per-edit and fixed-point postconditions, focused commands and bounded expansion handles; alternatives, source/proof bodies, secrets, and unknown semantics cannot broaden scope; stale/forged/partial/abstaining plans and path mismatches fail.
- Embedding query: atomic change propagation edit packet exact paths behavior proof llm step

## RPR-041 Route only admitted unresolved steps through llm_router

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-provider
- Depends on: RPR-040
- Goal id: RPR-G200
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/change_propagation_provider_router.py, test/api/test_agent_supervisor_change_propagation_provider_router.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_provider_router.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-provider
- Parallel lane: rpr-propagation-provider
- Resource class: cpu-medium
- Resource stage: materialize
- Token class: large
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/todo_daemon/change_propagation_provider_router.py, test/api/test_agent_supervisor_change_propagation_provider_router.py
- AST symbols: ChangePropagationProviderRouter, PropagationProviderEnvelope, PropagationProposalReceipt
- Interfaces: ChangePropagationEditPacket@1, ContractPacketProviderRouter, LLMRouter, ProviderReceipt, WriterLease
- Allow concurrent with:
- Conflict policy: Own the new propagation router/test; delegate to the existing bounded provider/reviewer/writer-lease path and canonical accelerator llm_router without direct datasets/model calls.
- Preconditions: Plan-bound packet schema and current exact provider capability receipts are available.
- Effects: Only syntax or bounded implementation gaps with already admitted semantics reach a model as redacted proposal-only requests.
- Evidence subset: analytical attempt disposition, admitted behavior, exact lease paths, prompt bounds, provider/model/config, review, proposal receipt
- Acceptance: Require an analytical non-success reason that is supported and behavior-complete; build a bounded redacted prompt with exact contract delta, chosen value mappings, behavior clauses, counterexamples, paths, postconditions and validations; call canonical llm_router through existing routing/receipt APIs; enforce time/token/context/tool/path bounds, provider/model/config identity and writer lease; model cannot choose a value source, behavior, owner, dependency, consumer set, plan order or path; proposed diff is untrusted until deterministic scope parsing, review, admission and post-edit proof; timeout/unavailable/refusal/malformed/scope escape creates no write.
- Embedding query: bounded llm router fallback admitted behavior exact write lease proposal only

## RPR-042 Project propagation tasks and gate immediately before providers

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-task-gate
- Depends on: RPR-040, RPR-041
- Goal id: RPR-G200
- Outputs: ipfs_accelerate_py/agent_supervisor/objectives/change_propagation_task_source.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_pre_provider_gate.py, test/api/test_agent_supervisor_change_propagation_task_source.py, test/api/test_agent_supervisor_change_propagation_pre_provider_gate.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_task_source.py test/api/test_agent_supervisor_change_propagation_pre_provider_gate.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-task-gate
- Parallel lane: rpr-propagation-task-gate
- Resource class: cpu-medium
- Resource stage: materialize
- Token class: large
- Estimated tokens: 28000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/objectives/change_propagation_task_source.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_pre_provider_gate.py, test/api/test_agent_supervisor_change_propagation_task_source.py, test/api/test_agent_supervisor_change_propagation_pre_provider_gate.py
- AST symbols: ChangePropagationTaskSource, ChangePropagationTaskProjection, ChangePropagationPreProviderGate, PropagationGateReceipt
- Interfaces: ChangePropagationEditPacket@1, AtomicPropagationPlan@1, ObjectiveTaskRecord, ProviderReceipt, WriterLease
- Allow concurrent with:
- Conflict policy: Own only the new task source/gate modules and tests; shared pipeline/refinery/daemon integration is reserved for RPR-043.
- Preconditions: Exact packet and provider-router contracts are stable and current repository/proof probes are callable without executing untrusted source.
- Effects: Plan steps become deterministic deduplicated supervisor tasks and stale or unproved work is rejected at the last pre-model boundary.
- Evidence subset: plan/packet/step/SCC/tree ids, exact files, behavior contract, proof refs, invalidators, provider capability, lease, validation
- Acceptance: Projection is deterministic/idempotent and task outputs exactly equal packet step authority; SCC order and dependency metadata are preserved; prompts expose admitted values/behavior, unsupported limits and fixed-point validation; duplicate plans do not duplicate tasks; analytical steps never invoke a provider; pre-provider gate revalidates tree/overlay, graph/index/model/config, target spans/hashes, translator/toolchain/policy, proof reconstruction, plan/packet completeness, provider identity, path lease and frontier; any drift, abstention, partial group, proof downgrade, escaped/read-only path or incomplete behavior blocks before llm_router.
- Embedding query: change propagation task source pre provider gate stale proof exact paths

## RPR-043 Implement checkpointed transactions and fixed-point validation primitives

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-validation
- Depends on: RPR-018, RPR-042
- Goal id: RPR-G210
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/change_propagation_transaction.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_validation.py, test/api/test_agent_supervisor_change_propagation_transaction.py, test/api/test_agent_supervisor_change_propagation_validation.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_transaction.py test/api/test_agent_supervisor_change_propagation_validation.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-validation
- Parallel lane: rpr-propagation-validation
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 36000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/change_propagation_transaction.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_validation.py, test/api/test_agent_supervisor_change_propagation_transaction.py, test/api/test_agent_supervisor_change_propagation_validation.py
- AST symbols: ChangePropagationTransaction, PropagationCheckpoint, PropagationRollbackReceipt, ChangePropagationValidator
- Interfaces: AtomicPropagationPlan@1, ChangePropagationEditPacket@1, ContractRepairValidator, CompletionEvidence, PropagationCompletionReceipt@1
- Allow concurrent with:
- Conflict policy: Own the new transaction/validation modules and tests; do not edit shared runtime pipelines yet and execute tests only in isolated explicit candidate roots.
- Preconditions: Plan-bound analytical or reviewed model diffs with exact receipts are available and existing patch-bound contract validation can be reused.
- Effects: The shared integration task receives tested all-or-rollback primitives and a concrete candidate-tree fixed-point completion gate.
- Evidence subset: checkpoint, step/group receipts, rollback, rebuilt indexes/graphs, redelta, reclosure, obligations, tests, fixed-point iterations
- Acceptance: Import and return the canonical RPR-022 AtomicPropagationPlan@1 and PropagationCompletionReceipt@1 records rather than redefining them; create a content-addressed checkpoint before mutation; verify each before hash and lease; execute dependency order with each SCC as one transaction group; on failure/drift/timeout/scope escape restore checkpoint and retain diagnostics; prevent partial merge/completion; validator rebuilds repository/AST/vector/graph rows/tombstones, re-extracts delta, re-resolves calls/data/schema/wiring, recomputes closure/frontier, verifies every original consumer once, discovers second-order impacts, reconstructs proofs and runs dependency-complete policy tools/tests; iterate to a policy bound; zero unresolved mandatory consumers, omitted resolved dependents, uncovered required frontier and unplanned breaking delta is required; bound exhaustion, weakened/deleted checks and skipped required tools fail.
- Embedding query: transaction primitive SCC checkpoint rollback fixed point validation reindex reprove

## RPR-044 Integrate transactional propagation and require fixed-point completion

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-integration
- Depends on: RPR-017, RPR-043
- Goal id: RPR-G210
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_pipeline.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_mismatch_refinery.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/contract_packet_provider_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_change_propagation_integration.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_integration.py test/api/test_agent_supervisor_change_propagation_transaction.py test/api/test_agent_supervisor_change_propagation_validation.py test/api/test_agent_supervisor_contract_repair_integration.py test/api/test_agent_supervisor_contract_mismatch_refinery.py test/api/test_agent_supervisor_contract_packet_provider_router.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-integration
- Parallel lane: rpr-propagation-integration-serialized
- Resource class: cpu-large
- Resource stage: materialize
- Token class: large
- Estimated tokens: 38000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_pipeline.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_mismatch_refinery.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/contract_packet_provider_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_change_propagation_integration.py
- AST symbols: ChangePropagationPipeline, AnalysisPipeline, ContractMismatchRefinery, ContractPacketProviderRouter, PortalImplementationDaemon
- Interfaces: ChangePropagationTaskSource, ChangePropagationPreProviderGate, ChangePropagationProviderRouter, ChangePropagationTransaction, ChangePropagationValidator, existing @1/@2 contract repair flow
- Allow concurrent with:
- Conflict policy: This is the single serialized shared-file cutover; preserve legacy entry points/tests, require the tested transaction and validator on the new path, and do not edit control-plane artifacts.
- Preconditions: All new-file adapters, packets, tasks, gates, provider routing, transaction, rollback and fixed-point validation are independently tested; existing RPR @2 integration is complete.
- Effects: The supervisor can execute an admitted intentional semantic change only through checkpointed mutation and candidate-tree fixed-point completion while retaining old broken-contract behavior.
- Evidence subset: feature-gated entry, base/candidate binding, pipeline order, analytical-first route, task dedupe, provider receipt, transaction, completion gate, legacy compatibility
- Acceptance: Add a versioned feature-gated flow that captures change set, extracts delta, builds graph/index, computes closure/frontier, inventories consumers, retrieves/proves values and behavior, admits an atomic plan, emits analytical or model steps, and rechecks the pre-provider gate; all mutations must invoke ChangePropagationTransaction and completion must invoke ChangePropagationValidator with its current PropagationCompletionReceipt; no direct or partial daemon write path may bypass checkpoint/SCC rollback/fixed-point proof; model steps use canonical bounded provider logic and analytical success makes no provider call; task/writer scopes equal admitted paths; failures preserve abstention and rollback; @1/@2 repair callers, daemon parsing, provider receipts and legacy tests remain compatible; import remains lazy and cold.
- Embedding query: integrate transactional change propagation pipeline daemon fixed point analytical llm router

## RPR-045 Benchmark adversarial transitive-change safety

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-benchmark
- Depends on: RPR-024, RPR-044
- Goal id: RPR-G220
- Outputs: test/api/test_agent_supervisor_change_propagation_benchmark.py, scripts/benchmark_change_propagation.py, data/agent_supervisor/proof_gated_change_propagation/benchmark/.gitkeep
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_benchmark.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-benchmark
- Parallel lane: rpr-propagation-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 28000
- Implementation timeout seconds: 7200
- Predicted files: test/api/test_agent_supervisor_change_propagation_benchmark.py, scripts/benchmark_change_propagation.py, data/agent_supervisor/proof_gated_change_propagation/benchmark/.gitkeep
- AST symbols: ChangePropagationBenchmark, ChangePropagationBenchmarkMetrics
- Interfaces: ChangePropagationFixtureManifest, AtomicPropagationPlan@1, PropagationCompletionReceipt@1
- Allow concurrent with:
- Conflict policy: Own benchmark runner/test/output placeholder; generated reports remain runtime artifacts unless separately reviewed.
- Preconditions: Adversarial corpus and end-to-end transaction/fixed-point validation are stable.
- Effects: Release decisions measure coverage, proof quality, plan safety, transactional integrity, completion correctness, latency, tokens, and context.
- Evidence subset: delta accuracy, impact recall, frontier quality, value-source precision, behavior authority, plan precision, rollback, fixed point, latency
- Acceptance: Deterministically run every fixture family with exact code/graph/index/model/translator/toolchain/policy roots; distinguish delta miss, graph miss, open frontier, missed consumer, retrieval miss, proof abstention, wrong value, behavior/placement error, plan omission, implementation error, rollback error, and false completion; record impact recall, consumer precision, proof-eligible value recall, unique-source precision, abstention, analytical coverage, LLM rate/scope escape, plan completeness, SCC rollback, iterations, closure success, latency/cache/tokens/context; safety floors require missed resolved impacted consumer, unproved/wrong value source, invented behavior, partial propagation completion, stale graph/index plan admission, and false fixed-point completion each zero alongside all legacy floors; repeated clean runs are identity-equivalent.
- Embedding query: benchmark transitive change impact missing caller wrong value atomic rollback fixed point safety

## RPR-046 Add propagation metrics, rollout flags, CLI, guide, and rollback

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: propagation-rollout
- Depends on: RPR-020, RPR-045
- Goal id: RPR-G220
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/change_propagation_rollout.py, scripts/validate_change_propagation.py, docs/guides/PROOF_GATED_CHANGE_PROPAGATION_GUIDE.md, test/api/test_agent_supervisor_change_propagation_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_rollout.py
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-rollout
- Parallel lane: rpr-propagation-rollout
- Resource class: cpu-small
- Resource stage: rollout
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/change_propagation_rollout.py, scripts/validate_change_propagation.py, docs/guides/PROOF_GATED_CHANGE_PROPAGATION_GUIDE.md, test/api/test_agent_supervisor_change_propagation_rollout.py
- AST symbols: ChangePropagationRolloutPolicy, ChangePropagationMetrics, ChangePropagationRollbackGate
- Interfaces: ChangePropagationBenchmarkMetrics, AtomicPropagationPlan@1, PropagationCompletionReceipt@1
- Allow concurrent with:
- Conflict policy: Own the new rollout module/CLI/guide/test; do not edit protected control-plane artifacts or default-enable model-authored, complex, public-schema, dynamic, generated, native, or cross-repository mutation.
- Preconditions: Benchmark meets reviewed legacy and propagation safety floors.
- Effects: Operators can doctor, inspect, replay, shadow, assist, narrowly automate analytical changes, and roll back on capability, coverage, proof, or metric regression.
- Evidence subset: capability/coverage health, delta/closure/plan dispositions, reason codes, benchmark floors, feature scope, release/rollback receipts
- Acceptance: CLI validates exact sources, capabilities, graph/index coverage, proof reconstruction, transaction health and benchmark floors; metrics expose every benchmark stage plus analytical/model split, tokens/context and fixed-point iterations; shadow is default; assist requires explicit policy; narrow-auto is limited to complete-frontier unique reconstructed analytical supported-Python transforms; model-authored, stateful behavior, public schema/API, dynamic/generated/native and cross-root changes remain approval-gated; stale roots, open frontier, capability regression, proof loss, wrong-value/missed-consumer/partial-plan/false-completion or any floor breach rolls back; guide documents trust, safety, memory, transaction and recovery boundaries.
- Embedding query: change propagation operations metrics shadow assist narrow auto rollback guide

## RPR-047 Extend end-to-end operations validation for propagation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: propagation-operations
- Depends on: RPR-046
- Goal id: RPR-G220
- Outputs: scripts/validate_proof_gated_contract_repair.py, test/api/test_agent_supervisor_contract_repair_rollout.py, test/api/test_agent_supervisor_change_propagation_end_to_end.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_rollout.py test/api/test_agent_supervisor_change_propagation_end_to_end.py && python scripts/validate_proof_gated_contract_repair.py --check-all
- Board namespace: agent-supervisor-proof-gated-contract-repair-v1
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-operations
- Parallel lane: rpr-propagation-operations
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 28000
- Implementation timeout seconds: 7200
- Predicted files: scripts/validate_proof_gated_contract_repair.py, test/api/test_agent_supervisor_contract_repair_rollout.py, test/api/test_agent_supervisor_change_propagation_end_to_end.py
- AST symbols: ProofGatedContractRepairOperations, ChangePropagationEndToEnd
- Interfaces: RPR-G110, RPR-G220, RPR-047, ChangePropagationRolloutPolicy, PropagationCompletionReceipt@1
- Allow concurrent with:
- Conflict policy: Own only the existing operations validator/test cutover and new end-to-end test; preserve RPR-020/RPR-G100 behavior and do not edit the protected plan/objective/taskboard/scheduler/launcher.
- Preconditions: Propagation rollout CLI, metrics, guide, benchmark, transaction, and fixed-point validation are complete.
- Effects: The canonical operations surface treats transitive propagation as required program work and can prove the final board and runtime are healthy.
- Evidence subset: terminal goal/task ids, full DAG, scheduler propagation policy, all safety floors, exact bindings, supervisor health, seeded end-to-end trace
- Acceptance: Extend validation without removing legacy RPR-020/RPR-G100 checks; require RPR-G110/RPR-G220 and terminal RPR-047, correct dependency chain, change_propagation_policy gates and six new zero safety floors; verify protected paths/refill isolation and exact source bindings; seeded two-to-three argument case detects all callers, proves one source or threads it, applies an atomic analytical plan, rediffs to a fixed point and emits completion; negative wrong-value, unknown-frontier, partial-SCC and LLM-scope cases fail; stopped/running supervisor health remains correctly reported and a clean four-shard board can drain.
- Embedding query: end to end operations validate change propagation taskboard supervisor fixed point
