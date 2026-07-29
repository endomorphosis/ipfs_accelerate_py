# Agent Supervisor Proof-Gated Contract Repair Objective Heap

This is the durable, machine-ingestible source of intent for the
proof-gated contract-repair program. The executable projection is
`agent_supervisor_proof_gated_contract_repair.todo.md`; the normative
architecture is `AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md`.

Program invariants:

- Vector, lexical, graph, history, test, and LLM results nominate or explain;
  they do not authorize a target.
- An admitted `ipfs_datasets_py.logic` proof/reconstruction result precedes
  eligibility, reranking, write-path selection, and provider invocation.
- Expected behavior is independent of the candidate implementation.
- Stale, ambiguous, unsupported, incomplete, or unreconstructed evidence
  fails closed.
- `ProgramContract@1.max_memory_bytes` is not memory-safety evidence; ownership,
  lifetime, unsafe, FFI, and allocator claims use `MemorySafetyFacet@1`.
- Active VFS and datasets programs retain ownership of their generic graph,
  resolver, contract, solver, finding, and repair-packet foundations.
- Cross-program capabilities are probed and version-bound; they are not unknown
  task dependencies on this board.
- Shared-file cutovers are serialized after new-file adapters are complete.

## RPR-G000 Deliver proof-gated target selection for broken code contracts

- Status: blocked
- Review only: true
- Parent:
- Priority: P0
- Track: proof-gated-contract-repair
- Bundle: agent-supervisor/proof-gated-contract-repair/control
- Parallel lane: control
- Conflict policy: Root is review/completion aggregation only; implementation belongs to child goals.
- Resource class: cpu-medium
- Token class: small
- Goal: For each broken call trace, derive a precise sender requirement, nominate possible moved receivers or insertion sites with a snapshot-bound code index, prove candidate compatibility or placement through ipfs_datasets_py logic, admit one exact target or abstain, and bind repair plus completion to the candidate tree.
- Evidence: RPR-G010, RPR-G020, RPR-G030, RPR-G040, RPR-G050, RPR-G060, RPR-G070, RPR-G080, RPR-G090, RPR-G100
- Outputs:
- Validation: python scripts/validate_proof_gated_contract_repair.py --check-all
- Acceptance: All mandatory child goals have current-tree evidence; adversarial wrong-path, failed-obligation override, stale/poison admission, and unsupported-memory-safety promotion rates are zero; a healthy isolated parallel supervisor can drain the board.
- Gap task: Review child-goal evidence and release gates only; do not perform an aggregate code edit.
- Refinement: Prefer abstention and explicit unsupported states to broad unearned correctness or memory-safety claims.
- Embedding query: broken code trace refactor rename vector code search sender receiver contract logic proof rerank exact write target
- AST query: BrokenContractTrace CallRequirementContract RepairCandidate RepairTargetDecision ContractRepairEditPacket

## RPR-G010 Define repair contracts, authority, and capability admission

- Status: active
- Parent: RPR-G000
- Priority: P0
- Track: contracts
- Bundle: agent-supervisor/proof-gated-contract-repair/contracts
- Parallel lane: rpr-contracts
- Conflict policy: Own only uniquely named RPR contracts/adapters; import ProgramContract@1 and existing assurance types without editing VFS-owned foundations.
- Resource class: cpu-small
- Token class: medium
- Goal: Define bounded content-addressed records for broken traces, sender requirements, memory-safety facets, repair candidates, target decisions, and upstream capability admission.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_contracts.py, ipfs_accelerate_py/agent_supervisor/integrations/contract_repair_capabilities.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_contracts.py, ipfs_accelerate_py/agent_supervisor/integrations/contract_repair_capabilities.py, test/api/test_agent_supervisor_contract_repair_contracts.py, test/api/test_agent_supervisor_contract_repair_capabilities.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_contracts.py test/api/test_agent_supervisor_contract_repair_capabilities.py
- Acceptance: Records bind exact tree/forest/graph/index/model/translator/toolchain/policy roots and bounded spans/references; invalid states, bodies, forged ids, unsupported promotions, write paths without a decision, and package-presence capability inference are rejected; missing upstream modules yield typed unavailable status.
- Gap task: Implement the RPR schema and capability boundary without creating a second ProgramContract or assurance lattice.
- Refinement: A capability is an exact compatible interface observation, not an installed-package guess.
- Embedding query: typed broken trace candidate target decision capability content identity bounds fail closed
- AST query: BrokenContractTrace CallRequirementContract MemorySafetyFacet RepairCandidate RepairTargetDecision CapabilityAdmissionReport

## RPR-G020 Build a snapshot-bound code-symbol vector index

- Status: active
- Parent: RPR-G000
- Priority: P0
- Track: code-index
- Bundle: agent-supervisor/proof-gated-contract-repair/code-index
- Parallel lane: rpr-code-index
- Conflict policy: Own the RPR code-symbol index and tests; do not edit RepositoryIndexer, AnalysisASTIndex, program_ast_adapters, or the objective todo-vector index.
- Resource class: cpu-medium
- Token class: medium
- Goal: Index bounded per-symbol signature, call, effect, error, documentation, test, ownership, and Git-lineage features with exact tree/model/config identities and incremental tombstones.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/code_symbol_vector_index.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/code_symbol_vector_index.py, test/api/test_agent_supervisor_code_symbol_vector_index.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_code_symbol_vector_index.py
- Acceptance: Rows are body-free and provenance-bearing; index roots bind tree, coverage, chunker, normalization, model/revision/dimensions/metric/config, inclusions, tombstones, and producer; stale, poisoned, cross-tree, dimension/config mismatch, forged lineage, and incomplete rebuild claims fail closed; vector results have no semantic authority.
- Gap task: Implement a code-specific index adapter over existing source/AST references with deterministic local fixtures.
- Refinement: This index embeds code symbols; the supervisor objective index embeds task text and is not reusable here.
- Embedding query: code symbol embedding index signature call graph effects git rename lineage tombstone snapshot
- AST query: CodeSymbolIndexRow CodeVectorIndexSnapshot CodeVectorQuery CodeVectorHit

## RPR-G030 Classify broken traces and recover candidate routes

- Status: active
- Parent: RPR-G000
- Depends on: RPR-G010
- Priority: P0
- Track: trace-analysis
- Bundle: agent-supervisor/proof-gated-contract-repair/trace
- Parallel lane: rpr-trace
- Conflict policy: Own the RPR trace adapter; consume program graph/resolver protocols when compatible and preserve unavailable/ambiguous frontiers.
- Resource class: cpu-medium
- Token class: medium
- Goal: Turn an unresolved or mismatched call site into a bounded BrokenContractTrace and distinguish moved/refactored, wiring drift, adapter-required, missing implementation, external, dynamic, ambiguous, and unsupported cases.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/broken_contract_trace.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/broken_contract_trace.py, test/api/test_agent_supervisor_broken_contract_trace.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_broken_contract_trace.py
- Acceptance: Exact caller span, call form, actual arguments, awaitedness, result uses, handled errors, effects/capabilities, graph frontier, completeness, and source roots are retained; same-name evidence does not manufacture a resolved edge; unavailable upstream resolver produces typed unsupported evidence rather than a guessed target.
- Gap task: Implement the conservative trace-classification adapter and seeded route fixtures.
- Refinement: An unresolved call remains unresolved until evidence closes the route.
- Embedding query: broken call trace unresolved function moved refactored alias reexport registration dynamic ambiguous
- AST query: BrokenTraceClassifier TraceDisposition ProgramCallResolver ProgramGraph

## RPR-G040 Derive precise sender, receiver, and memory-safety contracts

- Status: active
- Parent: RPR-G000
- Depends on: RPR-G010, RPR-G030
- Priority: P0
- Track: contract-synthesis
- Bundle: agent-supervisor/proof-gated-contract-repair/contract-synthesis
- Parallel lane: rpr-contract-synthesis
- Conflict policy: Own the RPR synthesis adapter; consume ProgramContract@1 and upstream extraction/checking through capability protocols without editing their files.
- Resource class: cpu-medium
- Token class: large
- Goal: Construct an independently sourced sender requirement and candidate receiver guarantees, including variance, errors, effects, lifecycle, resource bounds, and separately scoped ownership/lifetime/unsafe evidence.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/sender_receiver_contracts.py, ipfs_accelerate_py/agent_supervisor/analysis/memory_safety_facets.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/sender_receiver_contracts.py, ipfs_accelerate_py/agent_supervisor/analysis/memory_safety_facets.py, test/api/test_agent_supervisor_sender_receiver_contracts.py, test/api/test_agent_supervisor_memory_safety_facets.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_sender_receiver_contracts.py test/api/test_agent_supervisor_memory_safety_facets.py
- Acceptance: Inputs are contravariant, outputs covariant, and required/optional/keyword/overload/nullability/schema/sync/async/error/effect/capability/auth/temporal/resource clauses remain explicit; implementation observations cannot self-author expectations; unsupported reflection/FFI/lifetime semantics remain unknown; max_memory_bytes never becomes a memory-safety verdict.
- Gap task: Implement synthesis over reviewed evidence precedence and a separate memory-safety evidence facet.
- Refinement: Prove only the modeled language and toolchain subset.
- Embedding query: caller receiver precondition postcondition type variance errors effects async resource ownership lifetime memory safety
- AST query: SenderReceiverContractCompiler MemorySafetyFacet ExpectedProgramContract ObservedProgramContract

## RPR-G050 Nominate moved receivers and implementation sites

- Status: active
- Parent: RPR-G000
- Depends on: RPR-G020, RPR-G030, RPR-G040
- Priority: P0
- Track: candidate-retrieval
- Bundle: agent-supervisor/proof-gated-contract-repair/retrieval
- Parallel lane: rpr-retrieval
- Conflict policy: Own the RPR candidate retrieval adapter; existing analysis_retrieval remains context-only and VFS graph/query files remain untouched.
- Resource class: cpu-medium
- Token class: large
- Goal: Union exact history, structural fingerprints, resolver routes, dependency/ownership anchors, AST hints, lexical results, and vector results into a bounded snapshot-bound candidate set.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_candidate_retrieval.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_candidate_retrieval.py, test/api/test_agent_supervisor_contract_repair_candidate_retrieval.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_candidate_retrieval.py
- Acceptance: Candidate rows record evidence per signal and semantic_authority=false; pure rename, moved module, adapter, existing declaration, and new-site anchors are distinguished; same-name decoys, read-only targets, poisoned vectors, stale roots, partial results, and forged history are retained with rejection reasons or rejected; no winner or write path is chosen here.
- Gap task: Implement deterministic multi-signal nomination with fixed bounds and adversarial fixtures.
- Refinement: High recall precedes proof; retrieval is not adjudication.
- Embedding query: candidate retrieval rename refactor implementation site history graph ownership vector lexical
- AST query: ContractRepairCandidateRetriever RepairCandidate CodeVectorHit BoundRetrievalCandidate

## RPR-G060 Compile, solve, and reconstruct contract-repair obligations

- Status: active
- Parent: RPR-G000
- Depends on: RPR-G040, RPR-G050
- Priority: P0
- Track: logic-proof
- Bundle: agent-supervisor/proof-gated-contract-repair/proof
- Parallel lane: rpr-proof
- Conflict policy: Own RPR-specific obligation/prover adapters; use existing proof contracts, cache, and datasets provider; do not edit VFS code_contract_logic/prover/context files.
- Resource class: cpu-proof-solver
- Token class: large
- Goal: Lower supported substitution, rename-equivalence, adapter, and placement claims into exact ipfs_datasets_py logic obligations and admit only independently reconstructed current receipts.
- Evidence: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_obligations.py, ipfs_accelerate_py/agent_supervisor/proof/contract_repair_prover.py
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_obligations.py, ipfs_accelerate_py/agent_supervisor/proof/contract_repair_prover.py, test/api/test_agent_supervisor_contract_repair_obligations.py, test/api/test_agent_supervisor_contract_repair_prover.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_obligations.py test/api/test_agent_supervisor_contract_repair_prover.py
- Acceptance: Obligations bind source/premise/assumption/tree/translator/toolchain/policy ids; sender facts imply receiver preconditions and receiver guarantees imply caller requirements; error/effect/capability/lifecycle/resource/memory clauses are separate; pure rename needs bidirectional refinement plus lineage; solver candidates are non-authoritative until reconstructed; unknown, timeout, missing backend, incomplete slice, unsupported semantics, and stale cache remain non-conclusive.
- Gap task: Implement the RPR lowering and proof orchestration over capability-probed datasets logic.
- Refinement: One-way compatibility must not be mislabeled behavioral equivalence.
- Embedding query: logic obligation caller precondition receiver guarantee rename equivalence adapter placement cvc5 reconstruction
- AST query: ContractRepairObligationCompiler ContractRepairProver ProofObligation ProofReceipt

## RPR-G070 Admit and deterministically rank one repair target

- Status: active
- Parent: RPR-G000
- Depends on: RPR-G050, RPR-G060
- Priority: P0
- Track: adjudication
- Bundle: agent-supervisor/proof-gated-contract-repair/adjudication
- Parallel lane: rpr-adjudication
- Conflict policy: Own RPR eligibility/ranking/decision modules; do not mutate findings or packet materializers in this goal.
- Resource class: cpu-medium
- Token class: large
- Goal: Apply hard contract/proof/authority gates, then lexicographically rank eligible candidates and issue one exact, expiring target decision or abstention.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_reranker.py, ipfs_accelerate_py/agent_supervisor/planning/repair_target_admission.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/contract_repair_reranker.py, ipfs_accelerate_py/agent_supervisor/planning/repair_target_admission.py, test/api/test_agent_supervisor_contract_repair_reranker.py, test/api/test_agent_supervisor_repair_target_admission.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_reranker.py test/api/test_agent_supervisor_repair_target_admission.py
- Acceptance: Exact roots, target existence/placement, write authority, independent expectation, semantic completeness, reconstructed proof, and no counterexample are hard gates; score cannot override a failed obligation; ranking order is proof/coverage, lineage, graph/ownership, authoritative spec/test, AST, lexical, vector; ties/low margin/unsupported produce abstention; decision alone establishes exact read/write paths and invalidators.
- Gap task: Implement deterministic eligibility, ranking, target-decision identity, and replay validation.
- Refinement: The safest answer to insufficient discrimination is no automated write.
- Embedding query: proof gated rerank hard filter exact target decision write allowlist abstention margin
- AST query: ContractRepairReranker RepairTargetAdmission RepairTargetDecision

## RPR-G080 Hand admitted targets to the implementation supervisor

- Status: active
- Parent: RPR-G000
- Depends on: RPR-G070
- Priority: P0
- Track: supervisor-handoff
- Bundle: agent-supervisor/proof-gated-contract-repair/handoff
- Parallel lane: rpr-handoff-serialized
- Conflict policy: Serialized shared-file cutover; preserve @1 behavior for legacy callers and add an explicit @2 path rather than weakening existing packet validation.
- Resource class: cpu-medium
- Token class: large
- Goal: Materialize a versioned edit packet from an admitted target decision, project a precise repair task, and revalidate decision freshness before invoking the LLM.
- Evidence: ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_edit_packet.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_mismatch_refinery.py
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/contract_repair_edit_packet.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_repair_task_source.py, ipfs_accelerate_py/agent_supervisor/validation/contract_repair_pre_provider_gate.py, test/api/test_agent_supervisor_contract_repair_edit_packet.py, test/api/test_agent_supervisor_contract_repair_task_source.py, test/api/test_agent_supervisor_contract_repair_pre_provider_gate.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_edit_packet.py test/api/test_agent_supervisor_contract_repair_task_source.py test/api/test_agent_supervisor_contract_repair_pre_provider_gate.py
- Acceptance: @2 packets require an admitted current decision and derive write paths from it; compact prompt includes exact target, contract, proof/counterexample refs, postconditions, and validation; alternatives cannot expand scope; tree/index/target/translator/policy/proof drift blocks before provider invocation; @1 exact affected-path behavior remains intact until a separately reviewed cutover.
- Gap task: Implement new-file packet/task/gate adapters first; reserve any existing-file integration for a dedicated serialized follow-up.
- Refinement: The LLM implements an admitted decision; it does not choose the target.
- Embedding query: repair edit packet target decision exact write paths provider gate prompt contract
- AST query: ContractRepairEditPacket ContractRepairTaskSource ContractRepairPreProviderGate

## RPR-G090 Re-prove repairs and benchmark adversarial safety

- Status: active
- Parent: RPR-G000
- Depends on: RPR-G060, RPR-G080
- Priority: P0
- Track: validation
- Bundle: agent-supervisor/proof-gated-contract-repair/validation
- Parallel lane: rpr-validation
- Conflict policy: Own RPR post-edit validation and fixture/benchmark paths; do not weaken or skip policy-required tools.
- Resource class: cpu-large
- Token class: large
- Goal: Re-index, re-resolve, re-extract, re-prove, and run type/effect/resource/memory/test gates on the candidate tree, with an adversarial corpus measuring wrong-target and false-admission failures.
- Evidence: ipfs_accelerate_py/agent_supervisor/validation/contract_repair_validation.py, test/fixtures/agent_supervisor/contract_repair/manifest.json
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/contract_repair_validation.py, test/api/test_agent_supervisor_contract_repair_validation.py, test/fixtures/agent_supervisor/contract_repair/manifest.json, test/api/test_agent_supervisor_contract_repair_benchmark.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_validation.py test/api/test_agent_supervisor_contract_repair_benchmark.py
- Acceptance: Original edge resolves and every original/new obligation passes on the candidate tree; skipped required tools fail; dependency-complete impacted tests run; contract/test weakening and omitted dependants fail; corpus covers rename, move, decoys, poison, adapter, missing/new implementation, ambiguity, dynamic/FFI, stale roots, read-only targets, and tombstones; four safety floors are zero.
- Gap task: Implement patch-bound validation receipts and deterministic benchmark reporting.
- Refinement: A green provider run is not completion evidence without patch-bound reanalysis.
- Embedding query: patch bound reproof reindex resolver contract validation benchmark wrong file stale poison memory safety
- AST query: ContractRepairValidator ContractRepairBenchmark CompletionEvidence

## RPR-G100 Operate, observe, and safely roll out the feature

- Status: active
- Parent: RPR-G000
- Depends on: RPR-G080, RPR-G090
- Priority: P1
- Track: rollout
- Bundle: agent-supervisor/proof-gated-contract-repair/rollout
- Parallel lane: rpr-rollout
- Conflict policy: Own RPR CLI, metrics, operations, and feature-flag docs; no default auto-enable before safety floors.
- Resource class: cpu-small
- Token class: medium
- Goal: Provide doctor/status/replay commands, capability and decision metrics, shadow/assist/narrow-auto flags, rollback gates, and operator documentation.
- Evidence: scripts/proof_gated_contract_repair_supervisor.sh, docs/guides/PROOF_GATED_CONTRACT_REPAIR_GUIDE.md
- Outputs: scripts/validate_proof_gated_contract_repair.py, docs/guides/PROOF_GATED_CONTRACT_REPAIR_GUIDE.md, test/api/test_agent_supervisor_contract_repair_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_repair_rollout.py
- Acceptance: Shadow is default; assist and narrow-auto require explicit policy; capability regression, stale roots, proof reconstruction failure, or metric breach rolls back; metrics expose recall@K, proof-eligible recall, admitted precision, wrong-path rate, abstention, proof/cache latency, tokens, and context bytes; operations reproduce exact source bindings and supervisor health.
- Gap task: Implement validation CLI, feature flags, metrics, rollback, and operator guide after the proof-gated path exists.
- Refinement: Automation scope grows only after measured safety gates.
- Embedding query: shadow assist auto rollout rollback metrics supervisor doctor status replay
- AST query: ContractRepairRolloutPolicy ContractRepairMetrics validate_proof_gated_contract_repair
