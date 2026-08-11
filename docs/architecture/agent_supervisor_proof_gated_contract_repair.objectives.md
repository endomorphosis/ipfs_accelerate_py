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
- Every intentional semantic change has an exact base/candidate identity, a
  typed contract delta, one disposition per resolved transitive consumer, and
  an explicit unknown frontier.
- Knowledge graphs, vectors, history, runtime witnesses, and LLM proposals can
  nominate edges, values, behavior, or edits; none can independently authorize
  them.
- Missing values, constructors, and support behavior are proved or
  independently reconstructed before an atomic propagation plan is admitted.
- Analytical transforms precede bounded `llm_router` work; all edits are
  checkpointed and completion is a candidate-tree fixed point.

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
- Goal: For each broken call trace, derive a precise sender requirement, nominate possible moved receivers or insertion sites with a snapshot-bound code index, prove candidate compatibility or placement through ipfs_datasets_py logic, admit one exact target or abstain, and bind repair plus completion to the candidate tree; for intentional contract changes, compute the dependency-complete transitive impact closure, prove required value and behavior migrations, and admit one atomic fixed-point propagation plan.
- Evidence: RPR-G010, RPR-G020, RPR-G030, RPR-G040, RPR-G050, RPR-G060, RPR-G070, RPR-G080, RPR-G090, RPR-G100, RPR-G110
- Outputs:
- Validation: python scripts/validate_proof_gated_contract_repair.py --check-all
- Acceptance: All mandatory child goals have current-tree evidence; adversarial wrong-path, failed-obligation override, stale/poison admission, unsupported-memory-safety promotion, missed-consumer, unproved-value-source, invented-behavior, partial-plan, stale-impact-plan, and false-fixed-point completion rates are zero; a healthy isolated parallel supervisor can drain the board.
- Gap task: Review child-goal evidence and release gates only; do not perform an aggregate code edit.
- Refinement: Prefer abstention and explicit unsupported states to broad unearned correctness or memory-safety claims.
- Embedding query: broken code trace refactor rename vector code search sender receiver contract logic proof rerank exact write target transitive impact missing argument atomic propagation
- AST query: BrokenContractTrace CallRequirementContract RepairCandidate RepairTargetDecision ContractRepairEditPacket ProgramContractDelta ImpactClosureReceipt AtomicPropagationPlan

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

## RPR-G110 Deliver proof-gated transitive change propagation

- Status: blocked
- Review only: true
- Parent: RPR-G000
- Depends on: RPR-G100
- Priority: P0
- Track: change-propagation
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-control
- Parallel lane: control
- Conflict policy: Aggregate evidence and release gates only; implementation belongs to child goals and shared-file cutover remains serialized.
- Resource class: cpu-medium
- Token class: small
- Goal: Given an intentional semantic code change, identify every resolved direct and transitive consumer, preserve an explicit unknown frontier, derive and prove missing value or support behavior requirements, admit one exact atomic migration plan or abstain, prefer analytical transformations, bound any llm_router work to admitted behavior and paths, and validate the candidate tree to a fixed point.
- Evidence: RPR-G120, RPR-G130, RPR-G140, RPR-G150, RPR-G160, RPR-G170, RPR-G180, RPR-G190, RPR-G200, RPR-G210, RPR-G220
- Outputs:
- Validation: python scripts/validate_proof_gated_contract_repair.py check-dag
- Acceptance: All child goals have current-tree receipts; each resolved impacted consumer has exactly one disposition; unknown frontiers cannot be silently closed; all automated value mappings and behavior clauses have reconstructed authority; multi-file and SCC edits are atomic; analytical repair precedes model escalation; fixed-point validation meets every original and extension safety floor.
- Gap task: Review child-goal evidence and release gates only; do not perform an aggregate implementation edit.
- Refinement: A complete static closure is a proved coverage claim, not a synonym for a large search result.
- Embedding query: transitive code change impact missing argument value provenance theorem proof knowledge graph vector atomic migration fixed point
- AST query: ProgramContractDelta ImpactClosureReceipt MissingInputRequirement RequiredBehaviorContract AtomicPropagationPlan PropagationCompletionReceipt

## RPR-G120 Define propagation contracts and capability admission

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G100
- Priority: P0
- Track: propagation-foundations
- Bundle: agent-supervisor/proof-gated-contract-repair/change-propagation-contracts
- Parallel lane: rpr-propagation-contracts
- Conflict policy: Own uniquely named propagation contracts and capability adapters; reuse ProgramContract, repository identity, assurance, provider, and proof types without forking them.
- Resource class: cpu-small
- Token class: large
- Goal: Define bounded content-addressed records for change sets, deltas, impacted consumers, missing inputs, behavior contracts, plans, transactions, and completion, and probe exact graph, dataflow, logic, vector, and llm_router capabilities.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_contracts.py, ipfs_accelerate_py/agent_supervisor/integrations/change_propagation_capabilities.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_contracts.py, ipfs_accelerate_py/agent_supervisor/integrations/change_propagation_capabilities.py, test/api/test_agent_supervisor_change_propagation_contracts.py, test/api/test_agent_supervisor_change_propagation_capabilities.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_contracts.py test/api/test_agent_supervisor_change_propagation_capabilities.py
- Acceptance: Records bind exact base/candidate forest/tree/overlay plus graph/index/model/config/translator/toolchain/policy roots; state machines and bounds fail closed; capability is an exact lazy interface observation; unavailable or incompatible dataflow, solver, reconstruction, vector, graph, or llm_router support remains typed and cannot be inferred from package presence.
- Gap task: Implement extension contracts and capability probes without creating alternate identity, ProgramContract, graph-trust, or provider authority.
- Refinement: A typed unknown is preferable to a fabricated complete closure.
- Embedding query: change propagation immutable contracts capability exact roots bounds authority
- AST query: ProgramChangeSet ProgramContractDelta ImpactClosureReceipt MissingInputRequirement RequiredBehaviorContract AtomicPropagationPlan

## RPR-G130 Build the snapshot-bound program graph and value index

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G120
- Priority: P0
- Track: program-graph
- Bundle: agent-supervisor/proof-gated-contract-repair/change-program-graph
- Parallel lane: rpr-program-graph
- Conflict policy: Adapt RepositoryIndexer, AnalysisASTIndex, CodeImpactIndex, SemanticDependencyGraph, and CodeSymbolVectorIndex; do not fork their identity or grant graph/vector edges semantic authority.
- Resource class: cpu-large
- Token class: large
- Goal: Provide concrete ProgramGraph and ProgramCallResolver interfaces for typed call, data, state, schema, wiring, ownership, and validation edges, plus a snapshot-bound vector nomination index for values and behaviors.
- Evidence: ipfs_accelerate_py/agent_supervisor/program_graph.py, ipfs_accelerate_py/agent_supervisor/program_call_resolver.py, ipfs_accelerate_py/agent_supervisor/analysis/program_dependency_graph.py, ipfs_accelerate_py/agent_supervisor/analysis/change_value_vector_index.py
- Outputs: ipfs_accelerate_py/agent_supervisor/program_graph.py, ipfs_accelerate_py/agent_supervisor/program_call_resolver.py, ipfs_accelerate_py/agent_supervisor/analysis/program_dependency_graph.py, ipfs_accelerate_py/agent_supervisor/analysis/change_value_vector_index.py, test/api/test_agent_supervisor_program_dependency_graph.py, test/api/test_agent_supervisor_change_value_vector_index.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_dependency_graph.py test/api/test_agent_supervisor_change_value_vector_index.py
- Acceptance: Nodes and edges bind source/extractor roots and trusted or nominated provenance; calls, overrides, factories, DI, registries, imports, data/state flow, schemas, serializers, public surfaces, tests, generated/native boundaries, and ownership are represented where supported; incomplete dynamic edges remain frontiers; vector results are bounded, stale-safe, and semantic_authority=false.
- Gap task: Fill the concrete program graph/call resolver capability gap and add value/behavior nomination without replacing existing graph/index foundations.
- Refinement: A vector-nearest variable is not necessarily the information-bearing value a contract requires.
- Embedding query: program call data flow knowledge graph value vector index snapshot resolver dependency
- AST query: ProgramGraph ProgramCallResolver ProgramDependencyGraph ChangeValueVectorIndex

## RPR-G140 Compute a sound reverse impact closure and unknown frontier

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G130
- Priority: P0
- Track: impact-closure
- Bundle: agent-supervisor/proof-gated-contract-repair/change-impact
- Parallel lane: rpr-impact-closure
- Conflict policy: Consume the admitted program graph and existing CodeImpactIndex; keep nominated runtime, GraphRAG, vector, history, reflection, and FFI edges explicitly non-authoritative.
- Resource class: cpu-large
- Token class: large
- Goal: Starting from each semantic delta, compute the dependency-complete reverse transitive closure, group SCCs, and account for every resolved dependent or explicit unsupported/unknown frontier.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py, ipfs_accelerate_py/agent_supervisor/analysis/dynamic_impact_frontier.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py, ipfs_accelerate_py/agent_supervisor/analysis/dynamic_impact_frontier.py, test/api/test_agent_supervisor_contract_change_impact.py, test/api/test_agent_supervisor_dynamic_impact_frontier.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_contract_change_impact.py test/api/test_agent_supervisor_dynamic_impact_frontier.py
- Acceptance: Closure is deterministic and root-bound; includes direct and indirect calls, wrappers, overrides, schemas, wiring, generated surfaces, tests, and operational consumers; records SCCs, exclusions, bounds, and coverage; reflection, string dispatch, plugins, runtime registries, remote services, FFI, and excluded/generated sources remain explicit until admitted evidence closes them.
- Gap task: Implement fixed-point reverse impact accounting with no silent completeness promotion.
- Refinement: Zero discovered consumers is not proof that a changed symbol is unused.
- Embedding query: reverse transitive impact closure strongly connected component dynamic reflection FFI unknown frontier
- AST query: ContractChangeImpact ImpactClosureReceipt DynamicImpactFrontier

## RPR-G150 Detect semantic deltas and issue per-consumer obligations

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G120, RPR-G130
- Priority: P0
- Track: contract-delta
- Bundle: agent-supervisor/proof-gated-contract-repair/change-delta
- Parallel lane: rpr-contract-delta
- Conflict policy: Add before/after delta and consumer-ledger adapters over ProgramContract@1; do not introduce ProgramContract@2 or infer expectations from the changed implementation.
- Resource class: cpu-large
- Token class: large
- Goal: Normalize base/candidate semantic contract changes and create one compatibility or migration obligation for every impacted call site, schema/protocol consumer, constructor, serializer, test, mock, and public boundary.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/program_contract_delta.py, ipfs_accelerate_py/agent_supervisor/analysis/change_consumer_inventory.py, ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_contract_delta.py, ipfs_accelerate_py/agent_supervisor/analysis/change_consumer_inventory.py, ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py, test/api/test_agent_supervisor_program_contract_delta.py, test/api/test_agent_supervisor_change_consumer_inventory.py, test/api/test_agent_supervisor_schema_protocol_change_impact.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_contract_delta.py test/api/test_agent_supervisor_change_consumer_inventory.py test/api/test_agent_supervisor_schema_protocol_change_impact.py
- Acceptance: Formatting and moves do not manufacture deltas; parameter, result, type, schema, serialization, error, effect, capability, authorization, temporal, resource, memory, visibility, and wiring changes are explicit; a new required third argument identifies every two-argument caller separately; compatibility is consumer-domain-specific and no caller, schema, generated binding, fixture, or public surface is discharged by another consumer's success.
- Gap task: Implement typed semantic diff and exhaustive per-consumer compatibility ledgers.
- Refinement: Declaration compatibility and use-site compatibility are different claims.
- Embedding query: before after contract delta added required argument every caller schema protocol consumer
- AST query: ProgramContractDelta ChangeConsumerInventory ConsumerMigrationObligation SchemaProtocolImpact

## RPR-G160 Prove missing-value provenance and construction routes

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G140, RPR-G150
- Priority: P0
- Track: value-provenance
- Bundle: agent-supervisor/proof-gated-contract-repair/missing-input
- Parallel lane: rpr-value-provenance
- Conflict policy: Retrieval nominates only; reaching definitions, dominance, path conditions, information sufficiency, and contract proofs determine eligibility.
- Resource class: cpu-proof-solver
- Token class: xlarge
- Goal: For each missing input, nominate bounded in-scope, threaded, configured, injected, derived, or constructible values and prove the exact path on which one uniquely satisfies the consumer contract.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/missing_input_candidate_retrieval.py, ipfs_accelerate_py/agent_supervisor/analysis/value_provenance_graph.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/missing_input_candidate_retrieval.py, ipfs_accelerate_py/agent_supervisor/analysis/value_provenance_graph.py, test/api/test_agent_supervisor_missing_input_candidate_retrieval.py, test/api/test_agent_supervisor_value_provenance_graph.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_missing_input_candidate_retrieval.py test/api/test_agent_supervisor_value_provenance_graph.py
- Acceptance: Candidates record exact scope, definitions, aliases, dominance, path conditions, type/schema/range, information provenance, constructors, effects/capabilities/auth, ownership/lifetime, and dependency direction; vector/KG/history signals remain nomination only; same-typed wrong information, branch-local absence, forbidden config/env, partial constructors, cycles, ambiguity, and stale roots fail; unsatisfied inputs can be threaded upward without losing origin.
- Gap task: Implement value nomination and exact provenance graphs for analytical synthesis.
- Refinement: Type assignability is necessary but not sufficient for semantic information sufficiency.
- Embedding query: missing third argument reaching definition dominance path condition value provenance constructor dependency injection
- AST query: MissingInputCandidateRetriever ValueProvenanceGraph ValueProvenanceCandidate MissingInputRequirement

## RPR-G170 Synthesize and place complex support behavior

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G150, RPR-G160
- Priority: P0
- Track: behavior-synthesis
- Bundle: agent-supervisor/proof-gated-contract-repair/support-behavior
- Parallel lane: rpr-behavior-synthesis
- Conflict policy: Expected behavior comes from independent specifications, callers, schemas, and policy; implementation observations and LLM proposals cannot author their own acceptance contract.
- Resource class: cpu-proof-solver
- Token class: xlarge
- Goal: Define the fields, invariants, construction, methods, state machine, lifecycle, concurrency, persistence, errors, effects, compatibility, and tests required when propagation needs a new class, method, data structure, schema, provider, or factory, then prove an admissible owner and wiring path.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis/required_behavior_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/support_behavior_placement.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/required_behavior_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/support_behavior_placement.py, test/api/test_agent_supervisor_required_behavior_synthesis.py, test/api/test_agent_supervisor_support_behavior_placement.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_required_behavior_synthesis.py test/api/test_agent_supervisor_support_behavior_placement.py
- Acceptance: Behavior sources obey reviewed precedence and state assumptions/unknowns; constructors, invariants, transitions, ownership, lifetime, mutation, concurrency, serialization, migrations, errors, effects, auth, trust, resources, defaults, compatibility, tests, and telemetry are explicit; placement proves ownership, visibility, layering, registration/export wiring, write authority, and no forbidden cycle; insufficient or conflicting authority yields review-only abstention.
- Gap task: Implement independent behavior synthesis and exact placement admission for complex support types.
- Refinement: A plausible generated class is not a contract.
- Embedding query: synthesize new class method data structure behavior invariant state machine lifecycle placement
- AST query: RequiredBehaviorSynthesizer RequiredBehaviorContract SupportBehaviorPlacement

## RPR-G180 Prove mappings and prefer deterministic analytical transforms

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G160, RPR-G170
- Priority: P0
- Track: propagation-proof
- Bundle: agent-supervisor/proof-gated-contract-repair/change-proof
- Parallel lane: rpr-propagation-proof
- Conflict policy: Own propagation-specific obligation and synthesis adapters; use capability-admitted datasets logic, proof reconstruction, and existing assurance lattices without editing upstream solver foundations.
- Resource class: cpu-proof-solver
- Token class: xlarge
- Goal: Lower change, consumer, value-provenance, constructor, state, and placement claims into ipfs_datasets_py logic; prove, refute, and reconstruct them; and render only unique closed supported transformations analytically.
- Evidence: ipfs_accelerate_py/agent_supervisor/proof/change_propagation_obligations.py, ipfs_accelerate_py/agent_supervisor/proof/missing_input_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/analytical_change_transforms.py
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/change_propagation_obligations.py, ipfs_accelerate_py/agent_supervisor/proof/missing_input_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/analytical_change_transforms.py, test/api/test_agent_supervisor_change_propagation_obligations.py, test/api/test_agent_supervisor_missing_input_synthesis.py, test/api/test_agent_supervisor_analytical_change_transforms.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_obligations.py test/api/test_agent_supervisor_missing_input_synthesis.py test/api/test_agent_supervisor_analytical_change_transforms.py
- Acceptance: Obligations bind exact premises, assumptions, roots, translator, solver, reconstruction kernel, and policy; solver/premise candidates never grant authority; scope, type/schema, information, constructor totality, effects/capabilities/auth, ownership/lifetime, cycles, behavior, and placement are separate; unique proofs yield deterministic replayable argument/import/adapter/threading/schema transforms; unknown, timeout, counterexample, ambiguity, or unsupported semantics do not produce code.
- Gap task: Implement the analytical prove/refute/reconstruct path and a closed codemod set.
- Refinement: LLM escalation begins only after analytical synthesis has a typed non-success disposition.
- Embedding query: theorem prover missing input synthesis analytical codemod reconstruction ipfs datasets logic
- AST query: ChangePropagationObligationCompiler MissingInputSynthesizer AnalyticalChangeTransform

## RPR-G190 Admit one atomic transitive repair plan

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G140, RPR-G180
- Priority: P0
- Track: propagation-planning
- Bundle: agent-supervisor/proof-gated-contract-repair/change-plan
- Parallel lane: rpr-propagation-plan
- Conflict policy: Planning may consume admitted evidence only; exact paths, dependency order, SCC groups, checkpoints, and rollback are downstream of complete impact and proof gates.
- Resource class: cpu-large
- Token class: xlarge
- Goal: Build and deterministically admit one content-addressed plan that accounts for every affected consumer and groups all mutually dependent edits into atomic transactions, or abstain.
- Evidence: ipfs_accelerate_py/agent_supervisor/planning/change_propagation_plan.py
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/change_propagation_plan.py, test/api/test_agent_supervisor_change_propagation_plan.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_plan.py
- Acceptance: Plan identity covers the complete delta, closure, consumer ledger, proof set, candidate set, graph/index roots, exact reads/writes, topological steps, SCC groups, pre/postconditions, validation, checkpoint/rollback, invalidators, and fixed-point bound; every mandatory consumer has exactly one disposition; omissions, duplicate repairs, conflicting mappings, stale roots, incomplete frontiers, forbidden paths, cycles outside an atomic group, and alternative equally valid plans force abstention.
- Gap task: Implement exact multi-file plan construction, admission, replay, and abstention.
- Refinement: Atomic means no intermediate or partial group can be published as completion.
- Embedding query: atomic transitive change plan SCC dependency order exact paths checkpoint rollback
- AST query: ChangePropagationPlanner AtomicPropagationPlan PropagationPlanAdmission

## RPR-G200 Materialize bounded analytical or llm_router execution

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G190
- Priority: P0
- Track: propagation-handoff
- Bundle: agent-supervisor/proof-gated-contract-repair/change-handoff
- Parallel lane: rpr-propagation-handoff-serialized
- Conflict policy: Own new packet/router/task/gate modules only; preserve existing repair packet and provider behavior until the transaction-backed serialized cutover in RPR-G210.
- Resource class: cpu-large
- Token class: xlarge
- Goal: Materialize plan-bound analytical edits or exact LLM steps and route model work through the existing bounded llm_router proposal/review/lease path without giving the model semantic or scope authority.
- Evidence: ipfs_accelerate_py/agent_supervisor/proof/change_propagation_edit_packet.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/change_propagation_provider_router.py, ipfs_accelerate_py/agent_supervisor/objectives/change_propagation_task_source.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_pre_provider_gate.py
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/change_propagation_edit_packet.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/change_propagation_provider_router.py, ipfs_accelerate_py/agent_supervisor/objectives/change_propagation_task_source.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_pre_provider_gate.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_edit_packet.py test/api/test_agent_supervisor_change_propagation_provider_router.py test/api/test_agent_supervisor_change_propagation_task_source.py test/api/test_agent_supervisor_change_propagation_pre_provider_gate.py
- Acceptance: Analytical steps remain deterministic; LLM packets contain exact admitted behavior, values, counterexamples, paths, postconditions, validation, and unsupported limits; provider routing is bounded, redacted, receipt-bearing, proposal-only, and uses canonical llm_router; pre-provider drift or proof loss blocks; model output cannot select sources, invent behavior, add dependencies, expand scope, omit consumers, or weaken checks; legacy repair paths remain compatible.
- Gap task: Implement and test new-file handoff adapters; transaction-backed shared-file integration belongs to RPR-G210.
- Refinement: The model implements a plan step; it does not create the plan.
- Embedding query: llm router bounded edit packet exact behavior multi file task provider gate integration
- AST query: ChangePropagationEditPacket ChangePropagationProviderRouter ChangePropagationTaskSource ChangePropagationPreProviderGate

## RPR-G210 Execute transactionally and validate to a fixed point

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G200
- Priority: P0
- Track: propagation-validation
- Bundle: agent-supervisor/proof-gated-contract-repair/change-validation
- Parallel lane: rpr-propagation-validation
- Conflict policy: Implement new transaction/validation primitives first; then perform the only serialized shared-file cutover and require every new mutation/completion path to invoke them.
- Resource class: cpu-large
- Token class: xlarge
- Goal: Apply each plan in a checkpointed candidate worktree, roll back failed or partial SCC groups, repeatedly re-index, re-resolve, re-diff, re-prove, and validate to a fixed point, and integrate that mandatory path with the analysis/refinery/daemon flow.
- Evidence: ipfs_accelerate_py/agent_supervisor/planning/change_propagation_transaction.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_validation.py, ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_pipeline.py
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/change_propagation_transaction.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_validation.py, ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_pipeline.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_mismatch_refinery.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/contract_packet_provider_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_change_propagation_transaction.py, test/api/test_agent_supervisor_change_propagation_validation.py, test/api/test_agent_supervisor_change_propagation_integration.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_transaction.py test/api/test_agent_supervisor_change_propagation_validation.py test/api/test_agent_supervisor_change_propagation_integration.py
- Acceptance: Exact checkpoint, tree, plan, step, lease, and rollback receipts are content-addressed; partial groups cannot merge; candidate validation rebuilds indexes/graphs/tombstones, recomputes delta and closure, discharges each original obligation once, discovers second-order impacts, reconstructs proofs, and runs dependency-complete policy tools/tests; the integrated daemon cannot write or complete propagation work without the transaction and current fixed-point receipt; bound exhaustion, bypass, weakened tests/contracts, skipped tools, omitted consumers, unknown required frontier, or new unplanned delta prevents completion; legacy repair flow remains compatible.
- Gap task: Implement transactional and fixed-point primitives first and then wire them into the single serialized shared-file cutover.
- Refinement: A clean compile after a partial migration is not transitive completion evidence.
- Embedding query: transactional multi file code change fixed point reindex resolve rediff reprove rollback
- AST query: ChangePropagationTransaction ChangePropagationValidator PropagationCompletionReceipt

## RPR-G220 Benchmark, operate, and safely roll out propagation

- Status: active
- Parent: RPR-G110
- Depends on: RPR-G210
- Priority: P0
- Track: propagation-rollout
- Bundle: agent-supervisor/proof-gated-contract-repair/change-rollout
- Parallel lane: rpr-propagation-rollout
- Conflict policy: Own propagation fixtures, benchmark, metrics, validation CLI, guide, and final validator cutover; do not edit protected control-plane artifacts or default-enable broad automated mutation.
- Resource class: cpu-large
- Token class: large
- Goal: Exercise adversarial transitive changes, enforce old and new zero-tolerance safety floors, expose exact diagnostics and metrics, and roll out through shadow, assist, and narrowly scoped analytical automation.
- Evidence: test/fixtures/agent_supervisor/change_propagation/manifest.json, scripts/benchmark_change_propagation.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_rollout.py, scripts/validate_change_propagation.py, scripts/validate_proof_gated_contract_repair.py, docs/guides/PROOF_GATED_CHANGE_PROPAGATION_GUIDE.md, test/api/test_agent_supervisor_change_propagation_end_to_end.py
- Outputs: test/fixtures/agent_supervisor/change_propagation, scripts/benchmark_change_propagation.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_rollout.py, scripts/validate_change_propagation.py, scripts/validate_proof_gated_contract_repair.py, docs/guides/PROOF_GATED_CHANGE_PROPAGATION_GUIDE.md, test/api/test_agent_supervisor_change_propagation_benchmark.py, test/api/test_agent_supervisor_change_propagation_rollout.py, test/api/test_agent_supervisor_contract_repair_rollout.py, test/api/test_agent_supervisor_change_propagation_end_to_end.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_change_propagation_fixtures.py test/api/test_agent_supervisor_change_propagation_benchmark.py test/api/test_agent_supervisor_change_propagation_rollout.py test/api/test_agent_supervisor_change_propagation_end_to_end.py
- Acceptance: Corpus covers argument threading, wrong same-typed value, path-sensitive sources, schemas, constructors, stateful new behavior, cycles, generated/runtime/native frontiers, stale/poison evidence, partial failure, LLM scope escape, and second-order impacts; missed-consumer, unproved-value, invented-behavior, partial-plan, stale-impact-plan, and false-fixed-point completion rates are zero alongside legacy floors; shadow is default and auto is initially unique reconstructed analytical Python transformations with complete coverage; operations validator recognizes the extension terminal IDs and policies and a healthy four-shard supervisor can drain the board.
- Gap task: Seed fixtures early, then implement benchmark, rollout, and final operations validation after fixed-point execution exists.
- Refinement: Automation expands only with measured coverage and safety evidence for the exact change family.
- Embedding query: change propagation adversarial benchmark missed caller wrong value fixed point shadow assist rollback
- AST query: ChangePropagationBenchmark ChangePropagationMetrics ChangePropagationRolloutPolicy
