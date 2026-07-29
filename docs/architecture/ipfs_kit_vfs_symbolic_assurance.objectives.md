# IPFS Kit VFS Symbolic Assurance Objective Heap

This is the durable source of intent for the SwissKnife indexing, cross-package
contract assurance, and IPFS Kit VFS drift program. The companion `.todo.md`
file is an executable projection. Task status, discovery prose, cache presence,
or a ZK receipt by itself never completes a goal. Completion requires fresh,
typed evidence bound to the exact repository forest, objective revision,
policy, analyzers, assumptions, proof/runtime toolchains, and acceptance
criterion.

## VFS-G000 Prove and repair IPFS Kit VFS contracts from a content-addressed SwissKnife program graph

- Status: active
- Parent:
- Fib priority: 1
- Track: vfs-symbolic-assurance
- Priority: P0
- Bundle: vfs-assurance/root
- Goal: Build a bounded, symbolic-first supervisor loop that indexes SwissKnife and the three IPFS packages, proves or disproves declared contracts, and emits small validated repair tasks for VFS drift and security defects.
- Evidence: vfs/repository-forest-receipt@1, vfs/exhaustive-index-receipt@1, vfs/contract-assurance-root@1, vfs/autonomous-refill-exhaustion@1
- Outputs: docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md, data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py -q
- Acceptance: Every direct child has fresh criterion-level evidence; no unsupported or stale result is promoted to proof; the complete frozen repository inventory is accounted for; accepted repairs retain normal authorization, validation, and merge gates; drained work triggers bounded evidence-driven refill or a conclusive exhaustion receipt.
- Gap task: Implement the highest-priority uncovered child criterion without expanding repository write authority.
- Refinement: Preserve separate source identity, indexing, graph, contract, MCP++, proof, ZK, VFS, security, repair, refill, and rollout workstreams.
- Embedding query: ipfs kit virtual filesystem swissknife symbolic contract proof content addressed CID multihash MCP++ zero knowledge autonomous repair
- AST query: SupervisorControlService AnalysisPipeline VFSCore IPFSFSSpecFileSystem MCPPPServerConnector

## VFS-G010 Bind every observation to an explicit multi-repository authority forest

- Status: active
- Parent: VFS-G000
- Fib priority: 2
- Track: repository-identity
- Priority: P0
- Bundle: vfs-assurance/foundation
- Goal: Represent SwissKnife, ipfs_accelerate_py, ipfs_kit_py, ipfs_datasets_py, and recursive gitlinks as independently versioned repository descriptors with read/write authority and dirty-state policy.
- Evidence: vfs/repository-descriptor@1, vfs/repository-forest-manifest@1
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_forest.py, test/api/test_agent_supervisor_repository_forest.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_forest.py -q
- Acceptance: Paths cannot escape a descriptor root; sibling repositories are never conflated; commit, tree, gitlinks, dirty overlay, ignore policy, and authority affect identity; external SwissKnife is read-only in the initial policy.
- Gap task: Add the smallest missing repository identity or authority invariant.
- Refinement: Separate descriptor identity from inventory enumeration and mutation authorization.
- Embedding query: repository forest git tree dirty overlay submodule gitlink allowlist swissknife
- AST query: ResultBinding RepositoryIdentity checkout_mutation_lock_path

## VFS-G011 Freeze and replay the initial four-repository manifest

- Status: active
- Parent: VFS-G010
- Fib priority: 3
- Track: repository-identity
- Priority: P0
- Bundle: vfs-assurance/foundation
- Goal: Produce a canonical manifest and replay validator for the four configured checkouts without embedding host-specific paths in portable content identity.
- Evidence: vfs/repository-forest-replay@1
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_forest.py, test/api/test_agent_supervisor_repository_forest.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_forest.py -q
- Acceptance: Identical trees and policy reproduce the same portable forest CID; a changed commit, tree, gitlink, overlay, or policy changes it; unavailable roots fail closed with a typed reason.
- Gap task: Repair manifest determinism or replay diagnostics.
- Refinement: Keep portable identity distinct from local locator and credential state.
- Embedding query: canonical repository manifest replay CID portable locator
- AST query: canonical_runtime_json_bytes cid_for_dag_json validate_cid

## VFS-G020 Exhaustively inventory and incrementally parse the SwissKnife corpus

- Status: active
- Parent: VFS-G000
- Fib priority: 3
- Track: corpus-index
- Priority: P0
- Bundle: vfs-assurance/index
- Goal: Account for every admitted SwissKnife file and emit reusable content-bound AST, symbol, schema, documentation, test, and manifest records.
- Evidence: vfs/exhaustive-file-inventory@1, vfs/incremental-ast-index@1
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_corpus_index.py, ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, test/api/test_agent_supervisor_repository_corpus_index.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_corpus_index.py test/api/test_agent_supervisor_program_ast_adapters.py -q
- Acceptance: The scan publishes included and excluded populations with reasons; TypeScript/TSX/JavaScript/Python/JSON/Markdown inputs have provenance; unchanged blobs are reused; unexplained skips, parser failures, and truncation prevent an exhaustive verdict.
- Gap task: Add one missing inventory or language-evidence producer using the canonical AST record schema.
- Refinement: Split inventory, language adapters, and incremental persistence by conflict domain.
- Embedding query: swissknife exhaustive inventory TypeScript TSX JavaScript Python JSON Markdown AST incremental
- AST query: ASTBlobRecord index_ast_blob_records collect_ast_dataset_records

## VFS-G021 Resolve dynamic language features without inventing call edges

- Status: active
- Parent: VFS-G020
- Fib priority: 5
- Track: corpus-index
- Priority: P0
- Bundle: vfs-assurance/index
- Goal: Normalize imports, re-exports, decorators, callbacks, dependency injection, dynamic imports, subprocesses, RPC, HTTP, and libp2p boundaries as resolved, candidate, ambiguous, or external edges.
- Evidence: vfs/language-edge-resolution@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, ipfs_accelerate_py/agent_supervisor/program_graph.py, test/api/test_agent_supervisor_program_ast_adapters.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_ast_adapters.py test/api/test_agent_supervisor_program_graph.py -q
- Acceptance: Every edge cites a source span and resolver rule; ambiguous and unsupported constructs remain explicit; adversarial name collisions and re-exports cannot become forged direct calls.
- Gap task: Add one sound resolver rule or explicit unsupported classification.
- Refinement: Prefer conservative unknown edges over heuristic certainty.
- Embedding query: static call resolution dynamic import callback dependency injection RPC HTTP libp2p
- AST query: AnalysisASTIndex SemanticDependencyGraph CallEdge

## VFS-G030 Use canonical multiformats identities and dependency-aware content caches

- Status: active
- Parent: VFS-G000
- Fib priority: 5
- Track: content-addressing
- Priority: P0
- Bundle: vfs-assurance/cas
- Goal: Bridge supervisor content identities to strict DAG-JSON CIDv1 and multihash records and reuse only exact, fresh, dependency-complete analysis results.
- Evidence: vfs/cid-profile@1, vfs/dependency-cache@1
- Outputs: ipfs_accelerate_py/agent_supervisor/multiformats_identity.py, ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, test/api/test_agent_supervisor_multiformats_identity.py
- Validation: python -m pytest test/api/test_agent_supervisor_multiformats_identity.py test/api/test_agent_supervisor_program_analysis_cache.py -q
- Acceptance: CIDv1/base32/dag-json/sha2-256 bytes are cross-package reproducible; existing supervisor IDs retain compatibility mappings; all semantic dependencies and policy versions participate in cache keys; corruption and stale/negative results fail closed.
- Gap task: Repair one canonicalization, compatibility, invalidation, or retention invariant.
- Refinement: Keep immutable object identity separate from mutable current-tree projections.
- Embedding query: multiformats multihash CIDv1 DAG JSON content cache invalidation
- AST query: cid_for_dag_json validate_cid RuntimeCAS AnalysisCacheKey

## VFS-G031 Prove transitive cache invalidation and bounded storage

- Status: active
- Parent: VFS-G030
- Fib priority: 8
- Track: content-addressing
- Priority: P0
- Bundle: vfs-assurance/cas
- Goal: Invalidate only affected dependents across blob, AST, graph, contract, proof, runtime, and ZK artifacts while enforcing single-flight and storage quotas.
- Evidence: vfs/cache-invalidation-proof@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, test/api/test_agent_supervisor_program_analysis_cache.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_analysis_cache.py test/api/test_agent_supervisor_runtime_cas.py -q
- Acceptance: Every changed identity dimension has a test; unrelated components remain reusable; concurrent exact misses collapse; failed flights clean up; retained artifacts and compact receipts stay within declared count and byte bounds.
- Gap task: Add the smallest missing dependency or invalidation test and implementation.
- Refinement: Do not collapse draft, diagnostic, proposal, and authoritative namespaces.
- Embedding query: transitive cache invalidation single flight quota stale negative
- AST query: RuntimeCAS CacheCoordinator invalidate

## VFS-G040 Project a provenance-preserving program and GraphRAG evidence graph

- Status: active
- Parent: VFS-G000
- Fib priority: 8
- Track: program-graph
- Priority: P0
- Bundle: vfs-assurance/graph
- Goal: Build an IPLD/GraphRAG projection over repositories, blobs, symbols, calls, contracts, tests, MCP tools, proof obligations, and findings without giving retrieval authority to synthesize evidence.
- Evidence: vfs/program-graph@1, vfs/graphrag-projection@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_graph.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_program_graph_provider.py, test/api/test_agent_supervisor_program_graph.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_graph.py test/api/test_agent_supervisor_ipfs_datasets_program_graph_provider.py -q
- Acceptance: Node and edge provenance is content bound; graph chunks are deterministic and bounded; retrieval returns compact references and ranking reasons; provider absence degrades explicitly; GraphRAG output cannot create completion or proof authority.
- Gap task: Add one missing node/edge type, bounded query, or provider conformance case.
- Refinement: Separate canonical graph construction from optional GraphRAG ranking.
- Embedding query: GraphRAG IPLD knowledge graph program graph provenance bounded retrieval
- AST query: IpfsDatasetsAnalysisProvider AnalysisRetrieval CodeEvidenceGraph

## VFS-G041 Answer minimal call-slice and contract-impact queries

- Status: active
- Parent: VFS-G040
- Fib priority: 13
- Track: program-graph
- Priority: P1
- Bundle: vfs-assurance/graph
- Goal: Return the smallest dependency-complete slice for a symbol, MCP tool, contract, VFS operation, or changed blob with explicit truncation and unknown frontiers.
- Evidence: vfs/minimal-call-slice@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_graph_queries.py, test/api/test_agent_supervisor_program_graph_queries.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_graph_queries.py -q
- Acceptance: Seeded transitive callers/callees and MCP paths are complete within scope; unrelated source is omitted; limits never silently convert an incomplete slice into a complete result.
- Gap task: Implement one bounded query or frontier diagnostic.
- Refinement: Optimize query indexes without changing canonical graph identity.
- Embedding query: minimal call slice impact query dependency complete bounded context
- AST query: ProofDirectedRetriever retrieve_analysis_evidence

## VFS-G050 Extract versioned expected and observed contracts without circular reasoning

- Status: active
- Parent: VFS-G000
- Fib priority: 13
- Track: contract-ir
- Priority: P0
- Bundle: vfs-assurance/contracts
- Goal: Compile reviewed IDL/schema/types/tests/specification expectations and separately compile observed signatures, calls, effects, errors, and fallback behavior.
- Evidence: vfs/contract-ir@1, vfs/contract-source-precedence@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_contracts.py, ipfs_accelerate_py/agent_supervisor/contract_extractor.py, test/api/test_agent_supervisor_contract_extractor.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_contracts.py test/api/test_agent_supervisor_contract_extractor.py -q
- Acceptance: Contract sources and conflicts are explicit; implementation observations cannot define their own expectation; inputs, outputs, errors, effects, authorization, idempotence, ordering, atomicity, resources, and degradation are represented or marked unsupported.
- Gap task: Add one missing contract source, semantic field, or conflict case.
- Refinement: Keep contract extraction independent from satisfaction checking.
- Embedding query: contract IR schema IDL types tests expected observed precondition postcondition effect
- AST query: InterfaceContract FormalConstraint

## VFS-G051 Generate conclusive mismatch witnesses or explicit unknown results

- Status: active
- Parent: VFS-G050
- Fib priority: 21
- Track: contract-ir
- Priority: P0
- Bundle: vfs-assurance/contracts
- Goal: Compare expected and observed contract IR using typed lattice, dataflow, effect, and counterexample rules.
- Evidence: vfs/contract-check-result@1, vfs/contract-counterexample@1
- Outputs: ipfs_accelerate_py/agent_supervisor/contract_checker.py, test/api/test_agent_supervisor_contract_checker.py
- Validation: python -m pytest test/api/test_agent_supervisor_contract_checker.py -q
- Acceptance: Proven matches, proven mismatches, runtime witnesses, ambiguity, unsupported semantics, timeout, and stale results are distinct; seeded violations are found; compatible refinements are not false positives.
- Gap task: Add one sound contract rule or adversarial counterexample fixture.
- Refinement: Require exact repository, symbol, interface, policy, and freshness binding.
- Embedding query: contract mismatch counterexample subtyping effect dataflow unknown
- AST query: CodeProofObligation Counterexample

## VFS-G060 Prove whether SwissKnife MCP++ calls reach the intended IPFS implementation

- Status: active
- Parent: VFS-G000
- Fib priority: 21
- Track: mcp-interop
- Priority: P0
- Bundle: vfs-assurance/mcplusplus
- Goal: Resolve UI/service callers through connector, negotiated transport/profile, tools/list, tools/call, server registration, adapter, package implementation, and result/error mapping.
- Evidence: vfs/mcplusplus-call-path@1, vfs/mcplusplus-manifest-parity@1
- Outputs: ipfs_accelerate_py/agent_supervisor/mcplusplus_contract_resolver.py, test/api/test_agent_supervisor_mcplusplus_contract_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_mcplusplus_contract_resolver.py -q
- Acceptance: Same-name helpers, mocks, static payloads, copied manifests, and fallbacks do not prove invocation; ambiguous/dynamic registrations remain explicit; Python/TypeScript/schema names and errors are checked for parity.
- Gap task: Add one missing registration/transport resolver or interop fixture.
- Refinement: Split static resolution from hermetic runtime conformance.
- Embedding query: SwissKnife MCP++ connector tools list tools call server registration package implementation
- AST query: MCPPPServerConnector mcp_server tools_call

## VFS-G061 Witness selected MCP++ paths in a hermetic runtime

- Status: active
- Parent: VFS-G060
- Fib priority: 34
- Track: mcp-interop
- Priority: P1
- Bundle: vfs-assurance/mcplusplus
- Goal: Run bounded cross-language contract fixtures against real registered adapters and record request, result, error, capability, and transport identities.
- Evidence: vfs/mcplusplus-runtime-witness@1
- Outputs: test/api/test_agent_supervisor_mcplusplus_runtime_contracts.py, ipfs_accelerate_py/agent_supervisor/mcplusplus_runtime_witness.py
- Validation: python -m pytest test/api/test_agent_supervisor_mcplusplus_runtime_contracts.py -q
- Acceptance: Real adapter dispatch is distinguished from mocks; HTTP and mcp+p2p profiles use the same admitted contract where declared; failures and unavailable services are typed, bounded, and non-authoritative.
- Gap task: Add one real-tool runtime witness or failure-mode assertion.
- Refinement: Network remains disabled unless an exact fixture and egress policy permit it.
- Embedding query: MCP++ runtime conformance HTTP libp2p real adapter witness
- AST query: tools_list tools_call MCPp2pSession

## VFS-G070 Translate supported code contracts into kernel-checkable proof obligations

- Status: active
- Parent: VFS-G000
- Fib priority: 34
- Track: formal-proof
- Priority: P0
- Bundle: vfs-assurance/proof
- Goal: Translate finite call slices and contract predicates into the established formal vocabulary and route them through capability-probed proof providers and authoritative validation.
- Evidence: vfs/logic-translation@1, vfs/kernel-proof-receipt@1
- Outputs: ipfs_accelerate_py/agent_supervisor/code_contract_logic.py, ipfs_accelerate_py/agent_supervisor/code_contract_prover.py, test/api/test_agent_supervisor_code_contract_prover.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_contract_logic.py test/api/test_agent_supervisor_code_contract_prover.py -q
- Acceptance: Translation round trips are checked; assumptions and unsupported semantics are explicit; premise selectors and candidate solvers lack authority; wrong theorem, stale proof, omitted effect, and capability-loss cases fail closed.
- Gap task: Add one supported predicate or authoritative proof-validation case.
- Refinement: Keep translation, candidate search, and kernel validation separate.
- Embedding query: formal logic code contract theorem proof kernel premise selection
- AST query: FormalLogicVocabulary MultiProverRouter KernelVerification

## VFS-G071 Produce minimal proof and counterexample contexts

- Status: active
- Parent: VFS-G070
- Fib priority: 55
- Track: formal-proof
- Priority: P1
- Bundle: vfs-assurance/proof
- Goal: Select a dependency-complete proof context from the program graph and emit on-demand expansion handles rather than full source prompts.
- Evidence: vfs/minimal-proof-context@1
- Outputs: ipfs_accelerate_py/agent_supervisor/code_contract_prover.py, test/api/test_agent_supervisor_code_contract_proof_context.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_contract_proof_context.py -q
- Acceptance: Required axioms, contracts, effects, and call edges are never truncated; optional premises have inclusion reasons; identical requests reuse exact receipts; changed dependencies invalidate the proof context.
- Gap task: Repair one completeness, minimality, or invalidation invariant.
- Refinement: Measure proof context bytes independently from model context.
- Embedding query: minimal proof context premise dependency expansion handle
- AST query: ProofContext ProofDirectedRetriever

## VFS-G080 Attest supported deterministic analysis traces with qualified zero-knowledge proofs

- Status: active
- Parent: VFS-G000
- Fib priority: 55
- Track: zero-knowledge
- Priority: P1
- Bundle: vfs-assurance/zkp
- Goal: Define public commitments and circuits for supported repository, contract, call-slice, assumptions, trace, and result identities while keeping witness policy explicit.
- Evidence: vfs/zk-trace-statement@1, vfs/zk-verification-receipt@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, ipfs_datasets_py/ipfs_datasets_py/logic/zkp/provekit/circuits/program_contract_trace, test/api/test_agent_supervisor_program_analysis_zkp.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_analysis_zkp.py -q
- Acceptance: The statement proves only commitment openings and supported trace transitions; circuit, proving key, verifying key, public-input codec, backend, and ceremony identities are bound; verifier replay rejects tampering and stale inputs.
- Gap task: Add one circuit constraint, codec invariant, or verifier replay case.
- Refinement: Start in shadow and keep ZK attestation independent from semantic proof authority.
- Embedding query: zero knowledge proof program analysis trace commitment ProveKit Groth16
- AST query: ZKPBackend ProofStatement ProveKitBackend

## VFS-G081 Prevent simulated or placeholder ZK paths from acquiring authority

- Status: active
- Parent: VFS-G080
- Fib priority: 89
- Track: zero-knowledge
- Priority: P0
- Bundle: vfs-assurance/zkp
- Goal: Probe backend, circuit, setup, field encoding, ceremony, and verifier conformance and publish explicit shadow/degraded status.
- Evidence: vfs/zk-capability-conformance@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, test/api/test_agent_supervisor_program_analysis_zkp_conformance.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_analysis_zkp_conformance.py -q
- Acceptance: Simulated backends, placeholder encodings, unversioned circuits, missing setup, incompatible codecs, or absent independent verification cannot emit authoritative receipts; capability loss invalidates prior projections.
- Gap task: Add one fail-closed capability or adversarial forgery test.
- Refinement: Do not silently substitute one proof system for another.
- Embedding query: simulated ZKP placeholder production capability ceremony verifying key fail closed
- AST query: SimulatedBackend ZKPBackendProtocol probe_formal_verification_capabilities

## VFS-G090 Establish and check the canonical IPFS Kit VFS behavioral contract

- Status: active
- Parent: VFS-G000
- Fib priority: 89
- Track: vfs-drift
- Priority: P0
- Bundle: vfs-assurance/vfs
- Goal: Inventory competing VFS, fsspec, bucket, journal, version, handler, endpoint, tool, SDK, and variant surfaces and map each to one reviewed canonical operation contract.
- Evidence: vfs/canonical-operation-matrix@1, vfs/drift-inventory@1
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_contract_pack.py, test/api/test_agent_supervisor_vfs_contract_pack.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_contract_pack.py -q
- Acceptance: Path, IO, errors, sync/async, CID, atomicity, journal, cache, backend, authorization, and degradation semantics are covered; duplicate/variant modules and manifest drift are evidence-backed; presence alone is not labeled a defect.
- Gap task: Add one missing VFS surface or invariant to the canonical matrix.
- Refinement: Separate inventory findings from repair decisions.
- Embedding query: IPFS Kit VFS fsspec bucket journal version MCP handler endpoint drift
- AST query: VFSCore IPFSFSSpecFileSystem VFSManager BucketVFSManager

## VFS-G091 Differentially witness VFS facade and MCP behavior

- Status: active
- Parent: VFS-G090
- Fib priority: 144
- Track: vfs-drift
- Priority: P0
- Bundle: vfs-assurance/vfs
- Goal: Generate finite contract fixtures and compare canonical Python, CLI, MCP, MCP++, HTTP, libp2p, and backend results under the same operation model.
- Evidence: vfs/differential-contract-witness@1
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_differential_harness.py, test/api/test_agent_supervisor_vfs_differential_harness.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_differential_harness.py -q
- Acceptance: Seeded drift is detected; stable compatible behavior agrees across surfaces; mocks and unavailable backends are explicit; destructive and network operations use hermetic fixtures.
- Gap task: Add one operation family, surface, or adversarial drift fixture.
- Refinement: Keep runtime witnesses bound to exact fixture and toolchain CIDs.
- Embedding query: VFS differential contract Python CLI MCP MCP++ backend
- AST query: vfs_read vfs_write mcp_vfs_action

## VFS-G100 Generate a typed, deduplicated correctness and vulnerability ledger

- Status: active
- Parent: VFS-G000
- Fib priority: 144
- Track: finding-generation
- Priority: P0
- Bundle: vfs-assurance/findings
- Goal: Convert conclusive contract, dataflow, proof, runtime, and security-property evidence into stable findings and SARIF/JSON/Markdown projections.
- Evidence: vfs/finding-ledger@1, vfs/vulnerability-evidence-policy@1
- Outputs: ipfs_accelerate_py/agent_supervisor/contract_findings.py, ipfs_accelerate_py/agent_supervisor/finding_task_source.py, test/api/test_agent_supervisor_contract_findings.py
- Validation: python -m pytest test/api/test_agent_supervisor_contract_findings.py test/api/test_agent_supervisor_finding_task_source.py -q
- Acceptance: Severity, confidence, claim level, freshness, affected symbols, expected/observed contracts, witnesses, supersession, and remediation scope are bound; vulnerability labels require threat path and impact; duplicates and stale findings do not create work.
- Gap task: Add one finding policy, deduplication, SARIF, or task-source invariant.
- Refinement: Preserve append-only finding history and mutable current-tree projections separately.
- Embedding query: bug vulnerability finding SARIF contract evidence severity deduplication
- AST query: TaskProposal SecurityConstraint Counterexample

## VFS-G101 Materialize a second repair taskboard from admitted findings

- Status: active
- Parent: VFS-G100
- Fib priority: 233
- Track: finding-generation
- Priority: P0
- Bundle: vfs-assurance/findings
- Goal: Generate exact, dependency-aware repair tasks in the accelerator package from fresh finding CIDs without letting a report authorize edits.
- Evidence: vfs/finding-taskboard@1
- Outputs: ipfs_accelerate_py/agent_supervisor/finding_task_source.py, test/api/test_agent_supervisor_finding_task_source.py
- Validation: python -m pytest test/api/test_agent_supervisor_finding_task_source.py -q
- Acceptance: Each task has one root-cause family, exact outputs, allowed effects, validation/proof plan, context budget, goal lineage, dependencies, and stable identity; ambiguous or broad findings require review instead of executable authority.
- Gap task: Repair task sizing, identity, lineage, or authority projection.
- Refinement: Split tasks only when validation and merge fate can remain independent.
- Embedding query: finding taskboard repair exact output goal lineage stable identity
- AST query: TaskSource TaskProposalRouter canonical_task_identity

## VFS-G110 Give Grok and Codex only compact CID-addressed repair packets

- Status: active
- Parent: VFS-G000
- Fib priority: 233
- Track: low-context-repair
- Priority: P0
- Bundle: vfs-assurance/repair
- Goal: Compile the smallest sufficient repair packet and delta retry from finding, contract, call-slice, proof, and validation references.
- Evidence: vfs/compact-repair-packet@1, vfs/delta-repair-context@1
- Outputs: ipfs_accelerate_py/agent_supervisor/contract_repair_packet.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_contract_repair_packet.py
- Validation: python -m pytest test/api/test_agent_supervisor_contract_repair_packet.py -q
- Acceptance: Default canonical JSON is at most 16 KiB plus bounded source spans; required authority and acceptance fields survive provider budgets; expansion is handle-based; retries contain changed evidence; model output remains a proposal.
- Gap task: Remove unnecessary context or add one missing required binding.
- Refinement: Measure symbolic analysis and provider context separately.
- Embedding query: small targeted edit LLM context repair packet CID delta retry Grok Codex
- AST query: ContextCompiler CodexRepairPacket FormalReplanner

## VFS-G111 Run conflict-safe Grok Build and Codex implementation shards

- Status: active
- Parent: VFS-G110
- Fib priority: 377
- Track: low-context-repair
- Priority: P1
- Bundle: vfs-assurance/repair
- Goal: Use provider-probed deterministic shards, isolated worktrees, leases, protected paths, validation, serialized merge, and retry budgets.
- Evidence: vfs/provider-shard-receipt@1
- Outputs: scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance_control.sh, test/api/test_vfs_symbolic_assurance_control.py
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_control.py -q
- Acceptance: Both configured providers perform eligible work when healthy; one unavailable provider does not duplicate or block the other; objective/taskboard authority is protected; conflicts and exhausted retries become bounded follow-up work.
- Gap task: Repair provider admission, shard, worktree, lease, or merge behavior.
- Refinement: Provider preference never changes validation or completion authority.
- Embedding query: Grok Build Codex supervisor shard worktree merge lease retry
- AST query: PortalImplementationSupervisor task_shard_index

## VFS-G120 Refill goals, subgoals, and tasks from fresh symbolic evidence

- Status: active
- Parent: VFS-G000
- Fib priority: 377
- Track: autonomous-refill
- Priority: P0
- Bundle: vfs-assurance/refill
- Goal: Admit new failure families under bounded refinement while deduplicating repeated observations and preserving goal ancestry.
- Evidence: vfs/symbolic-refill-epoch@1, vfs/refill-idempotency@1
- Outputs: ipfs_accelerate_py/agent_supervisor/symbolic_finding_refill.py, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, test/api/test_agent_supervisor_symbolic_finding_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_symbolic_finding_refill.py -q
- Acceptance: Only fresh admitted findings produce work; existing goal families are reused; new children are bounded by breadth/depth/open-work/cooldown; replay is a no-op; unchanged failures back off; conclusive exhaustion creates no busywork.
- Gap task: Add one refill admission, lineage, idempotency, backoff, or exhaustion invariant.
- Refinement: Freeze root intent and assumptions during autonomous child refinement.
- Embedding query: autonomous refill goal subgoal task symbolic finding dedupe cooldown exhaustion
- AST query: AdaptiveGoalRefiner BacklogRefinery

## VFS-G121 Prove low-context symbolic-first operation and bounded resource use

- Status: active
- Parent: VFS-G120
- Fib priority: 610
- Track: autonomous-refill
- Priority: P1
- Bundle: vfs-assurance/refill
- Goal: Benchmark cold/warm scans, incremental changes, proof routing, cache reuse, task yield, false findings, provider tokens, storage, and idle behavior.
- Evidence: vfs/symbolic-efficiency-benchmark@1
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_symbolic_benchmark.py, test/api/test_agent_supervisor_vfs_symbolic_benchmark.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_symbolic_benchmark.py -q
- Acceptance: Scan, parse, identity, graph, contract comparison, cache, and supported proof stages use zero LLM calls; paired repair packets reduce median provider input by at least 80 percent versus repository-context baseline without lowering seeded finding coverage; resource ceilings and cache hit claims are measured.
- Gap task: Add one missing metric, paired fixture, or resource guard.
- Refinement: Never optimize token count by dropping required evidence or hiding unknowns.
- Embedding query: zero LLM symbolic analysis token benchmark cache reuse finding coverage
- AST query: SupervisorEfficiencyReport TokenLedger

## VFS-G130 Release only evidence-backed results through shadow and assist gates

- Status: active
- Parent: VFS-G000
- Fib priority: 610
- Track: assurance-rollout
- Priority: P0
- Bundle: vfs-assurance/rollout
- Goal: Validate completeness, precision, cache freshness, proof soundness, ZK conformance, repair safety, control-surface parity, recovery, and rollback before any automatic scope expands.
- Evidence: vfs/adversarial-e2e-gate@1, vfs/shadow-rollout-report@1
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_symbolic_rollout.py, test/api/test_vfs_symbolic_assurance_e2e.py
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py test/api/test_agent_supervisor_program_analysis_zkp_conformance.py -q
- Acceptance: Reproducible CIDs, complete inventories, zero stale authoritative hits, zero forged proof/ZK authority, seeded mismatch precision, deterministic tasks, Python/CLI/MCP parity, restart replay, and rollback are demonstrated on a frozen corpus.
- Gap task: Add one missing adversarial, recovery, parity, or rollback gate.
- Refinement: Promotion is per capability and returns to shadow on any binding or assurance regression.
- Embedding query: shadow rollout adversarial proof soundness cache freshness recovery rollback parity
- AST query: SelfImprovementRollout V2RolloutMode

## VFS-G131 Run the frozen SwissKnife and IPFS Kit VFS pilot

- Status: active
- Parent: VFS-G130
- Fib priority: 987
- Track: assurance-rollout
- Priority: P0
- Bundle: vfs-assurance/rollout
- Goal: Execute the admitted pipeline over the full SwissKnife inventory and the VFS-relevant closure of the three IPFS packages, then publish the baseline finding ledger and repair taskboard.
- Evidence: vfs/swissknife-vfs-pilot@1
- Outputs: data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/pilot, docs/architecture/ipfs_kit_vfs_symbolic_assurance.findings.todo.md
- Validation: python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot --verify
- Acceptance: Every admitted file is accounted for; every finding is reproducible from content-addressed evidence; inconclusive results remain non-actionable; the repair board is bounded, deduplicated, goal-backed, and independently reviewable.
- Gap task: Repair the earliest pipeline stage preventing a complete, reproducible pilot.
- Refinement: Do not broaden scope or write authority to make the pilot pass.
- Embedding query: SwissKnife full scan IPFS Kit VFS pilot finding ledger repair board
- AST query: vfs_symbolic_pilot

## VFS-G132 Prove vfs/repository-forest-receipt@1 for Prove and repair IPFS Kit VFS contracts from a content-addressed SwissKnife program graph

- Status: active
- Parent: VFS-G000
- Fib priority: 3000
- Track: vfs-symbolic-assurance
- Priority: P0
- Bundle: vfs-assurance/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/repository-forest-receipt@1`.
- Evidence: vfs/repository-forest-receipt@1
- Outputs: docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md, data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py -q
- Acceptance: Every direct child has fresh criterion-level evidence ; no unsupported or stale result is promoted to proof ; the complete frozen repository inventory is accounted for ; accepted repairs retain normal authorization ; validation ; and merge gates ; drained work triggers bounded evidence-driven refill or a conclusive exhaustion receipt.
- Refinement depth: 1
- Embedding query: vfs/repository-forest-receipt@1
- AST query: vfs/repository-forest-receipt@1
- Parallel lane: vfs-assurance/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/repository-forest-receipt@1` with a narrow, verifiable change.

## VFS-G133 Prove vfs/exhaustive-index-receipt@1 for Prove and repair IPFS Kit VFS contracts from a content-addressed SwissKnife program graph

- Status: active
- Parent: VFS-G000
- Fib priority: 3001
- Track: vfs-symbolic-assurance
- Priority: P0
- Bundle: vfs-assurance/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/exhaustive-index-receipt@1`.
- Evidence: vfs/exhaustive-index-receipt@1
- Outputs: docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md, data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py -q
- Acceptance: Every direct child has fresh criterion-level evidence ; no unsupported or stale result is promoted to proof ; the complete frozen repository inventory is accounted for ; accepted repairs retain normal authorization ; validation ; and merge gates ; drained work triggers bounded evidence-driven refill or a conclusive exhaustion receipt.
- Refinement depth: 1
- Embedding query: vfs/exhaustive-index-receipt@1
- AST query: vfs/exhaustive-index-receipt@1
- Parallel lane: vfs-assurance/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/exhaustive-index-receipt@1` with a narrow, verifiable change.

## VFS-G134 Prove vfs/contract-assurance-root@1 for Prove and repair IPFS Kit VFS contracts from a content-addressed SwissKnife program graph

- Status: active
- Parent: VFS-G000
- Fib priority: 3002
- Track: vfs-symbolic-assurance
- Priority: P0
- Bundle: vfs-assurance/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/contract-assurance-root@1`.
- Evidence: vfs/contract-assurance-root@1
- Outputs: docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md, data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py -q
- Acceptance: Every direct child has fresh criterion-level evidence ; no unsupported or stale result is promoted to proof ; the complete frozen repository inventory is accounted for ; accepted repairs retain normal authorization ; validation ; and merge gates ; drained work triggers bounded evidence-driven refill or a conclusive exhaustion receipt.
- Refinement depth: 1
- Embedding query: vfs/contract-assurance-root@1
- AST query: vfs/contract-assurance-root@1
- Parallel lane: vfs-assurance/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/contract-assurance-root@1` with a narrow, verifiable change.

## VFS-G135 Prove vfs/autonomous-refill-exhaustion@1 for Prove and repair IPFS Kit VFS contracts from a content-addressed SwissKnife program graph

- Status: active
- Parent: VFS-G000
- Fib priority: 3000
- Track: vfs-symbolic-assurance
- Priority: P0
- Bundle: vfs-assurance/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/autonomous-refill-exhaustion@1`.
- Evidence: vfs/autonomous-refill-exhaustion@1
- Outputs: docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md, data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py -q
- Acceptance: Every direct child has fresh criterion-level evidence ; no unsupported or stale result is promoted to proof ; the complete frozen repository inventory is accounted for ; accepted repairs retain normal authorization ; validation ; and merge gates ; drained work triggers bounded evidence-driven refill or a conclusive exhaustion receipt.
- Refinement depth: 1
- Embedding query: vfs/autonomous-refill-exhaustion@1
- AST query: vfs/autonomous-refill-exhaustion@1
- Parallel lane: vfs-assurance/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/autonomous-refill-exhaustion@1` with a narrow, verifiable change.

## VFS-G136 Prove vfs/repository-descriptor@1 for Bind every observation to an explicit multi-repository authority forest

- Status: active
- Parent: VFS-G010
- Fib priority: 5000
- Track: repository-identity
- Priority: P0
- Bundle: vfs-assurance/foundation
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/repository-descriptor@1`.
- Evidence: vfs/repository-descriptor@1
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_forest.py, test/api/test_agent_supervisor_repository_forest.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_forest.py -q
- Acceptance: Paths cannot escape a descriptor root ; sibling repositories are never conflated ; commit ; tree ; gitlinks ; dirty overlay ; ignore policy ; and authority affect identity ; external SwissKnife is read-only in the initial policy.
- Refinement depth: 2
- Embedding query: vfs/repository-descriptor@1
- AST query: vfs/repository-descriptor@1
- Parallel lane: vfs-assurance/foundation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/repository-descriptor@1` with a narrow, verifiable change.

## VFS-G137 Prove vfs/repository-forest-manifest@1 for Bind every observation to an explicit multi-repository authority forest

- Status: active
- Parent: VFS-G010
- Fib priority: 5001
- Track: repository-identity
- Priority: P0
- Bundle: vfs-assurance/foundation
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/repository-forest-manifest@1`.
- Evidence: vfs/repository-forest-manifest@1
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_forest.py, test/api/test_agent_supervisor_repository_forest.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_forest.py -q
- Acceptance: Paths cannot escape a descriptor root ; sibling repositories are never conflated ; commit ; tree ; gitlinks ; dirty overlay ; ignore policy ; and authority affect identity ; external SwissKnife is read-only in the initial policy.
- Refinement depth: 2
- Embedding query: vfs/repository-forest-manifest@1
- AST query: vfs/repository-forest-manifest@1
- Parallel lane: vfs-assurance/foundation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/repository-forest-manifest@1` with a narrow, verifiable change.

## VFS-G138 Prove vfs/exhaustive-file-inventory@1 for Exhaustively inventory and incrementally parse the SwissKnife corpus

- Status: active
- Parent: VFS-G020
- Fib priority: 5000
- Track: corpus-index
- Priority: P0
- Bundle: vfs-assurance/index
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/exhaustive-file-inventory@1`.
- Evidence: vfs/exhaustive-file-inventory@1
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_corpus_index.py, ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, test/api/test_agent_supervisor_repository_corpus_index.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_corpus_index.py test/api/test_agent_supervisor_program_ast_adapters.py -q
- Acceptance: The scan publishes included and excluded populations with reasons ; TypeScript/TSX/JavaScript/Python/JSON/Markdown inputs have provenance ; unchanged blobs are reused ; unexplained skips ; parser failures ; and truncation prevent an exhaustive verdict.
- Refinement depth: 2
- Embedding query: vfs/exhaustive-file-inventory@1
- AST query: vfs/exhaustive-file-inventory@1
- Parallel lane: vfs-assurance/index
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/exhaustive-file-inventory@1` with a narrow, verifiable change.

## VFS-G139 Prove vfs/incremental-ast-index@1 for Exhaustively inventory and incrementally parse the SwissKnife corpus

- Status: active
- Parent: VFS-G020
- Fib priority: 5001
- Track: corpus-index
- Priority: P0
- Bundle: vfs-assurance/index
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/incremental-ast-index@1`.
- Evidence: vfs/incremental-ast-index@1
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_corpus_index.py, ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, test/api/test_agent_supervisor_repository_corpus_index.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_corpus_index.py test/api/test_agent_supervisor_program_ast_adapters.py -q
- Acceptance: The scan publishes included and excluded populations with reasons ; TypeScript/TSX/JavaScript/Python/JSON/Markdown inputs have provenance ; unchanged blobs are reused ; unexplained skips ; parser failures ; and truncation prevent an exhaustive verdict.
- Refinement depth: 2
- Embedding query: vfs/incremental-ast-index@1
- AST query: vfs/incremental-ast-index@1
- Parallel lane: vfs-assurance/index
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/incremental-ast-index@1` with a narrow, verifiable change.

## VFS-G140 Prove vfs/repository-forest-replay@1 for Freeze and replay the initial four-repository manifest

- Status: active
- Parent: VFS-G011
- Fib priority: 8000
- Track: repository-identity
- Priority: P0
- Bundle: vfs-assurance/foundation
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/repository-forest-replay@1`.
- Evidence: vfs/repository-forest-replay@1
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_forest.py, test/api/test_agent_supervisor_repository_forest.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_forest.py -q
- Acceptance: Identical trees and policy reproduce the same portable forest CID ; a changed commit ; tree ; gitlink ; overlay ; or policy changes it ; unavailable roots fail closed with a typed reason.
- Refinement depth: 3
- Embedding query: vfs/repository-forest-replay@1
- AST query: vfs/repository-forest-replay@1
- Parallel lane: vfs-assurance/foundation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/repository-forest-replay@1` with a narrow, verifiable change.

## VFS-G141 Prove vfs/cid-profile@1 for Use canonical multiformats identities and dependency-aware content caches

- Status: active
- Parent: VFS-G030
- Fib priority: 5000
- Track: content-addressing
- Priority: P0
- Bundle: vfs-assurance/cas
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/cid-profile@1`.
- Evidence: vfs/cid-profile@1
- Outputs: ipfs_accelerate_py/agent_supervisor/multiformats_identity.py, ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, test/api/test_agent_supervisor_multiformats_identity.py
- Validation: python -m pytest test/api/test_agent_supervisor_multiformats_identity.py test/api/test_agent_supervisor_program_analysis_cache.py -q
- Acceptance: CIDv1/base32/dag-json/sha2-256 bytes are cross-package reproducible ; existing supervisor IDs retain compatibility mappings ; all semantic dependencies and policy versions participate in cache keys ; corruption and stale/negative results fail closed.
- Refinement depth: 2
- Embedding query: vfs/cid-profile@1
- AST query: vfs/cid-profile@1
- Parallel lane: vfs-assurance/cas
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/cid-profile@1` with a narrow, verifiable change.

## VFS-G142 Prove vfs/dependency-cache@1 for Use canonical multiformats identities and dependency-aware content caches

- Status: active
- Parent: VFS-G030
- Fib priority: 5001
- Track: content-addressing
- Priority: P0
- Bundle: vfs-assurance/cas
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/dependency-cache@1`.
- Evidence: vfs/dependency-cache@1
- Outputs: ipfs_accelerate_py/agent_supervisor/multiformats_identity.py, ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, test/api/test_agent_supervisor_multiformats_identity.py
- Validation: python -m pytest test/api/test_agent_supervisor_multiformats_identity.py test/api/test_agent_supervisor_program_analysis_cache.py -q
- Acceptance: CIDv1/base32/dag-json/sha2-256 bytes are cross-package reproducible ; existing supervisor IDs retain compatibility mappings ; all semantic dependencies and policy versions participate in cache keys ; corruption and stale/negative results fail closed.
- Refinement depth: 2
- Embedding query: vfs/dependency-cache@1
- AST query: vfs/dependency-cache@1
- Parallel lane: vfs-assurance/cas
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/dependency-cache@1` with a narrow, verifiable change.

## VFS-G143 Prove vfs/language-edge-resolution@1 for Resolve dynamic language features without inventing call edges

- Status: active
- Parent: VFS-G021
- Fib priority: 8000
- Track: corpus-index
- Priority: P0
- Bundle: vfs-assurance/index
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `vfs/language-edge-resolution@1`.
- Evidence: vfs/language-edge-resolution@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, ipfs_accelerate_py/agent_supervisor/program_graph.py, test/api/test_agent_supervisor_program_ast_adapters.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_ast_adapters.py test/api/test_agent_supervisor_program_graph.py -q
- Acceptance: Every edge cites a source span and resolver rule ; ambiguous and unsupported constructs remain explicit ; adversarial name collisions and re-exports cannot become forged direct calls.
- Refinement depth: 3
- Embedding query: vfs/language-edge-resolution@1
- AST query: vfs/language-edge-resolution@1
- Parallel lane: vfs-assurance/index
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `vfs/language-edge-resolution@1` with a narrow, verifiable change.

## VFS-G144 Prove objective validation repair for Project a provenance-preserving program and GraphRAG evidence graph

- Status: active
- Parent: VFS-G040
- Fib priority: 5000
- Track: program-graph
- Priority: P0
- Bundle: vfs-assurance/graph
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `objective validation repair`.
- Evidence: objective validation repair
- Outputs: ipfs_accelerate_py/agent_supervisor/program_graph.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_program_graph_provider.py, test/api/test_agent_supervisor_program_graph.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_graph.py test/api/test_agent_supervisor_ipfs_datasets_program_graph_provider.py -q
- Acceptance: Node and edge provenance is content bound ; graph chunks are deterministic and bounded ; retrieval returns compact references and ranking reasons ; provider absence degrades explicitly ; GraphRAG output cannot create completion or proof authority.
- Refinement depth: 2
- Embedding query: objective validation repair
- AST query: objective validation repair
- Parallel lane: vfs-assurance/graph
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `objective validation repair` with a narrow, verifiable change.

## VFS-G145 Prove objective validation repair for Prove transitive cache invalidation and bounded storage

- Status: active
- Parent: VFS-G031
- Fib priority: 8000
- Track: content-addressing
- Priority: P0
- Bundle: vfs-assurance/cas
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `objective validation repair`.
- Evidence: objective validation repair
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, test/api/test_agent_supervisor_program_analysis_cache.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_analysis_cache.py test/api/test_agent_supervisor_runtime_cas.py -q
- Acceptance: Every changed identity dimension has a test ; unrelated components remain reusable ; concurrent exact misses collapse ; failed flights clean up ; retained artifacts and compact receipts stay within declared count and byte bounds.
- Refinement depth: 3
- Embedding query: objective validation repair
- AST query: objective validation repair
- Parallel lane: vfs-assurance/cas
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `objective validation repair` with a narrow, verifiable change.

## VFS-G146 Prove objective validation repair for Extract versioned expected and observed contracts without circular reasoning

- Status: active
- Parent: VFS-G050
- Fib priority: 5000
- Track: contract-ir
- Priority: P0
- Bundle: vfs-assurance/contracts
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `objective validation repair`.
- Evidence: objective validation repair
- Outputs: ipfs_accelerate_py/agent_supervisor/program_contracts.py, ipfs_accelerate_py/agent_supervisor/contract_extractor.py, test/api/test_agent_supervisor_contract_extractor.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_contracts.py test/api/test_agent_supervisor_contract_extractor.py -q
- Acceptance: Contract sources and conflicts are explicit ; implementation observations cannot define their own expectation ; inputs ; outputs ; errors ; effects ; authorization ; idempotence ; ordering ; atomicity ; resources ; and degradation are represented or marked unsupported.
- Refinement depth: 2
- Embedding query: objective validation repair
- AST query: objective validation repair
- Parallel lane: vfs-assurance/contracts
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `objective validation repair` with a narrow, verifiable change.

## VFS-G147 Prove objective validation repair for Answer minimal call-slice and contract-impact queries

- Status: active
- Parent: VFS-G041
- Fib priority: 8000
- Track: program-graph
- Priority: P1
- Bundle: vfs-assurance/graph
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `objective validation repair`.
- Evidence: objective validation repair
- Outputs: ipfs_accelerate_py/agent_supervisor/program_graph_queries.py, test/api/test_agent_supervisor_program_graph_queries.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_graph_queries.py -q
- Acceptance: Seeded transitive callers/callees and MCP paths are complete within scope ; unrelated source is omitted ; limits never silently convert an incomplete slice into a complete result.
- Refinement depth: 3
- Embedding query: objective validation repair
- AST query: objective validation repair
- Parallel lane: vfs-assurance/graph
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `objective validation repair` with a narrow, verifiable change.

## VFS-G148 Prove objective validation repair for Prove whether SwissKnife MCP++ calls reach the intended IPFS implementation

- Status: active
- Parent: VFS-G060
- Fib priority: 5000
- Track: mcp-interop
- Priority: P0
- Bundle: vfs-assurance/mcplusplus
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `objective validation repair`.
- Evidence: objective validation repair
- Outputs: ipfs_accelerate_py/agent_supervisor/mcplusplus_contract_resolver.py, test/api/test_agent_supervisor_mcplusplus_contract_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_mcplusplus_contract_resolver.py -q
- Acceptance: Same-name helpers ; mocks ; static payloads ; copied manifests ; and fallbacks do not prove invocation ; ambiguous/dynamic registrations remain explicit ; Python/TypeScript/schema names and errors are checked for parity.
- Refinement depth: 2
- Embedding query: objective validation repair
- AST query: objective validation repair
- Parallel lane: vfs-assurance/mcplusplus
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `objective validation repair` with a narrow, verifiable change.
