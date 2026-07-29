# IPFS Kit VFS Symbolic Assurance Task Board

This board is consumed by the `ipfs_accelerate_py` implementation supervisor
with task prefix `VFS-`. Tasks implement the plan in
`IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md`. Large source, AST, graph, proof, and
witness bodies must remain content-addressed artifacts. Model prompts receive
compact references and bounded spans only.

## VFS-001 Define program-assurance evidence, claim, finding, and stage-receipt contracts

- Status: completed
- Completion: manual
- Priority: P0
- Track: assurance-contracts
- Depends on:
- Goal id: VFS-G000
- Outputs: ipfs_accelerate_py/agent_supervisor/program_assurance_contracts.py, test/api/test_agent_supervisor_program_assurance_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_assurance_contracts.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/foundation
- Parallel lane: assurance-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_assurance_contracts.py, test/api/test_agent_supervisor_program_assurance_contracts.py
- Conflict policy: Keep this standalone; do not edit agent_supervisor/__init__.py or wire orchestration yet.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: First inspect analysis_contracts.py, formal_verification_contracts.py, scan_receipts.py, supervisor_v2_contracts.py, and ipfs_datasets_py/logic/ir_core. Define immutable bounded records for repository observation, claim levels (`observed_syntax`, `resolved_static`, `model_proved`, `model_disproved`, `runtime_witnessed`, `zk_trace_attested`), explicit inconclusive states, expected/observed contracts, counterexamples, findings, stage receipts, and artifact references. Deterministic serialization must reject unbounded bodies, non-finite values, forged identities, stale authority, illegal claim promotion, and a ZK receipt presented as semantic proof. Add round-trip, identity, bounds, and invalid-state tests.

## VFS-002 Implement independently bound repository descriptors and authority forests

- Status: completed
- Completion: manual
- Priority: P0
- Track: repository-identity
- Depends on:
- Goal id: VFS-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_forest.py, test/api/test_agent_supervisor_repository_forest.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_forest.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/foundation
- Parallel lane: repository-forest
- Resource class: io-git
- Predicted files: ipfs_accelerate_py/agent_supervisor/repository_forest.py, test/api/test_agent_supervisor_repository_forest.py
- Conflict policy: Reuse existing checkout locks and ResultBinding identity conventions; do not change mutation authority.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Add canonical records for repository ID, portable tree/commit/gitlink closure, local locator, dirty overlay digest, ignore policy, case/Unicode policy, read/write authority, and a forest identity. Resolve roots and symlinks fail-closed; never infer that sibling roots share Git authority. Initial policy must model `/home/barberb/swissknife` as read-only and the accelerator checkout as the only write root. Test clean/dirty/submodule changes, path escapes, missing roots, duplicate aliases, portable replay, and deterministic identity.

## VFS-003 Add a frozen four-repository manifest loader and replay validator

- Status: todo
- Completion: manual
- Priority: P0
- Track: repository-identity
- Depends on: VFS-002
- Goal id: VFS-G011
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_forest.py, ipfs_accelerate_py/agent_supervisor/repository_forest_manifest.py, test/api/test_agent_supervisor_repository_forest_manifest.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_forest.py test/api/test_agent_supervisor_repository_forest_manifest.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/foundation
- Parallel lane: repository-manifest
- Resource class: io-git
- Predicted files: ipfs_accelerate_py/agent_supervisor/repository_forest_manifest.py, test/api/test_agent_supervisor_repository_forest_manifest.py
- Conflict policy: Extend the repository-forest contract after VFS-002; keep host locators outside portable CIDs.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: Load a reviewed manifest for SwissKnife, accelerator, kit, and datasets roots; derive fresh descriptors rather than trusting recorded commits; persist portable and local projections separately; validate expected roots/authority/policy on replay. A changed tree, gitlink, overlay, policy, or analyzer profile must change the relevant identity, while equivalent relocations retain portable identity. Never log credentials or environment secrets.

## VFS-004 Probe real ipfs_datasets_py AST, GraphRAG, IR, solver, multiformats, and ZKP capabilities

- Status: todo
- Completion: manual
- Priority: P0
- Track: provider-capabilities
- Depends on: VFS-001
- Goal id: VFS-G000
- Outputs: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_program_analysis_provider.py, test/api/test_agent_supervisor_ipfs_datasets_program_analysis_provider.py
- Validation: python -m pytest test/api/test_agent_supervisor_ipfs_datasets_program_analysis_provider.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/foundation
- Parallel lane: datasets-capabilities
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_program_analysis_provider.py, test/api/test_agent_supervisor_ipfs_datasets_program_analysis_provider.py
- Conflict policy: Extend the lazy optional-provider pattern; do not import providers during package discovery.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Build a closed, lazy capability matrix for strict CID helpers, knowledge-graph/GraphRAG queries, logic/ir_core claims and protocols, cvc5/z3 compiler and executable availability, AST producers, and ZKP backends/circuits. Current observations such as cvc5 available, z3 absent, simulated ZKP defaults, or the limited crypto-exchange Python AST extractor are diagnostics, not constants. Reject package-presence inference, broken method signatures, simulated/fallback ZKP authority, legacy pseudo-CIDs, unhealthy providers, incompatible schemas, and unbounded outputs. Add cold-import, unavailable, incompatible, partial, timeout, and current-probe tests.

## VFS-005 Build an exhaustive Git-aware multi-repository corpus inventory

- Status: todo
- Completion: manual
- Priority: P0
- Track: corpus-index
- Depends on: VFS-001, VFS-002
- Goal id: VFS-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/repository_corpus_index.py, test/api/test_agent_supervisor_repository_corpus_index.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_corpus_index.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/index
- Parallel lane: corpus-inventory
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/repository_corpus_index.py, test/api/test_agent_supervisor_repository_corpus_index.py
- Conflict policy: Inventory only; leave parsing and call resolution to later tasks.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Enumerate committed Git blobs plus an explicitly allowed dirty overlay for every repository descriptor. Classify source, generated source, schema, docs, tests, fixtures, vendored, archive, build output, symlink, submodule, ignored, binary, and oversized entries. Emit blob identities, canonical paths, modes, sizes, inclusion decisions, and reason codes. Unexplained omissions, path escapes, changing inputs, parser-eligible unreadable files, or bounds truncation prevent an exhaustive receipt. Test SwissKnife-like TS/TSX trees, submodules, symlinks, Unicode/case collisions, ignored output, dirty files, deterministic ordering, incremental reuse, and bounded manifests.

## VFS-006 Adapt TypeScript, TSX, and JavaScript evidence into canonical AST blob records

- Status: todo
- Completion: manual
- Priority: P0
- Track: corpus-index
- Depends on: VFS-005
- Goal id: VFS-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, test/api/test_agent_supervisor_program_ast_adapters_typescript.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_ast_adapters_typescript.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/index
- Parallel lane: typescript-ast
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, test/api/test_agent_supervisor_program_ast_adapters_typescript.py
- Conflict policy: Reuse conflict_graph.ASTBlobRecord and analysis_ast_index; do not add a competing AST schema.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: First inspect conflict_graph.py and existing JS/TS parsing. Emit content-bound definitions, exports/re-exports, imports/dynamic imports, calls/new expressions, types, decorators, callbacks, registrations, string literals relevant to MCP, and source spans with parser/version diagnostics. Preserve syntax errors, unsupported nodes, and ambiguity. Test SwissKnife-style service connectors, barrel exports, aliases, optional chaining, async calls, callback registration, name collisions, malformed files, exact reuse, and changed-blob invalidation.

## VFS-007 Adapt Python, JSON/Schema, Markdown, and manifest evidence into the same index

- Status: todo
- Completion: manual
- Priority: P0
- Track: corpus-index
- Depends on: VFS-005
- Goal id: VFS-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, test/api/test_agent_supervisor_program_ast_adapters_mixed.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_ast_adapters_mixed.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/index
- Parallel lane: mixed-ast
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, test/api/test_agent_supervisor_program_ast_adapters_mixed.py
- Conflict policy: Coordinate the shared adapter module with VFS-006; partition language dispatch and tests cleanly.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Reuse Python `ast` and canonical AST records; do not rely on the limited crypto-exchange heuristic extractor as a general call graph. Extract signatures, annotations, imports, definitions, calls, decorators, exceptions, context managers, async behavior, and source spans. Extract JSON Schema/MCP manifests and normative Markdown headings/code references as typed non-code evidence. Test malformed inputs, duplicate keys where detectable, schema refs, fenced examples versus normative text, generated manifests, import aliases, monkey-patch ambiguity, exact reuse, and explicit unsupported results.

## VFS-008 Build the canonical cross-repository program evidence graph

- Status: todo
- Completion: manual
- Priority: P0
- Track: program-graph
- Depends on: VFS-001, VFS-006, VFS-007
- Goal id: VFS-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/program_graph.py, test/api/test_agent_supervisor_program_graph.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_graph.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/graph
- Parallel lane: canonical-program-graph
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_graph.py, test/api/test_agent_supervisor_program_graph.py
- Conflict policy: Compose semantic_dependency_graph and code_evidence_graph contracts; do not change GraphRAG or contract extraction yet.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Add immutable bounded nodes/edges for repositories, blobs, modules, symbols, definitions, imports, exports, calls, types, schemas, docs, tests, MCP tools/registrations, transports, and artifacts. Every record binds producer, blob CID, span, resolver status, and forest identity. Graph chunks and indexes must be deterministic, content addressed, incrementally replace changed components, reject forged/dangling edges and cycles where illegal, and expose completeness/frontier metadata.

## VFS-009 Resolve cross-language calls conservatively and retain unknown frontiers

- Status: todo
- Completion: manual
- Priority: P0
- Track: program-graph
- Depends on: VFS-008
- Goal id: VFS-G021
- Outputs: ipfs_accelerate_py/agent_supervisor/program_call_resolver.py, test/api/test_agent_supervisor_program_call_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_call_resolver.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/graph
- Parallel lane: call-resolution
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_call_resolver.py, test/api/test_agent_supervisor_program_call_resolver.py
- Conflict policy: Consume the canonical graph without mutating source AST records.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Resolve relative/package imports, aliases, re-exports, class/member calls, known registrations, generated SDK methods, and explicit cross-package interfaces. Model dependency injection, callbacks, monkey patches, dynamic imports, subprocess, HTTP, RPC, libp2p, and MCP as typed candidate/ambiguous/external edges with required evidence. Test adversarial same-name functions, re-export loops, namespace packages, optional imports, generated clients, uninstalled dependencies, and deterministic confidence/reason codes. Never manufacture a direct edge to improve coverage.

## VFS-010 Add a strict DAG-JSON/CIDv1/multihash identity bridge

- Status: todo
- Completion: manual
- Priority: P0
- Track: content-addressing
- Depends on: VFS-001, VFS-004
- Goal id: VFS-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/multiformats_identity.py, test/api/test_agent_supervisor_multiformats_identity.py
- Validation: python -m pytest test/api/test_agent_supervisor_multiformats_identity.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/cas
- Parallel lane: multiformats-identity
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/multiformats_identity.py, test/api/test_agent_supervisor_multiformats_identity.py
- Conflict policy: Wrap `ipfs_datasets_py.utils.cid_utils`; preserve existing supervisor SHA identities with an explicit compatibility mapping.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: Use strict finite DAG-JSON bytes, CIDv1, base32, dag-json/raw codecs as declared, and sha2-256 multihashes. Validate canonical lowercase form, codec, version, base, digest size, and bytes. Add typed links between existing content_identity/runtime artifact IDs and CIDs without silently replacing persisted identities. Reject `default=repr`, unsorted JSON, NaN/infinity, timestamps in identity, truncated pseudo-CIDs, double hashing, ambiguous raw/string/file input, malformed multihashes, and cross-package codec drift. Test known vectors and independent round trips.

## VFS-011 Implement the dependency-aware program-analysis cache

- Status: todo
- Completion: manual
- Priority: P0
- Track: content-addressing
- Depends on: VFS-008, VFS-010
- Goal id: VFS-G031
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, test/api/test_agent_supervisor_program_analysis_cache.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_analysis_cache.py test/api/test_agent_supervisor_runtime_cas.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/cas
- Parallel lane: analysis-cache
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, test/api/test_agent_supervisor_program_analysis_cache.py
- Conflict policy: Reuse RuntimeCAS, AnalysisCache, CacheCoordinator, and ArtifactStore rather than implementing a new object store.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Key inventory, AST, graph, contract, proof, runtime, and ZK receipts by the complete forest/objective/policy/analyzer/schema/config/query/capability/assumption/toolchain dependency population. Store compact receipts and immutable artifact references. Prove exact hits, transitive invalidation, unrelated-component reuse, process and cross-process single-flight, atomic writes, corruption recovery, authority namespace isolation, negative TTL, quotas/GC, and zero stale authoritative hits under concurrency and restart.

## VFS-012 Add a bounded ipfs_datasets_py GraphRAG/IPLD projection provider

- Status: todo
- Completion: manual
- Priority: P1
- Track: program-graph
- Depends on: VFS-004, VFS-008, VFS-010
- Goal id: VFS-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_program_graph_provider.py, test/api/test_agent_supervisor_ipfs_datasets_program_graph_provider.py
- Validation: python -m pytest test/api/test_agent_supervisor_ipfs_datasets_program_graph_provider.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/graph
- Parallel lane: graphrag-projection
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_program_graph_provider.py, test/api/test_agent_supervisor_ipfs_datasets_program_graph_provider.py
- Conflict policy: Canonical graph remains owned by VFS-008; this is an optional lazy projection and query adapter.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Probe and use compatible `ipfs_datasets_py` graph/IPLD/index/query APIs, or return an explicit local-fallback/inconclusive result. Persist deterministic chunk CIDs and provenance links, not recursive unbounded graphs. Bound item, depth, byte, and time costs; return references, scores, and ranking reasons. GraphRAG may rank only canonical evidence and can never create calls, contracts, findings, proofs, completion, or mutation authority. Add unavailable, incompatible, partial, poisoned-result, and deterministic query tests.

## VFS-013 Implement minimal dependency-complete call and impact slice queries

- Status: todo
- Completion: manual
- Priority: P1
- Track: program-graph
- Depends on: VFS-009, VFS-012
- Goal id: VFS-G041
- Outputs: ipfs_accelerate_py/agent_supervisor/program_graph_queries.py, test/api/test_agent_supervisor_program_graph_queries.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_graph_queries.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/graph
- Parallel lane: graph-queries
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_graph_queries.py, test/api/test_agent_supervisor_program_graph_queries.py
- Conflict policy: Query immutable graph artifacts; do not embed source bodies in responses.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: Support symbol callers/callees, changed-blob impact, contract consumers/producers, MCP end-to-end routes, VFS operation surfaces, proof dependencies, and shortest counterexample slices. Results must be stable, bounded, provenance-bearing, and explicit about cycles, ambiguity, missing nodes, excluded repositories, and truncated frontiers. Prove minimality on seeded graphs without omitting required dependencies.

## VFS-014 Define a versioned expected/observed program contract IR

- Status: todo
- Completion: manual
- Priority: P0
- Track: contract-ir
- Depends on: VFS-001
- Goal id: VFS-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/program_contracts.py, test/api/test_agent_supervisor_program_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_contracts.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/contracts
- Parallel lane: contract-ir
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_contracts.py, test/api/test_agent_supervisor_program_contracts.py
- Conflict policy: Standalone contract definitions only; no source extraction or theorem translation in this task.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Define deterministic contracts for symbol/interface identity, source precedence, inputs/outputs, sync/async, errors, side effects, capabilities, authorization, idempotence, ordering, atomicity, consistency, resource bounds, and fallback/degradation. Represent refinements, conflicts, unsupported semantics, assumptions, and applicability. Separate expectations from observations so implementation behavior cannot validate itself. Add strict serialization, compatibility/version, bounds, conflict, subtyping, and forged-source tests.

## VFS-015 Extract contracts from IDL, schema, types, tests, specs, and observations

- Status: todo
- Completion: manual
- Priority: P0
- Track: contract-ir
- Depends on: VFS-008, VFS-014
- Goal id: VFS-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/contract_extractor.py, test/api/test_agent_supervisor_contract_extractor.py
- Validation: python -m pytest test/api/test_agent_supervisor_contract_extractor.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/contracts
- Parallel lane: contract-extraction
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/contract_extractor.py, test/api/test_agent_supervisor_contract_extractor.py
- Conflict policy: Consume canonical program graph and contract IR; retain raw sources only by artifact reference.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Implement explicit precedence across reviewed MCP++/MCP IDL and JSON Schema, stable typed interfaces, executable conformance tests, normative specs/docs, generated manifests/SDKs, and implementation observations. Emit source/spans/CIDs, extraction rule, confidence class, conflicts, and unsupported clauses. Distinguish examples, mocks, fixtures, deprecated variants, and generated copies. Test contradictory docs/types/tests, missing refs, overloads, async/errors, schema unions, version negotiation, optional capability, and circular self-expectations.

## VFS-016 Implement symbolic contract comparison and counterexample generation

- Status: todo
- Completion: manual
- Priority: P0
- Track: contract-ir
- Depends on: VFS-009, VFS-015
- Goal id: VFS-G051
- Outputs: ipfs_accelerate_py/agent_supervisor/contract_checker.py, test/api/test_agent_supervisor_contract_checker.py
- Validation: python -m pytest test/api/test_agent_supervisor_contract_checker.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/contracts
- Parallel lane: contract-checker
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/contract_checker.py, test/api/test_agent_supervisor_contract_checker.py
- Conflict policy: Pure checking over immutable IR; proof-provider routing follows in later tasks.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Compare input/output variance, required/optional fields, error maps, async shape, effects, authorization, idempotence, ordering, atomicity, bounds, and degradation across declared call paths. Emit proved-compatible only for closed supported rules, witnessed mismatch with a minimal counterexample when conclusive, or typed ambiguous/unsupported/timeout/stale results. Add seeded broken contracts, compatible refinements, dynamic dispatch uncertainty, omitted effects, path traversal, cache staleness, and adversarial same-name fixtures with deterministic identities.

## VFS-017 Resolve SwissKnife MCP++ calls to actual package registrations and implementations

- Status: todo
- Completion: manual
- Priority: P0
- Track: mcp-interop
- Depends on: VFS-009, VFS-015
- Goal id: VFS-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/mcplusplus_contract_resolver.py, test/api/test_agent_supervisor_mcplusplus_contract_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_mcplusplus_contract_resolver.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/mcplusplus
- Parallel lane: mcplusplus-static
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/mcplusplus_contract_resolver.py, test/api/test_agent_supervisor_mcplusplus_contract_resolver.py
- Conflict policy: Read SwissKnife and Mcp-Plus-Plus as external evidence; do not mutate their checkouts.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Trace caller -> connector -> negotiated profile/transport -> tools/list/interface -> tools/call -> server registry -> adapter -> package implementation -> result/error mapping. Resolve TypeScript/Python names, JSON Schema, manifests, aliases, version/profile negotiation, HTTP and mcp+p2p edges. Same-name local helpers, mocks, test servers, copied manifests, static dashboards, legacy fallbacks, or imports without call edges cannot prove invocation. Emit explicit ambiguous/external frontiers and manifest drift witnesses.

## VFS-018 Add hermetic MCP++ runtime contract witnesses

- Status: todo
- Completion: manual
- Priority: P1
- Track: mcp-interop
- Depends on: VFS-016, VFS-017
- Goal id: VFS-G061
- Outputs: ipfs_accelerate_py/agent_supervisor/mcplusplus_runtime_witness.py, test/api/test_agent_supervisor_mcplusplus_runtime_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_mcplusplus_runtime_contracts.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/mcplusplus
- Parallel lane: mcplusplus-runtime
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/mcplusplus_runtime_witness.py, test/api/test_agent_supervisor_mcplusplus_runtime_contracts.py
- Conflict policy: Use local fixtures and existing MCP++ conformance assets; network stays disabled by default.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: Run selected real registered adapters in a bounded subprocess/runtime fixture and record tool discovery, input validation, dispatch target identity, output/error schema, capability negotiation, transport, timeout, and cleanup. Distinguish mock/fixture results from production implementations. Test malformed calls, missing tools, wrong schema, unavailable backend, cancellation, profile mismatch, stale manifest, and deterministic receipt replay. Runtime witnesses supplement rather than replace static completeness or formal proof.

## VFS-019 Translate supported contracts and call slices through ipfs_datasets_py IR

- Status: todo
- Completion: manual
- Priority: P0
- Track: formal-proof
- Depends on: VFS-004, VFS-013, VFS-016
- Goal id: VFS-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/code_contract_logic.py, test/api/test_agent_supervisor_code_contract_logic.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_contract_logic.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/proof
- Parallel lane: logic-translation
- Resource class: cpu-proof-solver
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_contract_logic.py, test/api/test_agent_supervisor_code_contract_logic.py
- Conflict policy: Reuse formal_logic_vocabulary and ipfs_datasets_py/logic/ir_core claims/protocols; do not create a separate theorem language.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Translate finite supported predicates for types, nullability, errors, effects, authorization, state transitions, ordering, idempotence, and bounded reachability into immutable claims/obligations with source and assumption CIDs. Emit a round-trip/conformance receipt and explicit unsupported semantics. Reject unbound axioms, name capture, sort mismatch, partial call slices presented as closed, silent approximation, and changed translator/ruleset reuse. Test valid, invalid, ambiguous, and unsupported translations.

## VFS-020 Route code-contract obligations through capability-probed solvers and authoritative checks

- Status: todo
- Completion: manual
- Priority: P0
- Track: formal-proof
- Depends on: VFS-019
- Goal id: VFS-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/code_contract_prover.py, test/api/test_agent_supervisor_code_contract_prover.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_contract_prover.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/proof
- Parallel lane: contract-prover
- Resource class: cpu-proof-solver
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_contract_prover.py, test/api/test_agent_supervisor_code_contract_prover.py
- Conflict policy: Compose existing multi-prover, kernel, and ipfs_datasets logic-provider contracts; candidate output has no authority.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Probe cvc5, z3, and any other admitted backend per run; compile deterministic bounded requests through compatible ipfs_datasets IR backends; retain attempt/results/receipts; independently validate proof or model/counterexample under policy. Missing z3, timeout, unknown, malformed output, wrong theorem, stale solver/toolchain, forged authority, omitted effects, inconsistent assumptions, and capability loss remain non-conclusive. Add solver fixture, unavailable-backend, portfolio, cancellation, cache, and replay tests.

## VFS-021 Compile minimal dependency-complete proof and counterexample contexts

- Status: todo
- Completion: manual
- Priority: P1
- Track: formal-proof
- Depends on: VFS-013, VFS-020
- Goal id: VFS-G071
- Outputs: ipfs_accelerate_py/agent_supervisor/code_contract_proof_context.py, test/api/test_agent_supervisor_code_contract_proof_context.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_contract_proof_context.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/proof
- Parallel lane: proof-context
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_contract_proof_context.py, test/api/test_agent_supervisor_code_contract_proof_context.py
- Conflict policy: Reuse proof_context and proof_directed_retrieval; do not embed full graph/source.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: Select the smallest closed set of contracts, calls, definitions, assumptions, effects, and rules required by one obligation. Persist deterministic inclusion/exclusion reasons and content-addressed expansion handles. Required inputs cannot be truncated; limits yield incomplete status. Identical contexts reuse exact receipts, changed dependencies invalidate, and delta retries transmit only new counterexample or requested evidence. Measure bytes and item counts without invoking an LLM.

## VFS-022 Define ZK public inputs, witness policy, and trace semantics for program assurance

- Status: todo
- Completion: manual
- Priority: P1
- Track: zero-knowledge
- Depends on: VFS-010, VFS-019, VFS-020
- Goal id: VFS-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, test/api/test_agent_supervisor_program_analysis_zkp.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_analysis_zkp.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/zkp
- Parallel lane: zkp-contracts
- Resource class: cpu-proof-solver
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, test/api/test_agent_supervisor_program_analysis_zkp.py
- Conflict policy: Define contracts and shadow workflow only; circuit implementation follows separately.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Define canonical public commitments for forest, inventory, contract, call slice, assumptions, analyzer/resolver/translator/prover versions, result, circuit, proving/verifying keys, ceremony, and public-input codec. Define private witness/redaction policy and supported deterministic trace transitions. The contract must state that trace validity does not prove inventory completeness, translator soundness, arbitrary runtime semantics, or a theorem beyond the committed supported result. Add tampering, replay, privacy-leak, version, and illegal claim-promotion tests.

## VFS-023 Implement the first bounded program-contract trace circuit and cross-codec vectors

- Status: todo
- Completion: manual
- Priority: P1
- Track: zero-knowledge
- Depends on: VFS-022
- Goal id: VFS-G080
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/provekit/circuits/program_contract_trace/Nargo.toml, ipfs_datasets_py/ipfs_datasets_py/logic/zkp/provekit/circuits/program_contract_trace/src/main.nr, ipfs_datasets_py/tests/unit/logic/zkp/test_program_contract_trace.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/zkp/test_program_contract_trace.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/zkp
- Parallel lane: zkp-circuit
- Resource class: cpu-proof-solver
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/provekit/circuits/program_contract_trace/Nargo.toml, ipfs_datasets_py/ipfs_datasets_py/logic/zkp/provekit/circuits/program_contract_trace/src/main.nr, ipfs_datasets_py/tests/unit/logic/zkp/test_program_contract_trace.py
- Conflict policy: Work only in the configured ipfs_datasets_py submodule; do not modify existing knowledge-of-axioms or TDFOL circuits.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Implement a versioned bounded circuit that verifies commitment openings and the declared supported trace transitions for one finite contract-check result. Provide canonical Python/Noir public-input and witness vectors with explicit field encoding and bounds. Reject reordered/omitted steps, forged result, wrong contract/call slice/forest/version, overflow, padding ambiguity, and altered key/circuit identity. Do not copy the simulated backend or claim general function-call semantics. If the real toolchain is unavailable, land deterministic vectors and explicit skipped capability tests without emitting an authoritative proof.

## VFS-024 Enforce production ZK capability, setup, ceremony, and verifier conformance

- Status: todo
- Completion: manual
- Priority: P0
- Track: zero-knowledge
- Depends on: VFS-004, VFS-022, VFS-023
- Goal id: VFS-G081
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, test/api/test_agent_supervisor_program_analysis_zkp_conformance.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_analysis_zkp.py test/api/test_agent_supervisor_program_analysis_zkp_conformance.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/zkp
- Parallel lane: zkp-conformance
- Resource class: cpu-proof-solver
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, test/api/test_agent_supervisor_program_analysis_zkp_conformance.py
- Conflict policy: Extend VFS-022 after the circuit lands; preserve all simulated paths as non-authoritative fixtures.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Probe executable/architecture, backend, circuit version, setup artifacts, ceremony, proving/verifying keys, public-input codec, proof schema, independent verifier, bounds, and cancellation. Simulated defaults, knowledge-graph fail-open fallback, placeholder field encoding, v1 nonzero-only circuits, incompatible TDFOL-only circuits, unversioned or missing artifacts, and stale capabilities must fail closed for authority. Verify deterministic proof receipt identity, independent replay, corrupted proof/key/input rejection, capability loss invalidation, shadow-only rollout, and no semantic claim promotion.

## VFS-025 Inventory and classify all IPFS Kit VFS surfaces and variants

- Status: todo
- Completion: manual
- Priority: P0
- Track: vfs-drift
- Depends on: VFS-005, VFS-008, VFS-014
- Goal id: VFS-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_surface_inventory.py, test/api/test_agent_supervisor_vfs_surface_inventory.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_surface_inventory.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/vfs
- Parallel lane: vfs-inventory
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/vfs_surface_inventory.py, test/api/test_agent_supervisor_vfs_surface_inventory.py
- Conflict policy: Read ipfs_kit_py as evidence and emit inventory artifacts; do not repair VFS modules in this task.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Inventory `ipfs_fsspec.py`, `enhanced_fsspec.py`, VFS managers, bucket managers, journals/WAL, version/snapshot trackers, backend adapters, handlers, endpoints, controllers, tools, servers, SDK/manifests, exports, and `.fixed/.full/.new/.clean/.optimized/.broken` variants. Classify canonical, compatibility, generated, test, archive, placeholder, duplicate, shadow, and unknown using evidence. Map definitions/imports/callers/registrations/tests/docs and contradictions. Variant presence alone cannot become a defect. Publish completeness and unexplained-surface diagnostics.

## VFS-026 Define the canonical VFS operation and invariant contract pack

- Status: todo
- Completion: manual
- Priority: P0
- Track: vfs-drift
- Depends on: VFS-015, VFS-025
- Goal id: VFS-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_contract_pack.py, test/api/test_agent_supervisor_vfs_contract_pack.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_contract_pack.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/vfs
- Parallel lane: vfs-contract-pack
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/vfs_contract_pack.py, test/api/test_agent_supervisor_vfs_contract_pack.py
- Conflict policy: Build contracts from reviewed evidence and record conflicts; do not pick an implementation as canonical by popularity.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Define versioned path/Unicode/root/traversal/mount, read/write/open/close/seek/stat/list/mkdir/remove/rename/copy, bytes/text, sync/async, error, CID/size, atomicity, journal replay, versioning, cache/pin coherence, backend negotiation, authorization, resource, and degradation invariants. Map each public Python/CLI/MCP/MCP++/HTTP/libp2p surface to supported operations and source contracts. Missing or conflicting expectations stay unresolved. Add canonical vectors and compatible/incompatible facade examples.

## VFS-027 Build a hermetic differential VFS contract harness

- Status: todo
- Completion: manual
- Priority: P0
- Track: vfs-drift
- Depends on: VFS-016, VFS-026
- Goal id: VFS-G091
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_differential_harness.py, test/api/test_agent_supervisor_vfs_differential_harness.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_differential_harness.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/vfs
- Parallel lane: vfs-differential
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/vfs_differential_harness.py, test/api/test_agent_supervisor_vfs_differential_harness.py
- Conflict policy: Use temporary memory/local backends and bounded subprocesses; never touch user VFS state.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Generate finite canonical operation traces and compare selected real VFS/fsspec/manager/bucket/handler surfaces under identical fixtures. Normalize only contract-approved representation differences. Record exact runtime/toolchain/fixture/result/error/CID identities and cleanup. Detect seeded path, bytes/text, stat/list, rename atomicity, journal, cache, authorization, fallback, and silent-success drift; avoid false mismatches for compatible behavior. Unavailable or mock backends remain explicit and non-authoritative.

## VFS-028 Check VFS manifest, SDK, MCP, and MCP++ parity end to end

- Status: todo
- Completion: manual
- Priority: P0
- Track: vfs-drift
- Depends on: VFS-017, VFS-018, VFS-026
- Goal id: VFS-G091
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_mcp_contract_checker.py, test/api/test_agent_supervisor_vfs_mcp_contract_checker.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_mcp_contract_checker.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/vfs
- Parallel lane: vfs-mcp-parity
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/vfs_mcp_contract_checker.py, test/api/test_agent_supervisor_vfs_mcp_contract_checker.py
- Conflict policy: Consume MCP resolver/runtime receipts and VFS contracts; do not regenerate package manifests.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Compare VFS Python signatures, registered tools, tools/list schemas, generated JSON manifests/TypeScript SDKs, SwissKnife connector calls, transport profiles, result/error mappings, capability/degradation claims, and real implementation targets. Report stale generated artifacts, missing registrations, extra unreachable tools, wrong aliases/schema/errors, direct local bypass, mock/fallback dispatch, and ambiguous paths with minimal witnesses. Same text without a resolved call path is insufficient.

## VFS-029 Implement the append-only content-addressed contract finding ledger

- Status: todo
- Completion: manual
- Priority: P0
- Track: finding-generation
- Depends on: VFS-001, VFS-010, VFS-016
- Goal id: VFS-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/contract_findings.py, test/api/test_agent_supervisor_contract_findings.py
- Validation: python -m pytest test/api/test_agent_supervisor_contract_findings.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/findings
- Parallel lane: finding-ledger
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/contract_findings.py, test/api/test_agent_supervisor_contract_findings.py
- Conflict policy: Finding storage is diagnostic; it must not mutate tasks or source.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Persist immutable finding records with CID, claim level, status, severity, confidence, freshness, repositories/symbols/interfaces, expected/observed contract CIDs, call slice, counterexample/proof/runtime/ZK references, assumptions, analyzer versions, root-cause/merge family, remediation scope, supersession, and rejection reasons. Deduplicate only equal semantic/root-cause/merge-fate findings. Preserve history while current projections invalidate stale entries. Test collisions, stale evidence, contradictory claims, poisoned severity, partial findings, replay, concurrency, and bounds.

## VFS-030 Add security-property/dataflow findings and SARIF projection

- Status: todo
- Completion: manual
- Priority: P0
- Track: finding-generation
- Depends on: VFS-009, VFS-026, VFS-029
- Goal id: VFS-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/security_contract_analysis.py, ipfs_accelerate_py/agent_supervisor/finding_sarif.py, test/api/test_agent_supervisor_security_contract_analysis.py
- Validation: python -m pytest test/api/test_agent_supervisor_security_contract_analysis.py test/api/test_agent_supervisor_finding_sarif.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/findings
- Parallel lane: security-findings
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/security_contract_analysis.py, ipfs_accelerate_py/agent_supervisor/finding_sarif.py, test/api/test_agent_supervisor_security_contract_analysis.py, test/api/test_agent_supervisor_finding_sarif.py
- Conflict policy: Use conservative symbolic rules over canonical edges; no LLM vulnerability classification.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Add bounded interprocedural rules for path traversal/scope loss, authorization/capability bypass, unsafe deserialization/command construction, secret flow, CID/integrity bypass, cache poisoning/staleness, symlink escape, silent fallback/mock success, journal/atomicity violations, and MCP schema/dispatch confusion. A vulnerability requires a declared security property, reachable or declared threat path, impact, and evidence; otherwise classify correctness drift or suspicion. Export deterministic bounded SARIF with artifact references and no secret/source-body leakage. Test seeded true/false positives and unknown dynamic paths.

## VFS-031 Materialize a stable repair task source from admitted findings

- Status: todo
- Completion: manual
- Priority: P0
- Track: finding-generation
- Depends on: VFS-029, VFS-030
- Goal id: VFS-G101
- Outputs: ipfs_accelerate_py/agent_supervisor/finding_task_source.py, test/api/test_agent_supervisor_finding_task_source.py
- Validation: python -m pytest test/api/test_agent_supervisor_finding_task_source.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/findings
- Parallel lane: finding-task-source
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/finding_task_source.py, test/api/test_agent_supervisor_finding_task_source.py
- Conflict policy: Reuse canonical task identity, TaskSource, task quality, proposal router, and board locks.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: Convert only fresh admitted findings to goal-backed tasks with one root-cause family, exact output files/symbols/effects, dependencies, conflict domain, validation/proof plan, finding/provenance CIDs, risk, resource class, and context ceiling. Ambiguous/broad/out-of-root findings produce non-executable review records. Stable findings replay to no-op; changed evidence supersedes rather than duplicates; related tiny tasks coalesce only with shared validation and merge fate. Support bounded Markdown, DuckDB, JSON, and SARIF-linked projections without authority drift.

## VFS-032 Compile compact CID-addressed repair and delta-retry packets

- Status: todo
- Completion: manual
- Priority: P0
- Track: low-context-repair
- Depends on: VFS-013, VFS-021, VFS-031
- Goal id: VFS-G110
- Outputs: ipfs_accelerate_py/agent_supervisor/contract_repair_packet.py, test/api/test_agent_supervisor_contract_repair_packet.py
- Validation: python -m pytest test/api/test_agent_supervisor_contract_repair_packet.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/repair
- Parallel lane: repair-context
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/contract_repair_packet.py, test/api/test_agent_supervisor_contract_repair_packet.py
- Conflict policy: Compose ContextCompiler and FormalReplanner; do not alter provider invocation in this task.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: Compile a canonical packet containing task/finding/forest/tree/policy IDs, expected/observed contract refs, minimal call/counterexample slice, exact edit scope/effects, acceptance, validation/proof commands, risks, and content-addressed expansion handles. Default packet is <=16 KiB plus bounded source spans and never embeds full source/AST/graph/proof/witness. Required fields survive provider budgets. Delta retry binds prior decision and includes only changed/requested evidence. Test omission attacks, stale handles, secret/redaction policy, reconstruction, lower token/byte cost, and model proposal authority.

## VFS-033 Harden and validate the two-provider Grok Build/Codex supervisor control

- Status: todo
- Completion: manual
- Priority: P1
- Track: low-context-repair
- Depends on: VFS-032
- Goal id: VFS-G111
- Outputs: scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance_control.sh, test/api/test_vfs_symbolic_assurance_control.py
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_control.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/repair
- Parallel lane: provider-control
- Resource class: cpu-small
- Predicted files: scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance_control.sh, test/api/test_vfs_symbolic_assurance_control.py
- Conflict policy: Preserve operator start/status/stop semantics and protected paths; tests must not launch real providers.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: Validate idempotent start/status/stop, PID ownership, stale PID recovery, authenticated provider probes, deterministic shard count/index, isolated state/worktrees, shared merge queue, protected plan/objective/taskboard, submodule configuration, exact repository root, bounded timeouts/retries, and one refill owner. Grok and Codex shards must not duplicate tasks; provider loss degrades without expanding authority; tests use fake processes and temporary repos. Do not kill unrelated processes or store secrets in argv/logs.

## VFS-034 Refill goals and tasks from fresh symbolic finding families

- Status: todo
- Completion: manual
- Priority: P0
- Track: autonomous-refill
- Depends on: VFS-031
- Goal id: VFS-G120
- Outputs: ipfs_accelerate_py/agent_supervisor/symbolic_finding_refill.py, test/api/test_agent_supervisor_symbolic_finding_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_symbolic_finding_refill.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/refill
- Parallel lane: symbolic-refill
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/symbolic_finding_refill.py, test/api/test_agent_supervisor_symbolic_finding_refill.py
- Conflict policy: Compose AdaptiveGoalRefiner and BacklogRefinery under existing board/objective locks; do not create an independent daemon.
- Symbolic first: true
- LLM context budget bytes: 12288
- Acceptance: On low/drained backlog, ingest fresh ledger receipts, map them to an existing exact goal/root-cause family or propose bounded child goals, and materialize bounded tasks with valid ancestry. Enforce max three children, depth four, open-work ceiling, cooldown, stable semantic keys, dependency DAG, precise output scope, and policy/forest binding. Replay and repeated unchanged diagnostics are no-ops/backoff; stale/ambiguous/rejected evidence is retained but creates no work; exhausted retries become one bounded unblock/review task; conclusive healthy exhaustion creates no busywork.

## VFS-035 Benchmark symbolic-first coverage, reuse, context, and resources

- Status: todo
- Completion: manual
- Priority: P1
- Track: autonomous-refill
- Depends on: VFS-011, VFS-016, VFS-020, VFS-032, VFS-034
- Goal id: VFS-G121
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_symbolic_benchmark.py, test/api/test_agent_supervisor_vfs_symbolic_benchmark.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_symbolic_benchmark.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/refill
- Parallel lane: assurance-benchmark
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/vfs_symbolic_benchmark.py, test/api/test_agent_supervisor_vfs_symbolic_benchmark.py
- Conflict policy: Benchmark immutable fixtures and isolated caches; no adaptive promotion or provider network calls.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Measure cold/warm/exact/delta scans, inventory completeness, AST/graph/contract/proof cache reuse, invalidation precision, seeded true/false/unknown findings, time to counterexample, artifact bytes, CPU/RSS/process/disk, idle writes/CPU, task yield/deduplication, and provider packet bytes/tokens. Deterministic stages must make zero LLM calls. Paired packets must reduce median provider input by >=80 percent versus a bounded repository-context baseline without lowering required evidence or seeded finding coverage. Record all fixture/profile identities and avoid promotion claims from insufficient samples.

## VFS-036 Add adversarial end-to-end assurance, control parity, recovery, and rollback gates

- Status: todo
- Completion: manual
- Priority: P0
- Track: assurance-rollout
- Depends on: VFS-018, VFS-024, VFS-028, VFS-030, VFS-034, VFS-035
- Goal id: VFS-G130
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_symbolic_rollout.py, test/api/test_vfs_symbolic_assurance_e2e.py
- Validation: python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py -q
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/rollout
- Parallel lane: assurance-rollout
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/vfs_symbolic_rollout.py, test/api/test_vfs_symbolic_assurance_e2e.py
- Conflict policy: Integrate only after producer tasks land; preserve shadow default and existing SupervisorControlService authority.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Freeze a multi-repository fixture and test reproducible CIDs, complete inventory/exclusions, incremental reuse, stale/corrupt cache rejection, contract precision, wrong/unknown proof, simulated/forged/tampered ZK, MCP mock/bypass, VFS seeded drift, vulnerability false positives, task determinism, provider loss, restart/replay, lease/fence loss, merge conflict, bounded refill/exhaustion, and rollback. Publish equivalent bounded status/findings/receipts through Python, CLI, and MCP without provider imports or process starts during discovery. Automatic mutation remains disabled; any regression returns effective rollout to shadow.

## VFS-037 Run and verify the frozen SwissKnife/IPFS VFS pilot and emit the repair board

- Status: todo
- Completion: manual
- Priority: P0
- Track: assurance-rollout
- Depends on: VFS-036
- Goal id: VFS-G131
- Outputs: ipfs_accelerate_py/agent_supervisor/vfs_symbolic_pilot.py, docs/architecture/ipfs_kit_vfs_symbolic_assurance.findings.todo.md, test/api/test_agent_supervisor_vfs_symbolic_pilot.py
- Validation: python -m pytest test/api/test_agent_supervisor_vfs_symbolic_pilot.py -q; python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot --verify
- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1
- Bundle: vfs-assurance/rollout
- Parallel lane: frozen-pilot
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/vfs_symbolic_pilot.py, docs/architecture/ipfs_kit_vfs_symbolic_assurance.findings.todo.md, test/api/test_agent_supervisor_vfs_symbolic_pilot.py
- Conflict policy: The pilot reads the frozen external SwissKnife checkout and configured submodules; it may write only bounded accelerator artifacts and the generated findings board.
- Symbolic first: true
- LLM context budget bytes: 16384
- Acceptance: Provide dry-run and verify modes that freeze fresh repository descriptors, scan every admitted SwissKnife file plus the VFS-relevant IPFS closure, execute the admitted deterministic graph/contract/proof/ZK-shadow pipeline, and publish manifest/coverage/cache/proof/finding/taskboard CIDs. Every file and finding must be reproducible and provenance-bound; inconclusive items remain non-executable; the generated board is bounded, deduplicated, goal-backed, dependency-valid, and contains exact repair packets. Verification performs no provider call or source mutation and fails on changed trees, incomplete inventory, stale evidence, or noncanonical artifacts.
