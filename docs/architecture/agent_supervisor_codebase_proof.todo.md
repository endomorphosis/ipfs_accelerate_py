# Agent Supervisor Codebase-Proof Taskboard (CBP)

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix `## CBP-`.
Companion objectives: `agent_supervisor_codebase_proof.objectives.md`.
Human plan: `AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md`.
Scheduler config: `config/agent_supervisor_codebase_proof_scheduler.json`.

## Objective

Turn the agent supervisor into a **proof-carrying control plane** for codebase
change: reviewed properties → obligations → prove/refute with **trust-aware
proof caching** → queryable evidence → obligation-first agent context →
CodeEditPacket → implement → cache-aware re-prove.

Normative:

- `TrustAwareProofCache` / `formal_verification_cache` is the sole proof-receipt
  memoization trust boundary (hits re-derive assurance; never upgrade trust).
- Repository queries, GraphRAG projections, tests, runtime observations, static
  analysis, solver candidates, kernel-checked proofs, and cryptographic
  attestations are distinct evidence classes and must not be promoted across
  trust boundaries implicitly.
- Candidates and simulated ZK cannot grant completion / `ATTESTED` authority.
- Real ZK is considered only for an approved privacy/trust-boundary use case;
  it is not a code-correctness discovery mechanism.
- Domain semantic metrics (e.g. SRT e2e) remain separate from proof pass.

## Parallel lanes

| Lane | Owns |
| --- | --- |
| `cbp-docs` | Plan seal / doctrine inventory |
| `cbp-proof-cache` | Proof cache productization |
| `cbp-catalog` | Property catalog |
| `cbp-evidence` | Claim/evidence semantics and lifecycle |
| `cbp-obligations` | Obligation compiler |
| `cbp-queries` | Query API |
| `cbp-reproof` | Cache-aware re-proof |
| `cbp-context` | Obligation-first capsules |
| `cbp-context-delta` | proof_delta retries |
| `cbp-materialize` | CodeEditPacket materializer |
| `cbp-formal-plan` | require_proof preconditions |
| `cbp-bundles` | Bundle locality |
| `cbp-srt` | SRT bridge |
| `cbp-self` | Supervisor self-properties |
| `cbp-metrics` | Efficiency gates |
| `cbp-zk` | Attestation / ZK policy |

---

## CBP-000 Seal codebase-proof plan artifacts

- Status: completed
- Completion: auto
- Priority: P0
- Track: docs
- Depends on:
- Goal id: CBP-G000
- Outputs: docs/architecture/agent_supervisor_codebase_proof.objectives.md, docs/architecture/agent_supervisor_codebase_proof.todo.md, docs/architecture/AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md, config/agent_supervisor_codebase_proof_scheduler.json
- Validation: test -f docs/architecture/agent_supervisor_codebase_proof.todo.md && test -f docs/architecture/agent_supervisor_codebase_proof.objectives.md && test -f config/agent_supervisor_codebase_proof_scheduler.json
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/docs
- Parallel lane: cbp-docs
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 1800
- Predicted files: docs/architecture/agent_supervisor_codebase_proof.objectives.md, docs/architecture/agent_supervisor_codebase_proof.todo.md, docs/architecture/AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md, config/agent_supervisor_codebase_proof_scheduler.json
- Interfaces: CodebaseProofPlan@1
- Conflict policy: Own CBP planning artifacts only; do not edit ASI or PLAT sealed promotion reports.
- Preconditions:
- Effects: Supervisor can prepare/launch CBP board with proof_cache config flags.
- Evidence subset: cbp plan seal
- Acceptance: Objectives, taskboard, human plan, and scheduler config exist; scheduler JSON sets proof_cache.enabled true and prefer_cache_before_provider true; task prefix is `## CBP-`.

## CBP-010 Doctrine inventory and fail-closed policy tests

- Status: completed
- Completion: auto
- Priority: P0
- Track: docs
- Depends on: CBP-000
- Goal id: CBP-G010
- Outputs: test/api/test_agent_supervisor_code_proof_doctrine.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_doctrine.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/docs
- Parallel lane: cbp-docs
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 3600
- Predicted files: test/api/test_agent_supervisor_code_proof_doctrine.py
- Interfaces: AssuranceLevel, ProofReceipt
- Conflict policy: Own doctrine tests only; do not edit protected CBP plan/objectives/taskboard files; reference formal_verification_contracts without weakening ASI-G102.
- Preconditions: CBP-000 sealed.
- Effects: Automated guards for candidate non-authority and sim-ZK ≠ ATTESTED.
- Evidence subset: cbp doctrine inventory
- Acceptance: Tests assert (1) candidate assurance cannot satisfy kernel-required policy, (2) private_witness markers rejected from public receipt JSON, (3) simulated ZK/attestation path cannot produce AssuranceLevel.ATTESTED, (4) sealed plan file on disk documents formal_verification_cache as sole memoization trust boundary (read-only check; do not modify the plan).

## CBP-015 Productize trust-aware proof cache as default prove path

- Status: completed
- Completion: auto
- Priority: P0
- Track: proof-cache
- Depends on: CBP-000
- Goal id: CBP-G015
- Outputs: ipfs_accelerate_py/agent_supervisor/formal_verification_cache.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_formal_verification_cache.py, test/api/test_agent_supervisor_code_proof_cache_integration.py
- Validation: python -m pytest test/api/test_agent_supervisor_formal_verification_cache.py test/api/test_agent_supervisor_code_proof_cache_integration.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/proof-cache
- Parallel lane: cbp-proof-cache
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/formal_verification_cache.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_formal_verification_cache.py, test/api/test_agent_supervisor_code_proof_cache_integration.py
- Interfaces: TrustAwareProofCache, ProofCacheKey, ProofCacheEntry, CodeProofObligation
- Conflict policy: Extend existing cache and obligation modules; do not add a second trust root; preserve re-derive-on-hit semantics.
- Preconditions: formal_verification_cache module exists.
- Effects: Prove/re-prove callers prefer cache before provider; single-flight shared across workers; outcomes metrics available.
- Evidence subset: cbp proof cache productization
- Acceptance: Integration tests cover put/get with build_proof_cache_key bindings (obligation, tree, premises, toolchain, policy, required assurance); hit re-derives assurance; single-flight collapses concurrent identical keys; rejections for stale_tree, poisoned_entry, private_material, toolchain_drift; candidate-only receipts never admitted as authoritative hits; metrics expose hit/miss/reject counts.

## CBP-020 Reviewed property catalog

- Status: completed
- Completion: auto
- Priority: P0
- Track: property-catalog
- Depends on: CBP-000
- Goal id: CBP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/code_property_catalog.py, test/api/test_agent_supervisor_code_property_catalog.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_property_catalog.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/catalog
- Parallel lane: cbp-catalog
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_property_catalog.py, test/api/test_agent_supervisor_code_property_catalog.py
- Interfaces: CodePropertyCatalog@1, ReviewedCodeShape, ClaimFamily
- Conflict policy: Own new catalog module; register only reviewed template ids; no natural-language template invent.
- Preconditions: proof_obligation_templates registry exists.
- Effects: Stable property ids for queries, plans, and packets.
- Evidence subset: cbp property catalog
- Acceptance: Content-addressed catalog; every entry declares a reviewed claim family, specification/template version, required assurance, owner/reviewer metadata, assumption policy, and invalidation policy; seed families cover dependency/reachability facts, API contracts, behavioral invariants, security properties, semantic equivalence, supervisor lifecycle properties, and SRT structural constraints (non_vacuous_candidate, rule_cardinality_preserved, untriggered_projection_preserved); unknown ids and unreviewed natural-language templates fail closed; semantic_authority defaults false.

## CBP-025 Typed claim/evidence semantics and lifecycle

- Status: completed
- Completion: auto
- Priority: P0
- Track: evidence-contract
- Depends on: CBP-000
- Goal id: CBP-G025
- Outputs: ipfs_accelerate_py/agent_supervisor/code_claim_contracts.py, docs/architecture/agent_supervisor_code_claim_evidence_contract.md, test/api/test_agent_supervisor_code_claim_evidence_contract.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_claim_evidence_contract.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/evidence
- Parallel lane: cbp-evidence
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_claim_contracts.py, docs/architecture/agent_supervisor_code_claim_evidence_contract.md, test/api/test_agent_supervisor_code_claim_evidence_contract.py
- Interfaces: CodeClaimRecord@1, ClaimFamily, ClaimStatus, ProofEvidence, ImplementationResultEvidence, ProofReceipt
- Conflict policy: Add a normalization/lifecycle layer over existing formal_verification_contracts and code_proof_obligations types; do not duplicate assurance derivation, proof receipts, or the proof-cache trust root.
- Preconditions: Existing ProofEvidence, ImplementationResultEvidence, CodeProofObligation, and ProofReceipt contracts are importable.
- Effects: Every query and edit decision can state exactly what is claimed, what kind of evidence supports it, which assumptions and source revision it binds, and what invalidates it.
- Evidence subset: cbp claim evidence contract
- Acceptance: Content-addressed CodeClaimRecord binds property/obligation ids, claim family, repository/tree/scope ids, premise and assumption ids, producer/toolchain/policy/catalog versions, evidence ids, required assurance, and invalidation selectors; lifecycle distinguishes unknown, open, satisfied, refuted, unsupported, not_measured, and stale without treating cache miss as refutation; evidence tiers distinguish query facts, bounded test/runtime/static-analysis observations, solver/model-check candidates, kernel-checked proofs, and cryptographic attestations; query/GraphRAG facts and observations cannot independently mint kernel assurance; arbitrary natural-language claims fail closed; canonical round-trip and stale-evidence tests pass.

## CBP-030 Obligation compiler with cache-key binding

- Status: completed
- Completion: auto
- Priority: P0
- Track: obligations
- Depends on: CBP-000, CBP-020, CBP-025
- Goal id: CBP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_code_proof_scopes.py, test/api/test_agent_supervisor_code_proof_obligation_cache_keys.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_scopes.py test/api/test_agent_supervisor_code_proof_obligation_cache_keys.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/obligations
- Parallel lane: cbp-obligations
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_code_proof_scopes.py, test/api/test_agent_supervisor_code_proof_obligation_cache_keys.py
- Interfaces: CodeProofObligation, ProofCacheKey, CodePropertyCatalog@1, CodeClaimRecord@1
- Conflict policy: Extend code_proof_obligations; preserve ASI-G102 proof-candidate non-authority; do not put gold IR bodies into premises.
- Preconditions: CBP-020 catalog and CBP-025 claim/evidence contract available; build_proof_cache_key available.
- Effects: Every compiled obligation has a deterministic cache key identity for G015/G050.
- Evidence subset: cbp obligation compiler
- Acceptance: Compiles from tree + changed AST scope + optional formal-plan effects + optional residual refs; emits typed claim records with explicit premise/assumption ids and invalidation selectors; unsupported and not-measured shapes remain distinguishable; cache key identity binds property/catalog version, tree/scope, premise/assumption digests, toolchain, policy, and required assurance; repository-wide source dumps rejected as premises; unit tests cover dependency, API-contract, security, semantic-equivalence, and residual-ref cases without embedding secrets or gold bodies.

## CBP-040 Query API (open / satisfied / refuted / impact / proof_delta)

- Status: todo
- Completion: auto
- Priority: P0
- Track: queries
- Depends on: CBP-015, CBP-020, CBP-025, CBP-030
- Goal id: CBP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_query.py, ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, test/api/test_agent_supervisor_code_proof_query.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_query.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/queries
- Parallel lane: cbp-queries
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_proof_query.py, ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, test/api/test_agent_supervisor_code_proof_query.py
- Interfaces: CodeProofQuery@1, CodeClaimRecord@1, CodeEvidenceGraph, TrustAwareProofCache
- Conflict policy: Own query module; graph remains non-authoritative projection; GraphRAG cannot mint proof nodes.
- Preconditions: Cache productization, typed claim/evidence contract, and obligation compiler available.
- Effects: Agents and planners can ask what is already true/false/unknown without loading full source.
- Evidence subset: cbp query api
- Acceptance: Implements properties_satisfied, properties_open, properties_refuted, properties_unsupported, properties_not_measured, properties_stale, counterexamples, impact, and proof_delta; every result carries claim/evidence ids and provenance handles; cache miss is not refuted; open means a supported claim has no current valid evidence at required assurance; unsupported, not_measured, unknown, and stale remain distinct; proof_delta lists only invalidated obligations between parent and child trees with machine-readable reasons; results bounded and content-addressed.

## CBP-050 Cache-aware re-proof and invalidation

- Status: todo
- Completion: auto
- Priority: P0
- Track: reproof
- Depends on: CBP-015, CBP-030, CBP-040
- Goal id: CBP-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_verification_cache.py, test/api/test_agent_supervisor_code_proof_reproof.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_reproof.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/reproof
- Parallel lane: cbp-reproof
- Resource class: cpu-proof-solver
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_verification_cache.py, test/api/test_agent_supervisor_code_proof_reproof.py
- Interfaces: TrustAwareProofCache, CodeProofObligation, ProofReceipt
- Conflict policy: Coordinate with CBP-015 on cache APIs; prefer lookup-before-provider; serialize hot-path edits if both touch same functions.
- Preconditions: Query API and cache integration tests exist.
- Effects: Merge/completion re-proof is cheap on unchanged obligations and fail-closed on drift.
- Evidence subset: cbp cache-aware reproof
- Acceptance: Warm path serves unchanged obligations from cache after re-derivation; changed tree/blob/AST/path, dependency edge, premise/assumption digest, property catalog, toolchain, policy, or required assurance forces stale/open status and re-solve; every invalidation exposes a reason code and provenance edge; single-flight under parallel workers; wrong-tree binding never accepts foreign hit; provider not called on authoritative hit.

## CBP-060 Obligation-first context capsules

- Status: todo
- Completion: auto
- Priority: P0
- Track: context
- Depends on: CBP-040
- Goal id: CBP-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/decision_context.py, test/api/test_agent_supervisor_code_proof_context.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_context.py test/api/test_agent_supervisor_context_compiler.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/context
- Parallel lane: cbp-context
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/decision_context.py, test/api/test_agent_supervisor_code_proof_context.py
- Interfaces: ContextCapsule, DecisionContext, CodeProofQuery@1
- Conflict policy: Extend context profiles; preserve required-core non-truncation (ASI-G091); do not store raw prompts in receipts.
- Preconditions: Query API returns open/satisfied/refuted sets.
- Effects: Implementation agents receive a task-specific dependency/AST slice, open obligations, counterexamples, relevant specification handles, and failure traces instead of bulk source by default.
- Evidence subset: cbp obligation context
- Acceptance: Invariant core includes task/acceptance ids, open obligations, assumption ids, counterexamples, changed dependency/AST slice, relevant specification/contract handles, and bounded failure traces; satisfied proofs appear as receipt digests/handles only; optional source/evidence is value-of-information ranked with content-addressed expansion handles; untrusted repository text is labeled as data and cannot inject supervisor instructions; solver traces and unrelated source are excluded by default; required claim and acceptance coverage cannot be deferred as optional; capsule records token budget and omitted-handle manifest so coverage is auditable.

## CBP-070 Delta retry via proof_delta and cache hits

- Status: todo
- Completion: auto
- Priority: P1
- Track: context-delta
- Depends on: CBP-015, CBP-060
- Goal id: CBP-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_code_proof_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_context_delta.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/context-delta
- Parallel lane: cbp-context-delta
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_code_proof_context_delta.py
- Interfaces: ContextDeltaCapsule, CodeProofQuery@1, TrustAwareProofCache
- Conflict policy: Serialize with CBP-060 if both edit context_compiler hot paths; prefer sequential after 060.
- Preconditions: Obligation-first capsule profile exists.
- Effects: Retries ship only deltas; still-valid obligations reuse cache without re-prompting bodies.
- Evidence subset: cbp context delta
- Acceptance: Parent-bound reconstruct preserves core; proof_delta-only retries on fixtures; still-valid cached obligations not re-opened without impact reason; token count on retry path lower than cold path in fixture assertions.

## CBP-080 CodeEditPacket and supervisor materializer

- Status: todo
- Completion: auto
- Priority: P0
- Track: materialize
- Depends on: CBP-030, CBP-040
- Goal id: CBP-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/code_edit_packet.py, ipfs_accelerate_py/agent_supervisor/code_edit_materialize.py, test/api/test_agent_supervisor_code_edit_packet.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_edit_packet.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/materialize
- Parallel lane: cbp-materialize
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_edit_packet.py, ipfs_accelerate_py/agent_supervisor/code_edit_materialize.py, test/api/test_agent_supervisor_code_edit_packet.py
- Interfaces: CodeEditPacket@1, CodeClaimRecord@1, CodeProofQuery@1, TrustAwareProofCache
- Conflict policy: Own new packet modules under agent_supervisor; optional SRT bridge is additive only.
- Preconditions: Obligations and queries available.
- Effects: Open obligations become implementable supervisor tasks with validation_commands including cache-aware re-prove.
- Evidence subset: cbp code edit packet
- Acceptance: Packet is content-addressed and binds source tree, claim/obligation ids, assumptions, invalidation reasons, predicted files, and acceptance ids; implementable=false on reject/timeout/unsupported/not_measured/stale-required-input; prover fields carry semantic_authority=false; materializer emits validation_commands for tests, domain metrics, and cache-aware re-proof at declared assurance; cache and claim status recorded without embedding full proof bodies; round-trip serialize tests pass.

## CBP-090 Formal-plan require_proof preconditions

- Status: todo
- Completion: auto
- Priority: P1
- Track: formal-plan
- Depends on: CBP-020, CBP-040
- Goal id: CBP-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_formal_plan_proof_preconditions.py
- Validation: python -m pytest test/api/test_agent_supervisor_formal_plan_proof_preconditions.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/formal-plan
- Parallel lane: cbp-formal-plan
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_formal_plan_proof_preconditions.py
- Interfaces: FormalWorkPlan, CodePropertyCatalog@1, TrustAwareProofCache
- Conflict policy: Extend formal plan modules without LLM formula invent; keep unsupported semantics explicit.
- Preconditions: Property catalog and query API exist.
- Effects: Plans can gate work on cached kernel/solver receipts at declared assurance.
- Evidence subset: cbp formal plan require_proof
- Acceptance: requires_proof(property_id, assurance) preconditions compile; missing receipt fails admission; cache hit with re-derived assurance admits; candidate-only does not admit.

## CBP-100 Bundle optimizer locality for obligations and cache

- Status: todo
- Completion: auto
- Priority: P2
- Track: bundles
- Depends on: CBP-040, CBP-080
- Goal id: CBP-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, test/api/test_agent_supervisor_code_proof_bundles.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_bundles.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/bundles
- Parallel lane: cbp-bundles
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, test/api/test_agent_supervisor_code_proof_bundles.py
- Interfaces: BundleOptimizer, CodeProofQuery@1
- Conflict policy: Extend bundle optimizer; do not break existing ASI bundle tests.
- Preconditions: Query + materializer available.
- Effects: Parallel lanes co-locate shared obligations/cache namespaces; conflicts still serialize.
- Evidence subset: cbp bundle locality
- Acceptance: Independent obligation sets remain parallel; conflicting predicted files serialize; optimizer prefers shared proof-cache key prefixes without accepting wrong-tree hits.

## CBP-110 Semantic-roundtrip residual/structural bridge

- Status: todo
- Completion: auto
- Priority: P1
- Track: srt-vertical
- Depends on: CBP-020, CBP-040, CBP-080
- Goal id: CBP-G110
- Outputs: docs/architecture/agent_supervisor_codebase_proof_srt_bridge.md, ipfs_accelerate_py/agent_supervisor/code_proof_srt_bridge.py, test/api/test_agent_supervisor_code_proof_srt_bridge.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_srt_bridge.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/srt
- Parallel lane: cbp-srt
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: docs/architecture/agent_supervisor_codebase_proof_srt_bridge.md, ipfs_accelerate_py/agent_supervisor/code_proof_srt_bridge.py, test/api/test_agent_supervisor_code_proof_srt_bridge.py
- Interfaces: CodeEditPacket@1, StructuralAdmission@1, PlateauCodexPacket@1
- Conflict policy: Bridge only; do not rewrite sealed PLAT promotion snapshots or change production SRT arm defaults.
- Preconditions: Catalog, query, and packet materializer exist; SRT interfaces available via datasets or thin adapter.
- Effects: PLAT/PLAT2 residual catalogs, structural admissions, and supervisor packets project into CBP claims/queries/cache/context without moving experiment selection or semantic promotion authority into CBP.
- Evidence subset: cbp srt bridge
- Acceptance: Bridge documentation maps each heterogeneous method to its measured role rather than treating methods as interchangeable—autoencoder and spaCy as bounded guidance/diagnostics, SyMAI as orchestration, Leanstral as proposal teacher, Hammer/cvc5/Lean as declared structural gates, and the deterministic compiler/IR/decompiler as the edit target; PLAT residual catalog and PlateauCodexPacket@1 ids project into typed claims, counterexamples, context capsules, and CodeEditPacket@1; PLAT2 holdout artifacts remain separately preregistered and queryable; StructuralAdmission receipts project to graph/query with explicit non-semantic authority; cache keys exclude gold IR bodies; semantic round-trip e2e loss and holdout promotion gates remain authoritative; unit tests use fixtures, not live gold dumps or sealed-report rewrites.

## CBP-120 Supervisor self-properties (lease, merge, DAG, freshness)

- Status: todo
- Completion: auto
- Priority: P1
- Track: self-properties
- Depends on: CBP-020, CBP-050
- Goal id: CBP-G120
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_code_proof_self_properties.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_self_properties.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/self
- Parallel lane: cbp-self
- Resource class: cpu-proof-solver
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_code_proof_self_properties.py
- Interfaces: ReviewedCodeShape, TrustAwareProofCache, CodeProofObligation
- Conflict policy: Own self-property wiring tests; coordinate template registry only if new shapes required.
- Preconditions: Cache-aware re-proof works; catalog seeds include self shapes.
- Effects: Critical supervisor invariants prove on change with warm cache hits.
- Evidence subset: cbp self properties
- Acceptance: At least lease fencing, merge idempotence, DAG acyclicity, and evidence freshness shapes exercise prove→cache→reproof; mutations invalidate; warm path hits cache.

## CBP-130 Closed-loop quality, coverage, token, and proof-cost gates

- Status: todo
- Completion: auto
- Priority: P1
- Track: metrics
- Depends on: CBP-015, CBP-050, CBP-060, CBP-070, CBP-080
- Goal id: CBP-G130
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_code_proof_benchmark.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_code_proof_efficiency.py, docs/benchmarks/agent_supervisor_codebase_proof_evaluation.md
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_efficiency.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/metrics
- Parallel lane: cbp-metrics
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_code_proof_benchmark.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_code_proof_efficiency.py, docs/benchmarks/agent_supervisor_codebase_proof_evaluation.md
- Interfaces: CodebaseProofBenchmark@1, SupervisorEfficiencyMetrics, CodeEditPacket@1, TrustAwareProofCache, ContextCapsule
- Conflict policy: Extend efficiency metrics; store digests not prompt bodies.
- Preconditions: Cache-aware re-proof, obligation-first/delta context, and CodeEditPacket materialization available.
- Effects: Reproducible evidence shows whether claims are reliable and the proof-guided edit loop improves repair decisions while reducing context and prove cost.
- Evidence subset: cbp efficiency gates
- Acceptance: Preregister a fixed baseline and held-out mutation/repair suite spanning dependency, API-contract, behavioral, security, semantic-equivalence, and supervisor-lifecycle claims before outcome inspection; compare bulk-source and obligation-first paths on identical tasks; report claim coverage by family/evidence tier/required assurance, satisfied/refuted/open/unsupported/not_measured/stale counts, false-admit and false-refute rates on seeded mutations, stale-evidence detection, first-pass and eventual repair success, accepted-patch regression rate, input tokens per accepted criterion, retry tokens, provider calls, cache hit/reject rate, wall time, and proof cost; require zero false authoritative admissions in the fixture suite, no required-coverage loss, ≥40% fewer input tokens per accepted criterion, ≥60% retry token reduction, and warm prove cost improvement when cache hits dominate; live-model results, when present, are reported separately from deterministic fixture gates.

## CBP-200 Attestation and real ZK policy (deferred)

- Status: todo
- Completion: auto
- Priority: P2
- Track: attestation-zk
- Depends on: CBP-015, CBP-050
- Goal id: CBP-G200
- Outputs: ipfs_accelerate_py/agent_supervisor/proof_attestation.py, test/api/test_agent_supervisor_code_proof_attestation_policy.py, docs/architecture/agent_supervisor_codebase_proof_zk_threat_model.md, docs/architecture/agent_supervisor_codebase_proof_zk_policy.md
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_attestation_policy.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/zk
- Parallel lane: cbp-zk
- Resource class: cpu-proof-solver
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof_attestation.py, test/api/test_agent_supervisor_code_proof_attestation_policy.py, docs/architecture/agent_supervisor_codebase_proof_zk_threat_model.md, docs/architecture/agent_supervisor_codebase_proof_zk_policy.md
- Interfaces: AssuranceLevel.ATTESTED, PersistedAttestationRecord, TrustAwareProofCache
- Conflict policy: Threat model and fail-closed policy before backend selection; do not add production ZK merely to discover code correctness; simulated backends must hard-fail ATTESTED.
- Preconditions: Cache-aware re-proof and doctrine tests green.
- Effects: An approved private-witness/cross-trust-boundary case may gain third-party verification without exposing the witness; otherwise ZK remains explicitly not applicable.
- Evidence subset: cbp attestation zk policy
- Acceptance: Threat model names prover, verifier, protected witness, disclosure risk, trust boundary, replay/freshness requirements, and why ordinary signed/kernel receipts are insufficient; a reviewed use-case decision record is mandatory before selecting or implementing a Groth16/Plonk/other backend, and no qualifying use case yields a terminal not_applicable result without blocking core CBP; simulated ZKP/attestation cannot satisfy ATTESTED; public inputs bind property, repository/tree, obligation, toolchain, policy, and kernel-receipt digests; private witnesses rejected from public receipts and attestation cache entries; real attestations re-verify and fail closed on drift.

---

## Non-goals (do not schedule)

- Proving arbitrary natural-language claims about the repo
- LLM inventing proof templates or theorems
- Replacing pytest / domain e2e with SMT scores alone
- Treating repository queries, GraphRAG, tests, or static-analysis observations as kernel proofs
- Treating prompt wording alone as context reduction without dependency/evidence coverage
- Always-on full-repository formal verification
- A second proof-cache trust root that skips re-derivation on hit
- Treating simulated ZKP as production cryptography
- Building a real ZK backend without an approved private-witness threat model
- Rewriting sealed PLAT 2026-07-27 promotion reports
