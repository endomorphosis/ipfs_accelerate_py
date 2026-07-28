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
- Candidates and simulated ZK cannot grant completion / `ATTESTED` authority.
- Domain semantic metrics (e.g. SRT e2e) remain separate from proof pass.

## Parallel lanes

| Lane | Owns |
| --- | --- |
| `cbp-docs` | Plan seal / doctrine inventory |
| `cbp-proof-cache` | Proof cache productization |
| `cbp-catalog` | Property catalog |
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
- Interfaces: CodePropertyCatalog@1, ReviewedCodeShape
- Conflict policy: Own new catalog module; register only reviewed template ids; no natural-language template invent.
- Preconditions: proof_obligation_templates registry exists.
- Effects: Stable property ids for queries, plans, and packets.
- Evidence subset: cbp property catalog
- Acceptance: Content-addressed catalog; seed properties for existing ReviewedCodeShape values plus tags for SRT structural constraints (non_vacuous_candidate, rule_cardinality_preserved, untriggered_projection_preserved); unknown ids fail closed; semantic_authority defaults false.

## CBP-030 Obligation compiler with cache-key binding

- Status: todo
- Completion: auto
- Priority: P0
- Track: obligations
- Depends on: CBP-000, CBP-020
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
- Interfaces: CodeProofObligation, ProofCacheKey, CodePropertyCatalog@1
- Conflict policy: Extend code_proof_obligations; preserve ASI-G102 proof-candidate non-authority; do not put gold IR bodies into premises.
- Preconditions: CBP-020 catalog available; build_proof_cache_key available.
- Effects: Every compiled obligation has a deterministic cache key identity for G015/G050.
- Evidence subset: cbp obligation compiler
- Acceptance: Compiles from tree + changed AST scope + optional formal-plan effects + optional residual refs; unsupported shapes stay unsupported; cache key identity stable; repository-wide source dumps rejected as premises; unit tests cover residual-ref hook without embedding secrets.

## CBP-040 Query API (open / satisfied / refuted / impact / proof_delta)

- Status: todo
- Completion: auto
- Priority: P0
- Track: queries
- Depends on: CBP-015, CBP-020, CBP-030
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
- Interfaces: CodeProofQuery@1, CodeEvidenceGraph, TrustAwareProofCache
- Conflict policy: Own query module; graph remains non-authoritative projection; GraphRAG cannot mint proof nodes.
- Preconditions: Cache productization and obligation compiler available.
- Effects: Agents and planners can ask what is already true/false/unknown without loading full source.
- Evidence subset: cbp query api
- Acceptance: Implements properties_satisfied, properties_open, properties_refuted, counterexamples, impact, proof_delta; cache miss is not refuted; open means no valid hit at required assurance; proof_delta lists only invalidated obligations between parent and child trees; results bounded and content-addressed.

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
- Acceptance: Warm path serves unchanged obligations from cache after re-derivation; changed AST/path forces miss and re-solve; single-flight under parallel workers; wrong-tree binding never accepts foreign hit; provider not called on authoritative hit.

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
- Effects: Implementation agents receive obligations + counterexamples instead of bulk source by default.
- Evidence subset: cbp obligation context
- Acceptance: Invariant core includes open obligations and acceptance ids; satisfied proofs appear as receipt digests/handles only; optional evidence VoI-ranked with expansion handles; solver traces excluded by default; required coverage cannot be deferred as optional.

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
- Interfaces: CodeEditPacket@1, CodeProofQuery@1, TrustAwareProofCache
- Conflict policy: Own new packet modules under agent_supervisor; optional SRT bridge is additive only.
- Preconditions: Obligations and queries available.
- Effects: Open obligations become implementable supervisor tasks with validation_commands including cache-aware re-prove.
- Evidence subset: cbp code edit packet
- Acceptance: Packet is content-addressed; implementable=false on reject/timeout/unsupported; prover fields carry semantic_authority=false; materializer emits predicted_files + validation_commands; cache status (hit/miss/open) recorded without embedding full proof bodies; round-trip serialize tests pass.

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
- Effects: PLAT/PLAT2 loops can project structural admission + residuals into CBP queries/cache.
- Evidence subset: cbp srt bridge
- Acceptance: Structural constraint tags resolve in catalog; admission receipts project to graph/query; cache keys exclude gold IR bodies; e2e loss remains the promotion score in bridge docs; unit tests use fixtures not live gold dumps.

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

## CBP-130 Token and proof-cost efficiency gates

- Status: todo
- Completion: auto
- Priority: P1
- Track: metrics
- Depends on: CBP-015, CBP-060, CBP-070
- Goal id: CBP-G130
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_code_proof_efficiency.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_efficiency.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/metrics
- Parallel lane: cbp-metrics
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_code_proof_efficiency.py
- Interfaces: SupervisorEfficiencyMetrics, TrustAwareProofCache, ContextCapsule
- Conflict policy: Extend efficiency metrics; store digests not prompt bodies.
- Preconditions: Obligation-first context and proof cache productization available.
- Effects: Quantified proof that caching + capsules reduce cost without dropping coverage.
- Evidence subset: cbp efficiency gates
- Acceptance: Paired fixtures assert ≥40% fewer input tokens per accepted criterion for obligation-first vs bulk baseline, ≥60% retry token reduction, and warm prove cost improvement when cache hits dominate; required coverage preserved; cache hit rate reported.

## CBP-200 Attestation and real ZK policy (deferred)

- Status: todo
- Completion: auto
- Priority: P2
- Track: attestation-zk
- Depends on: CBP-015, CBP-050
- Goal id: CBP-G200
- Outputs: ipfs_accelerate_py/agent_supervisor/proof_attestation.py, test/api/test_agent_supervisor_code_proof_attestation_policy.py, docs/architecture/agent_supervisor_codebase_proof_zk_policy.md
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_attestation_policy.py -q
- Board namespace: agent-supervisor-codebase-proof-v1
- Bundle: agent-supervisor/codebase-proof/zk
- Parallel lane: cbp-zk
- Resource class: cpu-proof-solver
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof_attestation.py, test/api/test_agent_supervisor_code_proof_attestation_policy.py, docs/architecture/agent_supervisor_codebase_proof_zk_policy.md
- Interfaces: AssuranceLevel.ATTESTED, PersistedAttestationRecord, TrustAwareProofCache
- Conflict policy: Policy and fail-closed wiring only until a real crypto backend is selected; do not claim production ZK; simulated backends must hard-fail ATTESTED.
- Preconditions: Cache-aware re-proof and doctrine tests green.
- Effects: Optional third-party verify path without private witnesses in public cache.
- Evidence subset: cbp attestation zk policy
- Acceptance: Simulated ZKP/attestation cannot satisfy ATTESTED; public inputs bind property/tree/obligation digests only; private witnesses rejected from attestation cache entries; docs state upgrade path for real Groth16/Plonk backend.

---

## Non-goals (do not schedule)

- Proving arbitrary natural-language claims about the repo
- LLM inventing proof templates or theorems
- Replacing pytest / domain e2e with SMT scores alone
- Always-on full-repository formal verification
- A second proof-cache trust root that skips re-derivation on hit
- Treating simulated ZKP as production cryptography
- Rewriting sealed PLAT 2026-07-27 promotion reports
