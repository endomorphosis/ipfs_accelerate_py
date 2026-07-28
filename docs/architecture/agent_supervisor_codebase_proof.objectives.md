# Agent Supervisor Codebase-Proof Objective Heap (CBP)

This objective heap is machine-ingestible planning state for
`ipfs_accelerate_py.agent_supervisor` (objective daemon / bundle supervisor).
The companion taskboard
`agent_supervisor_codebase_proof.todo.md` is the executable projection
(task prefix `## CBP-`).

Human plan:
`AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md`.

## North star

**Make the agent supervisor a proof-carrying control plane for code change:**

1. State reviewed properties of the codebase as typed obligations (not free-form prose).
2. Discharge or refute them with fail-closed solvers / kernels / structural gates / tests.
3. **Memoize authoritative results in the trust-aware proof cache** (`formal_verification_cache.TrustAwareProofCache` / `ProofCache`) under exact semantic keys—hits re-derive assurance and never upgrade trust.
4. Index results so agents and humans can **query** satisfied / open / refuted / counterexamples / impact / proof_delta.
5. Drive edit decisions from unsatisfied obligations and counterexamples.
6. Compile implementation prompts as **obligation-first context capsules** (tiny core + expansion handles), shrinking tokens without dropping required coverage.
7. Optional **cryptographic attestation / ZK** only after property proofs + cache + queries are live; simulated ZK cannot emit `ATTESTED`.

First vertical consumer (parallel program, not a dependency of CBP root):
semantic-roundtrip plateau / holdout (`PlateauCodexPacket@1`, `StructuralAdmission@1`)
may bind residual obligations into the same catalog and cache.

## Existing foundations (reuse; do not fork trust models)

| Layer | Module / interface |
| --- | --- |
| Assurance lattice | `formal_verification_contracts` (`UNVERIFIED`…`ATTESTED`) |
| Provider isolation | `formal_verification_provider` |
| **Proof cache** | `formal_verification_cache` (`build_proof_cache_key`, single-flight, DuckDB) |
| Code obligations | `code_proof_obligations`, `proof_obligation_templates` |
| Evidence / queries | `code_evidence_graph`, `analysis_ast_index` |
| Context | `context_compiler`, `context_contracts`, `decision_context` |
| Formal plans | `formal_plan_compiler`, planning contracts |
| Domain gates | Hammer/cvc5/Lean via datasets logic; `StructuralAdmission@1` |
| ZK (later) | `ipfs_datasets_py.logic.zkp` (sim until real backend bound) |

## Composition doctrine (normative)

```text
Property catalog + obligation compiler
        │
        ▼
Prove / refute (provider) ──single-flight──► TrustAwareProofCache
        │                         ▲
        │                         │ exact key hit → re-derive assurance
        ▼                         │
ProofReceipt + graph projection ──┘
        │
        ├── Query API (what to change)
        ├── Obligation-first ContextCapsule (what to show the agent)
        └── CodeEditPacket → supervisor implement → re-prove (cache-aware)
```

Forbidden without explicit promotion:

- Candidate ATP/LLM/solver claims as completion or merge authority
- Simulated ZKP raising `AssuranceLevel.ATTESTED`
- Private witnesses / gold / secrets in public receipts or default agent context
- Proof pass substituting for domain semantic metrics (e.g. SRT e2e loss)
- Second independent proof-cache trust model that bypasses `formal_verification_cache`

## Goal tree

```text
CBP-G000  Proof-carrying supervisor control plane
├── CBP-G010  Doctrine seal + inventory (contracts, cache, ZK status)
├── CBP-G015  Proof-cache productization (default path + metrics)
├── CBP-G020  Reviewed property catalog
├── CBP-G030  Obligation compiler (AST + plan + domain residuals)
├── CBP-G040  Query API over evidence graph + cache projections
├── CBP-G050  Cache-aware re-proof / impact invalidation
├── CBP-G060  Obligation-first context capsules + decision core
├── CBP-G070  Delta retry via proof_delta + cache hits
├── CBP-G080  CodeEditPacket materializer → supervisor tasks
├── CBP-G090  Formal-plan preconditions require_proof(...)
├── CBP-G100  Bundle/optimizer co-locate obligations + cache locality
├── CBP-G110  SRT vertical: residual/structural properties in catalog+cache
├── CBP-G120  Supervisor self-properties (lease, merge, DAG, freshness)
├── CBP-G130  Token / proof-cost efficiency gates
└── CBP-G200  Attestation / real ZK (after A/B solid; sim fail-closed)
```

## Parallelism

| Wave | Goals | Notes |
| --- | --- | --- |
| 0 Seal | G010 | Root docs; unlocks all |
| 1 Foundation (max parallel) | G015, G020, G030 start after G010 | **G015 proof cache is independent of G020** |
| 2 Prove path | G040, G050 | Depend on G015+G020+G030 |
| 3 Context | G060, G070 | Depend on G040; G070 also G015 |
| 4 Materialize | G080, G090, G100 | After G040; G080 may start with G030 packets |
| 5 Domain | G110, G120 | Parallel after G040 |
| 6 Metrics | G130 | After G060+G015 |
| 7 ZK | G200 | Only after G050+G015 acceptance |

## CBP-G000 Proof-carrying supervisor control plane

- Status: active
- Parent:
- Priority: P0
- Track: codebase-proof
- Bundle: agent-supervisor/codebase-proof/root
- Goal: Deliver a schedulable objective heap and taskboard so the agent supervisor can prove reviewed codebase statements, cache results trust-aware, query them for edit judgment, and shrink agent context via obligation-first prompts—without unearned candidate or sim-ZK authority.
- Evidence: CBPEV000ROOT
- Outputs: docs/architecture/agent_supervisor_codebase_proof.objectives.md, docs/architecture/agent_supervisor_codebase_proof.todo.md, docs/architecture/AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md, config/agent_supervisor_codebase_proof_scheduler.json
- Validation: test -f docs/architecture/agent_supervisor_codebase_proof.todo.md && test -f config/agent_supervisor_codebase_proof_scheduler.json
- Acceptance: All child goals bound or explicitly blocked; board prefix `## CBP-` parseable; proof cache is a first-class lane; no production default that skips cache key bindings.
- Gap task: Execute child workstreams in parallel lanes per the taskboard.
- Conflict policy: Own CBP planning artifacts; do not reopen ASI completion IDs or rewrite PLAT sealed promotion reports.

## CBP-G010 Doctrine seal and inventory

- Status: active
- Parent: CBP-G000
- Priority: P0
- Track: docs
- Bundle: agent-supervisor/codebase-proof/docs
- Goal: Seal doctrine (assurance lattice, cache non-upgrade, candidate non-authority, sim ZK ≠ ATTESTED) and inventory reusable modules with explicit reuse vs extend decisions.
- Evidence: CBPEV010DOCS
- Outputs: docs/architecture/AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md, docs/architecture/agent_supervisor_codebase_proof.objectives.md, docs/architecture/agent_supervisor_codebase_proof.todo.md
- Validation: test -f docs/architecture/AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md
- Acceptance: Plan names `formal_verification_cache` as the sole proof-receipt memoization trust boundary; lists three proof kinds (property, receipt, crypto ZK); links SRT as optional vertical.
- Conflict policy: Own CBP docs only.

## CBP-G015 Proof-cache productization

- Status: active
- Parent: CBP-G000
- Priority: P0
- Track: proof-cache
- Bundle: agent-supervisor/codebase-proof/proof-cache
- Goal: Make `TrustAwareProofCache` / `ProofCache` the default memoization path for code-proof obligations: exact keys via `build_proof_cache_key`, single-flight for parallel supervisors, hit/miss/reject metrics, and fail-closed stale/poisoned/toolchain-drift handling.
- Evidence: CBPEV015CACHE
- Outputs: ipfs_accelerate_py/agent_supervisor/formal_verification_cache.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_formal_verification_cache.py, test/api/test_agent_supervisor_code_proof_cache_integration.py
- Validation: python -m pytest test/api/test_agent_supervisor_formal_verification_cache.py test/api/test_agent_supervisor_code_proof_cache_integration.py -q
- Acceptance: Every successful prove path records a cacheable receipt; hits re-derive assurance from typed evidence; cache key includes obligation, tree, premises, toolchain, policy, and required assurance; single-flight collapses concurrent identical work; negative tests for stale tree, poisoned entry, private material, and candidate-as-authoritative hit; metrics expose hit rate and rejection reason codes.
- Conflict policy: Extend existing cache module; do not introduce a parallel trust root or bypass re-derivation on hit.

## CBP-G020 Reviewed property catalog

- Status: active
- Parent: CBP-G000
- Priority: P0
- Track: property-catalog
- Bundle: agent-supervisor/codebase-proof/catalog
- Goal: Versioned, content-addressed catalog of reviewed properties (id, template, sorts, required assurance, query tags, semantic_authority=false default).
- Evidence: CBPEV020CAT
- Outputs: ipfs_accelerate_py/agent_supervisor/code_property_catalog.py, test/api/test_agent_supervisor_code_property_catalog.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_property_catalog.py -q
- Acceptance: Closed registration API; unknown property IDs fail closed; seed entries cover existing `ReviewedCodeShape` templates plus SRT structural constraint names as tags; catalog digests stable under key sort.
- Conflict policy: Own catalog module; reference templates by id only (no NL invent).

## CBP-G030 Obligation compiler

- Status: active
- Parent: CBP-G000, CBP-G020
- Priority: P0
- Track: obligations
- Bundle: agent-supervisor/codebase-proof/obligations
- Goal: Compile tree + changed AST scope + formal-plan effects (+ optional domain residual refs) into `CodeProofObligation` sets bound for cache keys and prove dispatch.
- Evidence: CBPEV030OBL
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_code_proof_scopes.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_scopes.py -q
- Acceptance: Unsupported shapes stay unsupported; cache-key builder invoked for each obligation; repository-wide opaque dumps never become premises; unit tests cover residual-ref binding hooks for SRT without importing live gold into receipts.
- Conflict policy: Extend `code_proof_obligations`; preserve ASI-G102 candidate non-authority.

## CBP-G040 Query API over evidence and cache

- Status: active
- Parent: CBP-G000, CBP-G015, CBP-G020, CBP-G030
- Priority: P0
- Track: queries
- Bundle: agent-supervisor/codebase-proof/queries
- Goal: Deterministic query surface: properties_satisfied, properties_open, properties_refuted, counterexamples, impact, proof_delta—projecting graph nodes and **cache lookup outcomes** without treating GraphRAG as proof.
- Evidence: CBPEV040QRY
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_query.py, ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, test/api/test_agent_supervisor_code_proof_query.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_query.py -q
- Acceptance: Queries return bounded, content-addressed results; cache miss ≠ refuted; open = no valid cache hit at required assurance; impact reopens obligations on path/AST change; proof_delta only lists invalidated obligations between parent/child trees.
- Conflict policy: Own query module; graph remains non-authoritative projection.

## CBP-G050 Cache-aware re-proof and invalidation

- Status: active
- Parent: CBP-G000, CBP-G015, CBP-G030, CBP-G040
- Priority: P0
- Track: reproof
- Bundle: agent-supervisor/codebase-proof/reproof
- Goal: On candidate trees, re-prove only open/invalidated obligations; satisfy from cache when bindings match; fail closed on drift.
- Evidence: CBPEV050REP
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_verification_cache.py, test/api/test_agent_supervisor_code_proof_reproof.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_reproof.py -q
- Acceptance: Warm re-proof uses cache hits for unchanged obligations; changed scope forces miss; parallel workers single-flight; merge/completion still re-bind receipts to current tree.
- Conflict policy: Coordinate with G015 on cache APIs only.

## CBP-G060 Obligation-first context capsules

- Status: active
- Parent: CBP-G000, CBP-G040
- Priority: P0
- Track: context
- Bundle: agent-supervisor/codebase-proof/context
- Goal: ContextCompiler profile that puts open obligations, refuted counterexamples, and satisfied receipt digests (not bodies) in the invariant core; optional evidence VoI-ranked; expansion handles for source.
- Evidence: CBPEV060CTX
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/decision_context.py, test/api/test_agent_supervisor_code_proof_context.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_context.py test/api/test_agent_supervisor_context_compiler.py -q
- Acceptance: Required core cannot drop open obligations; satisfied proofs appear as digests/handles only; cache hit metadata may be included as non-authority stats; raw solver traces excluded by default.
- Conflict policy: Extend context compiler profiles; preserve ASI-G091 required-coverage rules.

## CBP-G070 Delta retry via proof_delta and cache

- Status: active
- Parent: CBP-G000, CBP-G015, CBP-G060
- Priority: P1
- Track: context-delta
- Bundle: agent-supervisor/codebase-proof/context-delta
- Goal: Retry capsules carry only proof_delta + newly invalidated obligations; reuse cache hits for still-valid obligations without re-prompting full evidence.
- Evidence: CBPEV070DELTA
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_code_proof_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_context_delta.py -q
- Acceptance: Parent-bound deltas reconstruct without weakening core; token count on retry fixtures drops vs cold path; never re-opens satisfied cached obligations without impact reason.
- Conflict policy: Same context lane family as G060 after G060 lands, or serialize if file conflict.

## CBP-G080 CodeEditPacket materializer

- Status: active
- Parent: CBP-G000, CBP-G030, CBP-G040
- Priority: P0
- Track: materialize
- Bundle: agent-supervisor/codebase-proof/materialize
- Goal: Generalize plateau materializer pattern to `CodeEditPacket@1`: open obligations, cache status, implementable flag, predicted files, validation_commands (re-prove + tests).
- Evidence: CBPEV080MAT
- Outputs: ipfs_accelerate_py/agent_supervisor/code_edit_packet.py, ipfs_accelerate_py/agent_supervisor/code_edit_materialize.py, test/api/test_agent_supervisor_code_edit_packet.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_edit_packet.py -q
- Acceptance: implementable=false on reject/timeout/unsupported; prover fields semantic_authority=false; materializer emits supervisor-ready task fields; optional bridge from PlateauCodexPacket without coupling SRT gold into cache keys.
- Conflict policy: Own new packet modules under agent_supervisor; SRT bridge is additive.

## CBP-G090 Formal-plan require_proof preconditions

- Status: active
- Parent: CBP-G000, CBP-G020, CBP-G040
- Priority: P1
- Track: formal-plan
- Bundle: agent-supervisor/codebase-proof/formal-plan
- Goal: Formal work plans may declare `requires_proof(property_id, assurance)`; admission fails closed until cache-backed receipt satisfies.
- Evidence: CBPEV090PLAN
- Outputs: ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_formal_plan_proof_preconditions.py
- Validation: python -m pytest test/api/test_agent_supervisor_formal_plan_proof_preconditions.py -q
- Acceptance: Missing receipt blocks admission; cache hit with re-derived assurance admits; candidate-only does not.
- Conflict policy: Extend formal plan modules carefully; no LLM formula invent.

## CBP-G100 Bundle locality for obligations and cache

- Status: active
- Parent: CBP-G000, CBP-G040, CBP-G080
- Priority: P2
- Track: bundles
- Bundle: agent-supervisor/codebase-proof/bundles
- Goal: Bundle optimizer prefers co-located open obligations and shared proof-cache namespaces to maximize single-flight and context prefix reuse.
- Evidence: CBPEV100BUN
- Outputs: ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, test/api/test_agent_supervisor_code_proof_bundles.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_bundles.py -q
- Acceptance: Conflicting file scopes still serialize; independent obligation sets stay parallel; metrics show cache key locality preference without wrong-tree hits.
- Conflict policy: Extend bundle optimizer; do not break existing ASI bundle tests.

## CBP-G110 SRT residual/structural vertical

- Status: active
- Parent: CBP-G000, CBP-G020, CBP-G040, CBP-G080
- Priority: P1
- Track: srt-vertical
- Bundle: agent-supervisor/codebase-proof/srt
- Goal: Bind SRT structural constraints and residual facets into the property catalog + cache-aware packets so PLAT/PLAT2 loops share the CBP query and context path.
- Evidence: CBPEV110SRT
- Outputs: docs/architecture/agent_supervisor_codebase_proof_srt_bridge.md, integration tests under test/api/, optional thin adapter module
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_srt_bridge.py -q
- Acceptance: Structural admission receipts project into graph/query; e2e loss remains the promotion score; cache keys never include gold IR bodies.
- Conflict policy: Bridge only; do not mutate sealed PLAT promotion snapshots.

## CBP-G120 Supervisor self-properties

- Status: active
- Parent: CBP-G000, CBP-G020, CBP-G050
- Priority: P1
- Track: self-properties
- Bundle: agent-supervisor/codebase-proof/self
- Goal: Always-on (or policy-gated) obligations for lease fencing, merge idempotence, DAG acyclicity, evidence freshness using existing templates + cache.
- Evidence: CBPEV120SELF
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_code_proof_self_properties.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_self_properties.py -q
- Acceptance: Templates selected by exact ReviewedCodeShape; warm runs cache-hit; mutations invalidate correctly.
- Conflict policy: Own self-property wiring tests; coordinate template registry extensions only if needed.

## CBP-G130 Efficiency gates (tokens + proof cost)

- Status: active
- Parent: CBP-G000, CBP-G015, CBP-G060, CBP-G070
- Priority: P1
- Track: metrics
- Bundle: agent-supervisor/codebase-proof/metrics
- Goal: Paired fixtures prove context reduction and proof-cache benefit: cold vs warm prove cost, retry tokens, hit rate.
- Evidence: CBPEV130MET
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_code_proof_efficiency.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_efficiency.py -q
- Acceptance: Documented targets—≥40% fewer input tokens per accepted criterion (cold vs obligation-first), ≥60% retry token reduction, warm prove wall-time improvement when cache hits dominate; all with required coverage preserved.
- Conflict policy: Extend efficiency metrics; no prompt bodies in receipts.

## CBP-G200 Attestation and real ZK (deferred)

- Status: active
- Parent: CBP-G000, CBP-G015, CBP-G050
- Priority: P2
- Track: attestation-zk
- Bundle: agent-supervisor/codebase-proof/zk
- Goal: Optional path from kernel receipts to `ATTESTED` only with real cryptographic backend; public inputs = property/tree/obligation digests; sim ZK hard-fails ATTESTED.
- Evidence: CBPEV200ZK
- Outputs: ipfs_accelerate_py/agent_supervisor/proof_attestation.py, integration notes for datasets zkp backend, test/api/test_agent_supervisor_code_proof_attestation_policy.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_attestation_policy.py -q
- Acceptance: Simulated proofs cannot satisfy ATTESTED policy; attestation cache entries re-verify; private witnesses never enter public cache entries.
- Conflict policy: Policy and wiring only until real backend is selected; do not claim production ZK prematurely.

## Non-goals

- Proving arbitrary natural-language claims
- LLM-invented proof templates
- Replacing pytest / domain e2e with SMT scores
- Always-on full-repo formal verification
- Simulated ZKP as production cryptography
- A second proof-cache that skips `TrustAwareProofCache` re-derivation
