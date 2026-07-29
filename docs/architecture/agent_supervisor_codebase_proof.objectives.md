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
2. Represent claim family, evidence tier, assumptions, provenance, source revision,
   and invalidation policy explicitly; repository queries and observations are not
   silently promoted into mathematical proofs.
3. Discharge or refute supported obligations with fail-closed solvers / kernels /
   structural gates while retaining bounded test/runtime/static evidence as its
   own evidence tier.
4. **Memoize authoritative results in the trust-aware proof cache** (`formal_verification_cache.TrustAwareProofCache` / `ProofCache`) under exact semantic keys—hits re-derive assurance and never upgrade trust.
5. Index results so agents and humans can **query** satisfied / open / refuted /
   unsupported / not-measured / stale / counterexamples / impact / proof_delta.
6. Drive edit decisions from unsatisfied obligations and counterexamples.
7. Compile implementation prompts as **obligation-first context capsules** (tiny
   dependency/spec/failure core + expansion handles), shrinking tokens without
   dropping required coverage.
8. Measure claim reliability, repair quality, proof coverage, context tokens, and
   prove cost on preregistered baseline and held-out fixtures.
9. Optional **cryptographic attestation / ZK** only after property proofs + cache
   + queries are live and a private-witness threat model is approved; simulated ZK
   cannot emit `ATTESTED`.

First vertical consumer (parallel program, not a dependency of CBP root):
semantic-roundtrip plateau / holdout (`PlateauCodexPacket@1`, `StructuralAdmission@1`)
may bind residual obligations into the same catalog and cache.

## Existing foundations (reuse; do not fork trust models)

| Layer | Module / interface |
| --- | --- |
| Assurance lattice | `formal_verification_contracts` (`UNVERIFIED`…`ATTESTED`) |
| Provider isolation | `formal_verification_provider` |
| **Proof cache** | `formal_verification_cache` (`build_proof_cache_key`, single-flight, DuckDB) |
| Claim/evidence contracts | `formal_verification_contracts`, `code_proof_obligations` |
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
Typed claim record (family + assumptions + provenance + invalidators)
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
- Query/GraphRAG facts or bounded observations promoted into kernel assurance
- Simulated ZKP raising `AssuranceLevel.ATTESTED`
- Real ZK implementation without an approved private-witness threat model
- Private witnesses / gold / secrets in public receipts or default agent context
- Proof pass substituting for domain semantic metrics (e.g. SRT e2e loss)
- Second independent proof-cache trust model that bypasses `formal_verification_cache`

## Goal tree

```text
CBP-G000  Proof-carrying supervisor control plane
├── CBP-G010  Doctrine seal + inventory (contracts, cache, ZK status)
├── CBP-G015  Proof-cache productization (default path + metrics)
├── CBP-G020  Reviewed property catalog
├── CBP-G025  Typed claim/evidence semantics + lifecycle
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
├── CBP-G130  Closed-loop quality / coverage / token / proof-cost gates
└── CBP-G200  Attestation / real ZK (after A/B solid; sim fail-closed)
```

## Parallelism

| Wave | Goals | Notes |
| --- | --- | --- |
| 0 Seal | G010 | Root docs; unlocks all |
| 1 Foundation (max parallel) | G015, G020, G025; G030 after G020+G025 | **G015 proof cache is independent of G020/G025** |
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
- Goal: Versioned, content-addressed catalog of reviewed properties (id, claim family, template/spec version, sorts, required assurance, owner/reviewer, assumption and invalidation policies, query tags, semantic_authority=false default).
- Evidence: CBPEV020CAT
- Outputs: ipfs_accelerate_py/agent_supervisor/code_property_catalog.py, test/api/test_agent_supervisor_code_property_catalog.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_property_catalog.py -q
- Acceptance: Closed registration API; unknown property IDs and unreviewed natural-language templates fail closed; seed entries cover dependency/reachability, API contracts, behavioral invariants, security properties, semantic equivalence, supervisor lifecycle properties, existing `ReviewedCodeShape` templates, and SRT structural constraints; catalog digests stable under key sort.
- Conflict policy: Own catalog module; reference templates by id only (no NL invent).

## CBP-G025 Typed claim/evidence semantics and lifecycle

- Status: active
- Parent: CBP-G000
- Priority: P0
- Track: evidence-contract
- Bundle: agent-supervisor/codebase-proof/evidence
- Goal: Normalize existing property, obligation, evidence, and receipt contracts into a content-addressed `CodeClaimRecord@1` that makes claim family, evidence tier, assumptions, provenance, source revision, status, required assurance, and invalidation selectors queryable without creating a second assurance model.
- Evidence: CBPEV025SEM
- Outputs: ipfs_accelerate_py/agent_supervisor/code_claim_contracts.py, docs/architecture/agent_supervisor_code_claim_evidence_contract.md, test/api/test_agent_supervisor_code_claim_evidence_contract.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_claim_evidence_contract.py -q
- Acceptance: Record binds property/obligation ids, repository/tree/scope, premises/assumptions, producer/toolchain/policy/catalog versions, evidence ids, required assurance, and invalidators; lifecycle distinguishes unknown/open/satisfied/refuted/unsupported/not_measured/stale; evidence tiers distinguish query facts, bounded observations, solver/model-check candidates, kernel-checked proofs, and cryptographic attestations; no query, GraphRAG projection, test, runtime trace, or static-analysis result can independently mint kernel assurance; arbitrary NL claims fail closed; canonical round-trip and stale-evidence tests pass.
- Conflict policy: Build adapters over `formal_verification_contracts` and `code_proof_obligations`; do not duplicate ProofEvidence, ProofReceipt, assurance derivation, or proof cache.

## CBP-G030 Obligation compiler

- Status: active
- Parent: CBP-G000, CBP-G020, CBP-G025
- Priority: P0
- Track: obligations
- Bundle: agent-supervisor/codebase-proof/obligations
- Goal: Compile tree + changed AST scope + formal-plan effects (+ optional domain residual refs) into `CodeProofObligation` sets and typed claim records bound for cache keys and prove dispatch.
- Evidence: CBPEV030OBL
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, test/api/test_agent_supervisor_code_proof_scopes.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_scopes.py -q
- Acceptance: Unsupported and not-measured shapes remain distinct; each obligation carries typed premises/assumptions and invalidation selectors; cache-key builder binds property/catalog version, tree/scope, premise/assumption digests, toolchain, policy, and required assurance; repository-wide opaque dumps never become premises; tests cover dependency, API-contract, security, semantic-equivalence, and SRT residual-ref hooks without importing live gold into receipts.
- Conflict policy: Extend `code_proof_obligations`; preserve ASI-G102 candidate non-authority.

## CBP-G040 Query API over evidence and cache

- Status: active
- Parent: CBP-G000, CBP-G015, CBP-G020, CBP-G025, CBP-G030
- Priority: P0
- Track: queries
- Bundle: agent-supervisor/codebase-proof/queries
- Goal: Deterministic query surface: satisfied, open, refuted, unsupported, not-measured, stale, counterexamples, impact, and proof_delta—projecting typed claim provenance, graph nodes, and **cache lookup outcomes** without treating GraphRAG as proof.
- Evidence: CBPEV040QRY
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_query.py, ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, test/api/test_agent_supervisor_code_proof_query.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_query.py -q
- Acceptance: Queries return bounded, content-addressed results with claim/evidence ids and provenance handles; cache miss ≠ refuted; open = supported claim with no current evidence at required assurance; unsupported/not_measured/unknown/stale remain distinct; impact reopens obligations on any declared invalidator; proof_delta lists invalidated obligations and machine-readable reasons between parent/child trees.
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
- Acceptance: Warm re-proof uses cache hits for unchanged obligations; changed tree/blob/AST/path, dependency edge, premise/assumption digest, catalog, toolchain, policy, or assurance forces stale/open with a reason code and re-solve; parallel workers single-flight; merge/completion still re-bind receipts to the current tree.
- Conflict policy: Coordinate with G015 on cache APIs only.

## CBP-G060 Obligation-first context capsules

- Status: active
- Parent: CBP-G000, CBP-G040
- Priority: P0
- Track: context
- Bundle: agent-supervisor/codebase-proof/context
- Goal: ContextCompiler profile that puts the task/acceptance ids, dependency/AST slice, open obligations, assumptions, refuted counterexamples, relevant spec handles, bounded failure traces, and satisfied receipt digests (not bodies) in the invariant core; optional evidence VoI-ranked; expansion handles for source.
- Evidence: CBPEV060CTX
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/decision_context.py, test/api/test_agent_supervisor_code_proof_context.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_context.py test/api/test_agent_supervisor_context_compiler.py -q
- Acceptance: Required core cannot drop acceptance ids, open obligations, assumptions, or changed dependency/AST coverage; satisfied proofs appear as digests/handles only; relevant specs/failures are bounded; omitted material has a content-addressed handle manifest; untrusted repository text is labeled as data; cache metadata may be included as non-authority stats; raw solver traces and unrelated source are excluded by default.
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
- Acceptance: Packet binds source tree, claim/obligation ids, assumptions, invalidation reasons, predicted files, and acceptance ids; implementable=false on reject/timeout/unsupported/not_measured/stale-required-input; prover fields semantic_authority=false; materializer emits tests, domain metrics, and cache-aware re-proof commands at declared assurance; optional bridge from PlateauCodexPacket without coupling SRT gold into cache keys.
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
- Goal: Bind SRT structural constraints and residual facets into the property catalog + cache-aware packets so PLAT/PLAT2 loops share the CBP query and context path while preserving their benchmark, holdout, method-role, and promotion authority.
- Evidence: CBPEV110SRT
- Outputs: docs/architecture/agent_supervisor_codebase_proof_srt_bridge.md, integration tests under test/api/, optional thin adapter module
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_srt_bridge.py -q
- Acceptance: Bridge docs map autoencoder/spaCy to bounded guidance/diagnostics, SyMAI to orchestration, Leanstral to proposal teaching, Hammer/cvc5/Lean to declared structural gates, and the deterministic compiler/IR/decompiler to the edit target; PLAT residual-catalog and PlateauCodexPacket ids project into typed claims, counterexamples, context capsules, and CodeEditPacket; PLAT2 holdout artifacts remain preregistered and separately queryable; structural receipts have explicit non-semantic authority; e2e loss and holdout gates remain the promotion authority; cache keys never include gold IR bodies.
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

## CBP-G130 Closed-loop quality, coverage, token, and proof-cost gates

- Status: active
- Parent: CBP-G000, CBP-G015, CBP-G050, CBP-G060, CBP-G070, CBP-G080
- Priority: P1
- Track: metrics
- Bundle: agent-supervisor/codebase-proof/metrics
- Goal: Preregister a baseline and held-out mutation/repair suite that measures claim reliability, proof coverage, proof-guided edit quality, context reduction, and proof-cache benefit on identical bulk-source versus obligation-first tasks.
- Evidence: CBPEV130MET
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_code_proof_benchmark.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_code_proof_efficiency.py, docs/benchmarks/agent_supervisor_codebase_proof_evaluation.md
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_efficiency.py -q
- Acceptance: Held-out suite spans dependency, API-contract, behavioral, security, semantic-equivalence, and supervisor-lifecycle claims; report coverage by family/evidence tier/assurance, all lifecycle-state counts, false-admit/refute rates, stale-evidence detection, first-pass/eventual repair success, accepted-patch regressions, tokens, calls, cache outcomes, wall time, and proof cost; require zero false authoritative admissions, no required-coverage loss, ≥40% fewer input tokens per accepted criterion, ≥60% retry token reduction, and warm prove cost improvement; live-model results remain separate from deterministic fixture gates.
- Conflict policy: Extend efficiency metrics; no prompt bodies in receipts.

## CBP-G200 Attestation and real ZK (deferred)

- Status: active
- Parent: CBP-G000, CBP-G015, CBP-G050
- Priority: P2
- Track: attestation-zk
- Bundle: agent-supervisor/codebase-proof/zk
- Goal: Decide from an explicit private-witness/cross-trust-boundary threat model whether ZK is warranted; only then provide an optional path from kernel receipts to `ATTESTED` with a real cryptographic backend; sim ZK hard-fails ATTESTED.
- Evidence: CBPEV200ZK
- Outputs: ipfs_accelerate_py/agent_supervisor/proof_attestation.py, docs/architecture/agent_supervisor_codebase_proof_zk_threat_model.md, integration notes for datasets zkp backend, test/api/test_agent_supervisor_code_proof_attestation_policy.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_attestation_policy.py -q
- Acceptance: Threat model names prover, verifier, protected witness, disclosure risk, trust boundary, freshness/replay requirements, and why signed/kernel receipts are insufficient; backend selection requires a reviewed qualifying-use-case decision, while no qualifying case terminates as not_applicable without blocking core CBP; simulated proofs cannot satisfy ATTESTED; public inputs bind property/repository/tree/obligation/toolchain/policy/kernel-receipt digests; attestation cache entries re-verify and fail closed on drift; private witnesses never enter public receipts or cache entries.
- Conflict policy: Policy and wiring only until real backend is selected; do not claim production ZK prematurely.

## Non-goals

- Proving arbitrary natural-language claims
- LLM-invented proof templates
- Replacing pytest / domain e2e with SMT scores
- Promoting queries, GraphRAG, tests, runtime observations, or static analysis into kernel proof authority
- Prompt-only context compression without dependency/evidence coverage
- Always-on full-repo formal verification
- Simulated ZKP as production cryptography
- A real ZK backend without an approved private-witness threat model
- A second proof-cache that skips `TrustAwareProofCache` re-derivation
