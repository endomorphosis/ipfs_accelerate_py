# Agent Supervisor Codebase-Proof + Context Plan (CBP)

**Status:** ready for agent-supervisor handoff  
**Objectives:** `agent_supervisor_codebase_proof.objectives.md`  
**Taskboard:** `agent_supervisor_codebase_proof.todo.md`  
**Task prefix:** `## CBP-`  
**Namespace:** `agent-supervisor-codebase-proof-v1`  
**Scheduler config:** `config/agent_supervisor_codebase_proof_scheduler.json`

## Why

We need the supervisor to **prove reviewed statements about the codebase**, use
those results to **judge what must change** (queries + counterexamples), and
**shrink agent context** via obligation-first prompt capsules. Optional methods
and theorem provers remain gates/teachers; semantic metrics stay authoritative
for domain loss (e.g. semantic round-trip e2e).

Claims bind a reviewed family, assumptions, provenance, source revision, and
invalidation policy. Repository queries, GraphRAG projections, tests, runtime
observations, static analysis, solver candidates, kernel proofs, and
cryptographic attestations remain distinct evidence tiers. The closed loop is
evaluated for claim reliability and repair quality as well as token and prove
cost.

**Minor but first-class design choice:** all prove/re-prove paths go through the
existing **trust-aware proof cache**
(`ipfs_accelerate_py.agent_supervisor.formal_verification_cache`). Cache hits
are memoization under exact keys, never an assurance upgrade. Parallel workers
share single-flight leases so expensive solves are not duplicated.

## Doctrine

```text
Teachers / residuals / AST scopes (offline, non-authority)
        │
        ▼
Reviewed property catalog → CodeProofObligation[]
        │
        ▼
CodeClaimRecord (family + assumptions + provenance + invalidators)
        │
        ▼
Prove | refute (fail-closed providers)
        │
        ├── miss → solver/kernel → ProofReceipt → TrustAwareProofCache.put
        └── hit  → reconstruct receipt → re-derive assurance → admit only if OK
        │
        ▼
CodeEvidenceGraph + Query API
        │
        ├── open / refuted → what to change
        ├── satisfied digests → omit bodies from prompts
        └── proof_delta → retry context
        │
        ▼
Obligation-first ContextCapsule → agent supervisor edit
        │
        ▼
Re-prove (cache-aware) + tests → merge/completion gates
```

### Evidence and proof tiers

| Tier | Role | Authority |
| --- | --- | --- |
| Repository query / GraphRAG fact | Navigate and select relevant scope | Non-authoritative projection |
| Test / runtime / static-analysis observation | Bounded evidence about one revision and run | Observation only; never silently a theorem |
| Solver / model-check / ATP candidate | Search, refute, or draft a proof | Candidate unless policy independently verifies it |
| Kernel-checked property proof | Discharge a reviewed obligation at declared assurance | Authoritative only under exact receipt bindings |
| Cryptographic ZK / attestation | Verify without revealing a private witness | `ATTESTED` only with a real backend and approved threat model; **sim ≠ ATTESTED** |

Content-addressed evidence receipts support audit/replay and compact context but
do not change the authority of the evidence they reference.

### Normative rules

1. Candidate ATP/LLM drafts never complete or merge alone (ASI-G102).
2. Cache hit must re-derive assurance from typed evidence.
3. Private witnesses forbidden in public receipts and default capsules.
4. Proof pass does not lower domain semantic loss by itself.
5. **One cache trust boundary:** `formal_verification_cache` (+ attestation subcache). Do not invent a parallel root.
6. Every claim exposes assumptions, provenance, source binding, lifecycle state,
   and machine-readable invalidation reasons.
7. Real ZK work requires an approved private-witness/cross-trust-boundary use
   case; otherwise it terminates as not applicable.

## Parallel lanes

| Lane | Owns |
| --- | --- |
| `cbp-docs` | Plan seal |
| `cbp-proof-cache` | TrustAwareProofCache productization + metrics |
| `cbp-catalog` | Property catalog |
| `cbp-evidence` | Typed claim/evidence semantics + lifecycle |
| `cbp-obligations` | Obligation compiler |
| `cbp-queries` | Query API + graph projection |
| `cbp-reproof` | Cache-aware re-proof / invalidation |
| `cbp-context` | Obligation-first capsules |
| `cbp-context-delta` | proof_delta retries |
| `cbp-materialize` | CodeEditPacket + materializer |
| `cbp-formal-plan` | require_proof preconditions |
| `cbp-bundles` | Bundle locality |
| `cbp-srt` | SRT bridge (optional vertical) |
| `cbp-self` | Supervisor self-properties |
| `cbp-metrics` | Closed-loop quality, coverage, token, and proof-cost gates |
| `cbp-zk` | Attestation / real ZK policy |

## Task index

| Task | Depends on | Lane | Purpose |
| --- | --- | --- | --- |
| CBP-000 | — | cbp-docs | Seal plan artifacts |
| CBP-010 | 000 | cbp-docs | Doctrine inventory tests |
| **CBP-015** | 000 | **cbp-proof-cache** | **Proof cache default path + integration tests** |
| CBP-020 | 000 | cbp-catalog | Property catalog |
| CBP-025 | 000, 020 | cbp-evidence | Claim/evidence taxonomy, provenance, lifecycle |
| CBP-030 | 000, 020, 025 | cbp-obligations | Obligation compiler + cache key binding |
| CBP-040 | 015, 020, 025, 030 | cbp-queries | Query API |
| CBP-050 | 015, 030, 040 | cbp-reproof | Cache-aware re-proof |
| CBP-060 | 040 | cbp-context | Obligation-first capsules |
| CBP-070 | 015, 060 | cbp-context-delta | Delta retry |
| CBP-080 | 030, 040 | cbp-materialize | CodeEditPacket materializer |
| CBP-090 | 020, 040 | cbp-formal-plan | require_proof |
| CBP-100 | 040, 080 | cbp-bundles | Bundle locality |
| CBP-110 | 020, 040, 080 | cbp-srt | SRT bridge |
| CBP-120 | 020, 050 | cbp-self | Self-properties |
| CBP-130 | 015, 050, 060, 070, 080 | cbp-metrics | Closed-loop quality, coverage, and efficiency gates |
| CBP-200 | 015, 050 | cbp-zk | Attestation / ZK policy |

### Waves

1. **Foundation:** 010, **015**, and 020; then 025; then 030
2. **Prove path:** 040 → 050  
3. **Context:** 060 → 070  
4. **Materialize / plan / bundles:** 080, 090, 100  
5. **Domain:** 110, 120  
6. **Metrics / held-out repair evaluation:** 130
7. **ZK threat-model decision (deferred):** 200

## Proof caching specifics

Reuse:

- `build_proof_cache_key` / `ProofCacheKey` / `ProofCacheEntry`
- `TrustAwareProofCache` DuckDB boundary + single-flight
- Rejection reasons: stale, poisoned, toolchain_drift, private_material, solver_only, etc.
- Artifact store `proof_cache_outcomes` metrics where available

Productization acceptance for CBP-015/050:

- Put path on every independent successful prove
- Get path on every re-prove attempt before provider call
- Bind claim/catalog version, assumptions, scopes, and declared invalidators
- Concurrent identical keys → one flight
- Wrong tree / wrong obligation / candidate-only → miss or reject, never silent admit

## Relation to PLAT / PLAT2

PLAT pilots already showed residual → prover → packet → supervisor deterministic
edits and own the method-comparison evidence: autoencoder/spaCy are bounded
guidance/diagnostics, SyMAI is orchestration, Leanstral is a proposal teacher,
Hammer/cvc5/Lean are declared structural gates, and the deterministic
compiler/IR/decompiler is the edit target. CBP generalizes that loop beyond SRT
and **adds typed claims + cache + query + context** as shared infrastructure.
PLAT2 owns preregistered holdout scoring and may run **in parallel**; CBP-110
bridges artifact IDs and residuals without duplicating experiments, importing
gold bodies, or rewriting sealed promotion reports.

## Success criteria

| Level | Criterion |
| --- | --- |
| Board live | CBP-000 sealed; scheduler config present |
| Claim contract live | Claim family/evidence tier/provenance/lifecycle/invalidation semantics round-trip |
| Cache live | Warm re-prove hits cache; metrics + negative tests green |
| Query live | open/satisfied/refuted/unsupported/not_measured/stale/impact/proof_delta for seeded properties |
| Context win | Dependency/spec/failure coverage preserved with ≥40% input-token and ≥60% retry-token reductions |
| Edit loop | CodeEditPacket materializes implementable tasks from open obligations |
| Quality | Held-out mutations show zero false authoritative admissions; repair success and regression rates reported |
| Safety | 0 candidate or sim-ZK promotions to completion authority |
| ZK gate | No backend selection without an approved private-witness threat model |

## Non-goals

- Full-repo formal verification
- NL theorem invent
- Second proof-cache trust root
- Queries/tests/static analysis treated as kernel proof
- Prompt-only compression without dependency/evidence coverage
- Simulated ZK as production crypto
- Real ZK without a qualifying threat model
- Proof scores replacing domain e2e metrics
