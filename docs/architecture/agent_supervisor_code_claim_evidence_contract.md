# Code Claim / Evidence Contract (CBP-025)

**Interface:** `CodeClaimRecord@1`  
**Module:** `ipfs_accelerate_py.agent_supervisor.code_claim_contracts`  
**Schema:** `ipfs_accelerate_py/agent-supervisor/code-claim-record@1`

## Purpose

Normalize existing property, obligation, evidence, and receipt contracts into a
single content-addressed **claim record** that makes claim family, evidence tier,
assumptions, provenance, source revision, lifecycle status, required assurance,
and invalidation selectors **queryable** without creating a second assurance
model.

Authoritative assurance remains a projection of typed `ProofEvidence` via
`assess_assurance` / `derive_assurance` in `formal_verification_contracts`.
This module only **binds**, **classifies**, and **lifecycles** those results.

## Doctrine

```text
Reviewed property / obligation
        │
        ▼
CodeClaimRecord@1  (family + premises + assumptions + provenance + invalidators)
        │
        ├── query / GraphRAG fact     → tier query_fact / graphrag_fact
        ├── test/runtime/static obs.  → tier observation
        ├── solver / model-check      → tier solver_candidate
        ├── kernel-checked proof      → tier kernel_proof
        └── cryptographic attestation → tier cryptographic_attestation
        │
        ▼
Lifecycle: unknown | open | satisfied | refuted | unsupported | not_measured | stale
```

## CodeClaimRecord bindings

Every record binds:

| Field | Role |
| --- | --- |
| `property_id` | Reviewed catalog property (when known) |
| `obligation_id` | Content id of `CodeProofObligation` |
| `claim_family` | Closed family (see below) |
| `repository_id` / `repository_tree_id` | Source revision binding |
| `scope_ids` | AST / path scopes |
| `premise_ids` / `assumption_ids` | Semantic inputs |
| `producer_id` / `toolchain_id` / `policy_id` / `catalog_version` | Provenance versions |
| `evidence_ids` / `evidence_tiers` | Supporting evidence |
| `required_assurance` / `derived_assurance` | Policy requirement vs re-derived level |
| `invalidation_selectors` | Machine-readable stale triggers |
| `status` | Lifecycle (below) |
| `cache_lookup` | Optional hit/miss/stale/rejected memoization outcome |

`claim_id` is the content identity of the canonical payload (never trusted when
supplied without matching the re-derived digest).

## Claim families (closed)

| Family | Typical use |
| --- | --- |
| `dependency_reachability` | Import/call/dependency facts |
| `api_contract` | Interface / API obligations |
| `behavioral_invariant` | State machines, DAG, cache-key completeness |
| `security_property` | Authorization, lease fencing |
| `semantic_equivalence` | Projection / equivalence |
| `supervisor_lifecycle` | Merge, completion, supervisor self-properties |
| `srt_structural` | SRT structural tags (non_vacuous, cardinality, …) |
| `unsupported` | Explicit fail-closed unsupported shapes |

Arbitrary natural-language claims **fail closed**. A freeform statement without
a reviewed `property_id` or `obligation_id` is rejected.

## Lifecycle (`ClaimStatus`)

| Status | Meaning |
| --- | --- |
| `unknown` | No evaluation started |
| `open` | Admitted; awaiting evidence (**includes cache miss**) |
| `satisfied` | Settled positively at required assurance |
| `refuted` | Settled negatively with independent counterexample evidence |
| `unsupported` | Reviewed shape/template refuses the claim |
| `not_measured` | Measurement path not executed or out of bounds |
| `stale` | Prior evidence no longer binds current selectors |

### Cache miss is not refutation

A trust-aware proof-cache **miss** is memoization absence. `apply_cache_lookup`
maps miss → `open` (or preserves `unsupported` / `not_measured` / `stale`).
It never maps miss → `refuted`.

## Evidence tiers

| Tier | Role | Independent assurance ceiling |
| --- | --- | --- |
| `query_fact` | Repository query fact | `unverified` |
| `graphrag_fact` | GraphRAG / enrichment projection | `unverified` |
| `observation` | Bounded test / runtime / static-analysis | `candidate` |
| `solver_candidate` | ATP / SMT / solver / model-check candidate | `solver_checked` |
| `kernel_proof` | Independent kernel-checked proof | `kernel_verified` |
| `cryptographic_attestation` | Real attestation over a kernel receipt | `attested` |

### Non-upgrade rules

1. Query facts, GraphRAG projections, and observations **cannot independently
   mint** `kernel_verified` or `attested` assurance.
2. Solver candidates remain non-authoritative for merge/completion unless policy
   independently verifies them (ASI-G102).
3. Cache hits must **re-derive** assurance from typed evidence; a hit is not an
   upgrade.
4. Simulated attestations never reach `attested`.

## Adapters (no duplication)

| Source | Adapter |
| --- | --- |
| `CodeProofObligation` | `claim_from_obligation` → status `open` (or `unsupported`) |
| `ProofReceipt` | `claim_from_receipt` → re-derives via `assess_assurance` |
| `ImplementationResultEvidence` | `claim_from_implementation_evidence` → observation tier only |
| Query / GraphRAG | `claim_from_query_fact` → non-authoritative |
| Proof cache | `apply_cache_lookup` |

Invalidation: `build_invalidation_selectors`, `evaluate_invalidation`,
`mark_claim_stale`.

## Relationship to the proof cache

There is **one** cache trust boundary:
`formal_verification_cache.TrustAwareProofCache`. Claim records may record
lookup outcomes for queries and metrics; they do not store a parallel cache
root or alternate assurance lattice.

## Tests

`test/api/test_agent_supervisor_code_claim_evidence_contract.py` covers:

* Canonical round-trip and content identity
* Lifecycle distinctions (including cache miss ≠ refutation)
* Evidence tier ceilings and kernel non-mint rules
* Stale-evidence transitions
* Natural-language fail-closed behavior
* Adapters over obligation / receipt / implementation evidence
