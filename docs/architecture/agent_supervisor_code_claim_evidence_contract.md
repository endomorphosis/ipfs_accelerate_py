# Code Claim / Evidence Contract (CBP-025)

**Interface:** `CodeClaimRecord@1`  
**Module:** `ipfs_accelerate_py.agent_supervisor.code_claim_contracts`

## Purpose

Normalize “what is claimed about the codebase” into a content-addressed record
that binds:

- property / obligation ids  
- claim family  
- repository + tree + scope  
- premises / assumptions  
- producer / toolchain / policy / catalog versions  
- evidence ids + coarse evidence tiers  
- required assurance  
- invalidation selectors  

This layer does **not** re-derive assurance and does **not** replace
`ProofReceipt` or `TrustAwareProofCache`.

## Claim lifecycle (`ClaimStatus`)

| Status | Meaning |
| --- | --- |
| `unknown` | No claim recorded |
| `open` | Claim exists; evidence incomplete |
| `satisfied` | Bound evidence meets policy (kernel tier required for kernel assurance) |
| `refuted` | Bound evidence shows failure |
| `unsupported` | No reviewed template/shape |
| `not_measured` | Intentionally deferred |
| `stale` | Was open/satisfied; invalidation selectors fired |

**Cache miss is never refutation** — use `open` / `not_measured`
(`cache_miss_status()` → `open`).

## Evidence tiers (`EvidenceTier`)

| Tier | Sources | May mint kernel assurance alone? |
| --- | --- | --- |
| `query_fact` | GraphRAG / retrieval-like | **No** |
| `observation` | tests, runtime, static analysis, `ImplementationResultEvidence` | **No** |
| `solver_candidate` | ATP/SMT/LLM/solver | **No** |
| `kernel_proof` | Lean/kernel verification | **Yes** (with independent evidence) |
| `cryptographic_attestation` | ZK/attestation over a kernel receipt | Not alone |

## Normative rules

1. Arbitrary natural-language claims fail closed (long/multiline prose rejected
   unless `natural_language_allowed=true` for audited text).
2. `status=satisfied` + `required_assurance=kernel_verified|attested` requires
   a kernel proof tier in `evidence_tiers`.
3. Invalidation selectors force `stale` when tree/scope/premises/policy/toolchain/catalog change.
4. Project from `ProofReceipt` via `claim_from_proof_receipt` without re-deriving
   authoritative assurance (candidate receipts stay `open`).

## Relation to CBP later tasks

- **CBP-030** emits claims with obligations  
- **CBP-040** queries claim status sets (open/satisfied/refuted/stale)  
- **CBP-060** puts open claims in context capsules  
- **CBP-080** materializes edit packets from open/refuted claims  
