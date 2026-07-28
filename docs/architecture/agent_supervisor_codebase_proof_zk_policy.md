# Agent Supervisor Codebase Proof — Real ZK / Attestation Policy (CBP-200)

Status: **reviewed decision recorded**  
Threat model: [`agent_supervisor_codebase_proof_zk_threat_model.md`](./agent_supervisor_codebase_proof_zk_threat_model.md)  
Implementation: `ipfs_accelerate_py/agent_supervisor/proof_attestation.py`  
Validation: `test/api/test_agent_supervisor_code_proof_attestation_policy.py`

## Purpose

Define when (if ever) the codebase-proof program may select or implement a real
cryptographic attestation backend, and the fail-closed rules that apply to any
path that could claim `AssuranceLevel.ATTESTED`.

## Decision gate (mandatory)

**No** Groth16, Plonk, Halo2, Marlin, Fflonk, Nova, ProveKit, or other production
proving-system selection or implementation is allowed until a reviewed
`ZkUseCaseDecisionRecord` exists with:

- `disposition = approved`
- `qualifying_private_witness = true`
- `qualifying_cross_trust_boundary = true`
- a non-empty `approved_backend_families` list that includes the family to select

Enforcement helper: `require_zk_backend_selection_authorized(decision, backend_family=...)`.

Simulated, mock, demo, or educational backends **cannot** be authorized for
production selection and **cannot** satisfy `ATTESTED`.

## Reviewed use-case decision record (core CBP)

| Field | Value |
| --- | --- |
| `use_case_id` | `cbp-core-codebase-proof` |
| `disposition` | **`not_applicable`** (terminal) |
| `blocks_core_cbp` | **`false`** |
| `qualifying_private_witness` | `false` |
| `qualifying_cross_trust_boundary` | `false` |
| `approved_backend_families` | _(empty)_ |
| `reviewed_by` | `cbp-architecture-review` |
| `reviewed_at` | `2026-07-28T00:00:00Z` |

### Rationale

Core codebase-proof work proves properties of **public** repository trees with
kernel-checked receipts inside a **single operator trust domain**. There is no
qualifying private witness that must be hidden from a third-party verifier, and
no cross-trust-boundary consumer that must accept `ATTESTED` without learning
that witness. Therefore a real ZK backend is **not warranted** for core CBP.

### Program impact

- Terminal **`not_applicable`** does **not** block core CBP completion, merge,
  cache, reproof, or kernel-verified assurance paths.
- A `not_applicable` decision with `blocks_core_cbp=true` is **invalid** and
  rejected by contract validation.
- Production ceiling for core CBP remains `KERNEL_VERIFIED` unless a **new**
  approved decision is reviewed later for a distinct use case id.

Machine-readable constant: `CORE_CBP_ZK_USE_CASE_DECISION` /
`core_cbp_zk_use_case_decision()`.

## Trust boundary

The sole memoization trust boundary remains **`formal_verification_cache`**
(TrustAwareProofCache) plus its attestation subcache / sidecar records. Cache
hits re-derive assurance; attestation sidecars never mint a second proof-cache
root. Simulated paths cannot promote into that boundary as `ATTESTED`.

## Assurance rules

| Rule | Enforcement |
| --- | --- |
| Simulated ZKP / attestation ≠ `ATTESTED` | Envelope mode `simulated` → non-authoritative; `authoritative_assurance` is `UNVERIFIED` |
| Real path only over kernel receipts | `ProofReceipt.require_kernel_verified()` before statement build |
| Public inputs bind CBP identities | property, repository/tree, obligation, toolchain, policy, kernel-receipt **ids and digests** via `require_cbp_public_bindings` / `build_cbp_public_bindings` |
| Private witnesses stay private | `PrivateAttestationWitness`; `reject_private_witness_from_public_payload`; cache entry helper |
| Re-verify, fail closed on drift | `reproduce_attestation_verification` / `PersistedAttestationRecord.is_current_at` |
| No simulated fallback after crypto failure | `execute_cryptographic_attestation` raises or returns rejected/error only |

## Public-input binding (CBP)

When a CBP-grade attestation statement is built (`require_cbp_bindings=True` or
full identity slots populated), public inputs **must** include:

**Identities:** `property_id`, `repository_id`, `repository_tree_id`,
`obligation_id`, `toolchain_id`, `policy_id`, `kernel_receipt_id`

**Digests:** `property_digest`, `repository_digest`, `repository_tree_digest`,
`obligation_digest`, `toolchain_digest`, `policy_digest`,
`kernel_receipt_digest`

Managed backends additionally pin backend policy, circuit, public-input schema,
and verification-key versions.

## Backend health (when a future use case is approved)

Before production verification:

1. Backend configured and available  
2. All required fixtures pass: golden, negative, stale_key, malformed_proof,
   witness_no_leak  
3. Verification key current at use time  
4. Independent verifier success recorded  

Failure modes never fall back to simulated success.

## Integration notes (datasets / external ZKP)

Optional adapters (e.g. datasets ZKP bridges) may:

- generate **non-authoritative** educational envelopes for serialization tests;
- supply production provers **only** after an **approved** use-case decision and
  pinned `AttestationBackendPolicy`.

They must not:

- write private witnesses into receipts or attestation cache entries;
- claim `ATTESTED` without independent cryptographic verification;
- bypass `require_zk_backend_selection_authorized` for backend selection.

## Future approved cases

If a later program needs third-party verification of a private witness:

1. Write/update the threat model for that use case id.  
2. Record an **approved** `ZkUseCaseDecisionRecord` with named backend families.  
3. Pin circuit, public-input schema, verification key, and health fixtures.  
4. Wire attestation only as a sidecar over existing kernel receipts.  
5. Keep core CBP operable if that sidecar is unavailable.

## Non-goals

- Using ZK to discover ordinary repository correctness  
- Simulated ZK as production cryptography  
- A second proof-cache trust root  
- Selecting a proving system “to finish CBP-200” without a qualifying use case
