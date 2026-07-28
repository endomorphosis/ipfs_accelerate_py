# Agent Supervisor Codebase Proof — ZK / Attestation Threat Model (CBP-200)

Status: reviewed for the core codebase-proof (CBP) program  
Related policy: [`agent_supervisor_codebase_proof_zk_policy.md`](./agent_supervisor_codebase_proof_zk_policy.md)  
Contracts: `ipfs_accelerate_py/agent_supervisor/proof_attestation.py`

This document names the threat model that must be accepted **before** selecting
or implementing a real zero-knowledge proving backend (Groth16, Plonk, Halo2,
ProveKit, or other). It does **not** claim that ZK discovers ordinary code
correctness; kernel-checked receipts already do that inside one trust domain.

## Roles

| Role | Responsibility |
| --- | --- |
| **Prover** | Holds a private witness and an existing, current, kernel-verified `ProofReceipt`. Emits only a public statement, proof artifact id, and proof digest. Never asserts assurance. |
| **Verifier** | Independently checks a public envelope against a pinned verification key, circuit, and public-input schema. Derives authority only after successful re-verification. |
| **Operator / supervisor** | Builds obligations, runs solvers/kernels, owns `formal_verification_cache` (+ attestation subcache), and enforces fail-closed gates. |
| **Third-party consumer** (optional) | Accepts an `ATTESTED` claim without learning the protected witness, only when a real backend and approved use case exist. |

## Protected witness

A **protected witness** is any proving input that must not appear in:

- public `ProofReceipt` JSON,
- attestation envelopes, verification records, or cache entries,
- context capsules, logs, or completion references.

Examples (when a qualifying use case exists): private premises, kernel
transcripts used only for proving, secret credentials bound into a circuit, or
cross-org proprietary intermediate artifacts.

In core CBP development, ordinary premises and kernel checks stay inside the
operator domain. There is **no** protected witness that must be hidden from a
third-party verifier for core property proofs—see the policy decision record.

Implementation: `PrivateAttestationWitness` is non-serializable, redacted in
`repr`, and rejected by public-artifact and cache helpers.

## Disclosure risk

| Channel | Risk if mishandled |
| --- | --- |
| Public receipt / statement | Leaks private field names or values via metadata or open maps |
| Attestation cache entry | Replays a witness-bearing object as if it were public evidence |
| Simulated ZKP path | Trains callers to treat demo proofs as `ATTESTED` |
| Logging / capsules | Serializes proving requests that still hold a witness |
| Backend adapters | Echo witness bytes into proof artifacts or diagnostics |

Mitigations: separate witness object, recursive private-marker rejection on
public payloads, witness no-leak backend fixtures, and **sim ≠ ATTESTED**.

## Trust boundary

There is **one** proof memoization trust boundary for CBP:

1. `formal_verification_cache` (TrustAwareProofCache), and  
2. its attestation subcache / sidecar records (`PersistedAttestationRecord`).

Rules:

- Cache hits **re-derive** assurance from typed evidence; they never upgrade it.
- Attestation sidecars bind an immutable kernel-verified receipt and expire
  independently of that receipt.
- No second proof-cache root may skip re-derivation or mint `ATTESTED`.

For core CBP, the prover and verifier are the same operator-controlled
supervisor/kernel path. A **cross-trust-boundary** need appears only when an
external party must verify a claim without learning a protected witness.

## Replay and freshness requirements

| Requirement | Rule |
| --- | --- |
| Base receipt freshness | Only `EvidenceFreshness.CURRENT`, kernel-verified, proved receipts may be attested |
| Public-input binding | Statement commits to property, repository/tree, obligation, toolchain, policy, and kernel-receipt digests (plus backend/circuit/vk pins when managed) |
| Sidecar lifetime | `created_at < expires_at`, and expiry cannot exceed verification-key expiry |
| Re-verification | Consumers must re-run an independent verifier from persisted public contracts; serialized `authoritative` / `verified` flags are never sufficient |
| Drift / expiry | Expired records, stale keys, binding mismatch, or verifier rejection **fail closed** (no simulated fallback) |
| Replay resistance | A valid envelope is only meaningful for its exact public inputs and pinned verification key; tree, policy, or toolchain drift invalidates reuse |

## Why ordinary signed or kernel receipts are insufficient

Kernel receipts and ordinary signed artifacts are the right tools for **operator
domain** assurance:

- they bind obligation, tree, toolchain, and policy identities;
- they support reconstruction and audit inside one trust domain;
- they can reach `AssuranceLevel.KERNEL_VERIFIED`.

They are **insufficient** when the goal is third-party verification **without**
disclosing a protected witness:

1. **Disclosure** — a signed receipt or kernel artifact typically reveals the
   full public statement and any attached evidence; it does not hide private
   proving inputs from a distrusting consumer.
2. **Trust model** — accepting a signature still requires trusting the signer’s
   key and process; it does not cryptographically prove that a private witness
   satisfied a circuit relation.
3. **Authority ceiling** — without a real cryptographic attestation over a
   kernel receipt, the lattice must not emit `AssuranceLevel.ATTESTED`.
4. **Simulation** — demo or simulated “ZK” envelopes exercise serialization only
   and are permanently non-authoritative (`sim ≠ ATTESTED`).

Therefore ZK is **optional and deferred** until a reviewed use case shows both a
protected witness and a cross-trust-boundary verifier. Core CBP remains fully
operable on kernel-verified receipts alone.

## Actors and abuse cases (summary)

| Abuse | Fail-closed response |
| --- | --- |
| Claim `ATTESTED` from simulated envelope | Authority stays `UNVERIFIED` / non-authoritative |
| Cache a witness-bearing proving request | `WitnessDisclosureError` |
| Forge public-input digest or statement id | Load rejected |
| Use expired verification key or sidecar | Not current; effective assurance falls back to kernel |
| Select Groth16/Plonk without approved decision | `require_zk_backend_selection_authorized` raises |
| Treat cache hit as upgraded assurance | Re-derivation only; no silent promotion |

## Normative references

- CBP plan: evidence tier “Cryptographic ZK / attestation”; **sim ≠ ATTESTED**
- Objectives CBP-G200: private-witness / cross-trust-boundary gate
- Module: `proof_attestation.py` (`ZkUseCaseDecisionRecord`,
  `ReceiptAttestationStatement`, `PersistedAttestationRecord`,
  `reproduce_attestation_verification`)
