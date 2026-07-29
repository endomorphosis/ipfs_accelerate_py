# SwissKnife Contract Assurance — ZK Attestation Policy

Status: reviewed, normative policy for SCA-080

Threat model:
[`SWISSKNIFE_CONTRACT_ZK_THREAT_MODEL.md`](./SWISSKNIFE_CONTRACT_ZK_THREAT_MODEL.md)

Extends:
[`agent_supervisor_codebase_proof_zk_policy.md`](./agent_supervisor_codebase_proof_zk_policy.md)

Implementation anchors: `agent_supervisor.proof.proof_attestation`,
`agent_supervisor.proof.mcp_contract_obligations`, and the optional
`ipfs_datasets_py.logic.bridge.zkp_attestation`

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, and **MAY** are
normative.

## Policy decision

SwissKnife contract assurance does not require ZK to prove ordinary contract
properties. The core operator-domain use case is reviewed as:

| Field | Decision |
| --- | --- |
| `use_case_id` | `swissknife-contract-assurance-core` |
| `disposition` | **`not_applicable`** (terminal) |
| `blocks_contract_assurance` | `false` |
| `qualifying_private_witness` | `false` |
| `qualifying_cross_trust_boundary` | `false` |
| `approved_backend_families` | empty |
| `reviewed_by` | `swissknife-contract-assurance-architecture-review` |
| `reviewed_at` | `2026-07-29T00:00:00Z` |
| production assurance ceiling | `KERNEL_VERIFIED` |
| rationale | Contract properties concern versioned repository and interface snapshots handled inside the supervisor trust domain. A current kernel-checked receipt is sufficient; ZK adds no privacy benefit. |

This terminal outcome MUST NOT block analysis, proving, caching, repair, merge,
or completion that requires no ZK. It MUST NOT authorize a real backend.

A distinct use case MAY be reviewed later only for one of the three eligible
predicate classes below. No backend implementation or selection is authorized
merely because a class is listed.

## Mandatory separation of assurances

An implementation MUST evaluate these gates in order:

1. **Property proof gate** — validate that the obligation is supported, proved,
   current, independently checked, and classified at least
   `KERNEL_VERIFIED`.
2. **Receipt integrity gate** — recompute canonical bytes and all declared
   identity/profile bindings, validate freshness and policy, and admit or load
   the receipt only through `TrustAwareProofCache`.
3. **ZK applicability and attestation gate** — evaluate a reviewed use-case
   decision, capability report, public statement, real proof, and independent
   verifier result.

Failure of gate 1 or 2 MUST terminate before witness construction or backend
dispatch. Gate 3 MUST NOT fill missing evidence, reinterpret an unsupported
obligation, or increase the base property's truth status.

`ATTESTED` describes the independently verified attestation predicate. It is
not a synonym for `proved`, `kernel_verified`, “receipt exists,” or “code is
correct.”

## Closed predicate catalog

`ProofAttestationPolicy.predicate_kind` MUST be one of:

| Predicate kind | Normative relation | Required private witness | Required public anchor |
| --- | --- | --- | --- |
| `receipt_possession` | Prover knows the canonical receipt preimage/opening whose approved commitment equals the public receipt CID, and the referenced receipt has already passed gates 1 and 2. | Receipt preimage/opening or approved possession secret | Receipt/property/snapshot/policy and backend/setup bindings |
| `receipt_membership` | Prover knows a valid path/opening that places the bound receipt commitment in the committed result set. | Private leaf and membership path/opening | Receipt commitment and exact result-set root, plus all common bindings |
| `private_reviewed_predicate` | Prover knows private values satisfying the exact separately reviewed, versioned relation. | Only fields declared by that predicate manifest | All common bindings plus predicate-specific reviewed public inputs |

No fourth generic/custom/arbitrary kind is permitted. A
`private_reviewed_predicate` MUST name a specific immutable predicate manifest;
it MUST NOT accept source text, free-form logic, dynamically generated
constraints, an LLM-authored theorem, or caller-defined witness fields.

The relation MUST be fully constrained by the reviewed circuit. Receipt
possession and membership prove only their stated knowledge/membership facts;
they do not independently prove receipt validity or the underlying contract
property.

## Applicability and terminal outcomes

Before backend discovery, the supervisor MUST produce a reviewed use-case
decision containing at least:

- use-case and predicate IDs/revisions;
- terminal disposition and review identity/time;
- private-witness and cross-trust-boundary qualification;
- protected-witness, disclosure, replay, and signed-receipt-insufficiency
  rationale;
- verifier domain and data-retention boundary; and
- permitted backend families, if approved.

Disposition rules:

| Condition | Required disposition/result | Retry or fallback |
| --- | --- | --- |
| Core in-domain property assurance; no qualifying private witness or external verifier | `not_applicable` | Terminal for this use-case revision; continue without ZK |
| Predicate outside the closed catalog | `not_applicable` | Terminal; no simulation, backend search, or predicate substitution |
| Catalog kind selected but concrete private predicate manifest is missing/unreviewed | `not_applicable` | Terminal for the request; a new reviewed use-case revision is required |
| Review is explicitly underway and a decision is required by the caller | `pending_review` | Below `ATTESTED`; MUST NOT select a production backend |
| Review finds unacceptable leakage, setup, soundness, or operational risk | `rejected` | Terminal for that revision |
| Qualifying witness and cross-boundary need with complete review | `approved` | Continue only with an authorized backend family |

Unknown, missing, malformed, or contradictory decision data MUST fail closed.
An unsupported use case MUST terminate as `not_applicable`; it MUST NOT be
reported as a transient backend error and MUST NOT invoke any provider.

## Backend selection and capability policy

A real backend MAY be selected only when all of the following hold:

- the use-case disposition is `approved`;
- the use case qualifies both a private witness and a cross-trust-boundary
  verifier;
- the predicate kind is in the closed catalog and its exact manifest is
  reviewed;
- the backend family appears in the decision record;
- an immutable `AttestationBackendPolicy` pins backend, circuit, public-input
  schema, setup manifest, proving key, verification key, versions, identity
  profiles, security level, and expiry/rotation;
- a current capability report is bound to that backend policy and use-case
  revision; and
- an independent verifier is configured for the same pins.

Configuration, importability, a product name, environment opt-in, provider
`available=true`, or a successful smoke proof is not sufficient.

The capability report MUST distinguish `simulated`, `unavailable`,
`configured`, `available`, `degraded`, and `verified`. Production eligibility
requires the real cryptographic mode and `verified` health. The report MUST
include current evidence for:

- a golden valid proof;
- a negative/false-witness proof;
- wrong statement and public-input substitution;
- stale or wrong verification key and setup;
- malformed proof;
- replay/domain/freshness rejection;
- cross-profile identity mismatch; and
- witness no-leak across every public boundary.

Capability or pin drift after proof generation MUST invalidate production
eligibility. A cryptographic failure MUST produce rejection/error; it MUST NOT
fall back to a simulated success.

## Simulation policy

Simulation is allowed only for serialization, UI, integration, and negative
tests. It MUST be labeled `simulated` at every boundary and MUST remain below
`ATTESTED`.

A simulated proof MAY be internally “verified” by its matching demo verifier,
but its authority is `NON_AUTHORITATIVE` and its contributed assurance is
`UNVERIFIED`. It MUST NOT satisfy production, authorization, completion, merge,
release, or compliance gates.

The datasets `ZkpAttestationBridgeAdapter` defaults to the simulated backend
and may emit `verified_by = ("zkp:simulated",)`. The SwissKnife adapter MUST
map that result to a typed simulated observation, never `ATTESTED`. Likewise,
the existence or opt-in availability of datasets Groth16/ProveKit does not
bypass the use-case decision, backend policy, capability fixtures, or
independent verifier.

## Base receipt requirements

An attestation request MUST reference one exact current proof receipt that:

- is `proved`, not refuted, unsupported, inconclusive, timed out, or failed;
- meets the obligation's required assurance and is at least
  `KERNEL_VERIFIED` before attestation;
- is independently checked when its producer is only a candidate solver;
- binds snapshot/scope, property and catalog revision, exact premises and
  assumptions, solver/kernel/toolchain, policy, capability report, and
  invalidators;
- has canonical bytes whose multihash is recomputed under the declared
  content-identity profile; and
- is admitted by or freshly reconstructed for `TrustAwareProofCache`.

`ATTESTED` MUST NOT be requested for a stale receipt. Historical attestations
MAY remain as audit evidence only when marked stale and excluded from current
authority.

## Required public inputs

The canonical public statement MUST bind the following pairs. Both the CID and
its content-identity profile ID are required; display labels or a digest
without its profile are insufficient.

| Binding | Requirement |
| --- | --- |
| Receipt | receipt CID/profile and canonical receipt schema/version |
| Property/obligation | property CID/profile, obligation CID/profile, and required assurance |
| Snapshot | repository ID plus snapshot/root CID/profile and scope CID/profile |
| Policy | use-case-decision CID/profile and attestation-policy CID/profile |
| Predicate | kind, predicate-manifest CID/profile, circuit CID/profile/version, public-input-schema CID/profile/version |
| Backend | backend-policy and implementation CID/profile/version, backend family and real/simulated mode |
| Setup | setup-manifest, proving-key, and verification-key CID/profile/version plus key expiry/epoch |
| Result set | committed result-set root CID/profile for membership; otherwise a policy-defined, domain-separated `not_applicable` sentinel |
| Replay domain | verifier/domain ID, challenge or nonce, issued-at, expires-at, and approved revocation epoch |
| Envelope | envelope/proof schema versions and canonicalization version |

Implementations MAY expose individual fields or a domain-separated digest of
their canonical closed record, but the circuit and verifier MUST constrain the
complete set. All identity fields MUST be reconstructed from validated
artifacts. Ambient defaults, mutable configuration, open extension maps, and
provider-supplied authority flags MUST NOT influence verification.

Changing any bound value creates a different statement. A proof for one
receipt, profile, snapshot, property, policy, backend, setup, result root,
verifier domain, or challenge MUST fail for every other value.

## Witness handling

Witness custody begins only after every public precondition and capability
gate passes. Implementations:

- MUST use a dedicated non-serializable witness type with redacted `repr`;
- MUST reject witness objects and witness-bearing proving requests from public
  artifact and cache helpers;
- MUST NOT copy a witness into a receipt, public envelope, context capsule,
  prompt, log, trace, metric, exception, cache record, CAS object, completion
  reference, or provider diagnostic;
- MUST NOT expose witness field names, lengths, low-entropy hashes, derived
  values, or reversible encodings through those channels;
- MUST pass the witness directly to a local, isolated prover through a
  narrowly scoped read-only view;
- MUST bound ephemeral storage and clean it on success, rejection, exception,
  timeout, and cancellation;
- MUST zeroize mutable buffers where supported and document stronger process
  isolation where the runtime cannot guarantee erasure; and
- MUST retain only the public statement, proof artifact/digest, backend and
  setup pins, independent verification, and non-secret diagnostic code.

A safe boolean marker such as `private_witness_redacted: true` MAY be public
when it reveals no witness metadata. Redaction after serialization is not
acceptable.

## Setup and key policy

The setup policy MUST state whether setup is trusted or transparent. A trusted
setup manifest MUST bind ceremony provenance, contributors/trust assumption,
transcript, circuit, parameters, keys, artifact digests, custody, rotation,
expiry, revocation, and toxic-waste handling. Deterministic developer setup
MUST NOT be treated as a production ceremony.

Verification keys MUST be authenticated, current at verification time, and
pinned to the circuit and public-input schema. Proof expiry MUST NOT exceed
verification-key or policy expiry. Setup, circuit, schema, or key rotation
changes the backend-policy identity and prevents silent reuse.

## Verification and assurance derivation

The supervisor, not the provider, derives authority. A verifier MUST:

1. parse with a closed versioned schema and reject unknown critical fields;
2. validate the approved use-case decision and predicate;
3. revalidate the base property receipt and its current integrity;
4. reconstruct every public input from trusted artifacts;
5. validate backend policy, setup/key pins, capability report, and expiry;
6. compare the reconstructed statement to the envelope byte-for-byte under
   canonical encoding;
7. cryptographically verify using the independent pinned verifier; and
8. recheck verifier domain, nonce/challenge, freshness, revocation, and cache
   invalidators.

Effective attestation assurance is:

```text
ATTESTED
  only if base_receipt_valid_and_current
       and use_case_approved
       and predicate_reviewed
       and backend_mode == cryptographic
       and capability_health == verified
       and all_public_bindings_match
       and independent_verdict == verified
       and replay_and_expiry_checks_pass
else UNVERIFIED (for the attestation contribution)
```

The base receipt retains its separately derived assurance. An unavailable or
failed optional attestation MUST NOT erase a valid `KERNEL_VERIFIED` property
receipt, but it cannot satisfy a gate that explicitly requires `ATTESTED`.

## Persistence and cache policy

Only public attestation sidecars MAY be persisted. A sidecar MUST bind the
immutable receipt, statement, proof digest/artifact, independent verification,
capability report, policy, backend/setup, creation, expiry, and invalidators.
It MUST be rejected if recursive witness-marker scanning detects secret
material.

Attestation sidecars:

- MUST remain subordinate to `TrustAwareProofCache`;
- MUST expire independently and no later than their receipt, key, capability,
  or policy validity;
- MUST re-run schema, binding, freshness, and independent-verification checks
  on load;
- MUST NOT trust serialized `verified`, `authoritative`, or assurance fields;
  and
- MUST NOT promote a cache hit or datasets bridge cache record.

## Failure and degradation contract

| Event | Public outcome | Maximum attestation contribution |
| --- | --- | --- |
| Use case outside policy | terminal `not_applicable` | `UNVERIFIED` |
| Pending review | `pending_review` | `UNVERIFIED` |
| Simulation succeeds | `simulated` | `UNVERIFIED` |
| Backend missing or not qualified | `unavailable` or `degraded` | `UNVERIFIED` |
| Generation timeout/error | `error` | `UNVERIFIED` |
| Verification rejects | `rejected` | `UNVERIFIED` |
| Binding, replay, freshness, setup, key, or profile mismatch | `rejected` | `UNVERIFIED` |
| Witness disclosure detected | `error` plus security diagnostic; discard public artifact | `UNVERIFIED` |
| Every production rule passes | `attested` | `ATTESTED` |

Failures MUST use bounded non-secret reason codes. The absence of ZK MUST not
be disguised as success, and unsupported predicates MUST not be retried
through a different predicate or provider.

## `ProofAttestationPolicy` minimum contract

The implementation introduced after SCA-080 MUST use an immutable,
content-addressed policy record containing at least:

- schema/version, policy ID/profile, use-case decision ID/profile;
- predicate kind and predicate-manifest ID/profile;
- required base assurance and accepted receipt schemas;
- every required public-binding name and canonicalization/profile rule;
- authorized backend families and backend-policy ID/profile;
- verifier domain, challenge/freshness, expiry, and revocation rules;
- witness retention and isolation mode;
- result-set-root rule;
- capability fixture requirements; and
- review identity/time and policy expiry.

Missing fields, open-ended predicate kinds, or contradictory rules make the
policy invalid. Policy revision changes its content identity and invalidates
reuse under the prior statement.

## Conformance requirements

Before any SwissKnife path can emit `ATTESTED`, tests MUST demonstrate:

- property proof, receipt integrity, and ZK are three independent gates;
- no ZK request occurs for unproved, stale, forged, unsupported, or
  cache-ineligible receipts;
- only the three catalog predicate kinds parse;
- unsupported use cases terminate `not_applicable` without provider dispatch;
- simulation and provider-local `verified` never satisfy production or
  completion;
- every required public binding, including each identity-profile ID, is
  substitution-tested;
- changed policy, backend, circuit, setup, key, snapshot, receipt, property,
  result root, verifier domain, challenge, expiry, and capability report fail
  closed;
- malformed proofs and cross-profile identities fail closed;
- witness probes never appear in receipt, context, prompt, log, trace, metric,
  cache, exception, or public proof artifacts; and
- cache reload re-derives authority rather than trusting serialized flags.

## Change control

Adding a predicate, backend family, verifier domain, identity profile,
canonicalization version, setup mode, aggregation/recursion feature, or witness
transport requires a new reviewed policy revision and threat-model review.
Emergency disabling of an affected backend or key MUST immediately remove
production eligibility without enabling simulation fallback.
