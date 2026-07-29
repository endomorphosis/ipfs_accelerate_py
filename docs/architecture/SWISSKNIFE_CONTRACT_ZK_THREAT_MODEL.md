# SwissKnife Contract Assurance — ZK Threat Model

Status: normative architecture baseline for SCA-080

Applies to: SwissKnife MCP contract-assurance receipt attestation

Companion policy:
[`SWISSKNIFE_CONTRACT_ZK_POLICY.md`](./SWISSKNIFE_CONTRACT_ZK_POLICY.md)

Extends:
[`agent_supervisor_codebase_proof_zk_threat_model.md`](./agent_supervisor_codebase_proof_zk_threat_model.md)

## Purpose and security claim

Zero-knowledge (ZK) is an optional, post-proof disclosure mechanism. It can let
a verifier check one reviewed statement about an already validated proof
receipt without learning the statement's private witness. It does not analyze
source, discover a contract property, repair a failed proof, or turn
unsupported reasoning into proof.

The system keeps three questions separate:

| Layer | Question answered | Authoritative mechanism | ZK effect |
| --- | --- | --- | --- |
| **Property proof** | Does the reviewed contract property hold for this exact obligation and snapshot? | Trusted deterministic checker or independently checked proof kernel, producing a typed receipt classified at least `KERNEL_VERIFIED` | None. ZK cannot create, replace, or strengthen the property proof. |
| **Receipt integrity** | Are these the exact current receipt bytes and bindings admitted by policy? | Canonical encoding, content identity/profile validation, multihash recomputation, freshness checks, and `TrustAwareProofCache` admission | None. ZK cannot make a forged, stale, incomplete, or cache-ineligible receipt valid. |
| **ZK attestation** | Does a holder know a witness satisfying one approved predicate over that valid receipt? | A real, capability-qualified backend plus independent verification of exact public inputs | May add `ATTESTED` only after the first two layers pass. |

Consequently, an attestation must never be interpreted as “this code is
correct.” Its narrow claim is “the approved attestation predicate was verified
for the public inputs committed by this envelope.”

## Scope

This model covers:

- SwissKnife MCP contract obligations compiled by the canonical obligation
  layer;
- current proof receipts admitted through the sole trust-aware proof cache;
- receipt-possession, result-set-membership, and specifically reviewed
  private-witness predicates;
- lazy integration with `agent_supervisor.proof.proof_attestation` and an
  optional `ipfs_datasets_py` ZKP provider; and
- public envelopes, sidecar cache records, verifier results, and capability
  reports consumed by production or completion gates.

It excludes arbitrary program correctness, proof discovery, natural-language
claims, authorization based only on a provider assertion, and any use case
without a reviewed predicate and qualifying privacy boundary.

## Actors and trust boundaries

| Actor | Responsibility | Must not be trusted to assert |
| --- | --- | --- |
| Contract prover/kernel | Discharge a supported property obligation and emit a current typed receipt | Its own assurance without independent validation |
| Attestation prover | Hold the private witness and generate a proof for the exact public statement | `ATTESTED`, verifier success, freshness, or policy eligibility |
| Independent verifier | Reconstruct public inputs, validate pins, and verify the cryptographic proof | Property correctness that is absent from the base receipt |
| Supervisor/policy gate | Validate receipt integrity, capability, use-case decision, and effective assurance | Mutable provider metadata or serialized authority flags |
| Optional external consumer | Consume only a reproduced verifier result for its verifier domain | A proof copied from another domain, snapshot, policy, or setup |
| Cache/CAS | Retain canonical public evidence under exact identities | Witness custody or assurance promotion |

There is one proof memoization trust boundary:
`TrustAwareProofCache`, including its validated attestation sidecars. A
datasets bridge cache, provider-local proof cache, model context, or external
artifact store is not a second authority root. Every cache hit re-derives
receipt integrity, freshness, policy eligibility, and attestation assurance.

## Protected assets and data classification

### Public, integrity-sensitive inputs

The following are public commitments or identifiers, not secrets:

- receipt, property, obligation, repository/snapshot, policy, and result-set
  root identities;
- the content-identity profile ID paired with every CID;
- backend, circuit, public-input schema, setup, proving-key, and
  verification-key identities and versions;
- predicate/use-case ID, verifier domain, challenge or nonce, creation time,
  expiry, and proof digest; and
- a boolean redaction marker that reveals no witness name, type, length, or
  value.

They remain security-sensitive because substitution, ambiguity, or omission
can turn a valid proof into a false claim about different inputs.

### Private witness and secrets

A private witness is any value used to satisfy the circuit relation that is not
a declared public input. Depending on the approved predicate, this may be:

- canonical receipt bytes or a secret opening used for proof of possession;
- a Merkle or vector-commitment membership path and private leaf;
- private reviewed inputs to a specifically versioned predicate; or
- prover randomness and secret proving material.

Credentials, tokens, encryption keys, private repository content, unpublished
premises, private paths, and setup toxic waste are secrets even when they are
not part of the mathematical witness.

**No witness, secret, witness field name, value-derived diagnostic, reversible
encoding, or low-entropy unsalted witness hash may enter a proof receipt,
attestation envelope, context capsule, prompt, cache record, event/audit log,
trace, metric label, exception, crash report, or completion reference.**
Commitments are public inputs only when the reviewed predicate defines their
construction and leakage analysis.

## Approved predicate boundary

The approved predicate vocabulary is closed and limited to these three
classes:

1. **Receipt possession** — the prover knows the canonical preimage/opening for
   the public CID of one independently validated, current property-proof
   receipt.
2. **Receipt membership** — the prover knows a valid membership witness placing
   a receipt commitment in the exact public committed result-set root.
3. **Private reviewed predicate** — the prover knows private inputs satisfying
   one explicitly named, versioned, circuit-reviewed relation over a current
   validated receipt and declared public inputs.

“Private reviewed predicate” is not an extensibility escape hatch. Each
relation requires its own review record, circuit and schema pins, leakage
analysis, test vectors, setup decision, and use-case disposition. Free-form
theorems, source text, LLM-generated predicates, arbitrary solver claims, and
“receipt is valid/correct” without an independently validated base receipt are
outside the boundary.

Catalog eligibility does not authorize deployment. The companion policy
requires a concrete private-witness and cross-trust-boundary need before
selecting a real backend.

## Security assumptions

An `ATTESTED` result depends on all of these assumptions:

- the property receipt was independently validated at the required assurance
  and is current for its bound snapshot;
- canonical bytes and each `(identity_profile_id, CID)` pair are recomputed,
  not trusted from serialized claims;
- the reviewed circuit implements the named predicate and constrains every
  public input;
- the proving system's soundness and zero-knowledge assumptions hold at the
  approved security level;
- setup artifacts and verification keys have authentic provenance, are
  unexpired, and match the pinned circuit;
- the verifier is independent of the untrusted prover response;
- secret handling and the proving implementation do not leak through storage,
  diagnostics, timing, memory reuse, or subprocess boundaries; and
- verifier-domain freshness data prevents reuse outside the intended decision.

ZK does not hide public-input values, access patterns, proof timing, proof
existence, or facts logically implied by the predicate. Small witness domains
can leak through public commitments or repeated queries even if the backend is
cryptographically sound.

## Adversary capabilities

Assume an attacker can:

- submit forged, malformed, stale, or cross-snapshot receipts;
- change serialized IDs, CIDs, profile IDs, result roots, policy IDs, backend
  labels, setup pins, timestamps, or assurance flags;
- replay a valid proof to another verifier, use case, snapshot, policy, result
  set, identity profile, or deployment;
- control an attestation prover and optional provider response;
- label simulation output as Groth16 or “verified”;
- poison provider-local and sidecar caches;
- trigger backend absence, timeout, partial capability, capability drift, and
  key expiry;
- cause chosen statements and repeated proofs in an attempt to infer the
  witness; and
- observe public artifacts, logs, prompts, metrics, error paths, proof size,
  and coarse timing.

The attacker is not assumed able to break an approved primitive at its stated
security level, compromise the independent verifier and supervisor policy root
simultaneously, or read correctly isolated prover memory. Compromise of those
roots is a residual operational risk, not something the proof can repair.

## Threats and fail-closed responses

| Threat | Required response |
| --- | --- |
| ZK used instead of property proof | Reject attestation before backend dispatch; report missing or insufficient base proof. |
| Forged receipt or mismatched canonical bytes | Reject on CID/profile/multihash or receipt validation failure. |
| Stale receipt/cache entry | Reject for production/completion; do not attest a historical receipt as current. |
| Predicate/circuit omits a public binding | Predicate version is ineligible; no proof generation or verification. |
| Receipt, property, snapshot, policy, backend, setup, result-root, or identity-profile substitution | Reconstructed public inputs differ, so verification fails. |
| Cross-profile CID confusion | Reject even if digest text appears equal; profile ID is part of the statement. |
| Proof replay | Reject domain, nonce/challenge, freshness, expiry, or exact-binding mismatch. |
| Simulated proof marked verified | Preserve a typed simulated/test result below `ATTESTED`; never admit it to production/completion. |
| Provider claims authority | Ignore the claim; authority is derived only by the supervisor from independent verification. |
| Backend/setup/key drift | Capability report or pin mismatch makes the backend ineligible; no simulated fallback. |
| Malformed or malleable proof | Reject with a non-secret diagnostic code and no partial authority. |
| Witness in a public boundary | Abort, discard the artifact, raise a disclosure error, and do not retry through logging or a fallback. |
| Low-entropy commitment enables guessing | Reject the predicate during leakage review or require a hiding commitment with adequate randomness. |
| Prover crash leaves witness material | Treat as backend failure; clean bounded ephemeral state and retain only public failure metadata. |
| Backend unavailable or times out | Emit typed unavailable/error below `ATTESTED`; keep the base property receipt unchanged. |
| Unsupported use case or predicate | Return terminal `not_applicable`; do not select a backend, simulate, or search for a looser predicate. |

## Replay and public-input binding

The verifier reconstructs the canonical public statement. It must bind:

- `use_case_id`, `predicate_id`, predicate/circuit version, and verifier domain;
- receipt CID and profile ID;
- property/obligation CID and profile ID;
- repository snapshot/root CID and profile ID;
- attestation-policy CID and profile ID;
- backend-policy and implementation CID/profile/version;
- setup manifest, circuit, public-input schema, and verification-key
  CID/profile/version;
- committed result-set root CID and profile ID, or a domain-separated
  not-applicable sentinel for a non-membership predicate;
- challenge/nonce, issued-at, expires-at, and any approved revocation epoch; and
- proof/envelope schema version.

The proof must constrain every field directly or constrain one canonical
domain-separated digest of the complete field set. Open maps, implicit
defaults, display names, ambient configuration, and caller-provided
`verified`/`authoritative` flags are prohibited.

A proof is reusable only when the policy explicitly permits reuse and every
binding remains identical and current. Verification-key expiry, policy
revision, capability drift, snapshot change, result-root change, or receipt
invalidation ends reuse.

## Setup and backend boundary

Backend availability is not backend qualification. A real backend is usable
only with a reviewed use case and a current capability report bound to the
backend policy. Qualification includes golden, negative, stale-key,
malformed-proof, replay/substitution, and witness-no-leak tests.

For trusted-setup systems, the setup manifest must identify the ceremony,
contributors or trust assumption, transcript, circuit, proving and
verification keys, artifact digests, creation/expiry or rotation policy, and
destruction/handling of toxic waste. For transparent systems, the manifest
must still pin parameters and security assumptions. Changing setup artifacts
creates a new policy identity and invalidates prior eligibility unless an
explicit verification migration is reviewed.

The current datasets `zkp_attestation` bridge defaults to a simulated backend
and may call its result `verified` within that bridge. Such a result validates
demo structure only. It is an observation/test artifact, not cryptographic
authority, and is permanently below `ATTESTED`. Enabling a datasets Groth16 or
ProveKit path is likewise insufficient without this policy's use-case,
capability, setup, and independent-verification gates.

## Leakage controls

- Construct the witness only inside the prover boundary after every public
  precondition passes.
- Use a non-serializable, redacted witness holder and pass a read-only view
  directly to the prover.
- Do not place a proving request containing a witness in CAS or any cache.
- Disable payload-level tracing for the prover boundary and allowlist public
  diagnostic codes.
- Use bounded ephemeral files or memory; restrict permissions and remove them
  on success, error, cancellation, and timeout.
- Zeroize mutable buffers when the runtime permits. Because immutable Python
  objects cannot guarantee erasure, avoid materializing secrets as strings and
  isolate native/subprocess proving when stronger erasure is required.
- Run witness-no-leak fixtures against receipts, envelopes, context capsules,
  logs, cache records, exceptions, metrics, and subprocess output.
- Rate-limit chosen-statement and repeated-proof access where predicate
  inference is possible.

## Assurance and residual risk

`ATTESTED` is derived only when the base receipt remains valid, the use case is
approved, the backend is real and capability-qualified, all public bindings
match, the proof is independently verified, and freshness holds. Simulation,
unavailability, rejection, errors, and pending review add no assurance and
cannot lower or corrupt the separately derived property-proof assurance.

Even a valid attestation does not prove the source snapshot is benign, the
reviewed property is sufficient, the circuit has no review error, the
cryptographic implementation has no vulnerability, or the endpoint will obey
the contract in the future. Those risks remain with contract coverage,
property review, implementation audit, key operations, and runtime policy.

## Review triggers

Re-review this model before:

- adding a predicate class or changing a predicate relation;
- introducing a new verifier domain or cross-organization consumer;
- changing canonicalization or any identity profile;
- changing backend family, circuit, setup, key, public-input schema, or
  security level;
- allowing proof reuse, aggregation, recursion, or on-chain verification; or
- changing witness transport, logging, caching, subprocess, or retention
  behavior.
