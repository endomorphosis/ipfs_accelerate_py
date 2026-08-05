# Proof-Backed Test Reuse: ZK Threat Model and Authority Doctrine

Status: normative security doctrine
Contract: `ZkThreatModel@1`
Statement: `TestPassStatementV1`
Decision actions: `RUN` or `SKIP`

## Purpose and security objective

This document defines the authority boundary for proof-backed reuse of a prior
pytest pass. The optimization is allowed to avoid an execution only when the
supervisor can independently establish that one exact, trusted prior pass
receipt applies to the exact execution that would otherwise run now.

The security objective is **zero stale or false authoritative skips**. A false
miss only spends time by running a test. A false hit suppresses evidence that
the test runner was asked to produce, so `SKIP` is a fail-closed allow decision
and `RUN` is the safe default.

This model inherits the codebase-proof rule that observations do not become
proof merely by being serialized, cached, content-addressed, signed, or
presented by a provider. It specializes that rule for pytest outcomes and does
not promote an AST index, runtime trace, CID, cache hit, signature, or prover
claim into pass authority.

## Non-negotiable doctrine

The following sentences are normative and intentionally direct:

1. **ZK proves possession of the exact trusted pass receipt; it does not prove
   that changed code passes or repair an untrusted receipt.**
2. **AST similarity never means pass.** AST analysis can only identify and
   invalidate dependencies under an admitted completeness policy.
3. **A CID only identifies bytes.** It says nothing about whether those bytes
   describe a pass, are trusted, are fresh, or are applicable to the current
   execution.
4. **Simulated ZK never skips.** Mock, demo, educational, and simulated proofs
   are permanently non-authoritative, even if an adapter labels them
   `verified` or their bytes match an expected fixture.
5. **Every uncertainty executes the test.** Missing, unknown, ambiguous,
   unavailable, unsupported, malformed, corrupt, stale, expired, revoked,
   incomplete, over-budget, timed-out, or exceptional states resolve to
   `RUN`, never `SKIP`.

A valid ZK proof is narrow evidence about a pinned relation. For
`TestPassStatementV1`, its claim is that the prover possesses the canonical
preimage of the public receipt CID and that the constrained fields in that
receipt match the exact public statement. It is not a proof of general program
correctness, future behavior, dependency-trace completeness, deterministic
execution, or semantic equivalence between two versions of a test.

## Scope and non-goals

This threat model covers:

- immutable `TestPassReceipt@1` bytes and their strict CID profile;
- `TestExecutionKey@1`, trace-completeness, outcome, policy, issuer, epoch,
  circuit, backend, and verification-key bindings;
- real and simulated ZK certificate production and local verification;
- mutable lookup indexes and immutable local or shared content stores;
- reuse decisions made during pytest collection; and
- diagnostics, audit records, caches, subprocess boundaries, and deferred
  proving work that could disclose a private receipt witness.

It does not claim to prove arbitrary source correctness, infer pass status from
source similarity, make an incomplete trace complete, make uncontrolled
external effects deterministic, or turn availability of a proving service
into authority. It does not authorize reuse of ordinary skips, xfail, xpass,
rerun-only success, interrupted runs, incomplete setup or teardown, coverage
runs, benchmarks, mutation runs, debugger sessions, or leak-detection runs.

## Actors and trust boundaries

| Actor | Responsibility | Authority it does not possess alone |
| --- | --- | --- |
| Prior pytest runner | Execute setup, call, and teardown and propose a canonical pass receipt | It cannot authorize a future skip merely by reporting `passed`. |
| Receipt issuer / trust policy | Authenticate an admitted runner and its receipt commitment | A signature cannot establish current execution identity or trace completeness. |
| ZK prover | Hold the private receipt witness and produce a proof for exact public inputs | It cannot assert that its proof is valid, current, real, or policy-admitted. |
| Independent local verifier | Reconstruct public inputs and verify a pinned real proof | It cannot repair an untrusted receipt, wrong execution key, or incomplete trace. |
| Reuse policy gate | Recompute identities, validate receipt/certificate/trust/revocation, and choose `RUN` or `SKIP` | It must not trust mutable index entries, provider labels, or serialized authority flags. |
| Mutable locator index | Return a bounded list of candidate certificate CIDs | It is a hint and has no pass or skip authority. |
| Immutable CAS / IPFS transport | Retain or transport exact bytes | Content presence and CID equality do not establish pass authority. |
| AST and runtime tracers | Describe a bounded dependency/effect frontier for invalidation | Similarity and trace observations never establish that a test passed. |

The authoritative verifier and reuse policy execute locally. A prover, cache,
index, CAS gateway, datasets bridge, IPFS peer, or remote service is outside
that trust boundary. Existing accelerator trust-aware proof-cache and evidence
contracts remain the memoization trust root; test reuse does not create a
second root. Every candidate hit re-derives authority from retained bytes and
current local policy.

## Protected assets and data classification

### Public, integrity-sensitive bindings

The verifier reconstructs public inputs rather than accepting them from the
prover. The complete, domain-separated public statement binds:

- statement schema and version (`TestPassStatementV1`);
- proof-system, backend implementation, and backend-policy identities;
- circuit, public-input schema, setup manifest, and verification-key CIDs and
  versions;
- receipt CID and its canonical content-identity profile;
- exact current execution-key CID and locator CID;
- outcome bits requiring setup, call, and teardown to pass and every
  disqualifying bit to be clear;
- static/runtime trace roots, completeness-policy CID, and a positive
  completeness result;
- reuse-policy CID and version;
- runner trust domain, issuer/key commitment, and revocation epoch;
- repository-forest, dependency, environment, and capability roots admitted by
  the current execution key; and
- verifier domain, nonce or challenge, issued-at time, expiry, and allowed
  epoch data.

These fields are public but security-sensitive. Omitting or ambiguously
encoding one can allow a proof for one test, repository, circuit, key, policy,
issuer, or epoch to be substituted for another. Every field must be directly
constrained or included in one canonical domain-separated digest constrained
by the reviewed circuit. Open maps, display names, implicit defaults, ambient
configuration, and prover-supplied `verified`, `real`, or `authoritative` flags
are forbidden at the authority boundary.

### Private witness and secrets

The minimum private witness is the canonical trusted receipt bytes, or a
reviewed hiding opening to those exact bytes, needed to prove receipt
possession. Prover randomness and secret proving material are also private.
Test secrets, credentials, unrestricted environment values, source bodies,
private paths, captured stdout/stderr, and arbitrary fixture values are not
made acceptable merely by calling them witness fields.

No private witness, secret, witness field name, reversible encoding,
value-derived diagnostic, low-entropy unsalted witness hash, or private path
may enter a public certificate, receipt index, CAS metadata record, prompt,
trace, log, event, metric label, exception, crash report, or pytest skip
reason. Receipt witness bytes are constructed only inside the prover boundary
after public preconditions pass, are never sent to the verifier, and are never
written to a proving-request cache.

## Authority lattice and allow decision

The lattice is deliberately non-promotional. Lower layers can reject a
candidate, narrow invalidation, or supply an input to a higher check; no lower
layer alone can authorize `SKIP`.

| Layer | Examples | Permitted effect |
| --- | --- | --- |
| Observation | AST similarity, runtime trace, provider response, cache/index hit | Diagnose, locate candidates, or force `RUN`; never establish pass. |
| Content identity | Valid CID over retained canonical bytes | Establish exact byte identity only; never establish meaning, trust, freshness, or pass. |
| Candidate receipt | Structurally valid receipt that claims pass | Continue validation or choose `RUN`; never skip before issuer, outcome, trace, identity, and policy checks. |
| Trusted exact pass receipt | Canonical receipt admitted by runner/issuer policy with complete passing phases | Eligible input to certificate validation; still not a skip by itself. |
| Non-attested certificate | Simulated, mock, provider-asserted, unavailable, or unverified result | Exercise adapters or report diagnostics; always `RUN`. |
| Verified real certificate | Cryptographic proof verified locally against pinned exact public inputs | Evidence of possession of the exact trusted receipt; still subject to all current bindings and revocation checks. |
| Authoritative reuse decision | All required checks below succeed conjunctively in the same local decision | `SKIP`; no other state may choose it. |

`SKIP` is authorized if and only if all of the following are true at decision
time:

1. retained canonical receipt and certificate bytes independently reproduce
   their declared strict CIDv1/base32/dag-json/sha2-256 identities;
2. the receipt is schema-valid, trusted under the admitted runner/issuer
   policy, unrevoked, unexpired, and records complete passing setup, call, and
   teardown with no disqualifying outcome;
3. the receipt execution-key CID equals the freshly recomputed exact current
   execution-key CID;
4. the static/runtime trace policy is admitted and reports complete, with all
   required dependency, repository-forest, environment, and capability roots
   bound;
5. the reuse policy, statement, circuit, setup, backend, public-input schema,
   verification key, trust domain, nonce/domain, and epoch are exact, current,
   approved, and mutually compatible;
6. the certificate was produced by a real approved backend and its proof is
   independently verified by the local pinned verifier over reconstructed
   public inputs; and
7. bounded verification completes without absence, ambiguity, timeout,
   exception, or resource-limit failure.

This is a conjunction, not a score or majority vote. Serialized action fields
are ignored and the action is re-derived. Failure or uncertainty in any item
chooses `RUN` with a bounded, non-secret reason code.

## `TestPassStatementV1` claim boundary

The reviewed circuit relation must establish all of the following:

1. the private canonical receipt bytes hash under the declared content profile
   to the public receipt CID;
2. constrained receipt fields equal the public execution-key, locator, policy,
   trace, issuer, and epoch bindings;
3. setup, call, and teardown outcome bits are all pass;
4. skipped, xfailed, xpassed, rerun-only, interrupted, timed-out,
   leaked-resource, incomplete-trace, and other disqualifying bits are clear;
5. the trace-completeness identifier and positive result match the public
   admitted completeness policy; and
6. the issuer signature or commitment satisfies the pinned runner trust
   relation.

The supervisor validates receipt trust and recomputes the current execution
key outside the circuit as independent prerequisites. The circuit cannot
declare its own trusted issuer, circuit identity, verification key, or public
inputs. ZK privacy hides only the private witness values promised by the
reviewed proving system; it does not hide public inputs, proof existence,
timing, access patterns, or facts logically implied by the statement.

## Adversary capabilities and assumptions

Assume an attacker can control a prover and a remote provider, publish hostile
CAS/index entries, alter serialized fields and authority labels, replay valid
old artifacts, choose malformed or oversized inputs, interrupt writes, cause
timeouts and optional-dependency failures, observe public artifacts and coarse
timing, and attempt to induce secret-bearing diagnostics. The attacker may
present a proof from another test, receipt, execution key, repository forest,
policy, issuer, circuit, setup, verification key, backend, verifier domain, or
epoch. The attacker may label deterministic fixture bytes as a real proof.

Authority assumes that approved cryptographic primitives remain sound, the
local verifier and policy root are not both compromised, the current execution
identity compiler faithfully binds every component required by the admitted
policy, trusted runner keys are protected and revocable, and witness isolation
works as reviewed. Compromise of these roots is residual operational risk; ZK
cannot repair it.

## Threats and required fail-closed responses

| Threat | Attack | Required response |
| --- | --- | --- |
| Receipt forgery | Supply invented pass fields, a provider signature, or parsed fields that do not match retained bytes. | Recompute canonical bytes and CID, validate schema and issuer trust; on any mismatch choose `RUN`. |
| Receipt substitution | Prove possession of a valid receipt for another node, parameter, execution key, or repository forest. | Reconstruct exact public inputs and require receipt/execution/locator/root equality; otherwise `RUN`. |
| Proof replay or rollback | Reuse a once-valid proof across nonce, verifier domain, policy, revocation epoch, expiry, or changed execution. | Bind and validate domain, challenge, time, epoch, policy, and exact current key; stale or mismatched artifacts choose `RUN`. |
| AST similarity promoted to pass | Claim similar or unchanged syntax means behavior passed. | Treat AST only as invalidation input. Similarity has no pass authority and chooses no action other than continued checking or `RUN`. |
| Trace incompleteness | Hide a dynamic import, fixture, hook, data read, subprocess, network call, overflow, or unsupported event. | Require a positive completeness receipt for the admitted class; unknown frontier, overflow, or unsupported instrumentation chooses `RUN`. |
| CID semantic confusion | Present a correct CID for forged, stale, simulated, or irrelevant bytes, or a legacy pseudo-CID. | Validate strict profile and retained bytes, then separately validate semantics and trust; absence or mismatch chooses `RUN`. |
| Cross-profile CID confusion | Reuse digest text under another codec, base, multihash, version, or canonicalization rule. | Bind the complete identity profile and independently decode/re-hash; ambiguity or legacy formats choose `RUN`. |
| Circuit or verification-key confusion | Verify a valid proof with the wrong circuit, public-input schema, setup, backend, or key. | Pin all identities and versions in policy and public inputs; any drift, expiry, or incompatibility chooses `RUN`. |
| Issuer or trust-domain confusion | Accept a receipt signed by an untrusted, revoked, or wrong-domain runner. | Re-derive issuer policy, domain, key, and revocation epoch locally; any uncertainty chooses `RUN`. |
| Public-input omission | A circuit leaves a security-relevant field unconstrained or accepts an implicit default. | Mark the circuit/policy ineligible; do not dispatch proving or verification and choose `RUN`. |
| Simulated-backend mislabeling / downgrade | Label mock output as Groth16/ProveKit, or fall back to simulation after real verification fails. | Derive backend mode from pinned local capability/policy. Simulated ZK never skips; real-path failure remains `RUN` with no fallback promotion. |
| Provider authority spoofing | Set `verified=true`, `passed=true`, `authoritative=true`, or serialized action `SKIP`. | Ignore asserted authority and locally re-derive the decision; malformed or insufficient evidence chooses `RUN`. |
| Malformed, oversized, or malleable proof | Exhaust the verifier or exploit permissive decoding. | Enforce byte/time/shape bounds and canonical encodings before verification; reject safely and choose `RUN`. |
| Witness leakage | Place receipt bytes, secrets, private paths, fixture values, or secret-derived details in public artifacts or diagnostics. | Abort and discard the artifact, emit only a bounded non-secret code, clean ephemeral state, and choose `RUN`. |
| Cache/index poisoning | Point a locator at hostile, partial, corrupt, path-escaping, or excessive candidates. | Treat the index as a bounded hint, rehash immutable bytes, reject/quarantine safely, and choose `RUN`. |
| Mutable-state race | Revoke a key/policy or alter current state during lookup and verification. | Use one coherent decision epoch, recheck freshness/revocation before action, and choose `RUN` on a race or contradiction. |
| Optional capability failure | Remove or break multiformats, datasets ZK, CAS, IPFS, verifier, key, or issuer support. | Catch defined absence and ordinary boundary exceptions; every uncertainty executes the test (`RUN`). |
| Verification timeout or resource exhaustion | Delay a cache, parser, or verifier beyond its bounded budget. | Stop reuse evaluation, record a bounded diagnostic, and choose `RUN`; never delay normal execution indefinitely. |

## Replay, substitution, and freshness

Certificate reuse is permitted only when policy explicitly allows it and every
public binding remains identical and current. The verifier must rebuild the
statement from the current item and retained canonical artifacts, not from an
index record or prover-provided statement. A repository, dirty overlay,
parameter, fixture, hook, dependency, config, environment, capability, trace,
policy, circuit, key, issuer, or epoch change ends reuse.

Time alone is not freshness. Freshness combines exact current execution
identity, policy/key/issuer validity, revocation epoch, and an allowed
nonce/domain/time policy. Rollback protection must prevent an older but once
valid mutable index or policy snapshot from restoring authority.

## Trace completeness and semantic similarity

Static AST and runtime traces reduce the invalidation set only after their
completeness class is reviewed. They are not positive outcome evidence. An
identical AST can run under different fixtures, hooks, imports, native code,
data, environment, hardware, clock, randomness, subprocesses, services, or
parameters. Different ASTs may be semantically similar and still expose a
regression. Therefore no similarity threshold, embedding score, model verdict,
runtime overlap, or unchanged-line heuristic participates in the `SKIP`
authority calculation.

The first production policy binds the complete admitted repository forest.
Narrower roots require separate mutation and trace-completeness evidence.
Dynamic import, reflection, native extension, opaque decorator, unresolved
fixture, uncontrolled external effect, trace overflow, instrumentation loss,
or secret-dependent behavior is an explicit unknown and resolves to `RUN`.

## Backend qualification and downgrade resistance

Availability is not qualification. A real backend is eligible only when local
policy approves its exact family and version, the circuit and public-input
schema are reviewed, setup and verification-key provenance is pinned, keys are
current, and golden, negative, replay, substitution, stale-key,
malformed-proof, and witness-no-leak fixtures pass. Independent local
verification is mandatory.

A simulated backend may test serialization, fixtures, diagnostics, and adapter
control flow. Its authority is `non_attested` and cannot be upgraded by a
provider flag, signature, cache admission, successful round trip, or backend
unavailability. Failure of a real prover or verifier does not retry through a
simulated backend. It records a non-secret diagnostic and executes the test.

Proof issuance is deferred until after an eligible complete pass. Prover
absence or failure never changes that original pytest result and never blocks a
later normal test execution. An existing real certificate may be reused while
the prover is unavailable only if all local verification, identity,
freshness, and policy checks still succeed.

## Witness leakage controls

- Complete every public eligibility and trust check before constructing the
  private witness.
- Keep the witness in a redacted, non-serializable holder and pass it directly
  to the prover through a bounded interface.
- Disable payload tracing and body logging around proving; allowlist public
  reason codes.
- Use isolated bounded memory or permission-restricted ephemeral files and
  clean them on success, rejection, timeout, cancellation, and crash recovery.
- Never cache a proving request containing private material. Cache only
  canonical public artifacts after a recursive private-material scan.
- Do not expose witness length, field names, values, private paths, or
  value-dependent errors. Apply size and timing controls where the reviewed
  leakage analysis requires them.
- Treat any suspected disclosure as an artifact-invalidating security event:
  discard it, revoke affected material when applicable, and execute the test.

## Decision procedure and audit contract

For each collected pytest item, the policy gate performs a bounded lookup and
returns exactly one typed action:

```text
recompute current execution key
  -> no bounded candidate                         => RUN
  -> candidate bytes/CID/schema/trust invalid     => RUN
  -> receipt not exact, current, complete pass    => RUN
  -> trace/policy/issuer/key/circuit uncertain    => RUN
  -> proof simulated/unverified/malformed         => RUN
  -> local real-proof verification fails/unknown  => RUN
  -> every exact check succeeds                   => SKIP
```

There is no implicit truthy result and no `UNKNOWN -> SKIP` transition. The
plugin catches optional-boundary failures, records bounded reason codes, and
runs the test normally. A strict audit job may separately report degraded
capability after collection, but it must not manufacture a skip.

An authoritative audit record names the receipt and certificate CIDs, exact
execution key, policy, issuer/trust epoch, circuit and verification key,
independent verifier result, decision epoch, and bounded reason. It contains no
witness. pytest reports an authorized hit using the standard skip mechanism
and a bounded `proof-cache-hit:<certificate-cid>` reason.

## Required validation population

The doctrine is not complete without negative tests for forged receipt/proof
and authority flags; receipt, execution-key, test-node, parameter, repository,
policy, issuer, circuit, key, and epoch substitution; replay and rollback;
stale, expired, and revoked artifacts; trace gaps and overflow; strict CID
profile confusion; malformed and oversized inputs; partial writes and index
poisoning; witness strings across every public surface; simulated-backend
mislabeling and crypto-to-simulation downgrade; missing optional dependencies;
timeouts and exceptions; and concurrent revocation or publication races.

All implementation tests for proof-backed reuse must themselves run with
`IPFS_TEST_PROOF_REUSE_MODE=off`. The feature may never use its own cached
claim to validate the code that decides whether cached claims are authoritative.

## Residual risks and review triggers

Residual risks include compromise of the trusted runner or local policy root,
bugs in the identity compiler or reviewed circuit, cryptographic failure,
side-channel leakage beyond the qualified backend model, and an admitted
completeness policy that omits a real dependency. Conservative repository-
forest binding, forced-execution sampling, mutation populations, metrics with
zero tolerated false skips, key/policy revocation, and an immediate `off` mode
limit these risks but do not make them disappear.

Any new eligibility class, narrower dependency root, content-identity profile,
statement version, circuit, backend family, setup, verification-key policy,
issuer trust domain, external snapshot adapter, public input, or witness shape
requires security review and new negative vectors. Until that review is
admitted, the changed or unknown configuration resolves to `RUN`.
