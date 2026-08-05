# Automatic proof-reuse runtime activation contracts

`ProofReuseActivationContract@1` seals the fail-closed composition boundary for
automatic proof-backed pytest reuse. It sits above the existing execution
identity, pass-receipt, certificate, and lookup contracts, and below the
session-scoped factories that later wire identity services, candidate-context
storage, revalidation, deferred issuance, and the pytest plugin.

This module does not authorize skips, import pytest, open a network socket, or
install packages. Importing it is side-effect free.

## Artifact roles (non-interchangeable)

| Role | Interface | Authority |
| --- | --- | --- |
| Locator hint | `LocatorHint@1` | Mutable retrieval narrowing only. Points at retained bytes. Never authorizes `SKIP`. |
| Immutable candidate context | `CandidateExecutionContext@1` | Exact pass-time execution key, AST/static/runtime traces, forest, environment, policy, and receipt identities. Content-addressed. Never authorizes `SKIP` alone. |
| Fresh current context | `CurrentExecutionContext@1` | Rebuilt from live source/AST/fixtures/locks/environment/policy. Historical traces cannot be relabeled as current. Never authorizes `SKIP` alone. |
| Trusted pass receipt | `TrustedPassReceiptBinding@1` | Binding to an admitted complete-pass receipt after one real setup/call/teardown. Eligible input to deferred issuance and certificates; not a skip by itself. |
| Deferred proof request | `DeferredProofRequest@1` | Public-only issuance envelope. No private witness material. Missing proving capability yields `DEFERRED`/`RUN` while retaining the receipt. |
| Authoritative certificate | `AuthoritativeCertificateBinding@1` | Only role that may authorize `SKIP`, and only after exact current-context comparison plus local verification of a real cryptographic certificate. Simulated certificates are permanently non-authoritative. |

These roles are sealed by `ArtifactRole`. A locator hint cannot be promoted to a
candidate context, a prior runtime trace cannot be promoted to a current
context, and a deferred request cannot become an authoritative certificate.

## Content-addressed boundaries

At every content-addressed boundary the contract requires:

1. retained **exact canonical DAG-JSON bytes**;
2. **re-canonicalization** equality (pretty-printed or reordered JSON is rejected);
3. **CID rehash** to CIDv1 / lowercase base32 / dag-json / sha2-256; and
4. agreement between the claimed CID and the rehashed CID.

Helpers:

* `rehash_retained_canonical_bytes(data)`
* `admit_content_addressed_boundary(role, claimed_cid, canonical_bytes)`
* `require_content_addressed_boundary(...)` (strict, raises on miss)

Admission failures surface typed non-admitted results. They never become skip
authority and must not abort pytest collection.

## Authority sequence

The sealed sequence is fixed (`ACTIVATION_AUTHORITY_SEQUENCE`):

1. Compute a stable locator from the collected item and session identity.
2. Resolve a bounded candidate descriptor from the mutable index (hint only).
3. Load retained canonical candidate bytes and rehash every CID.
4. Rebuild the current admitted dependency frontier from live state named by the candidate.
5. Require exact comparison plus local verification of a real, exactly bound certificate before emitting `proof-cache-hit:<cid>`.
6. Otherwise execute setup/call/teardown exactly once.
7. On terminal pass, record post-pass runtime observations and the receipt **without** re-invoking the test body.
8. Request deferred proof issuance with public-only envelopes.
9. Publish candidate/certificate state atomically from the controller.

This ordering prevents both circular runtime-key prediction and duplicate test
execution. Historical AST/runtime traces may narrow revalidation work but never
assert that the current test passes.

## Pre-SKIP comparison

Before `SKIP`, `compare_contexts_for_skip` requires exact agreement on:

* AST (`test_ast_cid`)
* static dependency frontier (`static_trace_root_cid`)
* runtime dependency frontier (`runtime_trace_root_cid`)
* environment (`environment_cid`)
* policy (`policy_cid`)

plus locator, execution-key, and repository-forest identities. Incomplete,
unresolvable, or changed dimensions return `RUN`.

`ProofReuseActivationContract.evaluate_skip_admission` composes rehash,
comparison, and certificate authority into a single disposition. Simulated or
non-attested certificates always produce `RUN`.

## Post-pass runtime observations

`PostPassRuntimeObservation@1` / `record_post_pass_runtime_observation` capture
the observed runtime frontier after the single real lifecycle:

* `observation_source` must be `post_pass_lifecycle`
* `duplicate_test_call_forbidden` must be true
* `test_call_count`, `setup_call_count`, and `teardown_call_count` must each be exactly `1`

Warm admission must not run the test once to predict whether it can skip. Cold
passes execute once, record the frontier afterward, and retain immutable
candidate context for later comparison.

## Optional capability degradation

`disposition_for_optional_capability_fault` maps every fault class:

* `missing`
* `malformed`
* `incompatible`
* `timed_out`
* `exceptional`

to `RUN` or `DEFERRED`. Valid dispositions always set `collection_failed=False`.
`SKIP` is unreachable from this path. When a trusted pass receipt was retained
and proving infrastructure is missing, incompatible, or timed out, the preferred
disposition is `DEFERRED` so issuance can proceed later without re-running
collection or losing the receipt.

## Dispositions

`RuntimeReuseDisposition@1` is the activation-boundary action set:

| Action | Meaning |
| --- | --- |
| `RUN` | Execute the test. Safe default for absence, mismatch, integrity failure, and uncertainty. |
| `SKIP` | Authoritative proof-cache hit after exact current comparison and local verification. |
| `DEFERRED` | Receipt retained; certificate issuance postponed. Test already passed or continues as RUN for the caller’s phase. |

## Relationship to later repair tasks

| Task | Builds on this contract |
| --- | --- |
| PTR-134 | Session-scoped default identity services producing locators and current static components |
| PTR-135 | Immutable candidate-context store that rehashes retained bytes |
| PTR-136 | Runtime revalidation comparing candidate vs current contexts |
| PTR-137 | Typed deferred certificate requests and lazy issuers |
| PTR-138+ | Plugin composition and repository bootstrap using the sealed sequence |

## Non-goals

* No per-test path registry or item attribute as skip authority
* No environment flag that bypasses certificate verification
* No mutable locator index as authority
* No historical runtime trace accepted as current evidence
* No simulated ZK, cache presence, installer success, or bootstrap success as authority
