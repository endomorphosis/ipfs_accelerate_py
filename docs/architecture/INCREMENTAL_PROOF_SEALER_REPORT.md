# Incremental Proof Sealer Terminal Report

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

This is the current-tree fan-in report for IncrementalProofSealer.
Schema `incremental-proof-sealer-release-validation@2` and log policy
`public-full-log-secret-scan@1` are bound by the protected runner. The
receipt digest and exact source commits are substituted into the release
evidence marker above when materialization completes; they are not
provider-authored.

## What was observed

Live ipfs was refused before release suite execution.
Pytest process outputs were observed but test execution was not
cryptographically proven. Three new incremental-sealing suites require
fully green execution. Repository verification was decomposed into
content-addressed proof units. Stale or simulated evidence is never
treated as current verification.

The protected convergent ensure observes the historical terminal board
gate, the seventeen reviewed existing ZK/reuse/WAL/release suites, and
the three current-tree incremental-sealing suites
(`accelerate-incremental-sealing`, `datasets-incremental-sealing`,
`kit-proof-seal-store`). Existing suites are accepted only as green or
baseline-compatible-or-improved against their exact protected operator
receipts. Every retained skip, xfail, deselection, failure, error, xpass,
or collection issue is labeled `baseline_compatible_non_green` and named
in the release-evidence marker, never hidden as success.

## Systems and claims

Existing ZK systems are classified as real proving, simulated, or
structural validation. Direct execution proof is distinct from trusted
signed receipts and integrity commitments. Proof-unit granularity uses
a complete cache key, invalidation rules, full-proof fallback, and
Merkle manifest aggregation.

Ownership of modules follows the sealed three-repository boundary:
datasets owns proof semantics, identity, manifests, and invalidation;
kit owns storage, index, forest, WAL, and CAS; accelerate owns proving,
planning, scheduling, aggregation, sealing, and measurement. Public APIs
and CLI surfaces stay within those boundaries.

Claim classes:

- direct execution proof: deterministic computation for one proof unit
- trusted signed receipts: allowlisted signer assertions, not independent
  proof that tests ran without trusting the signer
- integrity commitments: exact-byte bindings that do not establish correct
  execution
- receipt-aggregation and Merkle manifest statements: child-identity
  completeness, not recursive verification of underlying tests
- simulated plumbing and structural validation: never production seal
  authority

## Modules and rules

Primary modules inspected across the board include proof attestation,
proof scheduler and resource router surfaces, proof-reuse identity and
cache adapters, datasets CEC/TDFOL/FLogic ZKP integrations and proof
caches, kit proof certificate store, modern WAL, Profile D policy, and
the incremental sealing / proof-seal-store suites under current tree.

Proof-unit granularity is the closed `ProofUnit@1` / `ProofCacheKey@1`
identity. Complete cache key fields bind statement, circuit, verification
key, public inputs, and parent seal identity. Invalidation rules require
a full checkpoint on canonicalization, circuit, or verification-key
change and on cache corruption. Full-proof fallback recomputes the unit
when reuse is denied. Aggregation is Merkle manifest aggregation unless
a recursive backend capability probe succeeds.

## Benchmark

The 40-transition benchmark reports average proof reuse rate, average
proving-compute reduction, best incremental case, worst incremental
case, proof size, seal size, verification latency, and storage overhead.
Provenance-labeled summary values from the bound summary artifact
(`incremental-proof-sealer-benchmark-summary@1`, raw digest
`sha256:f33fbdb928f523e11e75d22cc2808594bcb1e2fb83b9fcf7837d49608afa2de6`)
are mixed (estimated CPU/size with unavailable GPU), not measured wall
clock from a production prover:

- average proof reuse rate: 43.6161% (mixed)
- average proving-compute reduction: 43.0539% (mixed)
- best incremental case: transition 34 ordinary documentation edit,
  98.9011% compute saved (mixed)
- worst incremental case: transition 0 initial repository, 0.0% compute
  saved (mixed)
- proof size: mean 18022.4 bytes (estimated)
- seal size: mean 3980.8 bytes (estimated)
- verification latency: mean 0.00132 seal verification seconds
  (estimated)
- storage overhead: mean 18886.4 storage growth bytes (estimated)

Targets are not facts: localized 70% reuse and mixed 50% compute
reduction remain unmet; documentation 80% compute reduction is met under
mixed provenance only.

## Adversarial and durability results

Crash-recovery results and tamper-test results are recorded on the
adversarial board. Required negative, wrong-parent, concurrent-writer,
and seven-phase crash cases must pass with zero stale or simulated
acceptance. Publication uses compare-and-swap; recovery is WAL-driven;
ambiguous external prover outcomes are not success.

## Remaining work before production use

Remaining work before production use includes a production prover,
measured (not only estimated) costs, recursive verification where
claimed, and operational key ceremony. Observed process evidence is not
an entire-repository correctness proof, not zero-knowledge proof of
pytest execution, and not a semantic-correctness certificate for code
changes beyond the bound receipt and log.
