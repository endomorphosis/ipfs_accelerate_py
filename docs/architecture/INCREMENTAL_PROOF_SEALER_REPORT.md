# Incremental Proof Sealer Terminal Report

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

This is the current-tree fan-in report for IncrementalProofSealer.
Schema `incremental-proof-sealer-release-validation@2` and log policy
`public-full-log-secret-scan@1` are bound by the protected runner. The
narrow claim is: repository verification was decomposed into
content-addressed proof units; unchanged units were reused only when
their complete dependency and trust context remained unchanged;
invalidated units were re-proven; affected Merkle branches were
updated; and a new seal was generated from an accepted parent seal.
Stale or simulated evidence is never treated as current verification.

## What was observed

Live ipfs was refused before release suite execution.
Pytest process outputs were observed but test execution was not
cryptographically proven. Three new incremental-sealing suites require
fully green execution. Existing reviewed ZK/reuse/WAL/release suites
must be green or baseline-compatible-or-improved; every retained skip,
xfail, deselection, failure, error, xpass, or collection issue is
labeled `baseline_compatible_non_green` and named here when retained.

Protected operator baselines show non-green existing suites that the
release runner must either improve or retain as
`baseline_compatible_non_green` without hiding them as success:
`accelerate-proof-focused-core-15`,
`accelerate-proof-focused-wide-36`,
`accelerate-proof-reuse-migration`,
`accelerate-proof-reuse-cross-repo`,
`datasets-zkp-focused-current`,
`datasets-zkp-unit-wide-current`,
`datasets-proof-cache-adapters`,
`datasets-zkp-broad-safe-current`.
The protected runner's single marker substitution lists the actual
retained IDs after current-tree observation.

## Systems, tests, and claim classes

Existing ZK systems are classified as real proving, simulated, or
structural validation:

- **Real proving (bounded):** datasets arkworks Groth16 backends prove
  declared statements only (v1 commitment, v2 bounded Horn TDFOL
  derivation, v3 event digest/root/count). They do not prove pytest
  execution.
- **Simulated / structural:** wallet/PDF simulated paths, structural
  validation helpers, and test plumbing fixtures. Simulated required
  units cannot produce a production seal.
- **Integrity / transport:** kit `proof_certificate_store` is exact-byte
  CID transport and integrity commitments, not cryptographic verification
  or reuse authority. Event-DAG Merkle helpers and pseudo-CID utilities
  are not proof-seal authorities.
- **Signed assertions and receipts:** trusted signed receipts and
  receipt-aggregation statements bind admitted fields only; they are not
  direct execution proof of tests.
- **Direct execution proof:** direct execution proof remains distinct
  from trusted signed receipts and integrity commitments. No inspected
  kit suite performs real proving of test execution; recursion is
  unsupported and defaults to Merkle manifest aggregation.

Discovered tests include the 17 protected existing suites (accelerate
proof-focused/reuse, datasets zkp/cache, kit certificate/WAL/release/
coordination) plus the three new current-tree suites
`accelerate-incremental-sealing`, `datasets-incremental-sealing`, and
`kit-proof-seal-store`.

## Modules and granularity

Modules changed across the program include proof-unit planners, cache
key construction, invalidation rules, full-proof fallback, seal/WAL
durability, CLI surfaces (`full`, `incremental`, `verify`, `plan`,
`explain-reuse`, `explain-invalidation`, `benchmark`, `cache-status`,
`force-full`, `compact`), and the three-repository public APIs.

Proof-unit granularity uses a complete cache key, invalidation rules,
full-proof fallback, and Merkle manifest aggregation. Cache-key fields
cover source, schema, fixture, config, and proof dependencies plus
aggregate containment, supersession, and invalidation edges. Full
checkpoint seals and delta seals bind parent acceptance, compare-and-swap
identity, and chain compaction.

## Benchmark (40-transition, provenance-labeled)

The 40-transition benchmark reports average proof reuse rate, average
proving-compute reduction, best incremental case, worst incremental
case, proof size, seal size, verification latency, and storage overhead.
From the bound summary artifact (mixed/estimated planner costs, not
production wall-clock measurements):

| Metric | Value | Provenance |
| --- | --- | --- |
| average proof reuse rate | 43.62% | mixed |
| average proving-compute reduction | 43.05% | mixed |
| best incremental case | ordinary documentation edit (98.90% compute saved) | mixed |
| worst incremental case | initial repository (0% compute saved) | mixed |
| proof size (mean / max bytes) | 18022.4 / 36864.0 | estimated |
| seal size (mean / max bytes) | 3980.8 / 4608.0 | estimated |
| verification latency (mean / max s) | 0.00132 / 0.00270 | estimated |
| storage overhead (mean / max growth bytes) | 18886.4 / 36864.0 | estimated |

All cost and timing fields in that run are estimated planner resource
costs or unavailable; none are measured wall-clock observations from a
production prover. GPU time is unavailable. Simulated required units are
not counted as production proving.

## Crash recovery and tamper

Crash-recovery results and tamper-test results are recorded on the
adversarial board: seven-phase WAL crash cases, wrong-parent rejection,
concurrent-writer fencing, and positive invalidation/seal/compaction
paths. Zero stale or simulated production acceptance is required for
those cases.

## Remaining work before production use

Remaining work before production use includes a production prover with
measured (not only estimated) costs, operational key ceremony and
allowlisted verification keys, recursive verification if required by
policy (currently unsupported), and operational rollout of the narrow
three-repository CLI boundaries under production key material.
