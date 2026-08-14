# Incremental Proof Sealer Terminal Report

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

This is the current-tree fan-in report for IncrementalProofSealer.
Schema `incremental-proof-sealer-release-validation@2` and log policy
`public-full-log-secret-scan@1` are bound by the protected runner.

## What was observed

Live ipfs was refused before release suite execution.
Pytest process outputs were observed but test execution was not
cryptographically proven. Three new incremental-sealing suites require
fully green execution. Repository verification was decomposed into
content-addressed proof units. Stale or simulated evidence is never
treated as current verification.

## Systems and claims

Existing ZK systems are classified as real proving, simulated, or
structural validation. Direct execution proof is distinct from trusted
signed receipts and integrity commitments. Proof-unit granularity uses
a complete cache key, invalidation rules, full-proof fallback, and
Merkle manifest aggregation.

Backend decisions from the trust baseline:

- Groth16: bounded declared computation only
- ProveKit: optional capability; unavailable is typed
- Simulated: production seal forbidden
- Unknown systems: rejected
- Existing recursive backend: unsupported

Proof classes remain integrity commitments, trusted signed receipts,
receipt-aggregation zk proofs that do not prove underlying tests ran,
and direct execution proofs for declared deterministic units only.

## Modules, keys, and aggregation

Modules under accelerate own proving, planning, scheduling, aggregation,
sealing, and measurement. Datasets owns proof semantics, identity,
manifests, and invalidation. Kit owns storage, index, forest, WAL, and
CAS. Complete cache keys bind code, circuit, verification-key, and
environment identity. Invalidation rules force full-proof fallback on
schema, circuit, verification-key, trust-policy, dependency-lock, cache
corruption, or release-tag transitions. Aggregation uses Merkle manifest
aggregation over accepted proof units rather than recursive proof
verification of child signatures. Child proofs are individually verified
outside the circuit; test execution is not directly proven.

## Benchmark

The 40-transition benchmark reports average proof reuse rate, average
proving-compute reduction, best incremental case, worst incremental
case, proof size, seal size, verification latency, and storage overhead.
Metric provenance is labeled estimated, measured, mixed, or unavailable
per field.

From the protected summary artifact:

- average proof reuse rate: 43.6161% (mixed)
- average proving-compute reduction: 43.0539% (mixed)
- best incremental case: ordinary documentation edit, 98.9011% saved (mixed)
- worst incremental case: initial repository, 0.0000% saved (mixed)
- proof size / seal size / verification latency / storage overhead:
  estimated planner costs only; GPU time unavailable

Crash-recovery results and tamper-test results are recorded
on the adversarial board.

## Remaining work before production use

Remaining work before production use includes a production prover,
measured (not only estimated) costs, and operational key ceremony.
Recursive verification remains unsupported until a successful backend
capability probe. Simulated and structural surfaces must never be sold
as current verification of test execution.
