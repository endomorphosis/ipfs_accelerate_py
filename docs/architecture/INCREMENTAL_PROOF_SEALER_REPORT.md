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
content-addressed proof units. Unchanged units were safely reused when
their complete dependency and trust context remained unchanged.
Invalidated units were re-proven, affected Merkle branches were updated,
and a new seal was generated from an accepted parent seal, reducing
proving compute without treating stale or simulated evidence as current
verification. Existing suite outcomes that remain non-green solely
because they match protected operator baselines are labeled
baseline_compatible_non_green and named by the protected runner evidence
binding rather than reclassified as success.

## Systems and claims

Existing ZK systems are classified as real proving, simulated, or
structural validation. Direct execution proof is distinct from trusted
signed receipts and integrity commitments. Proof-unit granularity uses
a complete cache key, invalidation rules, full-proof fallback, and
Merkle manifest aggregation. Discovered modules, systems, and tests are
bound by the protected runner receipt rather than provider prose.

## Benchmark

The 40-transition benchmark reports average proof reuse rate, average
proving-compute reduction, best incremental case, worst incremental
case, proof size, seal size, verification latency, and storage overhead.
Crash-recovery results and tamper-test results are recorded on the
adversarial board. Provenance-labeled metrics remain estimated or mixed
where a production prover was unavailable.

## Remaining work before production use

Remaining work before production use includes a production prover,
measured (not only estimated) costs, and operational key ceremony.
