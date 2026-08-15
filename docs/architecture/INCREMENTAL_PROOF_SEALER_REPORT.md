# Incremental Proof Sealer Terminal Report

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

This is the current-tree fan-in report for IncrementalProofSealer.
Schema `incremental-proof-sealer-release-validation@2` and log policy
`public-full-log-secret-scan@1` are bound by the protected runner after it
consumes the closed materialization request and observes terminal/suite
process evidence. The sealed provider authors only this request-shaped
report and the exact JSON/log materialization requests; it does not claim
pass/fail for the release suites.

## What was observed

Live ipfs was refused before release suite execution.
Pytest process outputs were observed but test execution was not cryptographically proven.
Three new incremental-sealing suites require fully green execution.
Repository verification was decomposed into content-addressed proof units.
Stale or simulated evidence is never treated as current verification.
Any retained baseline non-pass suite is labeled `baseline_compatible_non_green`
and named after materialization; none of those residual issues is renamed success.

## Systems and claims

Existing ZK systems are classified as real proving, simulated, or
structural validation. Direct execution proof is distinct from trusted signed receipts
and integrity commitments. Proof-unit granularity uses a complete cache key,
invalidation rules, full-proof fallback, and Merkle manifest aggregation.
Modules follow the datasets/kit/accelerate ownership split: datasets owns
proof semantics and identity, kit owns immutable storage/WAL/CAS, and
accelerate owns planning, proving, scheduling, aggregation, sealing, and measurement.

## Benchmark

The 40-transition benchmark reports average proof reuse rate, average proving-compute reduction,
best incremental case, worst incremental case, proof size, seal size,
verification latency, and storage overhead.
From the protected analysis artifact (`incremental-proof-sealer-benchmark-summary@1`,
raw digest `sha256:f33fbdb928f523e11e75d22cc2808594bcb1e2fb83b9fcf7837d49608afa2de6`):

- average proof reuse rate: 43.6161% (mixed / estimated planner costs)
- average proving-compute reduction: 43.0539% (mixed)
- best incremental case: transition 34 ordinary documentation edit, 98.9011% saved (mixed)
- worst incremental case: transition 0 initial repository, 0.0000% saved (mixed)
- proof size / seal size / verification latency / storage overhead remain
  provenance-labeled estimated fields; GPU time is unavailable
- crash-recovery results and tamper-test results are recorded on the
  adversarial board (IPS-048 through IPS-051)

Localized 70% reuse and mixed-history 50% compute reduction targets are
unmet; documentation 80% compute reduction is met under mixed provenance.
Targets are not facts. Receipt aggregation does not prove test execution.
Simulated required units cannot satisfy a production seal.

## Trust boundary

Integrity commitments, trusted signed receipts, direct execution proofs,
and parent-bound incremental seals keep their exact nonclaims. Current
production aggregation is Merkle manifest aggregation, not recursive proof
verification. Unknown systems, arbitrary circuits, and simulated-as-real
evidence are rejected.

## Remaining work before production use

Remaining work before production use includes a production prover,
measured (not only estimated) costs, recursive verification where claimed,
and operational key ceremony.
