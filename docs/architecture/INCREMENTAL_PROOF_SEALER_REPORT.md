# Incremental Proof Sealer Terminal Report

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

This is the current-tree fan-in report for IncrementalProofSealer.
Schema `incremental-proof-sealer-release-validation@2` and log policy
`public-full-log-secret-scan@1` are bound by the protected runner. The
protected `--run-release-validation` ensure substitutes the marker above
with the fresh receipt digest, exact three source revisions, and ordered
`baseline_compatible_non_green` suite IDs. Provider prose never authors
those bindings.

## What was observed

Live ipfs was refused before release suite execution.
Pytest process outputs were observed but test execution was not
cryptographically proven. Three new incremental-sealing suites require
fully green execution. Repository verification was decomposed into
content-addressed proof units. Stale or simulated evidence is never
treated as current verification. The narrow final claim is
`pytest_process_outputs_observed_only_not_a_proof_of_test_execution`.

## Systems and claims

Existing ZK systems are classified as real proving, simulated, or
structural validation. Inventories also label hybrid bridges, integrity
envelopes, integrity transport, and unavailable optional backends.
Direct execution proof is distinct from trusted signed receipts and
integrity commitments. Receipt-aggregation zk proofs commit to child
receipt identity completeness only; they do not prove underlying tests
ran. Direct execution proof remains declared-computation only.
Incremental commit seals are parent-bound verified leaf transitions.

Discovered modules and surfaces include:

- **datasets** (semantic authority): CEC/TDFOL/F-logic bridges, proof-unit
  identity, manifests, dependency graphs, invalidation, commitment codecs,
  wallet/PDF simulated paths (production-seal forbidden).
- **kit** (storage authority): proof-certificate CID transport, modern WAL,
  CAS/forest indexes, Profile-D policy adapters, MCP++/Iroh release
  receipts (unsigned remain unsigned), planned proof-seal-store tests.
- **accelerate** (execution authority): proof attestation contracts,
  unsigned TestPassReceipt/ProofReceipt envelopes, schedulers, multi-prover
  resources, proof-reuse CLI/runtime, disposable test-only Groth16 fixture.

Proof-unit granularity uses a complete cache key, invalidation rules,
full-proof fallback, and Merkle manifest aggregation. Unsupported
recursion defaults to Merkle manifest aggregation with individually
verified child leaves. Recursive self-verification is unavailable.

## Benchmark

The 40-transition benchmark reports average proof reuse rate, average
proving-compute reduction, best incremental case, worst incremental
case, proof size, seal size, verification latency, and storage overhead.
All forty transition rows carry mixed metric provenance (estimated CPU
and size fields; GPU unavailable). Summary binding
`sha256:f33fbdb928f523e11e75d22cc2808594bcb1e2fb83b9fcf7837d49608afa2de6`:

| Metric | Value | Provenance |
| --- | --- | --- |
| average proof reuse rate | 43.6161% | mixed |
| average proving-compute reduction | 43.0539% | mixed |
| best incremental case | transition 34 ordinary documentation edit, 98.9011% saved | mixed |
| worst incremental case | transition 0 initial repository, 0.0% saved | mixed |
| proof size (mean) | 18022.4 bytes | estimated |
| seal size (mean) | 3980.8 bytes | estimated |
| verification latency (mean seal_verification_seconds) | 0.00132 s | estimated |
| storage overhead (mean storage_growth_bytes) | 18886.4 bytes | estimated |

Target honesty: localized 70% reuse unmet (43.0159%); mixed-history 50%
compute reduction unmet (43.0539%); documentation 80% compute reduction
met (98.7124%). Estimates are never sold as measurements.

Crash-recovery results and tamper-test results are recorded on the
adversarial board: wrong-parent rejection, concurrent-writer CAS
discipline, seven-phase WAL crash recovery, and negative suites that
refuse stale or simulated production seals.

## Remaining work before production use

Remaining work before production use includes a production prover,
measured (not only estimated) costs, recursive verification where
claimed, operational key ceremony, and allowlisted production
verification keys. Test-only keys must never enter a production
allowlist. Unknown proof systems remain rejected.
