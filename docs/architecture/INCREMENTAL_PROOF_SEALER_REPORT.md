# Incremental Proof Sealer Terminal Report

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

This is the current-tree fan-in report for IncrementalProofSealer.
Schema `incremental-proof-sealer-release-validation@2` and log policy
`public-full-log-secret-scan@1` are bound by the protected runner. Exact
source revisions and the release receipt digest are substituted into the
evidence marker by that runner; provider prose never invents those values.

## What was observed

Live ipfs was refused before release suite execution.
Pytest process outputs were observed but test execution was not cryptographically proven.
Three new incremental-sealing suites require fully green execution.
Repository verification was decomposed into content-addressed proof units.
Stale or simulated evidence is never treated as current verification.

The protected release runner observes `--check-terminal` plus exactly
seventeen existing reviewed ZK/reuse/WAL/release suites and three new
current-tree incremental-sealing suites. Every retained non-green existing
suite outcome is labeled `baseline_compatible_non_green` and named in the
materialized evidence binding; non-green outcomes are never hidden as
success.

## Systems and claims

Existing ZK systems are classified as real proving, simulated, or
structural validation. Direct execution proof is distinct from trusted
signed receipts and integrity commitments. Proof-unit granularity uses
a complete cache key, invalidation rules, full-proof fallback, and
Merkle manifest aggregation.

### Discovered systems and tests

- **Real proving**: bounded declared Groth16 computation only when tools and
  keys are present; ProveKit remains an optional capability whose absence is
  typed unavailable rather than a silent pass.
- **Simulated**: plumbing-only fixtures and simulated required units; they
  cannot satisfy a production seal.
- **Structural validation**: inventory surfaces, unsigned integrity envelopes,
  attestation boundaries, and process-observed pytest suites that record
  outcomes without cryptographic execution proof.
- **Direct execution proof**: declared deterministic computation for one
  proof unit only; an incremental or recursive commit seal is accepted only
  against an accepted parent.

Authorities remain split across modules:

| Authority | Module |
| --- | --- |
| proof unit / manifest / identity | `ipfs_datasets_py` |
| proof object / cache / forest / WAL / CAS | `ipfs_kit_py` |
| prover scheduler / aggregation planner / metrics | `ipfs_accelerate_py` |

### Cache, invalidation, fallback, aggregation

- Complete cache key fields bind statement, circuit, verification key, public
  inputs, environment trust policy, and related identity material.
- Invalidation rules close over the dependency graph; unknown or truncated
  closure broadens invalidation or forces a full-proof fallback.
- Full-proof fallback covers first state, low reuse, schema or key change,
  trust-policy change, cache corruption, periodic cadence, and release tags.
- Aggregation is Merkle manifest aggregation of verified child identities,
  not recursive self-verification of the tests themselves.

### Direct, signed, and integrity claims

- Integrity commitments bind exact bytes only.
- Trusted signed receipts are signer assertions against the current allowlist.
- Receipt-aggregation zk proofs commit to child receipt identities; they do
  not prove underlying tests ran.
- Direct execution proofs remain declared computation only.

## Benchmark

The 40-transition benchmark reports average proof reuse rate, average
proving-compute reduction, best incremental case, worst incremental
case, proof size, seal size, verification latency, and storage overhead.
All forty transition metrics are provenance-labeled `mixed` because GPU
time is unavailable while CPU, size, and storage fields are estimated;
none are measured wall-clock observations from a production prover.

Summary bound to
`artifacts/agent_supervisor/incremental_proof_sealer/summary.json`
and raw digest
`sha256:f33fbdb928f523e11e75d22cc2808594bcb1e2fb83b9fcf7837d49608afa2de6`:

| Metric | Value | Provenance |
| --- | --- | --- |
| average proof reuse rate | 43.6161% | mixed |
| average proving-compute reduction | 43.0539% | mixed |
| best incremental case | transition 34 ordinary documentation edit, 98.9011% saved | mixed |
| worst incremental case | transition 0 initial repository, 0.0000% saved | mixed |
| proof size (mean) | 18022.4 bytes | estimated |
| seal size (mean) | 3980.8 bytes | estimated |
| verification latency (mean) | 0.00132 s | estimated |
| storage overhead (mean growth) | 18886.4 bytes | estimated |

Target assessments remain honest: localized 70% reuse and mixed-history 50%
compute reduction are unmet; documentation 80% compute reduction is met.
Estimates are never sold as measurements.

## Crash and tamper board

Crash-recovery results and tamper-test results are recorded on the
adversarial board. Publication uses compare-and-swap; recovery is WAL-driven;
ambiguous external prover outcomes are not success. Wrong-parent, schema,
verification-key, trust-policy, and cache-corruption cases force full
checkpoints rather than silent reuse.

## Remaining work before production use

Remaining work before production use includes a production prover,
measured (not only estimated) costs, recursive verification where claimed,
and operational key ceremony. Test-only keys cannot enter a production
allowlist. Unknown proof systems stay rejected.
