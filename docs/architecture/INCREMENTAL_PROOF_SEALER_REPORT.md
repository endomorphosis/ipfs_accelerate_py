# Incremental Proof Sealer Terminal Report

<!-- IPS-056 RELEASE EVIDENCE (materialized by protected runner)
receipt_digest: sha256:f6a219d173232d03f692532aa9c7d9d5b9626b6ef593abee1abcd957e599bded
accelerate_revision: 20d76085f28adf6f104262d8d726caa348a935d6
datasets_revision: 1480ea2b4c54dda94c64b792c0af621cd764dbbb
kit_revision: 8799e8d3cc39bd8f2e58b819dacb6b3879b517c0
baseline_compatible_non_green: none
-->

This is the current-tree fan-in report for IncrementalProofSealer.
Schema `incremental-proof-sealer-release-validation@2` and log policy
`public-full-log-secret-scan@1` are bound by the protected runner.
The receipt digest and exact accelerate, datasets, and kit commits are
substituted into the release-evidence marker by that runner only.

## What was observed

Live ipfs was refused before release suite execution.
Pytest process outputs were observed but test execution was not
cryptographically proven. Three new incremental-sealing suites require
fully green execution. Repository verification was decomposed into
content-addressed proof units. Stale or simulated evidence is never
treated as current verification.

Narrow final claim: repository verification was decomposed into
content-addressed proof units; unchanged units were safely reused when
their complete dependency and trust context remained unchanged;
invalidated units were re-proven, affected Merkle branches were
updated, and a new seal was generated from an accepted parent seal,
reducing proving compute without treating stale or simulated evidence
as current verification.

## Systems, modules, and claims

Existing ZK systems are classified as real proving, simulated, or
structural validation. Direct execution proof is distinct from trusted
signed receipts and integrity commitments. Discovered surfaces include
datasets Groth16/ProveKit ZKP paths, accelerate proof schedulers and
reuse caches, and kit proof-certificate/store plus modern WAL modules.
New modules under `ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing`,
datasets incremental-sealing unit paths, and kit `proof_seal_store`
implement the sealed product.

Proof-unit granularity uses a complete cache key, invalidation rules,
full-proof fallback, and Merkle manifest aggregation. Cache-key fields
bind source, schema, fixture, config, proof, and aggregate edges.
Invalidation covers source implementation, public interface, test
source, deleted tests, fixtures, configuration, circuit/key, trust
policy, and schema changes. Aggregation is Merkle-manifest completeness
commitment, not recursive verification, unless a backend capability
probe admits recursion.

## Benchmark and adversarial results

The 40-transition benchmark reports average proof reuse rate
(about 43.62 percent, mixed provenance), average proving-compute
reduction (about 43.05 percent, mixed provenance), best incremental
case (ordinary documentation edit, about 98.90 percent compute saved),
worst incremental case (initial repository, 0 percent), proof size,
seal size, verification latency, and storage overhead. Metric
provenance labels each cost and size field as estimated, measured, or
unavailable; this run used estimated planner costs and no production
prover wall-clock measurements. Crash-recovery results and
tamper-test results are recorded on the adversarial board and in the
new incremental-sealing suites; wrong-parent and concurrent-writer
attempts are rejected.

## Baseline-compatible non-green retained issues

Every retained skip, xfail, deselection, failure, error, xpass, or
collection issue from existing suites is labeled
`baseline_compatible_non_green` and named here when still present after
current-tree observation. Operator-baseline non-green suite IDs that
remain acceptable only as baseline-compatible-or-improved include:
accelerate-proof-focused-core-15, accelerate-proof-focused-wide-36,
accelerate-proof-reuse-migration, accelerate-proof-reuse-cross-repo,
kit-coordination, kit-proof-reuse-bootstrap, kit-agent-receipts, and
kit-release-receipt, plus any datasets suite that still carries
baseline non-pass outcomes. Exact ordered IDs observed by the protected
runner are bound in the release-evidence marker. They are never renamed
success.

## Remaining work before production use

Remaining work before production use includes a production prover,
recursive verification where claimed, measured (not only estimated)
costs, operational key ceremony, and production key allowlisting for
test-only keys.
