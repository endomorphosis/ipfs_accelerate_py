# Incremental Proof Sealer Terminal Report

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

This is the current-tree fan-in report for IncrementalProofSealer.
Schema `incremental-proof-sealer-release-validation@2` and log policy
`public-full-log-secret-scan@1` are bound by the protected runner.
Protected convergent `--run-release-validation` materializes the receipt
digest and exact accelerate, datasets, and kit source revisions into the
release-evidence block above; provider prose never invents those digests.

## Narrow final claim

Repository verification was decomposed into content-addressed proof units.
Unchanged units were safely reused when their complete dependency and trust
context remained unchanged. Invalidated units were re-proven, affected Merkle
branches were updated, and a new seal was generated from an accepted parent
seal, reducing proving compute without treating stale or simulated evidence
as current verification.

This report never claims the repository was proven correct, that all pytest
execution was proven in zero knowledge, that a change is semantically
correct, or that tests passed when only hashes or receipt aggregation were
verified.

## What was observed

Live ipfs was refused before release suite execution.
Pytest process outputs were observed but test execution was not cryptographically proven.
Three new incremental-sealing suites require fully green execution
(complete nonzero collection, exit zero, and zero
failed/error/xpassed/skipped/xfailed/deselected outcomes). Existing
reviewed suites must be green or baseline-compatible-or-improved; every
retained skip, xfail, deselection, failure, error, xpass, or collection
issue is labeled `baseline_compatible_non_green` and named in the
materialized evidence block, never hidden as success.

## Systems and claims

Existing ZK systems are classified as real proving, simulated, or
structural validation:

- **accelerate** (`ipfs_accelerate_py`): prover scheduler, aggregation
  planner, trust/key allowlists, CLI, and metrics. Inventory modules
  include proof attestation, conformance, multi-prover routing, formal
  verification policy, and proof-reuse activation/publication paths.
- **datasets** (`ipfs_datasets_py`): proof-unit identity, manifests,
  dependency graphs, requirement discovery, invalidation, and commitment
  codecs.
- **kit** (`ipfs_kit_py`): immutable proof objects, exact-key cache
  indexes that never decide acceptance, proof-forest persistence, WAL
  durability, and repository/branch-namespaced current-seal CAS.

Direct execution proof is distinct from trusted signed receipts and
integrity commitments:

| Claim class | Bound meaning |
| --- | --- |
| integrity commitments | exact bytes, digest, CID, Merkle inclusion; not execution |
| trusted signed receipts | allowlisted signer assertion; not independent execution proof |
| receipt-aggregation zk | completeness of committed receipts; not underlying pytest |
| direct execution proof | declared deterministic computation for one proof unit only |
| incremental commit seal | parent-bound verified leaf transition and new root |

Backend decisions: groth16 is bounded declared computation only; ProveKit
is typed unavailable when absent; simulated production seals are forbidden;
unknown systems are rejected. Recursive self-verification is unsupported;
aggregation is Merkle manifest aggregation with individually verified
children.

## Modules, granularity, and cache rules

Proof-unit granularity uses the smallest sound module or symbol closure for
static analysis, and one collected pytest node plus canonical parameters for
tests. A complete cache key (`ProofCacheKey@1`) binds statement CID,
public-input CID, private-input commitment, source artifact CIDs, dependency
proof-unit roots, environment, dependency-lock, fixtures, tool/prover
identity, proof-system and evidence class, circuit and key IDs, configuration,
network-policy, proof-schema and canonicalization versions, test-selector, and
policy CID.

Invalidation rules walk forward from changed prerequisites to dependants.
Source, test, fixture, configuration, dependency-lock, circuit/key,
canonicalization, and environment-policy changes broaden invalidation or force
full-proof fallback; documentation-only changes preserve execution proofs
unless the document is a checked specification. Full-proof fallback applies on
low reuse, periodic cadence, cache corruption, schema change, trust-policy
change, and release-tag compaction. Aggregation strategy is Merkle manifest
aggregation, not recursive proof verification.

## Benchmark (40-transition, provenance-labeled)

The 40-transition benchmark reports average proof reuse rate, average
proving-compute reduction, best incremental case, worst incremental case,
proof size, seal size, verification latency, and storage overhead. Raw
artifact digest
`sha256:f33fbdb928f523e11e75d22cc2808594bcb1e2fb83b9fcf7837d49608afa2de6`
(`incremental-proof-sealer-benchmark-results@2`). Every transition row is
`mixed` provenance: CPU/size/storage fields are estimated planner costs; GPU
is unavailable; no field is measured wall-clock from a production prover.

| Metric | Value | Provenance |
| --- | --- | --- |
| average proof reuse rate | 43.6161% | mixed |
| average proving-compute reduction | 43.0539% | mixed |
| best incremental case | transition 34 ordinary documentation edit, 98.9011% saved | mixed |
| worst incremental case | transition 0 initial repository, 0.0% saved | mixed |
| proof size (mean / max bytes) | 18022.4 / 36864.0 | estimated |
| seal size (mean / max bytes) | 3980.8 / 4608.0 | estimated |
| verification latency (mean seal_verification_seconds) | 0.00132 | estimated |
| storage overhead (mean storage_growth_bytes) | 18886.4 | estimated |
| prover_cpu_seconds mean | 4.5 | estimated |
| prover_gpu_seconds | unavailable | unavailable |

Target assessment (targets are not facts): localized 70% reuse unmet
(43.0159%); mixed-history 50% compute reduction unmet (43.0539%);
documentation 80% compute reduction met (98.7124%).

## Crash-recovery and tamper results

Crash-recovery results and tamper-test results are recorded on the
adversarial board: seven-phase crash matrix, wrong-parent rejection,
concurrent-writer CAS, signature and integrity tamper paths, and WAL-driven
recovery. Stale or simulated evidence is never treated as current
verification. Publication uses compare-and-swap; ambiguous external prover
outcomes are not success.

## Current-tree validation suites

Protected release observation covers exactly 17 existing reviewed
ZK/reuse/WAL/release suites plus 3 new current-tree suites:

- new fully-green suites: `accelerate-incremental-sealing`,
  `datasets-incremental-sealing`, `kit-proof-seal-store`
- existing suites: accelerate proof-focused core/wide and reuse
  migration/cross-repo; datasets zkp focused/unit-wide/broad-safe and
  proof-cache adapters; kit proof-certificate, reuse-capabilities,
  profile-d, coordination, modern-wal, proof-reuse-bootstrap,
  agent-receipts, iroh-release, release-receipt

Process observation is not cryptographic proof of test execution. Any
baseline-compatible non-green remaining issues are listed in the
materialized `baseline_compatible_non_green` field of the release-evidence
block.

## Remaining work before production use

Remaining work before production use includes a production prover backend,
recursive verification where claimed, measured (not only estimated) costs,
and operational key ceremony. ProveKit and GPU prover capability remain
typed unavailable in this environment.
