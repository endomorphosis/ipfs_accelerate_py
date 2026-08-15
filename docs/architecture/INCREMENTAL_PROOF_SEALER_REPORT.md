# IncrementalProofSealer Final Report (IPS-056)

Status: terminal release fan-in for `agent/incremental-proof-sealer-v1`.  
Schema observed by protected validation: `incremental-proof-sealer-release-validation@2`.  
Public log witness policy: `public-full-log-secret-scan@1`.

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

## Narrow claim

Repository verification was decomposed into content-addressed proof units.
Unchanged units were safely reused only when their complete dependency and trust
context remained unchanged. Invalidated units were re-proven, affected Merkle
branches were updated, and a new seal was generated from an accepted parent
seal. This reduces proving compute without treating stale or simulated evidence
as current verification.

This report does **not** claim that the entire repository was proven correct, that
all pytest execution was proven in zero knowledge, or that every code change is
semantically correct. Pytest process outputs were observed but test execution was
not cryptographically proven.

## Bound sources and ownership

| Authority | Repository surface | Role |
| --- | --- | --- |
| Datasets | `ipfs_datasets_py/.../logic/zkp/incremental_sealing` | Proof classes, ProofUnit, statements, identity, complete cache key, dependency graph, invalidation rules, manifest, forest codec, migration |
| Kit | `ipfs_kit_py/.../proof_seal_store` | Immutable store, cache index, forest, optional IPFS transport, WAL, CAS pointer, recovery |
| Accelerate | `ipfs_accelerate_py/.../proof/incremental_sealing` | Admission, backends, provers, planner, executor, aggregation, sealer, checkpoints, delta seals, compaction, CLI, metrics |

Exact source revisions and the release receipt digest are bound only by the
protected runner substitution above. Planning revisions remain provenance, not
completion evidence.

## Existing ZK systems and proof classes

Classification of existing zk systems (from the trust baseline and inventories):

| Class | Meaning | Production use |
| --- | --- | --- |
| Real proving | Cryptographic verification of a declared statement | Admissible only after allowlisted keys and successful verification |
| Simulated | Plumbing / mock success paths | Never satisfies a production seal |
| Structural validation | Schema, graph, or envelope checks | Integrity of structure only |
| Integrity commitments | Exact-byte or Merkle commitments | Bind bytes; do not establish correct execution |
| Trusted signed receipts | Signer assertions over allowlisted keys | Receipt consistency, not direct execution |
| Direct execution proof | Declared deterministic computation for one unit | Required when claiming execution |
| Receipt aggregation | Commits to child receipt identities | Does not prove underlying tests ran |

Unknown proof systems are rejected. Test-only keys cannot enter production
allowlists. No assurance upgrade occurs during migration: accept, adapt,
reverify, reject, or keep simulated labels.

## Proof-unit design

- **Proof-unit granularity**: one content-addressed unit per closed statement,
  dependency set, tool/circuit/key/selector/policy context.
- **Complete cache key**: statement identity, dependency digests, environment,
  tool, circuit, verification key, selector, and policy fields. Any mutation
  invalidates reuse.
- **Invalidation rules**: reason-labeled dependency edges; uncertain coverage
  broadens closure; documentation-only edits preserve unrelated units; test
  add/delete is explicit.
- **Full-proof fallback**: forced when cache corruption, canonicalization change,
  circuit/key change, unknown dependency, or trust-context drift is detected.
- **Merkle manifest aggregation**: production aggregation is Merkle manifest
  completeness over verified leaves. Recursive verification is admitted only
  after a successful backend capability probe; default remains individually
  verified leaves plus manifest commitment.

## Durability, recovery, and adversarial gates

Kit WAL/CAS publication uses compare-and-swap. Recovery is WAL-driven.
Crash-recovery results cover the reviewed multi-phase interruption cases and
require deterministic reassembly of the accepted parent root without inventing
success from ambiguous external prover outcomes.

Tamper-test results cover wrong-parent seals, concurrent writers, leaf/order
mutation, lost leaves, and store corruption. Each fails closed with zero
stale or simulated evidence acceptance.

## Public surfaces

Public APIs and CLI in accelerate operate only through the narrow datasets and
kit authorities. Migration reclassifies legacy objects without upgrading
evidence class. Packaging remains hermetic on ordinary import: missing optional
provers are typed unavailable, never silently fabricated.

## 40-transition benchmark

The fixed 40-transition benchmark artifact and summary bind estimated planner
costs (not measured production prover wall-clock). Summary highlights:

| Metric | Value | Provenance |
| --- | --- | --- |
| Average proof reuse rate | 43.6161% | mixed |
| Average proving-compute reduction | 43.0539% | mixed |
| Best incremental case | transition 34 ordinary documentation edit, 98.9011% saved | mixed |
| Worst incremental case | transition 0 initial repository, 0.0000% saved | mixed |
| Proof size | mean ~18022 bytes (estimated) | mixed |
| Seal size | mean ~3981 bytes (estimated) | mixed |
| Verification latency | mean ~0.00132 s seal verification (estimated) | mixed |
| Storage overhead | mean growth ~18886 bytes (estimated) | mixed |

Localized 70% reuse and mixed 50% compute-reduction targets are unmet.
Documentation 80% compute reduction is met. Unmet goals are reported without
inflation. GPU prover time is unavailable.

## Release validation contract

Protected convergent `--run-release-validation`:

1. Consumes only the exact closed materialization-request bundle for IPS-056.
2. Refuses if live ipfs was refused before release suite execution (fixed PATH
   must not resolve `ipfs`).
3. Observes `--check-terminal`, the 17 existing reviewed ZK/reuse/WAL/release
   suites, and the three new incremental-sealing suites from a verified
   read-only materialization.
4. Three new incremental-sealing suites require fully green execution
   (complete collection, exit zero, zero failed/error/xpass/skip/xfail/deselected).
5. Existing suites must be green or baseline-compatible-or-improved; retained
   blockers stay labeled `baseline_compatible_non_green`, never renamed success.
6. Retained public log uses `public-full-log-secret-scan@1`.
7. Assurance is process observation only: pytest process outputs were observed
   but test execution was not cryptographically proven.

## Remaining work before production use

Remaining work before production use includes:

- measured (not only estimated) prover and GPU timing evidence
- production key-ceremony origin and allowlist documentation
- recursive verification where claimed, only after capability probe success
- operational runbooks for checkpoint force, cache corruption, and seal CAS
  contention
- continued separation of simulated test plumbing from production seals

Stale or simulated evidence must never be treated as current verification.

## Explicit nonclaims

- Integrity commitments bind bytes only.
- Trusted signed receipts are signer assertions, not direct execution proofs.
- Receipt aggregation does not prove tests ran.
- Simulated or structural validation paths do not satisfy production seals.
- Process-observed pytest is not a zero-knowledge proof of execution.
