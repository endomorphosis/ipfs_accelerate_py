# IncrementalProofSealer final report (IPS-056)

Status: terminal release report for `agent/incremental-proof-sealer-v1`.

This document is the sealed-provider substantive report for IPS-056. It does
**not** authorize itself. Protected convergent validation materializes the
exact closed request, runs the terminal board gate plus release suites, writes
canonical `incremental-proof-sealer-release-validation@2` process evidence, and
replaces only the single marker below with the fresh receipt digest, source
revisions, and ordered `baseline_compatible_non_green` suite IDs.

<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->

Public log policy for retained release evidence is
`public-full-log-secret-scan@1`. Live IPFS was refused before release suite execution.
Three new incremental-sealing suites require fully green execution.
Pytest process outputs were observed but test execution was not cryptographically proven.

## Narrow release claim

Repository verification was decomposed into content-addressed proof units.
Unchanged units were safely reused when their complete dependency and trust
context remained unchanged. Invalidated units were re-proven, affected Merkle
branches were updated, and a new seal was generated from an accepted parent
seal, reducing proving compute without treating stale or simulated evidence as
current verification.

This report never claims that the repository was proven correct, that all
pytest execution was proven in zero knowledge, that a change is semantically
correct, or that tests passed when only hashes or receipt aggregation were
verified.

## Bound source state and planning revisions

Planning revisions are provenance, not completion evidence:

| Repository | Planning revision | Role |
|---|---|---|
| `ipfs_accelerate_py` | `8881344bb2162f3f8d82f22d8348bc0ac7536f95` | prover, scheduler, planner, aggregation, measurement |
| `ipfs_datasets_py` | `bd2ff6245ebe476fc744d45c7c66235c92b0e19c` | proof-unit identity, manifests, invalidation |
| `ipfs_kit_py` | `5a7a2df8181cfdc33bc19be09989df7ff83f2d4e` | proof store, forest, WAL, CAS |

Operator baseline receipts (protected pins):

| Task | Receipt digest | Source revision |
|---|---|---|
| IPS-001 accelerate | `sha256:a85bc27f70dabbbea49d26200fee27e43dbf61102beb6a9789df7a15f367474e` | `9b43e0ea1c3cf651d884f5489ca46d7eda2ae41b` |
| IPS-002 datasets | `sha256:7dbe6a2f3b6fcbf56c841d2c757951a76b7dc91a6dc91614c6e618e4c38cdb81` | `cae71d992f82ae7a0975ba4f5ed0c575b1479253` |
| IPS-003 kit | `sha256:22d4f9663e3346fd2264efb38538cbedead642c3ba7fe403b5c7b9fc6545f982` | `b2c8e625b184c41fa865d906e0037915e3fb9179` |

Trust baseline synthesis (IPS-004) is bound at
`synthesis_worktree_parent_revision` `96d104da51950f01fa379c7f9f9d50fd47d3c09c`
under schema `incremental-proof-sealer-trust-baseline@2`. Inventory completion
revisions recorded in the matrix:

- accelerate inventory completion: `bb5d184ab8355def188ad5775a664a6790e54a63`
- datasets inventory completion: `7ea9822f11af9a6c3024ac6c29ced6270aa4321d`
- kit inventory completion: `f7074e1175505e8e1f0ea44a9e1f1db5ea2891db`

Benchmark evidence parent and source map (IPS-053):

| Field | Value |
|---|---|
| `benchmark_worktree_parent_revision` | `b11c1e84ce70840f99df66fd01f78a38985ec617` |
| accelerate source | `b11c1e84ce70840f99df66fd01f78a38985ec617` |
| datasets source | `1480ea2b4c54dda94c64b792c0af621cd764dbbb` |
| kit source | `8799e8d3cc39bd8f2e58b819dacb6b3879b517c0` |
| raw digest | `sha256:f33fbdb928f523e11e75d22cc2808594bcb1e2fb83b9fcf7837d49608afa2de6` |

Exact current-tree revisions for the IPS-056 release receipt are written only
by the protected runner into the evidence marker above.

## Existing ZK systems and proof classifications

Existing ZK systems were inventoried and classified from executable code and
tests, not from documentation alone.

### Real proving

- Datasets Groth16 v2 proves one bounded Horn-style TDFOL derivation over
  declared public inputs. Planning-time probe produced a 1,762-byte proof that
  verified in ~0.008s. That is real proving of a declared computation only; it
  is not pytest execution proof.
- Groth16 v1 proves a nonzero public commitment; v3 commits event
  digest/root/count. Neither proves test execution.

### Simulated

- Simulated units, mock hardware paths, disposable test fixtures, and plumbing
  backends are production-seal-forbidden. A simulated required unit forces
  `simulated_only` and can never produce `sealed_full` or `sealed_incremental`
  under production policy.
- Wallet/PDF simulated paths, ProveKit absence recorded as typed
  `unavailable`, and test-only keys remain non-production.

### Structural validation

- Accelerate unsigned `TestPassReceipt` / `ProofReceipt` envelopes are
  structural/integrity assertions, not signed receipts.
- Cache/admission gates, attestation stores that bind existing receipts, and
  kit Event-DAG hash commitments are structural validation or integrity only.
- Kit has no real proving test and no recursive verifier in inventory.

### Direct execution proof

- Direct execution proof is admitted only as a declared deterministic
  computation for one proof unit. Receipt aggregation never upgrades into
  direct execution. Direct-computation claim language is reserved for that
  class alone.

Closed evidence classes remain:

| Class | Establishes | Does not establish |
|---|---|---|
| Integrity commitments | exact bytes, digest, CID, Merkle inclusion | execution or semantic correctness |
| Trusted signed receipts | allowlisted signer asserted execution | independent proof without signer trust |
| Receipt aggregation ZK | committed receipt completeness/order | underlying tests ran |
| Direct execution proof | declared program over committed inputs | correctness beyond that program |
| Incremental commit seal | parent-bound leaf transition + complete manifest | arbitrary repository correctness |

## Modules delivered

### Accelerate execution authority

Package: `ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing`

| Module | Responsibility |
|---|---|
| `admission.py` | evidence-class verification before cache admission |
| `aggregation.py` | Merkle manifest aggregation; recursion capability-gated |
| `backends.py` | backend capability probes |
| `bootstrap.py` | seal bootstrap helpers |
| `checkpoint_policy.py` | full-checkpoint triggers and fail-closed defaults |
| `cli.py` | focused `zk-seal` CLI |
| `compaction.py` | chain compaction with retention |
| `delta_seal.py` | parent-bound incremental transitions |
| `executor.py` | plan execution, process bounds, outcome typing |
| `explanations.py` | reuse/invalidation/cost explanations |
| `full_checkpoint.py` | full seal creation |
| `metrics.py` | measured/estimated/unavailable provenance |
| `migration.py` | accept/adapt/reverify/reject/simulated dispositions |
| `planner.py` | full-versus-incremental planning |
| `process_control.py` | process-tree termination and output bounds |
| `provers.py` | prover adapters without silent install/download |
| `scheduling.py` | resource admission and scheduling stages |
| `sealer.py` | atomic seal publication |
| `trust.py` | verification-key/signer allowlists and setup origin |
| `verification.py` | seal verification against trusted policy |

Public facade APIs: `create_full_checkpoint`, `create_incremental_plan`,
`execute_incremental_plan`, `verify_seal`, `explain_reuse`,
`explain_invalidation`, `compare_full_and_incremental`.

### Datasets semantic authority

Package: `ipfs_datasets_py.logic.zkp.incremental_sealing` owns proof-unit
granularity, closed enums, manifests, complete cache key schemas, dependency
graph edges, invalidation rules, and leaf/category/repository commitment
codecs.

### Kit storage authority

Package: `ipfs_kit_py.proof_seal_store` owns immutable proof objects, exact-key
cache indexes (never acceptance), proof-forest branch updates, current-root
compare-and-swap, durable seal-transition WAL, recovery, and concurrent-writer
rejection on modern `core/wal` primitives.

## Proof-unit granularity and complete cache key

Proof-unit granularity is one content-addressed unit per selected
source/test/property/circuit binding, never a whole-repository blob and never
an undeclared helper side file.

A complete cache key binds at least:

- repository identity and source-root commitment
- statement / public-input commitment
- dependency closure and environment policy
- selector and verification policy
- circuit identity and content-addressed verification key
- proof system and evidence class
- parent seal identity for incremental transitions

Omitting any complete-key component never regains reuse.

## Invalidation rules and full-proof fallback

Invalidation rules re-prove when any complete-key field changes, including
source root, environment, selector, verification key, circuit, dependency
closure, public input, policy, fixture, configuration, or network mismatch.
Unauthorized test removal, missing invalidated units, reordered/duplicate
manifest leaves, and simulated evidence presented as real reject closed.

Full-proof fallback fires on first state, missing parent, periodic cadence,
release tags, circuit/key/lock/trust/schema/canonicalization/environment
change, cache corruption, low reuse ratio, excessive delta-chain depth, or
explicit force. Incremental callers cannot override a fired trigger.

## Aggregation strategy

Default production aggregation is Merkle manifest aggregation:

- child proofs are individually verified before inclusion
- exact child identities, count, order, duplicate rejection, root, terminal
  status, repository, and environment are bound
- recursive self-verification is unsupported unless a backend capability probe
  succeeds
- receipt aggregation does not prove test execution

Backend decisions frozen in the trust baseline:

| Backend | Decision |
|---|---|
| existing recursive backend | unsupported |
| groth16 | bounded_declared_computation_only |
| provekit | optional_capability_unavailable_is_typed |
| simulated | production_seal_forbidden |
| unknown | rejected |

## 40-transition benchmark results

Canonical summary schema `incremental-proof-sealer-benchmark-summary@1` binds
raw artifact `sha256:f33fbdb928f523e11e75d22cc2808594bcb1e2fb83b9fcf7837d49608afa2de6`.
All forty transitions are labeled `mixed` (CPU/size/storage estimated; GPU
unavailable). Targets are comparisons, never facts.

| Metric | Value | Provenance |
|---|---|---|
| average proof reuse rate | 43.6161% | mixed |
| average proving-compute reduction | 43.0539% | mixed |
| best incremental case | transition 34 ordinary documentation edit, 98.9011% saved | mixed |
| worst incremental case | transition 0 initial repository, 0.0000% saved | mixed |
| fallback indices | 0, 12, 17, 20, 22, 24, 27, 29, 30, 32, 35, 39 | observed policy |

Target assessments:

| Goal | Target | Actual | Met |
|---|---|---|---|
| localized reuse | 70% | 43.0159% | no |
| mixed compute reduction | 50% | 43.0539% | no |
| documentation compute reduction | 80% | 98.7124% | yes |

Unmet goals are reported honestly without manufacturing savings.

### Size, latency, and storage overhead

| Metric | min | mean | max | notes |
|---|---|---|---|---|
| proof size | 0 B | 18022.4 B | 36864 B | estimated |
| seal size | 3584 B | 3980.8 B | 4608 B | estimated |
| verification latency | 0 s | 0.00132 s | 0.0027 s | estimated |
| storage overhead / growth | 1792 B | 18886.4 B | 36864 B | estimated |
| prover_cpu_seconds | 0.1 | 4.5 | 9.1 | estimated |
| prover_gpu_seconds | n/a | n/a | n/a | unavailable (40/40) |

Simulated work is excluded from production proving-compute claims.

## Crash-recovery results and tamper-test results

Crash-recovery results: seven interrupted transition phases, concurrent
writers, and WAL-driven recovery reject ambiguous external prover outcomes.
Publication uses compare-and-swap; exactly one current-root writer wins and the
prior accepted seal remains recoverable. Crash cases pass with zero
stale/simulated acceptance.

Tamper-test results: single-field cache-context mutations, unauthorized
deletion, changed manifest with old aggregate, wrong parent, missing
invalidated unit, missing unaffected leaf, duplicate/reordered leaf,
corruption, and cache poisoning fail closed with typed reasons. Joined
adversarial e2e cases refuse poisoned candidates, stale parent/branch replay,
missing required replacements, simulated/unknown/timeout outcomes, corrupted
artifacts, and racing writers. Stale or mismatched proof acceptance floor is
zero; concurrent stale writer acceptance floor is zero.

## Release validation posture

Protected `--run-release-validation` is the only process authority for terminal
evidence. It:

1. consumes exact `incremental-proof-sealer-materialization-request@1` for
   `IPS-056` on the JSON/log paths
2. materializes a pristine no-local outer/nested source tree
3. refuses any live `ipfs` resolved from the fixed PATH
4. runs `--check-terminal`
5. runs all 17 existing reviewed ZK/reuse/WAL/release suites
6. runs the three new suites:
   - `accelerate-incremental-sealing` → `test/api/incremental_sealing`
   - `datasets-incremental-sealing` → `tests/unit/logic/zkp/incremental_sealing`
   - `kit-proof-seal-store` → `tests/proof_seal_store`
7. secret-scans the combined public log under `public-full-log-secret-scan@1`
8. writes `incremental-proof-sealer-release-validation@2` with
   `assurance.process_observed_only` and
   `test_execution_cryptographically_proven: false`

Existing suites must be green or baseline-compatible-or-improved against the
operator receipts. Every retained skip, xfail, deselection, failure, error,
xpass, or collection issue is labeled `baseline_compatible_non_green` and named
in the evidence marker; it is never hidden as success. The three new suites
must collect completely, exit zero, and show zero
failed/error/xpassed/skipped/xfailed/deselected outcomes.

## Remaining work before production use

Remaining work before production use includes:

1. operational recursive verification backend after a successful capability
   probe, if recursive claims are desired
2. measured (not only estimated) prover wall-clock and GPU metrics on real
   backends
3. production key-ceremony documentation and allowlisted production
   verification keys with recorded setup origin
4. closing the outer `ZKPProof.public_inputs` binding gap to inner verified
   Groth16 public inputs before production reuse of those artifacts
5. raising localized reuse and mixed-history compute reduction to their
   published targets without relaxing invalidation or admission rules
6. operational runbooks for WAL recovery, retention, and concurrent-writer
   operations outside the hermetic conformance matrix

## Migration note

Existing proof receipts and caches migrate by explicit classification only
(accept / adapt / reverify / reject / simulated). There is no assurance
upgrade. Evidence classes stay integrity-only, signed receipt, or direct
execution. See `INCREMENTAL_PROOF_SEALER_MIGRATION.md` and
`INCREMENTAL_PROOF_SEALER_TRUST_MODEL.md`.

## Declared outputs for IPS-056

| Path | Role |
|---|---|
| `docs/architecture/INCREMENTAL_PROOF_SEALER_REPORT.md` | this report |
| `artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json` | closed materialization request, then release receipt |
| `artifacts/agent_supervisor/incremental_proof_sealer/release_validation.log` | closed materialization request, then retained public log |
