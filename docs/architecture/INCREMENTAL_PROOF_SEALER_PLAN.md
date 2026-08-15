# IncrementalProofSealer comprehensive implementation plan

Status: supervisor bootstrap plan for `agent/incremental-proof-sealer-v1`.

This plan delivers a focused, cross-repository subsystem that safely reuses
previously verified proof units when, and only when, their complete dependency
and trust context is unchanged. It does not create a new agent framework, a
general-purpose zkVM, or a generic cache authority.

The narrow intended claim is:

> Repository verification was decomposed into content-addressed proof units.
> Unchanged units were safely reused when their complete dependency and trust
> context remained unchanged. Invalidated units were re-proven, affected Merkle
> branches were updated, and a new seal was generated from an accepted parent
> seal, reducing proving compute without treating stale or simulated evidence
> as current verification.

## 1. Bound source state and isolation

The supervisor control checkout is isolated from the dirty developer trees and
is bound to these fetched canonical revisions:

| Repository | Canonical planning revision | Original developer checkout observed |
|---|---|---|
| `endomorphosis/ipfs_accelerate_py` | `8881344bb2162f3f8d82f22d8348bc0ac7536f95` | `ea11293bb996f052d620eae989f5377a956764b1`, dirty feature branch |
| `endomorphosis/ipfs_datasets_py` | `bd2ff6245ebe476fc744d45c7c66235c92b0e19c` | `a2f5400b7cb89c8481819379a1b7b9959fe81d45`, dirty detached feature state |
| `endomorphosis/ipfs_kit_py` | `5a7a2df8181cfdc33bc19be09989df7ff83f2d4e` | `69091bf8f11a3ef1fb0e04e11a6d8a4c87f3fa78`, clean but behind upstream |

Planning revisions are provenance, not completion evidence. Before IPS-001
through IPS-003 run, the operator-owned baseline capture script executes the
fixed pytest suites and commits canonical, source-bound receipts plus their full
retained logs. Both capture and validation parse the protected, closed JSON
suite registry as data; validation never executes working-tree registry code.
The no-shell inventory providers consume those protected inputs
and inspect code; they do not run pytest or turn static inspection into a passed
test. IPS-004 must synthesize the validated receipts and inventories before any
implementation task can become ready. A changed tested source revision requires
a fresh operator capture and inventory.

No package, optional prover, proving key, verification key, language model,
fixture corpus, or daemon may be installed or downloaded by import. Tests use
existing tools only. Missing optional capability is recorded as typed
`unavailable`, never silently fabricated.

## 2. Executable-code truth known at planning time

These observations are provisional until protected operator receipts and
IPS-001 through IPS-004 commit the current-tree inventory:

- `ipfs_datasets_py.logic.zkp` contains the canonical ZK surface, including a
  Rust Groth16 backend, verification-key registry, circuit/statement helpers,
  test-execution certificate code, and several simulated/structural paths.
- `ipfs_kit_py.proof_certificate_store` is exact-byte CID transport. It does
  not verify cryptography or decide reuse. Kit's Event-DAG fallback is a hash
  commitment explicitly described as non-ZK; no inspected kit test performs
  real proving or directly proves test execution.
- Kit's modern `core/wal` contracts, coordinator, writer, checkpoint, and
  recovery modules are the durability base. Legacy WAL modules, `merkle_clock`,
  the Event-DAG Merkle helper, and pseudo-CID utilities are not proof-seal
  authorities.
- Accelerate already contains proof schedulers, resource scheduling,
  cancellation/process helpers, proof/test receipt adapters, Groth16/ProveKit
  integration surfaces, and proof-reuse caches. Presence or documentation is
  not evidence that recursion, direct execution proving, or production key
  trust is operational.
- Existing receipt, release, cache, and Merkle structures are not automatically
  reusable. Every candidate is classified from executable verification code and
  tests as real proving, simulated/mock, structural validation, integrity only,
  signed assertion, receipt aggregation, or direct execution proof.

Planning-time executable reconnaissance found no reliable recursive verifier in
any repository, so the default design is individually verified leaves plus a
Merkle manifest completeness commitment. In datasets, the existing Rust
arkworks Groth16 v2 circuit directly proves one bounded Horn-style TDFOL
derivation; a manual existing-artifact probe produced a 1,762-byte proof in
21.342593 seconds and verified it in 0.007902 seconds. Groth16 v1 proves a
nonzero public commitment, v3 commits event digest/root/count, and neither
proves pytest execution. Existing v1/v2 key files lack production-origin and
allowlist evidence and therefore remain test-only candidates. The outer
`ZKPProof.public_inputs` metadata is not currently bound to the inner verified
Groth16 public input, an admission gap that must be closed before reuse.

Planning reconnaissance mentioned a 257-result accelerate slice, but its exact
argv and full transcript were not retained. It is therefore non-reproducible
planning prose, not a baseline receipt and not a count that an inventory may
copy. The prior cross-repository 15/10 figures and the isolated 36-file
704/26/2 figures are likewise historical context rather than the current
controlled bundle. The reviewed capture registry now separates four new
observations: `accelerate-proof-focused-core-15`,
`accelerate-proof-focused-wide-36`, `accelerate-proof-reuse-migration`, and
`accelerate-proof-reuse-cross-repo`. Only their config-pinned receipts and full
logs are current evidence; no historical count is reconstructed. These pytest
observations do not prove test execution cryptographically. Independently, the
certification helper that labels a path live/cryptographic checks only a SHA
digest and is not admissible as Groth16 evidence. The reusable scheduler
provides dependency DAGs, leases/fencing, critical-path ordering, resource
admission, and subprocess process-tree termination, but needs proof-seal stages
and GPU-memory propagation.

Historical kit reconnaissance, gathered without installs or downloads, is a set
of separate commands, not one aggregate pytest invocation.  In the original
sibling checkout, the green proof-certificate, reuse-capability, Profile-D,
coordination-store, and modern-WAL slices contributed 12, 5, 2, 7, and 69
passes respectively; the bootstrap slice separately produced 6 passes and 2
failures, and the agent-receipt slice failed collection.  A repeat in the
isolated supervisor layout on 2026-08-11 produced 12, 5, 2, and 69 passes for
the hermetic slices, 7 coordination-store setup errors because its hard-coded
external MCP++ vector path is absent, 8 bootstrap skips because the optional
accelerate pytest plugin is not discoverable in that invocation, and the same
agent-receipt collection error.  An inventory must bind each count to the exact
command, checkout layout, interpreter, duration, exit code, and output digest;
it must never attach the aggregate counts to a command that aborts at
collection. Those numbers remain context until replaced by the nine separate
controlled kit receipts; they cannot be combined under a collection-aborting
command. Kit has no real proving test, signature-verified execution receipt, or
recursive verifier. Non-pass results are recorded, not relabeled as passes.

Historical datasets reconnaissance mentioned focused 177/3, unit-wide
749/31/13, and broad 1180/33/17 observations. They were produced by different
argv/layouts and remain planning prose rather than a current combined result.
The controlled registry instead creates four distinct observations:
`datasets-zkp-focused-current` for the focused current unit/integration paths,
`datasets-zkp-unit-wide-current` for the legacy unit-wide directory,
`datasets-proof-cache-adapters` for CEC/TDFOL/F-logic cache adapters, and
`datasets-zkp-broad-safe-current` for their explicit safe union plus selected
MCP++, wallet, and PDF paths. The focused/broad commands exclude the available-
toolchain Nargo path; the broad command also excludes the installed-multiformats
runtime case. The controlled environment disables Groth16/ProveKit enablement,
builds, downloads, and auto-install paths, so these pytest observations must not
be reported as new real proving. Existing manual Groth16 evidence remains a
separate planning-time probe, not proof that pytest ran. The earlier
`pending-local-run` zero-count record is invalid evidence.
All counts, commands, environment controls, and non-pass nodes are reified by
operator receipts whose canonical digests are pinned in the protected scheduler
configuration. The operator capture is one all-repository operation over fresh
no-local, no-hardlink materialized Git trees at one shared source epoch and
capture ID. Receipt schema `incremental-proof-sealer-baseline-receipt@4` binds
the fixed non-inherited Git environment policy, exact clean source trees and
gitlinks, an empty non-source-input set, interpreter/plugin provenance, each
reviewed pytest argv, and every full retained log. The authority checkout's
incidental redirected caches are not execution evidence; the gate separately
denies ignored task-worktree inputs outside its closed irrelevant cache roots.
These are integrity-protected process observations, not cryptographic proofs
that pytest execution occurred.

IPS-001 through IPS-004 reference those receipts and record key
provenance/classifications so later workers do not rely on this prose.

The operator bootstrap is a two-phase, fail-closed sequence; providers never
capture or pin their own evidence:

1. Commit the reviewed validator, capture script, closed suite registry,
   plan/objectives/taskboard, scheduler with empty receipt pins, tests, and
   ignore policy. Run the phase-specific `--check-bootstrap`; it requires a
   pristine committed tree, exact empty pins/base protected paths, the reviewed
   registry/capture controls, and zero canonical receipts/logs/work entries.
2. Run `python scripts/capture_incremental_proof_sealer_baselines.py capture --repository all`.
   This one-shot all-repository capture executes all 17 suites from one clean
   materialized source epoch and publishes the three receipts/full logs.
3. Run `python scripts/capture_incremental_proof_sealer_baselines.py render-pins`.
   This revalidates the whole bundle and prints, without writing, the canonical
   `operator_baseline_receipts` and exact-file `protected_paths` projection.
4. The operator applies only that printed scheduler projection with
   `apply_patch`; no provider or capture code mutates scheduler authority.
5. Run `python scripts/capture_incremental_proof_sealer_baselines.py validate-only --repository all`,
   then commit the receipts, logs, and exact scheduler pins/protected paths as
   one evidence commit.
6. Run the no-argument/`--check-all` preflight. Only then may IPS-001/002/003
   read the reference-only evidence and create their two declared outputs.

Before IPS-004, preflight requires the receipt epoch to remain current except
for the exact pin/evidence/inventory transition. After IPS-004 commits the
closed `incremental-proof-sealer-trust-baseline@2` authority/proof-class/
aggregation/backend/nonclaim milestone, restarts intrinsically verify those receipts as
historical evidence and validate the captured-to-synthesis lineage without
requiring later implementation HEADs to equal the tested baseline.

Inventory acceptance is fail-closed. The validator recomputes canonical receipt
digests, full-log hashes and sizes, fixed suite/command digests, pytest outcome
counts, exact non-pass nodes, and the schema-bound absence of ignored proof,
key, or build inputs against the config-pinned operator bundle.
Provider-owned inventories contain a reference-only `baseline_evidence` record;
they cannot replace or copy the measurements. Placeholder results such as
`pending-local-run`, zero-count success, prose copied from this plan, static
scans labeled as passed, patterned hashes, or undeclared helper outputs are not
current baseline evidence. Inventories distinguish the bound planning revision
from their worktree parent; the supervisor completion receipt supplies the final
task commit because an artifact cannot truthfully self-embed that future commit.
Individual test paths/functions are classified where proof behavior differs. In
particular:

- accelerate's unsigned `TestPassReceipt` and `ProofReceipt` structures are
  assertions/integrity envelopes, not signed receipts; its cache/admission
  gates are not receipt-aggregation proofs.  Inventory includes proof
  attestation stores/backends, real-Groth16 fixtures, runtime activation and v4
  publication, kernel/prover conformance and fallback paths, metrics and
  benchmark paths, evidence stores, doctor/MCP caches, manual completion and
  release evidence, and canonical CID/repository-forest implementations;
- datasets' test-execution certificate has no receipt-signature verifier,
  `TestPassStatementV1` is a Python predicate rather than an implemented ZK
  aggregation circuit, and callback-based proof-receipt attestation is
  structural unless a real backend is observed.  Inventory includes CEC,
  TDFOL, F-logic, Event-DAG v3, ProveKit FFI, wallet/PDF simulated paths, their
  caches, setup/key generation surfaces, exact key identities, and the v3
  artifact-unavailable state.  Groth16 v2 proves its bounded declared
  derivation but does not prove pytest execution; those axes stay separate;
- kit inventory includes Profile-D policy, MCP++ artifact receipts, Iroh/KITA
  release receipts, joined release-receipt checks, and opt-in Filecoin proving
  parameter download surfaces.  Planned-but-absent `proof_seal_store` code is
  not counted as an existing proof path.  Each test is classified as real,
  simulated, mock, structural, or integrity-only, rather than inferred from a
  suite-level label.

## 3. Single-authority repository boundaries

### `ipfs_datasets_py`: semantic authority

Own exactly one canonical implementation of:

- proof evidence classes and closed proof-unit/status enums;
- `ProofUnit`, `VerificationRequirementManifest`, `VerificationPolicy`,
  repository state, statement, test selector, and complete cache-key schemas;
- canonical proof statements, source/artifact/symbol/test/property identities;
- proof dependency graph edges, reason labels, transitive dependency roots;
- requirement discovery, test/property selection, repository diff
  classification, invalidation rules, and invalidation explanations;
- the canonical leaf/category/repository commitment codec and known vectors.

Proposed package: `ipfs_datasets_py.logic.zkp.incremental_sealing`.

### `ipfs_kit_py`: storage authority

Own exactly one narrow proof-seal storage adapter for:

- immutable proof objects, receipts, admitted verification keys, manifests,
  Merkle nodes, checkpoint/delta seals, tombstones, and invalidation records;
- exact-key candidate cache indexes that never decide acceptance;
- proof-forest persistence and affected-branch updates;
- repository/branch-namespaced current-seal compare-and-swap;
- durable seal-transition WAL, deterministic replay, corruption detection,
  recovery, retention references, and concurrent-writer rejection;
- a mandatory hermetic local store and optional injected IPFS transport.

Proposed package: `ipfs_kit_py.proof_seal_store`. It builds on strict CID and
modern `core.wal` primitives, not legacy proof caches or Event-DAG trees.

### `ipfs_accelerate_py`: execution authority

Own exactly one orchestration implementation for:

- proof adapter discovery, real backend probing, proof verification and cache
  admission, signed-receipt verification, and trust/key allowlists;
- full-versus-incremental planning and full-checkpoint fallback decisions;
- proving and verification scheduling, resource admission, cancellation,
  deadlines, process-tree termination, and bounded output;
- recursive aggregation only after an operational backend-specific safety
  probe, otherwise precisely labeled Merkle manifest aggregation;
- checkpoint creation, delta transition execution, seal verification,
  compaction, explanations, CLI, and full-versus-incremental measurements.

Proposed package:
`ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing`.

Kit never decides whether a cached proof is valid. Datasets never persists a
second proof cache. Accelerate never invents a competing CID, manifest, Merkle,
or storage authority.

## 4. Required proof semantics

The datasets schema exposes a closed discriminated union. No generic
`zk_verified: true` field may erase the distinction.

| Evidence class | Establishes | Explicitly does not establish |
|---|---|---|
| `IntegrityCommitment` | exact bytes, digest, CID, and Merkle inclusion | execution or semantic correctness |
| `SignedExecutionReceipt` | an allowlisted signer asserted execution; receipt integrity and signature validity | independent proof that execution occurred without trusting the signer |
| `ReceiptAggregationZkProof` | admitted committed receipt fields satisfy the aggregation circuit; exact required receipt set/count/order has no blocking circuit status | underlying tests ran unless signature verification and signer trust are inside the declared statement |
| `DirectExecutionProof` | the declared program/verifier ran inside the proof system over committed inputs and produced the committed output/property | correctness beyond that exact program, inputs, outputs, and proof-system assumptions |
| `IncrementalCommitSeal` | an accepted parent, explicit state transition, valid reused/replacement leaves, complete new manifest, and new repository verification root | arbitrary repository correctness or direct test execution unless child leaves prove it |

`ProofMode` is separately closed to
`direct_execution_proof`, `theorem_certificate`, `signed_receipt`,
`receipt_aggregation`, `integrity_only`, and `simulated`.

Only direct execution evidence may use direct-computation claim language.
Receipt aggregation must say that it establishes consistency and completeness
of admitted receipts. A simulated required unit forces `simulated_only` and can
never produce `sealed_full` or `sealed_incremental` under production policy.

## 5. Canonical schemas

### 5.1 `ProofUnit@1`

The closed immutable model includes, at minimum:

```text
proof_unit_id                 proof_unit_kind
repository_id                source_root_cid
repository_state_cid         source_closure_schema_version
source_artifact_cids          source_symbol_ids
test_ids                      property_id
statement_cid                 public_input_cid
private_input_commitment      dependency_unit_ids
dependency_unit_roots         environment_cid
dependency_lock_cid           tool_or_prover_id
tool_or_prover_version        circuit_id
circuit_version               proving_key_id
verification_key_id           configuration_cid
fixture_cids                  network_policy_cid
test_selector_cid             policy_cid
canonicalization_version      dependency_graph_schema_version
proof_system_id               evidence_class
proof_schema_version          required_for_seal
risk_class                    proof_mode
terminal_status               proof_object_cid
receipt_cid                   logical_epoch
```

`source_root_cid` is deliberately the proof unit's complete relevant source-
closure root, not the repository-wide tree root. This is the root against which
a different-source-root candidate is rejected. The manifest and seal separately
bind `repository_state_cid`, the repository-wide source root, and revision. Thus
an unrelated repository edit can preserve a unit's source-closure root and allow
reuse, while any relevant imported/transitive source change rejects it.

Non-applicable fields use one canonical typed absence representation; they are
never silently omitted. `logical_epoch`, not wall-clock creation time, enters
deterministic commitments. Any operational timestamp is envelope metadata and
is excluded from the canonical unit identity.

Required `ProofUnitKind` values are `static_analysis`, `type_check`,
`unit_test`, `integration_test`, `property_test`, `formal_obligation`,
`direct_zk_computation`, `receipt_aggregation`, and `release_invariant`.

Closed unit terminal statuses distinguish `proved`, `disproved`,
`integrity_verified`, `signed_assertion_verified`, `not_modeled`, `failed`,
`proof_failed`, `unknown`, `timeout`, `unavailable`, `cancelled`, `invalid`,
`simulated`, and `stale`. Acceptance is statement- and mode-specific:
`integrity_verified` satisfies only an integrity requirement;
`signed_assertion_verified` satisfies only a policy-admitted trusted assertion;
and `proved` carries only the exact theorem/direct/aggregation statement. No
generic `passed` status upgrades these classes.

### 5.2 `VerificationRequirementManifest@1`

The manifest binds repository/revision/source root, exact sorted required unit
IDs and unit descriptor CIDs, policy and selector CIDs, environment and lock
CIDs, schema/canonicalization/graph versions, permitted removals with policy
authorization, logical epoch, and manifest CID/root. Duplicate IDs or non-
canonical order are rejected rather than normalized after receipt.

### 5.3 Complete cache key

`ProofCacheKey@1` binds all of:

```text
statement CID
public-input CID
applicable private-input commitment
sorted source artifact CIDs
sorted dependency proof-unit roots
environment CID
dependency-lock CID
sorted fixture CIDs
tool/prover ID and version
proof-system ID, evidence class, unit kind, and proof mode
circuit ID and version
proving-key ID
verification-key ID
configuration CID
network-policy CID
proof-schema version
canonicalization version
test-selector CID
policy CID
```

The key builder rejects missing, duplicate, noncanonical, unknown, or secret
fields. Cache lookup returns an immutable candidate only. Accelerate rehashes
the candidate, recomputes the complete key, checks kind/mode/status and policy,
verifies its signature or cryptography with an allowlisted key, then admits it.
No target-file-only shortcut exists.

## 6. Dependency graph and invalidation

The directed graph stores `(from, to, edge_type, reason_cid)` with closed edge
types `source_depends_on`, `imports`, `calls`, `schema_depends_on`,
`test_covers`, `fixture_depends_on`, `config_depends_on`, `proof_depends_on`,
`aggregate_contains`, `supersedes`, and `invalidates`.

Direction is normative: `from` is the prerequisite and `to` is the dependent.
For example, source symbol -> static-analysis unit -> covered test -> formal
obligation -> aggregate. Invalidation walks forward from a changed prerequisite
to dependants; a proof unit's dependency-root calculation walks incoming edges
back to all statement-relevant prerequisites. `test_covers` is source -> test,
`aggregate_contains` is child -> aggregate, and `supersedes` is old -> new.

Dependency roots commit to every transitive node and reason relevant to the
statement. Unknown or truncated closure broadens invalidation or requires a
full checkpoint; it never narrows reuse.

Required rules:

- Source implementation change invalidates directly bound units, interface
  dependants when the public interface changes, covering tests, dependent
  formal properties, and affected aggregates only.
- Test source change invalidates that test unit, selector manifests containing
  it, and affected aggregates.
- Deleted tests create explicit manifest removals and tombstones. The new seal
  requires current-policy authorization for every removed required unit.
- Added selected tests become required new units and must be proven.
- Dependency-lock change invalidates every unit whose dependency/environment
  closure uses the changed lock; policy may require a full checkpoint.
- Fixture or configuration change invalidates every unit whose statement or
  execution context binds it.
- Circuit or proving/verification-key change invalidates all users and, under
  the default policy, requires a full checkpoint unless an admitted formal key
  migration proof exists.
- Canonicalization or dependency-graph schema change requires a full checkpoint
  unless an explicit migration proof is implemented and admitted.
- Environment-policy change requires a full checkpoint. Other policy changes
  recompute the exact required manifest and invalidate old aggregate decisions;
  trust/schema changes fall back to full.
- Documentation-only changes preserve execution proofs unless the document is
  a checked specification, generated input, fixture, or policy artifact.

`explain_invalidation` reports the changed key fields, direct reasons,
transitive graph paths, affected aggregates, and fallback policy. `explain_reuse`
reports every equal cache-key field and the fresh verification/admission
evidence; it never says merely “file unchanged.”

### 6.1 Proof-unit granularity and stable IDs

- Static-analysis and type-check units use the smallest module or symbol closure
  for which the selected tool can soundly report an independent result.
- Unit, integration, and property tests use one collected node plus canonical
  parameter case; a parametrized test therefore yields independently cacheable
  units.
- Formal units use one declared property/obligation and theorem-certificate
  statement. Direct ZK units use one fixed program/circuit/statement profile.
- Receipt aggregates and release invariants are explicit dependent units; they
  are never silently folded into test leaves.

`proof_unit_id` is the stable logical identity CID over repository ID, unit kind,
canonical locator/selector or property ID, and proof-unit identity schema. It
excludes the current source closure, repository state, proof object, status, and
logical epoch. Those changing values belong to the unit descriptor/cache key.
Thus an edited test at the same canonical node retains its logical ID but needs
a new proof; a renamed/deleted node is an explicit remove plus add.

## 7. Deterministic Merkle proof forest

The canonical datasets codec uses the existing strict content identity provider
and domain-separated hashes. It defines exact encoding for empty, leaf, unary,
and binary nodes. Leaves are sorted by canonical proof-unit ID bytes within a
category. Duplicate unit IDs, duplicate leaf positions, unknown categories,
and caller-provided noncanonical order are rejected.

```text
RepositoryProofForest
├── source_integrity_root
├── static_analysis_root
├── type_check_root
├── unit_test_root
├── integration_test_root
├── property_test_root
├── formal_obligation_root
├── direct_zk_root
├── receipt_aggregation_root
└── release_invariant_root
```

Each category root commits to the exact sorted proof-unit leaves. The repository
proof root commits to source root, required-manifest root, every named category
root including empty roots, environment root, policy root, proof schema,
canonicalization and dependency-graph versions, exact parent seal CID or
canonical genesis value, current repository revision, and all repository parent
revision IDs.

Kit persists immutable nodes and path indexes and recomputes every changed path.
An incremental update must prove by equality tests that every unaffected leaf
survived. A changed manifest paired with an old aggregate or a dropped
unaffected leaf is rejected.

## 8. Seal and transition model

### 8.1 Full checkpoint

`create_full_checkpoint` discovers and verifies/proves every required unit,
builds all category roots and the complete manifest, verifies the result, writes
immutable artifacts, and CAS-publishes a `FullCheckpointSeal`.

A full checkpoint is mandatory for genesis, canonicalization/schema/graph
changes, circuit or key changes, environment trust-policy changes, uncertain
cache integrity, explicit policy, configured release qualification, and any
case in which incremental reuse cannot be justified. Policy can additionally
require full checkpoints for dependency-lock classes, every N commits, low
reuse ratio, or excessive delta-chain depth.

### 8.2 Delta seal

`DeltaSeal@1` binds one exact accepted parent seal, the declared repository
parent set, old/new repository and source roots, complete diff algorithm and
changed-artifact commitment, old/new manifest roots, reused/replaced/added/
removed unit sets, removal authorizations, old/new forest roots, updated branch
proofs, aggregation evidence, policy/environment/schema/key identities, logical
epoch, revision, and transition evidence.

Verification establishes all fourteen normative invariants:

1. parent seal is accepted under current policy;
2. old root matches the declared parent state;
3. new root matches current source state;
4. changed-artifact set is complete for the bound diff algorithm;
5. every invalidated unit has a newly admitted proof;
6. every reused unit has an unchanged complete cache key and freshly verified evidence;
7. every deleted required unit is explicitly and permissibly removed;
8. every added required unit is present and proven;
9. the new required-unit manifest is complete;
10. the new forest root commits to the exact unit set;
11. no stale or mismatched proof was reused;
12. no blocking, unknown, unavailable, invalid, cancelled, timeout, or simulated unit passed;
13. the seal binds the exact parent seal;
14. branch/parent/revision binding prevents replay against an unrelated history.

Merge commits declare all repository parents. Incremental transition from one
accepted selected parent is allowed only when the diff algorithm and complete
manifest resolve the merge without ambiguity and every other parent identity is
bound. Otherwise the planner requires a full checkpoint. Rollbacks create new
parent-bound transitions; an old seal cannot be replayed merely because the
source root repeats.

### 8.3 Atomic flow

```text
load and verify parent seal
  -> compute complete repository diff
  -> rebuild required-unit manifest
  -> calculate transitive invalidation closure
  -> load immutable cache candidates
  -> recompute keys and verify candidate proofs
  -> prove invalidated/added units
  -> verify every new proof
  -> update affected forest branches
  -> build bounded aggregate or manifest-completeness evidence
  -> verify transition evidence and new seal
  -> persist new seal
  -> compare-and-swap current seal root
  -> commit/clean transition WAL
```

No current pointer changes until every required changed unit passed, every
reused unit was revalidated, all aggregate roots are current, transition
evidence verifies, and CAS succeeds.

## 9. Persistence, WAL, and recovery

The kit adapter exposes closed artifact kinds: `proof_object`, `proof_receipt`,
`verification_key`, `proof_manifest`, `merkle_node`, `checkpoint_seal`,
`delta_seal`, `tombstone`, and `invalidation_record`. A public API rejects
proving keys and witness material. Every read rehashes bytes. Mutable indexes
are hints and are rebuildable from immutable admitted records.

The local hermetic implementation uses explicit test roots, atomic file/SQLite
transactions, file and parent-directory durability, and a repository/branch
namespace. Optional IPFS transport is injected, bounded, content rehashed, and
never needed by unit tests or ordinary import.

WAL phases and expected recovery:

| Failure point | Recovery disposition |
|---|---|
| before proof execution | resume the unstarted job |
| after proof execution, before receipt persistence | never guess; verify a durable prover artifact if present, otherwise mark ambiguous and reprove/discard under policy |
| after receipt persistence, before forest update | rehash and verify the receipt/proof, then replay forest update |
| after forest update, before aggregate generation | verify persisted branch nodes and resume aggregation |
| after aggregate generation, before seal persistence | verify existing aggregate and resume seal creation |
| after seal persistence, before current-root CAS | verify seal and retry CAS only if the expected parent remains current; otherwise `stale_parent` |
| after CAS, before transaction cleanup | detect that current pointer equals the new seal, finalize commit/cleanup idempotently |

Corrupt tails preserve valid committed prefixes. Ambiguous external prover
outcomes never become success. Recovery is deterministic and repeatable.

## 10. Proving, verification, and aggregation

Accelerate uses its modern scheduler and resource primitives for a job DAG with
CPU, memory, optional real GPU, process, deadline, cancellation, and bounded
fan-in leases. Priority order is:

1. cheap invalidation and completeness checks;
2. cached-proof retrieval and cryptographic/signature verification;
3. small independent required units;
4. units on the critical aggregate path;
5. expensive direct-execution proofs;
6. full reproving only when required.

Provider commands are fixed registry entries and explicit argv, never dynamic
paths or shell strings. Cancellation kills the admitted process tree and fences
late output from cache or seal publication. Timeout, resource rejection, and
unavailable backend remain closed statuses.

IPS-029 must execute a backend-specific capability probe. Recursive aggregation
is implemented only for a backend already capable of verifying its child proofs
under stable circuit/key assumptions. Such an aggregate binds child validity,
exact unit IDs, count, duplicate rejection, deterministic order, child root,
terminal status, repository, environment, and policy.

If no backend passes that probe, the required implementation is:

- individually verifiable proof leaves;
- a deterministic Merkleized proof manifest;
- integrity/completeness transition evidence over the manifest;
- the explicit label `manifest_aggregation`, never recursive verification.

Bounded fan-in is leaf -> small batch -> category -> repository. Only affected
branches are rebuilt. The optional direct incremental-verifiable-computation
demonstration is permitted only if the capability probe succeeds; otherwise the
report records `not_implemented_backend_unavailable` without adding a new IVC
framework.

## 11. Public API and CLI

The accelerate facade provides:

```python
create_full_checkpoint(repository_state, verification_policy) -> FullCheckpointSeal
create_incremental_plan(parent_seal, old_repository_state, new_repository_state, verification_policy) -> IncrementalProofPlan
execute_incremental_plan(plan, resource_policy) -> IncrementalProofResult
verify_seal(seal, trusted_keys, verification_policy) -> SealVerificationResult
explain_reuse(seal, proof_unit_id) -> ProofReuseExplanation
explain_invalidation(plan, proof_unit_id) -> ProofInvalidationExplanation
compare_full_and_incremental(repository_state, parent_seal, verification_policy) -> ProofCostComparison
compact_seal_chain(current_seal, retention_policy, verification_policy) -> FullCheckpointSeal
```

The focused `zk-seal` CLI exposes `full`, `incremental`, `verify`, `plan`,
`explain-reuse`, `explain-invalidation`, `benchmark`, `cache-status`,
`force-full`, and `compact`. Imports remain lazy and side-effect free.

Closed seal outcomes are `sealed_full`, `sealed_incremental`,
`verification_failed`, `proof_failed`, `unknown`, `timeout`, `unavailable`,
`stale_parent`, `invalid_cache`, `incomplete_manifest`,
`full_reproof_required`, `cancelled`, and `simulated_only`.

## 12. Deterministic fixtures and tests

Fixture repositories and graphs cover:

- independent implementation edit and public-interface edit;
- test edit, authorized/unauthorized deletion, and selected addition;
- fixture, dependency lock, configuration, circuit, proving key, verification
  key, selector, policy, network policy, canonicalization, and environment edits;
- documentation-only and checked-specification documentation edits;
- two independent modules, wrong-parent branch, merge, rollback, and repeated root;
- concurrent writers and all seven interrupted transition phases;
- corrupt candidate, cache poisoning, missing required unit, duplicate leaf,
  reordered manifest, stale parent, old aggregate, lost unaffected leaf, and
  simulated evidence presented as real;
- valid unchanged proof reuse after an unrelated edit.

Critical negative tests mutate one authority field at a time and reject:

- source root, environment, selector, verification key, circuit, dependency
  closure, public input, policy, fixture, configuration, or network mismatch;
- valid format with invalid cryptography, unknown proof system, unallowlisted
  verification key, unsigned required receipt, and invalid signer;
- unauthorized test removal, changed manifest with old aggregate, wrong parent,
  missing invalidated unit, simulated production unit, unknown/timeout treated
  as pass, lost unaffected leaf, and stale CAS writer;
- arbitrary circuit/executable paths, proof-before-verification cache admission,
  proving-key/witness disclosure, path escape, symlink substitution, oversized
  artifacts, duplicate JSON keys, and non-finite canonical values.

Positive tests never use simulated proof success as production evidence.
Structural and simulated fixtures are explicitly labeled and test only rejection
or non-production plumbing. Real backend tests run only when the already-present
backend and reviewed test-only keys are operational; absence is an honest skip
or `unavailable`, never production evidence.

## 13. Forty-transition benchmark

The controlled benchmark uses a deterministic 40-transition history. Rejected
wrong-parent/poison attempts are side observations and do not masquerade as
accepted commits.

| Transition | Scenario | Expected checkpoint behavior |
|---:|---|---|
| 00 | initial repository | full genesis |
| 01 | localized private source edit | incremental |
| 02 | unrelated documentation | incremental, near-total reuse |
| 03 | one test-source edit | incremental |
| 04 | one fixture edit | incremental |
| 05 | unrelated module edit | incremental |
| 06 | public-interface edit | broader incremental closure |
| 07 | dependent module edit | incremental |
| 08 | selected test addition | add/prove leaf |
| 09 | authorized test deletion | tombstone/removal proof |
| 10 | relevant configuration edit | incremental invalidation |
| 11 | ordinary documentation | incremental, near-total reuse |
| 12 | dependency-lock class upgrade | policy full checkpoint |
| 13 | localized source edit | incremental |
| 14 | two independent module edits | parallel incremental |
| 15 | branch A edit | parent-bound incremental |
| 16 | branch B edit from prior accepted parent | parent-bound incremental |
| 17 | merge A/B | incremental only if complete, else full fallback |
| 18 | rollback of source bytes | new parent-bound delta, no replay |
| 19 | property-test edit | incremental |
| 20 | periodic N-commit checkpoint | full checkpoint |
| 21 | documentation-only | incremental |
| 22 | circuit version change | full checkpoint |
| 23 | localized source edit | incremental |
| 24 | verification-key change | full checkpoint |
| 25 | test-selector change | recompute manifest/invalidate |
| 26 | network-policy change | affected units invalidated |
| 27 | environment trust-policy change | full checkpoint |
| 28 | integration fixture edit | incremental |
| 29 | requirement policy change | manifest recompute/fallback as classified |
| 30 | periodic checkpoint | full checkpoint |
| 31 | integration-test addition | add/prove leaf |
| 32 | proof schema/canonicalization change | full checkpoint |
| 33 | checked-specification document edit | invalidate bound units |
| 34 | ordinary documentation edit | incremental, near-total reuse |
| 35 | injected cache corruption detection | full checkpoint required |
| 36 | two independent modules | parallel incremental |
| 37 | wrong-parent delta attempt then valid transition | reject attempt; accept bound transition |
| 38 | merge plus unaffected proof reuse | bounded incremental or honest fallback |
| 39 | release-tag qualification and chain compaction | full checkpoint |

For every accepted transition the harness runs or explicitly estimates both
full and incremental work and records measurement provenance. Metrics are:
required/reused/invalidated/added/removed/newly proved units, hit rate, leaf
proving time, aggregation time, total prover CPU and GPU time, peak memory,
proof/seal size, storage growth, seal verification latency, wall time, full and
incremental cost, percentage compute saved, chain depth, and fallback reason.

Targets are not facts: >=70% reuse for localized commits, >=50% mixed-history
proving-compute reduction, and >=80% reduction for ordinary docs/unrelated
edits. A metric based on a cost model is labeled `estimated`; a simulated unit
is excluded from production proving-compute claims. Best, worst, fallback, and
unavailable cases are all reported.

IPS-053 cannot satisfy this contract by authoring JSON/CSV, and its sealed
provider has no process authority. The provider therefore writes both declared
paths as exact closed `incremental-proof-sealer-materialization-request@1`
markers. Protected `--run-benchmark` Validation consumes only that complete
request, invokes the IPS-052 CLI with seed `20260811`, count `40`, and the two
exact output paths using direct argv, then runs the artifact gate. Before
consuming the request it rejects every staged, unstaged, untracked, or ignored
candidate-worktree mutation outside those two exact paths and the fixed ignored
`release-work` root, including skip-worktree/assume-unchanged index flags and
link or special-file substitution. It then makes no-local clones of the exact
outer and nested HEAD/gitlink commits inside that root, removes every remote,
rejects object alternates, replacement refs, and grafts, materializes only the
two owned nested repositories, and requires every deliberately absent non-owned
gitlink to match a closed reviewed path-to-object-ID allowlist. Unknown,
missing, present-on-disk, or object-ID-drifted gitlinks reject. It hashes every
physical tracked leaf against its Git blob and makes the verified source
read-only through held no-follow descriptors.
The CLI runs only there and writes to a sibling staged-output tree; the runner
rehashes the source, ingests exactly two bounded regular files through held
directory descriptors, and atomically publishes them to the candidate paths.
Pre/post candidate binding catches concurrent checkout/index mutation, while
the isolated clone prevents such mutation from becoming executed code. An
absent request never starts benchmark work. Each JSON/CSV artifact is bounded to one
MiB and the task declares the exact reviewed proposal envelope. The one
allowed stabilization invocation recognizes the complete valid bundle and is
read-only, so candidate hashes converge; partial, unknown, or invalid existing
bytes fail closed. `benchmark.json` is canonical
`incremental-proof-sealer-benchmark-results@2`; it binds the exact
`benchmark_worktree_parent_revision` and source revisions/trees. Candidate
validation requires that parent to be current; dependent analysis proves the
exact output-only candidate child and no-ff integration merge, then admits only
byte-exact daemon status-only descendants while keeping evidence blobs
unchanged. Each row binds the exact ordered scenario, observed planner counts,
complete arithmetic, full/incremental roots, checkpoint/fallback status,
wrong-parent side observation, zero production simulated units, and per-metric
`measured`/`estimated`/`unavailable` provenance. `benchmark.csv` is an exact
ordered scalar projection. IPS-054's canonical summary binds the raw JSON
SHA-256 and is recomputed from those rows; targets remain comparisons, never
retrofitted observations.

## 14. Checkpoint policy and chain compaction

Policy parameters include full checkpoint every N accepted seals, release tags,
circuit/key/lock classes, corruption, minimum reuse ratio, and maximum chain
depth. Defaults fail closed.

Compaction verifies the complete chain, current required manifest, and every
current proof unit; rebuilds a full forest; writes and verifies a new checkpoint;
CAS-publishes it; retains historical seal references and all evidence required
by retention policy. It does not rewrite history or silently delete evidence.

## 15. Security and privacy gates

- Production mode never generates or downloads proof keys.
- Test-only keys carry a machine-checked `test_only` marker and cannot enter a
  production policy allowlist.
- Verification keys are content addressed, origin documented, and selected
  only by trusted policy; proving keys are never returned by public APIs.
- Every proof is cryptographically/signature verified before cache admission
  and again after lookup; cache indexes cannot bypass verification.
- Receipt signatures and signer allowlists are verified wherever assertions
  contribute to acceptance.
- Unknown proof systems, circuits, tools, executables, keys, and dynamic paths
  are rejected.
- Sensitive witness data is excluded from logs, receipts, Merkle leaves, CLI
  JSON, and public storage. Each direct proof states public and private inputs.
- Optional network/IPFS operations are explicit policy-controlled adapters;
  ordinary imports are hermetic.

## 16. Supervisor dependency waves

```text
Wave 0: protected operator baseline capture, then IPS-001 / 002 / 003 inventories
    -> IPS-004 cross-repository trust and ownership synthesis

Wave 1: three safe per-repository chains begin in parallel:
    datasets proof contracts/identities/manifests/statements
    kit store protocol/local transport
    accelerate evidence/backend/key/prover admission

Wave 2: dependency graph/invalidation
    + kit store protocol/local transport
    + accelerate trust/key adapters

Wave 3: kit cache/forest/WAL/CAS/recovery
    + accelerate planner/scheduler/process fencing
    + deterministic fixtures

Wave 4: proof execution/admission/aggregation
    + full and delta seals/public APIs/compaction

Wave 5: CLI/migration, positive, tamper, signature, crash, and e2e tests

Wave 6: 40-transition benchmark, target analysis, trust docs, final report
```

Tasks declare predicted files and conflict policy. Tasks that modify the same
nested repository are serialized through an explicit dependency/fan-in because
even disjoint nested files produce competing gitlink descendants. Datasets and
kit chains may run in parallel with each other and with accelerate-only work.
Cross-repository tasks must
make and validate independent nested commits before the accelerate gitlink
update. The shared merge queue serializes gitlink publication. Broad snapshots,
dirty nested completion, and edits to this plan/board/config by workers are
forbidden.

The scheduler uses three strict SHA-256 task-ID shards, bounded retry budgets, a
10,800-second implementation/validation outer timeout (below the 14,400-second
hard maximum), a 20-minute log-stall threshold, and current-tree validation. Runtime files are
ignored and are not completion authority. Refill scanning is disabled so the
fixed reviewed board cannot silently expand into unrelated work.

## 17. Completion and final report

IPS-056 may close only after every goal/task dependency is terminal. Its sealed
provider writes exact JSON/log materialization requests and a substantive report
containing exactly one closed release-evidence marker. Protected convergent
`--run-release-validation`, not a provider shell string or provider-authored
pass receipt, consumes only that complete request and observes the
historical terminal board gate, all 17 existing reviewed ZK/reuse/WAL/release
suites, and the three new cross-repository incremental-sealing suites with
explicit argv, bounded timeouts, fixed offline/autoinstall-disabled workspaces,
and one content-bound retained log. Before request consumption the release
runner applies the same closed candidate index/worktree check, no-local
outer/nested clone, remote/alternate rejection, exact per-leaf Git-blob hashing,
and read-only source materialization as the benchmark runner. The terminal gate
and every pytest command run only from that materialization with caches and
temporary state redirected outside it; a post-run full rehash rejects source
mutation, and the release staging channel must remain empty. Only the bounded
secret-scanned receipt/log and single report-marker substitution are published
to the three declared candidate paths. An absent request never starts release
work. Each child is capped at 256 KiB of output; the combined public log must be
strict UTF-8 without NUL bytes and is capped at six MiB, the receipt at two MiB,
and the report at one MiB;
the complete three-file evidence remains below the exact 12-MiB patch and
20-MiB serialized proposal envelope. It writes canonical
`incremental-proof-sealer-release-validation@2`, independently reparses every
pytest summary/non-pass/collection result, binds unchanged source revisions and
trees, rejects any public log containing a private-key header, named secret,
authorization token, GitHub/AWS token, or JWT, and states that process output
observation is not cryptographic proof of test execution, and replaces only the
single report marker with the fresh receipt digest and exact source revisions.
Before starting a suite it also refuses a fixed PATH that resolves a live
`ipfs` executable. Its ignored fixed `release-work` tree is securely cleaned
before and after materialization; final evidence paths remain unignored.
The three new incremental-sealing suites must be fully green, including zero
skipped, xfailed, or deselected outcomes. Each existing
suite must be green or baseline-compatible-or-improved against its protected
operator receipt: no new non-pass identity/category, no increased
failure/error/xpass/skip/xfail/deselection count, no lost passes, consistent
selected/collection arithmetic, and no new collection abort. An
unchanged baseline collection blocker is retained as
`baseline_compatible_non_green`, never renamed success, and the final report
must name every such remaining blocker. Current-tree evidence must establish:

The first IPS-056 Validation materializes only the exact closed request. The
supervisor's one stabilization rerun securely reparses the complete bundle but
neither reruns a benchmark/test subprocess nor changes an output, and therefore
observes identical candidate hashes. Read-only Git binding subprocesses still
run. Partial requests, duplicate/missing report markers,
or provider-authored completed-looking evidence that fails the closed artifact
gate reject.

The release receipt binds the task worktree parent, not its unknowable future
completion commit. Candidate validation requires exact parent equality;
post-commit verification locates the exact output-only candidate and no-ff
integration merge, permits only validated daemon status-only descendants, and
requires all report/receipt/log blobs to remain unchanged.

- public APIs and CLI work through the narrow three-repository boundaries;
- positive invalidation/seal/compaction tests pass;
- every required negative, tamper, wrong-parent, concurrent-writer, and seven-
  phase crash case passes with zero stale/simulated acceptance;
- deterministic roots match across repeated runs;
- focused existing ZK/storage/scheduler regressions are rerun and pre-existing
  unrelated failures remain truthfully separated;
- the 40-transition artifact distinguishes actual versus estimated work;
- final trust documentation states which claims are direct proofs, signed
  assertions, receipt-aggregation statements, theorem certificates, integrity
  commitments, simulated test plumbing, or unavailable;
- exact commits inspected, modules changed, granularity, key fields,
  invalidation/fallback rules, aggregation strategy, benchmark sequence,
  reuse/compute results, sizes, latency, storage, recovery/tamper results, and
  remaining production work are recorded without overclaiming.

The terminal report must never say “the repository was proven correct,” “all
pytest execution was proven in zero knowledge,” “the change is semantically
correct,” or “the tests passed” when only hashes or receipt aggregation were
verified.
