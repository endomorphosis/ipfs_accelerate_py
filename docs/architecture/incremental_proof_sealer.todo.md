# IncrementalProofSealer fixed supervisor board

Reviewed executable projection of `INCREMENTAL_PROOF_SEALER_PLAN.md`.

Global invariants for every task:

- Bound repositories are `ipfs_accelerate_py@8881344b`,
  `ipfs_datasets_py@bd2ff624`, and `ipfs_kit_py@5a7a2df8`, or explicit
  descendant commits produced by this board.
- IPS-001, IPS-002, and IPS-003 run before implementation; IPS-004 is the
  mandatory cross-repository trust/ownership join.
- Do not install dependencies, download/generate proof keys, start a network
  daemon, or mutate user state. Ordinary imports must remain hermetic.
- Never use simulated/mock proof success to satisfy a production seal.
- Datasets owns proof semantics/identity/manifests/invalidation; kit owns
  storage/index/forest/WAL/CAS; accelerate owns proving/planning/scheduling/
  aggregation/sealing/measurement.
- A cross-repository task commits each nested repository independently and
  updates gitlinks through the serialized merge queue. Broad snapshots and
  dirty/detached nested completion are forbidden.
- Protected plan, objective, board, scheduler, validator, and `.gitignore`
  files are operator-owned and may not be edited by implementation workers.
- The operator-owned baseline capture script, canonical receipts, and retained
  pytest logs are protected inputs. The no-shell inventory providers consume
  those receipts; they do not execute pytest, regenerate evidence, or claim
  that static source inspection is an executed test result.
- The one-shot operator capture uses the closed 17-suite registry and clean
  materialized trees for all three repositories at one capture ID. Its `@4`
  receipts bind a fixed non-inherited Git environment and process observation
  only; current task gates deny undeclared ignored inputs outside explicit
  redirected cache roots.
- Bootstrap order is fixed: commit all empty-pin infrastructure; run
  `python scripts/capture_incremental_proof_sealer_baselines.py capture --repository all`;
  run `python scripts/capture_incremental_proof_sealer_baselines.py render-pins`;
  have the operator apply exactly that read-only
  projection to scheduler pins/protected paths; run
  `python scripts/capture_incremental_proof_sealer_baselines.py validate-only --repository all`;
  commit the evidence/pins; then run no-argument/`--check-all` preflight.
  Providers never capture, pin, or rewrite operator evidence.

## IPS-000 Freeze the reviewed source binding and supervisor projection

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: bootstrap
- Depends on:
- Goal id: IPS-G010
- Outputs: docs/architecture/INCREMENTAL_PROOF_SEALER_PLAN.md, docs/architecture/incremental_proof_sealer.objectives.md, docs/architecture/incremental_proof_sealer.todo.md, config/agent_supervisor_incremental_proof_sealer_scheduler.json
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-bootstrap
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/baseline
- Parallel lane: operator
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 0
- Implementation timeout seconds: 300
- Predicted files: none
- Submodules: none
- Interfaces: reviewed fixed board
- Allow concurrent with:
- Conflict policy: Operator-owned planning evidence only; completion is not implementation evidence.
- Preconditions: Canonical upstream revisions fetched and isolated control worktree created.
- Effects: Freezes the 57-task DAG, ownership boundaries, source revisions, and completion policy.
- Evidence subset: ips/planning-projection@1
- Symbolic first: true
- Acceptance: Control documents are internally consistent, source-bound, reviewed, and do not assume unrelated semantic-index/capsule/cache work.
- Embedding query: incremental proof sealer fixed board source binding

## IPS-001 Inventory accelerate proof backends, tests, receipts, schedulers, and baselines

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-accelerate
- Depends on: IPS-000
- Goal id: IPS-G010
- Outputs: docs/architecture/incremental_proof_sealer_inventory/accelerate.json, docs/architecture/incremental_proof_sealer_inventory/accelerate.md
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-001
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/baseline
- Parallel lane: baseline-accelerate
- Resource class: cpu-medium
- Resource stage: reconnaissance
- Estimated tokens: 14000
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/incremental_proof_sealer_inventory/accelerate.json, docs/architecture/incremental_proof_sealer_inventory/accelerate.md
- Submodules: none
- Interfaces: executable proof inventory
- Inputs: config/incremental_proof_sealer_baseline_suite_registry.json, artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/accelerate.json, its receipt-declared retained logs, the protected scheduler digest pin, and read-only accelerate source at the bound planning lineage
- Allow concurrent with:
- Conflict policy: Read only existing bound code and protected operator evidence; the provider has no shell authority and must not run pytest, regenerate/edit receipts or logs, install, download, build, create keys, or write outside the two declared inventory outputs.
- Preconditions: The accelerate task starts at one clean committed parent descended from the receipt-tested control revision solely through the exact operator pin/evidence transaction and previously admitted inventory/status transactions; that task-start parent remains embedded after publication.
- Effects: Records every Groth16/ProveKit/ZK/formal/simulated adapter, scheduler/resource/cancel path, receipt/cache/seal/CLI path, and key assumption from static inspection, while adding only a reference to the separately captured operator pytest receipt.
- Evidence subset: ips/accelerate-inventory@2
- Symbolic first: true
- Acceptance: Record `planning_revision` and the immutable `inventory_worktree_parent_revision`, with the latter exactly equal to the accelerate task-start parent during candidate validation and remaining that parent after publication; the final task commit is supplied by the supervisor completion receipt and is not self-embedded. A proposed candidate dirties exactly the two declared regular-file outputs at that parent. A committed result is accepted only when one direct output-only child of the embedded parent is integrated by one exact two-parent no-ff merge, both output blob IDs equal the reviewed candidate blobs, neither output changes afterward, and a later exact Implementation Daemon board-only `<IPS-ID>: mark todo completed` commit completes this task with both outputs already present. The captured tested revision may advance to the task-start parent only through the semantic scheduler receipt-pin/protected-path projection, the exact pinned receipt/log files, already-merged declared inventory documents/gitlinks whose nested deltas are inventory-document-only, and exact Implementation Daemon status commits. Each admitted board commit is replayed chronologically, changes only canonical `Status` bytes monotonically to `completed`, names a transitioned task, and has dependencies and outputs present at that commit. Any other plan, objectives, taskboard, validator, capture script, closed suite registry, `.gitignore`, source, test, dependency, scheduler, candidate, merge, output-blob, or status-lineage change rejects and requires recapture or reimplementation. This gate covers committed history and current-tree integrity; it does not prove read-only execution isolation against transient uncommitted edits. `baseline_evidence` is the exact reference-only projection `{path, receipt_digest, required_command_ids, evidence_origin=operator_capture, assurance=process_observed_only, nonclaim=pytest_execution_not_cryptographically_proven}` for `accelerate-proof-focused-core-15`, `accelerate-proof-focused-wide-36`, `accelerate-proof-reuse-migration`, and `accelerate-proof-reuse-cross-repo`; no command/result/count/log/transcript alias or execution-success claim is permitted elsewhere. The validator parses the protected closed JSON suite registry without executing it, independently recomputes exact suite preimages/digests, argv and controlled-offline environment, and checks pinned canonical receipt/log/count/non-pass evidence. Static inspection reports `surfaces_found`, never pass/fail; placeholders, plan-derived counts, patterned hashes, undeclared helpers, and extra outputs reject. Every proof/attestation backend and store, real-Groth16 fixture, runtime/v4 publication path, kernel/prover/fallback path, metric/benchmark/evidence store, doctor/MCP cache, manual/release seal, CID/canonicalization/Merkle path, scheduler, and focused test is classified. Unsigned TestPassReceipt/ProofReceipt values are not called signed, cache admission is not receipt aggregation, and direct-execution/recursion/setup claims are tied to executable evidence.
- Embedding query: accelerate zero knowledge provekit groth16 proof scheduler receipt real simulated test baseline

## IPS-002 Inventory datasets ZK, identity, manifest, dependency, and baseline paths

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-datasets
- Depends on: IPS-000
- Goal id: IPS-G010
- Outputs: ipfs_datasets_py/docs/architecture/incremental_proof_sealer_inventory.json, ipfs_datasets_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-002
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/baseline
- Parallel lane: baseline-datasets
- Resource class: cpu-medium
- Resource stage: reconnaissance
- Estimated tokens: 16000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_datasets_py/docs/architecture/incremental_proof_sealer_inventory.json, ipfs_datasets_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md
- Submodules: ipfs_datasets_py
- Interfaces: executable proof and identity inventory
- Inputs: config/incremental_proof_sealer_baseline_suite_registry.json, artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/datasets.json, its receipt-declared retained logs, the protected scheduler digest pin, and read-only datasets source at the bound planning lineage
- Allow concurrent with:
- Conflict policy: Documentation inventory only from read-only code and protected operator evidence; the provider has no shell authority and must not run pytest, regenerate/edit receipts or logs, build the Rust backend, provision keys, auto-install, trust archived claims, or write outside the two declared inventory outputs.
- Preconditions: The datasets task starts at the clean receipt-tested nested source revision, which is embedded as the immutable inventory parent and remains the candidate commit's direct parent after publication.
- Effects: Records ZK backends/circuits/keys/statements, proof/test receipts, caches, canonicalization/CID/Merkle paths, real/simulated test surfaces, and ownership candidates from static inspection, while adding only a reference to the separately captured operator pytest receipt.
- Evidence subset: ips/datasets-inventory@2
- Symbolic first: true
- Acceptance: Record `planning_revision` and immutable `inventory_worktree_parent_revision`; the latter equals the receipt-tested datasets source revision. Candidate validation requires current datasets HEAD to equal that parent and exactly the two declared outputs to be dirty. A committed result requires one direct nested two-output child of that parent, one direct outer candidate changing only the datasets gitlink from that parent to the nested child, one exact two-parent no-ff merge preserving that gitlink, no later gitlink rewrite, and a later exact Implementation Daemon board-only `IPS-002: mark todo completed` commit with both outputs already present. The final nested/outer/status commits come from supervisor completion evidence and are not self-embedded. Outer committed-history relevance admits only the exact operator bundle/inventory transactions and chronologically replayed board-only status commits whose sole byte changes are monotonic canonical `Status` values with dependencies and outputs already present; every other taskboard/control/source/candidate/merge/blob change rejects. This is committed-history and current-tree integrity, not proof of isolation from transient uncommitted edits. `baseline_evidence` is the exact reference-only operator projection for `datasets-zkp-focused-current`, `datasets-zkp-unit-wide-current`, `datasets-proof-cache-adapters`, and `datasets-zkp-broad-safe-current`, with fixed path/digest/IDs plus `evidence_origin=operator_capture`, `assurance=process_observed_only`, and `nonclaim=pytest_execution_not_cryptographically_proven`; no copied command/result/count/log/transcript alias or execution-success claim is permitted elsewhere. The validator binds the protected reviewed registry and independently recomputes suite/argv/environment/digest/log/count/non-pass evidence. Static scans report `surfaces_found`, never pass/fail. Placeholder, plan-derived, patterned, missing/swapped/duplicate-suite, helper, stale-source, or extra-output evidence rejects. Inventory CEC, TDFOL, F-logic, Event-DAG v3, ProveKit FFI, wallet/PDF simulated paths, all proof caches, setup/key-generation and exact key identity/provenance surfaces, and individual focused tests with valid repository-relative paths. Test-execution certificates without signature verification are not signed receipts; TestPassStatementV1 is not an implemented ZK circuit; callback attestation is structural unless a real backend ran; Groth16 v2's bounded computation-proof axis is distinct from pytest-execution proof; absent v3 artifacts and the reduced-field digest binding are explicit.
- Embedding query: datasets zkp groth16 canonicalization cid merkle proof cache receipt tests real simulated

## IPS-003 Inventory kit proof storage, CID, Merkle, WAL, CAS, and baseline paths

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-kit
- Depends on: IPS-000
- Goal id: IPS-G010
- Outputs: ipfs_kit_py/docs/architecture/incremental_proof_sealer_inventory.json, ipfs_kit_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-003
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/baseline
- Parallel lane: baseline-kit
- Resource class: cpu-medium
- Resource stage: reconnaissance
- Estimated tokens: 14000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_kit_py/docs/architecture/incremental_proof_sealer_inventory.json, ipfs_kit_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md
- Submodules: ipfs_kit_py
- Interfaces: executable storage and durability inventory
- Inputs: config/incremental_proof_sealer_baseline_suite_registry.json, artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/kit.json, its receipt-declared retained logs, the protected scheduler digest pin, and read-only kit source at the bound planning lineage
- Allow concurrent with:
- Conflict policy: Documentation inventory only from read-only code and protected operator evidence; the provider has no shell authority and must not run pytest, regenerate/edit receipts or logs, start IPFS, auto-install, perform proof setup, use pseudo-CID/legacy WAL as authority, or write outside the two declared inventory outputs.
- Preconditions: The kit task starts at the clean receipt-tested nested source revision, which is embedded as the immutable inventory parent and remains the candidate commit's direct parent after publication.
- Effects: Records strict/pseudo CID paths, proof transport, receipts, Merkle helpers, modern/legacy WAL, CAS candidates, and corruption/recovery behavior from static inspection, while adding only a reference to the separately captured operator pytest receipt.
- Evidence subset: ips/kit-inventory@2
- Symbolic first: true
- Acceptance: Record `planning_revision` and immutable `inventory_worktree_parent_revision`; the latter equals the receipt-tested kit source revision. Candidate validation requires current kit HEAD to equal that parent and exactly the two declared outputs to be dirty. A committed result requires one direct nested two-output child of that parent, one direct outer candidate changing only the kit gitlink from that parent to the nested child, one exact two-parent no-ff merge preserving that gitlink, no later gitlink rewrite, and a later exact Implementation Daemon board-only `IPS-003: mark todo completed` commit with both outputs already present. The final nested/outer/status commits come from supervisor completion evidence and are not self-embedded. Outer committed-history relevance admits only the exact operator bundle/inventory transactions and chronologically replayed board-only status commits whose sole byte changes are monotonic canonical `Status` values with dependencies and outputs already present; every other taskboard/control/source/candidate/merge/blob change rejects. This is committed-history and current-tree integrity, not proof of isolation from transient uncommitted edits. `baseline_evidence` is the exact reference-only operator projection for `kit-proof-certificate`, `kit-reuse-capabilities`, `kit-profile-d`, `kit-coordination`, `kit-modern-wal`, `kit-proof-reuse-bootstrap`, `kit-agent-receipts`, `kit-iroh-release`, and `kit-release-receipt`, with fixed path/digest/IDs plus `evidence_origin=operator_capture`, `assurance=process_observed_only`, and `nonclaim=pytest_execution_not_cryptographically_proven`; no copied command/result/count/log/transcript alias or execution-success claim is allowed elsewhere. The protected validator binds the reviewed registry and independently checks each suite preimage, exact argv/environment, receipt/log/count/non-pass evidence, and source relevance; it rejects combined counts under an aborting command, missing/swapped/duplicate suites, copied plan numbers, static-as-passed, patterned hashes, helpers, stale sources, and extra outputs. Include Profile-D policy, MCP++ artifact receipts, Iroh/KITA and joined release receipts, install_lotus opt-in proving-parameter downloads, every focused test, and explicit mock/simulated classifications. Planned-but-absent proof_seal_store is not counted as current structure. Every path is correctly labeled integrity/structural/mock/real, unsigned receipts remain unsigned, and direct-execution/recursion/key-download status is explicit.
- Embedding query: kit proof certificate store merkle wal compare and swap corruption receipt baseline

## IPS-004 Synthesize the cross-repository trust matrix and implementation boundary

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-synthesis
- Depends on: IPS-001, IPS-002, IPS-003
- Goal id: IPS-G010
- Outputs: docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_BASELINE.md, docs/architecture/incremental_proof_sealer_inventory/matrix.json
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-004
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/baseline
- Parallel lane: integration
- Resource class: cpu-small
- Resource stage: design-gate
- Estimated tokens: 12000
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_BASELINE.md, docs/architecture/incremental_proof_sealer_inventory/matrix.json
- Submodules: none
- Interfaces: ProofCapabilityMatrix, repository ownership decision
- Allow concurrent with:
- Conflict policy: Synthesize executable evidence only; do not claim recursion, direct execution, signatures, or trusted setup without current tests.
- Preconditions: All three inventory receipts are current for the exact source binding.
- Effects: Freezes canonical reuse/nonreuse decisions, real/mock/structural/direct classifications, the datasets/kit/accelerate authority split, proof-class claim boundaries, manifest-versus-recursive aggregation, backend capability/fallback decisions, pre-existing blockers, and explicit trust nonclaims.
- Evidence subset: ips/trust-matrix@1
- Symbolic first: true
- Acceptance: Every later implementation task can cite one exact semantic/storage/execution authority; unsupported recursion defaults to Merkle manifest aggregation and baseline failures are not hidden. The canonical `matrix.json` uses `incremental-proof-sealer-trust-baseline@2` with the closed keys `schema_version`, `synthesis_worktree_parent_revision`, `baseline_receipts`, `inventory_artifacts`, `repository_authorities`, `proof_class_decisions`, `aggregation_decision`, `backend_decisions`, and `trust_nonclaims`. It assigns proof-unit/manifest/identity to datasets, immutable cache/forest/WAL/CAS to kit, and prover/scheduler/planner/metrics to accelerate. It distinguishes integrity-only, trusted-signer receipt, receipt-completeness aggregation, declared direct computation, and parent-bound leaf-transition claims; records recursion unsupported and `merkle_manifest_aggregation`; rejects unknown/simulated production evidence; and states that repository correctness, pytest execution proof, semantic change correctness, and recursive proof verification are not established. `synthesis_worktree_parent_revision` equals the candidate task worktree HEAD, `baseline_receipts` equals the protected three-pin scheduler map, and each IPS-001/002/003 inventory artifact binds its exact inventory/report paths and inventory-output-only completion revision. The task-specific gate validates both proposed outputs and their current source relevance before supervisor completion; after merge, preflight recognizes the milestone only when both outputs were committed together directly after the declared parent. The Markdown repeats every exact matrix decision, task ID, receipt path/digest, and `pytest_execution_not_cryptographically_proven`. This immutable milestone lets later supervisor restarts treat the still-intrinsically-validated operator observations as historical rather than requiring implementation HEADs to equal the captured source state.
- Embedding query: cross repository proof trust matrix ownership executable evidence

## IPS-005 Define closed proof evidence classes, modes, kinds, and statuses

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-contracts
- Depends on: IPS-004
- Goal id: IPS-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/evidence.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_evidence.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_evidence.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets-contracts
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 14000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/evidence.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_evidence.py
- Submodules: ipfs_datasets_py
- Interfaces: IntegrityCommitment, SignedExecutionReceipt, ReceiptAggregationZkProof, DirectExecutionProof, IncrementalCommitSeal, ProofMode, ProofUnitKind, ProofTerminalStatus, SealStatus
- Allow concurrent with:
- Conflict policy: One datasets authority; no generic ZK boolean; closed unknown rejection; simulated is never production accepting.
- Preconditions: Trust baseline identifies existing adapters and wire contracts.
- Effects: Adds strict finite discriminated records and precise establishes/does-not-establish semantics.
- Evidence subset: ips/proof-evidence-classes@1
- Symbolic first: true
- Acceptance: Required classes/modes/kinds/statuses round-trip canonically; illegal combinations and generic overclaims fail; direct execution claims require DirectExecutionProof.
- Embedding query: integrity signed receipt aggregation direct execution incremental seal closed proof classes

## IPS-006 Implement the closed versioned ProofUnit schema

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-contracts
- Depends on: IPS-005
- Goal id: IPS-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/proof_unit.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_proof_unit.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_proof_unit.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets-schema
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 16000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/proof_unit.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_proof_unit.py
- Submodules: ipfs_datasets_py
- Interfaces: ProofUnit
- Allow concurrent with:
- Conflict policy: Include every normative field plus cache/graph context; typed absence only; no secrets or nondeterministic timestamp in identity.
- Preconditions: Closed evidence enums are stable.
- Effects: Defines immutable ProofUnit@1 validation, canonical serialization, and mode/status invariants.
- Evidence subset: ips/proof-unit@1
- Symbolic first: true
- Acceptance: All required fields exist; missing/unknown/duplicate/nonfinite/secret values fail; required simulated/non-pass units cannot satisfy production.
- Embedding query: ProofUnit schema source artifact dependency environment circuit key fixture policy status

## IPS-007 Implement deterministic repository, source, symbol, test, property, and artifact identities

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-identity
- Depends on: IPS-006
- Goal id: IPS-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/identity.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_identity.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_identity.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets-identity
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 17000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/identity.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_identity.py
- Submodules: ipfs_datasets_py
- Interfaces: RepositoryState, SourceArtifactIdentity, SourceSymbolIdentity, TestSelectorIdentity, canonical_cid
- Allow concurrent with:
- Conflict policy: Reuse the strict datasets canonical CID provider; reject pseudo-CIDs, path ambiguity, floats, cycles, duplicate map keys, and nondeterministic metadata.
- Preconditions: Trust baseline selects the existing canonicalization/CID primitive.
- Effects: Produces version-bound deterministic identities and known vectors for clean trees, dirty overlays, revisions, artifacts, symbols, tests, and properties.
- Evidence subset: ips/canonical-identities@1
- Symbolic first: true
- Acceptance: Byte-identical states yield identical IDs; every admitted content/path/schema/canonicalization mutation changes the required identity; imports have no side effects.
- Embedding query: deterministic repository source artifact symbol test property identity cid canonicalization

## IPS-008 Define and test the complete ProofCacheKey

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-cache-key
- Depends on: IPS-007
- Goal id: IPS-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/cache_key.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_cache_key.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_cache_key.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets-cache-key
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 15000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/cache_key.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_cache_key.py
- Submodules: ipfs_datasets_py
- Interfaces: ProofCacheKey, build_proof_cache_key
- Allow concurrent with:
- Conflict policy: Target-file equality is insufficient; every normative statement/source/dependency/environment/lock/fixture/tool/circuit/key/config/network/schema/canonicalization/selector/policy field is mandatory.
- Preconditions: ProofUnit and identity codecs are stable.
- Effects: Adds strict key construction and single-field mutation vectors.
- Evidence subset: ips/cache-key@1, ips/cache-key-vectors@1
- Symbolic first: true
- Acceptance: Changing any required key field changes the CID; missing/transitively incomplete roots, duplicates, and secret values fail closed.
- Embedding query: complete proof cache key dependency roots environment lock fixture tool circuit verification key selector policy

## IPS-009 Implement VerificationPolicy and VerificationRequirementManifest schemas

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-manifest
- Depends on: IPS-008
- Goal id: IPS-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/manifest.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_manifest.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_manifest.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets-manifest
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/manifest.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_manifest.py
- Submodules: ipfs_datasets_py
- Interfaces: VerificationPolicy, VerificationRequirementManifest, RequiredUnitDescriptor, UnitRemovalAuthorization
- Allow concurrent with:
- Conflict policy: Exact sorted required set; duplicate/reordered input rejected; deleted required units need current-policy authorization.
- Preconditions: Evidence/unit/identity contracts are stable.
- Effects: Defines required-set completeness, selector/policy/environment/version binding, removal records, and periodic checkpoint controls.
- Evidence subset: ips/verification-manifest@1
- Symbolic first: true
- Acceptance: Added selected units are required; unauthorized disappearance fails; manifest root changes for every required set/policy/selector/context mutation.
- Embedding query: verification requirement manifest policy selected tests authorized removal complete unit set

## IPS-010 Implement canonical proof statements and public/private input declarations

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-statements
- Depends on: IPS-009
- Goal id: IPS-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/statements.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_statements.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_statements.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets-statements
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 16000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/statements.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_statements.py
- Submodules: ipfs_datasets_py
- Interfaces: CanonicalProofStatement, DirectExecutionStatement, ReceiptAggregationStatement, ForestTransitionStatement
- Allow concurrent with:
- Conflict policy: Claims bind exact program/circuit/inputs/outputs and state what remains trusted; witness bytes never enter public artifacts.
- Preconditions: Evidence/unit/identity contracts are stable.
- Effects: Adds domain-separated canonical statements for each evidence class and explicit public/private input descriptors.
- Evidence subset: ips/canonical-statements@1
- Symbolic first: true
- Acceptance: Receipt aggregation cannot serialize a direct-execution claim; direct proof statement binds declared computation; private commitments reveal no witness.
- Embedding query: canonical proof statement direct execution receipt aggregation state transition public private inputs

## IPS-011 Implement the deterministic proof-forest commitment codec and vectors

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-merkle-codec
- Depends on: IPS-010
- Goal id: IPS-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/forest_codec.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_forest_codec.py, ipfs_datasets_py/tests/fixtures/incremental_proof_sealer/forest_vectors.json
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_forest_codec.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets-merkle
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/forest_codec.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_forest_codec.py, ipfs_datasets_py/tests/fixtures/incremental_proof_sealer/forest_vectors.json
- Submodules: ipfs_datasets_py
- Interfaces: ProofForestLeaf, CategoryRoot, RepositoryProofRoot, compute_category_root, compute_repository_root
- Allow concurrent with:
- Conflict policy: Domain-separated exact encoding; canonical ID-byte order; explicit empty/unary/binary nodes; reject duplicates and reordered caller input.
- Preconditions: Manifest and statement identities are stable.
- Effects: Freezes portable known vectors for every category and repository-root field.
- Evidence subset: ips/forest-codec@1, ips/forest-codec-vectors@1
- Symbolic first: true
- Acceptance: Repeated runs match; one-bit changes propagate; duplicate/reordered/unknown category inputs fail; parent/revision/environment/schema all affect the final root.
- Embedding query: deterministic merkle proof forest leaf category repository root vectors duplicates ordering

## IPS-012 Freeze datasets public exports and migrate existing receipt adapters

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: datasets-integration
- Depends on: IPS-011
- Goal id: IPS-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/migration.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_migration.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-contracts
- Parallel lane: datasets-integration
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 15000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/migration.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_migration.py
- Submodules: ipfs_datasets_py
- Interfaces: public datasets incremental sealing contracts, classify_legacy_receipt
- Allow concurrent with:
- Conflict policy: Legacy receipts retain their actual integrity/signed/simulated/direct meaning; no schema adapter upgrades assurance.
- Preconditions: Canonical datasets contracts and vectors pass.
- Effects: Adds lazy public exports and explicit accept/adapt/reject migration results for existing ZK/test/proof receipts.
- Evidence subset: ips/datasets-public-api@1, ips/legacy-receipt-migration@1
- Symbolic first: true
- Acceptance: Cold import performs no optional import, install, key, process, network, or user-state action; every legacy path is truthfully classified.
- Embedding query: datasets public proof unit manifest migration existing receipt assurance classification

## IPS-013 Implement the reason-labeled ProofDependencyGraph

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-dependency-graph
- Depends on: IPS-012
- Goal id: IPS-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/dependency_graph.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_dependency_graph.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_dependency_graph.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-invalidation
- Parallel lane: datasets
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/dependency_graph.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_dependency_graph.py
- Submodules: ipfs_datasets_py
- Interfaces: ProofDependencyGraph, ProofDependencyEdge, DependencyEdgeType, compute_dependency_root
- Allow concurrent with:
- Conflict policy: Store prerequisite -> dependent direction and content-addressed reasons; reject unknown edges, duplicate contradictions, cycles where illegal, and truncated roots.
- Preconditions: Canonical datasets public contracts pass.
- Effects: Adds typed artifact/symbol/unit nodes, deterministic adjacency, transitive prerequisite roots, forward invalidation traversal, and explanation paths.
- Evidence subset: ips/dependency-graph@1
- Symbolic first: true
- Acceptance: All eleven required edge types are supported; insertion order cannot affect roots; changed prerequisite reaches every dependent aggregate; unrelated nodes remain outside closure.
- Embedding query: proof dependency graph source imports calls test covers fixture config aggregate invalidation

## IPS-014 Implement deterministic requirement discovery and test/property selection

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-requirements
- Depends on: IPS-013
- Goal id: IPS-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/discovery.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_discovery.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_discovery.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-invalidation
- Parallel lane: datasets
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 19000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/discovery.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_discovery.py
- Submodules: ipfs_datasets_py
- Interfaces: build_verification_requirement_manifest, build_proof_dependency_graph, ProofUnitSelector
- Allow concurrent with:
- Conflict policy: Deterministic bounded source/test/property discovery; incomplete import/coverage frontier is explicit and broadens selection.
- Preconditions: Dependency graph and canonical manifest contracts pass.
- Effects: Selects proof units at module/symbol, exact pytest node/parameter, property/obligation, direct computation, and release-invariant granularity.
- Evidence subset: ips/requirement-discovery@1
- Symbolic first: true
- Acceptance: Stable logical IDs survive context changes; renamed/deleted nodes become remove/add; selector policy determines required units; unknown frontiers cannot narrow requirements.
- Embedding query: proof unit discovery pytest node parameter property formal obligation source symbol selector

## IPS-015 Implement complete repository diff and change classification

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-diff
- Depends on: IPS-014
- Goal id: IPS-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/repository_diff.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_repository_diff.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_repository_diff.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-invalidation
- Parallel lane: datasets
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 17000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/repository_diff.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_repository_diff.py
- Submodules: ipfs_datasets_py
- Interfaces: RepositoryDiff, ChangedArtifact, ChangeClass, diff_repository_states
- Allow concurrent with:
- Conflict policy: Bind the exact diff algorithm/version and all Git parents; unknown/ambiguous changes force broad invalidation or full fallback.
- Preconditions: Repository identities, selection, and graph semantics pass.
- Effects: Classifies source implementation/interface, test, add/delete, fixture, lock, configuration, circuit, key, selector, policy, network, canonicalization, environment, and documentation changes.
- Evidence subset: ips/repository-diff@1
- Symbolic first: true
- Acceptance: Changed-artifact commitment is complete/deterministic; merges and dirty overlays are explicit; ordinary docs are distinct from checked specifications/generated inputs.
- Embedding query: repository diff complete changed artifact source test fixture lock circuit key docs merge

## IPS-016 Implement invalidation closure, full-fallback rules, and explanations

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-invalidation
- Depends on: IPS-015
- Goal id: IPS-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/invalidation.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_invalidation.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_invalidation.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-invalidation
- Parallel lane: datasets
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 21000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/invalidation.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_invalidation.py
- Submodules: ipfs_datasets_py
- Interfaces: compute_invalidation_closure, classify_full_fallback, explain_invalidation
- Allow concurrent with:
- Conflict policy: Traverse prerequisite -> dependent; unchanged file alone never authorizes reuse; unknown closure broadens or falls back full.
- Preconditions: Exact key, manifest, dependency graph, discovery, and complete diff pass.
- Effects: Implements every normative invalidation and checkpoint-trigger rule plus deterministic reason/path records.
- Evidence subset: ips/invalidation-engine@1
- Symbolic first: true
- Acceptance: Relevant source/interface/test/fixture/lock/config/circuit/key/policy/environment changes invalidate correctly; docs/unrelated edits preserve valid units; add/delete rules are explicit.
- Embedding query: invalidation closure full checkpoint fallback explain reuse dependency trust context

## IPS-017 Freeze datasets invalidation API and run its conformance matrix

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-conformance
- Depends on: IPS-016
- Goal id: IPS-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/__init__.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_conformance.py
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/datasets-invalidation
- Parallel lane: datasets
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 16000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/__init__.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_conformance.py
- Submodules: ipfs_datasets_py
- Interfaces: build_repository_state, build_verification_requirement_manifest, build_proof_dependency_graph, compute_proof_cache_key, diff_repository_states, compute_invalidation_closure, explain_invalidation
- Allow concurrent with:
- Conflict policy: Correct only demonstrated datasets contract gaps; no persistence, scheduling, or assurance upgrade.
- Preconditions: All datasets contract/graph/diff/invalidation tasks pass.
- Effects: Runs known vectors and positive/negative rule matrix and freezes the narrow public semantic API.
- Evidence subset: ips/datasets-conformance@1
- Symbolic first: true
- Acceptance: Stable IDs/source-closure semantics permit unrelated reuse while rejecting a different relevant source root; every required rule is covered and deterministic.
- Embedding query: datasets incremental proof sealing public invalidation conformance

## IPS-018 Define the narrow kit ProofSealStore protocol and closed artifact kinds

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kit-store-contracts
- Depends on: IPS-004
- Goal id: IPS-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/contracts.py, ipfs_kit_py/ipfs_kit_py/proof_seal_store/__init__.py, ipfs_kit_py/tests/proof_seal_store/test_contracts.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_contracts.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-storage
- Parallel lane: kit
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 14000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/contracts.py, ipfs_kit_py/ipfs_kit_py/proof_seal_store/__init__.py, ipfs_kit_py/tests/proof_seal_store/test_contracts.py
- Submodules: ipfs_kit_py
- Interfaces: ProofSealStore, ArtifactKind, ArtifactReference, CacheCandidate, CurrentSealPointer, SealTransitionRecord
- Allow concurrent with:
- Conflict policy: Storage types carry bytes/CIDs and canonical opaque records only; kit never decides proof validity; public proving-key/witness artifacts rejected.
- Preconditions: Trust matrix selects strict CID and modern WAL primitives.
- Effects: Defines bounded local/IPFS retrieval, candidate index, forest, WAL, CAS, and recovery protocol boundaries without importing datasets on cold import.
- Evidence subset: ips/store-protocol@1
- Symbolic first: true
- Acceptance: Closed kinds exactly cover required artifacts; explicit roots are mandatory; candidate versus admitted/current distinctions cannot be collapsed.
- Embedding query: kit proof seal store protocol immutable artifacts candidate current pointer transition

## IPS-019 Implement the hermetic immutable local proof-object store

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kit-local-store
- Depends on: IPS-018
- Goal id: IPS-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/local_store.py, ipfs_kit_py/tests/proof_seal_store/test_local_store.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_local_store.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-storage
- Parallel lane: kit
- Resource class: io-medium
- Resource stage: implementation
- Estimated tokens: 19000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/local_store.py, ipfs_kit_py/tests/proof_seal_store/test_local_store.py
- Submodules: ipfs_kit_py
- Interfaces: HermeticProofSealStore, put_immutable, get_verified_bytes
- Allow concurrent with:
- Conflict policy: Explicit root only; strict CID/readback rehash, fsync file and parent, symlink/path/size fencing; no default user state or daemon.
- Preconditions: Store contracts pass.
- Effects: Persists immutable closed-kind blobs atomically and detects corruption/substitution on every read.
- Evidence subset: ips/local-proof-store@1
- Symbolic first: true
- Acceptance: Identical bytes deduplicate; mismatched CID/kind/bytes, path escape, symlink, short write, fsync/readback failure, and corrupted object fail closed.
- Embedding query: hermetic local immutable proof store cid rehash fsync corruption

## IPS-020 Implement the optional injected IPFS transport adapter

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: kit-ipfs-adapter
- Depends on: IPS-019
- Goal id: IPS-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/ipfs_transport.py, ipfs_kit_py/tests/proof_seal_store/test_ipfs_transport.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_ipfs_transport.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-storage
- Parallel lane: kit
- Resource class: io-medium
- Resource stage: implementation
- Estimated tokens: 13000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/ipfs_transport.py, ipfs_kit_py/tests/proof_seal_store/test_ipfs_transport.py
- Submodules: ipfs_kit_py
- Interfaces: IpfsProofArtifactTransport, replicate_public_artifact, fetch_public_artifact
- Allow concurrent with:
- Conflict policy: Injected client only; public artifact allowlist; bounded bytes/time; rehash response; proving keys/witnesses forbidden; network absence typed unavailable.
- Preconditions: Hermetic local store passes.
- Effects: Adds optional replication/retrieval without making IPFS a unit-test or import requirement.
- Evidence subset: ips/ipfs-proof-transport@1
- Symbolic first: true
- Acceptance: Mocked corrupt/oversized/wrong-kind responses fail; backend ambiguity is recorded; local committed bytes remain reconcilable.
- Embedding query: optional ipfs proof artifact transport injected rehash public only

## IPS-021 Implement the exact-key candidate cache index and admission records

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kit-cache-index
- Depends on: IPS-020, IPS-008
- Goal id: IPS-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/cache_index.py, ipfs_kit_py/tests/proof_seal_store/test_cache_index.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_cache_index.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-storage
- Parallel lane: kit
- Resource class: io-medium
- Resource stage: implementation
- Estimated tokens: 19000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/cache_index.py, ipfs_kit_py/tests/proof_seal_store/test_cache_index.py
- Submodules: ipfs_kit_py
- Interfaces: ProofCacheIndex, CandidateAdmissionRecord, lookup_candidate, record_verified_admission, tombstone
- Allow concurrent with:
- Conflict policy: Index is a hint; only accelerate-issued verified admission records may be indexed; every lookup returns a candidate requiring fresh verification.
- Preconditions: Complete cache key and local/optional transport contracts pass.
- Effects: Adds exact-key indexing, atomic updates, quarantine, tombstones, corruption rebuild, and poisoning detection.
- Evidence subset: ips/proof-cache-index@1
- Symbolic first: true
- Acceptance: Key/CID/kind/admission mismatch misses or quarantines; unverified proof cannot enter; stale/simulated/non-pass metadata cannot be queried as accepted.
- Embedding query: exact proof cache index candidate admission quarantine poisoning tombstone

## IPS-022 Persist deterministic proof forests and update affected branches only

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kit-proof-forest
- Depends on: IPS-021, IPS-011
- Goal id: IPS-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/forest.py, ipfs_kit_py/tests/proof_seal_store/test_forest.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_forest.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-storage
- Parallel lane: kit
- Resource class: io-medium
- Resource stage: implementation
- Estimated tokens: 21000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/forest.py, ipfs_kit_py/tests/proof_seal_store/test_forest.py
- Submodules: ipfs_kit_py
- Interfaces: ProofForestStore, persist_forest, update_forest_branches, verify_unaffected_leaves
- Allow concurrent with:
- Conflict policy: Consume datasets codec/vectors; do not invent ordering/hash semantics; reject duplicate/reordered/lost leaves and old-root reuse.
- Preconditions: Canonical forest vectors and cache-index storage pass.
- Effects: Stores immutable nodes/category roots and incrementally recomputes changed paths with equality witnesses for unaffected leaves.
- Evidence subset: ips/proof-forest-store@1
- Symbolic first: true
- Acceptance: Root matches datasets vectors; identical replay is deterministic; one/two independent changes touch only expected branches; unaffected leaf loss and changed manifest/old aggregate fail.
- Embedding query: proof forest persistence affected merkle branches unaffected leaves deterministic

## IPS-023 Implement repository/branch current-seal compare-and-swap

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kit-seal-cas
- Depends on: IPS-022
- Goal id: IPS-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/pointer.py, ipfs_kit_py/tests/proof_seal_store/test_pointer_cas.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_pointer_cas.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-durability
- Parallel lane: kit
- Resource class: io-medium
- Resource stage: implementation
- Estimated tokens: 16000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/pointer.py, ipfs_kit_py/tests/proof_seal_store/test_pointer_cas.py
- Submodules: ipfs_kit_py
- Interfaces: CurrentSealRepository, compare_and_swap_current_seal
- Allow concurrent with:
- Conflict policy: Namespace repository and branch; bind generation/expected parent seal/root; stale writer never overwrites current.
- Preconditions: Immutable forest storage passes.
- Effects: Adds durable pointer reads and expected-parent CAS with concurrent-process fencing.
- Evidence subset: ips/current-seal-cas@1
- Symbolic first: true
- Acceptance: Exactly one concurrent writer wins; wrong branch/parent/generation rejects; pointer bytes are rehashed and directory durability is enforced.
- Embedding query: current seal pointer compare and swap stale concurrent writer branch parent

## IPS-024 Implement the WAL-backed seal transition state machine

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kit-seal-wal
- Depends on: IPS-023
- Goal id: IPS-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/wal.py, ipfs_kit_py/tests/proof_seal_store/test_wal.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_wal.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-durability
- Parallel lane: kit
- Resource class: io-large
- Resource stage: implementation
- Estimated tokens: 21000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/wal.py, ipfs_kit_py/tests/proof_seal_store/test_wal.py
- Submodules: ipfs_kit_py
- Interfaces: SealTransitionWal, begin_transition, record_phase, commit_transition, abort_transition
- Allow concurrent with:
- Conflict policy: Build on modern core/wal committed-only semantics; durable intent precedes effects; immutable CIDs bind every phase.
- Preconditions: Local store, forest, and pointer CAS pass.
- Effects: Journals proof start/result, receipt, forest, aggregate, seal, CAS, and cleanup phases with injection hooks.
- Evidence subset: ips/seal-transition-wal@1
- Symbolic first: true
- Acceptance: Partial/uncommitted records cannot become current; committed replay is deterministic; corrupt tail preserves valid prefix.
- Embedding query: seal transition wal intent receipt forest aggregate seal cas crash phases

## IPS-025 Implement deterministic transition recovery and ambiguous-outcome policy

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kit-recovery
- Depends on: IPS-024
- Goal id: IPS-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/recovery.py, ipfs_kit_py/tests/proof_seal_store/test_recovery.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_recovery.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-durability
- Parallel lane: kit
- Resource class: io-large
- Resource stage: implementation
- Estimated tokens: 21000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/recovery.py, ipfs_kit_py/tests/proof_seal_store/test_recovery.py
- Submodules: ipfs_kit_py
- Interfaces: recover_seal_transitions, RecoveryDisposition
- Allow concurrent with:
- Conflict policy: Never infer external prover success; verify durable artifacts or require reproof/repair; recovery is idempotent.
- Preconditions: WAL phase model passes.
- Effects: Produces resume/replay/verify-existing/discard-uncommitted/repair/full-reproof dispositions for every phase.
- Evidence subset: ips/transition-recovery@1
- Symbolic first: true
- Acceptance: Repeated restart converges; post-CAS cleanup recognizes committed pointer; stale parent after pre-CAS persistence rejects publication.
- Embedding query: crash recovery ambiguous prover outcome resume replay verify discard repair full reproof

## IPS-026 Prove kit corruption, concurrency, replay, and seven-phase crash conformance

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kit-adversarial
- Depends on: IPS-025
- Goal id: IPS-G050
- Outputs: ipfs_kit_py/tests/proof_seal_store/test_crash_matrix.py, ipfs_kit_py/tests/proof_seal_store/test_concurrency.py, ipfs_kit_py/tests/proof_seal_store/test_corruption.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_crash_matrix.py ipfs_kit_py/tests/proof_seal_store/test_concurrency.py ipfs_kit_py/tests/proof_seal_store/test_corruption.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-durability
- Parallel lane: kit
- Resource class: io-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_kit_py/tests/proof_seal_store/test_crash_matrix.py, ipfs_kit_py/tests/proof_seal_store/test_concurrency.py, ipfs_kit_py/tests/proof_seal_store/test_corruption.py
- Submodules: ipfs_kit_py
- Interfaces: kit durability conformance
- Allow concurrent with:
- Conflict policy: Fault injection only; do not weaken durability or accept ambiguous artifacts to make tests pass.
- Preconditions: Store/index/forest/CAS/WAL/recovery focused tests pass.
- Effects: Injects every required transition failure plus corrupt blob/index/WAL/pointer, stale writers, replay, and optional transport ambiguity.
- Evidence subset: ips/kit-crash-matrix@1, ips/kit-concurrency@1
- Symbolic first: true
- Acceptance: All seven recovery decisions match policy; zero lost committed pointer, zero stale writer win, zero corrupted candidate acceptance, deterministic repeated recovery.
- Embedding query: proof seal store crash matrix corruption concurrent writers deterministic replay

## IPS-027 Freeze the kit adapter, migrate proof transport, and document storage nonclaims

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: kit-integration
- Depends on: IPS-026, IPS-012
- Goal id: IPS-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/proof_seal_store/__init__.py, ipfs_kit_py/docs/architecture/INCREMENTAL_PROOF_SEAL_STORE.md, ipfs_kit_py/tests/proof_seal_store/test_migration.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store ipfs_kit_py/tests/test_proof_certificate_store.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/kit-durability
- Parallel lane: kit
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 17000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_kit_py/ipfs_kit_py/proof_seal_store/__init__.py, ipfs_kit_py/docs/architecture/INCREMENTAL_PROOF_SEAL_STORE.md, ipfs_kit_py/tests/proof_seal_store/test_migration.py
- Submodules: ipfs_kit_py
- Interfaces: public ProofSealStore adapter
- Allow concurrent with:
- Conflict policy: Existing proof_certificate_store remains integrity transport or delegates narrowly; kit docs cannot claim proof/execution/reuse authority.
- Preconditions: Kit conformance and datasets public schemas pass.
- Effects: Freezes lazy adapter exports, legacy blob migration, canonical record interop, and precise durability/trust documentation.
- Evidence subset: ips/kit-public-adapter@1, ips/kit-migration@1
- Symbolic first: true
- Acceptance: Cold import is hermetic; old exact-byte blobs can be staged but require accelerate verification before admission; proving keys/witnesses never surface.
- Embedding query: kit proof seal store public adapter legacy certificate transport migration trust

## IPS-028 Implement evidence-class verification and cache-admission decisions

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-admission
- Depends on: IPS-004
- Goal id: IPS-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/admission.py, test/api/incremental_sealing/test_admission.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_admission.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-trust
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/admission.py, test/api/incremental_sealing/test_admission.py
- Submodules: none
- Interfaces: EvidenceVerifier, AdmissionDecision, verify_for_admission
- Allow concurrent with:
- Conflict policy: Accelerate alone decides proof admission; hashes, receipts, structural checks, and simulations retain their declared assurance class and never become direct-execution evidence.
- Preconditions: The executable trust matrix identifies each existing backend and receipt path.
- Effects: Verifies evidence according to its closed class, returns typed establishes/does-not-establish claims, and requires successful verification before a cache-admission record can be issued.
- Evidence subset: ips/evidence-admission@1
- Symbolic first: true
- Acceptance: Unknown proof systems, malformed evidence, nonterminal/failed/simulated required units, unsigned required receipts, public-input mismatch, and verifier failure reject; receipt aggregation never claims the tests executed.
- Embedding query: evidence verification cache admission integrity signed receipt aggregation direct execution simulation rejection

## IPS-029 Probe backend capabilities and admit recursion only when demonstrated

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-backends
- Depends on: IPS-028
- Goal id: IPS-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/backends.py, test/api/incremental_sealing/test_backends.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_backends.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-trust
- Parallel lane: accelerate
- Resource class: cpu-proof
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 4800
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/backends.py, test/api/incremental_sealing/test_backends.py
- Submodules: none
- Interfaces: ProofBackendCapability, BackendCapabilityRegistry, probe_backend_capability
- Allow concurrent with:
- Conflict policy: Capability is executable evidence, not a documentation flag; absent or inconclusive recursive self-verification selects Merkleized manifest aggregation.
- Preconditions: Evidence admission rules and the trust baseline pass.
- Effects: Records operational prove/verify, signature, direct-computation, aggregation, recursive-verification, resource, timeout, and cancellation capabilities without installation or setup side effects.
- Evidence subset: ips/backend-capability-matrix@1, ips/recursion-probe@1
- Symbolic first: true
- Acceptance: Unknown/unavailable backends fail typed; recursion is enabled only by a reliable bounded prove-and-verify probe using preconfigured test-only material, and is otherwise explicitly false.
- Embedding query: proof backend capability probe recursive self verification manifest aggregation fallback

## IPS-030 Implement allowlisted verification-key, proving-key, and signer trust policy

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-trust-policy
- Depends on: IPS-029, IPS-012
- Goal id: IPS-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/trust.py, test/api/incremental_sealing/test_trust.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_trust.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-trust
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/trust.py, test/api/incremental_sealing/test_trust.py
- Submodules: none
- Interfaces: TrustedProofPolicy, VerificationKeyRegistry, SignerTrustRegistry, ProvingKeyHandle
- Allow concurrent with:
- Conflict policy: Policy configuration, not an untrusted caller or model, selects content-addressed keys/signers/circuits; proving-key bytes are private handles and never public API data.
- Preconditions: Datasets evidence/migration contracts and backend capabilities pass.
- Effects: Binds allowlisted verification-key CIDs, setup origin, production/test-only designation, circuit compatibility, signer scope, revocation epoch, and nonexportable proving-key references.
- Evidence subset: ips/key-registry@1, ips/signer-trust@1
- Symbolic first: true
- Acceptance: Old/substituted/unallowlisted/test-only-in-production keys and untrusted/revoked/out-of-scope signers reject; production mode never generates or downloads key material.
- Embedding query: verification key allowlist proving key private signer trust policy setup origin test only

## IPS-031 Implement bounded hermetic prover and verifier adapters

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-provers
- Depends on: IPS-030
- Goal id: IPS-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/provers.py, test/api/incremental_sealing/test_provers.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_provers.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-trust
- Parallel lane: accelerate
- Resource class: cpu-proof
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/provers.py, test/api/incremental_sealing/test_provers.py
- Submodules: none
- Interfaces: IncrementalProofBackendAdapter, ProverInvocation, VerificationInvocation, ProverOutcome
- Allow concurrent with:
- Conflict policy: Adapters invoke only statically registered programs/circuits and approved key handles; no arbitrary executable/path, implicit network, setup generation, or mock proof success.
- Preconditions: Backend capabilities and trust registries pass.
- Effects: Wraps existing real prover/verifier paths with committed inputs, bounded output, structured timeout/cancellation/unavailable outcomes, proof-byte verification, and witness-safe logging.
- Evidence subset: ips/prover-adapters@1
- Symbolic first: true
- Acceptance: Modified public input or invalid cryptography fails; sensitive witness/proving-key data is absent from receipts/logs; ambiguous external completion is never reported as proved.
- Embedding query: bounded hermetic prover verifier adapter committed input witness secrecy no arbitrary executable

## IPS-032 Implement full-versus-incremental planning and reuse explanations

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-planner
- Depends on: IPS-017, IPS-027, IPS-031
- Goal id: IPS-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/planner.py, test/api/incremental_sealing/test_planner.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_planner.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-orchestration
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 24000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/planner.py, test/api/incremental_sealing/test_planner.py
- Submodules: none
- Interfaces: IncrementalProofPlan, AggregationPlan, ResourceEstimate, FinalAcceptancePolicy, create_incremental_plan, plan_incremental_proof, explain_reuse
- Allow concurrent with:
- Conflict policy: Planner consumes datasets invalidation semantics and kit candidates without copying either authority; candidate presence alone never authorizes reuse.
- Preconditions: Datasets invalidation API, kit public adapter/durability, and bounded proof adapters pass.
- Effects: Verifies the declared parent context, computes reusable/invalidated/added/removed units, changed roots, fallback reasons, expected incremental/full resources, savings, bounded aggregation, and acceptance gates.
- Evidence subset: ips/incremental-plan@1, ips/reuse-explanation@1
- Symbolic first: true
- Acceptance: Every required plan field is deterministic; complete cache keys and transitive roots govern reuse; first state and required trust/schema changes choose full fallback; unrelated edits preserve only demonstrably valid units.
- Embedding query: incremental proof plan parent reusable invalidated added removed fallback resources savings aggregation acceptance

## IPS-033 Integrate proof work with modern scheduling and resource admission

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-scheduling
- Depends on: IPS-032
- Goal id: IPS-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/scheduling.py, test/api/incremental_sealing/test_scheduling.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_scheduling.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-orchestration
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 20000
- Implementation timeout seconds: 4800
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/scheduling.py, test/api/incremental_sealing/test_scheduling.py
- Submodules: none
- Interfaces: ProofWorkScheduler, ProofWorkItem, ProofResourcePolicy, build_proof_schedule
- Allow concurrent with:
- Conflict policy: Adapt the existing modern scheduler/admission primitives; do not create a second scheduler or use legacy mock hardware/simulated success.
- Preconditions: A deterministic incremental plan and backend resource declarations exist.
- Effects: Admits CPU/memory and actual GPU requirements, runs independent units in bounded parallelism, bounds aggregate fan-in, and applies invalidation/cache/small-unit/critical-path/expensive/full-fallback priorities.
- Evidence subset: ips/proof-schedule@1, ips/resource-admission@1
- Symbolic first: true
- Acceptance: Oversubscribed work waits or returns typed unavailable; priority and fan-in are deterministic; independent units parallelize while dependencies and publication ordering remain intact.
- Embedding query: modern scheduler proof work resource admission cpu memory gpu parallel priority bounded fan in

## IPS-034 Implement cancellation, timeout, and process-tree termination fencing

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-process-control
- Depends on: IPS-033
- Goal id: IPS-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/process_control.py, test/api/incremental_sealing/test_process_control.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_process_control.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-orchestration
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/process_control.py, test/api/incremental_sealing/test_process_control.py
- Submodules: none
- Interfaces: ProofProcessController, CancellationToken, ProcessTerminationResult
- Allow concurrent with:
- Conflict policy: Use existing safe process-control hooks; timeout/cancellation fences descendants and resources, and never fabricates a terminal proof result.
- Preconditions: Scheduled work has explicit timeout/resource identities.
- Effects: Implements cooperative cancellation, bounded terminate/kill escalation, process-group cleanup, late-result quarantine, and typed timeout/cancelled/unknown outcomes.
- Evidence subset: ips/process-fencing@1
- Symbolic first: true
- Acceptance: Interrupted proving leaves no live descendant or admitted proof; racey late output cannot cross cancellation generation; unknown or timeout never satisfies a required unit.
- Embedding query: proof cancellation timeout process tree termination late result quarantine

## IPS-035 Execute plans with fresh cache verification and verified admission

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-execution
- Depends on: IPS-034, IPS-021, IPS-027
- Goal id: IPS-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/executor.py, test/api/incremental_sealing/test_executor.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_executor.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-orchestration
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 25000
- Implementation timeout seconds: 6000
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/executor.py, test/api/incremental_sealing/test_executor.py
- Submodules: none
- Interfaces: IncrementalPlanExecutor, IncrementalProofResult, execute_incremental_plan
- Allow concurrent with:
- Conflict policy: Every candidate is fetched/rehash-checked and cryptographically or signature verified under the current policy before admission; no cache fast path bypasses verification.
- Preconditions: Scheduler/process fencing, kit candidate index/public adapter, and admission/prover paths pass.
- Effects: Verifies reusable units, proves invalidated/added units, verifies new evidence, records admissions/tombstones/invalidation, and fails closed on every required non-pass state.
- Evidence subset: ips/incremental-execution@1, ips/cache-reverification@1
- Symbolic first: true
- Acceptance: Reused and newly proved sets exactly cover plan requirements; stale/poisoned/corrupt/mismatched/simulated evidence is rejected; cancellation and unavailable outcomes cannot proceed to aggregation.
- Embedding query: execute incremental proof plan verify cached candidate prove invalidated admit reject stale simulated

## IPS-036 Implement bounded manifest aggregation and capability-gated recursion

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: accelerate-aggregation
- Depends on: IPS-035, IPS-011, IPS-029
- Goal id: IPS-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/aggregation.py, test/api/incremental_sealing/test_aggregation.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_aggregation.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-orchestration
- Parallel lane: accelerate
- Resource class: cpu-proof
- Resource stage: implementation
- Estimated tokens: 24000
- Implementation timeout seconds: 6000
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/aggregation.py, test/api/incremental_sealing/test_aggregation.py
- Submodules: none
- Interfaces: ProofAggregator, ManifestAggregationResult, RecursiveAggregationResult, aggregate_verified_units
- Allow concurrent with:
- Conflict policy: Recursion is selectable only from a successful capability probe; otherwise produce a Merkle integrity/completeness aggregation explicitly labeled as not recursively verifying child proofs or test execution.
- Preconditions: Verified plan execution, canonical forest codec, and backend capability matrix pass.
- Effects: Builds bounded fan-in leaf/batch/category/repository aggregates binding exact identities/count/order/no-duplicates/root/status/repository/environment and recomputes affected branches only.
- Evidence subset: ips/manifest-aggregation@1, ips/recursive-aggregation@1
- Symbolic first: true
- Acceptance: Missing/duplicate/reordered/failed child and changed manifest/old aggregate reject; recursive claims appear only when the backend actually verifies children; receipt aggregation states signer trust and does not claim underlying execution.
- Embedding query: bounded proof aggregation merkle manifest completeness capability gated recursion child identities no duplicates

## IPS-037 Measure proving, aggregation, verification, storage, and savings cost

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: accelerate-metrics
- Depends on: IPS-036
- Goal id: IPS-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/metrics.py, test/api/incremental_sealing/test_metrics.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_metrics.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/accelerate-orchestration
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 17000
- Implementation timeout seconds: 4200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/metrics.py, test/api/incremental_sealing/test_metrics.py
- Submodules: none
- Interfaces: ProofCostRecord, ProofCostComparison, ProofMetricsCollector, compare_costs
- Allow concurrent with:
- Conflict policy: Preserve measured versus estimated provenance and wall/CPU/GPU distinctions; absent counters remain unknown, never zero or fabricated savings.
- Preconditions: Executor and aggregation emit structured lifecycle timings/resources/sizes.
- Effects: Records required/reused/invalidated/proved counts, cache hit, leaf/aggregate/verify/wall time, CPU/GPU, peak memory, proof/seal size, storage growth, full/incremental cost, savings, and fallback reason.
- Evidence subset: ips/proof-cost@1
- Symbolic first: true
- Acceptance: Arithmetic and units are deterministic; compute saved compares equivalent required work; failed/fallback runs remain visible; estimates are never reported as measurements.
- Embedding query: proving compute saved full incremental cost cpu gpu memory proof size storage verification latency

## IPS-038 Implement full checkpoint seal construction

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: seal-full-checkpoint
- Depends on: IPS-037
- Goal id: IPS-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/full_checkpoint.py, test/api/incremental_sealing/test_full_checkpoint.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_full_checkpoint.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/seals
- Parallel lane: accelerate
- Resource class: cpu-proof
- Resource stage: implementation
- Estimated tokens: 23000
- Implementation timeout seconds: 6000
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/full_checkpoint.py, test/api/incremental_sealing/test_full_checkpoint.py
- Submodules: none
- Interfaces: FullCheckpointSeal, FullCheckpointBuilder, create_full_checkpoint
- Allow concurrent with:
- Conflict policy: A full checkpoint proves or freshly verifies every current required unit; cache reuse cannot be hidden inside a sealed_full result.
- Preconditions: Planning, verified execution, aggregation, forest persistence, and metrics pass.
- Effects: Builds a full manifest/forest and repository seal bound to repository root/revision, environment, policy, schemas/canonicalization, circuits/keys, and an explicitly empty or historical-parent checkpoint relation.
- Evidence subset: ips/full-seal@1
- Symbolic first: true
- Acceptance: First states and mandated fallback contexts seal only after all required units and roots verify; any required simulated/unknown/unavailable/failed unit prevents sealed_full.
- Embedding query: full checkpoint seal every required proof unit repository forest root production status

## IPS-039 Implement parent-bound delta seals with all transition invariants

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: seal-delta
- Depends on: IPS-038, IPS-016, IPS-025
- Goal id: IPS-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/delta_seal.py, test/api/incremental_sealing/test_delta_seal.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_delta_seal.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/seals
- Parallel lane: accelerate
- Resource class: cpu-proof
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 6600
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/delta_seal.py, test/api/incremental_sealing/test_delta_seal.py
- Submodules: none
- Interfaces: DeltaSeal, DeltaTransitionStatement, DeltaSealBuilder, build_delta_seal
- Allow concurrent with:
- Conflict policy: Delta construction cannot weaken a full-fallback result and binds one exact accepted parent/branch/revision so the transition cannot be replayed elsewhere.
- Preconditions: Full checkpoint type, datasets invalidation/fallback, and kit recovery contracts pass.
- Effects: Enforces accepted old seal/root, exact new source root, complete diff, new proofs for every invalidation/addition, exact keys for reuse, authorized removals, complete manifest, exact forest, blocking-status exclusion, exact parent, and anti-replay binding.
- Evidence subset: ips/delta-seal@1, ips/delta-fourteen-invariants@1
- Symbolic first: true
- Acceptance: All fourteen normative invariants are independently tested; wrong parent/branch, incomplete diff/manifest, stale reuse, unauthorized deletion, missing replacement, old aggregate, lost leaf, and simulated/non-pass evidence reject.
- Embedding query: delta seal fourteen invariants accepted parent complete diff invalidation reuse manifest forest anti replay

## IPS-040 Implement atomic WAL-backed seal publication and current-root CAS

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: seal-atomic-workflow
- Depends on: IPS-039, IPS-024, IPS-025
- Goal id: IPS-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/sealer.py, test/api/incremental_sealing/test_atomic_sealer.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_atomic_sealer.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/seals
- Parallel lane: accelerate
- Resource class: io-large
- Resource stage: integration
- Estimated tokens: 28000
- Implementation timeout seconds: 6600
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/sealer.py, test/api/incremental_sealing/test_atomic_sealer.py
- Submodules: none
- Interfaces: IncrementalProofSealer, SealPublicationResult, publish_full_checkpoint, publish_delta_seal
- Allow concurrent with:
- Conflict policy: Publication is the sole accelerate coordinator over kit WAL/CAS; no seal becomes current until changed/reused evidence, affected aggregates, transition evidence, and expected-parent CAS all pass.
- Preconditions: Full/delta builders and kit WAL/recovery/CAS pass.
- Effects: Executes load/verify parent, diff/manifest/invalidation, candidate verification, proving, forest update, affected aggregation, transition verification, seal persistence, CAS, and cleanup as one recoverable workflow.
- Evidence subset: ips/atomic-transition@1
- Symbolic first: true
- Acceptance: Failures at any pre-CAS phase leave the old pointer current; exactly one valid writer publishes; post-CAS recovery recognizes success; stale parent returns stale_parent without overwrite.
- Embedding query: atomic incremental proof sealer wal compare swap current seal publication workflow

## IPS-041 Implement seal verification, explanations, and cost comparison APIs

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: seal-verification
- Depends on: IPS-040
- Goal id: IPS-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/verification.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/explanations.py, test/api/incremental_sealing/test_verification.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_verification.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/seals
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 23000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/verification.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/explanations.py, test/api/incremental_sealing/test_verification.py
- Submodules: none
- Interfaces: SealVerificationResult, ProofReuseExplanation, ProofInvalidationExplanation, verify_seal, explain_reuse, explain_invalidation, compare_full_and_incremental
- Allow concurrent with:
- Conflict policy: Trusted keys/policy are supplied by configured authority; explanations expose reason paths without secrets and never substitute for verification.
- Preconditions: Atomic full/delta lifecycle and metrics pass.
- Effects: Revalidates seal type/status/chain/key/signature/proofs/manifest/forest/root/policy, returns typed failure, traces exact reuse/invalidation evidence, and compares equivalent full/incremental work.
- Evidence subset: ips/seal-verification@1, ips/reuse-invalidation-explanation@1, ips/full-incremental-comparison@1
- Symbolic first: true
- Acceptance: Unknown systems/statuses, wrong keys/policy/parent/root, modified inputs, incomplete history, and cryptographic failure reject; explanations identify every bound cache-key field and invalidation path.
- Embedding query: verify seal trusted keys policy explain reuse invalidation compare full incremental

## IPS-042 Implement periodic checkpoints and delta-chain compaction

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: seal-compaction
- Depends on: IPS-041
- Goal id: IPS-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/checkpoint_policy.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/compaction.py, test/api/incremental_sealing/test_compaction.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_compaction.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/seals
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 24000
- Implementation timeout seconds: 6000
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/checkpoint_policy.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/compaction.py, test/api/incremental_sealing/test_compaction.py
- Submodules: none
- Interfaces: CheckpointPolicy, CheckpointDecision, compact_seal_chain
- Allow concurrent with:
- Conflict policy: Compaction verifies rather than trusts the chain and preserves history/retention references; checkpoint triggers cannot be overridden by an incremental caller.
- Preconditions: Seal verification and full checkpoint construction pass.
- Effects: Triggers full checkpoints by cadence, release tag, circuit/key/lock/trust/schema/cache corruption, low reuse, and maximum depth; compacts a verified chain into a new complete checkpoint.
- Evidence subset: ips/checkpoint-policy@1, ips/chain-compaction@1
- Symbolic first: true
- Acceptance: Complete chain/current manifest/every current unit/new forest verify; historical references required by retention survive; broken chains or required evidence loss reject rather than compact.
- Embedding query: periodic full checkpoint policy delta chain depth compaction verify history retention

## IPS-043 Expose the required public APIs and narrowly scoped zk-seal CLI

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: public-api-cli
- Depends on: IPS-042
- Goal id: IPS-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/__init__.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/cli.py, test/api/incremental_sealing/test_public_api.py, test/api/incremental_sealing/test_cli.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_public_api.py test/api/incremental_sealing/test_cli.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/public
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 23000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/__init__.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/cli.py, test/api/incremental_sealing/test_public_api.py, test/api/incremental_sealing/test_cli.py
- Submodules: none
- Interfaces: create_full_checkpoint, create_incremental_plan, execute_incremental_plan, verify_seal, explain_reuse, explain_invalidation, compare_full_and_incremental, zk-seal CLI
- Allow concurrent with:
- Conflict policy: This task solely owns accelerate public exports and CLI spelling; the surface remains local/narrow and adds no service, GUI, agent framework, or auto-install behavior.
- Preconditions: Checkpoint, delta, execution, verification, explanation, comparison, and compaction APIs pass internally.
- Effects: Adds full, incremental --parent, verify, plan --parent, explain-reuse, explain-invalidation, benchmark, cache-status, and force-full commands with stable machine-readable statuses/errors.
- Evidence subset: ips/public-api@1, ips/cli@1
- Symbolic first: true
- Acceptance: All seven requested Python APIs and nine CLI operations are exercised; production seals reject simulated evidence; missing optional capabilities are typed and cold help/import has no process/network/key/state side effect.
- Embedding query: incremental proof public api zk seal full incremental verify plan explain benchmark cache status force full

## IPS-044 Add hermetic bootstrap and truthful legacy proof-receipt migration

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: public-migration
- Depends on: IPS-043, IPS-012, IPS-027
- Goal id: IPS-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/bootstrap.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/migration.py, pyproject.toml, ipfs_datasets_py/pyproject.toml, ipfs_kit_py/pyproject.toml, test/api/incremental_sealing/test_imports_and_migration.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_imports_and_migration.py test/api/test_proof_reuse_cross_repository_e2e.py test/api/test_proof_reuse_subprocess_benchmark.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/public
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 19000
- Implementation timeout seconds: 4800
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/bootstrap.py, ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/migration.py, pyproject.toml, ipfs_datasets_py/pyproject.toml, ipfs_kit_py/pyproject.toml, test/api/incremental_sealing/test_imports_and_migration.py
- Submodules: ipfs_datasets_py, ipfs_kit_py
- Interfaces: IncrementalSealingBootstrap, LegacyEvidenceMigrationResult, migrate_legacy_evidence
- Allow concurrent with:
- Conflict policy: This task solely owns accelerate bootstrap/migration code and the three pytest entry-point declarations; it consumes datasets and kit public adapters without cloning their schema/storage authorities.
- Preconditions: Accelerate public API/CLI and both nested public migration adapters pass.
- Effects: Performs explicit dependency injection and accept/adapt/reverify/reject migration for existing proof/test receipts and caches, reconciles the known cross-repository pytest proof-reuse entry-point drift, and preserves actual integrity/signed/direct/simulated meaning.
- Evidence subset: ips/import-hermeticity@1, ips/cross-repository-migration@1
- Symbolic first: true
- Acceptance: Ordinary package import creates no files, keys, subprocesses, installs, network access, or daemon dependency; legacy evidence never enters the reusable cache without current-policy verification.
- Embedding query: hermetic bootstrap legacy proof receipt migration cache reverify no import side effects

## IPS-045 Build the deterministic fixture repository and proof-graph generator

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: fixture-corpus
- Depends on: IPS-042
- Goal id: IPS-G100
- Outputs: test/fixtures/incremental_proof_sealer/generate_fixture_history.py, test/fixtures/incremental_proof_sealer/fixture_manifest.json, test/api/incremental_sealing/test_fixture_generator.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_fixture_generator.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/fixtures-positive
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: test-fixtures
- Estimated tokens: 19000
- Implementation timeout seconds: 4800
- Predicted files: test/fixtures/incremental_proof_sealer/generate_fixture_history.py, test/fixtures/incremental_proof_sealer/fixture_manifest.json, test/api/incremental_sealing/test_fixture_generator.py
- Submodules: none
- Interfaces: deterministic fixture repository/history/proof graph corpus
- Allow concurrent with:
- Conflict policy: This task solely owns fixture generation and checked-in fixture metadata; fake/simulated evidence is labeled and exists only to test rejection or nonproduction plumbing.
- Preconditions: Seal lifecycle and compaction contracts define the fixture wire format.
- Effects: Generates tiny byte-stable repositories with separate source implementation/public-interface, test selector/source/add/delete, fixture, relevant configuration, network policy, verification policy, dependency lock, tool/prover version, circuit/key, proof-schema, canonicalization, checked-specification document, ordinary-documentation, graph/manifest, branch, merge, rollback, corruption, and independent-module mutations without relying on wall time or host environment.
- Evidence subset: ips/fixture-corpus@1
- Symbolic first: true
- Acceptance: Two clean generations are byte-identical; selector, fixture, configuration, network-policy, policy, lock, tool, schema, canonicalization, checked-spec, source/test/circuit/key/documentation/branch/merge/rollback scenarios each have explicit parents, changed-artifact provenance, expected direct/transitive unit closure, aggregate effect, and full-fallback decision; no fixture silently models simulated proving as production success.
- Embedding query: deterministic fixture repository history proof graph source test fixture lock circuit branch merge rollback

## IPS-046 Cover the complete positive invalidation and reuse matrix

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: positive-invalidation
- Depends on: IPS-045
- Goal id: IPS-G100
- Outputs: test/api/incremental_sealing/test_positive_matrix.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_positive_matrix.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/fixtures-positive
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 5400
- Predicted files: test/api/incremental_sealing/test_positive_matrix.py
- Submodules: none
- Interfaces: positive invalidation and reuse conformance matrix
- Allow concurrent with:
- Conflict policy: This task solely owns the positive matrix and may correct only demonstrated product defects; unrelated units must not be broadly invalidated merely to pass.
- Preconditions: Deterministic fixture history reproduces exactly.
- Effects: Tests independent source, public interface, test source/selector/delete/add, fixture, dependency lock, relevant configuration, network policy, verification policy, tool/prover version, circuit, verification key, proof schema, canonicalization, checked-specification document, ordinary documentation, two-module, and unrelated-edit reuse behavior.
- Evidence subset: ips/invalidation-positive@1
- Symbolic first: true
- Acceptance: Expected invalidated/reused/added/removed/fallback sets and reason-labeled dependency closure match exactly for selector, fixture, configuration, network policy, policy, lock, tool, schema, canonicalization, checked-spec, source/test/circuit/key/documentation mutations; deleted tests require authorization, added selected tests are proven, ordinary documentation reuse remains valid, checked specifications invalidate their consumers, trust/schema/canonicalization/key/circuit changes force full proof where required, and roots repeat deterministically.
- Embedding query: positive invalidation source interface test add delete fixture dependency config circuit key docs reuse

## IPS-047 Cover full/delta lifecycle, branches, merge, rollback, and compaction

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: positive-lifecycle
- Depends on: IPS-046, IPS-042
- Goal id: IPS-G100
- Outputs: test/api/incremental_sealing/test_seal_lifecycle.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_seal_lifecycle.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/fixtures-positive
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 23000
- Implementation timeout seconds: 6000
- Predicted files: test/api/incremental_sealing/test_seal_lifecycle.py
- Submodules: none
- Interfaces: full/delta/checkpoint/compaction positive lifecycle conformance
- Allow concurrent with:
- Conflict policy: This task solely owns the positive lifecycle matrix; branch ancestry and merge parents remain exact and rollback creates a new parent-bound transition rather than rewriting history.
- Preconditions: Positive invalidation and complete seal lifecycle APIs pass.
- Effects: Exercises first full seal, localized deltas, periodic/forced full checkpoints, correct-parent branches, merge commits, rollback, concurrent nonstale branches, chain verification, retention, and compaction.
- Evidence subset: ips/seal-lifecycle-positive@1
- Symbolic first: true
- Acceptance: Each seal is accepted only on its declared lineage; complete current units survive merge/rollback/compaction; repeated histories yield deterministic roots and retained historical references.
- Embedding query: full delta seal lifecycle branch merge rollback checkpoint compaction deterministic root

## IPS-048 Reject cache-context, manifest, and proof-forest tampering

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adversarial-cache-forest
- Depends on: IPS-047
- Goal id: IPS-G110
- Outputs: test/api/incremental_sealing/test_tamper_matrix.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_tamper_matrix.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/adversarial
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: security-validation
- Estimated tokens: 24000
- Implementation timeout seconds: 6000
- Predicted files: test/api/incremental_sealing/test_tamper_matrix.py
- Submodules: none
- Interfaces: cache-key, manifest, forest, and parent tamper conformance
- Allow concurrent with:
- Conflict policy: This task solely owns cache/manifest/forest tamper cases; product fixes must preserve exact cache keys, completeness, ordering, and unaffected leaves rather than weaken assertions.
- Preconditions: Positive seal lifecycle passes on the same deterministic corpus.
- Effects: Mutates source root, environment, test selector, fixture, relevant configuration, network policy, verification policy, dependency lock, tool/prover version, proof schema, canonicalization version, checked-specification binding, dependency closure, manifest membership/order, aggregate binding, parent, required replacements, duplicate leaves, unaffected leaves, and cached object/index bytes one field at a time.
- Evidence subset: ips/cache-tamper@1, ips/forest-tamper@1
- Symbolic first: true
- Acceptance: Different source/environment/selector/fixture/config/network-policy/policy/lock/tool/schema/canonicalization/checked-spec/dependency roots, unauthorized deleted test, changed manifest with old aggregate, wrong parent, missing invalidated unit, missing unaffected leaf, duplicate/reordered leaf, corruption, and poisoning all reject with typed reasons; no complete-key component can be omitted to regain reuse.
- Embedding query: negative cache context source environment selector dependency manifest aggregate parent forest poisoning

## IPS-049 Reject cryptographic, signature, key, circuit, and claim tampering

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adversarial-crypto-trust
- Depends on: IPS-048
- Goal id: IPS-G110
- Outputs: test/api/incremental_sealing/test_crypto_trust_negative.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_crypto_trust_negative.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/adversarial
- Parallel lane: accelerate
- Resource class: cpu-proof
- Resource stage: security-validation
- Estimated tokens: 25000
- Implementation timeout seconds: 6600
- Predicted files: test/api/incremental_sealing/test_crypto_trust_negative.py
- Submodules: none
- Interfaces: proof, public-input, receipt-signature, key/circuit, and assurance-claim negative conformance
- Allow concurrent with:
- Conflict policy: This task solely owns crypto/trust negatives; backend unavailability is typed/skipped only where explicitly optional, never converted into a passing cryptographic assertion.
- Preconditions: Cache/forest tamper matrix and configured real verifier test vectors pass.
- Effects: Substitutes verification/proving key IDs, circuit versions, public inputs, proof bytes, receipt signatures/signers, proof-system IDs, test-only designations, and direct-versus-receipt claim tags.
- Evidence subset: ips/crypto-trust-negative@1
- Symbolic first: true
- Acceptance: Old/unallowlisted keys, old circuit, modified public input, valid-format invalid cryptography, absent/invalid/untrusted receipt signature, unknown proof system, simulated-as-real, exposed secrets, and receipt-as-execution claims reject.
- Embedding query: negative cryptography public input verification key circuit signature signer unknown proof system simulated claim

## IPS-050 Inject all seven joined seal-transition crash boundaries

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adversarial-crash-recovery
- Depends on: IPS-047, IPS-026, IPS-044
- Goal id: IPS-G110
- Outputs: ipfs_kit_py/tests/proof_seal_store/test_incremental_sealer_recovery_integration.py
- Validation: python -m pytest -q ipfs_kit_py/tests/proof_seal_store/test_incremental_sealer_recovery_integration.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/adversarial
- Parallel lane: kit
- Resource class: io-large
- Resource stage: security-validation
- Estimated tokens: 23000
- Implementation timeout seconds: 6000
- Predicted files: ipfs_kit_py/tests/proof_seal_store/test_incremental_sealer_recovery_integration.py
- Submodules: ipfs_kit_py
- Interfaces: joined sealer/store seven-boundary recovery conformance
- Allow concurrent with:
- Conflict policy: This task solely owns the joined kit recovery fixture; it extends rather than rewrites IPS-026 durability semantics and never guesses that an ambiguous external prover succeeded.
- Preconditions: Positive seal lifecycle and kit crash/recovery conformance pass.
- Effects: Injects before proof execution; after proof before receipt; after receipt before forest; after forest before aggregate; after aggregate before seal; after seal before CAS; and after CAS before cleanup.
- Evidence subset: ips/joined-crash-matrix@1
- Symbolic first: true
- Acceptance: Restart deterministically chooses resume, replay, verify existing artifact, discard uncommitted artifact, repair, or full reproof as appropriate; only the post-CAS case is current and repeated recovery converges.
- Embedding query: seven crash points proof receipt forest aggregate seal cas cleanup recovery

## IPS-051 Run the joined adversarial seal and concurrent-writer matrix

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adversarial-e2e
- Depends on: IPS-048, IPS-049, IPS-050
- Goal id: IPS-G110
- Outputs: test/api/incremental_sealing/test_adversarial_e2e.py
- Validation: python -m pytest -q test/api/incremental_sealing/test_adversarial_e2e.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/adversarial
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: security-validation
- Estimated tokens: 25000
- Implementation timeout seconds: 6600
- Predicted files: test/api/incremental_sealing/test_adversarial_e2e.py
- Submodules: none
- Interfaces: cross-repository fail-closed adversarial conformance
- Allow concurrent with:
- Conflict policy: This task solely owns joined adversarial scenarios and may repair demonstrated integration bugs without changing datasets/kit authority or weakening a negative assertion.
- Preconditions: Cache/forest, cryptographic/trust, and joined crash matrices pass independently.
- Effects: Combines poisoned candidates, stale parent/branch replay, missing invalidated/required units, simulated/unknown/timeout outcomes, corrupted artifacts, old aggregates, unaffected-leaf loss, and racing writers through the public workflow.
- Evidence subset: ips/e2e-adversarial@1
- Symbolic first: true
- Acceptance: No stale/mismatched/corrupt/simulated/unknown/timeout evidence becomes sealed; an incremental seal missing one required replacement rejects; exactly one current-root writer wins and the prior accepted seal remains recoverable.
- Embedding query: end to end adversarial incremental seal stale parent missing proof simulated timeout corrupted cache concurrent writer

## IPS-052 Implement the deterministic forty-transition benchmark workload

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: benchmark-workload
- Depends on: IPS-051
- Goal id: IPS-G120
- Outputs: benchmarks/agent_supervisor/incremental_proof_sealer.py, test/benchmarks/test_incremental_proof_sealer_benchmark.py
- Validation: python -m pytest -q test/benchmarks/test_incremental_proof_sealer_benchmark.py
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/benchmark
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: benchmark
- Estimated tokens: 24000
- Implementation timeout seconds: 6600
- Predicted files: benchmarks/agent_supervisor/incremental_proof_sealer.py, test/benchmarks/test_incremental_proof_sealer_benchmark.py
- Submodules: none
- Interfaces: IncrementalProofBenchmark, forty-transition workload and result schema
- Allow concurrent with:
- Conflict policy: This task solely owns benchmark code/workload; it cannot bypass proof verification, alter invalidation/fallback, count simulations as proving, or conflate estimated and measured values.
- Preconditions: Joined positive and adversarial conformance pass.
- Effects: Encodes exactly 40 sequential controlled transitions spanning local/docs/test/schema/multimodule/lock/circuit/key/policy edits, periodic full checkpoints, branch, merge, rollback, and compaction, evaluating equivalent full and incremental work.
- Evidence subset: ips/benchmark-workload@1
- Symbolic first: true
- Acceptance: Stable seed/input produces the same task sequence and expected unit sets; every transition records required/reused/invalidated/new counts, cache hit, leaf/aggregate/verify/wall/CPU/GPU/memory/size/storage/cost/savings/fallback provenance.
- Embedding query: forty transition full versus incremental proof benchmark commits documentation test schema dependency circuit branch merge

## IPS-053 Execute the benchmark and persist provenance-rich result artifacts

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: benchmark-execution
- Depends on: IPS-052
- Goal id: IPS-G120
- Outputs: artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json, artifacts/agent_supervisor/incremental_proof_sealer/benchmark.csv
- Validation: python scripts/validate_incremental_proof_sealer_board.py --run-benchmark
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/benchmark
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: benchmark
- Estimated tokens: 15000
- Implementation timeout seconds: 10800
- Predicted files: artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json, artifacts/agent_supervisor/incremental_proof_sealer/benchmark.csv
- Proposal artifact envelope: {"max_file_bytes":2000000,"max_output_bytes":8000000,"max_patch_bytes":4000000,"paths":["artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json","artifacts/agent_supervisor/incremental_proof_sealer/benchmark.csv"],"schema":"ipfs_accelerate_py/agent-supervisor/task-artifact-envelope@1"}
- Submodules: none
- Interfaces: benchmark.json@2, benchmark.csv@1
- Allow concurrent with:
- Conflict policy: This task solely owns raw benchmark artifacts; record failures and missing counters honestly, keep host/revision/config/backend provenance, and never hand-edit savings to meet targets.
- Preconditions: Forty-transition benchmark code and conformance test pass.
- Effects: Because the sealed provider has no process authority, it writes the two declared result paths as the exact closed materialization requests `{"schema_version":"incremental-proof-sealer-materialization-request@1","task_id":"IPS-053"}` plus newline and `incremental-proof-sealer-materialization-request@1,IPS-053` plus newline. The protected `--run-benchmark` Validation is a convergent ensure operation. Before consuming exactly both request files it rejects every candidate index/worktree mutation outside those paths and the fixed ignored `release-work` root, including assume-unchanged/skip-worktree state, nested-repository dirt, ignored or untracked inputs, and link/special-file substitution. It makes no-local outer/nested clones of the exact bound commits, removes remotes, rejects object alternates, replacement refs, and grafts, and materializes only the two owned nested repositories. Every deliberately absent non-owned gitlink must match the closed reviewed path-to-object-ID allowlist; unknown, missing, present-on-disk, or drifted entries reject. It verifies every physical tracked leaf against its Git blob, makes the source read-only through held no-follow descriptors, and invokes the checked-in IPS-052 CLI there with fixed seed `20260811` and exactly 40 transitions. The CLI writes only to a sibling staged channel; the runner rehashes the source, admits exactly the two bounded regular outputs through held directory descriptors, atomically publishes them through held candidate-parent descriptors, then independently validates the closed result schema, ordered workload, parent-bound source revisions/trees, unit arithmetic, per-metric measured/estimated/unavailable provenance, fallback/status rules, deterministic roots, wrong-parent rejection, exact CSV projection, and the one-MiB-per-file bound within the reviewed proposal envelope. Pre/post candidate binding rejects concurrent checkout/index mutation, while the isolated materialization prevents such mutation from becoming executed code. An absent request does not start work. The supervisor stabilization invocation recognizes the complete valid bundle and performs read-only validation without rerunning the benchmark or rewriting outputs; read-only Git binding subprocesses still run. Partial bundles, unknown pre-existing bytes, or invalid completed bundles fail closed. Candidate validation requires `benchmark_worktree_parent_revision` to equal current HEAD; IPS-054 later accepts the exact no-ff output merge followed by tightly validated daemon status-only descendants, with unchanged JSON/CSV evidence and no source/config/test mutation.
- Evidence subset: ips/benchmark-results@1
- Symbolic first: false
- Acceptance: `benchmark.json` is canonical `incremental-proof-sealer-benchmark-results@2` for `incremental-proof-sealer-40-transition@1` and binds its exact task-parent revision without self-referencing a future completion commit; every row has the exact 00–39 scenario/index, observed planner counts with `required=reused+newly_proved` and `newly_proved=invalidated+added`, derived cache hit/savings, full/incremental cost, CPU/GPU/memory/time/size/storage values with per-field measured/estimated/unavailable provenance, exact mandatory full-checkpoint indices/reasons, and an honest reason/status when merge or policy transitions 17/29/38 conditionally fall back full. Full and incremental roots match, simulated required units are zero, and row 37 retains the typed stale-parent attempt. CSV is an exact ordered scalar projection. First Validation must turn the exact two-file request into process-observed evidence; one stabilization rerun must be read-only and hash-identical. Process failure, partial/request drift, source-lineage drift, non-output completion changes, or artifact drift exits nonzero; provider prose or hand-authored result files do not replace the protected materialization.
- Embedding query: run incremental proof benchmark result json csv provenance measured estimated savings

## IPS-054 Analyze actual reuse, compute, latency, size, and storage results

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: benchmark-analysis
- Depends on: IPS-053
- Goal id: IPS-G120
- Outputs: docs/architecture/INCREMENTAL_PROOF_SEALER_BENCHMARK.md, artifacts/agent_supervisor/incremental_proof_sealer/summary.json
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-054
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/benchmark
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 17000
- Implementation timeout seconds: 4200
- Predicted files: docs/architecture/INCREMENTAL_PROOF_SEALER_BENCHMARK.md, artifacts/agent_supervisor/incremental_proof_sealer/summary.json
- Submodules: none
- Interfaces: performance evidence summary
- Allow concurrent with:
- Conflict policy: This task solely owns benchmark interpretation; report arithmetic from raw artifacts, distinguish estimates, and state unmet 70/50/80-percent goals without excuse or inflation.
- Preconditions: Raw benchmark artifacts validate and retain provenance.
- Effects: Computes average reuse/compute reduction, localized/docs cases, best/worst/fallback, proof/seal sizes, verification latency, storage growth, CPU/GPU/memory, and deterministic-root evidence with uncertainty and limitations.
- Evidence subset: ips/performance-analysis@1
- Symbolic first: true
- Acceptance: Canonical `incremental-proof-sealer-benchmark-summary@1` binds the raw benchmark SHA-256 and exactly recomputes provenance counts, unweighted reuse and available-cost reduction means, best/worst cases, forced-full indices, per-metric availability/min/max/mean, and the localized-70/mixed-50/documentation-80 target assessments. Markdown cites the digest and schema, labels measured/estimated/unavailable values, says targets are not facts, receipt aggregation does not prove test execution, simulated required units cannot satisfy a production seal, and reports unavailable evidence honestly.
- Embedding query: benchmark analysis reuse compute reduction best worst proof seal size latency storage target actual

## IPS-055 Document the precise trust model and migration guidance

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-documentation
- Depends on: IPS-044, IPS-051, IPS-054
- Goal id: IPS-G130
- Outputs: docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_MODEL.md, docs/architecture/INCREMENTAL_PROOF_SEALER_MIGRATION.md
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-055
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/release
- Parallel lane: accelerate
- Resource class: cpu-medium
- Resource stage: documentation
- Estimated tokens: 21000
- Implementation timeout seconds: 4800
- Predicted files: docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_MODEL.md, docs/architecture/INCREMENTAL_PROOF_SEALER_MIGRATION.md
- Submodules: none
- Interfaces: production trust model and existing-receipt/cache migration guide
- Allow concurrent with:
- Conflict policy: This task solely owns trust/migration documentation and must describe measured executable behavior; no documentation claim upgrades receipts, manifest aggregation, or integrity commitments.
- Preconditions: Public migration, adversarial tests, and benchmark analysis are final.
- Effects: Documents proof-class claims/nonclaims, public/private inputs, signer trust, setup/key origins/allowlists, recursion or manifest strategy, network/environment trust, cache admission, fallback, retention/recovery, and staged legacy migration.
- Evidence subset: ips/trust-model@1, ips/migration-guide@1
- Symbolic first: true
- Acceptance: It separately defines the five proof classes and their exact nonclaims; public/private inputs and sensitive-witness handling; signer/signature trust; whether child signatures are circuit-verified and whether test execution is directly proven; manifest aggregation versus recursion; setup origin, test-only keys, content-addressed allowlisted verification keys, no silent production key generation, unknown-system/arbitrary-circuit rejection, verify-before-cache admission, full-checkpoint triggers, CAS/WAL/ambiguous-prover recovery, and remaining production work. Migration explicitly classifies accept/adapt/reverify/reject/simulated and integrity/signed/direct evidence without assurance upgrade, including allowlist, schema/canonicalization fallback, staged rollback, unknown legacy systems, and witness/key handling.
- Embedding query: incremental proof trust model setup keys signer receipts direct execution manifest aggregation migration

## IPS-056 Run terminal current-tree validation and publish the final report

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-fan-in
- Depends on: IPS-055
- Goal id: IPS-G130
- Outputs: docs/architecture/INCREMENTAL_PROOF_SEALER_REPORT.md, artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json, artifacts/agent_supervisor/incremental_proof_sealer/release_validation.log
- Validation: python scripts/validate_incremental_proof_sealer_board.py --run-release-validation
- Board namespace: incremental-proof-sealer-v1
- Bundle: incremental-proof-sealer/release
- Parallel lane: accelerate
- Resource class: cpu-large
- Resource stage: release-validation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: docs/architecture/INCREMENTAL_PROOF_SEALER_REPORT.md, artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json, artifacts/agent_supervisor/incremental_proof_sealer/release_validation.log
- Proposal artifact envelope: {"max_file_bytes":7000000,"max_output_bytes":20000000,"max_patch_bytes":12000000,"paths":["docs/architecture/INCREMENTAL_PROOF_SEALER_REPORT.md","artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json","artifacts/agent_supervisor/incremental_proof_sealer/release_validation.log"],"schema":"ipfs_accelerate_py/agent-supervisor/task-artifact-envelope@1"}
- Submodules: none
- Interfaces: terminal release report and current-tree validation receipt
- Allow concurrent with:
- Conflict policy: This task solely owns the final report/validation receipt; it may repair only demonstrated integration/regression defects, must commit nested repositories before gitlink updates, and may not edit operator-owned board controls.
- Preconditions: Public/migration, adversarial, benchmark, trust, and migration-documentation fan-ins pass.
- Effects: Because the sealed provider has no process authority, it writes a substantive final report containing exactly one `<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->` marker and writes the declared JSON/log paths as the exact closed `incremental-proof-sealer-materialization-request@1` requests for `IPS-056`. The protected `--run-release-validation` Validation is a convergent ensure operation. Before consuming the exact request bundle it applies the same closed outer/nested index/worktree gate and verified no-local, no-remote, no-alternates, replacement/graft rejection, closed absent-gitlink path/OID policy, and per-leaf-hashed descriptor-read-only source materialization as IPS-053. Only from that isolated source it refuses any live `ipfs` resolved from its fixed PATH, executes `--check-terminal`, all 17 protected existing ZK/reuse/WAL/release suites, and the three new incremental-sealing suites with explicit argv, fixed offline/autoinstall-disabled workspaces including workspace-local pytest-benchmark storage, bounded timeouts, a 256-KiB per-process output limit, and Linux subreaper plus `/proc` process-tree termination, including session-detached descendants. A live residual descendant is killed and makes the command non-completed. A post-run full rehash rejects materialized-source mutation, pre/post candidate binding rejects concurrent task-worktree/index mutation, and the release staged channel must stay empty. It retains at most six MiB of combined secret-scanned strict-UTF-8/NUL-free text log evidence, bounds the receipt at two MiB and report at one MiB, and keeps the aggregate under the reviewed 12-MiB patch/20-MiB serialized text proposal envelope. It writes canonical `incremental-proof-sealer-release-validation@2` process evidence, then replaces only the single report marker with the fresh receipt digest, exact three source revisions, and actual ordered `baseline_compatible_non_green` IDs before full artifact validation. An absent request does not start work. The supervisor stabilization invocation recognizes the complete valid report/receipt/log bundle, requires the materialization marker to be absent, and performs read-only validation without rerunning terminal/pytest subprocesses or rewriting outputs; read-only Git binding subprocesses still run. Partial, duplicated/missing-marker, unknown pre-existing, or invalid completed bundles fail closed. Post-commit verification finds the exact no-ff output merge and permits only tightly validated daemon status-only descendants while requiring unchanged evidence bytes.
- Evidence subset: ips/final-report@1, ips/release-conformance@1
- Symbolic first: true
- Acceptance: The first protected release ensure invocation, not provider-authored prose, observes a successful terminal gate plus exactly 17 existing and 3 new current-tree suites, exact contiguous secret-scanned log evidence, and unchanged source revisions/trees; its single bounded report substitution binds that new evidence, and the stabilization invocation is read-only with identical report/receipt/log hashes. All three new suites require complete nonzero collection, exit zero, and zero failed/error/xpassed/skipped/xfailed/deselected outcomes. Each existing suite must be green or baseline-compatible-or-improved against its exact protected operator observation: no new non-pass node/category, no increased failed/error/xpassed/skipped/xfailed/deselected outcomes, no lost passes, consistent selected/collection arithmetic, and no new collection abort; an incomplete baseline collection must either retain the exact blocker or improve to complete collection. Every retained skip, xfail, deselection, failure, error, xpass, or collection issue is labeled `baseline_compatible_non_green` and named in the final report, never hidden as success. The final report binds the receipt digest and exact commits; discovered real/simulated/structural/direct systems/tests; modules, granularity, complete key/invalidation/fallback/Merkle-manifest rules; all provenance-labeled 40-transition metrics; size/latency/storage; crash/tamper results; direct/signed/integrity claims; and remaining production work. It says pytest process outputs were observed but execution was not cryptographically proven and uses the narrow requested final claim without treating stale or simulated evidence as current verification.
- Embedding query: final incremental proof sealer report exact commits tests benchmark crash tamper trust claims remaining production work

## IPS-057 Resolve merge retry-budget failure for IPS-003

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: IPS-000
- Outputs: ipfs_kit_py/docs/architecture/incremental_proof_sealer_inventory.json, ipfs_kit_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-003
- Parallel lane: baseline-kit
- Predicted files: ipfs_kit_py/docs/architecture/incremental_proof_sealer_inventory.json, ipfs_kit_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md
- Conflict policy: Documentation inventory only from read-only code and protected operator evidence; the provider has no shell authority and must not run pytest, regenerate/edit receipts or logs, start IPFS, auto-install, perform proof setup, use pseudo-CID/legacy WAL as authority, or write outside the two declared inventory outputs.
- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1
- Retry repair source: IPS-003
- Retry failure kind: merge
- Retry repair discovery: /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-057-ips-003-merge-retry-budget.md
- Canonical board task: false
- Acceptance: Merge retry-budget guardrail filed this from repeated merge failures in IPS-003. Use evidence in /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-057-ips-003-merge-retry-budget.md to fix the merge blocker, verify the intended implementation changes are committed in their owning repository or submodule, run `ipfs-accelerate-agent-merge-resolver --events-path ... --apply` when the conflict is semantic, then mark this repair task completed so the supervisor can release IPS-003 from strategy blocked_tasks.

## IPS-058 Resolve validation retry-budget failure for IPS-001

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: IPS-000
- Outputs: docs/architecture/incremental_proof_sealer_inventory/accelerate.json, docs/architecture/incremental_proof_sealer_inventory/accelerate.md
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-001
- Parallel lane: baseline-accelerate
- Predicted files: docs/architecture/incremental_proof_sealer_inventory/accelerate.json, docs/architecture/incremental_proof_sealer_inventory/accelerate.md
- Conflict policy: Read only existing bound code and protected operator evidence; the provider has no shell authority and must not run pytest, regenerate/edit receipts or logs, install, download, build, create keys, or write outside the two declared inventory outputs.
- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1
- Retry repair source: IPS-001
- Retry failure kind: validation
- Retry repair discovery: /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-058-ips-001-retry-budget.md
- Canonical board task: false

- Acceptance: Retry-budget guardrail filed this from repeated validation failures in IPS-001. Use evidence in /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-058-ips-001-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release IPS-001 from strategy blocked_tasks.

## IPS-059 Resolve 1 dirty backlogged worktrees blocked by content_not_in_target

- Status: blocked
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P2
- Track: ops
- Generated by: ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1
- Reconciliation kind: dirty_backlogged_worktree
- Reconciliation reason: content_not_in_target
- Reconciliation fingerprint: 9a1e2455799a7c0c5dd02a935e37a657840a58f9
- Reconciliation discovery: /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-059-reconciliation-9a1e2455799a.md
- Canonical board task: false
- Fingerprint: 9a1e2455799a7c0c5dd02a935e37a657840a58f9
- Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:content_not_in_target
- Depends on:
- Outputs: data/agent_supervisor/incremental_proof_sealer/state/discovery, docs/architecture/incremental_proof_sealer.todo.md
- Board namespace: incremental-proof-sealer-v1
- Goal id: IPS-G130
- Bundle: incremental-proof-sealer/release
- Parallel lane: accelerate
- Resource class: cpu-large
- Validation: test -f /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-059-reconciliation-9a1e2455799a.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by content_not_in_target. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-059-reconciliation-9a1e2455799a.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## IPS-060 Resolve 3 preflight-conflicting backlogged worktree merges

- Status: blocked
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Generated by: ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1
- Reconciliation kind: preflight_merge_conflict
- Reconciliation reason: preflight_merge_conflict
- Reconciliation fingerprint: 4a4ccc39bf0ba238ec50bf8c44e9105bc8a6ed8f
- Reconciliation discovery: /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-060-reconciliation-4a4ccc39bf0b.md
- Canonical board task: false
- Fingerprint: 4a4ccc39bf0ba238ec50bf8c44e9105bc8a6ed8f
- Dedupe key: reconciliation_guardrail:preflight_merge_conflict
- Depends on:
- Outputs: data/agent_supervisor/incremental_proof_sealer/state/discovery, docs/architecture/incremental_proof_sealer.todo.md
- Board namespace: incremental-proof-sealer-v1
- Goal id: IPS-G130
- Bundle: incremental-proof-sealer/release
- Parallel lane: accelerate
- Resource class: cpu-large
- Validation: test -f /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-060-reconciliation-4a4ccc39bf0b.md
- Acceptance: Reconciliation guardrail filed this because 3 branch or worktree cleanup candidates are blocked by preflight_merge_conflict. Use evidence and the machine-readable reconciliation plan in /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-060-reconciliation-4a4ccc39bf0b.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## IPS-061 Resolve dirty main checkout blocking 1 worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Generated by: ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1
- Reconciliation kind: main_checkout_dirty
- Reconciliation reason: main_checkout_dirty
- Reconciliation fingerprint: 8de091b195730ec5fc6097a945072def7e4f2832
- Reconciliation discovery: /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-061-reconciliation-8de091b19573.md
- Canonical board task: false
- Fingerprint: 8de091b195730ec5fc6097a945072def7e4f2832
- Dedupe key: reconciliation_guardrail:main_checkout_dirty
- Depends on:
- Outputs: data/agent_supervisor/incremental_proof_sealer/state/discovery, docs/architecture/incremental_proof_sealer.todo.md
- Board namespace: incremental-proof-sealer-v1
- Goal id: IPS-G130
- Bundle: incremental-proof-sealer/release
- Parallel lane: accelerate
- Resource class: cpu-large
- Validation: test -f /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-061-reconciliation-8de091b19573.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by main_checkout_dirty. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-061-reconciliation-8de091b19573.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## IPS-062 Resolve dirty main checkout blocking 2 worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Generated by: ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1
- Reconciliation kind: main_checkout_dirty
- Reconciliation reason: main_checkout_dirty
- Reconciliation fingerprint: 111fede4c879a1059cbe6ee46e203c56c495e9d4
- Reconciliation discovery: /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-062-reconciliation-111fede4c879.md
- Canonical board task: false
- Fingerprint: 111fede4c879a1059cbe6ee46e203c56c495e9d4
- Dedupe key: reconciliation_guardrail:main_checkout_dirty
- Depends on:
- Outputs: data/agent_supervisor/incremental_proof_sealer/state/discovery, docs/architecture/incremental_proof_sealer.todo.md
- Board namespace: incremental-proof-sealer-v1
- Goal id: IPS-G130
- Bundle: incremental-proof-sealer/release
- Parallel lane: accelerate
- Resource class: cpu-large
- Validation: test -f /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-062-reconciliation-111fede4c879.md
- Acceptance: Reconciliation guardrail filed this because 2 branch or worktree cleanup candidates are blocked by main_checkout_dirty. Use evidence and the machine-readable reconciliation plan in /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-062-reconciliation-111fede4c879.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## IPS-063 Resolve dirty main checkout blocking 3 worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Generated by: ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1
- Reconciliation kind: main_checkout_dirty
- Reconciliation reason: main_checkout_dirty
- Reconciliation fingerprint: 79743a495e084c938856b7149b23d02979bf80e1
- Reconciliation discovery: /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-063-reconciliation-79743a495e08.md
- Canonical board task: false
- Fingerprint: 79743a495e084c938856b7149b23d02979bf80e1
- Dedupe key: reconciliation_guardrail:main_checkout_dirty
- Depends on:
- Outputs: data/agent_supervisor/incremental_proof_sealer/state/discovery, docs/architecture/incremental_proof_sealer.todo.md
- Board namespace: incremental-proof-sealer-v1
- Goal id: IPS-G130
- Bundle: incremental-proof-sealer/release
- Parallel lane: accelerate
- Resource class: cpu-large
- Validation: test -f /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-063-reconciliation-79743a495e08.md
- Acceptance: Reconciliation guardrail filed this because 3 branch or worktree cleanup candidates are blocked by main_checkout_dirty. Use evidence and the machine-readable reconciliation plan in /home/barberb/.local/state/ipfs_accelerate_py/incremental-proof-sealer-v1/repo/data/agent_supervisor/incremental_proof_sealer/state/discovery/2026-08-12-ips-063-reconciliation-79743a495e08.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.
