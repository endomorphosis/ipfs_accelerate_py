# IncrementalProofSealer trust baseline (IPS-004)

Immutable cross-repository trust matrix and implementation boundary for
`incremental-proof-sealer-trust-baseline@2`. Later implementation tasks cite
exactly one semantic, storage, or execution authority from this milestone.
Unsupported recursion defaults to Merkle manifest aggregation. Baseline
failures remain visible process observations and are not hidden.

This report freezes operator-captured receipts and IPS-001/002/003 inventories
as historical, intrinsically revalidated evidence. After this milestone merges,
supervisor restarts do not require later implementation HEADs to equal the
captured tested source state.

## Schema and parent binding

| Field | Value |
| --- | --- |
| `schema_version` | `incremental-proof-sealer-trust-baseline@2` |
| `synthesis_worktree_parent_revision` | `96d104da51950f01fa379c7f9f9d50fd47d3c09c` |

`synthesis_worktree_parent_revision` equals the candidate task worktree HEAD
during IPS-004 candidate validation. After merge, preflight recognizes this
milestone only when both declared outputs were committed together as one direct
child of that declared parent.

Canonical machine authority:
`docs/architecture/incremental_proof_sealer_inventory/matrix.json`.

## Baseline receipts (protected three-pin scheduler map)

`baseline_receipts` equals the protected scheduler `operator_baseline_receipts`
map exactly. Each pin is integrity-protected process observation only:
`pytest_execution_not_cryptographically_proven`.

### IPS-001 accelerate

| Field | Value |
| --- | --- |
| task_id | `IPS-001` |
| path | `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/accelerate.json` |
| receipt_digest | `sha256:a85bc27f70dabbbea49d26200fee27e43dbf61102beb6a9789df7a15f367474e` |
| planning_revision | `8881344bb2162f3f8d82f22d8348bc0ac7536f95` |
| source_revision | `9b43e0ea1c3cf651d884f5489ca46d7eda2ae41b` |
| source_tree | `e64efa9fffd0882f235d34d27e2c2c423eb89c6d` |
| required_command_ids | `accelerate-proof-focused-core-15`, `accelerate-proof-focused-wide-36`, `accelerate-proof-reuse-migration`, `accelerate-proof-reuse-cross-repo` |
| nonclaim | `pytest_execution_not_cryptographically_proven` |

Retained logs:

- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/accelerate-proof-focused-core-15-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/accelerate-proof-focused-wide-36-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/accelerate-proof-reuse-migration-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/accelerate-proof-reuse-cross-repo-20260812T221104.565941Z-1812421.log`

Suite definition digests:

- `accelerate-proof-focused-core-15` = `sha256:c6701b3552e88b61e11e86907ea24863fd233401bec1cfdcddb2d44874eb157f`
- `accelerate-proof-focused-wide-36` = `sha256:92e1f8df4d56363f6899b29c797d9d34013a859e1504a9e018897dbd48cc09fb`
- `accelerate-proof-reuse-cross-repo` = `sha256:0c1510628b48aab11c76287d9adbf942253bb269bd792507b4897bc4732e471b`
- `accelerate-proof-reuse-migration` = `sha256:ca91a14fa1a51f07b5c61846fe71f85a921c57c52d6d95e0654c11eaddab9d45`

### IPS-002 datasets

| Field | Value |
| --- | --- |
| task_id | `IPS-002` |
| path | `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/datasets.json` |
| receipt_digest | `sha256:7dbe6a2f3b6fcbf56c841d2c757951a76b7dc91a6dc91614c6e618e4c38cdb81` |
| planning_revision | `bd2ff6245ebe476fc744d45c7c66235c92b0e19c` |
| source_revision | `cae71d992f82ae7a0975ba4f5ed0c575b1479253` |
| source_tree | `86fbb01962ac237b3c39a5d94d947026c84033ea` |
| required_command_ids | `datasets-zkp-focused-current`, `datasets-zkp-unit-wide-current`, `datasets-proof-cache-adapters`, `datasets-zkp-broad-safe-current` |
| nonclaim | `pytest_execution_not_cryptographically_proven` |

Retained logs:

- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/datasets-zkp-focused-current-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/datasets-zkp-unit-wide-current-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/datasets-proof-cache-adapters-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/datasets-zkp-broad-safe-current-20260812T221104.565941Z-1812421.log`

Suite definition digests:

- `datasets-proof-cache-adapters` = `sha256:a726bc186495c05852518d2d4036920f7c94c5f94fd339b10350eaaca3ab57bc`
- `datasets-zkp-broad-safe-current` = `sha256:bb2b89fd5e717ea8cc7a977f7a2e26f958151e1e1a1ffa14365c97c4cb387802`
- `datasets-zkp-focused-current` = `sha256:698e8b10542034fe7d993371efcf1479edfd1c09a06d94610215702a9eda4fa4`
- `datasets-zkp-unit-wide-current` = `sha256:04a74051a55e1fc36eaa3b1e12a41fc58cb8e04fc27938b8f4f89af7bd9dad02`

### IPS-003 kit

| Field | Value |
| --- | --- |
| task_id | `IPS-003` |
| path | `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/kit.json` |
| receipt_digest | `sha256:22d4f9663e3346fd2264efb38538cbedead642c3ba7fe403b5c7b9fc6545f982` |
| planning_revision | `5a7a2df8181cfdc33bc19be09989df7ff83f2d4e` |
| source_revision | `b2c8e625b184c41fa865d906e0037915e3fb9179` |
| source_tree | `d3dd2f1e5984124e1dfe94e9e5f3cfeb828b570e` |
| required_command_ids | `kit-proof-certificate`, `kit-reuse-capabilities`, `kit-profile-d`, `kit-coordination`, `kit-modern-wal`, `kit-proof-reuse-bootstrap`, `kit-agent-receipts`, `kit-iroh-release`, `kit-release-receipt` |
| nonclaim | `pytest_execution_not_cryptographically_proven` |

Retained logs:

- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-proof-certificate-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-reuse-capabilities-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-profile-d-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-coordination-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-modern-wal-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-proof-reuse-bootstrap-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-agent-receipts-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-iroh-release-20260812T221104.565941Z-1812421.log`
- `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/logs/kit-release-receipt-20260812T221104.565941Z-1812421.log`

Suite definition digests:

- `kit-agent-receipts` = `sha256:8d7942d828d7e82ffaf202bb118274896569bda983e72b44c184487e19b45a0c`
- `kit-coordination` = `sha256:f193fe07b5ba3109fd191b60c2e0fa0c09abd4fc211562ffb65f3585cc549d91`
- `kit-iroh-release` = `sha256:ddb1e17b9fd03e01e2a268198cae7ce960fa209fb92c477f406fff5fabe6725d`
- `kit-modern-wal` = `sha256:ecdc91c4ef4866b8f983417559dc0cb8baa92715f78ecab3bd846ae1d0931ee9`
- `kit-profile-d` = `sha256:3dca13e6aeaf8419e11f774d5c25c373143ffcfd26644ac8205a850e9efb727f`
- `kit-proof-certificate` = `sha256:876154cb523cb321c9caac891e63bd0073d0a8de3251f8399e6165be1dbb2e07`
- `kit-proof-reuse-bootstrap` = `sha256:59797fdea072393b1d45ba502523c1ab8c29d7a37db55ed860d8eaa3d2722915`
- `kit-release-receipt` = `sha256:0b221ff9296415784fdea1716fc02fb5a26f203eca209eb74300d96f6143161d`
- `kit-reuse-capabilities` = `sha256:437a313abdfdf2b6b7d24f925c85daf5aa84f8243f0750f388312344fa9daae3`

## Inventory artifacts (IPS-001 / IPS-002 / IPS-003)

Each inventory artifact binds exact inventory and report paths plus the
inventory-output-only completion revision.

| Task | inventory | report | completion_revision |
| --- | --- | --- | --- |
| `IPS-001` | `docs/architecture/incremental_proof_sealer_inventory/accelerate.json` | `docs/architecture/incremental_proof_sealer_inventory/accelerate.md` | `bb5d184ab8355def188ad5775a664a6790e54a63` |
| `IPS-002` | `ipfs_datasets_py/docs/architecture/incremental_proof_sealer_inventory.json` | `ipfs_datasets_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md` | `7ea9822f11af9a6c3024ac6c29ced6270aa4321d` |
| `IPS-003` | `ipfs_kit_py/docs/architecture/incremental_proof_sealer_inventory.json` | `ipfs_kit_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md` | `f7074e1175505e8e1f0ea44a9e1f1db5ea2891db` |

## Repository authorities

Exact single-authority split for later implementation tasks:

| Decision key | Authority |
| --- | --- |
| `proof_unit_manifest_identity` | `ipfs_datasets_py` |
| `proof_object_cache_forest_wal_cas` | `ipfs_kit_py` |
| `prover_scheduler_aggregation_planner_metrics` | `ipfs_accelerate_py` |

- **datasets** (`proof_unit_manifest_identity` = `ipfs_datasets_py`): semantic
  authority for proof units, manifests, identities, statements, dependency
  graphs, requirement discovery, invalidation, and commitment codecs.
- **kit** (`proof_object_cache_forest_wal_cas` = `ipfs_kit_py`): storage
  authority for immutable proof objects, exact-key cache indexes that never
  decide acceptance, proof-forest persistence, WAL durability, and
  repository/branch-namespaced current-seal compare-and-swap.
- **accelerate** (`prover_scheduler_aggregation_planner_metrics` =
  `ipfs_accelerate_py`): execution authority for provers, schedulers,
  planners, aggregation orchestration, trust/key allowlists, CLI, and metrics.

Kit never decides whether a cached proof is valid. Datasets never persists a
second proof cache. Accelerate never invents a competing CID, manifest, Merkle,
or storage authority.

## Proof-class decisions

Closed claim boundaries. No generic `zk_verified: true` may erase them.

| Decision key | Decision value |
| --- | --- |
| `integrity_commitment` | `integrity_only` |
| `signed_execution_receipt` | `trusted_signer_assertion_not_direct_execution` |
| `receipt_aggregation_zk_proof` | `receipt_completeness_not_test_execution` |
| `direct_execution_proof` | `declared_computation_only` |
| `incremental_commit_seal` | `parent_bound_verified_leaf_transition` |

Interpretation:

- integrity-only: exact bytes, digest, CID, and Merkle inclusion; not execution
  or semantic correctness.
- trusted-signer receipt: allowlisted signer asserted execution; not independent
  proof that execution occurred without trusting the signer.
- receipt-completeness aggregation: admitted committed receipt fields and exact
  required receipt set/count/order; not underlying test execution unless
  signature verification and signer trust are inside the declared statement.
- declared direct computation: the declared program/verifier ran over committed
  inputs and produced the committed output/property only.
- parent-bound leaf-transition: accepted parent, explicit state transition,
  valid reused/replacement leaves, complete new manifest, and new repository
  verification root; not arbitrary repository correctness.

## Aggregation decision

| Decision key | Decision value |
| --- | --- |
| `mode` | `merkle_manifest_aggregation` |
| `recursive_self_verification_supported` | `false` |
| `child_proofs_individually_verified` | `true` |
| `test_execution_directly_proven` | `false` |

Unsupported recursion defaults to `merkle_manifest_aggregation` with
individually verified child leaves and a Merkle completeness commitment.
Recursive self-verification is not supported (`false`). Test execution is not
directly proven (`false`).

## Backend decisions

| Decision key | Decision value |
| --- | --- |
| `existing_recursive_backend` | `unsupported` |
| `groth16` | `bounded_declared_computation_only` |
| `provekit` | `optional_capability_unavailable_is_typed` |
| `simulated` | `production_seal_forbidden` |
| `unknown` | `rejected` |

Unknown and simulated production evidence are rejected. Simulated required units
cannot produce production seals. Missing optional ProveKit capability is typed
unavailable, never fabricated.

## Trust nonclaims

The following are explicitly not established by this baseline:

1. `entire_repository_proven_correct`
2. `pytest_execution_cryptographically_proven`
3. `semantically_correct_change`
4. `recursive_proof_verification_available`

Repository correctness, pytest execution proof, semantic change correctness, and
recursive proof verification are not established. Operator receipts remain
`pytest_execution_not_cryptographically_proven` process observations.

## Milestone use

Every later implementation task can cite one exact semantic/storage/execution
authority from `repository_authorities`. Baseline failures recorded in the
pinned receipts are not hidden. After this immutable milestone merges, later
supervisor restarts treat the still-intrinsically-validated operator
observations as historical rather than requiring implementation HEADs to equal
the captured source state.
