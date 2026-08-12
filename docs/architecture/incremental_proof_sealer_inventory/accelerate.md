# IncrementalProofSealer accelerate inventory (IPS-001)

Static source inventory of `ipfs_accelerate_py` proof backends, schedulers,
receipts, caches, seals, CLI-adjacent surfaces, and baseline-reference pins at
the receipt-tested control lineage. This document is a companion to
`docs/architecture/incremental_proof_sealer_inventory/accelerate.json`.

## Revisions

| Field | Value |
| --- | --- |
| `planning_revision` | `8881344bb2162f3f8d82f22d8348bc0ac7536f95` |
| `inventory_worktree_parent_revision` | `1d7b2ef78e24cfa82e4f8437faefe37c3a45d29a` |

`inventory_worktree_parent_revision` is immutable and equals the accelerate
task-start parent during candidate validation. Final task commits come from
supervisor completion evidence and are not self-embedded here.

## Baseline evidence (reference only)

Operator-captured process observation only. This inventory does not restate
command lines, outcome tallies, logs, or execution claims.

| Field | Value |
| --- | --- |
| path | `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/accelerate.json` |
| receipt_digest | `sha256:8db3857c748ef217a9da15dcc4cef81d8522170e44d0ee7cf9c337bc9f4acc52` |
| required_command_ids | `accelerate-proof-focused-core-15`, `accelerate-proof-focused-wide-36`, `accelerate-proof-reuse-migration`, `accelerate-proof-reuse-cross-repo` |
| evidence_origin | `operator_capture` |
| assurance | `process_observed_only` |
| nonclaim | `pytest_execution_not_cryptographically_proven` |

The protected closed suite registry and validator independently recompute suite
preimages, argv, controlled-offline environment, digests, log sizes, counts,
and incomplete-collection evidence nodes. Providers only reference the pin above.

## Inspection method

- classification_method: static source inventory
- Static scans report `surfaces_found` only; they never assert suite outcomes
- Static inspection is not pytest execution and is not cryptographic proof
- Controlled-offline capture disables Groth16/ProveKit enablement, builds,
  downloads, and auto-install; the receipt reference does not establish new
  real proving

## Explicit nonclaims

1. `TestPassReceipt` and `ProofReceipt` are **unsigned** integrity/assertion
   envelopes, not signed receipts.
2. Cache admission gates are **not** receipt-aggregation proofs.
3. `proof_attestation` binds statements only to existing independently
   kernel-checked `ProofReceipt` values; it does not claim arbitrary Python
   correctness from a ZKP.
4. ProveKit/Groth16 binding surfaces close without tools/keys and never install
   packages; absence is typed unavailable.
5. `proof_reuse_real_groth16_fixture` is a disposable test-only fixture path;
   missing artifacts remain typed gaps.
6. No inspected accelerate surface implements a reliable recursive verifier;
   default aggregation remains Merkle manifest completeness.
7. Direct-execution, recursion, and production-key trust claims require
   executable evidence beyond static surface presence.
8. Planning-time historical counts and the unreproducible 257-result slice are
   **not** baseline evidence and are not reconstructed here.

## Surface families

### Proof attestation and contracts

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_attestation.py` | receipt-bound ZKP attestation contracts | structural_attestation_boundary |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py` | `ProofReceipt` and canonical contracts | integrity_envelope_unsigned |
| `ipfs_accelerate_py/agent_supervisor/proof/test_execution_contracts.py` | `TestPassReceipt` | integrity_envelope_unsigned |
| `ipfs_accelerate_py/testing/proof_reuse/receipt.py` | `TestPassReceiptCollector` | integrity_capture |

### Real-Groth16 fixture and backend binding

| Path | Role | Classification |
| --- | --- | --- |
| `test/api/proof_reuse_real_groth16_fixture.py` | disposable real Groth16 fixture | test_only_real_backend_fixture |
| `ipfs_accelerate_py/agent_supervisor/proof/provekit_setup.py` | ProveKit identity and self-test gate | real_backend_candidate_closed |
| `ipfs_accelerate_py/agent_supervisor/proof/ipfs_datasets_zk_attestation.py` | datasets Groth16/ProveKit binding | real_backend_candidate_closed |
| `ipfs_accelerate_py/agent_supervisor/proof/program_analysis_zkp.py` | program contract trace ZK surface | bounded_trace_zk_candidate |

### Kernel, conformance, fallbacks, metrics, evidence store

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py` | independent kernel reconstruction | closed_kernel_adapter |
| `ipfs_accelerate_py/agent_supervisor/proof/prover_conformance.py` | semantic conformance and quarantine | structural_conformance_gate |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_fallbacks.py` | bounded unsuccessful routing | diagnostic_fallback_not_assurance |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_metrics.py` | metrics and benchmark projection | observability_projection |
| `ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py` | portfolio evidence persistence | integrity_cache_not_aggregation |

### Manual completion seal and release evidence

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/control/manual_completion_seal.py` | operator seal check | external_operator_seal |
| `ipfs_accelerate_py/agent_supervisor/runtime/release_evidence.py` | release-domain evidence export/check | integrity_and_binding_export |

### Repository forest and canonicalization

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/analysis/repository_forest.py` | multi-checkout portable identity | integrity_identity_not_proof_seal |
| `ipfs_accelerate_py/agent_supervisor/analysis/repository_forest_manifest.py` | forest manifest binding | integrity_manifest |
| `ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py` | CID/multiformats identity | integrity_commitment |
| `ipfs_accelerate_py/agent_supervisor/analysis/content_identity_bridge.py` | content-identity bridge | integrity_commitment |

### Schedulers and resource admission

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py` | dependency-aware proof-plan execution | execution_orchestration |
| `ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py` | host/provider resource admission | resource_admission |
| `ipfs_accelerate_py/agent_supervisor/proof/multi_prover_resources.py` | multi-prover resource binding | resource_binding |
| `ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py` | multi-prover route selection | routing_structural |

### Doctor / MCP / formal caches

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py` | formal receipt cache | integrity_cache_not_aggregation |
| `ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py` | doctor federated cache gate | integrity_cache_not_aggregation |
| `ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_proof_cache.py` | MCP contract cache adapter | integrity_cache_not_aggregation |
| `ipfs_accelerate_py/agent_supervisor/proof/test_proof_cache.py` | test proof cache | integrity_cache_not_aggregation |
| `ipfs_accelerate_py/agent_supervisor/proof/test_certificate_store.py` | test certificate persistence | integrity_store |

### Runtime activation and v4 publication

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/testing/proof_reuse/plugin.py` | pytest proof-reuse plugin | runtime_activation_surface |
| `ipfs_accelerate_py/testing/proof_reuse/activation_contracts.py` | activation and v4 contracts | structural_runtime_contracts |
| `ipfs_accelerate_py/testing/proof_reuse/candidate_publication.py` | v4 candidate publication | publication_path_structural |
| `ipfs_accelerate_py/testing/proof_reuse/item_identity.py` | item identity / forest binding | identity_integrity |
| `ipfs_accelerate_py/testing/proof_reuse/lookup.py` | warm lookup | cache_lookup_integrity |
| `ipfs_accelerate_py/testing/proof_reuse/runtime_revalidation.py` | runtime revalidation | revalidation_integrity |
| `ipfs_accelerate_py/testing/proof_reuse/services.py` | service composition | service_composition |
| `ipfs_accelerate_py/testing/proof_reuse/xdist.py` | xdist publication authority | distributed_publication_boundary |

### Provider, policy, readiness

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py` | provider boundary | provider_boundary |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_capabilities.py` | capability health probe | capability_probe |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_policy.py` | outcome and rollout policy | policy_structural |
| `ipfs_accelerate_py/agent_supervisor/proof/solver_readiness.py` | readiness identifiers | readiness_probe |

### Focused test surfaces (registry-selected)

Static inventory only; suite outcomes are owned by the operator receipt above.

| Path | Role | Classification |
| --- | --- | --- |
| `test/api/test_agent_supervisor_proof_scheduler.py` | proof scheduler tests | focused_test_surface |
| `test/api/test_agent_supervisor_proof_resource_scheduler.py` | resource scheduler tests | focused_test_surface |
| `test/api/test_agent_supervisor_provekit_setup.py` | ProveKit setup tests | focused_test_surface |
| `test/api/test_agent_supervisor_code_proof_attestation_policy.py` | attestation policy tests | focused_test_surface |
| `test/api/test_proof_reuse_v4_publication_integration.py` | v4 publication integration | focused_test_surface |
| `test/api/test_proof_reuse_runtime_activation_e2e.py` | runtime activation e2e | focused_test_surface |
| `test/api/test_proof_reuse_cross_repository_e2e.py` | cross-repository e2e | focused_test_surface |
| `test/api/test_proof_reuse_accelerator_bootstrap.py` | accelerator bootstrap | focused_test_surface |

## Ownership proposal

Accelerate remains the execution authority for adapter discovery, backend
probing, cache admission, scheduling, cancellation, and seal orchestration.
Proposed package:
`ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing`.

Datasets remains semantic authority for proof units and manifests. Kit remains
storage authority for immutable seal objects and WAL durability. Accelerate
must not invent a competing CID, manifest, Merkle, or storage authority.
