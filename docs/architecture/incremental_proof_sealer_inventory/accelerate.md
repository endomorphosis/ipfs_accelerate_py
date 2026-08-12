# IncrementalProofSealer accelerate inventory (IPS-001)

Static source inventory of accelerate proof backends, attestation stores,
schedulers, receipts, caches, seals, CID/forest surfaces, focused tests, and
baseline-reference pins at the task-start parent. This document is a companion
to `docs/architecture/incremental_proof_sealer_inventory/accelerate.json`.

## Revisions

| Field | Value |
| --- | --- |
| `planning_revision` | `8881344bb2162f3f8d82f22d8348bc0ac7536f95` |
| `inventory_worktree_parent_revision` | `cea8ee497190c4373ff10eaa9088b2948c863293` |

`inventory_worktree_parent_revision` is immutable and equals the accelerate
task-start parent at candidate validation. The final task commit is supplied by
the supervisor completion receipt and is not self-embedded here.

## Baseline evidence (reference only)

Operator-captured process observation only. This inventory does not restate
argv, outcome tallies, logs, or operator capture claims.

| Field | Value |
| --- | --- |
| path | `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/accelerate.json` |
| receipt_digest | `sha256:bec760fcc447ba525a7e7b15e44d670eb224e637c73e6bc39854e7237fdfcde5` |
| required_command_ids | `accelerate-proof-focused-core-15`, `accelerate-proof-focused-wide-36`, `accelerate-proof-reuse-migration`, `accelerate-proof-reuse-cross-repo` |
| evidence_origin | `operator_capture` |
| assurance | `process_observed_only` |
| nonclaim | `pytest_execution_not_cryptographically_proven` |

The protected closed suite registry and validator independently recompute
preimages, argv, controlled-offline environment, digests, log sizes, tallies,
and incomplete-collection evidence nodes. Providers only reference the pin above.

## Inspection method

- classification_method: static source inventory
- Static scans report `surfaces_found` only; they never assert operator outcomes
- Static inspection is not a substitute for operator capture and is not
  cryptographic proof
- Controlled-offline capture disables Groth16/ProveKit enablement, builds,
  downloads, and auto-install; the receipt reference does not establish new
  real proving

## Explicit nonclaims

1. Unsigned `TestPassReceipt` and `ProofReceipt` values are integrity/assertion
   envelopes, **not** signed receipts.
2. Cache admission and locator-index hits are **not** receipt-aggregation proofs.
3. Direct-execution proving, recursion, and production setup claims require
   executable evidence from a real backend path; module presence alone does not
   establish those claims.
4. Simulated certificate backends are forced to `non_attested` authority and
   cannot authorize production `SKIP`.
5. No reliable recursive verifier surface is present in accelerate; default
   aggregation remains Merkle manifest completeness commitment.
6. `kernel_verification` rejects provider-claimed kernel status and re-derives
   trust from reconstruction digests and kernel acceptance checks.
7. `provekit_setup` and `ipfs_datasets_zk_attestation` never install tools or
   claim arbitrary source-code correctness from a ZK receipt alone.

## Surface families

### Proof attestation backends and stores

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_attestation.py` | attestation contracts | structural_attestation_boundary |
| `ipfs_accelerate_py/agent_supervisor/proof/ipfs_datasets_zk_attestation.py` | datasets ZK bridge | real_backend_candidate_fail_closed |
| `ipfs_accelerate_py/agent_supervisor/proof/program_analysis_zkp.py` | program-analysis ZK contracts | structural_with_real_backend_targets |
| `ipfs_accelerate_py/agent_supervisor/proof/provekit_setup.py` | ProveKit identity/self-test gate | real_backend_candidate_fail_closed |

### Receipts (unsigned integrity envelopes)

| Path | Role | Classification |
| --- | --- | --- |
| `formal_verification_contracts.py#ProofReceipt` | formal proof receipt | unsigned_integrity_envelope |
| `test_execution_contracts.py#TestPassReceipt` | three-phase test receipt | unsigned_integrity_envelope |
| `test_execution_contracts.py#TestProofCertificate` | certificate with authority enum | certificate_with_authority_enum |
| `testing/proof_reuse/receipt.py` | TestPassReceiptCollector | unsigned_receipt_capture |

### Cache admission (not receipt aggregation)

| Path | Role | Classification |
| --- | --- | --- |
| `proof/test_proof_cache.py` | TestProofCache admission | integrity_cache_admission_not_aggregation |
| `proof/test_certificate_store.py` | certificate CAS | integrity_cas_transport |
| `proof/formal_verification_cache.py` | formal receipt cache | integrity_cache_admission_not_aggregation |
| `proof/doctor_proof_cache.py` | doctor federated gate | integrity_cache_federation |
| `proof/mcp_contract_proof_cache.py` | MCP contract cache adapter | integrity_cache_adapter |

### Real-Groth16 fixture and runtime / v4 publication

| Path | Role | Classification |
| --- | --- | --- |
| `test/api/proof_reuse_real_groth16_fixture.py` | real Groth16 fixture (`proof_reuse_real_groth16_fixture`) | test_only_real_backend_fixture |
| `testing/proof_reuse/publication.py` | controller cold/v4 publication | real_backend_gated_publication |
| `testing/proof_reuse/candidate_publication.py` | candidate publication context | structural_publication_context |
| `testing/proof_reuse/activation_contracts.py` | runtime activation contracts | structural_runtime_contracts |
| `testing/proof_reuse/plugin.py` | pytest plugin | runtime_plugin_orchestration |
| `testing/proof_reuse/runtime_revalidation.py` | revalidation pins | structural_revalidation |
| `testing/proof_reuse/lookup.py` | two-stage lookup | integrity_lookup_not_aggregation |
| `testing/proof_reuse/services.py` | default services | service_composition |
| `testing/proof_reuse/xdist.py` | xdist controller | controller_publication_authority |

### Kernel / prover / fallback paths

| Path | Role | Classification |
| --- | --- | --- |
| `proof/kernel_verification.py` | independent kernel reconstruction | independent_kernel_reconstruction |
| `proof/prover_conformance.py` | semantic conformance gates | semantic_conformance_gate |
| `proof/proof_fallbacks.py` | bounded fallback routing | bounded_fallback_routing |
| `proof/formal_verification_provider.py` | provider subprocess + cancellation | fail_closed_provider_execution |
| `proof/multi_prover_router.py` | portfolio router | orchestration_not_proof |
| `proof/multi_prover_resources.py` | portfolio resource budgets | resource_admission |
| `proof/prover_matrix_registry.py` | prover matrix | capability_registry |

### Metrics, benchmarks, and evidence stores

| Path | Role | Classification |
| --- | --- | --- |
| `proof/proof_metrics.py` | metric/benchmark projection | observability_projection |
| `proof/supervisor_code_proof_benchmark.py` | code-proof benchmark | benchmark_measurement |
| `self_improvement/proof_reuse_benchmark.py` | proof-reuse benchmark | benchmark_measurement |
| `proof/prover_evidence_store.py` | portfolio evidence store | durable_portfolio_evidence |
| `proof/database_evidence_store.py` | DuckDB evidence authority | durable_validation_proof_store |

### Manual completion and release seals

| Path | Role | Classification |
| --- | --- | --- |
| `control/manual_completion_seal.py` | operator manual completion seal | operator_seal_integrity |
| `runtime/release_evidence.py` | release evidence export/replay | release_evidence_export_verify |

### CID / canonicalization / repository forest

| Path | Role | Classification |
| --- | --- | --- |
| `analysis/repository_forest.py` | repository_forest authority | canonical_repository_forest |
| `analysis/repository_forest_manifest.py` | forest manifest | forest_manifest_binding |
| `core/multiformats_identity.py` | CIDv1 profile bridge | canonical_cid_profile |
| `utils/cid_utils.py` | CID helpers | canonical_cid_helpers |
| `formal_verification_contracts.py#content_identity` | content identity | canonical_content_identity |

### Schedulers and resource admission

| Path | Role | Classification |
| --- | --- | --- |
| `proof/proof_scheduler.py` | dependency-DAG proof scheduler | dependency_dag_scheduler |
| `runtime/resource_scheduler.py` | host/provider resource admission | resource_admission |
| `proof/formal_verification_policy.py` | rollout/policy | policy_rollout |
| `proof/formal_verification_capabilities.py` | capability probes | capability_inventory |
| `proof/admissibility_enforcement.py` | admissibility gate | admission_gate |
| `proof/admissibility_bridge.py` | admissibility bridge | admission_bridge |

### Test-execution identity

| Path | Role | Classification |
| --- | --- | --- |
| `proof/test_execution_contracts.py` | reuse contracts module | typed_reuse_contracts |
| `analysis/test_execution_identity.py` | identity assembly | identity_assembly |
| `testing/proof_reuse/item_identity.py` | per-item identity | item_identity_assembly |
| `proof/test_candidate_context_store.py` | candidate context store | integrity_context_store |

## Focused tests (repository-relative)

### accelerate-proof-focused-core-15 paths

- `test/api/test_agent_supervisor_provekit_setup.py`
- `test/api/test_agent_supervisor_ipfs_datasets_zk_attestation.py`
- `test/api/test_agent_supervisor_program_analysis_zkp.py`
- `test/api/test_agent_supervisor_program_analysis_zkp_conformance.py`
- `test/api/test_agent_supervisor_proof_scheduler.py`
- `test/api/test_agent_supervisor_proof_resource_scheduler.py`
- `test/api/test_agent_supervisor_adaptive_resources.py`
- `test/api/test_agent_supervisor_multi_prover_resources.py`
- `test/api/test_agent_supervisor_multi_prover_router.py`
- `test/api/test_agent_supervisor_formal_verification_contracts.py`
- `test/api/test_agent_supervisor_formal_verification_cache.py`
- `test/api/test_agent_supervisor_formal_verification_capabilities.py`
- `test/api/test_agent_supervisor_formal_verification_provider.py`
- `test/api/test_agent_supervisor_formal_verification_policy.py`
- `test/api/test_agent_supervisor_code_proof_attestation_policy.py`

### accelerate-proof-focused-wide-36 additional paths

- `test/api/test_agent_supervisor_test_execution_identity.py`
- `test/api/test_agent_supervisor_test_execution_identity_vectors.py`
- `test/api/test_agent_supervisor_test_proof_reuse_doctrine.py`
- `test/api/test_proof_reuse_activation_contracts.py`
- `test/api/test_proof_reuse_receipt.py`
- `test/api/test_proof_reuse_runtime_activation_report.py`
- `test/api/test_proof_reuse_controller_issuance.py`
- `test/api/test_proof_reuse_candidate_publication_context.py`
- `test/api/test_proof_reuse_locator_first_collection.py`
- `test/api/test_proof_reuse_two_stage_warm_lookup.py`
- `test/api/test_proof_reuse_runtime_revalidation.py`
- `test/api/test_proof_reuse_degradation_matrix.py`
- `test/api/test_proof_reuse_cold_pass_publication.py`
- `test/api/test_proof_reuse_default_identity_services.py`
- `test/api/test_proof_reuse_default_runtime_services.py`
- `test/api/test_proof_reuse_security_concurrency.py`
- `test/api/test_proof_reuse_invalidation_mutations.py`
- `test/api/test_proof_reuse_issued_material_retention.py`
- `test/api/test_proof_reuse_setup_provisioning.py`
- `test/api/test_proof_reuse_service_injection.py`
- `test/api/test_proof_reuse_lazy_provisioning.py`

### accelerate-proof-reuse-migration paths

- `test/api/test_proof_reuse_v4_publication_integration.py`
- `test/api/test_proof_reuse_runtime_activation_e2e.py`
- `test/api/test_proof_reuse_runtime_composition.py`
- `test/api/test_pytest_proof_reuse_item_identity.py`
- `test/api/test_pytest_proof_reuse_lookup.py`
- `test/api/test_pytest_proof_reuse_plugin.py`
- `test/api/test_pytest_proof_reuse_receipt.py`
- `test/api/test_pytest_proof_reuse_xdist.py`

### accelerate-proof-reuse-cross-repo paths

- `test/api/test_proof_reuse_cross_repository_e2e.py`
- `test/api/test_proof_reuse_accelerator_bootstrap.py`

### Fixture helper (not a suite id)

- `test/api/proof_reuse_real_groth16_fixture.py` — real-Groth16 disposable fixture helpers

## Ownership candidate

Accelerate owns execution authority: adapter discovery, backend probing, cache
admission, scheduling/resource admission/cancellation, measurement, and sealing
orchestration. Proposed package:
`ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing`.

Datasets remains semantic authority for proof units, manifests, and identity.
Kit remains storage authority for CAS, WAL, and forest persistence.
