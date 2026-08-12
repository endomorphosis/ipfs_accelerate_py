# IncrementalProofSealer accelerate inventory (IPS-001)

Static source inventory of accelerate proof backends, attestation surfaces,
schedulers, receipts, caches, seals, CID/forest paths, and baseline-reference
surfaces at the task-start parent. Companion to
`docs/architecture/incremental_proof_sealer_inventory/accelerate.json`.

## Revisions

| Field | Value |
| --- | --- |
| `planning_revision` | `8881344bb2162f3f8d82f22d8348bc0ac7536f95` |
| `inventory_worktree_parent_revision` | `633de55f11ff266481d7c590ac905fe90d9136f1` |

`inventory_worktree_parent_revision` is immutable and equals the accelerate
task-start parent for this candidate. The final task commit is supplied by the
supervisor completion receipt and is not self-embedded here.

## Baseline evidence (reference only)

Operator-captured process observation only. This inventory does not restate
command lines, outcome tallies, logs, or execution claims.

| Field | Value |
| --- | --- |
| path | `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/accelerate.json` |
| receipt_digest | `sha256:2bf70cf540d04bac0d11fc58b8668bd924884ea89578cef28ec2e8c601efe8a9` |
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
- Controlled-offline capture disables network installs, key generation, and
  auto-provisioning; the receipt reference does not establish new real proving

## Explicit nonclaims

1. Unsigned `TestPassReceipt` and `ProofReceipt` values are **not** signed
   receipts; they are integrity/assertion envelopes.
2. Cache admission (formal verification, doctor, MCP, test-proof caches) is
   **not** receipt-aggregation proof of test execution.
3. Direct-execution and recursion claims require executable operational backend
   evidence; no reliable recursive verifier is classified as operational here.
4. Simulated backends and non-attested certificates cannot authorize production
   seals or production `SKIP`.
5. Presence of ProveKit/Groth16 integration surfaces is **not** evidence that
   production keys or direct-execution proving are operational.
6. Manual completion seals and release evidence are operator/content-addressed
   integrity envelopes, not ZK proofs of task execution.
7. Repository-forest and multiformats CID paths are identity/integrity
   commitments, not cryptographic test-execution proofs.

## Surface families

### Proof attestation and ZK bindings

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_attestation.py` | receipt-bound ZKP contracts | structural_unless_real_backend |
| `ipfs_accelerate_py/agent_supervisor/proof/ipfs_datasets_zk_attestation.py` | datasets Groth16/ProveKit binding | real_backend_candidate_fail_closed |
| `ipfs_accelerate_py/agent_supervisor/proof/provekit_setup.py` | ProveKit setup identity gate | real_backend_candidate_fail_closed |
| `ipfs_accelerate_py/agent_supervisor/proof/program_analysis_zkp.py` | program-analysis trace ZK contracts | structural_with_real_backend_targets |

### Kernel, conformance, fallbacks, metrics, evidence

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py` | independent kernel reconstruction | structural_integrity_with_live_reconstruction |
| `ipfs_accelerate_py/agent_supervisor/proof/prover_conformance.py` | semantic conformance gate | structural_integrity |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_fallbacks.py` | fallback routing | structural |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_metrics.py` | metrics projection | integrity_projection |
| `ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py` | multi-prover evidence store | integrity_cache |

### Schedulers and resources

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py` | proof-plan DAG scheduler | orchestration |
| `ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py` | resource admission | orchestration |
| `ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py` | multi-prover routing | orchestration |
| `ipfs_accelerate_py/agent_supervisor/proof/multi_prover_resources.py` | multi-prover resources | orchestration |
| `ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py` | board scheduler config | orchestration |
| `ipfs_accelerate_py/agent_supervisor/runtime/durable_process.py` | process-tree / cancellation helpers | orchestration |

### Caches (admission is not receipt aggregation)

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py` | formal verification cache | integrity_cache_not_receipt_aggregation |
| `ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py` | doctor cache federation | integrity_cache_not_receipt_aggregation |
| `ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_proof_cache.py` | MCP contract proof cache | integrity_cache_not_receipt_aggregation |
| `ipfs_accelerate_py/agent_supervisor/proof/test_proof_cache.py` | test proof-reuse cache | integrity_cache_not_receipt_aggregation |
| `ipfs_accelerate_py/agent_supervisor/proof/test_certificate_store.py` | test certificate store | integrity_store |

### Receipts and publication (unsigned envelopes; v4 publication)

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/test_execution_contracts.py` | `TestPassReceipt` / execution keys | unsigned_integrity_envelope |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py` | `ProofReceipt` contracts | unsigned_integrity_envelope |
| `ipfs_accelerate_py/testing/proof_reuse/receipt.py` | `TestPassReceiptCollector` | unsigned_integrity_envelope |
| `ipfs_accelerate_py/testing/proof_reuse/publication.py` | controller v4 publication | real_backend_candidate_fail_closed |
| `ipfs_accelerate_py/testing/proof_reuse/activation_contracts.py` | runtime activation contracts | structural_activation |
| `ipfs_accelerate_py/testing/proof_reuse/plugin.py` | pytest plugin | orchestration |
| `ipfs_accelerate_py/testing/proof_reuse/lookup.py` | candidate lookup | integrity_lookup |
| `ipfs_accelerate_py/testing/proof_reuse/runtime_revalidation.py` | current-context revalidation | integrity_comparison |

### Real-Groth16 fixture, manual/release seals, CID/forest

| Path | Role | Classification |
| --- | --- | --- |
| `test/api/proof_reuse_real_groth16_fixture.py` | real Groth16 v4 fixture helpers | real_backend_fixture_gap_tolerant |
| `ipfs_accelerate_py/agent_supervisor/control/manual_completion_seal.py` | manual completion seal | operator_integrity_envelope |
| `ipfs_accelerate_py/agent_supervisor/runtime/release_evidence.py` | release evidence export/verify | integrity_export |
| `ipfs_accelerate_py/agent_supervisor/analysis/repository_forest.py` | repository forest identity | integrity_commitment |
| `ipfs_accelerate_py/agent_supervisor/analysis/repository_forest_manifest.py` | forest manifest | integrity_commitment |
| `ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py` | multiformats CID bridge | integrity_commitment |
| `ipfs_accelerate_py/utils/cid_utils.py` | CID utilities | integrity_commitment |

### Backends and policy

| Path | Role | Classification |
| --- | --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py` | provider adapters | mixed_real_and_unavailable |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_capabilities.py` | capability probes | structural |
| `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_policy.py` | assurance policy | structural |
| `ipfs_accelerate_py/agent_supervisor/proof/prover_matrix_registry.py` | prover matrix | structural |
| `ipfs_accelerate_py/agent_supervisor/self_improvement/proof_reuse_benchmark.py` | proof-reuse benchmark | measurement_surface |

### Focused registry tests (static path classification only)

Core-15 / wide-36 / reuse-migration / cross-repo suite paths are classified as
`focused_test_path` with `surfaces_found` counts only. Individual paths include
ProveKit setup, datasets ZK attestation, program-analysis ZKP, proof and
resource schedulers, multi-prover router/resources, formal-verification
contracts/cache/capabilities/provider/policy, code proof attestation policy,
v4 publication integration, runtime activation e2e, cross-repository e2e,
accelerator bootstrap, and proof-reuse receipt/activation contracts.

### Operator-protected baseline controls

| Path | Role | Classification |
| --- | --- | --- |
| `config/agent_supervisor_incremental_proof_sealer_scheduler.json` | board scheduler pin namespace | operator_protected_config |
| `config/incremental_proof_sealer_baseline_suite_registry.json` | closed suite registry | operator_protected_config |

## Ownership note

Accelerate owns proving/planning/scheduling/aggregation/sealing/measurement
orchestration. Datasets remains semantic/identity/manifest authority; kit remains
storage/WAL/CAS/forest authority. Proposed package:
`ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing`.
