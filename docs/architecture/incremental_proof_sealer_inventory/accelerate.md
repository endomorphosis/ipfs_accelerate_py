# IPS-001 Accelerate proof inventory

Static source-code inventory of accelerate proof backends, receipts, schedulers,
caches, seals, CID/Merkle surfaces, and focused test modules. This document
reports **surfaces_found** classifications only. It does not restate operator
pytest outcomes and is not a cryptographic proof that pytest ran.

## Provenance

| Field | Value |
| --- | --- |
| Task | IPS-001 |
| Repository | accelerate |
| Planning revision | `8881344bb2162f3f8d82f22d8348bc0ac7536f95` |
| Inventory worktree parent revision | `006afc0e7c6169eda50d9daf37819134b42aa1e9` |
| Inspection mode | source_code_read |
| Inspection reports | surfaces_found |

The final task commit is supplied by the supervisor completion receipt and is
not self-embedded in this inventory.

## Baseline evidence (reference only)

Operator-captured process observation only. This inventory does not copy
commands, counts, logs, or transcripts.

| Field | Value |
| --- | --- |
| path | `artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts/accelerate.json` |
| receipt_digest | `sha256:bec760fcc447ba525a7e7b15e44d670eb224e637c73e6bc39854e7237fdfcde5` |
| required_command_ids | `accelerate-proof-focused-core-15`, `accelerate-proof-focused-wide-36`, `accelerate-proof-reuse-migration`, `accelerate-proof-reuse-cross-repo` |
| evidence_origin | `operator_capture` |
| assurance | `process_observed_only` |
| nonclaim | `pytest_execution_not_cryptographically_proven` |

## Nonclaims

- `TestPassReceipt` is **not signed**; it is an integrity and admission envelope.
- `ProofReceipt` is **not signed**; assurance is a projection of typed evidence.
- Cache admission (formal verification cache, test proof cache, doctor/MCP
  federated caches, prover evidence store, reuse lookup) is **not** receipt
  aggregation.
- Capability probes and setup identity gates do not prove pytest execution.
- Simulated attestation backends are non-authoritative.
- No operational recursive verifier was found in accelerate executable code.
- P2P merkle clock hash commitments are not ZK proof-seal authorities.
- Historical planning counts are not reconstructed here.

## Proof contracts and receipts

| Surface | Path | Class |
| --- | --- | --- |
| ProofReceipt contracts | `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py` | integrity_only (unsigned) |
| TestPassReceipt contracts | `ipfs_accelerate_py/agent_supervisor/proof/test_execution_contracts.py` | integrity_only (unsigned) |
| proof_attestation | `ipfs_accelerate_py/agent_supervisor/proof/proof_attestation.py` | attestation_binding; simulated backends non-authoritative |
| test certificate store | `ipfs_accelerate_py/agent_supervisor/proof/test_certificate_store.py` | integrity_only CAS |
| proof reuse receipt collection | `ipfs_accelerate_py/testing/proof_reuse/receipt.py` | integrity_only |

## Real proving, setup, and publication paths

Direct-execution and setup claims are tied only to executable evidence:

| Surface | Path | Class |
| --- | --- | --- |
| ProveKit setup identity gate | `ipfs_accelerate_py/agent_supervisor/proof/provekit_setup.py` | real_proving_path_when_artifacts_present; typed unavailable without install |
| datasets Groth16/ProveKit binding | `ipfs_accelerate_py/agent_supervisor/proof/ipfs_datasets_zk_attestation.py` | attestation_binding on already kernel-checked receipts |
| program analysis ZKP | `ipfs_accelerate_py/agent_supervisor/proof/program_analysis_zkp.py` | declared_computation_only |
| v4 publication | `ipfs_accelerate_py/testing/proof_reuse/publication.py` | real_proving_path_when_artifacts_present |
| real Groth16 fixture | `test/api/proof_reuse_real_groth16_fixture.py` | real_proving_fixture |

## Kernel, prover, fallback, metrics, evidence

| Surface | Path | Class |
| --- | --- | --- |
| kernel_verification.py | `ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py` | structural_and_kernel_checked |
| prover_conformance.py | `ipfs_accelerate_py/agent_supervisor/proof/prover_conformance.py` | structural_validation |
| proof_fallbacks.py | `ipfs_accelerate_py/agent_supervisor/proof/proof_fallbacks.py` | structural_validation |
| proof_metrics.py | `ipfs_accelerate_py/agent_supervisor/proof/proof_metrics.py` | measurement_only |
| prover_evidence_store.py | `ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py` | cache_admission |
| database_evidence_store.py | `ipfs_accelerate_py/agent_supervisor/proof/database_evidence_store.py` | integrity_only |
| multi_prover_router.py | `ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py` | scheduling_orchestration |
| multi_prover_resources.py | `ipfs_accelerate_py/agent_supervisor/proof/multi_prover_resources.py` | scheduling_orchestration |
| formal_verification_provider.py | `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py` | untrusted_provider_boundary |
| formal_verification_capabilities.py | `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_capabilities.py` | capability_discovery_only |
| formal_verification_policy.py | `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_policy.py` | structural_validation |
| solver_readiness.py | `ipfs_accelerate_py/agent_supervisor/proof/solver_readiness.py` | capability_discovery_only |
| prover_matrix_registry.py | `ipfs_accelerate_py/agent_supervisor/proof/prover_matrix_registry.py` | structural_validation |

## Caches (admission, not aggregation)

| Surface | Path |
| --- | --- |
| formal_verification_cache | `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py` |
| test_proof_cache | `ipfs_accelerate_py/agent_supervisor/proof/test_proof_cache.py` |
| doctor_proof_cache | `ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py` |
| mcp_contract_proof_cache | `ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_proof_cache.py` |
| proof reuse lookup | `ipfs_accelerate_py/testing/proof_reuse/lookup.py` |

## Schedulers and runtime activation

| Surface | Path | Class |
| --- | --- | --- |
| proof_scheduler | `ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py` | scheduling_orchestration |
| resource_scheduler | `ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py` | scheduling_orchestration |
| configured_board_scheduler | `ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py` | scheduling_orchestration |
| validation_scheduler | `ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py` | scheduling_orchestration |
| proof reuse plugin | `ipfs_accelerate_py/testing/proof_reuse/plugin.py` | runtime_activation |
| activation contracts | `ipfs_accelerate_py/testing/proof_reuse/activation_contracts.py` | structural_validation |
| runtime revalidation | `ipfs_accelerate_py/testing/proof_reuse/runtime_revalidation.py` | structural_validation |
| item identity | `ipfs_accelerate_py/testing/proof_reuse/item_identity.py` | structural_validation |
| xdist publication authority | `ipfs_accelerate_py/testing/proof_reuse/xdist.py` | runtime_activation |
| candidate publication | `ipfs_accelerate_py/testing/proof_reuse/candidate_publication.py` | structural_validation |
| default identity services | `ipfs_accelerate_py/testing/proof_reuse/default_identity_services.py` | structural_validation |
| runtime services | `ipfs_accelerate_py/testing/proof_reuse/services.py` | runtime_activation |

## Manual seal, release evidence, CID, forest, Merkle

| Surface | Path | Class |
| --- | --- | --- |
| manual_completion_seal.py | `ipfs_accelerate_py/agent_supervisor/control/manual_completion_seal.py` | signed_assertion (HMAC operator seal) |
| release_evidence.py | `ipfs_accelerate_py/agent_supervisor/runtime/release_evidence.py` | integrity_only |
| repository_forest | `ipfs_accelerate_py/agent_supervisor/analysis/repository_forest.py` | content_addressing |
| multiformats_identity | `ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py` | content_addressing |
| cid_utils | `ipfs_accelerate_py/utils/cid_utils.py` | content_addressing |
| P2P merkle clock | `ipfs_accelerate_py/p2p_workflow_scheduler.py` | integrity_only (not ZK seal authority) |

## Metrics and benchmarks

| Surface | Path | Class |
| --- | --- | --- |
| proof_metrics | `ipfs_accelerate_py/agent_supervisor/proof/proof_metrics.py` | measurement_only |
| supervisor_code_proof_benchmark | `ipfs_accelerate_py/agent_supervisor/proof/supervisor_code_proof_benchmark.py` | measurement_only |
| proof_reuse_benchmark | `ipfs_accelerate_py/agent_supervisor/self_improvement/proof_reuse_benchmark.py` | measurement_only |

## Focused tests (classification by surface behavior)

Focused modules listed by the closed suite registry command IDs
`accelerate-proof-focused-core-15`, `accelerate-proof-focused-wide-36`,
`accelerate-proof-reuse-migration`, and `accelerate-proof-reuse-cross-repo`
are each classified under `surfaces_found` in the companion JSON by repository-
relative path. Coverage includes ProveKit setup, ZK attestation, program-analysis
ZKP, proof/resource/multi-prover schedulers, formal verification contracts/cache/
capabilities/provider/policy, code-proof attestation policy, test-execution
identity, proof-reuse activation/receipt/publication/lookup/plugin/xdist/
controller/provisioning, and cross-repository bootstrap. This Markdown reports
only `surfaces_found` classifications and does not restate operator observations.

## Machine-readable companion

See `docs/architecture/incremental_proof_sealer_inventory/accelerate.json` for
the complete `surfaces_found` list, structured `classifications`, and the
exact reference-only `baseline_evidence` projection.
