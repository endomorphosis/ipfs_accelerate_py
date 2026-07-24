# ASI-078 completion-gate evidence map

- Date: 2026-07-24
- Task: ASI-078
- Goal: ASI-G103
- Parent: ASI-G070
- Requirement: `031486194157679117987393491870400400279`
- Source gap fingerprint: `dd2576a635bc38ba7542efcc5bc66e65585cf300`
- Evidence obligation: `objective-work/v1/65258c680ddf290ac7a8af37c6cf92c2284f41ac`
- Todo vector: `65258c680ddf290a`
- Merge role: `completion_gate`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-078 covers the missing completion-evidence terms without allowing this
task, document, or an operational parity witness to authorize completion:

1. `ControlSurfaceParityEvidence` derives the complete sorted `Operation`
   vocabulary, the generic request/result schema identities, every
   operation-specific request/result schema identity, and one
   `schema_population_id` over that exact population. Deserialization
   independently re-derives every identity and rejects an omitted, partial,
   reordered, or forged population. The Python discovery manifest and MCP
   registration resolve to the same transport-independent population.
2. The six mandatory criteria form one closed population shared literally by
   this heap and `CONTROL_SURFACE_PARITY_ACCEPTANCE_CRITERIA`.
   `ControlSurfaceParityEvidence.evaluate_objective_completion` requires one
   submitted validation per exact criterion and rejects missing, duplicate,
   extra, failed, stale, foreign-tree, wrong-policy, wrong-objective, or
   operationally detached records.
3. `GoalCoverageMap` projection requires every exact criterion row to name a
   concrete implementation surface and bind that criterion's submitted
   validation receipt on the operational tree. A verified summary cannot
   replace the closed row population or repair a missing implementation,
   detached validation, stale receipt, or foreign-tree binding.
4. Analyzer status is never inferred. The gate requires explicit `healthy`
   and `safe_for_completion_reasoning` values plus the exact G103 analyzer,
   objective, and current-tree binding.
5. `ControlSurfaceParityCompletionMemberHealth` binds an explicit healthy and
   completion-safe attestation to every independently named exhaustive member
   and receipt CID. `ControlSurfaceParityCompletionQuorumEvidence` requires
   exact population equality and binds the configured quorum to the G103
   requirement, operational receipt, validation policy, tree, analyzer,
   configuration, and ASI-078 objective revision. Under-count, duplicate
   member/receipt/channel, unhealthy, unsafe, non-exhaustive, stale,
   wrong-revision, foreign-policy, foreign-receipt, or foreign-tree evidence
   fails closed.
6. A complete passing evaluation from `active` advances only to
   `provisionally_complete`. Verified completion requires a separate later
   evaluation with the full proof population still valid. ASI-G103 and its
   ASI-G070 parent therefore remain actionable for supervisor ingestion of
   the final post-change receipts.

## Mandatory criterion map

| Criterion | Implementation | Validation |
| --- | --- | --- |
| The shared schema describes all operations | `Operation`, `operation_request_json_schema`, `operation_result_json_schema`, `ControlDiscoveryManifest`, and the derived `ControlSurfaceParityEvidence.request_schema_ids`, `result_schema_ids`, and `schema_population_id` | `test_shared_wire_schemas_cover_every_operation_and_mutation_guard`, `test_typed_surface_parity_evidence_proves_exact_requirement`, `test_surface_parity_evidence_rejects_behavior_or_schema_drift`, `test_registration_covers_every_operation_with_shared_schema` |
| every CLI/MCP adapter decodes and dispatches the canonical request directly | authoritative `decode_operation_request`, direct `run_agent_cli` dispatch, direct `execute_agent_supervisor_operation` dispatch, and `SupervisorControlService.execute` | `test_cli_read_result_is_exactly_the_python_service_record`, `test_mcp_result_is_exactly_the_python_service_record`, `test_hierarchical_dispatch_uses_direct_control_service`, `test_python_cli_mcp_matrix_emits_typed_parity_evidence` |
| canonical records are exactly equal to Python behavior | `ControlSurfaceParityCase` re-decodes and request-validates the full Python/CLI/MCP `OperationResult` records before exact equality | `test_python_cli_mcp_matrix_emits_typed_parity_evidence`, `test_typed_surface_parity_evidence_proves_exact_requirement`, `test_surface_parity_evidence_rejects_behavior_or_schema_drift` |
| bounded reads and watches cannot exceed contract limits | `ControlBounds`, bounded `OperationResult` construction, direct repository read adapters, and the bounded CLI watch loop | `test_read_client_uses_direct_repository_apis_and_bounded_results`, `test_allowlists_bounds_and_paths_fail_with_stable_errors`, `test_cli_watch_is_bounded_and_emits_one_canonical_record_per_line`, `test_cli_rejects_unbounded_watch_before_dispatch` |
| unsafe CLI defaults and unconfigured MCP mutation authority fail closed | CLI complete-request boundary and MCP decode-before-resolution configuration/allowlist boundary | `test_cli_rejects_ambiguous_roots_and_non_dry_run_mutation`, every CLI/MCP unsafe mutation parameter case, `test_unconfigured_mcp_adapter_fails_closed_without_request_roots` |
| and the exact requirement ID appears only in a tree/objective/policy-bound parity evidence record that rejects any surface, vocabulary, schema, or behavior drift. | `ControlSurfaceParityEvidence`, `ControlSurfaceParityCompletionMemberHealth`, `ControlSurfaceParityCompletionQuorumEvidence`, and the two-phase completion bridge | `test_typed_surface_parity_evidence_proves_exact_requirement`, `test_surface_parity_evidence_rejects_behavior_or_schema_drift`, `test_g103_completion_requires_bound_current_tree_validation_health_and_quorum` |

The completion matrix uses canonical `GoalCoverageMap`,
`AnalyzerHealthReport`, and `ExhaustionQuorumResult` producers wrapped by the
G103-specific member-health and quorum evidence contracts. It submits a
complete positive population and incomplete tasks plus failed, stale,
missing, duplicate, detached, incompletely mapped, implementation-free,
unsafe, analyzer-mismatched, under-count, duplicate-member,
duplicate-receipt, duplicate-channel, unhealthy, non-exhaustive, stale-quorum,
wrong-revision, foreign-policy, foreign-receipt, and foreign-tree variants.

## Validation observation

- Command: `python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q`
- Observed result during ASI-078 implementation: 56 passed

This discovery record is an audit index, not a completion receipt. It claims
neither the final post-change repository tree identity nor the independent
analyzer/quorum executions that can exist only after the change is finalized.
The supervisor must keep ASI-G103 and ASI-G070 actionable until a post-change
evaluation supplies fresh passing current-tree criterion receipts, explicit
healthy completion-safe analyzer evidence, the configured independent fresh
healthy exhaustive quorum, and a canonical completion-gate record.
Verification may occur only in a separate evaluation after provisional
completion.
