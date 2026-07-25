# ASI-076 completion-gate evidence map

- Date: 2026-07-24
- Task: ASI-076
- Goal: ASI-G105
- Parent: ASI-G070
- Requirement: `186773143401179107362964063059661378722`
- Source gap fingerprint: `8ac999078ba30bfa90ad80b22e82d9045357e91e`
- Evidence obligation: `objective-work/v1/9de0c7b66d9f7ecd7ae5af2eeefb440e657c8d77`
- Merge role: `completion_gate`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-076 covers the missing completion-evidence terms without allowing this
task, document, or an operational discovery record to authorize completion:

1. `ControlDiscoveryManifest.schema_population_id` binds the complete sorted
   operation vocabulary and the canonical per-operation request/result schema
   identities independently of transport. Every `ControlDiscoveryObservation`
   compares the canonical bytes returned by two calls on one surface.
   `ControlDiscoverySafetyEvidence` requires exactly one Python, CLI, and MCP
   observation and one identical transport-independent schema population.
2. The six mandatory criteria form one closed population shared literally by
   this heap and `CONTROL_DISCOVERY_SAFETY_ACCEPTANCE_CRITERIA`.
   `ControlDiscoverySafetyEvidence.evaluate_objective_completion` requires one
   submitted validation per exact criterion and rejects missing, duplicate,
   extra, failed, stale, foreign-tree, wrong-policy, wrong-objective, or
   operationally detached records.
3. `GoalCoverageMap` projection requires every exact criterion row to name a
   concrete implementation surface and bind that criterion's submitted
   validation receipt. A verified summary cannot replace the closed row
   population or repair a detached validation.
4. Analyzer status is never inferred. The gate requires explicit `healthy`
   and `safe_for_completion_reasoning` values plus the exact G105 analyzer,
   objective, and current-tree binding.
5. `ControlDiscoveryCompletionMemberHealth` binds an explicit healthy and
   completion-safe attestation to every independently named exhaustive member
   and receipt CID. `ControlDiscoveryCompletionQuorumEvidence` requires exact
   population equality and binds that quorum to the G105 requirement,
   operational receipt, validation policy, tree, analyzer, configuration, and
   ASI-076 objective revision. Missing, unhealthy, unsafe, duplicate,
   non-exhaustive, stale, or foreign members fail closed.
6. A complete passing evaluation from `active` advances only to
   `provisionally_complete`. Verified completion requires a separate later
   evaluation with the full proof population still valid. ASI-G105 and its
   ASI-G070 parent therefore remain actionable for supervisor ingestion of the
   final post-change receipts.

## Mandatory criterion map

| Criterion | Implementation | Validation |
| --- | --- | --- |
| Repeated Python, CLI, and MCP discovery is byte-deterministic and covers the same closed operation/schema population | `ControlDiscoveryManifest.schema_population_id`, canonical bytes in `ControlDiscoveryObservation`, `SupervisorControlService.discovery_manifest`, `agent_cli_discovery_manifest`, and `agent_supervisor_discovery_manifest` | `test_python_discovery_is_cached_deterministic_and_never_dispatches`, `test_cli_discovery_is_repeatable_and_initializes_no_runtime`, `test_discovery_safety_evidence_uses_observed_python_cli_and_mcp_runs`, `test_registration_covers_every_operation_with_shared_schema` |
| no backend or configured service factory is called | static manifest producers and the invocation-only Python/MCP dispatch paths | `test_python_discovery_is_cached_deterministic_and_never_dispatches`, `test_discovery_and_registration_do_not_resolve_a_service`, `test_discovery_safety_evidence_uses_observed_python_cli_and_mcp_runs` |
| optional supervisor provider imports and process starts remain independently observed at zero delta | `capture_control_discovery_runtime_state` module/process inventories and provider-load/process-start counters | `test_discovery_safety_evidence_uses_observed_python_cli_and_mcp_runs`, `test_cli_discovery_is_repeatable_and_initializes_no_runtime` |
| agent CLI discovery does not construct unrelated runtime state | lightweight `agent` parser registration and early unified-CLI dispatch in `ipfs_accelerate_py/cli.py` | `test_cli_discovery_is_repeatable_and_initializes_no_runtime`, `test_agent_group_covers_the_closed_operation_vocabulary` |
| only tool execution can increment MCP service resolution | invocation-only `_resolve_service` and `agent_supervisor_service_resolution_count` | `test_discovery_and_registration_do_not_resolve_a_service`, `test_discovery_safety_evidence_uses_observed_python_cli_and_mcp_runs`, `test_mcp_result_is_exactly_the_python_service_record` |
| and only the complete current-tree three-surface evidence emits the exact requirement ID. | `ControlDiscoverySafetyEvidence`, `ControlDiscoveryCompletionMemberHealth`, `ControlDiscoveryCompletionQuorumEvidence`, and the two-phase objective bridge | `test_discovery_safety_evidence_is_complete_content_addressed_and_strict`, `test_g105_completion_requires_bound_current_tree_validation_health_and_quorum` |

The completion matrix uses canonical `GoalCoverageMap`,
`AnalyzerHealthReport`, and `ExhaustionQuorumResult` producers wrapped by the
G105-specific member-health and quorum evidence contracts. It also submits
failed, stale, missing, detached, incompletely mapped, unsafe, unhealthy,
under-count, duplicate-channel, foreign-policy, foreign-receipt, and
foreign-tree variants.

## Validation observation

- Command: `python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q`
- Observed result during ASI-076 implementation: 47 passed

This discovery record is an audit index, not a completion receipt. It claims
neither the final post-change repository tree identity nor the independent
analyzer/quorum executions that can exist only after the change is finalized.
The supervisor must keep ASI-G105 and ASI-G070 actionable until a post-change
evaluation supplies fresh passing current-tree criterion receipts, explicit
healthy completion-safe analyzer evidence, the configured independent fresh
healthy exhaustive quorum, and a canonical completion-gate record.
Verification may occur only in a separate evaluation after provisional
completion.
