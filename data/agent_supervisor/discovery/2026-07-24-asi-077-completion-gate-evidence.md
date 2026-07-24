# ASI-077 completion-gate evidence map

- Date: 2026-07-24
- Task: ASI-077
- Goal: ASI-G104
- Parent: ASI-G070
- Requirement: `184125100306462690646212311073240043804`
- Source gap fingerprint: `d94d0ddd7f0204fc0b866864769a717297cae340`
- Evidence obligation: `objective-work/v1/bcdcc08d98c2bee4c0f4829ce757072367049df5`
- Merge role: `completion_gate`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-077 covers the missing completion-evidence terms without allowing this
task, document, or an operational mutation witness to authorize completion:

1. `CONTROL_MUTATION_GUARD_REJECTION_SCENARIOS` fixes the six unsafe mutation
   classes. `ControlMutationGuardEvidence` requires the exact cross product of
   those classes with Python, CLI, and MCP. Every `MutationGuardRejection`
   independently replays its payload through `OperationRequest` and binds
   unchanged service-resolution and backend-dispatch counters.
2. The six mandatory acceptance criteria form a second closed population
   shared literally by this heap and
   `CONTROL_MUTATION_GUARD_ACCEPTANCE_CRITERIA`.
   `ControlMutationGuardEvidence.evaluate_objective_completion` requires one
   submitted validation per exact criterion and rejects missing, duplicate,
   extra, failed, stale, foreign-tree, wrong-policy, wrong-objective, or
   operationally detached records.
3. `GoalCoverageMap` projection requires every exact criterion row to name a
   concrete implementation surface and bind that criterion's submitted
   validation receipt. A verified summary cannot replace the closed row
   population or repair a detached validation.
4. Analyzer status is never inferred. The gate requires explicit `healthy`
   and `safe_for_completion_reasoning` values plus the exact G104 analyzer,
   objective, and current-tree binding.
5. `ControlMutationCompletionMemberHealth` binds an explicit healthy and
   completion-safe attestation to every independently named exhaustive member
   and receipt CID. `ControlMutationCompletionQuorumEvidence` requires exact
   population equality and binds that quorum to the G104 requirement,
   operational receipt, validation policy, tree, analyzer, configuration, and
   ASI-077 objective revision. Missing, unhealthy, unsafe, duplicate,
   non-exhaustive, stale, or foreign members fail closed.
6. Producing tasks must be complete before the gate can pass. A complete
   passing evaluation from `active` advances only to
   `provisionally_complete`; verified completion requires a separate later
   evaluation with the full proof population still valid. ASI-G104 and its
   ASI-G070 parent therefore remain actionable for supervisor ingestion of
   final post-change receipts.

## Mandatory criterion map

| Criterion | Implementation | Validation |
| --- | --- | --- |
| Unauthorized, unscoped, unfenced, stale, path-escaping, or undeclared-effect mutations fail before dispatch on every surface | exact effect authorization in `OperationRequest._validate_mutation_bindings`; `SupervisorControlService._preflight_dispatch_boundary`; the closed `MutationGuardRejection` six-by-three matrix; decode-before-resolution CLI and MCP adapters | `test_mutation_guard_evidence_replays_all_required_fail_closed_cases`, `test_stale_fence_and_expired_authorization_fail_before_backend`, `test_cli_rejects_every_unsafe_real_mutation_before_service_resolution`, `test_mcp_rejects_every_unsafe_real_mutation_before_service_resolution` |
| dry-run stays proposal-only | the mutation dry-run branch in `SupervisorControlService.execute` and typed lifecycle/CLI adapters | `test_dry_run_never_calls_mutation_or_requires_a_live_lease`, `test_lifecycle_dry_run_binds_typed_command_without_dispatch`, `test_cli_proposal_and_dry_run_use_the_same_result_envelope` |
| a permitted current mutation emits a typed applied-effect audit receipt | exact authorization/effect binding, `ControlAuditReceipt`, result effect claims, and mutation runtime snapshots | `test_mutation_is_authorized_fenced_audited_and_idempotent`, `test_authorized_lifecycle_mutation_is_fenced_audited_and_idempotent` |
| exact retries and restart replay do not duplicate the backend effect | request-content identity, scoped idempotency store, persisted replay lookup, and `MutationGuardExecutionObservation` | `test_mutation_is_authorized_fenced_audited_and_idempotent`, `test_default_store_replays_exact_mutation_result_after_restart`, `test_mutation_guard_evidence_replays_all_required_fail_closed_cases` |
| conflicting reuse fails | `SupervisorControlService._check_idempotency` compares the stored request identity before dispatch | `test_same_idempotency_key_with_different_payload_conflicts` |
| and only the complete tamper-evident applied/replayed/rejection matrix emits the exact requirement ID. | `ControlMutationGuardEvidence`, `ControlMutationCompletionMemberHealth`, `ControlMutationCompletionQuorumEvidence`, and the two-phase objective bridge | `test_mutation_guard_evidence_replays_all_required_fail_closed_cases`, `test_g104_completion_requires_bound_validation_health_and_quorum` |

The completion matrix uses canonical `GoalCoverageMap`,
`AnalyzerHealthReport`, and `ExhaustionQuorumResult` producers wrapped by the
G104-specific member-health and quorum evidence contracts. It also submits
incomplete tasks and failed, stale, missing, detached, incompletely mapped,
unsafe, analyzer-mismatched, unhealthy, duplicate-channel, foreign-policy,
foreign-receipt, and foreign-tree variants.

“Undeclared effect” in the pre-dispatch criterion is the request-side safety
boundary: a real mutation must declare at least one mutation effect and the
permit's authorized effect IDs must exactly equal that declaration. A backend
response that reports an undeclared effect is still rejected, but because it
is post-call evidence it is not used to satisfy the pre-dispatch clause.

## Validation observation

- Command: `python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q`
- Observed result during ASI-077 implementation: 53 passed

This discovery record is an audit index, not a completion receipt. It claims
neither the final post-change repository tree identity nor the independent
analyzer/quorum executions that can exist only after the change is finalized.
The supervisor must keep ASI-G104 and ASI-G070 actionable until a post-change
evaluation supplies fresh passing current-tree criterion receipts, explicit
healthy completion-safe analyzer evidence, the configured independent fresh
healthy exhaustive quorum, and a canonical completion-gate record.
Verification may occur only in a separate evaluation after provisional
completion.
