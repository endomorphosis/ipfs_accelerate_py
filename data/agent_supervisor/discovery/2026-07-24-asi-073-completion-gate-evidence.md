# ASI-073 completion-gate evidence map

- Date: 2026-07-24
- Task: ASI-073
- Goal: ASI-G098
- Parent: ASI-G030
- Requirement: `003778425160038348524906247302938706902`
- Source gap fingerprint: `3fa69d82755036844ff2276752ac006f69795a0d`
- Evidence obligation: `objective-work/v1/2a7f341e9c9dec24c6e4403a835ec127ddb90b23`
- Merge role: `completion_gate`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-073 covers the three missing implementation evidence terms without
self-authorizing completion:

1. Completion criterion coverage is a closed six-clause population shared
   literally by the heap and
   `NEW_COUNTEREXAMPLE_REFINEMENT_ACCEPTANCE_CRITERIA`.
   `AdaptiveRefinementResult.evaluate_objective_completion` accepts the
   canonical `GoalCoverageMap` projection, selects ASI-G098 rows only, and
   requires every clause to bind a non-empty implementation surface and a
   validation receipt. Missing rows and missing implementation or validation
   bindings fail closed.
2. Completion analyzer health accepts the canonical `AnalyzerHealthReport`
   projection only when status, the healthy flag, and
   `safe_for_completion_reasoning` are all explicitly true. Status-only,
   unhealthy, and unsafe records fail closed.
3. Completion exhaustion quorum accepts a canonical evaluated
   `ExhaustionQuorumResult` or an explicitly attested persisted mapping.
   Configured count, unique members, unique receipt CIDs, independent evidence
   channels, exhaustive healthy completion-safe semantics, freshness, and
   exact tree/analyzer/configuration/objective bindings are required.
   Insufficient, duplicate, partial, unhealthy, unsafe, stale, or foreign
   evidence fails closed.

`ResponsiveReplanDecision` remains routing metadata. Its
`completion_evidence_roles` and `evidence_ids` are empty, so it cannot be used
as analyzer health, criterion coverage, exhaustion quorum, or objective proof.

## Mandatory criterion map

| Criterion | Implementation | Validation |
| --- | --- | --- |
| A changed typed counterexample can generate and admit at most one bounded refinement in the next cycle | `AdaptiveGoalRefiner.refine` root/cycle lock, persisted budgets, and `NewCounterexampleRefinementEvidence` | `test_new_counterexample_triggers_exactly_one_bounded_verified_refinement`, `test_distinct_changed_counterexamples_share_one_generation_slot_per_cycle`, `test_concurrent_same_evidence_performs_one_generation_and_admission` |
| The frozen root is never mutated | frozen-context and root-content admission checks | `test_frozen_root_and_assumptions_cannot_be_mutated` |
| The request and candidate remain on the frozen repository tree | request/candidate/plan tree equality checks | `test_candidate_must_bind_request_signal_kind_and_repository_tree`, `test_request_repository_tree_must_match_frozen_plan` |
| Admission is policy gated, the candidate declares the exact bounded changed-goal set, and verification binds the exact candidate plan with a boolean proof result | policy bounds, canonical changed-goal diff, and typed verifier binding | `test_changed_goal_declaration_and_change_budget_are_enforced`, `test_bare_boolean_verifier_cannot_assert_proof`, `test_non_boolean_verification_status_cannot_assert_proof`, `test_verification_for_another_candidate_plan_cannot_be_replayed` |
| The witness binds the exact requirement ID, trigger signal, request and evidence fingerprint, frozen root/tree/policy identities, previous and candidate plans, producer, verification receipt, refinement index, and content digest | content-addressed `NewCounterexampleRefinementEvidence` embedded in the durable adaptive receipt | `test_new_counterexample_triggers_exactly_one_bounded_verified_refinement`, `test_counterexample_witness_tampering_fails_closed` |
| Non-counterexample admissions remain non-authoritative for this requirement, and restored objective receipts reject unsupported versions, missing identities, and unknown fields | closed signal authority and fail-closed receipt restoration | `test_all_reviewed_typed_changes_are_eligible`, `test_persisted_objective_receipts_fail_closed_on_unreviewed_shape` |

The completion-gate matrix is exercised by
`test_g098_completion_requires_fresh_complete_current_tree_proof`. It uses the
canonical coverage, analyzer-health, and exhaustion-quorum producer types and
also submits malformed or incomplete variants for every fail-closed branch.

## Validation observation

- Command: `python -m pytest test/api/test_agent_supervisor_adaptive_goal_refiner.py -q`
- Observed result during ASI-073 implementation: 33 passed

This discovery record is an audit index, not a completion receipt. It does not
claim a final repository tree identity, analyzer execution receipt, or
independent exhaustion vote. The supervisor must keep ASI-G098 and ASI-G030
actionable until a post-change run supplies fresh passing current-tree
`Completion evidence records`, an explicit healthy completion-safe analyzer
record, the configured independent fresh exhaustive quorum, and a canonical
`Completion gate record`. Verification may occur only in a separate
evaluation after provisional completion.
