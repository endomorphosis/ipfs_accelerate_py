# ASI-074 completion-gate evidence map

- Date: 2026-07-24
- Task: ASI-074
- Goal: ASI-G093
- Parent: ASI-G010
- Requirement: `248026856102230635452423769994290240744`
- Source gap fingerprint: `4a168d424c57516150ea8c07f4719f96184fe14a`
- Evidence obligation: `objective-work/v1/573cd85b5929284ecec96a6deb6dc6ac1b0e7a9b`
- Merge role: `completion_gate`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-074 covers the missing completion-evidence terms without allowing this
task, document, or a diagnostic report to authorize completion:

1. `TerminalAcceptedWorkEvidence` is a bounded, content-addressed benchmark
   receipt containing the complete typed baseline and candidate populations.
   Construction and decoding independently replay the paired report, freeze
   one goal/tree/policy binding, and bind the source receipt identities,
   report identity, deterministic input digest, passing accounting result, and
   exact requirement. The completion bridge also rebuilds the artifact from
   its independently enumerated cohort. Reordering canonicalizes; omission,
   duplication, substitution, or altered content fails closed.
2. The four mandatory criteria form one closed population shared literally by
   the objective heap and
   `TERMINAL_ACCEPTED_WORK_ACCEPTANCE_CRITERIA`.
   `TerminalAcceptedWorkEvidence.evaluate_objective_completion` projects a
   canonical `GoalCoverageMap` to ASI-G093 and requires every row to bind both
   a concrete implementation surface and a submitted validation receipt
   identity.
3. Every submitted validation is checked by the canonical completion gate.
   Each mandatory criterion needs a fresh passing receipt on the current
   repository tree; a missing, extra failed, stale, or foreign-tree record
   cannot be masked by other passing records.
4. Analyzer health must be explicitly healthy and
   `safe_for_completion_reasoning`. The bridge accepts the canonical
   `AnalyzerHealthReport`; incomplete mappings cannot infer either fact.
5. The configured exhaustion quorum must contain independent fresh receipts
   with unique member, receipt, and evidence-channel identities. A canonical
   evaluated `ExhaustionQuorumResult` fixes repository, tree, analyzer,
   configuration, and objective revision. Persisted specialized records must
   additionally attest the goal, policy, requirement, terminal evidence,
   paired report, and benchmark input identities.
6. A passing first evaluation can move ASI-G093 only from active to
   provisional. Verification requires a separate later evaluation with the
   complete evidence population still valid.

## Mandatory criterion map

| Criterion | Implementation | Validation |
| --- | --- | --- |
| The exact requirement ID is emitted only by a bounded, content-addressed benchmark receipt carrying the complete typed baseline and candidate receipt populations, one frozen goal/tree/policy binding, the independently replayed paired result, source and report identities, a deterministic input digest, and a passing accounting result. A completion gate verifies the artifact against its independently enumerated benchmark cohort, so an omitted, duplicated, reordered, or substituted input is either canonicalized to the same evidence identity or fails closed. The accepted-task population must be non-empty and identical across arms | `EfficiencyReceipt`, `build_paired_efficiency_report`, `TerminalAcceptedWorkEvidence`, `build_terminal_accepted_work_evidence`, and `verify_terminal_accepted_work_evidence` | `test_terminal_accepted_work_evidence_replays_complete_source_populations`, `test_terminal_evidence_verifier_requires_the_independent_complete_cohort`, `test_g093_completion_requires_current_cohort_health_quorum_and_two_phases` |
| failed-only tasks never enter its denominator | Accepted-task intersection and denominator construction in `build_paired_efficiency_report` | `test_paired_report_measures_only_terminal_accepted_tasks_and_charges_attempts`, `test_terminal_accounting_rejects_nonempty_arms_without_accepted_work` |
| every supplied failed or retried lifecycle for work that eventually reaches acceptance remains charged | Task-reference lifecycle grouping in `build_paired_efficiency_report` and `WorkCost`/token aggregation | `test_paired_report_measures_only_terminal_accepted_tasks_and_charges_attempts`, `test_retry_records_are_contiguous_compact_and_accounted_in_totals` |
| duplicate acceptance, omitted or altered embedded receipts, stale bindings, forged totals or terminal IDs, detached cases, and serialization tampering fail closed. The separate 35 percent token-reduction and full-coverage gate does not redefine the accounting proof, but context and delta promotion require both the accepted-work witness and their own typed compiler evidence. | Receipt/evidence construction and decoding identity checks, the independent cohort verifier, `RequiredContextPromotionReport`, and `DeltaRetryPromotionReport` | `test_detached_or_tampered_terminal_accounting_cannot_claim_evidence`, `test_terminal_accounting_evidence_rejects_unpaired_or_stale_populations`, `test_required_context_promotion_fails_closed_for_gap_or_forgery`, `test_delta_retry_promotion_fails_closed_for_missing_stale_or_unverified_proof` |

The completion matrix is
`test_g093_completion_requires_current_cohort_health_quorum_and_two_phases`.
It exercises mapping-backed persisted evidence and the canonical
`GoalCoverageMap`, `AnalyzerHealthReport`, and evaluated
`ExhaustionQuorumResult` producers, plus the fail-closed paths for missing,
failed, stale, duplicated, unbound, unsafe, insufficient, and foreign proof.

## Validation observation

- Command: `python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q`
- Observed result during ASI-074 implementation: 68 passed

This discovery record is an audit index, not a completion receipt. It claims
neither a final post-change repository tree identity nor analyzer/quorum
execution receipts. The supervisor must keep ASI-G093 and ASI-G010 actionable
until a post-change evaluation supplies fresh passing current-tree
`Completion evidence records`, explicit healthy completion-safe analyzer
evidence, the configured independent fresh exhaustive quorum, and a canonical
`Completion gate record`. Verification may occur only in a separate
evaluation after provisional completion.
