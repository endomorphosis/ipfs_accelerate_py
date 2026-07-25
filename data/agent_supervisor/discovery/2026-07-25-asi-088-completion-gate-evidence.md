# ASI-088 completion-gate evidence map

- Date: 2026-07-25
- Task: ASI-088
- Goal: ASI-G010
- Parent: ASI-G000
- Child goals: ASI-G091, ASI-G092, ASI-G093
- Requirements: `208290439421789408250562066350459701853`, `306437607356117177048620815571362227127`, `248026856102230635452423769994290240744`
- Source gap fingerprint: `5d10fa4e142379c78582e21ccbd0a39ea7b6d690`
- Evidence obligation: `objective-work/v1/992e6c459e2a84ecc5369c9bc3bfe2d2a241c5a9`
- Todo vector: `992e6c459e2a84ec`
- Merge family: ASI-G010
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-088 supplies the missing parent completion boundary without allowing this
task, this document, a child producer, or a detached benchmark report to
authorize ASI-G010 completion:

1. The producing-task population is fixed to ASI-001, ASI-005, and ASI-006.
   `tasks_complete=True` is insufficient when any producer is absent,
   duplicated, unexpected, or nonterminal.
2. The direct descendant population is fixed to ASI-G091, ASI-G092, and
   ASI-G093. Every child must retain a passing verified gate, current-tree
   validation records, a fresh gate evaluation, and nonempty conclusive,
   current proof requirements. A reopened, stale, proofless, foreign-tree, or
   caller-substituted child keeps the parent actionable.
3. The four literal parent criteria are a closed population. Every coverage
   row must name a concrete implementation surface and validation surface and
   bind the receipt identity submitted for that exact criterion. Missing,
   duplicate, cross-bound, failed, stale, future, or foreign-tree validation
   cannot be masked by a passing sibling.
4. Analyzer status is not inferred from compiler or benchmark success. The
   record must explicitly be `healthy` and
   `safe_for_completion_reasoning` and bind repository, tree, ASI-G010,
   `ASI-G010@asi-088`, analyzer version, and configuration revision.
5. The configured quorum is fixed at two. Both receipts must be fresh,
   healthy, completion-safe, exhaustive, identically bound, and independent
   by member ID, evidence channel, and receipt CID. A caller cannot lower the
   configured count.
6. `ContextCapsule.invariant_core_id` identifies the complete goal, authority,
   scope, and acceptance core. Omission diagnostics now correspond exactly to
   optional expansion handles and cannot name an invariant field.
   `ContextCompileResult.required_context_preserved` and
   `ContextDeltaResult.invariant_core_preserved` expose the revalidated base
   and retry guarantees to completion evidence mapping.
7. A closed pass from `active` advances only to `provisionally_complete`.
   Verified completion requires a separate later evaluation while the entire
   proof population remains valid.

## Mandatory criterion map

| Criterion | Implementation | Validation |
| --- | --- | --- |
| Required goal, authority, scope, and acceptance context is never truncated | `ContextCapsule` mandatory fields, canonical `invariant_core_id`, omission/expansion one-to-one validation, `ContextCompiler.compile`, `RequiredContextBudgetEvidence`, `ContextCompileResult.required_context_preserved`, and `RequiredContextPromotionReport` | `test_required_fields_and_references_survive_effective_provider_budget`, `test_required_context_fails_closed_instead_of_truncating`, `test_required_evidence_cannot_be_deferred_as_expansion_handle`, `test_invariant_core_identity_cannot_be_described_as_truncated`, `test_g010_parent_completion_closes_producers_children_and_proof_gate` |
| optional evidence has deterministic inclusion reasons | Ranked optional selection, `EvidenceSelectionDecision`, explicit omission reasons, and bounded content-addressed expansion handles | `test_optional_evidence_has_deterministic_ranking_and_decisions`, `test_text_artifact_chunks_are_complete_content_addressed_expansion_units`, `test_g010_parent_completion_closes_producers_children_and_proof_gate` |
| retries use changed evidence rather than full replay | `ContextDeltaCapsule`, `ContextCompiler.compile_delta`, `reconstruct_context`, `ContextDeltaResult.invariant_core_preserved`, and `DeltaRetryPromotionReport` | `test_delta_transmits_changes_and_preserves_required_coverage`, `test_requested_unchanged_reference_is_not_masqueraded_as_changed`, `test_delta_must_be_smaller_than_full_replay`, `test_delta_result_exposes_exact_invariant_core_preservation`, `test_g010_parent_completion_closes_producers_children_and_proof_gate` |
| paired fixtures reduce median input tokens by at least 35 percent without lowering required evidence coverage or safety. | `PairedEfficiencyReport`, `TerminalAcceptedWorkEvidence`, `RequiredContextPromotionReport`, `DeltaRetryPromotionReport`, and the fixed parent evaluator | `test_paired_report_couples_token_reduction_to_required_coverage`, `test_paired_report_uses_median_same_task_reduction`, `test_delta_retry_gate_accepts_requested_only_and_enforces_35_percent`, `test_paired_semantic_retries_reduce_median_tokens_by_at_least_35_percent`, `test_g010_parent_completion_closes_producers_children_and_proof_gate` |

The parent matrix also exercises incomplete producers; stale descendants; an
additional failed validation beside valid receipts; detached criterion
coverage; unsafe analyzer evidence; duplicate quorum receipt identity; a
caller attempt to lower the configured count; and the required two-phase
lifecycle.

## Validation observation

- Command: `python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q`
- Observed result during ASI-088 implementation: 78 passed

This discovery record is an audit index, not a completion receipt. It claims
neither the final post-change tree identity nor independent analyzer/quorum
executions. The supervisor must keep ASI-G010 and ASI-G000 actionable until a
post-change evaluation supplies the fresh passing current-tree criterion
receipts, exact implementation/validation map, explicitly healthy
completion-safe analyzer evidence, configured independent exhaustive quorum,
verified current descendants, and a canonical completion-gate record.
Verification may occur only in a separate evaluation after provisional
completion.
