# ASI-091 completion-gate evidence map

- Date: 2026-07-25
- Task: ASI-091
- Goal: ASI-G100
- Parent: ASI-G040
- Producing task: ASI-031
- Requirement: `314133036252270790078901745919131980427`
- Source gap fingerprint: `374843191ffec232dd1acac397775bce4afc5fd9`
- Evidence obligation: `objective-work/v1/30849f363ca125ed40dd2bd99a348262925df890`
- Todo vector: `30849f363ca125ed`
- Merge family: ASI-G100
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- State after this implementation: active and supervisor-actionable

## Gap disposition

ASI-091 supplies the missing G100 completion boundary without allowing this
task, this document, a caller summary, or an expected rejection to authorize
completion:

1. The producing-task population is fixed to ASI-031. A summary
   `tasks_complete=True` is insufficient when that producer is missing,
   duplicated, replaced, or not terminal-successful.
2. The operational witness is the complete `ProposalValidationResult`,
   including its content-addressed `ProposalValidationReceipt` and
   `ProposalRejectionEvidence`. Admission replays the versioned proposal
   schema, frozen authority and baseline/candidate identities, effective
   change, normalized path safety, and the intersection of immutable
   `task_owned_paths` with policy `allowed_paths`. The policy therefore cannot
   widen task scope. Empty or effectless changes and every old or new path of
   a rejected add, modify, delete, rename, or copy remain inside the ordered
   pre-execution boundary.
3. The four literal G100 acceptance clauses form one closed criterion
   population. The completion evaluator projects a canonical
   `GoalCoverageMap` and requires every exact clause to name a concrete
   implementation surface and the identity of its submitted validation
   receipt. Missing, duplicate, extra, or detached rows fail closed.
4. The expected failed proposal is operational fail-fast evidence, not a
   passing completion validation. Every submitted criterion proof is checked:
   each exact criterion must have one fresh passing terminal receipt on the
   current repository tree, with unique provenance and an exact objective,
   requirement, policy, operational-receipt, and criterion binding. A failed,
   stale, future, foreign, duplicated, or additional invalid receipt cannot
   be hidden by a passing sibling.
5. Analyzer health is explicit rather than inferred from schema, proposal,
   scheduler, semantic, proof, or test results. Its record must be healthy,
   `safe_for_completion_reasoning`, and bound to the configured G100
   objective revision, analyzer version, configuration revision, repository,
   and current tree.
6. The configured exhaustion quorum is two and cannot be caller-lowered. Both
   members must be fresh, healthy, completion-safe, exhaustive, identically
   bound to the configured G100 objective revision, analyzer, configuration,
   repository, and current tree, and independent by member, evidence-channel,
   and receipt identities.
7. A closed pass from `active` advances only to `provisionally_complete`.
   Verified completion requires a separate later evaluation while the full
   producer, operational witness, criterion-validation, coverage, analyzer,
   and quorum population remains valid.

The rejection producer remains fail closed. Schema and authority failures,
unsafe or unowned paths, no effective change, altered proposal/baseline/scope
or diff content, incomplete gate traces, detached scheduler outcomes, cross-
tree replay, and forged proof, merge, freshness, completion, or generic
authority cannot claim G100. No smaller child goal is needed: ASI-031 already
owns the cohesive pre-execution admission boundary, while sibling ASI-G101
owns post-admission impact-selected execution.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| Proposal admission deterministically checks schema, authority, baseline and candidate identity, non-empty effective change, normalized path safety, and task-owned scope before any expensive validation. Empty or effectless diffs and every out-of-scope path fail closed with bounded typed diagnostics | `ImplementationProposal`, `ProposalValidationPolicy`, `validate_implementation_proposal`, ordered `ProposalGate` replay, normalized candidate-diff paths, immutable `task_owned_paths`, bounded `ProposalValidationFinding` records, and `ProposalValidationResult` | `test_router_accepts_only_the_exact_versioned_task_envelope`, `test_accepts_an_exactly_bound_effectful_proposal`, `test_rejects_python_comment_or_format_only_rewrite_as_non_semantic`, `test_noop_and_out_of_scope_rejections_are_typed_fail_fast_evidence`, and `test_g100_completion_requires_exact_current_tree_evidence_population` |
| policy cannot widen task scope | The effective path envelope is the normalized intersection of independently content-addressed `ProposalValidationPolicy.task_owned_paths` and policy `allowed_paths`; the bound `ProposalValidationResult` and `ProposalRejectionEvidence` persist both scopes and replay every old/new candidate path | `test_policy_allowance_cannot_widen_immutable_task_owned_scope`, `test_lossy_unsafe_rename_path_cannot_disappear_during_normalization`, the add/modify/delete/rename/copy path cases and receipt-tamper matrix in `test_agent_supervisor_proposal_validation.py`, plus `test_g100_completion_rejects_detached_unsafe_or_nonindependent_proof` |
| rejected output cannot claim proof, completion, merge eligibility, or authority | `ProposalValidationResult`, `ProposalValidationReceipt`, and `ProposalRejectionEvidence` fix all proof/code-proof/merge/freshness/completion/generic authority fields false and reject serialized claim tampering | `test_rejected_output_cannot_create_semantic_or_code_proof_obligations`, `test_serialized_result_rejects_tampered_identity_authority_and_verdict`, `test_v2_rejects_forged_proof_and_completion_authority`, and `test_g100_completion_rejects_detached_unsafe_or_nonindependent_proof` |
| the scheduler cannot be reached through the validated pipeline after preflight rejection. The exact requirement ID is emitted only by a tamper-evident receipt that binds the current tree, objective, policy, proposal, baseline, scope, normalized diff, complete ordered gate trace, failure result, proof that expensive dispatch remained closed, and content digest | `ValidationScheduler.run_validated` binds the closed-dispatch outcome and returns the fully bound `ProposalValidationResult`; `ProposalValidationReceipt` attaches `ProposalRejectionEvidence` only for the qualifying fail-fast finding population and zero expensive dispatch | `test_rejected_proposal_closes_dispatch_and_proves_exact_g100_requirement`, `test_rejection_receipt_rejects_detached_or_mutated_evidence`, `test_rejection_requirement_projection_cannot_be_erased`, `test_g100_completion_requires_exact_current_tree_evidence_population`, and `test_g100_completion_rejects_detached_unsafe_or_nonindependent_proof` |

`test_g100_completion_requires_exact_current_tree_evidence_population` and
`test_g100_completion_rejects_detached_unsafe_or_nonindependent_proof`
exercise the canonical `GoalCoverageMap`, explicit analyzer-health record,
evaluated independent exhaustion quorum, fixed ASI-031 producer population,
and two-phase lifecycle. Their negative population covers a missing,
nonterminal, substituted, or duplicated producer; a missing criterion proof;
a detached operational receipt; unbound coverage; unsafe analyzer evidence;
duplicate or stale quorum receipts; and a caller-lowered quorum. The shared
completion evaluator additionally checks every submitted proof for its
passing, freshness, provenance, and current-tree binding. The operational
proposal rejection is validated independently from the fresh passing receipts
that prove the four criteria.

## Validation route

The required current-tree command is:

```text
python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
```

- Observed result during ASI-091 implementation: 152 passed

This discovery record is an audit and provenance index, not a completion
receipt. It deliberately claims no final post-change tree identity, passing
criterion receipt, analyzer run, exhaustion vote, or lifecycle transition.
The submitting runner must execute the command after all artifact changes.
The supervisor must keep ASI-G100 and ASI-G040 actionable until ASI-031 is
terminal-successful; the exact four criteria have fresh passing current-tree
validations and receipt-bound implementation mappings; the G100 analyzer is
explicitly healthy, completion-safe, and fully bound; two independent fresh
healthy exhaustive receipts pass; and a separate provisional-to-verified
evaluation succeeds.
