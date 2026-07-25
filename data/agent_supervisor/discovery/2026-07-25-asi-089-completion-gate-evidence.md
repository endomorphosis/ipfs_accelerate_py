# ASI-089 completion-gate evidence map

- Date: 2026-07-25
- Task: ASI-089
- Goal: ASI-G040
- Parent: ASI-G000
- Child goals: ASI-G100, ASI-G101, ASI-G102
- Producing tasks: ASI-010, ASI-011, ASI-012
- Requirements: `314133036252270790078901745919131980427`, `266404049326363900535699811645710804440`, `006818797857632260116084792540150258746`
- Source gap fingerprint: `c3718670dcc69881205fa734efde8d996d092db3`
- Evidence obligation: `objective-work/v1/d943cb6487309dd72c6dc4230058503db4f10335`
- Todo vector: `d943cb6487309dd7`
- Merge family: ASI-G040
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-089 supplies the missing parent completion boundary without allowing this
task, this document, a completed producer, or a detached validation report to
authorize ASI-G040 completion:

1. The producing-task population is fixed to ASI-010, ASI-011, and ASI-012.
   `tasks_complete=True` is insufficient when any producer is missing,
   duplicated, unexpected, or not terminal-successful.
2. The direct descendant population is fixed to ASI-G100, ASI-G101, and
   ASI-G102. Every child must retain a verified passing gate evaluated
   freshly on the current repository tree, nonempty validation evidence, and
   nonempty conclusive, uncontradicted, assurance-sufficient proof
   requirements. A reopened, stale, proofless, foreign-tree, or
   caller-substituted child keeps the parent actionable.
3. The two literal parent acceptance clauses are a closed population. Every
   submitted validation participates in the decision, and each coverage row
   must name a concrete implementation surface, validation surface, and the
   submitted receipt identity for that exact criterion. Missing, duplicate,
   detached, failed, stale, future, or foreign-tree validation cannot be
   hidden by a passing sibling.
4. Analyzer health is not inferred from proposal, DAG, semantic, proof, or
   test success. Its record must explicitly be healthy and
   `safe_for_completion_reasoning` and bind the repository, tree, ASI-G040,
   `STRICT_VALIDATION_OBJECTIVE_REVISION`,
   `STRICT_VALIDATION_COMPLETION_ANALYZER_VERSION`, and
   `STRICT_VALIDATION_COMPLETION_CONFIGURATION_REVISION`.
5. The configured exhaustion quorum is two and cannot be caller-lowered.
   Both receipts must be fresh, healthy, completion-safe, exhaustive,
   identically bound, and independent by member ID, evidence channel, and
   receipt identity.
6. The parent gate is a typed three-owner join, not a caller-authored gate
   checklist. It reconstructs proposal-owned
   `ProposalValidationReceipt.proposal_gate_evidence`, scheduler-owned
   `StrictValidationDAGCompletionEvidence`, and proof-owned
   `StrictValidationProofCompletionEvidence`, verifies their separate
   tree/objective/receipt identities and false completion authority, and
   requires the exact owner union to equal `STRICT_VALIDATION_GATE_KINDS`.
   The complete failed validation DAG remains operational defect-detection
   evidence, not passing parent validation evidence.
7. A closed pass from `active` advances only to `provisionally_complete`.
   Verified completion requires a separate later evaluation while the entire
   producer, child, validation, coverage, analyzer, and quorum population
   remains valid.

No additional child goal is needed. G100, G101, and G102 are the stable
pre-execution proposal rejection, post-admission impact execution, and
semantic proof-to-code-completion authority partitions. The objective scan
records G100 as active and G101/G102 as provisionally complete, so ASI-G040
must remain supervisor-actionable regardless of the ASI-089 implementation
and test result.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| Schema, authority, patch, path, AST/interface, impact-test, semantic/proof, merge, and freshness gates are explicit. Validation declarations bind canonical impact targets, DAG dependencies, and downstream authority gates | `proposal_validation.PROPOSAL_OWNED_GATE_GROUPS`, `ProposalValidationReceipt.proposal_gate_evidence`, and its tree/objective/proposal/policy/receipt/diff binding; `validation_scheduler.STRICT_VALIDATION_STAGE_ORDER`, `ImpactDependencyGraph.validation_targets`, `ValidationDAGReceipt.strict_validation_completion_evidence`, `StrictValidationDAGCompletionEvidence`, and `REQUIRED_AUTHORITY_GATES`; `ProofCandidateNonAuthorityEvidence.strict_validation_completion_evidence` and `StrictValidationProofCompletionEvidence`, which reconstruct the fresh implementation-obligation/candidate/admission chain; and `formal_plan_conformance.evaluate_strict_validation_completion`, which requires the three producer-owned gate populations to equal `STRICT_VALIDATION_GATE_KINDS` | `test_receipt_projects_tree_bound_explicit_proposal_gate_evidence`, `test_receipt_rejects_partial_gate_trace_and_tampered_gate_projection`, `test_strict_validation_parent_projection_binds_complete_scheduler_gate_surface`, `test_strict_validation_parent_projection_rejects_tamper_and_non_witness_dag`, `test_seeded_transitive_failure_blocks_completion_despite_valid_proposal`, `test_provider_proof_candidate_never_becomes_code_completion_evidence`, and `test_g040_parent_completion_requires_closed_current_tree_proof_packet` |
| the receipt covers the complete selected population and schedules only dependency-ready checks under bounded parallelism. No required gate may be omitted, seeded adversarial defects do not escape, and failed output yields bounded typed diagnostics while closing proof, merge, freshness, and completion authority. | complete `ValidationDAGNodeRecord` selection/execution population, validated dependency edges, bounded ready-node scheduling, deterministic failure/block/omission dispositions, `TransitiveImpactValidationEvidence`, and the full `ValidationAuthorityGateRecord` population; `formal_plan_conformance.evaluate_strict_validation_completion` fixes `STRICT_VALIDATION_PRODUCING_TASK_IDS`, `STRICT_VALIDATION_CHILD_GOAL_IDS`, `STRICT_VALIDATION_ACCEPTANCE_CRITERIA`, the typed three-owner gate join, analyzer binding, and `STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS` before using the canonical two-phase completion evaluator | `test_declared_transitive_validation_cannot_be_omitted_from_population`, `test_dependency_aware_dag_parallelism_fail_fast_and_complete_receipt`, `test_transitive_failure_blocks_dependent_semantic_and_proof_nodes`, `test_fail_fast_same_stage_peer_is_recorded_without_false_dependency`, `test_validation_dag_receipt_rejects_tampering`, `test_formal_completion_rejects_any_submitted_invalid_receipt`, and `test_g040_parent_completion_requires_closed_current_tree_proof_packet` |

`test_g040_parent_rejects_open_or_unbound_completion_packet` exercises
missing, duplicate, or active producers; missing, duplicate, unverified,
stale-gated, proofless, or foreign-validation children; caller-authored
scheduler/proof projections; detached coverage; unsafe analyzer evidence; and
under-count, duplicate, or stale quorum members.
`test_g040_parent_rejects_caller_lowered_exhaustive_quorum` proves that the
configured count cannot be narrowed. The shared
`test_formal_completion_rejects_any_submitted_invalid_receipt` proves that an
additional failed or stale validation cannot be masked by a passing sibling,
and `test_g040_parent_completion_requires_closed_current_tree_proof_packet`
proves the typed reconstruction, exact populations, and required two-phase
lifecycle.

## Validation route

The required current-tree command is:

```text
python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
```

This discovery record is an audit and provenance index, not a completion
receipt. It deliberately claims no final post-change tree identity, passing
criterion receipt, analyzer run, exhaustion vote, or lifecycle transition.
The submitting runner must execute the command after all artifact changes.
The supervisor must keep ASI-G040 and ASI-G000 actionable until all three
producers are terminal-successful; G100, G101, and G102 are separately
verified with fresh, conclusive, uncontradicted, assurance-sufficient proof;
both exact parent criteria have fresh passing current-tree validations and
receipt-bound implementation mappings; analyzer health is explicitly healthy,
completion-safe, and fully bound; two independent fresh healthy exhaustive
receipts pass; and a separate provisional-to-verified evaluation succeeds.
