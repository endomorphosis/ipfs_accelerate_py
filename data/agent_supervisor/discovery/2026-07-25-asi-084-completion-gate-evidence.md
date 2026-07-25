# ASI-084 Task Generation Completion-Gate Evidence Map

- Date: 2026-07-25
- Task: ASI-084
- Goal: ASI-G050 — High-quality task generation and conflict-aware bundling
- Parent: ASI-G000
- Child goals: ASI-G106, ASI-G107, ASI-G108
- Requirements: `127990245919649912156052660092678945998`,
  `061582446926920746660485801841658333166`,
  `187052702852200236079602798955260586139`
- Source gap fingerprint: `803aec4e5425e8a14a1959bcaa30f61aa954eb19`
- Evidence obligation:
  `objective-work/v1/b4a99eb7f19006443473c4b528fac9a33e51dc75`
- Todo vector: `b4a99eb7f1900644`
- Merge family: ASI-G050
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- Lifecycle: provisionally complete and supervisor-actionable

## Gap disposition

ASI-084 closes the parent-gate implementation and audit gap without treating
this document, task status, a vector projection, or any one task-generation
witness as completion authority:

1. Producing-task closure is exact.
   `TASK_GENERATION_PRODUCING_TASK_IDS` fixes ASI-013 and ASI-014 as the
   original task-admission and bundle-optimization producer population.
   `evaluate_task_generation_completion` requires each task exactly once in a
   successful terminal state in addition to the caller's completion
   assertion. Missing, duplicate, foreign, or incomplete producers fail
   closed. ASI-084 is not a circular producing-task requirement.
2. Descendant closure is exact and recursive.
   `TASK_GENERATION_CHILD_GOAL_IDS` fixes ASI-G106, ASI-G107, and ASI-G108.
   Each child must be `verified_complete` with a fresh passing gate bound to
   the current repository tree. Every descendant proof requirement remains
   freshly proved, conclusive, uncontradicted, and satisfied at its required
   assurance. A split/refill witness cannot replace width or packet-completion
   proof, and no child can directly complete its parent.
3. `TASK_GENERATION_ACCEPTANCE_CRITERIA` fixes the parent acceptance
   population to the five literal ASI-G050 clauses. Callers cannot narrow or
   replace it. Every submitted validation participates in the decision; one
   failed, stale, malformed, contradictory, or foreign-bound sibling
   invalidates the submission even when another receipt for the same
   criterion passes.
4. Coverage is implementation- and receipt-bound. There is exactly one
   coverage row per literal criterion on the current tree. Each row names a
   concrete implementation binding and the provenance identity of a submitted
   fresh passing validation for that exact criterion. A requirement ID,
   child status, discovery record, optimization metric, or detached receipt
   identity cannot fill a coverage gap.
5. Analyzer health is a separate authority input. It must explicitly report
   healthy and safe for completion reasoning and bind the repository, tree,
   `TASK_GENERATION_OBJECTIVE_ID`, `TASK_GENERATION_OBJECTIVE_REVISION`,
   `TASK_GENERATION_COMPLETION_ANALYZER_VERSION`, and
   `TASK_GENERATION_COMPLETION_CONFIGURATION_REVISION`. Task-quality scores,
   bundle plans, conflict projections, vector packets, and this audit index
   cannot substitute.
6. Exhaustion is configured and independent.
   `TASK_GENERATION_REQUIRED_EXHAUSTIVE_RECEIPTS` fixes the count at two. Each
   member must be independently identified, fresh, healthy, completion-safe,
   exhaustive, identically bound to analyzer health, and unique by member,
   evidence-channel, and receipt identity. Caller-supplied under-counts or
   duplicate, stale, unhealthy, unsafe, non-exhaustive, or foreign-bound
   members fail closed.
7. Completion remains two phase. A fully passing active evaluation may move
   only to provisional completion. Verification requires a later separate
   evaluation while every producer, child, validation, mapping, analyzer, and
   exhaustion receipt remains valid. Later invalidation reopens a verified
   parent.

The existing child goals are the stable minimal partition: ASI-G106 owns the
complete split/refill and duplicate-suppression proof, ASI-G107 owns
independent critical-path width under conflict serialization, and ASI-G108
owns exact canonical packet-completion propagation. No additional child goal
is needed.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| Tasks bind one coherent acceptance/effect subset with predicted scope and costs | `TaskCandidate.work_contract` binds the exact acceptance/effect/evidence subset, predicted paths/symbols/context, execution boundary, and positive context/task/validation cost estimates into `work_contract_id` and canonical task identity; `TaskWorkContract` then binds that admitted contract to the canonical task key/CID and preserves it through conflict, bundle, and todo-vector projections | `test_work_contract_binds_exact_coherent_subset_scope_and_predicted_costs`, `test_candidate_projection_round_trip_preserves_identity_and_rejects_tampering`, `test_canonical_identity_changes_with_predicted_scope_or_any_mandatory_cost`, `test_admission_fails_closed_with_specific_reason_for_each_zero_cost`, split/coalesce contract cases in `test_agent_supervisor_task_quality.py`, and `test_optimizer_projection_preserves_coherent_contract_effects_and_costs` in `test_agent_supervisor_bundle_optimizer.py` |
| broad tasks split and compatible tiny tasks coalesce | `is_over_broad`, `split_task_candidate`, `can_coalesce_tasks`, `coalesce_task_candidates`, and `refine_task_candidates` preserve complete source acceptance/effect/evidence surfaces, prerequisites, execution context, and merge fate while enforcing policy bounds | `test_over_broad_task_splits_deterministically_and_preserves_dependencies`, `test_tiny_tasks_coalesce_only_with_shared_execution_and_merge_fate`, and the resizing-lineage cases in `test_agent_supervisor_task_quality.py` |
| semantic duplicates are rejected across refills | `canonical_semantic_identity`, historical similarity checks, bounded admission pressure, and producer-sealed `TaskSplitRefillEvidence` bind a complete first admission and zero-admission replay of the same canonical children | duplicate/failure-history, existing semantic duplicate, bounded refill, normalized alias, positive split/refill, and forged/tampered split/refill cases in `test_agent_supervisor_task_quality.py` |
| bundles preserve critical-path width and serialize conflicts | `conflict_graph.project_conflict_free_wave`, closed prerequisite resolution, `optimize_task_bundles`, and producer-sealed `CriticalPathWidthEvidence` replay exact dependency waves, blocking pairs, serialization edges, bundle membership, and preserved independent lanes | dependency-width, hierarchy-depth separation, conflicting-context, dependency-direction, path-coloring, closed-DAG width, unresolved-population, and projection-replay cases in `test_agent_supervisor_bundle_optimizer.py` |
| model calls per accepted work item improve without increasing merge conflicts. | `_plan_metrics`, `compare_bundle_plan_metrics`, and `BundlePlanComparison` derive accepted-work, model-call, context/validation reuse, completion, critical-path, and merge-conflict measures from the canonical optimized and current-planner populations | `test_plan_metrics_measure_real_bundle_reuse_and_compare_current_planner`, global AST conflict serialization, path-width conflict-rate, and bundle-supervisor comparison cases in `test_agent_supervisor_bundle_optimizer.py` |

The parent completion matrix additionally exercises incomplete producing
tasks; missing, duplicate, foreign, reopened, proofless, stale, or
foreign-tree children; failed, stale, missing, duplicate, detached, or
unmapped criterion receipts; unsafe or mismatched analyzer evidence; and
under-count, duplicate-member, duplicate-receipt, duplicate-channel,
unhealthy, unsafe, non-exhaustive, stale, or foreign-bound quorum members.
`test_g050_parent_completion_requires_closed_current_tree_proof_packet`
constructs a fully bound active-to-provisional packet and then performs the
separate provisional-to-verified evaluation.
`test_g050_parent_rejects_incomplete_wrong_or_duplicate_producers`,
`test_g050_parent_rejects_each_invalid_submitted_criterion_evidence`,
`test_g050_parent_rejects_incomplete_or_unbound_coverage`,
`test_g050_parent_rejects_missing_unhealthy_unsafe_or_foreign_analyzer`,
`test_g050_parent_rejects_nonindependent_or_unhealthy_exhaustive_quorum`, and
`test_g050_parent_rejects_unverified_stale_or_wrong_child_population` mutate
one authority input at a time.

## Validation observation

The mandatory current-tree command is:

```text
python -m pytest test/api/test_agent_supervisor_task_quality.py test/api/test_agent_supervisor_bundle_optimizer.py -q
```

This discovery record is an audit and provenance index, not a completion
receipt. It intentionally claims no final repository-tree identity, analyzer
execution, exhaustion vote, fresh validation result, or lifecycle transition.
The submitting runner's fresh passing post-change execution is the validation
receipt. ASI-G050 and ASI-G000 remain supervisor-actionable until ASI-013 and
ASI-014 are terminal-successful, all three child goals are verified with fresh
current-tree proof, every parent criterion has a fresh passing mapped
validation, analyzer health is explicitly healthy, completion-safe, and fully
bound, two independent fresh healthy exhaustive receipts pass, and the later
separate provisional-to-verified evaluation succeeds.
