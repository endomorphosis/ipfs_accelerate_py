# ASI-075 completion-gate evidence map

- Date: 2026-07-24
- Task: ASI-075
- Goal: ASI-G101
- Parent: ASI-G040
- Requirement: `266404049326363900535699811645710804440`
- Source gap fingerprint: `2f6255939201205ad114477605f57d57aae6fb42`
- Evidence obligation: `objective-work/v1/8433e3f318f68068174cbfe718c3c1e6f020cfae`
- Merge role: `completion_gate`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-075 covers the missing completion-evidence terms without turning the
expected adversarial failure, this document, or task status into completion
authority:

1. `ImpactDependencyGraph`, the validated command declarations, and
   `ValidationDAGReceipt` form the operational witness. The receipt
   independently recomputes the affected closure, mandatory validation
   population, strict ready-node barriers, node dispositions, failed
   prerequisites, and downstream authority-gate closure. Its content identity
   binds the accepted proposal receipt, current tree, objective, validation
   policy, graph, changes, declarations, selected population, results, seeded
   defect, and transitive path.
2. The six mandatory G101 clauses are one closed population shared with
   `TRANSITIVE_IMPACT_ACCEPTANCE_CRITERIA`.
   `ValidationDAGReceipt.evaluate_objective_completion` accepts the canonical
   `GoalCoverageMap` projection and requires each exact clause to name an
   implementation surface and the identity of its submitted validation
   receipt. Missing, duplicate, extra, or detached rows fail closed.
3. The failed consumer-test DAG is operational detection evidence, not a
   passing completion validation. Every submitted criterion proof is
   separately checked for a passing terminal result, current-tree identity,
   freshness, exact criterion membership, unique provenance, requirement,
   objective, validation policy, and operational-receipt binding. A failed,
   stale, foreign, duplicate, or additional invalid receipt cannot be hidden
   by a passing receipt.
4. Analyzer health is explicit. The bridge accepts the canonical
   `AnalyzerHealthReport` projection or the persisted equivalent only when it
   is healthy, `safe_for_completion_reasoning`, and bound to the configured
   G101 analyzer version.
5. Exhaustion requires the configured number of independent current-tree
   receipts. Canonical `ExhaustionQuorumResult` inputs and specialized
   persisted inputs require unique member, receipt, and evidence-channel
   identities; exhaustive scan modes; exact tree, analyzer, configuration,
   and objective-revision bindings; and fresh member completion times.
   Specialized records additionally bind the validation policy and
   operational DAG receipt.
6. A passing evaluation from `active` produces only
   `provisionally_complete`. Verified completion is reachable only through a
   separate later evaluation in which the entire evidence population remains
   valid. ASI-G040 therefore remains actionable until the supervisor performs
   that post-change evaluation.

The formal completion evidence evaluator also rejects a lane when any
submitted receipt is stale, failed, unbound, or invalidated; one passing
receipt can no longer mask an invalid sibling. The validation scheduler
records a same-stage command left undispatched by fail-fast without falsely
naming the failed peer as a dependency ancestor.

## Mandatory criterion map

| Criterion | Implementation | Validation |
| --- | --- | --- |
| The validation DAG is derived from the canonical changed-file and dependency/interface impact graph and validated declarations | `ImpactDependencyGraph`, `ProposalValidationResult.require_admitted_binding`, and `ValidationScheduler.run_validated` | `test_transitive_impact_selects_failing_test_and_proves_exact_g101_requirement`, `test_stale_impact_graph_is_rejected_before_runner_dispatch`, `test_impact_graph_rejects_multi_node_dependency_cycles` |
| The receipt contains the complete selected population and every mandatory direct and transitive validation exactly once | `ValidationDAGReceipt.required_validation_ids`, `selected_node_ids`, graph-derived coverage replay, and per-node selection records | `test_receipt_rejects_recanonicalized_incomplete_affected_closure`, `test_declared_transitive_validation_cannot_be_omitted_from_population`, `test_missing_or_uncovered_impact_fails_closed_without_false_completion` |
| Missing, stale, cyclic, inconsistent, or population-incomplete impact evidence fails closed before granting authority | graph, receipt, selected-population, dependency, authority, and digest validation in `ImpactDependencyGraph` and `ValidationDAGReceipt` | `test_validation_dag_receipt_rejects_tampering`, `test_empty_or_omitted_only_receipts_cannot_claim_a_passing_dag`, `test_nonqualifying_results_never_emit_transitive_requirement` |
| A seeded upstream defect selects and executes its transitively affected consumer validation and records the real failure | `TransitiveImpactValidationEvidence` bound to the failed mandatory node, result digest, seed, graph path, and DAG receipt | `test_transitive_impact_selects_failing_test_and_proves_exact_g101_requirement` |
| Semantic, proof, merge, freshness, and completion authority remain closed by explicit records bound to the failed validation | `ValidationAuthorityGateRecord`, `transitive_impact_blocks_proof_derivation`, and `evaluate_transitive_impact_admission_closure` | `test_transitive_failure_blocks_dependent_semantic_and_proof_nodes`, `test_seeded_transitive_failure_blocks_completion_despite_valid_proposal`, `test_fail_fast_same_stage_peer_is_recorded_without_false_dependency` |
| The exact transitive-impact requirement is emitted only by a tamper-evident current-tree witness | content-addressed `ValidationDAGReceipt` plus `TransitiveImpactValidationEvidence` and the closed objective bridge | `test_nonqualifying_results_never_emit_transitive_requirement`, `test_validation_dag_receipt_rejects_tampering`, `test_g101_objective_repair_requires_closed_two_phase_proof` |

The completion matrix in
`test_g101_objective_repair_requires_closed_two_phase_proof` exercises both
mapping-backed persistence and canonical coverage/quorum producers. It also
submits omitted, duplicate, failed, stale, unbound, unsafe, unhealthy,
non-exhaustive, under-count, inconsistent-count, duplicate-receipt, and
foreign-tree variants.

## Validation observation

- Command: `python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q`
- Observed result during ASI-075 implementation: 60 passed

This discovery record is an audit index, not a completion receipt. It claims
neither a final post-change repository tree identity nor analyzer/quorum
execution receipts. The supervisor must keep ASI-G101 and ASI-G040 actionable
until a post-change evaluation supplies fresh passing current-tree criterion
receipts, explicit healthy completion-safe analyzer evidence, the configured
independent fresh exhaustive quorum, and a canonical completion-gate record.
Verification may occur only in a separate evaluation after provisional
completion.
