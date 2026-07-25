# ASI-087 Bounded Self-Refill Completion-Gate Evidence Map

- Date: 2026-07-25
- Task: ASI-087
- Goal: ASI-G080 — Benchmark-driven bounded self-refill
- Parent: ASI-G000
- Producing task: ASI-022
- Child goals: ASI-G109, ASI-G110, ASI-G111
- Source gap fingerprint: `b13adfbc29b55d1a074b18e72b4cacc5cac035f7`
- Evidence obligation:
  `objective-work/v1/60ee13a4300b6f29dfd458062949af8a58479cb2`
- Todo vector: `60ee13a4300b6f29`
- Merge family: ASI-G080
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- Lifecycle: provisionally complete and supervisor-actionable

## Gap disposition

ASI-087 closes the missing parent-gate implementation and audit boundary. It
does not promote this discovery record, task status, an empty objective scan,
or any operational refill receipt into completion authority:

1. `SELF_IMPROVEMENT_PRODUCING_TASK_IDS` fixes ASI-022 as the complete
   original producer population. Missing, duplicate, foreign, or nonterminal
   producers fail even if a caller asserts `tasks_complete`. ASI-087 is a
   completion-gate repair, not a circular new producer.
2. `SELF_IMPROVEMENT_CHILD_GOAL_IDS` fixes G109, G110, and G111. Each child
   must remain verified with a fresh passing current-tree gate, validation
   evidence, and conclusive uncontradicted proof requirements. The three
   runtime witnesses remain leaf authority only.
3. `SELF_IMPROVEMENT_ACCEPTANCE_CRITERIA` fixes all five literal parent
   clauses. Exactly one submitted current-tree validation must name each
   clause. Every submission participates; a failed, stale, foreign, extra, or
   duplicate sibling invalidates the packet.
4. Coverage has exactly one row per criterion, names concrete implementation,
   and binds the sole submitted validation receipt identity for that row.
5. Analyzer health separately and explicitly says healthy and safe for
   completion reasoning. Its complete binding fixes repository, tree, G080,
   `ASI-G080@asi-087`, analyzer version, and configuration revision.
6. `SELF_IMPROVEMENT_REQUIRED_EXHAUSTIVE_RECEIPTS` fixes the quorum at two.
   Both members are fresh, healthy, completion-safe, exhaustive, identically
   bound, and independent by member, evidence channel, and receipt identity.
7. Completion remains two phase. A complete active packet can advance only to
   provisional completion. A later separate evaluation may verify while all
   inputs remain current; invalidation reopens a verified goal.
8. `completion_gate_actionable_goal_ids` and
   `align_completion_gate_force_goal_ids` keep a proof-incomplete G080 in the
   supervisor refill projection and stop forcing it only after a canonical
   verified decision with a passing gate and no actionable reasons.

ASI-086 described the same boundary as analyzer health, criterion coverage,
exhaustion quorum, and task closure under fingerprint
`d839bf3e1cdaf61c33f0e9d753afce59b06dff90`. The cohesive ASI-087 gate
dispositions that generic scan without creating another child. G109 owns
bounded successor materialization, G110 owns exact replay, and G111 owns
healthy no-work exhaustion, which is the stable minimal leaf partition.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| A drained board triggers one identity-bound evaluation epoch | `SelfImprovementEpochBinding`, `build_self_improvement_epoch_binding`, and `run_self_improvement_epoch` bind every meaningful input before invoking one observation provider | healthy/actionable epoch and meaningful-trigger tests in `test_agent_supervisor_self_improvement_refill.py` |
| measured gaps yield bounded goal proposals that pass quality, refinement, novelty, and policy checks | `SelfImprovementPolicy`, `materialize_self_improvement_successors`, the objective materialization transaction, and `SuccessorRefillEvidence` | supported-gap, bounded-policy, and pre-write rejection matrices |
| duplicate/cooldown work is suppressed | canonical successor filtering plus durable admission/cooldown records | `test_successor_filter_covers_terminal_lifecycle_and_durable_cooldown` |
| identical epochs are idempotent | strict ledger restoration and `EpochReplayEvidence` return before benchmark/proposal callbacks | exact replay, artifact immutability, and restoration-tamper tests |
| healthy no-gap epochs persist exhaustion quorum and wait for a meaningful trigger instead of looping. | `HealthyExhaustionEvidence`, `evaluate_self_improvement_epoch`, `record_self_improvement_exhaustion`, and the exact wait-state predicate | healthy exhaustion, nonqualifying population, replay repair, and changed-trigger tests |

The parent matrix adds exact producer/child population, one validation and one
coverage row per literal criterion, explicit analyzer health, independent
exhaustive quorum, active-to-provisional-to-verified lifecycle, and fail-closed
backlog projection. It mutates missing, duplicate, incomplete, failed, stale,
unsafe, non-exhaustive, proofless, and foreign-bound inputs one at a time.

## Validation observation

The mandatory current-tree command is:

```text
python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
```

This file is an audit and provenance index, not a completion receipt. It
claims no final repository-tree identity, analyzer execution, exhaustion vote,
fresh validation result, or lifecycle transition. The submitting runner's
fresh passing post-change execution is the validation receipt. ASI-G080 and
ASI-G000 remain supervisor-actionable until ASI-022 is terminal-successful,
G109/G110/G111 are currently verified with fresh proof, all five parent
criteria have fresh passing mapped current-tree validations, the analyzer is
explicitly healthy and completion-safe, two independent fresh healthy
exhaustive receipts pass, and a later separate provisional-to-verified
evaluation succeeds.
