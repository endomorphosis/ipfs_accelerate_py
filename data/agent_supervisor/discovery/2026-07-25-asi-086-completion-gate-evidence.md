# ASI-086 Bounded Self-Refill Completion Evidence

- Date: 2026-07-25
- Task: ASI-086
- Goal: ASI-G080 — Benchmark-driven bounded self-refill
- Parent: ASI-G000
- Source gap fingerprint: `d839bf3e1cdaf61c33f0e9d753afce59b06dff90`
- Evidence obligation:
  `objective-work/v1/eb80c46cde69abbe6c94dadd5651812c9722f084`
- Todo vector: `eb80c46cde69abbe`
- Merge family: ASI-G080
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- Lifecycle: provisionally complete and supervisor-actionable

## Gap disposition

ASI-086 is the generic completion-evidence projection of the same cohesive
ASI-G080 gate implemented by ASI-087. It does not create a duplicate child or
make ASI-086 a circular producer. ASI-022 remains the original producing task,
and G109, G110, and G111 remain the bounded-successor, exact-replay, and
healthy-exhaustion leaf owners.

The four missing evidence terms are covered as follows:

1. **completion task closure** — `evaluate_self_improvement_completion`
   requires the exact, unique ASI-022 producer population to be terminally
   successful and requires the caller's closure assertion to be the literal
   boolean `true`. The canonical gate projection retains required/submitted
   task identities, statuses, population completeness, and the derived
   closure verdict instead of reducing the proof to an unbound boolean.
2. **completion criterion coverage** — all five literal G080 criteria require
   exactly one fresh passing current-tree `CompletionEvidence`. The exact
   coverage population must name concrete implementation and bind each row to
   that criterion's submitted validation receipt identity.
3. **completion analyzer health** — analyzer health must explicitly be
   `healthy` and safe for completion reasoning under the exact repository,
   tree, G080, `ASI-G080@asi-087`, analyzer-version, and configuration binding.
   Operational benchmark health is not promoted into this parent input.
4. **completion exhaustion quorum** — exactly two independently named
   members, evidence channels, and receipt identities must each be fresh,
   healthy, completion-safe, exhaustive, and bound identically to the
   analyzer and current parent decision.

Every direct child must additionally remain verified with a fresh passing
current-tree gate, validation evidence, and fresh conclusive uncontradicted
proof requirements. The canonical lifecycle still requires separate
active-to-provisional and provisional-to-verified evaluations.

## Durable backlog alignment

A serialized historical decision cannot suppress G080 merely by claiming
`verified_complete`. `completion_gate_actionable_goal_ids` now requires the
complete canonical decision schema, closed tasks, the exact valid criterion
result population, empty missing/invalid/actionable populations, all six
independent passing gate checks, the full evaluated coverage/health/quorum
payload, a fresh evaluation time, and the current repository/tree binding.
Skeletal, stale, foreign-tree, incomplete, or malformed records fail closed.

`record_configured_objective_backlog_findings` derives the current completion
tree identity and passes it through
`align_completion_gate_force_goal_ids`. Therefore a provisional or invalidated
G080 is force-fed back to the supervisor, while only a complete, fresh,
current-tree verified decision can stop forcing it. A verified goal whose
proof expires transitions to `reopened` and is immediately actionable again.

## Proof map

| Obligation | Implementation witness | Focused validation |
| --- | --- | --- |
| completion task closure | `SELF_IMPROVEMENT_PRODUCING_TASK_IDS`, the exact producer check, and `producing_task_closure` in the evaluated coverage projection | `test_g080_parent_completion_requires_closed_current_tree_proof_packet`; `test_g080_parent_rejects_incomplete_wrong_or_duplicate_producers` |
| completion criterion coverage | `SELF_IMPROVEMENT_ACCEPTANCE_CRITERIA`, `CompletionEvidence`, and exact coverage-to-receipt binding in `evaluate_self_improvement_completion` | `test_g080_parent_rejects_each_invalid_submitted_criterion_evidence`; `test_g080_parent_rejects_incomplete_or_unbound_coverage` |
| completion analyzer health | exact `SELF_IMPROVEMENT_COMPLETION_*` binding plus explicit healthy/completion-safe checks | `test_g080_parent_requires_explicit_completion_safe_analyzer` |
| completion exhaustion quorum | configured two-member bound quorum with unique member/channel/receipt identity and fresh exhaustive members | `test_g080_parent_requires_independent_fresh_healthy_exhaustive_quorum` |
| parent remains actionable | current canonical durable-decision validation and current-tree propagation through configured refill | `test_g080_verified_completion_reopens_and_requeues_on_stale_proof`; `test_g080_durable_backlog_projection_requires_canonical_fresh_decision`; `test_configured_refill_receives_current_completion_alignment` |

## Validation boundary

The mandatory command is:

```text
python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
```

This discovery index, the objective heap, task status, a naked requirement ID,
and operational refill output are routing and audit records, not completion
receipts. ASI-G080 and ASI-G000 remain actionable until a post-change
current-tree evaluation supplies fresh passing validation for every criterion,
the exact coverage bindings, explicit completion-safe analyzer health, two
independent fresh healthy exhaustive receipts, current verified child proofs,
and then succeeds in a separate provisional-to-verified transition.
