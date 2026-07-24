# ASI-082 Root Supervisor Completion-Gate Evidence Map

- Date: 2026-07-24
- Task: ASI-082
- Goal: ASI-G000 — Efficient and trustworthy supervisor control loop
- Source gap fingerprint: `db30cef45181be9e37143b868c545c4e11ac23db`
- Evidence obligation: `objective-work/v1/5da092dd3d872f72a99219606961bc9a105d9c32`
- Todo vector: `5da092dd3d872f72`
- Merge family: `ASI-G000`
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- Lifecycle: provisionally complete and supervisor-actionable

## Gap disposition

1. Producing-task closure is exact. The root adapter fixes the original
   ASI-001 through ASI-024 population and requires every task exactly once in
   a terminal successful state in addition to `tasks_complete=True`. Missing,
   duplicate, foreign, or incomplete tasks fail closed. The checked-in board
   still has ASI-006, ASI-012, ASI-014 through ASI-017, and ASI-019 through
   ASI-024 open, so this repair cannot complete the root.
2. Child closure is exact and recursive. The adapter fixes ASI-G010 through
   ASI-G090 as the nine direct children. Every child must be
   `verified_complete` with a passed gate freshly evaluated for the current
   repository and tree, nonempty validation evidence, and nonempty proof
   requirements. The shared gate recursively rejects stale, unsupported,
   inconclusive, contradicted, or assurance-insufficient descendant proof.
   The current heap has no verified direct child, so ASI-G000 remains
   actionable.
3. The root criterion population cannot be narrowed. It is the four literal
   ASI-G000 clauses, and every submitted validation participates. One failed,
   stale, malformed, contradictory, or foreign-bound sibling invalidates the
   submission even when another receipt for the same criterion passes.
4. Coverage is implementation- and receipt-bound. There must be one unique
   row per literal criterion on the current tree. Each row names a concrete
   implementation binding and the provenance identity of a submitted fresh
   passing validation for that exact criterion. A status-only row, detached
   receipt identity, omitted criterion, or duplicate criterion is rejected.
5. Analyzer health is separate and explicit. It must state `status=healthy`,
   `healthy=True`, and `safe_for_completion_reasoning=True`, and bind the
   repository, tree, ASI-G000 objective, `ASI-G000@asi-082` revision, analyzer
   version, and configuration revision. Operational output, provider health,
   objective prose, and the audit index cannot substitute for it.
6. Exhaustion is configured and independent. The trusted count is two and
   cannot be lowered by a caller. Every supplied member must be fresh,
   healthy, completion-safe, exhaustive, identically bound to analyzer
   health, and unique by member ID, evidence channel, and receipt CID.
7. Lifecycle authority remains two-phase. A fully passing active evaluation
   can only record provisional completion. Verification requires a later
   evaluation while every task, child, validation, coverage row, analyzer
   binding, and exhaustive receipt remains valid. Subsequent invalidation
   reopens a verified root.

No additional child goal is needed. The existing nine workstream children are
the stable partition for measurement, analysis, planning, validation, task
generation, parallel runtime, control, refill, and rollout. ASI-082 supplies
the missing closed root adapter, validation matrix, policy documentation, and
audit index without duplicating their implementation work.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| Every child goal has fresh tree-bound evidence | `SELF_IMPROVEMENT_ROOT_CHILD_GOAL_IDS`, `_child_goal_is_current`, and recursive descendant proof revalidation in `goal_completion.evaluate_completion_gate` | `test_root_requires_exact_fresh_tree_bound_child_population` and `test_root_revalidates_every_descendant_proof_requirement` |
| rollout has zero false completion or authority-boundary violations | exact producer/child populations, all-submitted-validation semantics, fixed root criteria, and shared two-phase lifecycle evaluation | producer, validation, coverage, lifecycle, and current-live-state cases in `test_agent_supervisor_self_improvement_e2e.py` |
| Python, CLI, and MCP controls agree | mandatory G070 child population entry retains the G103/G104/G105 parity, authorization, and discovery proof gates and cannot be omitted or replaced by a caller | exact child-population E2E matrix plus the G070 control-contract validation suites |
| a drained board runs bounded evidence-driven refill rather than stopping or creating duplicate busywork. | mandatory G080 child population entry retains G109/G110/G111 observation, materialization, and healthy-exhaustion proof; empty task or child populations never pass | exact child-population and independent-exhaustion E2E matrices plus the G080 refill validation suite |

The E2E positive fixture is deliberately hypothetical. It demonstrates that a
complete fresh packet can move active to provisional and, on a later call,
provisional to verified. It does not describe the current heap. Negative
fixtures remove or corrupt one authority input at a time to prove:

- `tasks_complete=True` cannot conceal an incomplete producer population;
- a coverage summary cannot conceal a missing implementation or detached
  validation receipt;
- a nominally healthy analyzer cannot omit explicit completion safety or its
  full binding;
- a declared quorum cannot conceal duplicate, stale, unhealthy, unsafe,
  non-exhaustive, insufficient, or foreign-bound members;
- a parent cannot conceal a missing, duplicate, reopened, proofless,
  stale-gated, or foreign-tree child; and
- fresh evidence cannot mask a failed or stale submitted sibling.

## Validation observation

The required current-tree command is:

```text
python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py -q
```

Fresh ASI-082 candidate-tree observation on 2026-07-24: **39 passed, 0
failed**. The submitting runner must execute the same command again after all
artifact changes; that final result, rather than this prose, is the
current-tree validation receipt. The command exercises the immutable
populations, positive two-phase route, each fail-closed root gate, export
contract, and the checked-in actionable-state assertions.

This file is an audit and provenance index, not completion evidence. It claims
no final tree-bound criterion receipt, analyzer run, exhaustion vote, or
lifecycle transition. ASI-G000 remains provisionally complete and
supervisor-actionable until ASI-082 and every other original producing task
are terminal-successful; every descendant is verified with fresh, conclusive,
uncontradicted, assurance-sufficient proof; all four current-tree criterion
validations and their implementation mappings pass; analyzer health is
explicitly healthy, completion-safe, and fully bound; two independent fresh
healthy exhaustive receipts pass; and a separate provisional-to-verified
evaluation succeeds.
