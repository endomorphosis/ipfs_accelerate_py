# ASI-080 Evidence-aware Planning Completion-Gate Evidence Map

- Date: 2026-07-24
- Task: ASI-080
- Goal: ASI-G030 — Evidence-aware planning and responsive goal refinement
- Parent: ASI-G000
- Requirements: `173075880069453142914839090434430341799`, `003778425160038348524906247302938706902`, `312819945606360295782005228058369235550`
- Source gap fingerprint: `95ff2e6c9e3e4e53e69056e51159ca5b2a1c1da9`
- Evidence obligation: `objective-work/v1/941b45b4a475a78a114da4e82944d2023025eaab`
- Todo vector: `941b45b4a475a78a`
- Merge role: `completion_gate`
- Lifecycle: provisionally complete and supervisor-actionable

## Gap disposition

1. Producing-task closure is exact. The parent requires ASI-008 and ASI-009
   once each in a successful terminal state in addition to the caller's
   `tasks_complete=True`; an omitted, duplicate, unexpected, or incomplete
   task keeps the parent actionable.
2. Descendant closure is exact. ASI-G097, ASI-G098, and ASI-G115 must all be
   `verified_complete` with passed gates bound to the current tree. Every
   aggregated proof requirement must remain fresh, proved, conclusive,
   uncontradicted, and satisfied at its required assurance.
3. The parent acceptance population is the four literal ASI-G030 clauses.
   Callers cannot narrow it. Every submitted validation participates in the
   decision, every clause needs a fresh passing current-tree receipt, and
   every coverage row must bind its concrete implementation to that clause's
   submitted validation receipt identity.
4. `EvidenceAwarePlanningCompletionEvidence` binds the complete routed plan
   population and its seven-dimension evaluations, one qualifying
   `NewCounterexampleRefinementEvidence` admission receipt, and one qualifying
   `UnchangedFailureBackoffEvidence` plus its exact persisted source failure.
   All records use one repository tree and expose the exact three requirement
   IDs. The cohort is content-addressed but explicitly has no completion
   authority and is unsafe for completion reasoning.
5. Analyzer health is separate and explicit. It must state `healthy=True` and
   `safe_for_completion_reasoning=True` and bind the repository, tree,
   objective/revision, analyzer version, and configuration revision. Planner,
   refiner, provider, formal-repair, and routing outputs cannot substitute.
6. Exhaustion is configured and independent. ASI-G030 fixes the count at two
   and requires unique member, evidence-channel, and receipt identities;
   exhaustive, healthy, completion-safe, fresh semantics; and the exact same
   binding as analyzer health. A caller cannot lower the trusted count.
7. Completion remains two phase. A first passing evaluation may only move
   active to provisional; verified completion requires a later separate
   evaluation while every receipt and descendant proof is still valid.

`ResponsiveReplanDecision` remains routing-only. Its serialized
`completion_evidence_roles` is empty and both `completion_authority` and
`safe_for_completion_reasoning` are false, so it cannot replace any runtime
producer witness or completion proof class.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| Every plan is evaluated for acceptance coverage, assumptions, semantics, dependencies, conflicts, validation/proof feasibility, novelty, and resource/token cost | Exact seven-member `PlanEvaluationDimension` assessment for every routed candidate, deterministic recomputation, and `AdaptivePlanningRunReceipt` population equality | `test_every_quality_dimension_is_evaluated_and_weighting_is_deterministic` and `test_g030_parent_completion_requires_all_producers_and_fresh_descendants` |
| hard safety failures cannot be traded away | Independent authority/scope/safety/proof receipts and non-compensable hard gates in `adaptive_planner.py` and `plan_evaluator.py`; formal repair metadata is bound but non-authoritative | cheaper-authority, other-hard-failure, adversarial-quality, gate-replay, and G030 parent-matrix tests |
| unchanged failures back off | Semantic repeated-failure identity, persisted failed source, finite retry deadline, and causal `UnchangedFailureBackoffEvidence` in `adaptive_goal_refiner.py` | no-second-call, JSONL restart, changed/deadline, causal tamper, routing-boundary, and G030 parent-matrix tests |
| changed evidence can trigger a bounded verified refinement in the next cycle without mutating the frozen root. | Root/cycle transaction lock, policy bounds, exact changed-goal and verifier binding, and content-addressed `NewCounterexampleRefinementEvidence` | new-counterexample, per-cycle/concurrency, frozen-root/tree, verifier replay/type, receipt tamper, and G030 parent-matrix tests |

The parent matrix also proves that an incomplete producing-task population,
missing descendant, stale descendant proof, unbound validation mapping, unsafe
analyzer, duplicate exhaustion receipt, caller-lowered quorum, or tampered
operational cohort cannot verify ASI-G030. Invalidating a verified parent
reopens it.

## Validation observation

The required current-tree command is:

```text
python -m pytest test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
```

The implementation run collected 79 tests and passed all 79 with no failures.
That post-change run, rather than this prose index, is the validation receipt.

This file is an audit index, not completion evidence. It claims no final
repository-tree identity, analyzer execution, exhaustion vote, or lifecycle
transition. ASI-G030 and ASI-G000 remain actionable until the supervisor
ingests fresh passing current-tree receipts for all four criteria, fully bound
healthy analyzer evidence, the configured two independent fresh exhaustive
receipts, and fresh conclusive proof for ASI-G097/G098/G115, then performs the
separate provisional-to-verified evaluation.
