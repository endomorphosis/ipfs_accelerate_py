# ASI-083 completion-gate evidence map

- Date: 2026-07-25
- Task: ASI-083
- Goal: ASI-G060 — Adaptive parallel execution and acceptance throughput
- Parent: ASI-G000
- Child goals: none
- Requirement IDs: `122080003600146794820964010047426915846`, `124037811551945145648172208272779822741`, `185033715568272291470322170325431455647`
- Source gap fingerprint: `df2b5b9186e148afdf4afa8a8c8a75a447fe01b4`
- Evidence obligation: `objective-work/v1/e0d8763c9c31e15fe8579faddc628167717f936e`
- Todo vector: `e0d8763c9c31e15f`
- Merge family: ASI-G060
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- Lifecycle disposition: provisionally complete and supervisor-actionable

## Gap disposition

ASI-G060 has one immutable completion boundary. Its direct producing-task
population is exactly ASI-015, ASI-016, and ASI-017, and every producer must be
terminal-successful before completion can be requested. ASI-083 supplies the
parent gate and is deliberately not a producer in its own decision. ASI-042 is
the completed canonical aggregate and routing record; ASI-036 and ASI-037 are
retired aliases. Resource scheduling, provider batching, and parallel
acceptance are producer lanes rather than separately completable child goals,
so another child partition would add no independent proof boundary.

The five literal acceptance criteria are fixed by code. Every submitted
validation participates in the decision, must be fresh, passing, and bound to
the current repository tree, and must be the single receipt named by exactly
one criterion coverage row. Every row must also name a concrete implementation
binding. A failed, stale, foreign, duplicate, unmapped, or caller-substituted
receipt makes the parent packet non-completing even when another sibling is
valid.

Analyzer health and exhaustion are separate proof inputs. Analyzer evidence
must say both `healthy` and `safe_for_completion_reasoning` and carry the full
repository, tree, ASI-G060, `ASI-G060@asi-083`, analyzer-version, and
configuration-revision binding. Scheduler telemetry, provider health,
validation output, merge receipts, and this Markdown file cannot substitute
for analyzer evidence. The trusted exhaustive count is exactly two. Both
members must be fresh, healthy, completion-safe, exhaustive, identically bound,
and unique by member ID, evidence channel, and receipt CID; callers cannot
lower the count.

The shared lifecycle remains two phase: a first closed evaluation may move
`active` to `provisionally_complete`, and only a later separate closed
evaluation may move it to `verified_complete`. Invalidating any proof input
reopens a verified goal.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation binding | Focused validation proof |
|---|---|---|
| Resource pools expose backpressure and fair admission | `ResourceScheduler.adaptive_stage_capacity`, `_fair_requirements`, `evaluate`, `reserve`, `ResourceAdmissionLease`, `ResourceScheduleSnapshot`, and `metrics_snapshot`/`AdaptiveResourceMetrics` expose live capacity, bounded admission debt, round-robin service, critical-path priority with aging, aggregate resource accounting, cancellation release, and recovery. | `test_resource_pools_expose_fair_order_and_backpressure`, `test_adaptive_admission_round_robins_stages_and_exports_lane_metrics`, the pressure/recovery/starvation/cancellation/aggregate-overadmission/backpressure-event tests, and the ASI-G060 parent packet matrix. |
| compatible provider work shares model capacity | `ProviderBatchKey`, `ProviderBatchScheduler.execute_many`, `partial_cancellation_evidence`, `ProviderBatchEvidenceReceipt`, and `ResourceSchedulerBatchAdmission` share only compatible work while retaining bounded capacity and member-local cancellation, budget, provenance, result, and terminal authority. | Compatible-sharing, partial-cancellation, single-flight, concurrency/capacity, admission/release, fallback, and round-robin fairness tests in `test_agent_supervisor_provider_batch_scheduler.py`, plus the parent criterion map. |
| independent validation and merge preflight run concurrently | `ValidationScheduler._run_parallel_stage` measures bounded DAG execution while `MergeTrain.drain_parallel` runs non-mutating preflights concurrently. | `test_validation_lane_reports_measured_parallel_throughput` and `test_parallel_preflights_keep_mutation_serial_and_gate_every_completion`. |
| target-branch mutation remains fenced and serialized | `MergeTrain._consumer_lease`, synthesized-tree post-merge validation, compare-and-swap mutation, `ParallelAcceptanceReceipt`, and `MergeQueue` debt and claim fences retain one repository mutation authority. | The synthesized-tree-before-CAS, serial mutation, stale-target repair, conflict-ordering, foreign-receipt, restart, and merge-debt tests in `test_agent_supervisor_parallel_acceptance_flow.py`. |
| paired independent fixtures achieve at least twice single-lane throughput without duplicate execution, stale acceptance, resource overcommit, or merge-conflict regression | `AdaptiveThroughputBenchmarkReceipt` and `benchmark_adaptive_execution` revalidate identical fixture populations and policy/resource bounds; provider and parallel-acceptance receipts independently close cancellation, stale-target, and final-gate failure modes. | `test_benchmark_runner_proves_two_x_without_duplicates_or_overcommit`, the fail-closed adaptive benchmark matrix, provider partial-cancellation/capacity tests, merge synthesized-tree and final-gate tests, and the closed parent packet test. |

## Backlog and lifecycle alignment

The objective heap records the evaluator and this audit index but does not
claim that documentation is a receipt. The supervisor-fed todo, runtime shard,
todo-vector index, generated objective data, status, confidence, analyzer JSON,
quorum JSON, and transition timestamps are not edited by ASI-083. Normal
supervisor regeneration reconciles those projections after the implementation
daemon records a fresh run.

ASI-G060 and ASI-G000 therefore remain provisional and actionable until a
post-change evaluation supplies terminal producer state, one fresh passing
current-tree receipt and exact implementation mapping for every criterion,
fully bound healthy completion-safe analyzer evidence, and two independent
fresh healthy exhaustive receipts. Verified completion additionally requires
the later, separate lifecycle evaluation.

## Validation

The mandatory command is:

```text
python -m pytest test/api/test_agent_supervisor_adaptive_resources.py test/api/test_agent_supervisor_provider_batch_scheduler.py test/api/test_agent_supervisor_parallel_acceptance_flow.py -q
```

This file is a provenance and audit index, not final-tree validation,
analyzer-health, exhaustion-quorum, or lifecycle authority. The submitting
runner's fresh passing post-change execution is the validation receipt.
