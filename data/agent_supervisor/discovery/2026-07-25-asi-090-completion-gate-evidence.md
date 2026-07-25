# ASI-090 completion-gate evidence map

- Date: 2026-07-25
- Task: ASI-090
- Goal: ASI-G090
- Parent: ASI-G000
- Producing tasks: ASI-023, ASI-024
- Child goals: ASI-G112, ASI-G113, ASI-G114
- Requirements: `109590900757783560279417463762322084165`, `146189916032404266364029134505159070240`, `300500866741873729474343907613893393545`
- Source gap fingerprint: `075ea1951c0b4657bae0a00df5e516d8024e076a`
- Evidence obligation: `objective-work/v1/bef0ffbeecaeda580a753344fcab98755066d821`
- Todo vector: `bef0ffbeecaeda58`
- Merge family: ASI-G090
- Merge role: `completion_gate`
- Work scope: `bounded_objective_generation`
- State after this implementation: `provisionally_complete` and supervisor-actionable

## Gap disposition

ASI-090 adds the missing closed parent completion boundary without allowing
this task, this document, a detached report, or operator prose to authorize
ASI-G090 completion:

1. The producing population is exactly ASI-023 and ASI-024. A caller's
   `tasks_complete=True` summary cannot mask an absent, duplicate, substituted,
   or nonterminal producer.
2. The direct descendant population is exactly ASI-G112, ASI-G113, and
   ASI-G114. Each child must remain freshly `verified_complete` with a passing
   current-tree gate, validation evidence, and a conclusive, satisfied,
   uncontradicted current proof requirement.
3. The paired report is re-derived from all twelve typed fixtures and must be
   fresh and pass every non-negotiable, paired, token, cache, planning, and
   throughput gate. Its G112/G113 projections must be strictly restored,
   satisfied, unique, and bound to the current repository/tree. They are
   operational prerequisites, not parent validations or analyzer authority.
4. The five literal parent clauses are closed. Exactly one fresh passing
   current-tree validation is submitted for each, and each coverage row names
   concrete implementation plus the exact receipt identity. Every submitted
   record participates.
5. Analyzer health is explicit and fixed to ASI-G090 revision
   `ASI-G090@asi-090`, `paired-rollout-completion@1`, and
   `paired-rollout-completion-policy@1`, with complete repository/tree
   binding and `safe_for_completion_reasoning=true`.
6. The configured quorum is exactly two. Both members are fresh, healthy,
   completion-safe, exhaustive, identically bound, and independent by member
   ID, evidence channel, and receipt CID. The caller cannot lower this count.
7. A closed pass from `active` advances only to `provisionally_complete`.
   Verified completion requires a separate later evaluation while the entire
   producer, report, projection, criterion, coverage, analyzer, quorum, and
   child population remains valid.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| Paired cold/warm, failure, adversarial, parallel, restart, and refill fixtures satisfy every non-negotiable safety gate and the documented token/cache/planning/throughput gates | `REQUIRED_PAIRED_FIXTURE_KINDS`, `PairedRolloutPolicy`, `evaluate_paired_self_improvement_rollout`, component gate projections, strict report restoration | `test_closed_paired_population_passes_every_asi_023_gate`, `test_g090_fixture_families_share_every_required_gate_and_safety_invariant`, `test_each_paired_regression_independently_forces_shadow`, `test_paired_rollout_gate_survives_process_restart_without_state_drift`, `test_seeded_false_completion_proves_shadow_blocking_only_for_closed_population` |
| optional integrations degrade correctly | Provider-unavailable outcomes are limited to degraded/fallback/rejected; package lazy discovery resolves no optional provider | `test_fault_fixtures_must_fail_closed_and_restart_must_be_stable`, `test_stable_rollout_exports_remain_lazy_without_optional_providers` |
| stable exports remain lazy | Package-owned lazy-export requirement/goal, exact immutable manifest, `_LAZY_STABLE_EXPORTS`, and `__getattr__` | `test_stable_rollout_exports_remain_lazy_without_optional_providers`, `test_benchmark_uses_complete_stable_package_root_rollout_surface` |
| operators have verified smoke and production profiles | Deterministic smoke recipe and production go/no-go checklist require fresh preflight, complete v2 report, restored projections, exact bindings, persisted reload, and separate authorization/completion | `test_operator_profiles_document_the_g090_completion_contract` and the mandatory full e2e/benchmark run |
| failed gates retain shadow mode and produce bounded diagnostics. | Effective-mode coercion, stable bounded reason codes, artifact/report limits, and append-only store | `test_nonnegotiable_violation_always_forces_shadow`, `test_each_paired_regression_independently_forces_shadow`, `test_candidate_artifact_aggregate_is_hard_bounded`, `test_any_end_to_end_authority_failure_keeps_candidate_in_shadow` |

The G090 completion tests also cover missing producers and descendants,
incomplete criterion evidence, detached coverage, unsafe analyzer data,
duplicate quorum identities, missing report projections, stale/failed reports,
caller-lowered quorum, and the mandatory two-phase lifecycle.

## Validation route

```text
python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py test/api/test_agent_supervisor_self_improvement_benchmark.py -q
```

This file is an audit and provenance index, not a completion receipt. It
claims no final post-change tree identity, validation execution, analyzer run,
exhaustion vote, descendant transition, or lifecycle transition. The
submitting runner must execute the command after all changes. The supervisor
must keep ASI-G090 and ASI-G000 actionable until the exact producers are
terminal; every criterion has fresh passing current-tree validation and
receipt-bound implementation coverage; every exact child remains freshly
verified; the analyzer is explicitly healthy and completion-safe; two
independent fresh healthy exhaustive receipts pass; and a separate
provisional-to-verified evaluation succeeds.
