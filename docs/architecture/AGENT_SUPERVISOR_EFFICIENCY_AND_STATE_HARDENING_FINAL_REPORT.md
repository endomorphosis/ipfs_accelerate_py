# Agent Supervisor Efficiency and State Hardening final report

Program: `agent-supervisor-efficiency-and-state-hardening-v1`
Task: `ASEH-075`
Schema: `ipfs_accelerate_py/agent-supervisor/aseh-release-report@1`
Interface: `AsehReleaseReport@1`
Identity: `baguqeera5pv363r2jl4twtuzh73aip56nknt7kj5jzrufqam4ia6waoys6rq`
Plan revision: `ASEH-PLAN-R1`
Policy: `ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1`
Disposition: `non_promoted_unmeasured`
Disposition kind: `unmeasured`

This human report reconciles to the machine report at
`docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json`
and to the admitted receipts listed below. Reporting cannot complete the board,
create follow-on tasks, mutate a policy pointer, or authorize promotion.

## Exact commits changed and trees

### ipfs_accelerate_py

- Planning commit `755f45475cc2d13dacd8b330036c1d597afeddde` tree `729da9f8293ecfa046a0136381a3d3808f9ed140`
- Authority-inventory capture commit `7c03c4ded2a80d50fb72130591a521d415386c97` tree `959ac0df9f57f56c759529b747a6a818e6de4fbf`
- Hermetic qualification commit `1d77c3ba0f5498fb0b7615391c58811683301d25` tree `9f8c1b41450a8309f3b7e23dc5932dee62f48b7c`
- Historical qualification commit `a044c001b2d69b11534d8c1def717484c1c12c4f` tree `9d0aab8a8b1269e6158ce129e844ff0a853e1430`
- Live-shadow qualification commit `5b44a281738d4aa1946de3967a29775e8c04448f` tree `e131452f1feeef0335c38b75f0c1c3b89a3a03f5`
- Canary qualification commit `fadcbe90c776698871e380fb7285c03fbdf5977f` tree `d879202f67a9ff9b799fd7f2bffde78141c963ec`
- Implementation envelope tree id `1fa4d8e2bad45ede8d132eb6e077a623c38a1bd4` is not current-head or promotion identity
- Role: supervisor_state_execution_and_admission_authority

### ipfs_datasets_py

- Campaign snapshot commit `209dbe2765593fbc6efe8e9281c34f2e8f6e37a6` tree `95f54df34585d0b736706fd90c83f55954489ad9`
- Additional admitted campaign commit: `false`
- Role: semantic_identity_ir_context_and_formal_obligation_authority

### ipfs_kit_py

- Campaign snapshot commit `ba5508d940fb5b23a6d0d9b2084f5195cd26a671` tree `7c71efa93c4e4124d12fa05515868df3a5344b2e`
- Additional admitted campaign commit: `false`
- Role: durable_bytes_cid_current_root_cas_wal_and_recovery_authority

A moving branch is not the release identity.

## Board completion status

- Markdown board is completion authority: `false`
- Markdown status (observation): `todo`
- Production complete: `false`
- Drain is bookkeeping only: `true`
- This task authority: `reporting_only`

Admitted receipts:

- `ASEH-000` `inventory` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json`
- `ASEH-001` `sealed_not_qualified` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json`
- `ASEH-014` `sealed` — `benchmarks/agent_supervisor/efficiency_state_hardening/historical_manifest.json`
- `ASEH-015` `insufficient_evidence` — `benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json`
- `ASEH-035` `sealed` — `benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_manifest.json`
- `ASEH-045` `bounded_hermetic_qualification` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/state_machine_qualification.json`
- `ASEH-060` `authority_disposition` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/migration_matrix.json`
- `ASEH-062` `current_head_installed_package_contract_qualification` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/cross_repository_qualification.json`
- `ASEH-070` `evidence_qualified` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json`
- `ASEH-071` `insufficient_evidence` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json`
- `ASEH-072` `insufficient_evidence` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json`
- `ASEH-073` `not_admitted` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json`
- `ASEH-074` `non_promoted_unmeasured` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/promotion_decision.json`
- `ASEH-075` `reporting_only` — `docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json`

## Canonical architecture selected

Selected: `true`
ADR: `docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md`
Production cutover deferred to: `ASEH-061`

Handoff: immutable reviewed objectives and task board -> one offline materialization -> IntentRepository@1 / DatabaseTaskSource@1 in DuckDB -> exclusive loopback QuackStateServer@1 typed owner -> configured_board_scheduler.py -> existing multi_supervisor_runner.py -> admitted validators and current-tree merge receipts -> one terminalization and an independent promotion decision

Mutable-fact owners:

- `context`: `ContextPackSelector@1` at `ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack_selector.py`
- `merge`: `PatchAdmission@1` at `ipfs_accelerate_py/agent_supervisor/merge/patch_admission.py`
- `promotion`: `PromotionAdmission@1` at `ipfs_accelerate_py/agent_supervisor/control/promotion_admission.py`
- `receipts`: `TypedStateOwnerCommandGateway@1` at `ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py`
- `recovery`: `SupervisorRecovery` at `ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py`
- `repair`: `DeterministicDoctorSynthesizer@1` at `ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py`
- `reuse`: `ContextPackSelector@1` at `ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack_selector.py`
- `routing`: `ModelRouting@1` at `ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py`
- `task_objective_state`: `TypedStateOwnerCommandGateway@1` at `ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py`

Cross-repository owns/must-not-own boundaries remain those sealed in the cross-repository qualification receipt.

## Duplicate paths deprecated

- `docs/architecture/agent_supervisor_efficiency_state_hardening.todo.md` (`markdown_task_board`) replaces with `IntentRepository@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py` (`AutonomousRepairEngine`) replaces with `DeterministicDoctorSynthesizer@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/autonomy/promotion.py` (`AutonomyPromotionController@1`) replaces with `PromotionAdmission@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py` (`autonomy_repair_controller`) replaces with `DeterministicDoctorSynthesizer@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/context/logic_repair_context.py` (`logic_repair_context`) replaces with `ContextPackSelector@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/context/planner_doctor_context.py` (`planner_doctor_context`) replaces with `ContextPackSelector@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/federation/merge.py` (`federation_merge`) replaces with `PatchAdmission@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/federation/recovery.py` (`federation_recovery`) replaces with `SupervisorRecovery`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py` (`llm_merge_resolver`) replaces with `PatchAdmission@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/objectives/objective_daemon.py` (`objective_daemon_backlog`) replaces with `DatabaseTaskSource@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py` (`provider_fallback_runner`) replaces with `ModelRouting@1`; writable=false; deletion_supported=false
- `ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py` (`DuckDBTaskSource`) replaces with `DatabaseTaskSource@1`; writable=false; deletion_supported=false

Public deletion remains unsupported. Deprecated paths are not executing replacements and cannot independently claim authority.

## Benchmark population sample sizes

- Hermetic sealed fixtures: `72`; pair_count measured 72 count; disposition `evidence_qualified`; live `false`; sufficient for production promotion `false`
- Historical sealed corpus: `28`; pair_count unavailable (not_yet_measured); disposition `insufficient_evidence`
- Live-shadow pair_count unavailable (not_yet_measured); disposition `insufficient_evidence`
- Live cohort count `0` of minimum `10`; disposition `insufficient_evidence`; deadline `2026-10-02T00:00:00Z`
- Canary pair_count unavailable (not_yet_measured); disposition `not_admitted`

## Baseline, current, and candidate token use

Hermetic paired median input-token difference: measured 265 tokens
Hermetic paired mean input-token difference: measured 433 tokens
Hermetic arm totals: unavailable (arm_totals_not_published_in_admitted_receipt)
Live baseline/current/candidate token use: unavailable (not_yet_measured)
Historical: unavailable (not_yet_measured)
Live-shadow: unavailable (not_yet_measured)
Canary: unavailable (not_yet_measured)
Usable for production promotion: `false`

## Baseline, current, and candidate compute use

Hermetic paired median terminal time: measured 22612 seconds_millionths
Hermetic paired mean terminal time: measured 22612 seconds_millionths
Hermetic median time to terminal: measured 92962 seconds_millionths
ContextPack hermetic audit compute units: measured 244204 compute_units
CPU seconds: unavailable (not_published_in_admitted_qualification_receipts)
GPU seconds: unavailable (not_published_in_admitted_qualification_receipts)
Peak memory: unavailable (not_published_in_admitted_qualification_receipts)
Hermetic arm totals: unavailable (arm_totals_not_published_in_admitted_receipt)
Live compute use: unavailable (not_yet_measured)

## Provider cost

Hermetic paired median cost: measured 605 microusd
Hermetic paired mean cost: measured 991 microusd
Hermetic quality-adjusted median candidate cost: measured 2760 microusd
Hermetic audit overhead: measured 10170 microusd
Historical audit overhead: unavailable (not_yet_measured)
Live audit overhead: unavailable (not_yet_measured)
Live weighted provider-cost reduction: unavailable (not_yet_measured)
Net savings after audit overhead: unavailable (not_yet_measured)

## Model-call distribution

Hermetic: unavailable (not_published_in_admitted_qualification_receipts)
Historical: unavailable (not_yet_measured)
Live-shadow: unavailable (not_yet_measured)
Live: unavailable (not_yet_measured)
Canary: unavailable (not_yet_measured)

## Deterministic, small, medium, frontier, and human route shares

- deterministic: unavailable (not_published_in_admitted_qualification_receipts)
- small: unavailable (not_published_in_admitted_qualification_receipts)
- medium: unavailable (not_published_in_admitted_qualification_receipts)
- frontier: unavailable (not_published_in_admitted_qualification_receipts)
- human: unavailable (not_published_in_admitted_qualification_receipts)
- live low-risk deterministic or small-model share: unavailable (not_yet_measured)
- live frontier-call reduction: unavailable (not_yet_measured)

## Test and proof reuse

ContextPack hermetic eligible reuse `1` of `17`; live reuse granted `false`; stale packs admitted `0`; stale packs rejected `7`; critical omissions accepted `0`.
Live eligible ContextPack reuse percent: unavailable (not_yet_measured)
Live test reuse: unavailable (not_yet_measured)
Live proof reuse: unavailable (not_yet_measured)

## Retry and recovery rate

Hermetic retry_rescue class count measured 6 count is not a rate.
Live retry rate: unavailable (not_yet_measured)
Live manual recovery rate: unavailable (not_yet_measured)
Retry token reduction: unavailable (not_yet_measured)

## False-positive and false-negative results

False positives: unavailable (not_published_in_admitted_qualification_receipts)
Hermetic observed selected-test false negatives: `aseh-h26`
Hermetic escaped selected-test false negatives: `(none)`
Live-shadow / historical / canary observed and escaped selected-test false negatives: none recorded.

## Quality results

Hermetic accepted-patch rate: measured 500000 ratio_millionths (`36` of `72`)
Hermetic outlier fixtures: `aseh-h68`
Accepted-patch quality statistically meaningful degradation: unavailable (not_yet_measured)
Usable for production promotion: `false`

## Safety results

Hard-gate violation: `false`
Hermetic escaped critical seeded defects: measured 0 count
Hermetic simulated-as-live outcomes: measured 0 count
Hermetic live cohort: unavailable (fixture_only)
Historical / live-shadow / canary escaped critical seeded defects, simulated-as-live outcomes, and live cohort: unavailable (`not_yet_measured`)
Live cohort present: `false`
Usable for production promotion: `false`

## Promotion status

Exact disposition `non_promoted_unmeasured` / `unmeasured` from ASEH-074 identity `baguqeerasbt75apwn2fj7urqpuesz44sfyxlzvhsl7kddfvbche3jx74wwoa`.
Eligible for operator authorization: `false`
Promotion authorized: `false`
Self-authorized: `false`
Policy pointer mutated: `false`
Reasons: `absent_live_cohort`, `historical_replay_insufficient_evidence`, `live_shadow_insufficient_evidence`, `canary_not_admitted`, `historical_audit_overhead_unmeasured`, `live_audit_overhead_unmeasured`, `live_safety_unmeasured`, `live_efficiency_unmeasured`

## Limits

- Deferred ideas non-executing: `true`
- Follow-on tasks created: `false`
- Hermetic cannot satisfy live or promotion: `true`
- Missing measurements recorded as zero: `false`
- Policy pointer mutation: `false`
- Reporting cannot authorize promotion: `true`
- Thresholds lowered: `false`

Named unavailable fields include:

- `baseline_current_candidate_compute_use.canary`
- `baseline_current_candidate_compute_use.cpu_seconds`
- `baseline_current_candidate_compute_use.gpu_seconds`
- `baseline_current_candidate_compute_use.hermetic.arm_totals.direct_minimal_orchestration_baseline`
- `baseline_current_candidate_compute_use.hermetic.arm_totals.sealed_current_supervisor_baseline`
- `baseline_current_candidate_compute_use.hermetic.arm_totals.candidate_optimized_supervisor`
- `baseline_current_candidate_compute_use.historical`
- `baseline_current_candidate_compute_use.live`
- `baseline_current_candidate_compute_use.live_shadow`
- `baseline_current_candidate_compute_use.peak_memory`
- `baseline_current_candidate_token_use.canary`
- `baseline_current_candidate_token_use.hermetic.arm_totals.direct_minimal_orchestration_baseline`
- `baseline_current_candidate_token_use.hermetic.arm_totals.sealed_current_supervisor_baseline`
- `baseline_current_candidate_token_use.hermetic.arm_totals.candidate_optimized_supervisor`
- `baseline_current_candidate_token_use.historical`
- `baseline_current_candidate_token_use.live`
- `baseline_current_candidate_token_use.live_shadow`
- `benchmark_population_sample_sizes.canary.pair_count`
- `benchmark_population_sample_sizes.historical.pair_count`
- `benchmark_population_sample_sizes.live_shadow.pair_count`
- `deterministic_small_medium_frontier_and_human_route_shares.frontier_model_call_reduction_percent.observed`
- `deterministic_small_medium_frontier_and_human_route_shares.hermetic.deterministic`
- `deterministic_small_medium_frontier_and_human_route_shares.hermetic.small`
- `deterministic_small_medium_frontier_and_human_route_shares.hermetic.medium`
- `deterministic_small_medium_frontier_and_human_route_shares.hermetic.frontier`
- `deterministic_small_medium_frontier_and_human_route_shares.hermetic.human`
- `deterministic_small_medium_frontier_and_human_route_shares.historical`
- `deterministic_small_medium_frontier_and_human_route_shares.live`
- `deterministic_small_medium_frontier_and_human_route_shares.live_shadow`
- `deterministic_small_medium_frontier_and_human_route_shares.low_risk_deterministic_or_small_model_share_percent.observed`
- `false_positive_and_false_negative_results.false_positives`
- `model_call_distribution.canary`
- `model_call_distribution.hermetic`
- `model_call_distribution.historical`
- `model_call_distribution.live`
- `model_call_distribution.live_shadow`
- `provider_cost.audit_overhead.canary`
- `provider_cost.audit_overhead.historical`
- `provider_cost.audit_overhead.live`
- `provider_cost.audit_overhead.live_shadow`
- … 27 additional unavailable fields

## Residual risks

- `absent_live_cohort` (blocks_promotion): No live cohort is enrolled; live measurement remains unavailable.
- `historical_replay_unmeasured` (blocks_promotion): A sealed historical corpus exists but paired replay is unavailable on the current tree.
- `canary_not_admitted` (blocks_promotion): Canary mutation is not admitted because shadow evidence is not qualified.
- `live_efficiency_unmeasured` (blocks_promotion): Live token, cost, frontier, retry, route-share, reuse, and recovery thresholds remain unmeasured.
- `hermetic_selected_test_false_negative` (observed_not_escaped): Hermetic fixture aseh-h26 is an observed selected-test false negative and did not escape.
- `production_cutover_deferred` (residual): Owner-paused staged production cutover remains deferred to ASEH-061.
- `markdown_is_not_authority` (residual): The protected Markdown board remains todo and cannot complete, drain, or promote work.

## Highest-return next work

Automatically executed: `false`
May create follow-on tasks: `false`

- `enroll_live_cohort` executing=false: Enroll at least 10 distinct live tasks before the sealed 2026-10-02T00:00:00Z deadline.
- `replay_historical_corpus` executing=false: Replay the sealed 28-vector historical corpus on the current tree until pair_count is measured.
- `shadow_then_canary` executing=false: Run live-shadow to qualification, then a separately gated low-risk canary only if shadow admits it.
- `measure_live_efficiency` executing=false: Measure live token, cost, frontier, retry, route-share, reuse, recovery, and audit-inclusive net savings against the sealed thresholds.
- `operator_cas_promotion_only_after_evidence` executing=false: Leave policy-pointer mutation to an operator-authorized CAS action after an eligible disposition; this report must not perform it.

These items are residual-gap guidance only. They are not executing work, not a backlog mutation, and not a promotion.

## Nonclaims

- This report does not mutate a policy pointer.
- This report does not authorize promotion.
- This report does not create follow-on tasks.
- Hermetic evidence is not live and cannot satisfy production promotion.
- Missing measurements are unavailable, never numeric zero.
- Markdown board status is an observation and is not completion authority.
- Deferred highest-return work is non-executing.
