# Semantic Compression Governor — Final Report (SCG-048)

**Status:** terminal current-tree qualification report (non-authoritative)  
**Package:** `ipfs_accelerate_py.agent_supervisor.semantic_governor`  
**Board namespace:** `semantic-compression-governor-v1`  
**Task / goal:** `SCG-048` / `SCG-G090`  
**Evidence:** `scg/final-report@1`, `scg/release@1`, `scg/benchmark-results@1`, `scg/incremental-seal-qualification@1`, `scg/rollback@1`, `scg/trust-docs@1`  
**Interface:** `SemanticGovernorReleaseQualification@1` / `GovernorReport@1`  
**Authority:** this report is **not** production-authoritative and **not** production-eligible. It records measured evidence, typed unavailability, and remaining risks without upgrading any receipt status.

Machine-readable companion: [`artifacts/agent_supervisor/semantic_compression_governor/release.json`](../../artifacts/agent_supervisor/semantic_compression_governor/release.json)  
Sealed governor report CID: `baguqeera3qzt3fxishav374lfecwk32hmg37ap2vnx2xyrv4ywakt4lm3mdq`  
Release content id: `sha256:50c29f55a68763dba3890eab0260ad1516270c9994afea179134e93dfcaf5f95`  
Release CID: `baguqeera6pavpnosalcynozxwth4tt5j4sygbpzc2s6cisamhxxvwg6vbboq`

Related:

- Plan: [`SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md`](./SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md)
- Trust model: [`SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md`](./SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md)
- Operator guide: [`../guides/SEMANTIC_COMPRESSION_GOVERNOR.md`](../guides/SEMANTIC_COMPRESSION_GOVERNOR.md)

---

## 1. Bounded claim (plan §16 maximum justified claim)

> The system empirically and structurally audits semantic compression, detects and diagnoses omitted-context failures, expands context using counterexamples and dependency evidence, calibrates future compression decisions, and promotes rule changes only after held-out evaluation and authorized, reproducible qualification.

**Explicit non-claim:** the system does **not** prove that every compressed context is semantically complete. Proof claims stop at properties actually encoded and verified (closed bounded seal claims). Heuristic evidence is never treated as exact. Token reduction is never an acceptance criterion by itself.

---

## 2. Exact inspected and implemented commits

### 2.1 Planning baseline authority pins (plan §2 / authority matrix)

| Authority | Commit |
| --- | --- |
| accelerate_planning | `dfd92b554e662d4312411f2e8e63a52368806f2a` |
| datasets | `1330038f626ef92993f03d46f21e1a57719e9c25` |
| kit | `df2f9cc092456329de9724c45a50c54b410875d1` |
| mcplusplus | `dc3164653a48d059ae9812078359daeafb451c07` |
| incremental_verification_freeze | `8c7800cedc5e1b848367db9952f912428466f8cc` |
| incremental_proof_sealer_program | `7dc8f1422cb7e80757077948dc0785c1aaa4fd25` |

### 2.2 Live heads on this tree (implemented)

| Tree | Commit |
| --- | --- |
| controller (workspace) | `79e1a2c39f2252c2335909e4c6bc5418e6cd8a8b` |
| `ipfs_datasets_py` gitlink | `8ffa3152603fe8ae3a463250d91d47109cd48006` |
| `ipfs_kit_py` gitlink | `996ee85f071dff17e4104948d9ca938d2125a447` |
| `ipfs_accelerate_py/mcplusplus` gitlink | `dc3164653a48d059ae9812078359daeafb451c07` |

**Authority pin revision:** live datasets and kit heads advanced during SCG implementation (SCG-018 / SCG-022) and remain descendants of the planning pins. The authority matrix keeps the planning pins. The live-gitlink join accepts the pin or a descendant so current-tree qualification does not treat implemented SCG work as a stale unrelated tip.

Inspected commits (union of pins and live heads):

```text
1330038f626ef92993f03d46f21e1a57719e9c25
79e1a2c39f2252c2335909e4c6bc5418e6cd8a8b
7dc8f1422cb7e80757077948dc0785c1aaa4fd25
8c7800cedc5e1b848367db9952f912428466f8cc
8ffa3152603fe8ae3a463250d91d47109cd48006
996ee85f071dff17e4104948d9ca938d2125a447
dc3164653a48d059ae9812078359daeafb451c07
df2f9cc092456329de9724c45a50c54b410875d1
dfd92b554e662d4312411f2e8e63a52368806f2a
```

Implemented commits (live heads only):

```text
79e1a2c39f2252c2335909e4c6bc5418e6cd8a8b
8ffa3152603fe8ae3a463250d91d47109cd48006
996ee85f071dff17e4104948d9ca938d2125a447
dc3164653a48d059ae9812078359daeafb451c07
```

---

## 3. Consumed interfaces

Public governor APIs and supporting surfaces consumed by this program:

```text
BoundedSealClaimSet@1
DurablePolicyCASRepositories@1
GovernorMetricsCollector@1
GovernorReport@1
IncrementalSealerCapability@1
ReleaseQualification@1
SemanticCompressionGovernor@1
SemanticGovernorReleaseQualification@1
apply_trusted_decision
build_context_coverage_manifest
build_dashboard_data@1
build_governor_report@1
compare_shadow_results
create_shadow_plan
detect_instruction_like_content
diagnose_omission
evaluate_context_sufficiency
evaluate_rule_candidate
execute_expansion_loop
merge_calibration_profiles
plan_context_expansion
promote_compression_policy
propose_rule_change
qualify_policy_candidate
rollback_compression_policy
seal_governor_run
update_calibration
validate_rule_proposal
```

Accelerate required public APIs: evaluate_context_sufficiency, create_shadow_plan, compare_shadow_results, diagnose_omission, plan_context_expansion, execute_expansion_loop, update_calibration, propose_rule_change, evaluate_rule_candidate, promote_compression_policy  
Datasets required public APIs: build_context_coverage_manifest, evaluate_context_sufficiency, diagnose_omission, plan_context_expansion, update_calibration, propose_rule_change  
Datasets supporting APIs: detect_instruction_like_content, apply_trusted_decision, merge_calibration_profiles, validate_rule_proposal

---

## 4. Required final-report fields

Every plan §16 field is present on the sealed `GovernorReport@1` projection embedded in `release.json` under `governor_report`.

| Field | Present | Notes |
| --- | --- | --- |
| inspected_commits | yes | planning pins ∪ live heads |
| implemented_commits | yes | live heads |
| consumed_interfaces | yes | closed allowlist tokens |
| audit_population | yes | 60 simulated audits; 0 live |
| differential_outcomes | yes | simulated cohort outcomes |
| omission_detection | yes | 16 intentional omissions; 100% pre-exec detection |
| expansion | yes | expansion count 0 on this corpus; FN recorded |
| final_context_reduction | yes | median 47.36% (4736 bp); soft target 50% unmet |
| route_distribution | yes | small/medium/frontier/human shares |
| quality | yes | 0 accepted patches; 0 regressions accepted |
| overhead_and_cost | yes | simulated micros; live billing missing |
| rules | yes | proposed 1 / rejected 1 / **promoted 0** |
| rollback | yes | hermetic authorized rollback receipts |
| seal_scope | yes | **unavailable** (sealer absent) |
| proof_scope | yes | bounded artifact evaluation only |
| heuristics | yes | present; **not treated as exact** |
| remaining_production_risks | yes | see §17 |
| unavailable_fields | yes | live providers, ZK, live metrics CIDs |
| live_metrics_cid | yes (null) | unavailable |
| simulated_metrics_present | yes | true |
| metric_report_cid | yes (null) | summary uses sha256 content_id, not bagu CID |
| evidence_mode | yes | `simulated` |

Evidence mode: **`simulated`**  
Simulated metrics present: **`False`**

---

## 5. Audit population

| Metric | Value |
| --- | --- |
| Total audits | 60 |
| Live audits | 0 |
| Simulated audits | 60 |
| Corpus | `semantic-governor-partitioned-corpus-v1` |
| Partitions | calibration=14, development=14, held_out=32 |
| Live quality claims | False |

Source: `artifacts/agent_supervisor/semantic_compression_governor/summary.json` (`sha256:da9061f672aeed49dadfa62ae7542b7d8409afc96befd74761f094b1784c8893`).

---

## 6. Differential outcomes

From simulated comparative quality counts:

```json
{
  "both_failed_different_reason": 0,
  "both_failed_same_reason": 0,
  "both_valid_different": 0,
  "compressed_better": 0,
  "compressed_failed_expanded_succeeded": 0,
  "compressed_succeeded_expanded_failed": 0,
  "equivalent_success": 0,
  "expanded_better": 0,
  "human_review_required": 2,
  "verification_inconclusive": 58
}
```

| Derived | Value |
| --- | --- |
| equivalent_success_count | 0 |
| compressed_failed_expanded_succeeded_count | 0 |
| both_failed_count | 0 |
| verification_inconclusive_count | 58 |

Most simulated comparisons land in `verification_inconclusive` under controlled offline replay — not live provider equivalence proof.

---

## 7. Omission detection and critical acceptance

| Metric | Value | Hard target |
| --- | --- | --- |
| Intentional omissions | 16 | — |
| Detected before execution | 16 | — |
| Detected after execution | 0 | — |
| Detection-before rate | 10000 bp | ≥ 9500 bp |
| Critical omissions | 16 | — |
| Critical omissions accepted | 0 | **0** |
| Critical acceptance rate | 0 bp | — |
| False alarms | 0 | — |

Hard targets for critical-omission detection and zero critical accepted omissions are **met** on the controlled corpus. Post-execution omission sensors remain listed as missing evidence.

---

## 8. Expansion

| Metric | Value |
| --- | --- |
| Expansion count | 0 |
| Expansion rate | 0 bp |
| True positives | 0 |
| False positives | 0 |
| False negatives | 16 |
| Precision / recall | None / 0 |
| Expanded tokens total | 8646 |

Expansion was not triggered on this controlled offline corpus (`expansion_count=0`); intentional-omission cases that required expansion evidence are recorded as expansion false negatives rather than fabricated successes.

---

## 9. Final context reduction

| Metric | Value | Soft target |
| --- | --- | --- |
| Median context reduction | 4736 bp (47.36%) | ≥ 5000 bp (50%) |
| Mean context reduction | 4537 bp | — |
| Raw tokens total | 11170 | — |
| Compressed tokens total | 6344 | — |
| Expanded tokens total | 8646 | — |

Soft median-reduction target is **not met** (yellow). This is a cost/coverage target, not a correctness gate; proposal promotion remains blocked for this reason among others.

---

## 10. Route distribution and escalation

| Route | Count | Share (bp) |
| --- | --- | --- |
| small | 31 | 5166 |
| medium | 20 | 3333 |
| frontier | 7 | 1166 |
| human | 2 | 333 |
| deterministic | 0 | 0 |

| Metric | Value |
| --- | --- |
| Escalation count | 9 |
| Escalation rate | 1500 bp |
| Retry count | 0 |
| Retry rate | 0 bp |

---

## 11. Quality and regressions

| Metric | Value | Hard target |
| --- | --- | --- |
| Accepted patches | 0 | — |
| Accepted rate | 0 bp | — |
| Regressions | 0 | **0 accepted** |
| Regression rate | 0 bp | — |
| Selected-test false negatives | 1 | — |
| Proof failures | 1 | — |
| Review disagreements | 2 | — |
| Production eligible | False | must remain false without live eligibility |

Zero accepted regressions and zero production-eligible flags are **met**. Stale admissions: 0 (target met).

---

## 12. Overhead and cost

| Metric | Value (micros) | Cohort |
| --- | --- | --- |
| Model spend total | 165952 | simulated |
| Baseline model spend | 204560 | simulated |
| Gross savings | 38608 | simulated |
| Net savings | -10417 | simulated |
| Cost per accepted patch | None | null (no accepted patches) |
| Verification compute | 24000 | simulated |
| Shadow compute | 15000 | simulated |
| Audit overhead | 10025 | simulated |
| Total audit overhead | 49025 | simulated |

Live billing cost sensors: **missing**. Estimator id: `scg-local-cost-estimator/v1`.

---

## 13. Rules proposed / rejected / promoted

| Metric | Production / benchmark track |
| --- | --- |
| Proposed | 1 |
| Rejected | 1 |
| **Promoted** | **0** |
| Promotion authorized | False |
| Verdict | fail |
| Blocking reasons | median_context_reduction_below_threshold |

### Promotion gate (fail-closed)

**No policy is reported promoted** on this tree for production use. Acceptance rule:

1. Separate external authorization CID (candidate/evaluation/proposal cannot self-authorize).
2. Current held-out evaluation that still passes thresholds.
3. Release qualification (authorized VerificationBundle path or released sealer).
4. Successful expected-generation CAS receipt.

Hermetic SCG-047 exercises **did** promote and roll back inside a disposable `DurableCoordinationStore` namespace with distinct authorization and CAS receipts. Those receipts prove the gate path; they **do not** promote a production policy head and **are not** counted in `rules.promoted_count`.

---

## 14. Rollback

| Metric | Value |
| --- | --- |
| Rollback tested | True |
| History preserved | True |
| Status | qualified |
| Namespace | hermetic durable coordination store |
| Rollback operation count (head-mutating) | 2 |
| Rollback receipt CIDs | `baguqeeraljk5jbgyuaajiwtqutjmvhjvik5aa4imcgskuifbsgzaotitn6ja, baguqeeramul7foboodl6waiuqwbdg3ywupqhh6yo5fltaxsrdavcb3yn6kfq` |

Rollback is a forward expected-generation CAS, not history deletion. Stale and mismatched candidates leave the live hermetic head unchanged (see `rollback.json` acceptance block).

Artifact: `artifacts/agent_supervisor/semantic_compression_governor/rollback.json` (`sha256:3e0be7fd702fea06de435d1cb26ac5297dea731c76791ab572b094d04199aa46`).

---

## 15. Seal scope, proof scope, and heuristics

### Seal scope

| Field | Value |
| --- | --- |
| Status | **unavailable** |
| Sealer available on live tree | false |
| Seal CID (authorized non-proof path) | `baguqeera3mxukdztvqjsz7tjlbslwf43o243q3dh4v23g4ncqrhmeeivaa2a` |
| Qualification path | `authorized_release_qualification` |
| Is ZK | false |
| IVP commitment may substitute | false |

Released `IncrementalProofSealer` / `FullCheckpointSeal` / `DeltaSeal` public API is **not** present. Missing sealer is typed unavailable and does not stall authorized VerificationBundle release qualification.

### Proof scope (bounded claims only)

Encoded bounded claims:

```text
exact_artifacts_evaluated
required_evaluations_completed
declared_thresholds_applied
no_blocking_status_omitted
promoted_policy_equals_evaluated_candidate
```

Forbidden / non-claims (never asserted):

```text
execution_proof
full_suite_implied
ivp_commitment_is_sealer
model_agreement_is_equivalence
proof_of_test_execution
semantic_sufficiency
universal_semantic_completeness
zero_knowledge_proof
zk_proof
```

| Field | Value |
| --- | --- |
| Kind | bounded_artifact_evaluation |
| Claims semantic sufficiency | **false** |
| Is zero knowledge | **false** |

### Heuristics

| Field | Value |
| --- | --- |
| Classification | excluded_from_exact |
| Labels | coverage_heuristic, capsule_confidence_heuristic, route_calibration_empirical |
| Treated as exact | **false** |

---

## 16. Current-tree qualification results

### 16.1 Pytest (required validation command)

```text
PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q test/api/semantic_governor test/benchmarks/test_semantic_compression_governor_benchmark.py
```

| Result | Value |
| --- | --- |
| Collected | 710 |
| Passed | 710 |
| Failed | 0 |
| Duration | 29.37 s |
| Status | **green** |

**Hard failures:** none.

**Order-dependent failures:** none (hermetic package-import test `test_package_import_is_hermetic_and_lazy` now runs in a subprocess so it cannot dual-load governor classes mid-suite).

**Authority-matrix live-gitlink join:** pass. Planning pins remain the matrix authority; live datasets/kit heads `8ffa3152...` / `996ee85f...` are accepted as descendants. Hermetic package-import isolation runs in a child interpreter so the suite does not dual-load governor classes.

### 16.2 Board validator

```text
python3 scripts/validate_semantic_compression_governor_board.py --check-all
```

| Result | Value |
| --- | --- |
| valid | **true** |
| terminal_task_id | SCG-048 |
| ready_task_ids | SCG-048 |
| errors | none |

### 16.3 Prior release artifacts

| Artifact | Content id | Notes |
| --- | --- | --- |
| benchmark.json | `sha256:40a64d49405abbaaecee16c4002ed4c3f6f57bd903c3f387853cdde3ec572ea6` | simulated controlled corpus |
| summary.json | `sha256:da9061f672aeed49dadfa62ae7542b7d8409afc96befd74761f094b1784c8893` | status `yellow` |
| seal_qualification.json | `sha256:6bf42dca9b93e33aa68e9ccc0c85244b64a670e68379214942a215056ad9095a` | sealer unavailable |
| rollback.json | `sha256:3e0be7fd702fea06de435d1cb26ac5297dea731c76791ab572b094d04199aa46` | hermetic CAS |

---

## 17. Remaining production risks

- `authority_matrix_planning_pins_retained_live_heads_are_descendants`
- `expansion_false_negatives_on_simulated_corpus`
- `incremental_sealer_unavailable_on_current_tree`
- `live_billing_cost_sensors_missing`
- `live_model_quality_cohort_missing`
- `live_provider_receipts_missing`
- `median_context_reduction_below_soft_threshold`
- `no_production_policy_cas_promotion_on_this_tree`
- `production_eligible_false_by_design`
- `promotion_requires_explicit_authorization`
- `zk_seal_scope_unavailable`

### Missing evidence (union of benchmark + seal qualification)

- `live_billing_cost_sensors`
- `live_delta_or_full_checkpoint_seal`
- `live_model_quality_cohort`
- `live_observation_count`
- `live_provider_receipts`
- `post_execution_omission_detection_sensors`
- `released_IncrementalProofSealer_public_api`
- `zk_seal_scope`

---

## 18. Promotion recommendation

**Recommendation:** **do not promote** any production compression policy from this tree.

| Gate | Result |
| --- | --- |
| Held-out evaluation / proposal verdict | fail (`median_context_reduction_below_threshold`) |
| Separate authorization for production head | absent |
| Successful production CAS receipt | absent |
| Production eligible | false |
| Sealer-backed proof promotion | blocked (sealer unavailable) |
| Authorized VerificationBundle release-qual path | available for hermetic qualification only |
| Critical omission hard targets | met on simulated corpus |
| Zero accepted regressions | met |
| Authority matrix live pins | **green** (planning pins retained; live heads are descendants) |
| Full suite green | **yes** |

Operators may continue to use the governor as an **auditor and proposal system**. Promotion remains disabled until:

1. held-out evaluation meets protected thresholds (or an authorized threshold change is recorded),
2. live model quality / billing / provider evidence gaps are closed or explicitly waived by authorization,
3. a released IncrementalProofSealer (or an authorized non-sealer qualification path) is available when proof-backed promotion is claimed,
4. a distinct authorization CID plus successful CAS receipt are present for the exact candidate and evaluation.

---

## 19. Sealed machine projection

The privacy-filtered sealed `GovernorReport@1` identity is recomputed from structured fields (CID `baguqeera3qzt3fxishav374lfecwk32hmg37ap2vnx2xyrv4ywakt4lm3mdq`). Full JSON is published in:

`artifacts/agent_supervisor/semantic_compression_governor/release.json` → key `governor_report`.

```json
{
  "audit_population": {
    "history_cids": [],
    "live_audits": 0,
    "simulated_audits": 60,
    "source_receipt_cids": [],
    "total_audits": 60
  },
  "differential_outcomes": {
    "both_failed_count": 0,
    "compressed_failed_expanded_succeeded_count": 0,
    "equivalent_success_count": 0,
    "outcome_counts": {
      "both_failed_different_reason": 0,
      "both_failed_same_reason": 0,
      "both_valid_different": 0,
      "compressed_better": 0,
      "compressed_failed_expanded_succeeded": 0,
      "compressed_succeeded_expanded_failed": 0,
      "equivalent_success": 0,
      "expanded_better": 0,
      "human_review_required": 2,
      "verification_inconclusive": 58
    },
    "unavailable": false,
    "verification_inconclusive_count": 58
  },
  "evidence": "scg/final-report@1",
  "evidence_mode": "simulated",
  "expansion": {
    "expanded_tokens_total": 8646,
    "expansion_count": 0,
    "expansion_false_negative_count": 16,
    "expansion_false_positive_count": 0,
    "expansion_precision_bp": null,
    "expansion_rate_bp": 0,
    "expansion_recall_bp": 0,
    "expansion_true_positive_count": 0,
    "unavailable": false
  },
  "final_context_reduction": {
    "compressed_tokens_total": 6344,
    "mean_context_reduction_bp": 4537,
    "median_context_reduction_bp": 4736,
    "raw_tokens_total": 11170,
    "unavailable": false
  },
  "heuristics": {
    "classification": "excluded_from_exact",
    "heuristic_labels": [
      "capsule_confidence_heuristic",
      "coverage_heuristic",
      "route_calibration_empirical"
    ],
    "treated_as_exact": false,
    "unavailable": false
  },
  "implemented_commits": [
    "79e1a2c39f2252c2335909e4c6bc5418e6cd8a8b",
    "8ffa3152603fe8ae3a463250d91d47109cd48006",
    "996ee85f071dff17e4104948d9ca938d2125a447",
    "dc3164653a48d059ae9812078359daeafb451c07"
  ],
  "inspected_commits": [
    "1330038f626ef92993f03d46f21e1a57719e9c25",
    "79e1a2c39f2252c2335909e4c6bc5418e6cd8a8b",
    "7dc8f1422cb7e80757077948dc0785c1aaa4fd25",
    "8c7800cedc5e1b848367db9952f912428466f8cc",
    "8ffa3152603fe8ae3a463250d91d47109cd48006",
    "996ee85f071dff17e4104948d9ca938d2125a447",
    "dc3164653a48d059ae9812078359daeafb451c07",
    "df2f9cc092456329de9724c45a50c54b410875d1",
    "dfd92b554e662d4312411f2e8e63a52368806f2a"
  ],
  "interface_id": "GovernorReport@1",
  "live_metrics_cid": null,
  "metric_report_cid": null,
  "omission_detection": {
    "critical_acceptance_rate_bp": 0,
    "critical_omission_count": 16,
    "critical_omissions_accepted_count": 0,
    "detected_after_execution_count": 0,
    "detected_before_execution_count": 16,
    "detection_before_rate_bp": 10000,
    "false_alarm_count": 0,
    "intentional_omission_count": 16,
    "unavailable": false
  },
  "overhead_and_cost": {
    "audit_overhead_micros_total": 10025,
    "cost_per_accepted_patch_micros": null,
    "gross_savings_micros": 38608,
    "model_spend_micros_total": 165952,
    "net_savings_micros": -10417,
    "shadow_compute_micros_total": 15000,
    "unavailable": false,
    "verification_compute_micros_total": 24000
  },
  "proof_scope": {
    "claim_kinds": [
      "declared_thresholds_applied",
      "exact_artifacts_evaluated",
      "no_blocking_status_omitted",
      "promoted_policy_equals_evaluated_candidate",
      "required_evaluations_completed"
    ],
    "claims_semantic_sufficiency": false,
    "commitment_cid": null,
    "is_zero_knowledge": false,
    "kind": "bounded_artifact_evaluation",
    "unavailable": false
  },
  "quality": {
    "accepted_patch_count": 0,
    "accepted_rate_bp": 0,
    "proof_failure_count": 1,
    "regression_count": 0,
    "regression_rate_bp": 0,
    "review_disagreement_count": 2,
    "selected_test_false_negative_count": 1,
    "unavailable": false
  },
  "remaining_production_risks": [
    "authority_matrix_revised_from_planning_baseline_to_live_heads",
    "expansion_false_negatives_on_simulated_corpus",
    "incremental_sealer_unavailable_on_current_tree",
    "live_billing_cost_sensors_missing",
    "live_model_quality_cohort_missing",
    "live_provider_receipts_missing",
    "median_context_reduction_below_soft_threshold",
    "no_production_policy_cas_promotion_on_this_tree",
    "production_eligible_false_by_design",
    "promotion_requires_explicit_authorization",
    "zk_seal_scope_unavailable"
],
  "report_cid": "baguqeera3qzt3fxishav374lfecwk32hmg37ap2vnx2xyrv4ywakt4lm3mdq",
  "rollback": {
    "last_rollback_decision_cid": "baguqeeramul7foboodl6waiuqwbdg3ywupqhh6yo5fltaxsrdavcb3yn6kfq",
    "rollback_count": 2,
    "rollback_decision_cids": [
      "baguqeeraljk5jbgyuaajiwtqutjmvhjvik5aa4imcgskuifbsgzaotitn6ja",
      "baguqeeramul7foboodl6waiuqwbdg3ywupqhh6yo5fltaxsrdavcb3yn6kfq"
    ],
    "unavailable": false
  },
  "route_distribution": {
    "escalation_count": 9,
    "escalation_rate_bp": 1500,
    "retry_count": 0,
    "route_share_bp": {
      "deterministic": 0,
      "frontier": 1166,
      "human": 333,
      "medium": 3333,
      "small": 5166
    },
    "route_share_counts": {
      "deterministic": 0,
      "frontier": 7,
      "human": 2,
      "medium": 20,
      "small": 31
    },
    "unavailable": false
  },
  "rules": {
    "candidate_cids": [],
    "evaluation_report_cids": [],
    "promoted_count": 0,
    "proposed_count": 1,
    "rejected_count": 1,
    "unavailable": false
  },
  "schema": "ipfs_accelerate_py/agent-supervisor/semantic-governor/governor-report@1",
  "seal_scope": {
    "bound_artifact_cids": [
      "baguqeera3crgykbppugzs6m3suric2stjjumtlpdekj2d3xicvzsjfio4v2q",
      "baguqeera5dmhbjfv3p3qfzc7mszwid6gyhmkwuqjugff6t7psivsvuszvpkq",
      "baguqeera6bnxbtkhgt3r3flx5lfknbvlzhvrtmhvd5x6ozosccjzmfqyytva",
      "baguqeerai3qvrzd2xji7k7bgwieqzg53dxml57frq4w6di7u46a6mpxam2xa",
      "baguqeeralcva4uyn3g23zp6r4tigddft32b4roeusmdgf6olov6puaqfsy7a",
      "baguqeeralvpol33ycpfhciefwnc7rkr26lzgc2firlckomw3lhm4dvsdtwwa",
      "baguqeeramqq2gluupbpz7oqatt745wrgljz2w4wgf67gbkyrsz27sxw772xq",
      "baguqeeranwkwpg4yohtcpq73pdg5l34fp7q7euicsubs2xcnymf4xkfcopyq",
      "baguqeeraql5udwm5dwkjsux34dx44gu7ewgudqagyt5winuqvizgtkg7ohba"
    ],
    "qualification_path": "authorized_release_qualification",
    "seal_cid": "baguqeera3mxukdztvqjsz7tjlbslwf43o243q3dh4v23g4ncqrhmeeivaa2a",
    "sealer_interface_id": "BoundedSealClaimSet@1",
    "status": "unavailable",
    "unavailable": true
  },
  "simulated_metrics_present": false,
  "unavailable_fields": [
    "audit_population",
    "consumed_interfaces",
    "differential_outcomes",
    "evidence_mode",
    "expansion",
    "final_context_reduction",
    "heuristics",
    "implemented_commits",
    "inspected_commits",
    "live_billing_cost_sensors",
    "live_metrics_cid",
    "live_model_quality_cohort",
    "live_observation_count",
    "live_provider_receipts",
    "metric_report_cid",
    "omission_detection",
    "overhead_and_cost",
    "proof_scope",
    "quality",
    "released_incremental_proof_sealer_public_api",
    "remaining_production_risks",
    "rollback",
    "route_distribution",
    "rules",
    "seal_scope",
    "simulated_metrics_present",
    "unavailable_fields",
    "zk_seal_scope"
  ]
}
```

---

## 20. Environment

| Field | Value |
| --- | --- |
| Platform | Linux-6.17.0-1014-nvidia-aarch64-with-glibc2.39 |
| Python | 3.12.3 |
| Machine | aarch64 |
| Generated (UTC) | 2026-08-13T19:50:00Z |
| Task | SCG-048 |
| Goal | SCG-G090 |

---

*End of SCG-048 final report. Maximum justified claim is the bounded claim in §1. No universal semantic completeness and no unauthorized production policy promotion are asserted.*
