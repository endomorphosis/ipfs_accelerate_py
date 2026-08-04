"""PDR-082: quality-safe Pareto and anti-gaming Planner/Doctor rollout gates."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement import (
    planner_doctor_rollout as rollout,
)


QUALIFIED_AT = "2026-08-01T00:00:00Z"
CURRENT_AT = "2026-08-01T01:00:00Z"
HOLDOUT_AT = "2026-08-01T02:00:00Z"


def _binding(tree_id: str | None = None) -> rollout.PlannerDoctorRolloutBinding:
    if tree_id is None:
        return rollout.default_rollout_binding()
    return rollout.default_rollout_binding(tree_id=tree_id)


def _policy(**kwargs):
    return rollout.default_rollout_policy(**kwargs)


def _qualification(**kwargs):
    return rollout.build_passing_observation(
        observation_id="observation:qualification@1",
        observed_at=QUALIFIED_AT,
        role=rollout.ObservationRole.QUALIFICATION,
        **kwargs,
    )


def _current(**kwargs):
    defaults = {
        "observation_id": "observation:current@1",
        "observed_at": CURRENT_AT,
        "role": rollout.ObservationRole.CURRENT_TREE,
        "tree_id": "sha256:" + ("b" * 64),
    }
    defaults.update(kwargs)
    return rollout.build_passing_observation(**defaults)


def _holdout(**kwargs):
    defaults = {
        "observation_id": "observation:holdout@1",
        "observed_at": HOLDOUT_AT,
        "role": rollout.ObservationRole.HOLDOUT,
        "tree_id": "sha256:" + ("b" * 64),
    }
    defaults.update(kwargs)
    return rollout.build_passing_observation(**defaults)


def _inputs(*, allow_automatic: bool = True, fresh_root: bool = True):
    qualification = _qualification()
    current = _current()
    holdout = _holdout()
    binding = _binding(tree_id=current.tree_id)
    policy = _policy(
        allow_automatic=allow_automatic,
        operator_fresh_root_approved=fresh_root and allow_automatic,
        operator_fresh_root_tree_id=current.tree_id if fresh_root else "",
        operator_fresh_root_evidence_id=(
            "evidence:operator-fresh-root@1" if fresh_root else ""
        ),
    )
    return qualification, current, holdout, binding, policy


# ---------------------------------------------------------------------------
# Interfaces / schemas
# ---------------------------------------------------------------------------


def test_interfaces_and_schemas_are_stable() -> None:
    assert (
        rollout.PLANNER_DOCTOR_ROLLOUT_POLICY_INTERFACE
        == "PlannerDoctorRolloutPolicy@1"
    )
    assert (
        rollout.PLANNER_DOCTOR_PROMOTION_RECEIPT_INTERFACE
        == "PlannerDoctorPromotionReceipt@1"
    )
    assert rollout.PLANNER_DOCTOR_ROLLOUT_PRODUCER_TASK_ID == "PDR-082"
    assert rollout.PLANNER_DOCTOR_ROLLOUT_GOAL_ID == "PDR-G090"
    assert hasattr(rollout, "PlannerDoctorRolloutPolicy")
    assert hasattr(rollout, "PlannerDoctorPromotionReceipt")
    assert hasattr(rollout, "evaluate_planner_doctor_rollout")
    assert hasattr(rollout, "recompute_planner_doctor_gates")


def test_mode_ladder_is_closed() -> None:
    assert [m.value for m in rollout.PlannerDoctorRolloutMode] == [
        "off",
        "observe",
        "shadow",
        "assist",
        "canary",
        "automatic",
    ]


def test_default_policy_excludes_automatic_and_keeps_preregistered_method() -> None:
    policy = rollout.default_rollout_policy()
    assert rollout.PlannerDoctorRolloutMode.AUTOMATIC not in policy.allowed_modes
    assert not policy.automatic_approved
    assert (
        policy.non_inferiority_method
        == rollout.PREREGISTERED_NON_INFERIORITY_METHOD
    )
    assert policy.absolute_quality_margin == 0
    assert (
        policy.material_relative_improvement_millionths
        == rollout.MATERIAL_RELATIVE_IMPROVEMENT_MILLIONTHS
    )
    payload = policy.to_dict()
    assert payload["interface"] == "PlannerDoctorRolloutPolicy@1"
    restored = rollout.PlannerDoctorRolloutPolicy.from_dict(payload)
    assert restored == policy


def test_policy_rejects_cherry_picked_statistical_method() -> None:
    with pytest.raises(rollout.PlannerDoctorRolloutError, match="preregistered"):
        rollout.PlannerDoctorRolloutPolicy(
            non_inferiority_method="pick-best-p-value"
        )
    with pytest.raises(rollout.PlannerDoctorRolloutError, match="zero"):
        rollout.PlannerDoctorRolloutPolicy(absolute_quality_margin=1)
    with pytest.raises(rollout.PlannerDoctorRolloutError, match="material"):
        rollout.PlannerDoctorRolloutPolicy(
            material_relative_improvement_millionths=1
        )


# ---------------------------------------------------------------------------
# Gate recomputation
# ---------------------------------------------------------------------------


def test_complete_passing_observation_clears_all_gates() -> None:
    observation = _qualification()
    result = rollout.recompute_planner_doctor_gates(observation)

    assert result.passed
    assert result.safety_passed
    assert result.authority_passed
    assert result.quality_passed
    assert result.pareto_passed
    assert result.anti_gaming_passed
    assert result.exact_rollback_ok
    assert result.material_improvements
    assert not result.safety_floor_violations
    assert not result.authority_floor_violations
    assert not result.quality_non_inferiority_failures
    assert not any(result.anti_gaming_failures.values())
    assert not result.failure_codes


def test_safety_floor_is_non_compensable_by_pareto() -> None:
    baseline = rollout.build_clean_arm_metrics(
        pareto=rollout._baseline_pareto()
    )
    # Huge resource win but non-zero safety floor.
    challenger = rollout.build_clean_arm_metrics(
        pareto={name: 100_000 for name in rollout.PARETO_RESOURCE_METRICS},
        safety_overrides={"authority_violation_count": 1},
    )
    observation = replace(
        _qualification(),
        baseline=baseline,
        challenger=challenger,
    )
    result = rollout.recompute_planner_doctor_gates(observation)

    assert not result.safety_passed
    assert not result.authority_passed
    assert result.pareto_passed  # resource win alone is not enough
    assert not result.passed
    assert "safety-floor:authority_violation_count" in result.failure_codes
    assert "authority-floor:authority_violation_count" in result.failure_codes


def test_quality_regression_is_non_compensable_by_speed() -> None:
    baseline = rollout.build_clean_arm_metrics()
    challenger = rollout.build_clean_arm_metrics(
        pareto={name: 100_000 for name in rollout.PARETO_RESOURCE_METRICS},
        quality_higher_overrides={
            "first_valid_plan_rate_millionths": 900_000
        },
    )
    observation = replace(
        _qualification(),
        baseline=baseline,
        challenger=challenger,
    )
    result = rollout.recompute_planner_doctor_gates(observation)

    assert not result.quality_passed
    assert result.pareto_passed
    assert not result.passed
    assert (
        "quality-regression:first_valid_plan_rate_millionths"
        in result.quality_non_inferiority_failures
    )


def test_synthetic_skipped_unavailable_required_evidence_rejects() -> None:
    for status, kind in (
        ("synthetic", "paired_live_receipt"),
        ("skipped", "oracle_receipt"),
        ("unavailable", "telemetry_receipt"),
    ):
        challenger = rollout.build_clean_arm_metrics(
            pareto=rollout._improved_pareto(),
            evidence_overrides={kind: status},
        )
        observation = replace(_qualification(), challenger=challenger)
        result = rollout.recompute_planner_doctor_gates(observation)
        assert not result.passed
        assert any(
            code.startswith(f"evidence-{status}:")
            for code in result.evidence_admission_failures
        )

    synthetic_obs = replace(_qualification(), synthetic=True)
    result = rollout.recompute_planner_doctor_gates(synthetic_obs)
    assert "synthetic-observation" in result.evidence_admission_failures
    assert not result.passed


def test_no_material_pareto_improvement_rejects() -> None:
    observation = replace(
        _qualification(),
        baseline=rollout.build_clean_arm_metrics(
            pareto=rollout._baseline_pareto()
        ),
        challenger=rollout.build_clean_arm_metrics(
            pareto=rollout._baseline_pareto()
        ),
    )
    result = rollout.recompute_planner_doctor_gates(observation)
    assert not result.pareto_passed
    assert "no-material-pareto-improvement" in result.pareto_failures
    assert not result.passed


def test_resource_ceiling_regression_rejects_even_with_material_win() -> None:
    baseline = rollout.build_clean_arm_metrics(pareto=rollout._baseline_pareto())
    challenger = rollout.build_clean_arm_metrics(
        pareto=rollout._improved_pareto(),
        ceiling_overrides={"peak_rss_bytes": 4_000_000_000},
    )
    observation = replace(
        _qualification(), baseline=baseline, challenger=challenger
    )
    result = rollout.recompute_planner_doctor_gates(observation)
    assert not result.pareto_passed
    assert "resource-ceiling-regression:peak_rss_bytes" in result.pareto_failures
    assert not result.passed


def test_pareto_metric_regression_rejects() -> None:
    baseline = rollout.build_clean_arm_metrics(pareto=rollout._baseline_pareto())
    worse = dict(rollout._improved_pareto())
    worse["end_to_end_makespan_seconds"] = 1_100_000
    challenger = rollout.build_clean_arm_metrics(pareto=worse)
    observation = replace(
        _qualification(), baseline=baseline, challenger=challenger
    )
    result = rollout.recompute_planner_doctor_gates(observation)
    assert "pareto-regression:end_to_end_makespan_seconds" in result.pareto_failures
    assert not result.passed


def test_exact_rollback_failure_overrides_scores() -> None:
    challenger = rollout.build_clean_arm_metrics(
        pareto=rollout._improved_pareto(),
        exact_rollback_succeeded=False,
    )
    observation = replace(_qualification(), challenger=challenger)
    result = rollout.recompute_planner_doctor_gates(observation)
    assert not result.exact_rollback_ok
    assert not result.passed
    assert "exact-rollback-failure" in result.failure_codes


@pytest.mark.parametrize(
    "check",
    list(rollout.ANTI_GAMING_CHECKS),
)
def test_anti_gaming_detects_each_preregistered_check(check: str) -> None:
    challenger = rollout.build_clean_arm_metrics(
        pareto=rollout._improved_pareto(),
        anti_gaming_overrides={check: True},
    )
    observation = replace(_qualification(), challenger=challenger)
    result = rollout.recompute_planner_doctor_gates(observation)
    assert result.anti_gaming_failures[check]
    assert not result.anti_gaming_passed
    assert not result.passed
    assert any(
        code.startswith(f"anti-gaming:{check}:") for code in result.failure_codes
    )


def test_task_status_and_self_report_leakage_reject() -> None:
    challenger = rollout.build_clean_arm_metrics(
        pareto=rollout._improved_pareto(),
        task_status_used_as_quality=True,
        candidate_self_report_used=True,
    )
    observation = replace(_qualification(), challenger=challenger)
    result = rollout.recompute_planner_doctor_gates(observation)
    assert "task-status-as-quality" in result.evidence_admission_failures
    assert "candidate-self-report" in result.evidence_admission_failures
    assert result.anti_gaming_failures["task_status_leakage"]
    assert result.anti_gaming_failures["metric_leakage"]
    assert not result.passed


def test_denominator_mismatch_between_qualification_and_current() -> None:
    qualification = _qualification()
    mismatched = replace(
        _current(),
        denominator=rollout.build_default_denominator(
            case_ids=("case:only-one",),
            input_seal_id="seal:different@1",
        ),
    )
    result = rollout.recompute_planner_doctor_gates(
        mismatched,
        reference_denominator=qualification.denominator,
    )
    assert "paired-denominator-mismatch" in result.denominator_failures
    assert not result.passed


def test_holdout_must_be_disjoint_from_qualification_cases() -> None:
    qualification = _qualification()
    overlapping = replace(
        _holdout(),
        denominator=rollout.build_default_denominator(
            partition="holdout",
            case_ids=qualification.denominator.case_ids,
            input_seal_id="seal:holdout-overlap@1",
        ),
        holdout_partition="holdout",
        holdout_manifest_id="manifest:holdout@1",
    )
    result = rollout.recompute_planner_doctor_gates(
        overlapping,
        reference_denominator=qualification.denominator,
    )
    assert "holdout-case-overlap" in result.denominator_failures
    assert not result.passed


# ---------------------------------------------------------------------------
# Mode decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("desired", "expected"),
    (
        (rollout.PlannerDoctorRolloutMode.OFF, rollout.PlannerDoctorRolloutMode.OFF),
        (
            rollout.PlannerDoctorRolloutMode.OBSERVE,
            rollout.PlannerDoctorRolloutMode.OBSERVE,
        ),
        (
            rollout.PlannerDoctorRolloutMode.SHADOW,
            rollout.PlannerDoctorRolloutMode.SHADOW,
        ),
        (
            rollout.PlannerDoctorRolloutMode.ASSIST,
            rollout.PlannerDoctorRolloutMode.ASSIST,
        ),
    ),
)
def test_off_observe_shadow_assist_have_bound_deterministic_modes(
    desired, expected
) -> None:
    qualification, _, _, binding, policy = _inputs(allow_automatic=False)

    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding if desired is not rollout.PlannerDoctorRolloutMode.ASSIST
        else replace(
            binding,
            tree_id=qualification.tree_id,
        ),
        policy=policy,
        desired_mode=desired,
    )

    assert receipt.desired_mode is desired
    assert receipt.effective_mode is expected
    if desired is rollout.PlannerDoctorRolloutMode.ASSIST:
        assert receipt.qualification_gate_passed
        assert receipt.promotion_allowed
    else:
        assert not receipt.promotion_allowed or desired is (
            rollout.PlannerDoctorRolloutMode.ASSIST
        )


def test_canary_requires_current_tree_and_independent_holdout() -> None:
    qualification, current, holdout, binding, policy = _inputs(
        allow_automatic=False
    )
    binding = replace(binding, tree_id=qualification.tree_id)

    missing = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.CANARY,
    )
    assert missing.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert "current-tree-evaluation-required" in missing.reason_codes
    assert "independent-holdout-required" in missing.reason_codes
    assert not missing.canary_ready

    current_only = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=replace(binding, tree_id=current.tree_id),
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.CANARY,
        current_observation=current,
    )
    assert current_only.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert "independent-holdout-required" in current_only.reason_codes

    promoted = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=replace(binding, tree_id=current.tree_id),
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.CANARY,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert promoted.effective_mode is rollout.PlannerDoctorRolloutMode.CANARY
    assert promoted.canary_ready
    assert promoted.current_tree_gate_passed
    assert promoted.holdout_gate_passed
    assert promoted.promotion_allowed
    assert promoted.reason_codes == ()


def test_automatic_requires_explicit_policy_and_operator_fresh_root() -> None:
    qualification, current, holdout, binding, _ = _inputs()
    binding = replace(binding, tree_id=current.tree_id)

    no_auto = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=_policy(allow_automatic=False),
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert no_auto.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert not no_auto.automatic_ready
    assert "policy-mode-not-approved:automatic" in no_auto.reason_codes

    no_fresh = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=_policy(
            allow_automatic=True,
            operator_fresh_root_approved=False,
        ),
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert no_fresh.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert "operator-fresh-root-approval-required" in no_fresh.reason_codes

    promoted = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=_policy(
            allow_automatic=True,
            operator_fresh_root_approved=True,
            operator_fresh_root_tree_id=current.tree_id,
            operator_fresh_root_evidence_id="evidence:operator-fresh-root@1",
        ),
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert promoted.effective_mode is rollout.PlannerDoctorRolloutMode.AUTOMATIC
    assert promoted.automatic_ready
    assert promoted.promotion_allowed
    assert promoted.reason_codes == ()


def test_automatic_requires_later_separate_current_tree_evaluation() -> None:
    qualification, current, holdout, binding, policy = _inputs()
    binding = replace(binding, tree_id=current.tree_id)

    missing = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        holdout_observation=holdout,
    )
    assert missing.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert "current-tree-evaluation-required" in missing.reason_codes

    # Same observation reused as current is not separate / later.
    same = replace(
        qualification,
        role=rollout.ObservationRole.CURRENT_TREE,
    )
    replayed = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=replace(binding, tree_id=qualification.tree_id),
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=same,
        holdout_observation=holdout,
    )
    assert replayed.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert "current-evaluation-not-separate" in replayed.reason_codes
    assert "current-evaluation-not-later" in replayed.reason_codes


def test_kill_switch_overrides_all_scores_and_forces_off() -> None:
    qualification, current, holdout, binding, _ = _inputs()
    binding = replace(binding, tree_id=current.tree_id)
    policy = _policy(
        allow_automatic=True,
        kill_switch_engaged=True,
        operator_fresh_root_approved=True,
        operator_fresh_root_tree_id=current.tree_id,
        operator_fresh_root_evidence_id="evidence:operator-fresh-root@1",
    )

    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert receipt.kill_switch_override
    assert receipt.effective_mode is rollout.PlannerDoctorRolloutMode.OFF
    assert not receipt.automatic_ready
    assert not receipt.promotion_allowed
    assert "kill-switch-engaged" in receipt.reason_codes
    assert receipt.rollback_applied


def test_safety_failure_on_current_tree_rolls_back_assist() -> None:
    qualification = _qualification()
    bad_challenger = rollout.build_clean_arm_metrics(
        pareto=rollout._improved_pareto(),
        safety_overrides={"false_completion_count": 1},
    )
    current = replace(
        _current(),
        challenger=bad_challenger,
    )
    binding = replace(_binding(), tree_id=current.tree_id)
    policy = _policy(allow_automatic=False)

    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.ASSIST,
        current_observation=current,
    )
    assert receipt.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert receipt.rollback_applied
    assert any(
        code.startswith("current:safety-floor:") for code in receipt.reason_codes
    )


def test_cross_evaluation_quality_deterioration_rolls_back() -> None:
    qualification, _, holdout, binding, policy = _inputs()
    worse_quality = rollout.build_clean_arm_metrics(
        pareto=rollout._improved_pareto(),
        quality_higher_overrides={
            "independent_test_pass_millionths": 999_000
        },
    )
    # Still non-inferior to its own baseline (perfect), but worse than
    # qualification challenger — triggers cross-evaluation regression.
    current = replace(
        _current(),
        baseline=rollout.build_clean_arm_metrics(
            quality_higher_overrides={
                "independent_test_pass_millionths": 999_000
            }
        ),
        challenger=worse_quality,
    )
    binding = replace(binding, tree_id=current.tree_id)

    # Ensure qualification itself still passes absolute gates.
    assert rollout.recompute_planner_doctor_gates(current).passed

    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert receipt.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert receipt.rollback_applied
    assert (
        "regression:quality_higher:independent_test_pass_millionths"
        in receipt.reason_codes
    )


def test_stale_policy_capability_or_tree_binding_rolls_back() -> None:
    qualification, current, holdout, binding, _ = _inputs()
    stale_binding = replace(
        binding,
        tree_id="sha256:" + ("f" * 64),
        policy_revision="stale-policy-rev",
        capability_revision="stale-cap-rev",
    )
    stale_policy = _policy(
        allow_automatic=True,
        operator_fresh_root_approved=True,
        operator_fresh_root_tree_id=current.tree_id,
        operator_fresh_root_evidence_id="evidence:operator-fresh-root@1",
    )
    stale_policy = replace(
        stale_policy,
        policy_revision=stale_binding.policy_revision,
        approved_capability_ids=(stale_binding.capability_id,),
    )

    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=stale_binding,
        policy=stale_policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert receipt.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert receipt.rollback_applied
    assert "stale-binding:qualification" in receipt.reason_codes
    assert "stale-binding:current" in receipt.reason_codes


def test_receipt_restore_replays_sources_and_rejects_tampering() -> None:
    qualification, current, holdout, binding, policy = _inputs()
    binding = replace(binding, tree_id=current.tree_id)
    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=current,
        holdout_observation=holdout,
    )

    restored = rollout.PlannerDoctorPromotionReceipt.from_json(
        receipt.to_json(),
        qualification=qualification,
        current=current,
        holdout=holdout,
    )
    assert restored == receipt
    assert (
        rollout.verify_planner_doctor_promotion_receipt(
            receipt,
            qualification,
            current_observation=current,
            holdout_observation=holdout,
        )
        == receipt
    )
    tampered = receipt.to_dict(include_receipt_id=True)
    tampered["effective_mode"] = "shadow"
    with pytest.raises(
        rollout.PlannerDoctorRolloutError, match="source replay"
    ):
        rollout.PlannerDoctorPromotionReceipt.from_dict(
            tampered,
            qualification=qualification,
            current=current,
            holdout=holdout,
        )


def test_forbidden_unbounded_content_is_rejected_on_restore() -> None:
    qualification, current, holdout, binding, policy = _inputs()
    binding = replace(binding, tree_id=current.tree_id)
    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.CANARY,
        current_observation=current,
        holdout_observation=holdout,
    )
    payload = receipt.to_dict(include_receipt_id=True)
    payload["oracle_body"] = "secret gold answers"
    with pytest.raises(
        rollout.PlannerDoctorRolloutError, match="forbidden"
    ):
        rollout.PlannerDoctorPromotionReceipt.from_dict(
            payload,
            qualification=qualification,
            current=current,
            holdout=holdout,
        )


def test_observe_never_grants_promotion_authority() -> None:
    qualification = _qualification()
    binding = replace(_binding(), tree_id=qualification.tree_id)
    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=_policy(),
        desired_mode=rollout.PlannerDoctorRolloutMode.OBSERVE,
    )
    assert receipt.effective_mode is rollout.PlannerDoctorRolloutMode.OBSERVE
    assert not receipt.promotion_allowed


def test_assist_without_passing_qualification_stays_shadow() -> None:
    bad = replace(
        _qualification(),
        challenger=rollout.build_clean_arm_metrics(
            pareto=rollout._baseline_pareto()
        ),
    )
    binding = replace(_binding(), tree_id=bad.tree_id)
    receipt = rollout.evaluate_planner_doctor_rollout(
        bad,
        binding=binding,
        policy=_policy(),
        desired_mode=rollout.PlannerDoctorRolloutMode.ASSIST,
    )
    assert receipt.effective_mode is rollout.PlannerDoctorRolloutMode.SHADOW
    assert not receipt.qualification_gate_passed
    assert not receipt.promotion_allowed


def test_promotion_receipt_interface_fields_are_present() -> None:
    qualification, current, holdout, binding, policy = _inputs()
    binding = replace(binding, tree_id=current.tree_id)
    receipt = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.CANARY,
        current_observation=current,
        holdout_observation=holdout,
    )
    payload = receipt.to_dict(include_receipt_id=True)
    assert payload["interface"] == "PlannerDoctorPromotionReceipt@1"
    assert payload["producer_task_id"] == "PDR-082"
    assert payload["goal_id"] == "PDR-G090"
    assert payload["receipt_id"].startswith("sha256:")
    assert payload["qualification"]["pareto_passed"] is True
    assert payload["current"]["passed"] is True
    assert payload["holdout"]["passed"] is True


def test_replay_helper_matches_evaluate() -> None:
    qualification, current, holdout, binding, policy = _inputs()
    binding = replace(binding, tree_id=current.tree_id)
    expected = rollout.evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.CANARY,
        current_observation=current,
        holdout_observation=holdout,
    )
    replayed = rollout.replay_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=rollout.PlannerDoctorRolloutMode.CANARY,
        current_observation=current,
        holdout_observation=holdout,
        expected_receipt=expected,
    )
    assert replayed == expected
