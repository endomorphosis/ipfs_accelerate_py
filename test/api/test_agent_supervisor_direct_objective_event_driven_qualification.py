"""PCPR-001 fail-closed direct-objective and event-driven qualification."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.direct_objective_event_driven_qualification import (
    CASE_UNAVAILABLE_REASONS,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    DIRECT_OBJECTIVE_EVENT_DRIVEN_QUALIFICATION_INTERFACE,
    HARD_ZERO_INVARIANTS,
    HERMETIC_CANDIDATE_SUITES,
    LIVE_OBJECTIVE_MINIMUM,
    LIVE_REPLAY_MINIMUM,
    PCPR_PHASE0_GOAL_ID,
    PCPR_PHASE0_TASK_ID,
    REQUIRED_COHORT_CASES,
    REQUIRED_TARGETS,
    TARGET_CONTEXTPACK_REUSE_BPS,
    TARGET_ELIGIBLE_DECISIONS_WITHOUT_FRONTIER_BPS,
    TARGET_FRONTIER_MODEL_CALL_REDUCTION_BPS,
    TARGET_MAX_MANUAL_RECOVERY_BPS,
    TARGET_MAX_UNNECESSARY_TASK_CHURN_BPS,
    TARGET_MEDIAN_INPUT_TOKEN_REDUCTION_BPS,
    TARGET_ORDINARY_REFILLS_WITHOUT_LLM_BPS,
    TARGET_RECEIPT_SPEC,
    TARGET_UNAVAILABLE_REASONS,
    CohortEvidence,
    DirectObjectiveEventDrivenQualificationError,
    SafetyVector,
    TargetMeasurement,
    current_head_pcpr_phase0_receipt_promotion,
    current_head_pcpr_phase0_receipt_sections,
    current_head_unavailable_inputs,
    pcpr_phase0_receipt_efficiency_targets,
    pcpr_phase0_receipt_live_cohort,
    pcpr_phase0_receipt_promotion,
    qualify_current_head_without_live_campaign,
    qualify_direct_objective_event_driven,
    validate_pcpr_phase0_outer_receipt,
)


def test_closed_vocabularies_match_phase_zero_requirements() -> None:
    assert len(REQUIRED_COHORT_CASES) == 13
    assert len(REQUIRED_TARGETS) == 10
    assert len(HARD_ZERO_INVARIANTS) == 16
    assert LIVE_OBJECTIVE_MINIMUM == 10
    assert LIVE_REPLAY_MINIMUM == 20
    assert TARGET_MEDIAN_INPUT_TOKEN_REDUCTION_BPS == 3_000
    assert TARGET_FRONTIER_MODEL_CALL_REDUCTION_BPS == 4_000
    assert TARGET_ELIGIBLE_DECISIONS_WITHOUT_FRONTIER_BPS == 6_000
    assert TARGET_ORDINARY_REFILLS_WITHOUT_LLM_BPS == 8_000
    assert TARGET_CONTEXTPACK_REUSE_BPS == 5_000
    assert TARGET_MAX_UNNECESSARY_TASK_CHURN_BPS == 500
    assert TARGET_MAX_MANUAL_RECOVERY_BPS == 200
    assert DIRECT_OBJECTIVE_EVENT_DRIVEN_QUALIFICATION_INTERFACE == (
        "DirectObjectiveEventDrivenQualification@1"
    )
    assert set(HERMETIC_CANDIDATE_SUITES) == set(REQUIRED_COHORT_CASES)
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "non_promoted_supervisor_unqualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert "supervisor_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert set(CASE_UNAVAILABLE_REASONS) == set(REQUIRED_COHORT_CASES)
    assert set(TARGET_UNAVAILABLE_REASONS) == set(REQUIRED_TARGETS)
    assert set(TARGET_RECEIPT_SPEC) == set(REQUIRED_TARGETS)
    assert "test/api/test_agent_supervisor_todo_daemon_port.py" in HERMETIC_CANDIDATE_SUITES[
        "automatic_task_frontier_refill"
    ]
    assert "test/api/test_agent_supervisor_database_portal_bridge.py" in HERMETIC_CANDIDATE_SUITES[
        "automatic_task_frontier_refill"
    ]


def test_missing_live_campaign_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_without_live_campaign()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert set(verdict.missed_live_cohort) == set(REQUIRED_COHORT_CASES)
    assert set(verdict.missed_targets) == set(REQUIRED_TARGETS)
    payload = verdict.to_mapping()
    assert payload["closed_release_outcome"] is None
    assert all(item["live_count"] is None for item in payload["cohort"])
    assert all(item["evidence_kind"] == "unavailable" for item in payload["targets"])
    assert all(item["observed_bps"] is None for item in payload["targets"])
    assert all(item["observed_count"] is None for item in payload["targets"])
    assert verdict.verdict_cid == CURRENT_HEAD_UNAVAILABLE_VERDICT_CID
    section = current_head_pcpr_phase0_receipt_promotion()
    assert section == pcpr_phase0_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["duckdb_or_quack_state_written"] is False
    assert section["missed_live_cohort_count"] == 13
    assert section["missed_target_count"] == 10
    assert PCPR_PHASE0_TASK_ID == "PCPR-001"
    assert PCPR_PHASE0_GOAL_ID == "PCPR-G120"


def test_unavailable_metrics_are_not_represented_as_zero() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    token_target = next(
        item for item in targets if item.target_id == "median_end_to_end_input_token_reduction"
    )
    assert token_target.observed_bps is None
    assert token_target.evidence_kind == "unavailable"
    edits = next(item for item in targets if item.target_id == "manual_task_table_edits")
    assert edits.observed_count is None
    assert safety.counts == MappingProxyType({})
    verdict = qualify_direct_objective_event_driven(cohort, targets, safety)
    assert verdict.promotion_status == "rnd_non_promoted"


def test_hermetic_coverage_cannot_satisfy_live_objective_or_replay_counts() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    replaced = []
    for record in cohort:
        if record.case_id in {
            "ten_consecutive_bounded_objectives",
            "twenty_historical_task_replays",
        }:
            replaced.append(
                CohortEvidence(
                    case_id=record.case_id,
                    evidence_kind="measured_hermetic",
                    status="passed",
                    reason="hermetic suite passed; not a live campaign",
                    live_count=20,
                    live_environment_id="",
                    hermetic_suite_paths=record.hermetic_suite_paths,
                )
            )
        else:
            replaced.append(record)
    verdict = qualify_direct_objective_event_driven(replaced, targets, safety)
    assert "ten_consecutive_bounded_objectives" in verdict.missed_live_cohort
    assert "twenty_historical_task_replays" in verdict.missed_live_cohort
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.release_claim is False


def test_simulated_success_cannot_promote() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    simulated = []
    for record in cohort:
        simulated.append(
            CohortEvidence(
                case_id=record.case_id,
                evidence_kind="simulated",
                status="passed",
                reason="fixture",
                live_count=10,
                live_environment_id="sim",
                hermetic_suite_paths=record.hermetic_suite_paths,
            )
        )
    verdict = qualify_direct_objective_event_driven(simulated, targets, safety)
    assert verdict.promotion_status == "rnd_non_promoted"
    assert set(verdict.missed_live_cohort) == set(REQUIRED_COHORT_CASES)
    assert all(item.status != "passed" or item.evidence_kind != "simulated" for item in verdict.cohort)


def test_estimated_numeric_targets_are_rejected() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    forged = []
    for record in targets:
        if record.target_id == "median_end_to_end_input_token_reduction":
            forged.append(
                TargetMeasurement(
                    target_id=record.target_id,
                    evidence_kind="estimated",
                    observed_bps=9_000,
                    reason="guess",
                )
            )
        else:
            forged.append(record)
    with pytest.raises(DirectObjectiveEventDrivenQualificationError, match="estimated"):
        qualify_direct_objective_event_driven(cohort, forged, safety)


def test_live_objective_count_below_minimum_does_not_qualify() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    replaced = [
        CohortEvidence(
            case_id="ten_consecutive_bounded_objectives",
            evidence_kind="measured_live",
            status="passed",
            reason="nine live objectives",
            live_count=9,
            live_environment_id="pcpr-live-cohort",
        )
        if record.case_id == "ten_consecutive_bounded_objectives"
        else record
        for record in cohort
    ]
    verdict = qualify_direct_objective_event_driven(replaced, targets, safety)
    assert "ten_consecutive_bounded_objectives" in verdict.missed_live_cohort
    assert verdict.promotion_status == "rnd_non_promoted"


def test_blocked_cohort_case_is_typed_blocked() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    replaced = [
        CohortEvidence(
            case_id="lease_and_fencing_races",
            evidence_kind="measured_live",
            status="blocked",
            reason="live fencing environment unavailable",
            live_environment_id="pcpr-live-cohort",
        )
        if record.case_id == "lease_and_fencing_races"
        else record
        for record in cohort
    ]
    verdict = qualify_direct_objective_event_driven(replaced, targets, safety)
    assert verdict.promotion_status == "typed_blocked"
    assert "lease_and_fencing_races" in verdict.blockers
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False


def test_manual_task_table_edits_block_promotion() -> None:
    cohort, targets, safety = _live_passing_inputs()
    edited_targets = [
        TargetMeasurement(
            target_id="manual_task_table_edits",
            evidence_kind="measured_live",
            observed_count=1,
            reason="direct table write observed",
        )
        if record.target_id == "manual_task_table_edits"
        else record
        for record in targets
    ]
    verdict = qualify_direct_objective_event_driven(
        cohort,
        edited_targets,
        safety,
        live_campaign_identity="pcpr-live-cohort",
    )
    assert verdict.promotion_status != "supervisor_promoted"
    assert "manual_task_table_edits" in verdict.blockers
    assert verdict.closed_release_outcome is None


def test_hard_safety_failure_blocks_promotion() -> None:
    cohort, targets, safety = _live_passing_inputs()
    counts = dict(safety.counts)
    counts["false_completions"] = 1
    verdict = qualify_direct_objective_event_driven(
        cohort,
        targets,
        SafetyVector(counts=counts, evidence_kind="measured_live", reason="observed"),
        live_campaign_identity="pcpr-live-cohort",
    )
    assert "hard_safety_failures" in verdict.blockers
    assert verdict.promotion_status != "supervisor_promoted"
    assert verdict.release_claim is False


def test_complete_live_evidence_promotes_supervisor_without_release_claim() -> None:
    cohort, targets, safety = _live_passing_inputs()
    verdict = qualify_direct_objective_event_driven(
        cohort,
        targets,
        safety,
        live_campaign_identity="pcpr-live-cohort",
    )
    assert verdict.promotion_status == "supervisor_promoted"
    assert verdict.supervisor_disposition == "supervisor_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.missed_live_cohort == ()
    assert verdict.missed_targets == ()
    assert verdict.blockers == ()
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    section = pcpr_phase0_receipt_promotion(verdict)
    assert section["promotion_status"] == "supervisor_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["duckdb_or_quack_state_written"] is False


def test_unknown_or_duplicate_cohort_fails_closed() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    with pytest.raises(DirectObjectiveEventDrivenQualificationError, match="duplicate"):
        qualify_direct_objective_event_driven(cohort + cohort[:1], targets, safety)
    extra = cohort[1:] + (
        CohortEvidence(
            case_id="not_a_required_case",
            evidence_kind="unavailable",
            status="unavailable",
            reason="unknown",
        ),
    )
    with pytest.raises(DirectObjectiveEventDrivenQualificationError, match="unknown"):
        qualify_direct_objective_event_driven(extra, targets, safety)


def test_measured_live_requires_environment_identity() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    replaced = [
        CohortEvidence(
            case_id=record.case_id,
            evidence_kind="measured_live",
            status="passed",
            reason="missing environment",
            live_count=10,
        )
        if record.case_id == "external_python_and_mcp_submission"
        else record
        for record in cohort
    ]
    with pytest.raises(DirectObjectiveEventDrivenQualificationError, match="live_environment_id"):
        qualify_direct_objective_event_driven(replaced, targets, safety)


def test_net_cost_zero_is_not_positive_reduction() -> None:
    cohort, targets, safety = _live_passing_inputs()
    replaced = [
        TargetMeasurement(
            target_id="net_cost_reduction_after_audit",
            evidence_kind="measured_live",
            observed_net_cost_units=0,
            reason="break-even after audit",
        )
        if record.target_id == "net_cost_reduction_after_audit"
        else record
        for record in targets
    ]
    verdict = qualify_direct_objective_event_driven(
        cohort,
        replaced,
        safety,
        live_campaign_identity="pcpr-live-cohort",
    )
    assert "net_cost_reduction_after_audit" in verdict.missed_targets
    assert verdict.promotion_status == "rnd_non_promoted"


def _live_passing_inputs() -> tuple[
    tuple[CohortEvidence, ...],
    tuple[TargetMeasurement, ...],
    SafetyVector,
]:
    cohort = tuple(
        CohortEvidence(
            case_id=case_id,
            evidence_kind="measured_live",
            status="passed",
            reason="live campaign",
            live_count=10 if case_id == "ten_consecutive_bounded_objectives" else 20
            if case_id == "twenty_historical_task_replays"
            else None,
            live_environment_id="pcpr-live-cohort",
        )
        for case_id in REQUIRED_COHORT_CASES
    )
    measurements = {
        "median_end_to_end_input_token_reduction": TargetMeasurement(
            target_id="median_end_to_end_input_token_reduction",
            evidence_kind="measured_live",
            observed_bps=3_000,
            reason="live",
        ),
        "frontier_model_call_reduction": TargetMeasurement(
            target_id="frontier_model_call_reduction",
            evidence_kind="measured_live",
            observed_bps=4_000,
            reason="live",
        ),
        "net_cost_reduction_after_audit": TargetMeasurement(
            target_id="net_cost_reduction_after_audit",
            evidence_kind="measured_live",
            observed_net_cost_units=1,
            reason="live",
        ),
        "eligible_decisions_without_frontier_model": TargetMeasurement(
            target_id="eligible_decisions_without_frontier_model",
            evidence_kind="measured_live",
            observed_bps=6_000,
            reason="live",
        ),
        "ordinary_refills_without_llm": TargetMeasurement(
            target_id="ordinary_refills_without_llm",
            evidence_kind="measured_live",
            observed_bps=8_000,
            reason="live",
        ),
        "contextpack_reuse_on_eligible_tasks": TargetMeasurement(
            target_id="contextpack_reuse_on_eligible_tasks",
            evidence_kind="measured_live",
            observed_bps=5_000,
            reason="live",
        ),
        "unnecessary_task_churn": TargetMeasurement(
            target_id="unnecessary_task_churn",
            evidence_kind="measured_live",
            observed_bps=0,
            reason="live",
        ),
        "manual_recovery": TargetMeasurement(
            target_id="manual_recovery",
            evidence_kind="measured_live",
            observed_bps=0,
            reason="live",
        ),
        "manual_task_table_edits": TargetMeasurement(
            target_id="manual_task_table_edits",
            evidence_kind="measured_live",
            observed_count=0,
            reason="live",
        ),
        "hard_safety_failures": TargetMeasurement(
            target_id="hard_safety_failures",
            evidence_kind="measured_live",
            observed_count=0,
            reason="live",
        ),
    }
    targets = tuple(measurements[target_id] for target_id in REQUIRED_TARGETS)
    safety = SafetyVector(
        counts={name: 0 for name in HARD_ZERO_INVARIANTS},
        evidence_kind="measured_live",
        reason="live campaign counters",
    )
    return cohort, targets, safety


def test_current_head_receipt_sections_are_rnd_non_promoted_and_not_a_release() -> None:
    sections = current_head_pcpr_phase0_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_UNAVAILABLE_VERDICT_CID
    cohort = sections["required_live_cohort"]
    assert cohort["live_campaign_executed"] is False
    assert cohort["minimum_consecutive_bounded_objectives"] == 10
    assert cohort["minimum_historical_task_replays"] == 20
    assert cohort["manual_database_edits_in_this_task"] == 0
    assert [item["case_id"] for item in cohort["cases"]] == list(REQUIRED_COHORT_CASES)
    assert all(item["live_status"] == "unavailable" for item in cohort["cases"])
    assert all(item["evidence_kind"] == "unavailable" for item in cohort["cases"])
    assert all(item["live_count"] is None for item in cohort["cases"] if "live_count" in item)
    assert {
        item["case_id"]: item["reason"] for item in cohort["cases"]
    } == dict(CASE_UNAVAILABLE_REASONS)
    targets = sections["efficiency_targets"]
    assert set(targets) == set(REQUIRED_TARGETS)
    assert all(entry["observed_bps"] is None for entry in targets.values() if "observed_bps" in entry)
    assert all(
        entry["observed_count"] is None for entry in targets.values() if "observed_count" in entry
    )
    assert targets["net_cost_reduction_after_audit"]["observed_net_cost_units"] is None
    assert targets["manual_task_table_edits"]["this_task_direct_writes"] == 0
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True


def test_outer_receipt_validator_accepts_generated_missing_live_receipt() -> None:
    sections = current_head_pcpr_phase0_receipt_sections()
    payload = {
        "task_id": PCPR_PHASE0_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "required_live_cohort": sections["required_live_cohort"],
        "efficiency_targets": sections["efficiency_targets"],
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
        },
    }
    checked = validate_pcpr_phase0_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_UNAVAILABLE_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_phase0_receipt_sections()
    forged = {
        "task_id": PCPR_PHASE0_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "required_live_cohort": sections["required_live_cohort"],
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(DirectObjectiveEventDrivenQualificationError, match="closed PCPR release"):
        validate_pcpr_phase0_outer_receipt(forged)
    forged_status = {
        "task_id": PCPR_PHASE0_TASK_ID,
        "status": "non_promoted_supervisor_unqualified",
        "qualification_verdict": sections["qualification_verdict"],
    }
    with pytest.raises(DirectObjectiveEventDrivenQualificationError, match="closed PCPR release"):
        validate_pcpr_phase0_outer_receipt(forged_status)
    forged_verdict = {
        "task_id": PCPR_PHASE0_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "closed_release_outcome": "non_promoted_unmeasured",
        },
    }
    with pytest.raises(DirectObjectiveEventDrivenQualificationError, match="must be null"):
        validate_pcpr_phase0_outer_receipt(forged_verdict)


def test_hermetic_auto_start_suites_cannot_satisfy_live_refill() -> None:
    cohort, targets, safety = current_head_unavailable_inputs()
    verdict = qualify_direct_objective_event_driven(cohort, targets, safety)
    refill = next(
        item for item in verdict.cohort if item.case_id == "automatic_task_frontier_refill"
    )
    assert "test/api/test_agent_supervisor_todo_daemon_port.py" in refill.hermetic_suite_paths
    assert "test/api/test_agent_supervisor_database_portal_bridge.py" in refill.hermetic_suite_paths
    assert refill.evidence_kind == "unavailable"
    assert "automatic_task_frontier_refill" in verdict.missed_live_cohort
    section = pcpr_phase0_receipt_live_cohort(verdict)
    assert section["live_campaign_executed"] is False
    targets_section = pcpr_phase0_receipt_efficiency_targets(verdict)
    assert targets_section["median_end_to_end_input_token_reduction"]["observed_bps"] is None
