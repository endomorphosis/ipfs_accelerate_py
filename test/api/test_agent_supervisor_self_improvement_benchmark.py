from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import ipfs_accelerate_py.agent_supervisor as supervisor_api
from ipfs_accelerate_py.agent_supervisor import (
    MAX_CANDIDATE_ARTIFACT_BYTES,
    MAX_CANDIDATE_ARTIFACT_COUNT,
    MIN_INDEPENDENT_LANE_THROUGHPUT_BPS,
    MIN_MEDIAN_INPUT_TOKEN_REDUCTION_BPS,
    MIN_REPEATED_FIXTURE_CACHE_REUSE_BPS,
    PAIRED_EFFICIENCY_REQUIREMENT_ID,
    PAIRED_ROLLOUT_ACCEPTANCE_CRITERIA,
    PAIRED_ROLLOUT_CHILD_GOAL_IDS,
    PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID,
    PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID,
    PAIRED_ROLLOUT_OBJECTIVE_ID,
    PAIRED_ROLLOUT_PRODUCING_TASK_IDS,
    PAIRED_ROLLOUT_REQUIRED_EXHAUSTIVE_RECEIPTS,
    SHADOW_FALSE_COMPLETION_REQUIREMENT_ID,
    PairedFixtureKind,
    PairedRolloutFixture,
    PairedRolloutPolicy,
    PairedRolloutRequirementEvidence,
    PairedRolloutReport,
    PairedRolloutReportStore,
    PairedRolloutValidationError,
    REQUIRED_PAIRED_FIXTURE_KINDS,
    RolloutBehaviorMeasurement,
    SelfImprovementRolloutMode,
    evaluate_paired_self_improvement_rollout,
)
from ipfs_accelerate_py.agent_supervisor import self_improvement_rollout as rollout_module


NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
STATE_DIGEST = "sha256:" + "a" * 64
REPOSITORY_ID = "ipfs-accelerate-py"
REPOSITORY_TREE = "sha256:" + "c" * 64


def _measurement(
    kind: PairedFixtureKind,
    *,
    candidate: bool,
) -> RolloutBehaviorMeasurement:
    rejecting = {
        PairedFixtureKind.CONTRADICTORY,
        PairedFixtureKind.MALFORMED_OUTPUT,
        PairedFixtureKind.FAILED_VALIDATION,
    }
    no_acceptance = rejecting | {
        PairedFixtureKind.PROVIDER_UNAVAILABLE,
        PairedFixtureKind.DRAINED_REFILL,
    }
    terminal_outcome = (
        "rejected"
        if kind in rejecting
        else "degraded"
        if kind is PairedFixtureKind.PROVIDER_UNAVAILABLE
        else "exhausted"
        if kind is PairedFixtureKind.DRAINED_REFILL
        else "accepted"
    )
    accepted_work = (
        0
        if kind in no_acceptance
        else 4
        if kind is PairedFixtureKind.INDEPENDENT_PARALLEL
        else 2
        if kind is PairedFixtureKind.CONFLICTING_PARALLEL
        else 1
    )
    seeded_defects = 1 if kind is PairedFixtureKind.FAILED_VALIDATION else 0
    repeated = kind in {
        PairedFixtureKind.WARM,
        PairedFixtureKind.RESTART,
    }
    return RolloutBehaviorMeasurement(
        input_tokens=600 if candidate else 1_000,
        cache_lookups=10 if repeated else 2,
        cache_hits=(8 if candidate else 2) if repeated else 0,
        false_completions=0,
        authority_violations=0,
        stale_authoritative_hits=0,
        artifact_count=2,
        artifact_bytes=1_024,
        elapsed_ms=(
            1_800
            if candidate and kind is PairedFixtureKind.INDEPENDENT_PARALLEL
            else 4_000
            if kind is PairedFixtureKind.INDEPENDENT_PARALLEL
            else 900
            if candidate
            else 1_000
        ),
        completed_work=max(1, accepted_work),
        accepted_work=accepted_work,
        evidence_coverage_bps=9_200 if candidate else 9_000,
        quality_score_bps=9_200 if candidate else 9_000,
        invalid_plan_branches=4 if candidate else 5,
        seeded_defects=seeded_defects,
        detected_defects=seeded_defects,
        escaped_defects=0,
        false_rejections=0,
        merge_conflicts=(
            1 if kind is PairedFixtureKind.CONFLICTING_PARALLEL else 0
        ),
        duplicate_executions=0,
        unauthorized_mutations=0,
        terminal_outcome=terminal_outcome,
        state_digest_before=(
            STATE_DIGEST if kind is PairedFixtureKind.RESTART else ""
        ),
        state_digest_after=(
            STATE_DIGEST if kind is PairedFixtureKind.RESTART else ""
        ),
    )


def _fixtures() -> tuple[PairedRolloutFixture, ...]:
    return tuple(
        PairedRolloutFixture(
            fixture_id=f"asi-023:{kind.value}",
            fixture_kind=kind,
            fixture_revision="asi-023-fixtures@1",
            input_digest="sha256:" + f"{index:064x}",
            baseline=_measurement(kind, candidate=False),
            candidate=_measurement(kind, candidate=True),
        )
        for index, kind in enumerate(REQUIRED_PAIRED_FIXTURE_KINDS, start=1)
    )


def _replace_candidate(
    fixtures: tuple[PairedRolloutFixture, ...],
    kind: PairedFixtureKind,
    **changes: object,
) -> tuple[PairedRolloutFixture, ...]:
    return tuple(
        replace(item, candidate=replace(item.candidate, **changes))
        if item.fixture_kind is kind
        else item
        for item in fixtures
    )


def _requirement_evidence(
    report: PairedRolloutReport,
    requirement_id: str,
) -> PairedRolloutRequirementEvidence:
    return report.evidence_for(
        requirement_id,
        repository_id=REPOSITORY_ID,
        repository_tree=REPOSITORY_TREE,
    )


def test_benchmark_uses_complete_stable_package_root_rollout_surface() -> None:
    stable_exports = supervisor_api.PAIRED_ROLLOUT_STABLE_EXPORTS

    assert PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID == (
        "300500866741873729474343907613893393545"
    )
    assert PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID == "ASI-G114"
    assert set(stable_exports) == set(rollout_module.__all__)
    assert len(stable_exports) == len(set(stable_exports))
    assert all(name in supervisor_api.__all__ for name in stable_exports)
    assert all(
        getattr(supervisor_api, name) is getattr(rollout_module, name)
        for name in stable_exports
    )
    assert (
        supervisor_api.evaluate_paired_self_improvement_rollout
        is rollout_module.evaluate_paired_self_improvement_rollout
    )

    package_report = supervisor_api.evaluate_paired_self_improvement_rollout(
        _fixtures(),
        desired_mode=supervisor_api.SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )
    direct_report = rollout_module.evaluate_paired_self_improvement_rollout(
        _fixtures(),
        desired_mode=rollout_module.SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )
    assert package_report.report_id == direct_report.report_id
    assert package_report.to_dict() == direct_report.to_dict()


def test_closed_paired_population_passes_every_asi_023_gate() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _fixtures(),
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )

    assert report.promotion_allowed
    assert report.effective_mode is SelfImprovementRolloutMode.AUTOMATIC
    assert report["gate_passed"]
    assert report["nonnegotiable_gate_passed"]
    assert report["paired_gate_passed"]
    assert report["fixture_count"] == len(REQUIRED_PAIRED_FIXTURE_KINDS)
    assert {
        item["fixture_kind"] for item in report["fixtures"]
    } == {kind.value for kind in REQUIRED_PAIRED_FIXTURE_KINDS}
    assert report.reason_codes == ()

    metrics = report["metrics"]
    assert (
        metrics["median_input_token_reduction_bps"]
        >= MIN_MEDIAN_INPUT_TOKEN_REDUCTION_BPS
    )
    assert (
        metrics["repeated_fixture_cache_reuse_bps"]
        >= MIN_REPEATED_FIXTURE_CACHE_REUSE_BPS
    )
    assert (
        metrics["independent_lane_throughput_bps"]
        >= MIN_INDEPENDENT_LANE_THROUGHPUT_BPS
    )
    assert metrics["candidate_false_completions"] == 0
    assert metrics["candidate_authority_violations"] == 0
    assert metrics["candidate_stale_authoritative_hits"] == 0
    assert metrics["candidate_artifact_count"] <= MAX_CANDIDATE_ARTIFACT_COUNT
    assert metrics["candidate_artifact_bytes"] <= MAX_CANDIDATE_ARTIFACT_BYTES
    assert metrics["invalid_plan_branch_reduction_bps"] >= 2_000
    assert _requirement_evidence(
        report, SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
    ).requirement_satisfied
    assert _requirement_evidence(
        report, PAIRED_EFFICIENCY_REQUIREMENT_ID
    ).requirement_satisfied


def test_g090_fixture_families_share_every_required_gate_and_safety_invariant() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _fixtures(),
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )
    families = {
        "cold_warm": {
            PairedFixtureKind.COLD,
            PairedFixtureKind.WARM,
        },
        "failure": {
            PairedFixtureKind.FAILED_VALIDATION,
            PairedFixtureKind.PROVIDER_UNAVAILABLE,
        },
        "adversarial": {
            PairedFixtureKind.BROAD_GOAL,
            PairedFixtureKind.CONTRADICTORY,
            PairedFixtureKind.MALFORMED_OUTPUT,
            PairedFixtureKind.STALE_CACHE,
        },
        "parallel": {
            PairedFixtureKind.INDEPENDENT_PARALLEL,
            PairedFixtureKind.CONFLICTING_PARALLEL,
        },
        "restart": {PairedFixtureKind.RESTART},
        "refill": {PairedFixtureKind.DRAINED_REFILL},
    }

    assert set().union(*families.values()) == set(
        REQUIRED_PAIRED_FIXTURE_KINDS
    )
    assert all(
        report[name]
        for name in (
            "gate_passed",
            "nonnegotiable_gate_passed",
            "paired_gate_passed",
            "token_gate_passed",
            "cache_gate_passed",
            "planning_gate_passed",
            "throughput_gate_passed",
        )
    )
    assert PAIRED_ROLLOUT_OBJECTIVE_ID == "ASI-G090"
    assert PAIRED_ROLLOUT_PRODUCING_TASK_IDS == ("ASI-023", "ASI-024")
    assert PAIRED_ROLLOUT_CHILD_GOAL_IDS == (
        "ASI-G112",
        "ASI-G113",
        "ASI-G114",
    )
    assert len(PAIRED_ROLLOUT_ACCEPTANCE_CRITERIA) == 5
    assert PAIRED_ROLLOUT_REQUIRED_EXHAUSTIVE_RECEIPTS == 2

    fixtures = {
        PairedFixtureKind(item["fixture_kind"]): item["candidate"]
        for item in report["fixtures"]
    }
    for measurement in fixtures.values():
        assert measurement["false_completions"] == 0
        assert measurement["authority_violations"] == 0
        assert measurement["stale_authoritative_hits"] == 0
        assert measurement["escaped_defects"] == 0
        assert measurement["duplicate_executions"] == 0
        assert measurement["unauthorized_mutations"] == 0
    assert fixtures[PairedFixtureKind.FAILED_VALIDATION][
        "detected_defects"
    ] == fixtures[PairedFixtureKind.FAILED_VALIDATION]["seeded_defects"]
    assert fixtures[PairedFixtureKind.PROVIDER_UNAVAILABLE][
        "terminal_outcome"
    ] == "degraded"
    assert fixtures[PairedFixtureKind.RESTART][
        "state_digest_before"
    ] == fixtures[PairedFixtureKind.RESTART]["state_digest_after"]
    assert fixtures[PairedFixtureKind.DRAINED_REFILL][
        "terminal_outcome"
    ] == "exhausted"


def test_omitted_rollout_mode_proves_gates_but_never_promotes() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _fixtures(),
        evaluated_at=NOW,
    )

    assert report["desired_mode"] == SelfImprovementRolloutMode.SHADOW.value
    assert report["gate_passed"]
    assert report["token_gate_passed"]
    assert report["cache_gate_passed"]
    assert report["planning_gate_passed"]
    assert report["throughput_gate_passed"]
    assert not report.promotion_allowed
    assert report.effective_mode is SelfImprovementRolloutMode.SHADOW
    assert _requirement_evidence(
        report, PAIRED_EFFICIENCY_REQUIREMENT_ID
    ).requirement_satisfied


def test_either_reviewed_planning_improvement_can_pass_the_gate() -> None:
    coverage_fixtures = tuple(
        replace(
            item,
            candidate=replace(
                item.candidate,
                evidence_coverage_bps=10_000,
                invalid_plan_branches=item.baseline.invalid_plan_branches,
            ),
        )
        for item in _fixtures()
    )
    report = evaluate_paired_self_improvement_rollout(
        coverage_fixtures,
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )

    assert report["planning_gate_passed"]
    assert (
        report["metrics"]["planning_coverage_improvement_bps"] >= 1_000
    )
    assert report["metrics"]["invalid_plan_branch_reduction_bps"] == 0
    assert report.promotion_allowed


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("false_completions", "candidate_false_completion"),
        ("authority_violations", "candidate_authority_violation"),
        (
            "stale_authoritative_hits",
            "candidate_stale_authoritative_hit",
        ),
        ("escaped_defects", "candidate_escaped_defect"),
        ("duplicate_executions", "candidate_duplicate_execution"),
        ("unauthorized_mutations", "candidate_unauthorized_mutation"),
    ],
)
def test_nonnegotiable_violation_always_forces_shadow(
    field: str,
    reason: str,
) -> None:
    kind = (
        PairedFixtureKind.FAILED_VALIDATION
        if field == "escaped_defects"
        else PairedFixtureKind.COLD
    )
    changes = {field: 1}
    if field == "escaped_defects":
        changes["detected_defects"] = 0
    report = evaluate_paired_self_improvement_rollout(
        _replace_candidate(_fixtures(), kind, **changes),
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )

    assert not report.promotion_allowed
    assert report.effective_mode is SelfImprovementRolloutMode.SHADOW
    assert not report["nonnegotiable_gate_passed"]
    assert reason in report.reason_codes
    assert not _requirement_evidence(
        report, PAIRED_EFFICIENCY_REQUIREMENT_ID
    ).requirement_satisfied


@pytest.mark.parametrize(
    ("fixtures", "reason"),
    [
        (
            tuple(
                replace(
                    item,
                    candidate=replace(item.candidate, input_tokens=700),
                )
                for item in _fixtures()
            ),
            "median_input_token_reduction_below_threshold",
        ),
        (
            _replace_candidate(
                _replace_candidate(
                    _fixtures(),
                    PairedFixtureKind.WARM,
                    cache_hits=6,
                ),
                PairedFixtureKind.RESTART,
                cache_hits=6,
            ),
            "repeated_fixture_cache_reuse_below_threshold",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.INDEPENDENT_PARALLEL,
                elapsed_ms=2_100,
            ),
            "independent_lane_throughput_below_threshold",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.BROAD_GOAL,
                quality_score_bps=8_999,
            ),
            "quality_regression:broad_goal",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.BROAD_GOAL,
                invalid_plan_branches=5,
            ),
            "planning_improvement_below_threshold",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.CONFLICTING_PARALLEL,
                merge_conflicts=2,
            ),
            "merge_conflict_regression:conflicting_parallel",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.BROAD_GOAL,
                terminal_outcome="rejected",
            ),
            "paired_outcome_regression:broad_goal",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.BROAD_GOAL,
                evidence_coverage_bps=8_999,
            ),
            "evidence_coverage_regression:broad_goal",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.COLD,
                false_rejections=1,
            ),
            "false_rejection_regression:cold",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.COLD,
                accepted_work=0,
            ),
            "accepted_work_regression:cold",
        ),
        (
            _replace_candidate(
                _fixtures(),
                PairedFixtureKind.FAILED_VALIDATION,
                detected_defects=0,
            ),
            "defect_detection_regression:failed_validation",
        ),
    ],
)
def test_each_paired_regression_independently_forces_shadow(
    fixtures: tuple[PairedRolloutFixture, ...],
    reason: str,
) -> None:
    report = evaluate_paired_self_improvement_rollout(
        fixtures,
        desired_mode="automatic",
        evaluated_at=NOW,
    )

    assert not report.promotion_allowed
    assert report.effective_mode is SelfImprovementRolloutMode.SHADOW
    assert not report["paired_gate_passed"]
    assert reason in report.reason_codes
    assert not _requirement_evidence(
        report, PAIRED_EFFICIENCY_REQUIREMENT_ID
    ).requirement_satisfied


def test_fault_fixtures_must_fail_closed_and_restart_must_be_stable() -> None:
    malformed = _replace_candidate(
        _fixtures(),
        PairedFixtureKind.MALFORMED_OUTPUT,
        terminal_outcome="accepted",
    )
    malformed_report = evaluate_paired_self_improvement_rollout(
        malformed, evaluated_at=NOW
    )
    assert "candidate_malformed_output_not_rejected" in (
        malformed_report.reason_codes
    )

    provider = _replace_candidate(
        _fixtures(),
        PairedFixtureKind.PROVIDER_UNAVAILABLE,
        terminal_outcome="accepted",
    )
    provider_report = evaluate_paired_self_improvement_rollout(
        provider, evaluated_at=NOW
    )
    assert "candidate_provider_unavailable_overclaimed" in (
        provider_report.reason_codes
    )

    contradictory = _replace_candidate(
        _fixtures(),
        PairedFixtureKind.CONTRADICTORY,
        terminal_outcome="accepted",
    )
    contradictory_report = evaluate_paired_self_improvement_rollout(
        contradictory, evaluated_at=NOW
    )
    assert "candidate_contradiction_not_rejected" in (
        contradictory_report.reason_codes
    )

    restart = _replace_candidate(
        _fixtures(),
        PairedFixtureKind.RESTART,
        state_digest_after="sha256:" + "b" * 64,
    )
    restart_report = evaluate_paired_self_improvement_rollout(
        restart, evaluated_at=NOW
    )
    assert "candidate_restart_unstable" in restart_report.reason_codes
    assert all(
        report.effective_mode is SelfImprovementRolloutMode.SHADOW
        for report in (
            malformed_report,
            provider_report,
            contradictory_report,
            restart_report,
        )
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"artifact_count": MAX_CANDIDATE_ARTIFACT_COUNT},
            "candidate_artifact_count_exceeded",
        ),
        (
            {"artifact_bytes": MAX_CANDIDATE_ARTIFACT_BYTES},
            "candidate_artifact_bytes_exceeded",
        ),
    ],
)
def test_candidate_artifact_aggregate_is_hard_bounded(
    changes: dict[str, int],
    reason: str,
) -> None:
    report = evaluate_paired_self_improvement_rollout(
        _replace_candidate(_fixtures(), PairedFixtureKind.COLD, **changes),
        evaluated_at=NOW,
    )

    assert not report.promotion_allowed
    assert not report["nonnegotiable_gate_passed"]
    assert reason in report.reason_codes


def test_missing_fixture_is_a_gate_failure_not_a_smaller_benchmark() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _fixtures()[:-1],
        evaluated_at=NOW,
    )

    assert not report.promotion_allowed
    assert report.effective_mode is SelfImprovementRolloutMode.SHADOW
    assert "required_fixture_missing:drained_refill" in report.reason_codes
    assert not _requirement_evidence(
        report, SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
    ).requirement_satisfied
    assert not _requirement_evidence(
        report, PAIRED_EFFICIENCY_REQUIREMENT_ID
    ).requirement_satisfied


def test_policy_cannot_weaken_thresholds_bounds_or_population() -> None:
    with pytest.raises(PairedRolloutValidationError, match="cannot weaken"):
        PairedRolloutPolicy(min_median_input_token_reduction_bps=3_499)
    with pytest.raises(PairedRolloutValidationError, match="cannot weaken"):
        PairedRolloutPolicy(min_repeated_fixture_cache_reuse_bps=6_999)
    with pytest.raises(PairedRolloutValidationError, match="cannot weaken"):
        PairedRolloutPolicy(min_independent_lane_throughput_bps=19_999)
    with pytest.raises(PairedRolloutValidationError, match="cannot weaken"):
        PairedRolloutPolicy(max_candidate_artifact_count=257)
    with pytest.raises(PairedRolloutValidationError, match="cannot weaken"):
        PairedRolloutPolicy(max_candidate_artifact_bytes=4 * 1024 * 1024 + 1)
    with pytest.raises(PairedRolloutValidationError, match="cannot weaken"):
        PairedRolloutPolicy(max_report_bytes=2 * 1024 * 1024 + 1)
    with pytest.raises(PairedRolloutValidationError, match="non-narrowable"):
        PairedRolloutPolicy(
            required_fixture_kinds=REQUIRED_PAIRED_FIXTURE_KINDS[:-1]
        )


def test_report_round_trip_recomputes_metrics_and_rejects_tampering() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _fixtures(), evaluated_at=NOW
    )
    assert PairedRolloutReport.from_dict(report.to_dict()).to_dict() == (
        report.to_dict()
    )

    tampered = report.to_dict()
    tampered["metrics"]["candidate_false_completions"] = 1
    material = {
        key: value
        for key, value in tampered.items()
        if key not in {"report_id", "evaluated_at"}
    }
    import hashlib
    import json

    tampered["report_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    forged_typed = PairedRolloutReport(tampered)
    with pytest.raises(PairedRolloutValidationError, match="fixture evidence"):
        _requirement_evidence(
            forged_typed, SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
        )
    with pytest.raises(PairedRolloutValidationError, match="fixture evidence"):
        PairedRolloutReport.from_dict(tampered)

    legacy = report.to_dict()
    legacy["schema_version"] = 1
    for fixture in legacy["fixtures"]:
        fixture["baseline"].pop("invalid_plan_branches")
        fixture["candidate"].pop("invalid_plan_branches")
    for name in (
        "token_gate_passed",
        "cache_gate_passed",
        "planning_gate_passed",
        "throughput_gate_passed",
    ):
        legacy.pop(name)
    for name in (
        "baseline_median_evidence_coverage_bps",
        "candidate_median_evidence_coverage_bps",
        "planning_coverage_improvement_bps",
        "baseline_invalid_plan_branches",
        "candidate_invalid_plan_branches",
        "invalid_plan_branch_reduction_bps",
    ):
        legacy["metrics"].pop(name)
    legacy_material = {
        key: value
        for key, value in legacy.items()
        if key not in {"report_id", "evaluated_at"}
    }
    legacy["report_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            legacy_material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    restored_legacy = PairedRolloutReport.from_dict(legacy)
    assert restored_legacy.to_dict() == legacy
    assert not _requirement_evidence(
        restored_legacy, PAIRED_EFFICIENCY_REQUIREMENT_ID
    ).requirement_satisfied


def test_requirement_evidence_rejects_unknown_ids_and_is_round_trip_stable() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _fixtures(), evaluated_at=NOW
    )
    restored = PairedRolloutReport.from_dict(report.to_dict())

    for requirement_id in (
        SHADOW_FALSE_COMPLETION_REQUIREMENT_ID,
        PAIRED_EFFICIENCY_REQUIREMENT_ID,
    ):
        first = _requirement_evidence(report, requirement_id)
        second = _requirement_evidence(restored, requirement_id)
        assert first.to_dict() == second.to_dict()
        assert first.evidence_id == second.evidence_id
        assert first.report_id == report.report_id
        assert (
            PairedRolloutRequirementEvidence.from_dict(
                first.to_dict(),
                report=restored,
            )
            == first
        )

    with pytest.raises(
        PairedRolloutValidationError,
        match="unsupported paired rollout requirement",
    ):
        _requirement_evidence(report, "not-a-reviewed-requirement")

    tampered = _requirement_evidence(
        report, PAIRED_EFFICIENCY_REQUIREMENT_ID
    ).to_dict()
    tampered["repository_tree"] = "sha256:" + "d" * 64
    with pytest.raises(
        PairedRolloutValidationError,
        match="evidence|binding|identity",
    ):
        PairedRolloutRequirementEvidence.from_dict(tampered, report=report)


def test_bounded_report_store_is_idempotent_and_restart_safe(tmp_path) -> None:
    report = evaluate_paired_self_improvement_rollout(
        _fixtures(),
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )
    store = PairedRolloutReportStore(tmp_path / "rollout-reports")
    path = store.persist(report)

    assert path.stat().st_mode & 0o777 == 0o600
    assert path.stat().st_size <= report["policy"]["max_report_bytes"]
    assert store.persist(report) == path
    reloaded = PairedRolloutReportStore(
        tmp_path / "rollout-reports"
    ).load(report.report_id)
    assert reloaded.to_dict() == report.to_dict()
    assert reloaded.promotion_allowed

    replay = evaluate_paired_self_improvement_rollout(
        _fixtures(),
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW + timedelta(minutes=5),
    )
    assert replay.report_id == report.report_id
    assert store.persist(replay) == path
