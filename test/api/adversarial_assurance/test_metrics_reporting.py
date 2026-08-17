"""Tests for disjoint campaign metrics and report builders (AAE-058).

Acceptance criteria enforced here:

* Mutation coverage, detection quality, gap, remediation, and economics
  populations are pairwise disjoint and reproducible.
* Kill-rate denominators exclude invalid, equivalent, and infrastructure
  cases as specified (plan §5 / §15).
* ``AssuranceMetrics@1`` and ``build_assurance_report`` seal stable identities.
* Cold import is side-effect free; no production policy change.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.metrics import (
    AAE_METRICS_EVIDENCE,
    ASSURANCE_METRICS_INTERFACE,
    BASIS_POINTS,
    DENOMINATOR_EXCLUDED_OUTCOMES,
    METRICS_POPULATION_KINDS,
    AssuranceMetrics,
    MetricsError,
    MetricsPopulationKind,
    assert_populations_disjoint,
    compute_assurance_metrics,
    coverage_bucket,
    denominator_excluded_outcomes,
    is_denominator_excluded,
    metrics_descriptor,
    metrics_population_kinds,
    population_member_id,
    verify_assurance_metrics_identity,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.reporting import (
    ASSURANCE_REPORT_INTERFACE,
    BUILD_ASSURANCE_REPORT_INTERFACE,
    DEFAULT_SUCCESS_TARGETS,
    AssuranceReport,
    ReportingError,
    build_assurance_report,
    reporting_descriptor,
    verify_assurance_report_identity,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    MutationOutcomeStatus,
    counts_as_killed,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/metrics.py"
)
REPORTING_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/reporting.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


PLAN_CID = _cid("plan-aae058")
RESULT_CID = _cid("result-aae058")
REPO_STATE = _cid("repo-state-aae058")
POLICY_CID = _cid("policy-aae058")


def _outcome(
    candidate_id: str,
    status: str,
    *,
    operator_class: str = "control_flow",
    risk_weight_bp: int = 5_000,
    predicted: list[str] | None = None,
    selected: list[str] | None = None,
    executed: list[str] | None = None,
    observed: list[str] | None = None,
    kinds: dict[str, str] | None = None,
    killing_id: str | None = None,
    killing_kind: str | None = None,
) -> dict[str, Any]:
    predicted = predicted if predicted is not None else ["det_test"]
    selected = selected if selected is not None else list(predicted)
    executed = executed if executed is not None else list(selected)
    observed = observed if observed is not None else []
    kinds = kinds if kinds is not None else {d: "unit_test" for d in predicted}
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "outcome_status": status,
        "operator_class": operator_class,
        "risk_weight_bp": risk_weight_bp,
        "predicted_detector_ids": predicted,
        "selected_detector_ids": selected,
        "executed_detector_ids": executed,
        "observed_detector_ids": observed,
        "detector_kinds": kinds,
    }
    if killing_id is not None:
        payload["killing_detector_id"] = killing_id
        payload["killing_detector_kind"] = killing_kind
    return payload


def _sample_campaign_inputs() -> dict[str, Any]:
    outcomes = [
        _outcome(
            "cand_kill",
            MutationOutcomeStatus.KILLED_BY_TEST.value,
            operator_class="authorization",
            risk_weight_bp=8_000,
            observed=["det_test"],
            killing_id="det_test",
            killing_kind="unit_test",
        ),
        _outcome(
            "cand_surv_sel",
            MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
            operator_class="authorization",
            risk_weight_bp=7_000,
            predicted=["det_test", "det_proof"],
            selected=["det_test", "det_proof"],
            executed=["det_test", "det_proof"],
            observed=[],
            kinds={"det_test": "unit_test", "det_proof": "formal_obligation"},
        ),
        _outcome(
            "cand_surv_full",
            MutationOutcomeStatus.SURVIVED_FULL_VERIFICATION.value,
            operator_class="data_schema",
            risk_weight_bp=6_000,
        ),
        _outcome(
            "cand_invalid",
            MutationOutcomeStatus.INVALID_MUTANT.value,
            operator_class="control_flow",
            risk_weight_bp=1_000,
        ),
        _outcome(
            "cand_equiv",
            MutationOutcomeStatus.EQUIVALENT.value,
            operator_class="control_flow",
            risk_weight_bp=1_000,
        ),
        _outcome(
            "cand_prob_equiv",
            MutationOutcomeStatus.PROBABLY_EQUIVALENT.value,
            operator_class="control_flow",
            risk_weight_bp=1_000,
        ),
        _outcome(
            "cand_infra",
            MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value,
            operator_class="control_flow",
            risk_weight_bp=1_000,
        ),
        _outcome(
            "cand_timeout",
            MutationOutcomeStatus.TIMEOUT.value,
            operator_class="control_flow",
            risk_weight_bp=1_000,
        ),
        _outcome(
            "cand_inconclusive",
            MutationOutcomeStatus.INCONCLUSIVE.value,
            operator_class="control_flow",
            risk_weight_bp=1_000,
        ),
        _outcome(
            "cand_full_suite_kill",
            MutationOutcomeStatus.KILLED_BY_FULL_SUITE.value,
            operator_class="authorization",
            risk_weight_bp=9_000,
            predicted=["det_test"],
            selected=["det_test"],
            executed=["det_test", "det_full"],
            observed=["det_full"],
            kinds={"det_test": "unit_test", "det_full": "full_suite"},
            killing_id="det_full",
            killing_kind="full_suite",
        ),
    ]
    gaps = [
        {
            "gap_id": "gap_missing_test",
            "gap_class": "missing_test",
            "risk_class": "critical_security",
        },
        {
            "gap_id": "gap_policy",
            "gap_class": "missing_policy_constraint",
            "risk_class": "authorization",
        },
        {
            "gap_id": "gap_unknown",
            "gap_class": "unknown",
            "risk_class": "low",
        },
    ]
    remediations = [
        {
            "remediation_id": "rem_accept",
            "disposition": "accepted",
            "held_out_kill_count": 2,
            "cost_cpu_ms": 250,
            "cost_wall_ms": 300,
        },
        {
            "remediation_id": "rem_reject",
            "disposition": "rejected",
            "regression": True,
            "held_out_kill_count": 0,
            "cost_cpu_ms": 100,
        },
        {
            "remediation_id": "rem_over",
            "disposition": "failed",
            "overconstraint": True,
            "cost_cpu_ms": 50,
        },
    ]
    economics_records = [
        {
            "economics_id": "eco_kill",
            "full_cpu_ms": 1_000,
            "full_wall_ms": 1_200,
            "incremental_cpu_ms": 400,
            "incremental_wall_ms": 500,
            "cache_hits": 4,
            "cache_misses": 1,
            "compute_saved_cpu_ms": 600,
            "compute_saved_wall_ms": 700,
            "model_calls": 1,
            "model_tokens": 128,
        },
        {
            "economics_id": "eco_surv",
            "full_cpu_ms": 800,
            "full_wall_ms": 900,
            "incremental_cpu_ms": 350,
            "incremental_wall_ms": 400,
            "cache_hits": 2,
            "cache_misses": 2,
            "compute_saved_cpu_ms": 450,
            "compute_saved_wall_ms": 500,
        },
    ]
    return {
        "outcomes": outcomes,
        "gaps": gaps,
        "remediations": remediations,
        "economics_records": economics_records,
    }


def _compute_sample() -> AssuranceMetrics:
    data = _sample_campaign_inputs()
    return compute_assurance_metrics(
        campaign_id="campaign_aae058",
        plan_id="plan_aae058",
        plan_cid=PLAN_CID,
        result_cid=RESULT_CID,
        repository_state_cid=REPO_STATE,
        generated_count=12,
        admitted_count=10,
        **data,
    )


# ---------------------------------------------------------------------------
# Module surface / cold import
# ---------------------------------------------------------------------------


def test_modules_exist_and_export_interfaces() -> None:
    assert METRICS_PATH.is_file()
    assert REPORTING_PATH.is_file()
    assert ASSURANCE_METRICS_INTERFACE == "AssuranceMetrics@1"
    assert BUILD_ASSURANCE_REPORT_INTERFACE == "build_assurance_report@1"
    assert ASSURANCE_REPORT_INTERFACE == "AssuranceReport@1"
    assert AAE_METRICS_EVIDENCE == "aae/metrics@1"
    desc = metrics_descriptor()
    assert desc["interface"] == ASSURANCE_METRICS_INTERFACE
    assert desc["production_policy_change"] is False
    assert list(desc["populations"]) == list(METRICS_POPULATION_KINDS)
    rdesc = reporting_descriptor()
    assert rdesc["api"] == "build_assurance_report"
    assert rdesc["targets_are_goals_not_results"] is True


def test_cold_import_is_side_effect_free() -> None:
    """Import graphs must not open network sockets or spawn subprocesses."""

    for path in (METRICS_PATH, REPORTING_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {
                    "system",
                    "Popen",
                    "urlopen",
                    "urlretrieve",
                }


def test_population_kinds_are_closed_and_fivefold() -> None:
    kinds = metrics_population_kinds()
    assert kinds == (
        "mutation_coverage",
        "detection_quality",
        "gap",
        "remediation",
        "economics",
    )
    assert set(kinds) == {item.value for item in MetricsPopulationKind}


# ---------------------------------------------------------------------------
# Denominator exclusions
# ---------------------------------------------------------------------------


def test_denominator_excludes_invalid_equivalent_infrastructure() -> None:
    excluded = set(denominator_excluded_outcomes())
    assert excluded == set(DENOMINATOR_EXCLUDED_OUTCOMES)
    for status in (
        MutationOutcomeStatus.INVALID_MUTANT.value,
        MutationOutcomeStatus.UNCOMPILABLE.value,
        MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value,
        MutationOutcomeStatus.TIMEOUT.value,
        MutationOutcomeStatus.INCONCLUSIVE.value,
        MutationOutcomeStatus.EQUIVALENT.value,
        MutationOutcomeStatus.PROBABLY_EQUIVALENT.value,
    ):
        assert is_denominator_excluded(status) is True
        assert counts_as_killed(status) is False

    for status in (
        MutationOutcomeStatus.KILLED_BY_TEST.value,
        MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
        MutationOutcomeStatus.SURVIVED_FULL_VERIFICATION.value,
        MutationOutcomeStatus.HUMAN_REVIEW_REQUIRED.value,
    ):
        assert is_denominator_excluded(status) is False


def test_kill_rate_denominator_excludes_infrastructure_and_equivalence() -> None:
    metrics = _compute_sample()
    cov = metrics.mutation_coverage

    # Scoring population: kill, surv_sel, surv_full, full_suite_kill = 4
    assert cov.scoring_denominator == 4
    # Excluded: invalid, equiv, probably_equiv, infra, timeout, inconclusive = 6
    assert cov.denominator_excluded_count == 6
    assert cov.killed_count == 2
    assert cov.selected_survivor_count == 1
    assert cov.full_survivor_count == 1
    assert cov.invalid_count == 1
    assert cov.equivalent_count == 1
    assert cov.probably_equivalent_count == 1
    assert cov.infrastructure_failure_count == 1
    assert cov.timeout_count == 1
    assert cov.inconclusive_count == 1

    # 2 killed / 4 scoring = 5000 bp
    assert cov.kill_rate_bp == 5_000
    assert cov.generated_count == 12
    assert cov.admitted_count == 10


def test_excluded_statuses_never_inflate_kill_rate() -> None:
    """Adding more invalid/equivalent/infra cases must not change kill rate."""

    base = [
        _outcome(
            "k1",
            MutationOutcomeStatus.KILLED_BY_TEST.value,
            observed=["det_test"],
            killing_id="det_test",
            killing_kind="unit_test",
        ),
        _outcome(
            "s1",
            MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
        ),
    ]
    with_noise = base + [
        _outcome("i1", MutationOutcomeStatus.INVALID_MUTANT.value),
        _outcome("e1", MutationOutcomeStatus.EQUIVALENT.value),
        _outcome("f1", MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value),
        _outcome("t1", MutationOutcomeStatus.TIMEOUT.value),
        _outcome("n1", MutationOutcomeStatus.INCONCLUSIVE.value),
        _outcome("p1", MutationOutcomeStatus.PROBABLY_EQUIVALENT.value),
    ]
    m_base = compute_assurance_metrics(campaign_id="c_base", outcomes=base)
    m_noise = compute_assurance_metrics(campaign_id="c_noise", outcomes=with_noise)
    assert m_base.mutation_coverage.kill_rate_bp == 5_000
    assert m_noise.mutation_coverage.kill_rate_bp == 5_000
    assert m_noise.mutation_coverage.scoring_denominator == 2
    assert m_noise.mutation_coverage.denominator_excluded_count == 6


def test_empty_scoring_denominator_yields_unknown_rate_not_zero() -> None:
    metrics = compute_assurance_metrics(
        campaign_id="only_excluded",
        outcomes=[
            _outcome("i1", MutationOutcomeStatus.INVALID_MUTANT.value),
            _outcome("e1", MutationOutcomeStatus.EQUIVALENT.value),
        ],
    )
    assert metrics.mutation_coverage.scoring_denominator == 0
    assert metrics.mutation_coverage.kill_rate_bp is None
    assert metrics.mutation_coverage.risk_weighted_score_bp is None
    assert "kill_rate_unavailable_empty_denominator" in metrics.reason_codes


# ---------------------------------------------------------------------------
# Disjoint populations
# ---------------------------------------------------------------------------


def test_populations_are_pairwise_disjoint() -> None:
    metrics = _compute_sample()
    assert_populations_disjoint(metrics.populations)

    sets = {
        kind: set(metrics.populations[kind].member_ids)
        for kind in METRICS_POPULATION_KINDS
    }
    kinds = list(METRICS_POPULATION_KINDS)
    for i, a in enumerate(kinds):
        for b in kinds[i + 1 :]:
            overlap = sets[a] & sets[b]
            assert not overlap, f"{a} overlaps {b}: {overlap}"

    # Prefix scoping
    for kind, members in sets.items():
        for mid in members:
            assert mid.startswith(f"{kind}:")


def test_population_member_ids_use_distinct_prefixes() -> None:
    ids = [
        population_member_id(MetricsPopulationKind.MUTATION_COVERAGE, "x"),
        population_member_id(MetricsPopulationKind.DETECTION_QUALITY, "x"),
        population_member_id(MetricsPopulationKind.GAP, "x"),
        population_member_id(MetricsPopulationKind.REMEDIATION, "x"),
        population_member_id(MetricsPopulationKind.ECONOMICS, "x"),
    ]
    assert len(ids) == len(set(ids))
    assert ids[0] == "mutation_coverage:x"


def test_forged_overlapping_populations_fail_closed() -> None:
    metrics = _compute_sample()
    payload = metrics.to_dict()
    # Force an economics member into the gap population.
    shared = payload["populations"]["economics"]["member_ids"][0]
    payload["populations"]["gap"]["member_ids"] = list(
        payload["populations"]["gap"]["member_ids"]
    ) + [shared]
    payload["populations"]["gap"]["count"] = len(
        payload["populations"]["gap"]["member_ids"]
    )
    payload["gaps"]["member_ids"] = list(payload["populations"]["gap"]["member_ids"])
    with pytest.raises(MetricsError) as excinfo:
        verify_assurance_metrics_identity(payload)
    assert excinfo.value.reason_code in {
        "populations_not_disjoint",
        "population_scope",
        "member_alignment",
    }


# ---------------------------------------------------------------------------
# Detection / gap / remediation / economics
# ---------------------------------------------------------------------------


def test_detection_quality_counts_predicted_observed_missed() -> None:
    metrics = _compute_sample()
    det = metrics.detection_quality
    assert det.predicted_detector_count > 0
    assert det.observed_detector_count > 0
    assert det.missed_detector_count > 0
    assert det.full_suite_only_detection_count == 1
    # Selected kinds include unit_test and formal_obligation.
    assert det.selected_test_rate_bp is not None
    assert det.selected_proof_rate_bp is not None


def test_gap_category_counts_are_closed() -> None:
    metrics = _compute_sample()
    gaps = metrics.gaps
    assert gaps.total_gaps == 3
    assert gaps.high_risk_survivor_gaps == 2
    assert gaps.category_counts["missing_test"] == 1
    assert gaps.category_counts["missing_policy_constraint"] == 1
    assert gaps.category_counts["unknown"] == 1


def test_remediation_metrics_track_promotions_and_regressions() -> None:
    metrics = _compute_sample()
    rem = metrics.remediation
    assert rem.candidate_count == 3
    assert rem.accepted_promotion_count == 1
    assert rem.rejected_promotion_count == 2
    assert rem.regression_count == 1
    assert rem.overconstraint_count == 1
    assert rem.held_out_kill_count == 2
    assert rem.total_cost_cpu_ms == 400


def test_economics_full_versus_incremental_and_cache_reuse() -> None:
    metrics = _compute_sample()
    eco = metrics.economics
    assert eco.mutant_cost_records == 2
    assert eco.full_cpu_ms_total == 1_800
    assert eco.incremental_cpu_ms_total == 750
    assert eco.compute_saved_cpu_ms == 1_050
    assert eco.proof_cache_hits == 6
    assert eco.proof_cache_misses == 3
    assert eco.proof_cache_reuse_rate_bp == (6 * BASIS_POINTS) // 9
    assert eco.savings_rate_bp == (1_050 * BASIS_POINTS) // 1_800
    assert eco.avg_full_cost_per_mutant_cpu_ms == 900
    assert eco.avg_incremental_cost_per_mutant_cpu_ms == 375
    assert eco.model_calls == 1
    assert eco.model_tokens == 128
    # cost per critical gap: inc_cpu // high_risk_gaps (2)
    assert eco.cost_per_critical_gap_cpu_ms == 750 // 2
    # accepted + rejected = 3 promotions
    assert eco.cost_per_promotion_cpu_ms == 400 // 3


def test_risk_weighted_score_uses_scoring_population_only() -> None:
    metrics = _compute_sample()
    # Scoring weights: kill 8000, surv_sel 7000, surv_full 6000, full_suite 9000
    # Killed weights: 8000 + 9000 = 17000; total = 30000 → 5666 bp
    assert metrics.mutation_coverage.risk_weighted_score_bp == (
        17_000 * BASIS_POINTS
    ) // 30_000
    rates = metrics.mutation_coverage.class_kill_rates_bp
    assert "authorization" in rates
    # authorization: kill + surv_sel + full_suite_kill = 2 kills / 3
    assert rates["authorization"] == (2 * BASIS_POINTS) // 3


def test_coverage_bucket_mapping() -> None:
    assert coverage_bucket("killed_by_test") == "killed"
    assert coverage_bucket("survived_selected_verification") == "selected_survivor"
    assert coverage_bucket("survived_full_verification") == "full_survivor"
    assert coverage_bucket("equivalent") == "equivalent"
    assert coverage_bucket("invalid_mutant") == "invalid"
    with pytest.raises(MetricsError):
        coverage_bucket("not_a_real_status")


# ---------------------------------------------------------------------------
# Reproducibility / identity
# ---------------------------------------------------------------------------


def test_metrics_identity_is_stable_and_verifiable() -> None:
    a = _compute_sample()
    b = _compute_sample()
    assert a.metrics_cid == b.metrics_cid
    assert a.to_dict() == b.to_dict()
    assert verify_assurance_metrics_identity(a) == a.metrics_cid
    assert verify_assurance_metrics_identity(a.to_dict()) == a.metrics_cid


def test_metrics_identity_detects_tampering() -> None:
    metrics = _compute_sample()
    payload = metrics.to_dict()
    payload["mutation_coverage"] = dict(payload["mutation_coverage"])
    payload["mutation_coverage"]["killed_count"] = (
        payload["mutation_coverage"]["killed_count"] + 1
    )
    with pytest.raises(MetricsError) as excinfo:
        verify_assurance_metrics_identity(payload)
    assert excinfo.value.reason_code in {
        "identity_mismatch",
        "member_alignment",
        "count_mismatch",
    }


def test_production_policy_change_rejected() -> None:
    metrics = _compute_sample()
    payload = metrics.to_dict()
    payload["production_policy_changed"] = True
    with pytest.raises(MetricsError) as excinfo:
        verify_assurance_metrics_identity(payload)
    assert excinfo.value.reason_code == "production_policy_change"


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def test_build_assurance_report_binds_metrics() -> None:
    data = _sample_campaign_inputs()
    campaign_result = {
        "plan_id": "plan_aae058",
        "plan_cid": PLAN_CID,
        "result_cid": RESULT_CID,
        "repository_state_cid": REPO_STATE,
        "verification_policy_cid": POLICY_CID,
        "killed_count": 2,
        "survivor_count": 2,
        "invalid_count": 1,
        "inconclusive_count": 1,
        "terminal_status": "complete",
        "reason_codes": ["campaign_executed"],
        "candidate_reports": [
            {
                "candidate_id": "cand_kill",
                "candidate_cid": _cid("cand-kill"),
                "terminal_status": "killed",
                "report_cid": _cid("rep-kill"),
            },
            {
                "candidate_id": "cand_surv_sel",
                "candidate_cid": _cid("cand-surv"),
                "terminal_status": "survivor",
                "report_cid": _cid("rep-surv"),
            },
        ],
        "require_sandbox": True,
        "network_disabled": True,
        "production_policy_changed": False,
    }
    report = build_assurance_report(
        campaign_result,
        plan={"plan_id": "plan_aae058", "plan_cid": PLAN_CID},
        notes="aae-058-report",
        outcomes=data["outcomes"],
        gaps=data["gaps"],
        remediations=data["remediations"],
        economics_records=data["economics_records"],
        generated_count=12,
        admitted_count=10,
    )
    assert isinstance(report, AssuranceReport)
    assert report.metrics_available is True
    assert report.metrics_cid is not None
    assert report.metrics is not None
    assert report.metrics["interface_id"] == ASSURANCE_METRICS_INTERFACE
    assert report.production_policy_changed is False
    assert report.success_targets["targets_are_goals_not_results"] is True
    assert report.success_targets["compute_savings_min_bp"] == 5_000
    assert report.candidate_report_count == 2
    assert report.killed_count == 2
    assert "metrics_bound" in report.reason_codes
    assert "populations_disjoint" in report.reason_codes
    assert verify_assurance_report_identity(report) == report.report_cid
    assert verify_assurance_report_identity(report.to_dict()) == report.report_cid


def test_build_assurance_report_from_candidate_reports_alone() -> None:
    """CLI-shaped input without explicit outcomes still projects metrics."""

    campaign_result = {
        "plan_id": "plan_cli",
        "plan_cid": PLAN_CID,
        "result_cid": RESULT_CID,
        "repository_state_cid": REPO_STATE,
        "verification_policy_cid": POLICY_CID,
        "killed_count": 1,
        "survivor_count": 1,
        "invalid_count": 0,
        "inconclusive_count": 0,
        "terminal_status": "complete",
        "candidate_reports": [
            {
                "candidate_id": "cand_1",
                "terminal_status": "killed",
                "candidate_cid": _cid("c1"),
            },
            {
                "candidate_id": "cand_2",
                "terminal_status": "survivor",
                "candidate_cid": _cid("c2"),
            },
        ],
        "require_sandbox": True,
        "network_disabled": True,
    }
    report = build_assurance_report(campaign_result)
    assert report.metrics_available is True
    assert report.metrics is not None
    cov = report.metrics["mutation_coverage"]
    assert cov["killed_count"] == 1
    assert cov["selected_survivor_count"] == 1
    assert cov["scoring_denominator"] == 2
    assert cov["kill_rate_bp"] == 5_000


def test_build_assurance_report_is_reproducible() -> None:
    data = _sample_campaign_inputs()
    campaign_result = {
        "plan_id": "plan_aae058",
        "plan_cid": PLAN_CID,
        "result_cid": RESULT_CID,
        "repository_state_cid": REPO_STATE,
        "verification_policy_cid": POLICY_CID,
        "terminal_status": "complete",
        "candidate_reports": [],
        "require_sandbox": True,
        "network_disabled": True,
    }
    a = build_assurance_report(
        campaign_result,
        outcomes=data["outcomes"],
        gaps=data["gaps"],
        remediations=data["remediations"],
        economics_records=data["economics_records"],
    )
    b = build_assurance_report(
        campaign_result,
        outcomes=data["outcomes"],
        gaps=data["gaps"],
        remediations=data["remediations"],
        economics_records=data["economics_records"],
    )
    assert a.report_cid == b.report_cid
    assert a.metrics_cid == b.metrics_cid


def test_success_targets_are_goals_not_results() -> None:
    assert DEFAULT_SUCCESS_TARGETS["targets_are_goals_not_results"] is True
    report = build_assurance_report(
        {
            "plan_id": "plan_t",
            "plan_cid": PLAN_CID,
            "result_cid": RESULT_CID,
            "repository_state_cid": REPO_STATE,
            "verification_policy_cid": POLICY_CID,
            "terminal_status": "complete",
            "candidate_reports": [],
            "require_sandbox": True,
            "network_disabled": True,
        },
        include_metrics=False,
    )
    assert report.metrics_available is False
    assert report.success_targets["targets_are_goals_not_results"] is True
    # Targets must not be rebranded as measured results in the summary.
    assert "targets_are_goals_not_results" not in (report.summary or "")


def test_report_rejects_absolute_path_exposure() -> None:
    with pytest.raises(ReportingError) as excinfo:
        build_assurance_report(
            {
                "plan_id": "plan_path",
                "plan_cid": PLAN_CID,
                "result_cid": RESULT_CID,
                "repository_state_cid": REPO_STATE,
                "verification_policy_cid": POLICY_CID,
                "terminal_status": "complete",
                "repo_root": "/home/secret/project",
                "candidate_reports": [],
                "require_sandbox": True,
                "network_disabled": True,
            }
        )
    assert excinfo.value.reason_code == "path_exposure"


def test_report_rejects_production_policy_change_claim() -> None:
    with pytest.raises(ReportingError) as excinfo:
        AssuranceReport(
            interface_id=ASSURANCE_REPORT_INTERFACE,
            plan_id="plan_x",
            plan_cid=PLAN_CID,
            result_cid=RESULT_CID,
            repository_state_cid=REPO_STATE,
            verification_policy_cid=POLICY_CID,
            terminal_status="complete",
            killed_count=0,
            survivor_count=0,
            invalid_count=0,
            inconclusive_count=0,
            candidate_report_count=0,
            candidate_reports=(),
            reason_codes=("report_built",),
            summary="empty",
            notes=None,
            metrics_available=False,
            metrics=None,
            metrics_cid=None,
            success_targets=dict(DEFAULT_SUCCESS_TARGETS),
            require_sandbox=True,
            network_disabled=True,
            production_policy_changed=True,
        )
    assert excinfo.value.reason_code == "production_policy_change"


def test_unknown_outcome_fails_closed() -> None:
    with pytest.raises(MetricsError) as excinfo:
        compute_assurance_metrics(
            campaign_id="bad",
            outcomes=[{"candidate_id": "c1", "outcome_status": "totally_made_up"}],
        )
    assert excinfo.value.reason_code == "unknown_outcome"


def test_class_kill_rates_exclude_denominator_cases() -> None:
    outcomes = [
        _outcome(
            "k1",
            MutationOutcomeStatus.KILLED_BY_POLICY.value,
            operator_class="policy_ops",
            observed=["p1"],
            predicted=["p1"],
            selected=["p1"],
            executed=["p1"],
            kinds={"p1": "policy_rule"},
            killing_id="p1",
            killing_kind="policy_rule",
        ),
        _outcome(
            "inv",
            MutationOutcomeStatus.INVALID_MUTANT.value,
            operator_class="policy_ops",
        ),
        _outcome(
            "eq",
            MutationOutcomeStatus.EQUIVALENT.value,
            operator_class="policy_ops",
        ),
    ]
    metrics = compute_assurance_metrics(campaign_id="cls", outcomes=outcomes)
    # Only k1 remains in scoring denominator for class policy_ops.
    assert metrics.mutation_coverage.class_kill_rates_bp["policy_ops"] == BASIS_POINTS
    assert metrics.mutation_coverage.scoring_denominator == 1
