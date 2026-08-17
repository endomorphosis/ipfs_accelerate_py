"""Tests for SCG-038 complete governor metrics.

Acceptance criteria enforced here:

* Simulated and live cohorts stay separate.
* Percentiles and costs are reproducible (deterministic integer nearest-rank).
* Audit overhead is included in net savings.
* Unavailable data is missing, not zero success.
* Exact integer/fixed-point accounting with provenance (report CID).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
    RouteTier,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ExecutionMode,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration_contracts import (
    EvidencePartition,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    ComparativeOutcome,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.metrics import (
    BASIS_POINTS,
    GOVERNOR_METRIC_REPORT_INTERFACE,
    GOVERNOR_METRICS_COLLECTOR_INTERFACE,
    SCG_METRICS_EVIDENCE,
    CalibrationMetrics,
    CompressionMetrics,
    EconomicMetrics,
    GovernorMetricReport,
    GovernorMetricsCollector,
    IntegerPercentileSummary,
    MetricsCohort,
    MetricsError,
    MetricsIngestDisposition,
    MetricsObservation,
    OmissionMetrics,
    QualityMetrics,
    RoutingMetrics,
    build_empirical_rate,
    build_percentile_summary,
    collect_metrics,
    governor_metric_report_interface_id,
    governor_metrics_collector_interface_id,
    metrics_cohorts,
    metrics_evidence_id,
    metrics_ingest_dispositions,
    nearest_rank_percentile,
    observation_from_receipt_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/metrics.py"
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _obs(
    observation_id: str = "obs_0001",
    *,
    cohort: str = MetricsCohort.LIVE.value,
    **overrides: Any,
) -> MetricsObservation:
    fields: dict[str, Any] = {
        "observation_id": observation_id,
        "receipt_cid": _cid(f"receipt-{observation_id}"),
        "cohort": cohort,
        "route_tier": RouteTier.MEDIUM.value,
        "comparative_outcome": ComparativeOutcome.EQUIVALENT_SUCCESS.value,
        "acceptance_disposition": AcceptanceDisposition.NOT_ACCEPTED.value,
        "raw_tokens": 1000,
        "retrieval_tokens": 800,
        "compressed_tokens": 400,
        "expanded_tokens": 600,
        "accepted_patch": False,
        "regression": False,
        "selected_test_false_negative": False,
        "proof_failure": False,
        "review_disagreement": False,
        "intentional_omission_present": False,
        "omission_detected_before_execution": False,
        "omission_detected_after_execution": False,
        "critical_omission": False,
        "critical_omission_accepted": False,
        "expansion_used": False,
        "expansion_true_positive": False,
        "expansion_false_positive": False,
        "expansion_false_negative": False,
        "escalated": False,
        "retried": False,
        "input_tokens": 400,
        "output_tokens": 100,
        "baseline_model_spend_micros": 10_000,
        "model_spend_micros": 4_000,
        "verification_compute_micros": 500,
        "shadow_compute_micros": 300,
        "audit_overhead_micros": 200,
        "calibration_use": False,
        "calibration_revision": None,
        "omission_failure": False,
        "task_class": "local_bug",
        "partition": EvidencePartition.DEVELOPMENT.value,
        "metadata": {},
    }
    fields.update(overrides)
    return MetricsObservation(**fields)


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


def test_module_exports_required_interfaces() -> None:
    assert SCG_METRICS_EVIDENCE == "scg/metrics@1"
    assert GOVERNOR_METRICS_COLLECTOR_INTERFACE == "GovernorMetricsCollector@1"
    assert GOVERNOR_METRIC_REPORT_INTERFACE == "GovernorMetricReport@1"
    assert metrics_evidence_id() == SCG_METRICS_EVIDENCE
    assert governor_metrics_collector_interface_id() == (
        GOVERNOR_METRICS_COLLECTOR_INTERFACE
    )
    assert governor_metric_report_interface_id() == GOVERNOR_METRIC_REPORT_INTERFACE
    assert set(metrics_cohorts()) == {"live", "simulated"}
    assert "applied" in metrics_ingest_dispositions()
    assert MODULE_PATH.is_file()


def test_module_is_pure_no_network_imports() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"socket", "http", "urllib", "requests", "aiohttp", "httpx"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            assert root not in forbidden


# ---------------------------------------------------------------------------
# Percentiles — deterministic / reproducible
# ---------------------------------------------------------------------------


def test_nearest_rank_percentile_reproducible() -> None:
    samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # p50 → rank = ceil(0.5 * 10) = 5 → index 4 → 50
    assert nearest_rank_percentile(samples, 5_000) == 50
    # p90 → rank = ceil(0.9 * 10) = 9 → index 8 → 90
    assert nearest_rank_percentile(samples, 9_000) == 90
    # p95 → rank = ceil(0.95 * 10) = 10 → index 9 → 100
    assert nearest_rank_percentile(samples, 9_500) == 100
    # Shuffled input yields identical result (sort is internal).
    assert nearest_rank_percentile([100, 10, 50, 30, 90, 20, 70, 40, 80, 60], 5_000) == 50


def test_empty_sample_percentiles_are_unavailable_not_zero() -> None:
    assert nearest_rank_percentile([], 5_000) is None
    summary = build_percentile_summary([], sample_kind="raw_tokens")
    assert summary.sample_count == 0
    assert summary.p50 is None
    assert summary.p95 is None
    assert summary.min_value is None
    payload = summary.to_dict()
    assert payload["p50"] is None
    assert payload["total"] == 0


def test_percentile_summary_round_trip() -> None:
    summary = build_percentile_summary([1, 2, 3, 4, 5], sample_kind="tokens")
    restored = IntegerPercentileSummary.from_dict(summary.to_dict())
    assert restored.to_dict() == summary.to_dict()
    assert restored.p50 == 3


# ---------------------------------------------------------------------------
# Cohort separation
# ---------------------------------------------------------------------------


def test_simulated_and_live_cohorts_stay_separate() -> None:
    live = _obs(
        "live_001",
        cohort=MetricsCohort.LIVE.value,
        acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        accepted_patch=True,
        raw_tokens=2000,
        compressed_tokens=500,
        model_spend_micros=2_000,
        baseline_model_spend_micros=8_000,
    )
    simulated = _obs(
        "sim_001",
        cohort=MetricsCohort.SIMULATED.value,
        acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        accepted_patch=True,
        raw_tokens=100,
        compressed_tokens=10,
        model_spend_micros=100,
        baseline_model_spend_micros=500,
        regression=True,
    )
    report = collect_metrics([live, simulated])

    assert report.live.observation_count == 1
    assert report.simulated.observation_count == 1
    assert report.live.quality.accepted_patch_count == 1
    assert report.simulated.quality.accepted_patch_count == 1
    # Simulated regression must not appear in live quality.
    assert report.live.quality.regression_count == 0
    assert report.simulated.quality.regression_count == 1
    # Token totals stay cohort-local.
    assert report.live.compression.raw_tokens_total == 2000
    assert report.simulated.compression.raw_tokens_total == 100
    # Live savings exclude simulated spend.
    assert report.live.economic.model_spend_micros_total == 2_000
    assert report.simulated.economic.model_spend_micros_total == 100
    assert report.total_observations == 2


def test_execution_mode_maps_to_cohort() -> None:
    obs = observation_from_receipt_fields(
        observation_id="mode_map",
        receipt_cid=_cid("receipt-mode-map"),
        cohort=ExecutionMode.SIMULATED,
        raw_tokens=10,
        compressed_tokens=5,
    )
    assert obs.cohort == MetricsCohort.SIMULATED.value
    assert obs.is_simulated is True
    assert obs.is_live is False


def test_collector_idempotent_by_receipt_cid() -> None:
    collector = GovernorMetricsCollector()
    obs = _obs("idem_001")
    first = collector.ingest(obs)
    second = collector.ingest(obs)
    assert first is MetricsIngestDisposition.APPLIED
    assert second is MetricsIngestDisposition.SKIPPED_IDEMPOTENT
    report = collector.build_report()
    assert report.applied_count == 1
    assert report.skipped_idempotent_count == 1
    assert report.live.observation_count == 1


# ---------------------------------------------------------------------------
# Compression metrics
# ---------------------------------------------------------------------------


def test_compression_metrics_tokens_and_expansion_rate() -> None:
    observations = [
        _obs(
            "c1",
            raw_tokens=1000,
            retrieval_tokens=900,
            compressed_tokens=400,
            expanded_tokens=400,
            expansion_used=False,
        ),
        _obs(
            "c2",
            raw_tokens=2000,
            retrieval_tokens=1800,
            compressed_tokens=500,
            expanded_tokens=1200,
            expansion_used=True,
        ),
        _obs(
            "c3",
            raw_tokens=1500,
            retrieval_tokens=1400,
            compressed_tokens=600,
            expanded_tokens=600,
            expansion_used=False,
        ),
    ]
    report = collect_metrics(observations)
    compression = report.live.compression
    assert compression.observation_count == 3
    assert compression.raw_tokens_total == 4500
    assert compression.compressed_tokens_total == 1500
    assert compression.expansion_count == 1
    assert compression.expansion_rate_bp == (1 * BASIS_POINTS) // 3
    assert compression.raw_tokens_percentiles.p50 is not None
    assert compression.median_context_reduction_bp is not None
    # Mean reduction is integer floor of average basis points.
    assert isinstance(compression.mean_context_reduction_bp, int)


def test_unavailable_tokens_do_not_become_zero_success() -> None:
    obs = _obs(
        "missing_tokens",
        raw_tokens=None,
        retrieval_tokens=None,
        compressed_tokens=None,
        expanded_tokens=None,
    )
    report = collect_metrics([obs])
    compression = report.live.compression
    assert compression.raw_tokens_samples == 0
    assert compression.raw_tokens_percentiles.p50 is None
    assert compression.median_context_reduction_bp is None
    assert compression.unavailable_token_fields >= 4


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def test_quality_metrics_outcome_distribution_and_rates() -> None:
    observations = [
        _obs(
            "q1",
            acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
            accepted_patch=True,
            comparative_outcome=ComparativeOutcome.EQUIVALENT_SUCCESS.value,
        ),
        _obs(
            "q2",
            regression=True,
            comparative_outcome=ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value,
        ),
        _obs(
            "q3",
            selected_test_false_negative=True,
            proof_failure=True,
            review_disagreement=True,
            comparative_outcome=ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value,
        ),
        _obs(
            "q4",
            comparative_outcome=ComparativeOutcome.EQUIVALENT_SUCCESS.value,
        ),
    ]
    report = collect_metrics(observations)
    quality = report.live.quality
    assert quality.accepted_patch_count == 1
    assert quality.regression_count == 1
    assert quality.selected_test_false_negative_count == 1
    assert quality.proof_failure_count == 1
    assert quality.review_disagreement_count == 1
    assert quality.accepted_rate_bp == (1 * BASIS_POINTS) // 4
    assert quality.outcome_counts[ComparativeOutcome.EQUIVALENT_SUCCESS.value] == 2
    assert (
        quality.outcome_counts[
            ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
        ]
        == 1
    )
    # Empty quality rates stay missing on empty cohort.
    assert report.simulated.quality.accepted_rate_bp is None


# ---------------------------------------------------------------------------
# Omission metrics
# ---------------------------------------------------------------------------


def test_omission_metrics_detection_precision_recall() -> None:
    observations = [
        _obs(
            "o1",
            intentional_omission_present=True,
            omission_detected_before_execution=True,
            critical_omission=True,
            expansion_used=True,
            expansion_true_positive=True,
            omission_failure=True,
        ),
        _obs(
            "o2",
            intentional_omission_present=True,
            omission_detected_after_execution=True,
            expansion_false_negative=True,
            omission_failure=True,
        ),
        _obs(
            "o3",
            intentional_omission_present=False,
            omission_detected_before_execution=True,  # false alarm
            expansion_false_positive=True,
            expansion_used=True,
        ),
        _obs(
            "o4",
            intentional_omission_present=True,
            critical_omission=True,
            critical_omission_accepted=True,
            expansion_false_negative=True,
        ),
    ]
    report = collect_metrics(observations)
    omission = report.live.omission
    assert omission.intentional_omission_count == 3
    assert omission.detected_before_execution_count == 2
    assert omission.detected_after_execution_count == 1
    assert omission.critical_omission_count == 2
    assert omission.critical_omissions_accepted_count == 1
    assert omission.false_alarm_count >= 1
    assert omission.expansion_true_positive_count == 1
    assert omission.expansion_false_positive_count == 1
    assert omission.expansion_false_negative_count == 2
    # precision = 1 / (1+1) = 5000 bp
    assert omission.expansion_precision_bp == 5_000
    # recall = 1 / (1+2) = 3333 bp
    assert omission.expansion_recall_bp == 3_333
    assert omission.empirical_omission_rate is not None
    assert omission.empirical_omission_rate.successes == 2
    assert omission.empirical_omission_rate.trials == 4
    assert (
        omission.empirical_omission_rate.interval_lower_bp
        <= omission.empirical_omission_rate.rate_bp
        <= omission.empirical_omission_rate.interval_upper_bp
    )


def test_critical_omission_accepted_requires_critical_flag() -> None:
    with pytest.raises(MetricsError):
        _obs("bad_critical", critical_omission_accepted=True, critical_omission=False)


# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------


def test_routing_metrics_share_escalation_retry() -> None:
    observations = [
        _obs("r1", route_tier=RouteTier.SMALL.value),
        _obs("r2", route_tier=RouteTier.MEDIUM.value, retried=True),
        _obs(
            "r3",
            route_tier=RouteTier.FRONTIER.value,
            escalated=True,
            retried=True,
        ),
        _obs("r4", route_tier=RouteTier.MEDIUM.value),
    ]
    report = collect_metrics(observations)
    routing = report.live.routing
    assert routing.route_share_counts[RouteTier.SMALL.value] == 1
    assert routing.route_share_counts[RouteTier.MEDIUM.value] == 2
    assert routing.route_share_counts[RouteTier.FRONTIER.value] == 1
    assert routing.route_share_bp[RouteTier.MEDIUM.value] == 5_000
    assert routing.escalation_count == 1
    assert routing.retry_count == 2
    assert routing.escalation_rate_bp == (1 * BASIS_POINTS) // 4
    assert routing.retry_rate_bp == (2 * BASIS_POINTS) // 4


# ---------------------------------------------------------------------------
# Economic metrics — audit overhead in net savings
# ---------------------------------------------------------------------------


def test_net_savings_include_audit_overhead() -> None:
    """gross = baseline - model; net = gross - (audit + verification + shadow)."""

    obs = _obs(
        "econ_001",
        acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        accepted_patch=True,
        baseline_model_spend_micros=10_000,
        model_spend_micros=4_000,
        audit_overhead_micros=500,
        verification_compute_micros=300,
        shadow_compute_micros=200,
    )
    report = collect_metrics([obs])
    economic = report.live.economic
    assert economic.gross_savings_micros == 6_000
    # total audit overhead = 500 + 300 + 200 = 1000
    assert economic.total_audit_overhead_micros == 1_000
    assert economic.audit_overhead_micros_total == 500
    assert economic.verification_compute_micros_total == 300
    assert economic.shadow_compute_micros_total == 200
    assert economic.net_savings_micros == 5_000
    assert economic.cost_per_accepted_patch_micros == 4_000


def test_net_savings_reproducible_across_order() -> None:
    a = _obs(
        "e_a",
        baseline_model_spend_micros=8_000,
        model_spend_micros=3_000,
        audit_overhead_micros=100,
        verification_compute_micros=50,
        shadow_compute_micros=25,
    )
    b = _obs(
        "e_b",
        baseline_model_spend_micros=12_000,
        model_spend_micros=5_000,
        audit_overhead_micros=200,
        verification_compute_micros=100,
        shadow_compute_micros=50,
        acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        accepted_patch=True,
    )
    report_ab = collect_metrics([a, b])
    report_ba = collect_metrics([b, a])
    assert (
        report_ab.live.economic.gross_savings_micros
        == report_ba.live.economic.gross_savings_micros
    )
    assert (
        report_ab.live.economic.net_savings_micros
        == report_ba.live.economic.net_savings_micros
    )
    assert (
        report_ab.live.economic.total_audit_overhead_micros
        == report_ba.live.economic.total_audit_overhead_micros
    )
    # gross = (8000-3000) + (12000-5000) = 12000
    assert report_ab.live.economic.gross_savings_micros == 12_000
    # overhead = (100+50+25) + (200+100+50) = 525
    assert report_ab.live.economic.total_audit_overhead_micros == 525
    assert report_ab.live.economic.net_savings_micros == 12_000 - 525


def test_missing_cost_sensors_leave_savings_unavailable() -> None:
    obs = _obs(
        "no_cost",
        baseline_model_spend_micros=None,
        model_spend_micros=None,
        audit_overhead_micros=None,
        verification_compute_micros=None,
        shadow_compute_micros=None,
    )
    report = collect_metrics([obs])
    economic = report.live.economic
    assert economic.gross_savings_micros is None
    assert economic.net_savings_micros is None
    assert economic.cost_per_accepted_patch_micros is None
    assert economic.unavailable_cost_fields > 0


def test_cost_per_accepted_unavailable_without_accepts() -> None:
    obs = _obs(
        "no_accept",
        model_spend_micros=1_000,
        baseline_model_spend_micros=2_000,
        accepted_patch=False,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED.value,
    )
    report = collect_metrics([obs])
    assert report.live.economic.cost_per_accepted_patch_micros is None
    assert report.live.economic.gross_savings_micros == 1_000


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def test_calibration_metrics_revision_coverage_and_rate() -> None:
    observations = [
        _obs(
            "cal1",
            calibration_use=True,
            calibration_revision=3,
            omission_failure=True,
            task_class="local_bug",
            partition=EvidencePartition.CALIBRATION.value,
        ),
        _obs(
            "cal2",
            calibration_use=True,
            calibration_revision=5,
            omission_failure=False,
            task_class="schema_migration",
            partition=EvidencePartition.DEVELOPMENT.value,
        ),
        _obs(
            "cal3",
            calibration_use=False,
            calibration_revision=4,
            task_class="local_bug",
            partition=EvidencePartition.HELD_OUT.value,
        ),
    ]
    report = collect_metrics(observations)
    calibration = report.live.calibration
    assert calibration.calibration_use_count == 2
    assert calibration.last_revision == 5
    assert calibration.task_coverage_count == 2
    assert set(calibration.task_classes_observed) == {
        "local_bug",
        "schema_migration",
    }
    assert calibration.task_class_counts["local_bug"] == 2
    assert calibration.partition_counts[EvidencePartition.HELD_OUT.value] == 1
    assert calibration.empirical_omission_rate is not None
    assert calibration.empirical_omission_rate.successes == 1
    assert calibration.empirical_omission_rate.trials == 3


def test_build_empirical_rate_matches_wilson_bounds() -> None:
    rate = build_empirical_rate(2, 10)
    assert rate.rate_bp == 2_000
    assert rate.interval_lower_bp <= 2_000 <= rate.interval_upper_bp
    assert rate.interval_method == "wilson_score_95"


# ---------------------------------------------------------------------------
# Report provenance / identity / round-trip
# ---------------------------------------------------------------------------


def test_report_identity_is_stable_and_verifiable() -> None:
    observations = [
        _obs("id1", raw_tokens=100, compressed_tokens=40),
        _obs(
            "id2",
            cohort=MetricsCohort.SIMULATED.value,
            raw_tokens=50,
            compressed_tokens=20,
        ),
    ]
    report_a = collect_metrics(observations)
    report_b = collect_metrics(observations)
    assert report_a.report_cid == report_b.report_cid
    payload = report_a.to_dict()
    assert payload["evidence"] == SCG_METRICS_EVIDENCE
    assert payload["report_cid"] == report_a.report_cid
    restored = GovernorMetricReport.from_dict(payload)
    assert restored.report_cid == report_a.report_cid
    assert restored.live.observation_count == 1
    assert restored.simulated.observation_count == 1


def test_report_rejects_tampered_cid() -> None:
    report = collect_metrics([_obs("tamper")])
    payload = report.to_dict()
    payload["report_cid"] = _cid("not-the-real-report")
    with pytest.raises(MetricsError, match="report_cid"):
        GovernorMetricReport.from_dict(payload)


def test_observation_identity_round_trip() -> None:
    obs = _obs(
        "round",
        acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        accepted_patch=True,
        expansion_used=True,
        expansion_true_positive=True,
        intentional_omission_present=True,
    )
    restored = MetricsObservation.from_dict(obs.to_dict())
    assert restored.observation_cid == obs.observation_cid
    assert restored.to_dict() == obs.to_dict()


def test_observation_from_receipt_fields_helper() -> None:
    obs = observation_from_receipt_fields(
        observation_id="helper_001",
        receipt_cid=_cid("receipt-helper"),
        cohort=MetricsCohort.LIVE,
        route_tier=RouteTier.DETERMINISTIC,
        acceptance_disposition=AcceptanceDisposition.ACCEPTED,
        raw_tokens=100,
        compressed_tokens=40,
        model_spend_micros=1000,
        baseline_model_spend_micros=2500,
        audit_overhead_micros=100,
        verification_compute_micros=50,
        shadow_compute_micros=25,
    )
    assert obs.accepted_patch is True
    assert obs.route_tier == RouteTier.DETERMINISTIC.value
    report = collect_metrics([obs])
    assert report.live.economic.net_savings_micros == (2500 - 1000) - (100 + 50 + 25)


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def test_rejects_floats_in_metadata() -> None:
    with pytest.raises(MetricsError):
        _obs("float_meta", metadata={"ratio": 0.5})


def test_rejects_negative_tokens() -> None:
    with pytest.raises(MetricsError):
        _obs("neg", raw_tokens=-1)


def test_rejects_mutually_exclusive_expansion_flags() -> None:
    with pytest.raises(MetricsError, match="mutually exclusive"):
        _obs(
            "both_flags",
            expansion_true_positive=True,
            expansion_false_positive=True,
        )


def test_rejects_bool_as_integer_count() -> None:
    with pytest.raises(MetricsError):
        _obs("bool_int", raw_tokens=True)  # type: ignore[arg-type]


def test_empty_collector_report_has_empty_cohorts() -> None:
    report = GovernorMetricsCollector().build_report()
    assert report.total_observations == 0
    assert report.live.observation_count == 0
    assert report.simulated.observation_count == 0
    assert report.live.compression.expansion_rate_bp is None
    assert report.live.economic.net_savings_micros is None
    assert report.live.quality.accepted_rate_bp is None
    assert isinstance(report.live.compression, CompressionMetrics)
    assert isinstance(report.live.quality, QualityMetrics)
    assert isinstance(report.live.omission, OmissionMetrics)
    assert isinstance(report.live.routing, RoutingMetrics)
    assert isinstance(report.live.economic, EconomicMetrics)
    assert isinstance(report.live.calibration, CalibrationMetrics)


def test_reset_clears_accumulator() -> None:
    collector = GovernorMetricsCollector()
    collector.ingest(_obs("reset_me"))
    assert collector.live_observation_count == 1
    collector.reset()
    assert collector.live_observation_count == 0
    assert collector.applied_count == 0
    report = collector.build_report()
    assert report.total_observations == 0


def test_full_family_bundle_present_on_live_cohort() -> None:
    """Plan §12: all six metric families must be present."""

    report = collect_metrics(
        [
            _obs(
                "full",
                acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
                accepted_patch=True,
                intentional_omission_present=True,
                omission_detected_before_execution=True,
                expansion_used=True,
                expansion_true_positive=True,
                escalated=True,
                retried=True,
                calibration_use=True,
                calibration_revision=1,
                omission_failure=False,
            )
        ]
    )
    live = report.live
    payload = live.to_dict()
    for key in (
        "compression",
        "quality",
        "omission",
        "routing",
        "economic",
        "calibration",
    ):
        assert key in payload
        assert isinstance(payload[key], dict)
    # No floats in durable payload leaves.
    serialized = report.to_dict()

    def _assert_no_floats(value: Any, path: str = "$") -> None:
        if isinstance(value, float):
            raise AssertionError(f"float at {path}: {value!r}")
        if isinstance(value, dict):
            for key, item in value.items():
                _assert_no_floats(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                _assert_no_floats(item, f"{path}[{index}]")

    _assert_no_floats(serialized)
