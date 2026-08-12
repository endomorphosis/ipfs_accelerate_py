"""WPD-050: LLM-avoidance metrics and attempt attribution.

Acceptance (from the sealed WPD board):

* Metrics reject negative counts
* Attribute zero provider calls for ``closed_deterministic``
* Missing telemetry marked unavailable not zero-success

Interface: ``LlmAvoidanceMetrics@1``
Evidence: ``wpd/llm-avoidance-metrics@1``

Observability only — not completion authority.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.llm_avoidance_metrics import (
    ATTEMPT_ATTRIBUTION_SCHEMA,
    AttemptAttribution,
    AttemptDisposition,
    BASIS_POINTS,
    DISPOSITION_POLICY_SENSOR,
    LLM_AVOIDANCE_METRICS_EVIDENCE,
    LLM_AVOIDANCE_METRICS_INTERFACE,
    LLM_AVOIDANCE_METRICS_VERSION,
    LlmAvoidanceMetrics,
    LlmAvoidanceMetricsError,
    MISSING_TELEMETRY_SENSOR,
    ProviderTelemetrySample,
    TelemetryStatus,
    UnavailableReason,
    aggregate_attempt_attributions,
    attribute_attempt,
    attributes_zero_provider_calls,
    closed_disposition_values,
    expected_provider_call_floor,
)


# ---------------------------------------------------------------------------
# Interface / closed vocabulary
# ---------------------------------------------------------------------------


def test_llm_avoidance_metrics_interface_identity() -> None:
    assert LLM_AVOIDANCE_METRICS_INTERFACE == "LlmAvoidanceMetrics@1"
    assert LLM_AVOIDANCE_METRICS_VERSION == 1
    assert LLM_AVOIDANCE_METRICS_EVIDENCE == "wpd/llm-avoidance-metrics@1"
    assert LlmAvoidanceMetrics.INTERFACE == LLM_AVOIDANCE_METRICS_INTERFACE
    assert LlmAvoidanceMetrics.EVIDENCE == LLM_AVOIDANCE_METRICS_EVIDENCE


def test_closed_disposition_vocabulary_matches_implementation_disposition() -> None:
    expected = frozenset(
        {
            "closed_deterministic",
            "residual_llm_authorized",
            "abstain_review",
            "defer_capability",
        }
    )
    assert closed_disposition_values() == expected
    assert {d.value for d in AttemptDisposition} == expected


@pytest.mark.parametrize(
    "disposition,zero_calls,floor",
    [
        (AttemptDisposition.CLOSED_DETERMINISTIC, True, 0),
        (AttemptDisposition.ABSTAIN_REVIEW, True, 0),
        (AttemptDisposition.DEFER_CAPABILITY, True, 0),
        (AttemptDisposition.RESIDUAL_LLM_AUTHORIZED, False, None),
    ],
)
def test_expected_provider_call_floor_by_disposition(
    disposition: AttemptDisposition,
    zero_calls: bool,
    floor: int | None,
) -> None:
    assert attributes_zero_provider_calls(disposition) is zero_calls
    assert expected_provider_call_floor(disposition) == floor
    assert expected_provider_call_floor(disposition.value) == floor


def test_unknown_disposition_rejected() -> None:
    with pytest.raises(LlmAvoidanceMetricsError, match="must be one of"):
        attribute_attempt(
            attempt_id="a1",
            task_cid="task:1",
            disposition="free_form_llm",
        )


# ---------------------------------------------------------------------------
# Negative counts rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    [
        "provider_calls",
        "input_tokens",
        "output_tokens",
        "reused_tokens",
        "retry_tokens",
        "attempt_index",
    ],
)
def test_metrics_reject_negative_counts(field_name: str) -> None:
    kwargs: dict = {
        "attempt_id": "neg-1",
        "task_cid": "task:neg",
        "disposition": "residual_llm_authorized",
        "provider_calls": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "reused_tokens": 0,
        "retry_tokens": 0,
        "attempt_index": 0,
    }
    kwargs[field_name] = -1
    with pytest.raises(
        LlmAvoidanceMetricsError,
        match="non-negative integer|negative counts rejected",
    ):
        attribute_attempt(**kwargs)


def test_metrics_reject_boolean_and_float_counts() -> None:
    with pytest.raises(LlmAvoidanceMetricsError, match="non-negative integer"):
        attribute_attempt(
            attempt_id="bool-1",
            task_cid="task:bool",
            disposition="residual_llm_authorized",
            provider_calls=True,  # type: ignore[arg-type]
        )
    with pytest.raises(LlmAvoidanceMetricsError, match="non-negative integer"):
        ProviderTelemetrySample.measured(
            "provider_calls",
            1.5,  # type: ignore[arg-type]
            sensor_id="sensor:test",
        )


def test_aggregate_counters_reject_negative_construction() -> None:
    metrics = LlmAvoidanceMetrics()
    metrics.record_attempt(
        attempt_id="ok-1",
        task_cid="task:ok",
        disposition="closed_deterministic",
    )
    report = metrics.aggregate()
    # Report fields are non-negative by construction.
    assert report.total_attempts >= 0
    assert report.measured_provider_calls_total >= 0
    for counter in report.disposition_counters:
        assert counter.attempt_count >= 0
        assert counter.provider_calls_total >= 0


# ---------------------------------------------------------------------------
# closed_deterministic attributes zero provider calls
# ---------------------------------------------------------------------------


def test_closed_deterministic_attributes_zero_provider_calls() -> None:
    attempt = attribute_attempt(
        attempt_id="cd-1",
        task_cid="task:cd",
        disposition="closed_deterministic",
    )
    assert attempt.disposition is AttemptDisposition.CLOSED_DETERMINISTIC
    assert attempt.provider_calls.is_measured
    assert attempt.provider_calls.value == 0
    assert attempt.provider_calls.sensor_id == DISPOSITION_POLICY_SENSOR
    assert attempt.provider_calls_measured == 0
    assert attempt.attributes_zero_provider_calls


def test_closed_deterministic_rejects_nonzero_provider_calls() -> None:
    with pytest.raises(
        LlmAvoidanceMetricsError,
        match="zero provider calls",
    ):
        attribute_attempt(
            attempt_id="cd-bad",
            task_cid="task:cd",
            disposition="closed_deterministic",
            provider_calls=1,
        )


@pytest.mark.parametrize(
    "disposition",
    [
        "closed_deterministic",
        "abstain_review",
        "defer_capability",
    ],
)
def test_non_residual_dispositions_attribute_zero_provider_calls(
    disposition: str,
) -> None:
    attempt = attribute_attempt(
        attempt_id=f"zero-{disposition}",
        task_cid="task:zero",
        disposition=disposition,
        provider_calls=0,
    )
    assert attempt.provider_calls.is_measured
    assert attempt.provider_calls.value == 0


def test_closed_deterministic_explicit_zero_uses_disposition_sensor() -> None:
    attempt = attribute_attempt(
        attempt_id="cd-explicit-zero",
        task_cid="task:cd",
        disposition=AttemptDisposition.CLOSED_DETERMINISTIC,
        provider_calls=0,
    )
    assert attempt.provider_calls.value == 0
    assert attempt.provider_calls.sensor_id == DISPOSITION_POLICY_SENSOR


# ---------------------------------------------------------------------------
# Missing telemetry → unavailable, not zero-success
# ---------------------------------------------------------------------------


def test_residual_missing_provider_telemetry_is_unavailable_not_zero() -> None:
    attempt = attribute_attempt(
        attempt_id="res-missing",
        task_cid="task:res",
        disposition="residual_llm_authorized",
    )
    assert attempt.provider_calls.is_unavailable
    assert attempt.provider_calls.status is TelemetryStatus.UNAVAILABLE
    assert (
        attempt.provider_calls.reason_code
        == UnavailableReason.TELEMETRY_MISSING.value
    )
    assert attempt.provider_calls.sensor_id == MISSING_TELEMETRY_SENSOR
    # Must not pretend measured zero-success.
    assert attempt.provider_calls_measured is None
    assert "value" not in attempt.provider_calls.to_dict()
    assert attempt.provider_calls.to_dict()["status"] == "unavailable"


def test_residual_missing_token_telemetry_is_unavailable_not_zero() -> None:
    attempt = attribute_attempt(
        attempt_id="res-tokens-missing",
        task_cid="task:res",
        disposition="residual_llm_authorized",
        provider_calls=2,
    )
    assert attempt.provider_calls.is_measured
    assert attempt.provider_calls.value == 2
    for sample in (
        attempt.input_tokens,
        attempt.output_tokens,
        attempt.reused_tokens,
        attempt.retry_tokens,
    ):
        assert sample.is_unavailable
        assert sample.reason_code == UnavailableReason.TELEMETRY_MISSING.value
        assert sample.measured_value() is None
        payload = sample.to_dict()
        assert payload["status"] == "unavailable"
        assert "value" not in payload


def test_unavailable_sample_must_not_encode_numeric_value() -> None:
    # Internal placeholder zero is allowed; serialization omits value.
    ok = ProviderTelemetrySample(
        metric_name="provider_calls",
        status=TelemetryStatus.UNAVAILABLE,
        sensor_id=MISSING_TELEMETRY_SENSOR,
        value=0,
        reason_code=UnavailableReason.TELEMETRY_MISSING.value,
    )
    assert "value" not in ok.to_dict()
    assert ok.measured_value() is None

    # Non-zero values must not ride on an unavailable sample.
    with pytest.raises(
        LlmAvoidanceMetricsError,
        match="must not encode a numeric value",
    ):
        ProviderTelemetrySample(
            metric_name="input_tokens",
            status=TelemetryStatus.UNAVAILABLE,
            sensor_id=MISSING_TELEMETRY_SENSOR,
            value=7,
            reason_code=UnavailableReason.SENSOR_ABSENT.value,
        )


def test_empty_population_ratio_is_unavailable_not_zero_success() -> None:
    metrics = LlmAvoidanceMetrics()
    report = metrics.aggregate()
    assert report.total_attempts == 0
    assert report.llm_avoidance_ratio_status is TelemetryStatus.UNAVAILABLE
    assert report.llm_avoidance_ratio_bps is None
    assert (
        report.llm_avoidance_ratio_reason_code
        == UnavailableReason.SENSOR_ABSENT.value
    )
    ratio = report.to_dict()["llm_avoidance_ratio"]
    assert ratio["status"] == "unavailable"
    assert "value_bps" not in ratio
    assert ratio["reason_code"] == UnavailableReason.SENSOR_ABSENT.value


# ---------------------------------------------------------------------------
# Aggregation by disposition
# ---------------------------------------------------------------------------


def test_aggregate_per_attempt_counters_by_disposition() -> None:
    metrics = LlmAvoidanceMetrics()
    metrics.record_attempt(
        attempt_id="a-closed",
        task_cid="task:1",
        disposition="closed_deterministic",
    )
    metrics.record_attempt(
        attempt_id="a-residual",
        task_cid="task:2",
        disposition="residual_llm_authorized",
        provider_calls=3,
        input_tokens=100,
        output_tokens=40,
        reused_tokens=10,
        retry_tokens=5,
    )
    metrics.record_attempt(
        attempt_id="a-abstain",
        task_cid="task:3",
        disposition="abstain_review",
    )
    metrics.record_attempt(
        attempt_id="a-defer",
        task_cid="task:4",
        disposition="defer_capability",
    )
    metrics.record_attempt(
        attempt_id="a-residual-missing",
        task_cid="task:5",
        disposition="residual_llm_authorized",
        # provider telemetry intentionally omitted
    )

    report = metrics.aggregate()
    assert report.total_attempts == 5
    assert report.closed_deterministic_attempts == 1
    assert report.residual_llm_attempts == 2
    assert report.abstain_review_attempts == 1
    assert report.defer_capability_attempts == 1

    closed = report.counters_for("closed_deterministic")
    assert closed.attempt_count == 1
    assert closed.provider_calls_total == 0
    assert closed.measured_provider_call_attempts == 1
    assert closed.unavailable_provider_call_attempts == 0

    residual = report.counters_for("residual_llm_authorized")
    assert residual.attempt_count == 2
    assert residual.provider_calls_total == 3
    assert residual.measured_provider_call_attempts == 1
    assert residual.unavailable_provider_call_attempts == 1
    assert residual.input_tokens_total == 100
    assert residual.output_tokens_total == 40

    assert report.measured_provider_calls_total == 3
    assert report.unavailable_provider_call_attempts == 1
    assert report.measured_input_tokens_total == 100
    assert report.measured_output_tokens_total == 40

    # Avoidance ratio: 3 non-residual of 5 attempts → 6000 bps
    assert report.llm_avoidance_ratio_status is TelemetryStatus.MEASURED
    assert report.llm_avoidance_ratio_bps == (3 * BASIS_POINTS) // 5

    payload = report.to_dict()
    assert payload["interface"] == LLM_AVOIDANCE_METRICS_INTERFACE
    assert payload["evidence"] == LLM_AVOIDANCE_METRICS_EVIDENCE
    assert payload["completion_authority"] is False
    assert payload["llm_avoidance_ratio"]["value_bps"] == 6000


def test_aggregate_helper_and_round_trip_dict() -> None:
    attempts = [
        attribute_attempt(
            attempt_id="rt-1",
            task_cid="task:rt",
            disposition="closed_deterministic",
        ),
        attribute_attempt(
            attempt_id="rt-2",
            task_cid="task:rt",
            disposition="residual_llm_authorized",
            provider_calls=1,
            input_tokens=20,
            output_tokens=8,
        ),
    ]
    report = aggregate_attempt_attributions(attempts)
    assert report.total_attempts == 2
    assert report.measured_provider_calls_total == 1

    restored = AttemptAttribution.from_dict(attempts[0].to_dict())
    assert restored.content_id == attempts[0].content_id
    assert restored.to_dict()["schema"] == ATTEMPT_ATTRIBUTION_SCHEMA

    metrics = LlmAvoidanceMetrics()
    metrics.extend(attempts)
    assert len(metrics) == 2
    assert metrics.aggregate().content_id == report.content_id


def test_duplicate_attempt_id_with_conflict_rejects() -> None:
    metrics = LlmAvoidanceMetrics()
    metrics.record_attempt(
        attempt_id="dup-1",
        task_cid="task:dup",
        disposition="closed_deterministic",
    )
    with pytest.raises(LlmAvoidanceMetricsError, match="already recorded"):
        metrics.record_attempt(
            attempt_id="dup-1",
            task_cid="task:dup",
            disposition="residual_llm_authorized",
            provider_calls=1,
        )


def test_duplicate_identical_attempt_is_idempotent() -> None:
    metrics = LlmAvoidanceMetrics()
    first = metrics.record_attempt(
        attempt_id="idem-1",
        task_cid="task:idem",
        disposition="closed_deterministic",
    )
    second = metrics.record_attempt(
        attempt_id="idem-1",
        task_cid="task:idem",
        disposition="closed_deterministic",
    )
    assert first.content_id == second.content_id
    assert len(metrics) == 1


def test_metric_labels_include_disposition_and_task() -> None:
    attempt = attribute_attempt(
        attempt_id="lbl-1",
        task_cid="task:labels",
        disposition="closed_deterministic",
    )
    labels = attempt.metric_labels()
    assert labels["disposition"] == "closed_deterministic"
    assert labels["provider_authorized"] == "false"
    assert labels["task_cid"] == "task:labels"
    assert labels["attempt_id"] == "lbl-1"


def test_residual_measured_zero_calls_is_allowed_when_sensor_says_so() -> None:
    """Residual may legitimately observe zero calls; that is measured, not policy."""

    attempt = attribute_attempt(
        attempt_id="res-zero-measured",
        task_cid="task:res",
        disposition="residual_llm_authorized",
        provider_calls=0,
        input_tokens=0,
        output_tokens=0,
    )
    assert attempt.provider_calls.is_measured
    assert attempt.provider_calls.value == 0
    assert attempt.provider_calls.sensor_id == "sensor:observed@1"


def test_explicit_unavailable_sample_preserved_for_residual() -> None:
    sample = ProviderTelemetrySample.unavailable(
        "provider_calls",
        UnavailableReason.COLLECTION_FAILED,
    )
    attempt = attribute_attempt(
        attempt_id="res-unavail",
        task_cid="task:res",
        disposition="residual_llm_authorized",
        provider_calls=sample,
    )
    assert attempt.provider_calls.is_unavailable
    assert (
        attempt.provider_calls.reason_code
        == UnavailableReason.COLLECTION_FAILED.value
    )


def test_collector_to_dict_includes_attempts_and_report_fields() -> None:
    metrics = LlmAvoidanceMetrics()
    metrics.record_attempt(
        attempt_id="ser-1",
        task_cid="task:ser",
        disposition="closed_deterministic",
    )
    payload = metrics.to_dict()
    assert payload["interface"] == LLM_AVOIDANCE_METRICS_INTERFACE
    assert payload["completion_authority"] is False
    assert len(payload["attempts"]) == 1
    assert payload["attempts"][0]["disposition"] == "closed_deterministic"
    assert payload["measured_provider_calls_total"] == 0
