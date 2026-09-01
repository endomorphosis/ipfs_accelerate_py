"""ASEH-011: provider/model usage maps onto admitted causal spans."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import (
    MAX_INTEGER,
    MODEL_CLASS_NAMES,
    PROVIDER_USAGE_METRIC_NAMES,
    UNIT_IDENTITY,
    AttributionRole,
    BenchmarkCausalSpan,
    BenchmarkProviderBinding,
    BenchmarkTelemetryError,
    BenchmarkTelemetrySession,
    EstimateMethod,
    ModelClass,
    ProviderUsageDisposition,
    ProviderUsageQuarantineReason,
    ProviderUsageRecord,
    SampleStatus,
    SpanKind,
    TelemetrySample,
    admit_provider_usage,
    certify_measurement_from_source_spans,
    map_provider_response_to_span,
    map_provider_responses_to_span,
)


PRICE_SNAPSHOT = "baguqeera3nas3dp546nrxqg3nr6qdalxzbld2wlbhxzrdt5bz5disttd3w6a"
SECRET_REQUEST = "sk-" + ("a" * 24)


def _provider() -> BenchmarkProviderBinding:
    return BenchmarkProviderBinding(
        provider_id="grok_cli",
        model_id="grok-4.6",
        model_revision="grok-4.6@2026-08-23",
        tokenizer_id="tokenizer:grok-native",
        endpoint_id="endpoint:grok-cli",
        max_context_tokens=128_000,
    )


def _span(**overrides: object) -> BenchmarkCausalSpan:
    fields: dict[str, object] = dict(
        span_id="span:task-aseh-011",
        kind=SpanKind.TASK,
        run_id="run:aseh-011",
        case_id="case:provider-usage",
        arm_id="arm:sealed-current",
        task_id="ASEH-011",
        attempt=1,
        process_id="pid:11",
        role=AttributionRole.PROVIDER,
        provider=_provider(),
        started_at_mono_ns=1_000_000_000,
        finished_at_mono_ns=2_000_000_000,
        monotonic_clock=True,
    )
    fields.update(overrides)
    return BenchmarkCausalSpan(**fields)  # type: ignore[arg-type]


def _response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider_id": "grok_cli",
        "model_id": "grok-4.6",
        "model_revision": "grok-4.6@2026-08-23",
        "model_class": ModelClass.REMOTE_FRONTIER_MODEL.value,
        "request_id": "req_7f3c91ab2e",
        "usage": {
            "input_tokens": 120,
            "output_tokens": 40,
            "cached_tokens": 16,
            "reasoning_tokens": 8,
        },
        "cost": {"reported_charge_microusd": 2500},
        "estimate": {
            "charge_microusd": 3100,
            "method": EstimateMethod.TOKEN_PRICE_SNAPSHOT.value,
            "estimator_id": "aseh-price-snapshot",
            "price_snapshot_identity": PRICE_SNAPSHOT,
        },
    }
    payload.update(overrides)
    return payload


def _assert_unavailable(sample: TelemetrySample) -> None:
    assert sample.status is SampleStatus.UNAVAILABLE
    envelope = sample.to_envelope()
    assert "value" not in envelope
    assert "unit" not in envelope
    assert envelope["reason_code"]
    assert sample.value == 0
    assert sample.unit == ""


def test_provider_response_maps_to_admitted_task_span() -> None:
    span = _span()
    session = BenchmarkTelemetrySession(span)
    record = map_provider_response_to_span(
        _response(),
        span,
        session=session,
        admitted_span_ids=(span.span_id,),
    )
    assert record.admitted is True
    assert record.quarantined is False
    assert record.span is not None
    assert record.span.span_id == span.span_id
    assert record.span.task_id == "ASEH-011"
    assert record.provider_id == "grok_cli"
    assert record.model_id == "grok-4.6"
    assert record.model_revision == "grok-4.6@2026-08-23"
    assert record.model_class == "remote_frontier_model"
    assert record.safe_request_ids == ("req_7f3c91ab2e",)
    assert record.price_snapshot_identity == PRICE_SNAPSHOT

    revision_sample = record.require_sample("provider_model_revision")
    assert revision_sample.status is SampleStatus.MEASURED
    assert revision_sample.unit == UNIT_IDENTITY
    assert 0 <= revision_sample.value <= MAX_INTEGER
    request_sample = record.require_sample("safe_provider_request_id")
    assert request_sample.status is SampleStatus.MEASURED
    assert request_sample.unit == UNIT_IDENTITY
    assert 0 <= request_sample.value <= MAX_INTEGER

    assert record.require_sample("provider_native_input_tokens").value == 120
    assert record.require_sample("provider_native_output_tokens").value == 40
    assert record.require_sample("provider_native_cached_tokens").value == 16
    assert record.require_sample("provider_native_reasoning_tokens").value == 8
    assert record.require_sample("model_call_count").value == 1
    assert record.require_sample("calls_remote_frontier_model").value == 1
    assert record.require_sample("calls_deterministic").value == 0
    assert record.require_sample("provider_reported_charge").status is (
        SampleStatus.MEASURED
    )
    assert record.require_sample("provider_reported_charge").value == 2500
    estimated = record.require_sample("token_based_estimated_charge")
    assert estimated.status is SampleStatus.ESTIMATED
    assert estimated.value == 3100
    assert estimated.method == "token_price_snapshot"
    assert estimated.estimator_id == "aseh-price-snapshot"
    assert estimated.price_snapshot_identity == PRICE_SNAPSHOT
    assert estimated.to_envelope()["status"] == "estimated"

    stored = session.record_provider_usage(record)
    assert stored.usage_id == record.usage_id
    assert session.provider_usages == (record,)
    measurement = session.measurements[0]
    assert measurement.span.span_id == span.span_id
    cert = certify_measurement_from_source_spans(measurement, (span,))
    assert cert.certified is True
    assert admit_provider_usage(record) is record


def test_absent_usage_and_cost_fields_stay_unavailable_not_zero() -> None:
    span = _span()
    record = map_provider_response_to_span(
        {
            "provider_id": "grok_cli",
            "model_id": "grok-4.6",
            "model_class": "remote_frontier_model",
        },
        span,
    )
    for name in (
        "provider_model_revision",
        "safe_provider_request_id",
        "provider_native_input_tokens",
        "provider_native_output_tokens",
        "provider_native_cached_tokens",
        "provider_native_reasoning_tokens",
        "provider_reported_charge",
        "token_based_estimated_charge",
    ):
        _assert_unavailable(record.require_sample(name))
        assert record.require_sample(name).reason_code == "not-reported"
    assert record.require_sample("model_call_count").value == 1
    assert record.model_revision == ""
    assert record.safe_request_ids == ()
    envelopes = record.to_envelopes()
    for name in (
        "provider_native_cached_tokens",
        "provider_native_reasoning_tokens",
        "provider_reported_charge",
        "token_based_estimated_charge",
    ):
        assert "value" not in envelopes[name]
        assert envelopes[name]["status"] == "unavailable"


def test_reported_zero_tokens_remain_measured_with_sensor_receipt() -> None:
    span = _span()
    record = map_provider_response_to_span(
        {
            "provider_id": "grok_cli",
            "model_id": "grok-4.6",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        span,
    )
    incoming = record.require_sample("provider_native_input_tokens")
    assert incoming.status is SampleStatus.MEASURED
    assert incoming.value == 0
    assert incoming.sensor_id
    _assert_unavailable(record.require_sample("provider_native_cached_tokens"))
    _assert_unavailable(record.require_sample("provider_reported_charge"))


def test_nested_openai_usage_details_are_recorded_when_reported() -> None:
    span = _span()
    record = map_provider_response_to_span(
        {
            "provider_id": "grok_cli",
            "model": "grok-4.6",
            "model_revision": "grok-4.6@2026-08-23",
            "id": "chatcmpl-9f21ab",
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 11},
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        },
        span,
    )
    assert record.require_sample("provider_native_input_tokens").value == 80
    assert record.require_sample("provider_native_output_tokens").value == 20
    assert record.require_sample("provider_native_cached_tokens").value == 11
    assert record.require_sample("provider_native_reasoning_tokens").value == 5
    assert record.safe_request_ids == ("chatcmpl-9f21ab",)
    assert record.model_revision == "grok-4.6@2026-08-23"


def test_estimate_does_not_overwrite_unavailable_reported_charge() -> None:
    span = _span()
    record = map_provider_response_to_span(
        {
            "provider_id": "grok_cli",
            "model_id": "grok-4.6",
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "estimate": {
                "charge_microusd": 900,
                "method": "token_price_snapshot",
                "estimator_id": "aseh-price-snapshot",
                "price_snapshot_identity": PRICE_SNAPSHOT,
            },
        },
        span,
    )
    _assert_unavailable(record.require_sample("provider_reported_charge"))
    estimated = record.require_sample("token_based_estimated_charge")
    assert estimated.status is SampleStatus.ESTIMATED
    assert estimated.value == 900
    measurement = record.to_measurement()
    assert measurement.measured_value("provider_reported_charge") is None
    assert measurement.measured_value("token_based_estimated_charge") is None
    assert (
        measurement.require_sample("token_based_estimated_charge").status
        is SampleStatus.ESTIMATED
    )


def test_unlabeled_estimate_stays_unavailable() -> None:
    span = _span()
    record = map_provider_response_to_span(
        {
            "provider_id": "grok_cli",
            "model_id": "grok-4.6",
            "estimate": {"charge_microusd": 12},
        },
        span,
    )
    _assert_unavailable(record.require_sample("token_based_estimated_charge"))


def test_unbound_usage_is_quarantined() -> None:
    span = _span()
    session = BenchmarkTelemetrySession(span)
    unbound = map_provider_response_to_span(_response(), None, session=session)
    assert unbound.quarantined is True
    assert (
        ProviderUsageQuarantineReason.UNBOUND_USAGE.value
        in unbound.quarantine_reasons
    )
    assert unbound.safe_request_ids == ()
    assert unbound.samples == ()
    with pytest.raises(BenchmarkTelemetryError, match="quarantined"):
        admit_provider_usage(unbound)
    with pytest.raises(BenchmarkTelemetryError, match="measurement"):
        unbound.to_measurement()

    foreign = _span(span_id="span:foreign", task_id="ASEH-011")
    unknown = map_provider_response_to_span(
        _response(),
        foreign,
        session=session,
        admitted_span_ids=(span.span_id,),
    )
    assert unknown.quarantined is True
    assert "unbound-usage" in unknown.quarantine_reasons
    stored = session.record_provider_usage(unknown)
    assert stored.quarantined is True
    assert session.measurements == ()
    assert session.quarantined_provider_usages == (unknown,)


def test_missing_causal_identity_is_quarantined() -> None:
    incomplete = _span(task_id="", run_id="run:aseh-011")
    record = map_provider_response_to_span(_response(), incomplete)
    assert record.quarantined is True
    assert (
        ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY.value
        in record.quarantine_reasons
    )
    assert record.span is None


def test_provider_binding_mismatch_is_unbound() -> None:
    span = _span()
    record = map_provider_response_to_span(
        _response(provider_id="codex", model_id="gpt-5.6-terra"),
        span,
    )
    assert record.quarantined is True
    assert "unbound-usage" in record.quarantine_reasons


def test_credential_shaped_request_id_quarantines_without_leakage() -> None:
    span = _span()
    record = map_provider_response_to_span(
        _response(request_id=SECRET_REQUEST),
        span,
    )
    assert record.quarantined is True
    assert (
        ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE.value
        in record.quarantine_reasons
    )
    encoded = record.to_json()
    assert SECRET_REQUEST not in encoded
    assert "sk-" not in encoded
    assert record.safe_request_ids == ()
    payload_with_key = _response()
    payload_with_key["api_key"] = SECRET_REQUEST
    leaked = map_provider_response_to_span(payload_with_key, span)
    assert leaked.quarantined is True
    assert "credential-leakage" in leaked.quarantine_reasons
    assert SECRET_REQUEST not in leaked.to_json()


def test_estimated_as_measured_is_quarantined() -> None:
    span = _span()
    labeled_measured = map_provider_response_to_span(
        _response(
            estimate={
                "charge_microusd": 12,
                "method": "token_price_snapshot",
                "estimator_id": "aseh-price-snapshot",
                "truth_state": "measured",
            }
        ),
        span,
    )
    assert labeled_measured.quarantined is True
    assert "estimated-as-measured" in labeled_measured.quarantine_reasons

    reported_as_estimate = map_provider_response_to_span(
        {
            "provider_id": "grok_cli",
            "model_id": "grok-4.6",
            "cost": {
                "provider_reported_charge": {
                    "truth_state": "estimated",
                    "value": 12,
                }
            },
        },
        span,
    )
    assert reported_as_estimate.quarantined is True
    assert "estimated-as-measured" in reported_as_estimate.quarantine_reasons

    with pytest.raises(BenchmarkTelemetryError, match="estimate labels"):
        TelemetrySample(
            metric_name="provider_reported_charge",
            status=SampleStatus.MEASURED,
            sensor_id="sensor:cost",
            unit="microusd",
            value=1,
            estimator_id="aseh-price-snapshot",
            method="token_price_snapshot",
        )
    with pytest.raises(BenchmarkTelemetryError, match="estimated-as-measured"):
        ProviderUsageRecord(
            usage_id="usage:bad-estimate",
            disposition=ProviderUsageDisposition.ADMITTED,
            span=_span(),
            samples=(
                TelemetrySample.measured(
                    "token_based_estimated_charge",
                    1,
                    unit="microusd",
                    sensor_id="sensor:est",
                ),
            ),
        )


def test_call_counts_and_classes_aggregate_without_inventing_zeros() -> None:
    span = _span()
    frontier = _response(request_id="req_front_1")
    specialist = _response(
        request_id="req_small_1",
        model_class=ModelClass.LOCAL_SMALL_MODEL.value,
        usage={"input_tokens": 5, "output_tokens": 2},
        cost={"reported_charge_microusd": 100},
        estimate={
            "charge_microusd": 140,
            "method": "token_price_snapshot",
            "estimator_id": "aseh-price-snapshot",
            "price_snapshot_identity": PRICE_SNAPSHOT,
        },
    )
    combined = map_provider_responses_to_span((frontier, specialist), span)
    assert combined.admitted is True
    assert combined.require_sample("model_call_count").value == 2
    assert combined.require_sample("calls_remote_frontier_model").value == 1
    assert combined.require_sample("calls_local_small_model").value == 1
    assert combined.require_sample("provider_native_input_tokens").value == 125
    assert combined.require_sample("provider_native_output_tokens").value == 42
    assert combined.require_sample("provider_reported_charge").value == 2600
    assert combined.require_sample("token_based_estimated_charge").status is (
        SampleStatus.ESTIMATED
    )
    assert combined.require_sample("token_based_estimated_charge").value == 3240
    _assert_unavailable(combined.require_sample("model_call_class"))
    assert combined.safe_request_ids == ("req_7f3c91ab2e", "req_small_1")

    omitted_cached = _response(
        request_id="req_front_2",
        usage={"input_tokens": 3, "output_tokens": 1},
    )
    mixed = map_provider_responses_to_span((frontier, omitted_cached), span)
    _assert_unavailable(mixed.require_sample("provider_native_cached_tokens"))
    _assert_unavailable(mixed.require_sample("provider_native_reasoning_tokens"))
    assert mixed.require_sample("provider_native_input_tokens").value == 123


def test_aggregate_quarantines_when_any_constituent_is_unbound() -> None:
    span = _span()
    mixed = map_provider_responses_to_span((_response(), _response()), None)
    assert mixed.quarantined is True
    assert "unbound-usage" in mixed.quarantine_reasons


def test_provider_usage_round_trips_and_rejects_unknown_fields() -> None:
    span = _span()
    record = map_provider_response_to_span(_response(), span)
    restored = ProviderUsageRecord.from_dict(record.to_record())
    assert restored.content_id == record.content_id
    assert restored.model_revision == record.model_revision
    assert restored.safe_request_ids == record.safe_request_ids
    estimated = restored.require_sample("token_based_estimated_charge")
    assert estimated.status is SampleStatus.ESTIMATED
    payload = record.to_record()
    payload["unexpected"] = 1
    with pytest.raises(BenchmarkTelemetryError, match="unknown fields"):
        ProviderUsageRecord.from_dict(payload)
    for name in PROVIDER_USAGE_METRIC_NAMES:
        assert record.sample(name) is not None
    for name in MODEL_CLASS_NAMES:
        assert record.sample(f"calls_{name}") is not None


def test_session_attributes_provider_usage_exactly_once() -> None:
    span = _span()
    session = BenchmarkTelemetrySession(span)
    record = map_provider_response_to_span(_response(), span, session=session)
    first = session.record_provider_usage(record)
    again = session.record_provider_usage(record)
    assert again.content_id == first.content_id
    colliding = map_provider_response_to_span(
        _response(usage={"input_tokens": 1, "output_tokens": 1}),
        span,
        session=session,
        usage_id=record.usage_id,
    )
    with pytest.raises(BenchmarkTelemetryError, match="collides"):
        session.record_provider_usage(colliding)
