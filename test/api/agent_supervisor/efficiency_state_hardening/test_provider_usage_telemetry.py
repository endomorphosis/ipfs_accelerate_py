"""ASEH-011: provider/model usage mapped onto admitted causal task spans."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import (
    AttributionRole,
    BenchmarkCausalSpan,
    BenchmarkProviderBinding,
    BenchmarkTelemetryError,
    BenchmarkTelemetrySession,
    EstimatorMethod,
    MODEL_CALL_CLASSES,
    ModelCallClass,
    PROVIDER_USAGE_METRIC_NAMES,
    ProviderUsageDisposition,
    ProviderUsageQuarantineReason,
    ProviderUsageRecord,
    SampleStatus,
    SpanKind,
    TelemetrySample,
    UNIT_MICROUSD,
    UnavailableReason,
    extract_safe_provider_request_ids,
    is_safe_provider_request_id,
    map_provider_response_to_admitted_span,
    project_provider_usage_samples,
    provider_response_contains_credentials,
)


PRICE_SNAPSHOT = "sha256:" + ("ab" * 32)


def _provider() -> BenchmarkProviderBinding:
    return BenchmarkProviderBinding(
        provider_id="grok_cli",
        model_id="grok-4.6",
        model_revision="grok-4.6-2026-08-15",
        tokenizer_id="tokenizer:grok-native",
        endpoint_id="endpoint:grok/v1",
        max_context_tokens=128_000,
    )


def _task_span(**overrides: object) -> BenchmarkCausalSpan:
    fields = dict(
        span_id="span:task-aseh-011",
        kind=SpanKind.TASK,
        run_id="run:aseh-011",
        case_id="case:provider-usage",
        arm_id="arm:candidate",
        task_id="ASEH-011",
        attempt=1,
        process_id="pid:11",
        role=AttributionRole.WORKER,
        provider=_provider(),
        started_at_mono_ns=1_000,
        finished_at_mono_ns=2_000,
        monotonic_clock=True,
    )
    fields.update(overrides)
    return BenchmarkCausalSpan(**fields)  # type: ignore[arg-type]


def _session(span: BenchmarkCausalSpan | None = None) -> tuple[
    BenchmarkTelemetrySession, BenchmarkCausalSpan
]:
    root = BenchmarkCausalSpan(
        span_id="span:run-aseh-011",
        kind=SpanKind.RUN,
        run_id="run:aseh-011",
        case_id="case:provider-usage",
        arm_id="arm:candidate",
        task_id="ASEH-011",
        attempt=0,
        process_id="pid:root",
        role=AttributionRole.ROOT,
        provider=_provider(),
        started_at_mono_ns=1,
        finished_at_mono_ns=9,
        monotonic_clock=True,
    )
    session = BenchmarkTelemetrySession(root)
    session.admit_task_span(root)
    task = span if span is not None else _task_span(
        parent_span_id=root.span_id,
        ancestry=(root.span_id,),
    )
    session.admit_task_span(task)
    return session, task


def _reported_response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chatcmpl-aseh011-safe",
        "provider_id": "grok_cli",
        "model": "grok-4.6",
        "model_revision": "grok-4.6-2026-08-15",
        "model_class": ModelCallClass.REMOTE_FRONTIER_MODEL.value,
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 16},
            "completion_tokens_details": {"reasoning_tokens": 8},
            "cost_microusd": 2500,
        },
    }
    payload.update(overrides)
    return payload


def _labeled_estimate(value: int = 3400) -> dict[str, object]:
    return {
        "truth_state": "estimated",
        "value": value,
        "unit": UNIT_MICROUSD,
        "estimator_id": "aseh-price-snapshot",
        "method": EstimatorMethod.TOKEN_PRICE_SNAPSHOT.value,
        "price_snapshot_identity": PRICE_SNAPSHOT,
    }


def test_reported_usage_maps_causally_to_admitted_task_span() -> None:
    session, span = _session()
    record = session.record_provider_response(
        _reported_response(),
        span=span,
        record_id="usage:reported",
        labeled_estimate=_labeled_estimate(),
        model_class=ModelCallClass.REMOTE_FRONTIER_MODEL,
    )
    assert record.admitted
    assert record.disposition is ProviderUsageDisposition.ADMITTED
    assert record.span_id == span.span_id
    assert record.task_id == "ASEH-011"
    assert record.provider_id == "grok_cli"
    assert record.model_id == "grok-4.6"
    assert record.model_revision == "grok-4.6-2026-08-15"
    assert record.safe_request_ids == ("chatcmpl-aseh011-safe",)
    assert record.input_tokens.status is SampleStatus.MEASURED
    assert record.input_tokens.value == 120
    assert record.output_tokens.value == 40
    assert record.cached_input_tokens.value == 16
    assert record.reasoning_tokens.value == 8
    assert record.number_of_calls.value == 1
    assert record.provider_reported_charge.value == 2500
    assert record.provider_reported_charge.status is SampleStatus.MEASURED
    assert record.token_based_estimated_charge.status is SampleStatus.ESTIMATED
    assert record.token_based_estimated_charge.value == 3400
    assert record.token_based_estimated_charge.method == "token_price_snapshot"
    assert "sensor_id" not in record.token_based_estimated_charge.to_envelope()
    classes = {item.metric_name: item for item in record.calls_by_model_class}
    assert classes["calls_remote_frontier_model"].value == 1
    assert classes["calls_deterministic"].value == 0
    assert classes["calls_deterministic"].status is SampleStatus.MEASURED
    measurement = record.to_resource_measurement()
    assert measurement.span.span_id == span.span_id
    assert measurement.span.task_id == "ASEH-011"
    assert session.measurements[0].measurement_id == measurement.measurement_id
    samples = project_provider_usage_samples(record)
    assert set(PROVIDER_USAGE_METRIC_NAMES) <= set(samples)
    receipt = session.seal_receipt()
    assert measurement.measurement_id in {
        item.measurement_id for item in receipt.measurements
    }
    bound = next(
        item for item in receipt.measurements if item.measurement_id == measurement.measurement_id
    )
    assert bound.span.span_id == span.span_id
    assert bound.span.task_id == "ASEH-011"
    assert bound.require_sample("provider_reported_charge_microusd").status is (
        SampleStatus.MEASURED
    )
    assert bound.require_sample("token_based_estimated_charge_microusd").status is (
        SampleStatus.ESTIMATED
    )
    certificates = receipt.certify_all()
    assert certificates and all(item.certified for item in certificates)


def test_provider_call_child_inherits_admitted_task_span() -> None:
    session, task = _session()
    call = task.child(
        span_id="span:provider-call-1",
        kind=SpanKind.PROVIDER_CALL,
        role=AttributionRole.PROVIDER,
        process_id="pid:provider-call",
        started_at_mono_ns=task.started_at_mono_ns + 1,
        finished_at_mono_ns=task.finished_at_mono_ns - 1,
    )
    session.register_span(call)
    record = map_provider_response_to_admitted_span(
        _reported_response(),
        call,
        session=session,
        record_id="usage:child",
        model_class="remote_frontier_model",
    )
    assert record.admitted
    assert record.span_id == call.span_id
    assert record.task_id == task.task_id
    assert call.span_id not in session.admitted_usage_span_ids
    assert task.span_id in session.admitted_usage_span_ids


def test_absent_usage_and_cost_fields_remain_unavailable_not_zero() -> None:
    session, span = _session()
    record = map_provider_response_to_admitted_span(
        {"id": "req_only-identity", "model": "grok-4.6"},
        span,
        session=session,
        record_id="usage:absent",
    )
    assert record.admitted
    for sample in (
        record.input_tokens,
        record.output_tokens,
        record.cached_input_tokens,
        record.reasoning_tokens,
        record.provider_reported_charge,
        record.token_based_estimated_charge,
    ):
        assert sample.status is SampleStatus.UNAVAILABLE
        assert sample.reason_code == UnavailableReason.NOT_REPORTED.value
        envelope = sample.to_envelope()
        assert "value" not in envelope
        assert envelope.get("value", "missing") != 0
        quantity = sample.to_quantity_envelope()
        assert quantity["truth_state"] == "unavailable"
        assert "value" not in quantity
        assert "count" not in quantity
    assert record.number_of_calls.status is SampleStatus.MEASURED
    assert record.number_of_calls.value == 1
    for item in record.calls_by_model_class:
        assert item.status is SampleStatus.UNAVAILABLE
        assert "value" not in item.to_envelope()
    assert "provider_native_input_tokens" in record.unavailable_fields()
    assert "provider_reported_charge_microusd" in record.unavailable_fields()


def test_explicit_zero_tokens_are_measured_with_sensor() -> None:
    session, span = _session()
    record = map_provider_response_to_admitted_span(
        {
            "id": "req_zero-cache",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 0,
                "cached_input_tokens": 0,
            },
        },
        span,
        session=session,
        record_id="usage:zero",
        model_class="remote_standard_model",
    )
    assert record.output_tokens.status is SampleStatus.MEASURED
    assert record.output_tokens.value == 0
    assert record.output_tokens.sensor_id
    assert record.cached_input_tokens.status is SampleStatus.MEASURED
    assert record.cached_input_tokens.value == 0
    assert record.reasoning_tokens.status is SampleStatus.UNAVAILABLE
    assert record.provider_reported_charge.status is SampleStatus.UNAVAILABLE


def test_unbound_usage_is_quarantined() -> None:
    span = _task_span()
    record = map_provider_response_to_admitted_span(
        _reported_response(),
        span,
        record_id="usage:unbound",
    )
    assert record.disposition is ProviderUsageDisposition.QUARANTINED
    assert record.quarantine_reason == ProviderUsageQuarantineReason.UNBOUND_USAGE.value
    assert record.input_tokens.status is SampleStatus.UNAVAILABLE
    assert record.input_tokens.reason_code == UnavailableReason.NOT_ADMITTED.value
    assert "value" not in record.input_tokens.to_envelope()
    with pytest.raises(BenchmarkTelemetryError, match="cannot seal"):
        record.to_resource_measurement()


def test_missing_causal_task_identity_is_quarantined() -> None:
    session, _task = _session()
    anonymous = _task_span(
        span_id="span:anonymous",
        task_id="",
        process_id="pid:anonymous",
        parent_span_id="span:run-aseh-011",
        ancestry=("span:run-aseh-011",),
    )
    session.register_span(anonymous)
    record = map_provider_response_to_admitted_span(
        _reported_response(),
        anonymous,
        session=session,
        record_id="usage:anonymous",
    )
    assert record.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        record.quarantine_reason
        == ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY.value
    )
    none_record = map_provider_response_to_admitted_span(
        _reported_response(),
        None,
        session=session,
        record_id="usage:none",
    )
    assert (
        none_record.quarantine_reason
        == ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY.value
    )
    with pytest.raises(BenchmarkTelemetryError, match="missing causal task identity"):
        session.admit_task_span(anonymous)


def test_credential_leakage_quarantines_and_never_persists_secrets() -> None:
    session, span = _session()
    leaked = _reported_response(api_key="sk-live-secret-material")
    record = map_provider_response_to_admitted_span(
        leaked,
        span,
        session=session,
        record_id="usage:leaked",
    )
    assert record.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        record.quarantine_reason
        == ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE.value
    )
    encoded = record.to_json()
    assert "sk-live-secret-material" not in encoded
    assert "api_key" not in encoded
    assert record.safe_request_ids == ()
    jwt_response = _reported_response(
        id="aaaaaaaabbbbbbbb.ccccccccdddddddd.eeeeeeeeffffffff"
    )
    jwt_record = map_provider_response_to_admitted_span(
        jwt_response,
        span,
        session=session,
        record_id="usage:jwt",
    )
    assert jwt_record.quarantine_reason == (
        ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE.value
    )
    assert provider_response_contains_credentials(leaked)
    assert not provider_response_contains_credentials(_reported_response())


def test_estimated_as_measured_is_quarantined() -> None:
    session, span = _session()
    record = map_provider_response_to_admitted_span(
        _reported_response(),
        span,
        session=session,
        record_id="usage:mislabel",
        labeled_estimate={
            "truth_state": "measured",
            "value": 99,
            "unit": UNIT_MICROUSD,
            "sensor_id": "sensor:fake",
            "estimator_id": "aseh-price-snapshot",
            "method": "token_price_snapshot",
            "price_snapshot_identity": PRICE_SNAPSHOT,
        },
    )
    assert record.disposition is ProviderUsageDisposition.QUARANTINED
    assert record.quarantine_reason == (
        ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED.value
    )
    nested = _reported_response(
        estimated_charge={
            "status": "measured",
            "value": 12,
            "sensor_id": "sensor:wrong",
        }
    )
    nested_record = map_provider_response_to_admitted_span(
        nested,
        span,
        session=session,
        record_id="usage:nested-mislabel",
    )
    assert nested_record.quarantine_reason == (
        ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED.value
    )


def test_estimated_sample_cannot_round_trip_as_measured() -> None:
    sample = TelemetrySample.estimated(
        "token_based_estimated_charge_microusd",
        12,
        unit=UNIT_MICROUSD,
        estimator_id="aseh-price-snapshot",
        method="token_price_snapshot",
        price_snapshot_identity=PRICE_SNAPSHOT,
    )
    payload = sample.to_record()
    payload["status"] = SampleStatus.MEASURED.value
    payload.pop("estimator_id", None)
    payload.pop("method", None)
    payload.pop("price_snapshot_identity", None)
    payload["sensor_id"] = "sensor:forged"
    with pytest.raises(BenchmarkTelemetryError):
        TelemetrySample.from_dict(payload)
    with pytest.raises(BenchmarkTelemetryError, match="estimator labels"):
        TelemetrySample(
            metric_name="provider_reported_charge_microusd",
            status=SampleStatus.MEASURED,
            sensor_id="sensor:x",
            unit=UNIT_MICROUSD,
            value=1,
            estimator_id="aseh-price-snapshot",
            method="token_price_snapshot",
        )


def test_unavailable_usage_cannot_encode_numeric_zero() -> None:
    with pytest.raises(BenchmarkTelemetryError, match="numeric value"):
        TelemetrySample(
            metric_name="provider_native_input_tokens",
            status=SampleStatus.UNAVAILABLE,
            sensor_id="sensor:x",
            reason_code=UnavailableReason.NOT_REPORTED.value,
            value=0,
            unit="tokens",
        )


def test_safe_request_ids_are_filtered() -> None:
    assert is_safe_provider_request_id("chatcmpl-aseh011-safe")
    assert is_safe_provider_request_id("req_01HZX")
    assert not is_safe_provider_request_id("sk-live-secret")
    assert not is_safe_provider_request_id("Bearer abc.def")
    ids = extract_safe_provider_request_ids(
        {
            "id": "chatcmpl-keep",
            "headers": {"x-request-id": "req_header-1"},
            "nested": {"request_id": "sk-drop-this"},
        }
    )
    assert ids == ("chatcmpl-keep", "req_header-1")


def test_record_round_trips_and_rejects_unknown_fields() -> None:
    session, span = _session()
    record = map_provider_response_to_admitted_span(
        _reported_response(),
        span,
        session=session,
        record_id="usage:round-trip",
        labeled_estimate=_labeled_estimate(),
        model_class="remote_frontier_model",
    )
    restored = ProviderUsageRecord.from_dict(record.to_record())
    assert restored.content_id == record.content_id
    assert restored.model_revision == record.model_revision
    payload = record.to_record()
    payload["unexpected"] = 1
    with pytest.raises(BenchmarkTelemetryError, match="unknown fields"):
        ProviderUsageRecord.from_dict(payload)


def test_model_use_projection_keeps_unavailable_distinct() -> None:
    session, span = _session()
    record = map_provider_response_to_admitted_span(
        _reported_response(),
        span,
        session=session,
        record_id="usage:model-use",
        labeled_estimate=_labeled_estimate(),
        model_class="remote_frontier_model",
    )
    model_use = record.to_model_use_fields()
    assert set(model_use) >= {
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "number_of_calls",
        "calls_by_model_class",
        "safe_provider_request_ids",
        "provider_reported_usage",
    }
    assert model_use["input_tokens"]["truth_state"] == "measured"
    assert model_use["safe_provider_request_ids"]["values"] == [
        "chatcmpl-aseh011-safe"
    ]
    assert set(model_use["calls_by_model_class"]) == set(MODEL_CALL_CLASSES)
    quarantined = map_provider_response_to_admitted_span(
        _reported_response(),
        span,
        record_id="usage:model-use-unbound",
    )
    quarantined_use = quarantined.to_model_use_fields()
    assert quarantined_use["provider_reported_usage"]["truth_state"] == "unavailable"
    assert "value" not in quarantined_use["input_tokens"]


def test_non_integer_usage_fails_closed() -> None:
    session, span = _session()
    with pytest.raises(BenchmarkTelemetryError, match="integer"):
        map_provider_response_to_admitted_span(
            {"id": "req_float", "usage": {"prompt_tokens": 1.5}},
            span,
            session=session,
            record_id="usage:float",
        )
    with pytest.raises(BenchmarkTelemetryError, match="integer"):
        map_provider_response_to_admitted_span(
            {"id": "req_bool", "usage": {"prompt_tokens": True}},
            span,
            session=session,
            record_id="usage:bool",
        )


def test_session_does_not_seal_quarantined_measurements() -> None:
    session, span = _session()
    leaked = session.record_provider_response(
        _reported_response(authorization="Bearer leaked"),
        span=span,
        record_id="usage:session-leaked",
    )
    assert leaked.disposition is ProviderUsageDisposition.QUARANTINED
    assert session.measurements == ()
    assert session.provider_usage[0].record_id == leaked.record_id
    admitted = session.record_provider_response(
        _reported_response(),
        span=span,
        record_id="usage:session-admitted",
        model_class="remote_frontier_model",
    )
    assert admitted.admitted
    assert len(session.measurements) == 1
    assert session.measurements[0].span.span_id == span.span_id
