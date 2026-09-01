"""ASEH-011 provider/model usage telemetry mapped to admitted causal spans."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import (
    PROVIDER_MODEL_CLASSES,
    PROVIDER_USAGE_METRIC_NAMES,
    PROVIDER_USAGE_OBSERVATION_INTERFACE,
    AttributionRole,
    BenchmarkCausalSpan,
    BenchmarkHardwareProfile,
    BenchmarkProviderBinding,
    BenchmarkTelemetryError,
    BenchmarkTelemetrySession,
    ProviderUsageDisposition,
    ProviderUsageObservation,
    ProviderUsageQuarantineReason,
    SampleStatus,
    SpanKind,
    TelemetrySample,
    UnavailableReason,
    build_provider_usage_measurement,
    certify_measurement_from_source_spans,
    map_provider_response_to_span,
    project_provider_usage_samples,
)


def _hardware() -> BenchmarkHardwareProfile:
    return BenchmarkHardwareProfile(
        profile_id="hw:aseh-011",
        hostname_alias="host-alias-011",
        cpu_model_id="cpu:test-x86",
        cpu_count=4,
        memory_bytes=8 * 1024**3,
        accelerator_present=False,
        platform="linux",
    )


def _provider() -> BenchmarkProviderBinding:
    return BenchmarkProviderBinding(
        provider_id="provider:grok_cli",
        model_id="model:grok-4.6",
        model_revision="grok-4.6-2026-08-25",
        tokenizer_id="tokenizer:grok-native",
        endpoint_id="endpoint:grok/v1",
        max_context_tokens=128_000,
    )


def _task_span(**overrides: object) -> BenchmarkCausalSpan:
    fields = dict(
        span_id="span:task-aseh-011",
        kind=SpanKind.TASK,
        run_id="run:aseh-011",
        case_id="case:hermetic-1",
        arm_id="arm:sealed-current",
        task_id="ASEH-011",
        attempt=1,
        process_id="pid:1",
        role=AttributionRole.WORKER,
        provider=_provider(),
        hardware=_hardware(),
        started_at_mono_ns=1_000_000_000,
        finished_at_mono_ns=3_000_000_000,
        monotonic_clock=True,
    )
    fields.update(overrides)
    return BenchmarkCausalSpan(**fields)  # type: ignore[arg-type]


def _reported_response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "req_safe_011",
        "provider_id": "provider:grok_cli",
        "model": "model:grok-4.6",
        "model_revision": "grok-4.6-2026-08-25",
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 8},
            "completion_tokens_details": {"reasoning_tokens": 5},
            "cost_microusd": 2100,
        },
    }
    payload.update(overrides)
    return payload


def _assert_unavailable(sample: TelemetrySample) -> None:
    assert sample.status is SampleStatus.UNAVAILABLE
    assert sample.reason_code
    assert sample.unit == ""
    envelope = sample.to_envelope()
    assert "value" not in envelope
    assert "unit" not in envelope


def test_provider_response_maps_causally_to_admitted_task_span() -> None:
    root = _task_span(span_id="span:run-root", kind=SpanKind.RUN, role=AttributionRole.ROOT)
    session = BenchmarkTelemetrySession(root)
    span = root.child(
        span_id="span:task-aseh-011",
        kind=SpanKind.TASK,
        role=AttributionRole.WORKER,
        task_id="ASEH-011",
        started_at_mono_ns=root.started_at_mono_ns + 10,
        finished_at_mono_ns=root.finished_at_mono_ns - 10,
    )
    assert span.parent_span_id == root.span_id
    assert span.process_id == root.process_id
    session.register_span(span)

    observation = session.record_provider_response(
        _reported_response(),
        span,
        model_class="remote_frontier_model",
        estimated_charge_microusd=3400,
        estimated_charge_method="token_price_snapshot",
        estimator_id="estimator:aseh-price-snapshot",
        price_snapshot_identity="sha256:" + "ab" * 32,
        observation_id="obs:aseh-011-admitted",
    )

    assert observation.disposition is ProviderUsageDisposition.ADMITTED
    assert observation.bound_to(span)
    assert observation.span_id == span.span_id
    assert observation.span_content_id == span.content_id
    assert observation.task_id == "ASEH-011"
    assert observation.run_id == span.run_id
    assert observation.INTERFACE == PROVIDER_USAGE_OBSERVATION_INTERFACE
    assert session.provider_observations == (observation,)
    measurement = session.measurements[0]
    assert measurement.span.span_id == span.span_id
    cert = certify_measurement_from_source_spans(measurement, session.spans)
    assert cert.certified is True


def test_exact_provider_model_revision_and_safe_request_ids() -> None:
    span = _task_span()
    observation = map_provider_response_to_span(
        _reported_response(),
        span,
        model_class="remote_frontier_model",
    )
    assert observation.provider_id == "provider:grok_cli"
    assert observation.model_id == "model:grok-4.6"
    assert observation.model_revision == "grok-4.6-2026-08-25"
    assert observation.safe_request_ids == ("req_safe_011",)
    assert (
        observation.require_sample("model_revision_identity").status
        is SampleStatus.MEASURED
    )
    assert (
        observation.require_sample("safe_provider_request_id_identity").status
        is SampleStatus.MEASURED
    )


def test_tokens_measured_where_reported_else_unavailable() -> None:
    span = _task_span()
    reported = map_provider_response_to_span(_reported_response(), span)
    assert reported.require_sample("provider_native_input_tokens").value == 40
    assert reported.require_sample("provider_native_output_tokens").value == 12
    assert reported.require_sample("provider_native_cached_tokens").value == 8
    assert reported.require_sample("provider_native_reasoning_tokens").value == 5
    assert reported.require_sample("provider_reported_charge").value == 2100

    omitted = map_provider_response_to_span({"id": "req_partial"}, span)
    for name in (
        "provider_native_input_tokens",
        "provider_native_output_tokens",
        "provider_native_cached_tokens",
        "provider_native_reasoning_tokens",
        "provider_reported_charge",
        "token_based_estimated_charge",
    ):
        sample = omitted.require_sample(name)
        _assert_unavailable(sample)
        assert sample.reason_code in {
            UnavailableReason.PROVIDER_OMITTED.value,
            UnavailableReason.SENSOR_ABSENT.value,
        }


def test_absent_usage_and_cost_never_encode_numeric_zero() -> None:
    span = _task_span()
    observation = map_provider_response_to_span({"model": "model:grok-4.6"}, span)
    for name in PROVIDER_USAGE_METRIC_NAMES:
        sample = observation.require_sample(name)
        if sample.status is SampleStatus.UNAVAILABLE:
            envelope = sample.to_envelope()
            assert "value" not in envelope
            assert sample.value == 0
            assert sample.unit == ""
    assert observation.require_sample("provider_native_input_tokens").status is (
        SampleStatus.UNAVAILABLE
    )
    assert observation.require_sample("provider_reported_charge").status is (
        SampleStatus.UNAVAILABLE
    )
    # A present zero is measured with a sensor, not an unavailable stand-in.
    zeroed = map_provider_response_to_span(
        {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost_microusd": 0}},
        span,
    )
    for name in (
        "provider_native_input_tokens",
        "provider_native_output_tokens",
        "provider_reported_charge",
    ):
        sample = zeroed.require_sample(name)
        assert sample.status is SampleStatus.MEASURED
        assert sample.value == 0
        assert sample.sensor_id


def test_call_counts_and_classes_are_recorded() -> None:
    span = _task_span()
    classified = map_provider_response_to_span(
        _reported_response(),
        span,
        model_class="remote_frontier_model",
    )
    assert classified.require_sample("model_call_count").value == 1
    assert classified.require_sample("calls_remote_frontier_model").value == 1
    for name in PROVIDER_MODEL_CLASSES:
        sample = classified.require_sample(f"calls_{name}")
        assert sample.status is SampleStatus.MEASURED
        assert sample.value == (1 if name == "remote_frontier_model" else 0)

    unclassified = map_provider_response_to_span(_reported_response(), span)
    assert unclassified.require_sample("model_call_count").value == 1
    for name in PROVIDER_MODEL_CLASSES:
        _assert_unavailable(unclassified.require_sample(f"calls_{name}"))


def test_reported_charge_and_labeled_estimate_are_distinct() -> None:
    span = _task_span()
    observation = map_provider_response_to_span(
        _reported_response(),
        span,
        estimated_charge_microusd=3400,
        estimated_charge_method="token_price_snapshot",
        estimator_id="estimator:aseh-price-snapshot",
    )
    reported = observation.require_sample("provider_reported_charge")
    estimated = observation.require_sample("token_based_estimated_charge")
    assert reported.status is SampleStatus.MEASURED
    assert reported.value == 2100
    assert "estimate_method" not in reported.to_envelope()
    assert estimated.status is SampleStatus.ESTIMATED
    assert estimated.value == 3400
    assert estimated.estimate_method == "token_price_snapshot"
    assert estimated.to_envelope()["status"] == "estimated"
    restored = TelemetrySample.from_dict(estimated.to_record())
    assert restored.status is SampleStatus.ESTIMATED
    assert restored.content_id == estimated.content_id


def test_unbound_usage_is_quarantined() -> None:
    root = _task_span(span_id="span:root", kind=SpanKind.RUN, role=AttributionRole.ROOT)
    session = BenchmarkTelemetrySession(root)
    foreign = _task_span(span_id="span:foreign-task")
    observation = session.record_provider_response(
        _reported_response(),
        foreign,
        observation_id="obs:unbound",
    )
    assert observation.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        observation.quarantine_reason
        == ProviderUsageQuarantineReason.UNBOUND_USAGE.value
    )
    for sample in observation.samples:
        _assert_unavailable(sample)
        assert sample.reason_code == UnavailableReason.QUARANTINED.value
    assert session.measurements == ()
    with pytest.raises(BenchmarkTelemetryError, match="quarantined"):
        build_provider_usage_measurement(
            measurement_id="meas:rejected",
            span=foreign,
            observation=observation,
        )


def test_credential_leakage_is_quarantined() -> None:
    span = _task_span()
    leaked = map_provider_response_to_span(
        _reported_response(authorization="not-empty-credential-field"),
        span,
    )
    assert leaked.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        leaked.quarantine_reason
        == ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE.value
    )
    assert leaked.safe_request_ids == ()
    serialized = json.dumps(leaked.to_record())
    assert "not-empty-credential-field" not in serialized
    assert "authorization" not in serialized

    secret_request = map_provider_response_to_span(
        {"id": "Bearer leaked-provider-token"},
        span,
    )
    assert secret_request.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        secret_request.quarantine_reason
        == ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE.value
    )
    assert "leaked-provider-token" not in json.dumps(secret_request.to_record())


def test_estimated_as_measured_is_quarantined() -> None:
    span = _task_span()
    flagged = map_provider_response_to_span(
        _reported_response(),
        span,
        estimated_charge_microusd=12,
        estimated_as_measured=True,
    )
    assert flagged.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        flagged.quarantine_reason
        == ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED.value
    )

    labeled = map_provider_response_to_span(
        {
            "id": "req_estimate",
            "token_based_estimated_charge": {
                "truth_state": "measured",
                "value": 99,
            },
        },
        span,
    )
    assert labeled.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        labeled.quarantine_reason
        == ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED.value
    )

    with pytest.raises(BenchmarkTelemetryError, match="cannot carry an estimate"):
        TelemetrySample(
            metric_name="token_based_estimated_charge",
            status=SampleStatus.MEASURED,
            sensor_id="sensor:price",
            unit="microusd",
            value=12,
            estimate_method="token_price_snapshot",
        )


def test_missing_causal_identity_is_quarantined() -> None:
    span = _task_span(task_id="", run_id="run:aseh-011")
    observation = map_provider_response_to_span(_reported_response(), span)
    assert observation.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        observation.quarantine_reason
        == ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY.value
    )
    assert observation.bound_to(span) is False


def test_observation_round_trips_and_rejects_unknown_fields() -> None:
    span = _task_span()
    observation = map_provider_response_to_span(
        _reported_response(),
        span,
        model_class="remote_frontier_model",
        estimated_charge_microusd=100,
        observation_id="obs:round-trip",
    )
    restored = ProviderUsageObservation.from_dict(observation.to_record())
    assert restored.content_id == observation.content_id
    assert restored.safe_request_ids == observation.safe_request_ids
    payload = observation.to_record()
    payload["unexpected"] = 1
    with pytest.raises(BenchmarkTelemetryError, match="unknown fields"):
        ProviderUsageObservation.from_dict(payload)
    with pytest.raises(BenchmarkTelemetryError, match="model_class"):
        map_provider_response_to_span(
            _reported_response(),
            span,
            model_class="frontier-plus",
        )


def test_projected_samples_preserve_unavailable_and_estimates() -> None:
    span = _task_span()
    observation = map_provider_response_to_span(
        {"id": "req_safe_011", "usage": {"prompt_tokens": 3}},
        span,
        estimated_charge_microusd=50,
        estimated_charge_method="provider_quote",
        estimator_id="estimator:quote",
    )
    samples = project_provider_usage_samples(observation)
    assert set(samples) == set(PROVIDER_USAGE_METRIC_NAMES)
    assert samples["provider_native_input_tokens"].value == 3
    _assert_unavailable(samples["provider_native_output_tokens"])
    _assert_unavailable(samples["provider_native_cached_tokens"])
    _assert_unavailable(samples["provider_native_reasoning_tokens"])
    _assert_unavailable(samples["provider_reported_charge"])
    assert samples["token_based_estimated_charge"].status is SampleStatus.ESTIMATED
    measurement = build_provider_usage_measurement(
        measurement_id="meas:provider-aseh-011",
        span=span,
        observation=observation,
    )
    assert measurement.require_sample("provider_native_input_tokens").value == 3
    assert certify_measurement_from_source_spans(measurement, (span,)).certified
