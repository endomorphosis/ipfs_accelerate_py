"""ASEH-011 provider/model usage mapped onto admitted causal task spans."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import (
    PROVIDER_USAGE_RECORD_INTERFACE,
    PROVIDER_USAGE_SAMPLE_NAMES,
    AttributionRole,
    BenchmarkCausalSpan,
    BenchmarkProviderBinding,
    BenchmarkTelemetryError,
    BenchmarkTelemetrySession,
    LabeledEstimate,
    ProviderUsageDisposition,
    ProviderUsageQuarantineReason,
    ProviderUsageRecord,
    SampleStatus,
    SpanKind,
    UnavailableReason,
    aggregate_provider_usage_for_span,
    extract_safe_request_id,
    map_provider_response_to_span,
    map_provider_responses,
    mono_ns,
)
from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    COST_FIELDS,
    MODEL_CLASSES,
    MODEL_USE_FIELDS,
    build_task_efficiency_receipt,
    collect_unavailable_fields,
    content_identity,
    observed_identity,
)


COMMIT = "755f45475cc2d13dacd8b330036c1d597afeddde"
TREE = "729da9f8293ecfa046a0136381a3d3808f9ed140"
POLICY = "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
OBJECTIVE_REVISION = "baguqeeray6iu7h6kiajjow44w3l6gexkiihui2423wrhtpou22thcm6sqhbq"
TASK_CID = content_identity({"task": "ASEH-011"})
PRICE_SNAPSHOT = content_identity({"price": "aseh-011-snapshot"})


def _provider() -> BenchmarkProviderBinding:
    return BenchmarkProviderBinding(
        provider_id="grok_cli",
        model_id="grok-4.6",
        model_revision="grok-4.6-2026-08-19",
        tokenizer_id="tokenizer:grok-native",
        endpoint_id="endpoint:grok/v1",
        max_context_tokens=128_000,
    )


def _task_span(**overrides: object) -> BenchmarkCausalSpan:
    started = mono_ns()
    fields: dict[str, object] = dict(
        span_id="span:task-aseh-011",
        kind=SpanKind.TASK,
        run_id="run:aseh-011",
        case_id="case:provider-usage",
        arm_id="arm:sealed-current",
        task_id="ASEH-011",
        attempt=1,
        process_id="pid:1",
        role=AttributionRole.ROOT,
        provider=_provider(),
        started_at_mono_ns=started,
        finished_at_mono_ns=started + 5_000_000_000,
        monotonic_clock=True,
    )
    fields.update(overrides)
    return BenchmarkCausalSpan(**fields)  # type: ignore[arg-type]


def _call_span(parent: BenchmarkCausalSpan, *, span_id: str = "span:provider-call-1") -> BenchmarkCausalSpan:
    return parent.child(
        span_id=span_id,
        kind=SpanKind.PROVIDER_CALL,
        role=AttributionRole.PROVIDER,
        started_at_mono_ns=parent.started_at_mono_ns + 10,
        finished_at_mono_ns=parent.finished_at_mono_ns - 10,
    )


def _complete_response(**overrides: object) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "span_id": "span:provider-call-1",
        "task_id": "ASEH-011",
        "provider_id": "grok_cli",
        "model_id": "grok-4.6",
        "model_revision": "grok-4.6-2026-08-19",
        "model_class": "remote_frontier_model",
        "request_id": "req_01JASEH011SAFEID",
        "usage": {
            "input_tokens": 120,
            "output_tokens": 40,
            "cached_input_tokens": 16,
            "reasoning_tokens": 8,
        },
        "charge": {"reported_microusd": 3400},
        "estimate": {
            "value": 3600,
            "unit": "microusd",
            "estimator_id": "aseh-price-snapshot",
            "method": "token_price_snapshot",
            "price_snapshot_identity": PRICE_SNAPSHOT,
            "truth_state": "estimated",
        },
    }
    payload.update(overrides)
    return payload


def _assert_unavailable_not_zero(sample) -> None:
    assert sample.status is SampleStatus.UNAVAILABLE
    assert sample.reason_code
    envelope = sample.to_envelope()
    assert "value" not in envelope
    assert "unit" not in envelope
    assert sample.unit == ""
    assert sample.value == 0


def test_complete_provider_response_maps_causally_to_admitted_span() -> None:
    task = _task_span()
    call = _call_span(task)
    session = BenchmarkTelemetrySession(task)
    session.register_span(call)

    records = session.record_provider_responses((_complete_response(),))
    assert len(records) == 1
    record = records[0]
    assert record.is_admitted
    assert record.disposition is ProviderUsageDisposition.ADMITTED
    assert record.span_id == call.span_id
    assert record.task_id == "ASEH-011"
    assert record.provider_id == "grok_cli"
    assert record.model_id == "grok-4.6"
    assert record.model_revision == "grok-4.6-2026-08-19"
    assert record.model_class == "remote_frontier_model"
    assert record.request_ids == ("req_01JASEH011SAFEID",)
    assert record.require_sample("provider_native_input_tokens").value == 120
    assert record.require_sample("provider_native_output_tokens").value == 40
    assert record.require_sample("provider_native_cached_input_tokens").value == 16
    assert record.require_sample("provider_native_reasoning_tokens").value == 8
    assert record.require_sample("model_call_count").value == 1
    assert record.require_sample("provider_reported_charge_microusd").value == 3400
    assert record.estimates and record.estimates[0].value == 3600
    assert record.estimates[0].method == "token_price_snapshot"
    assert "sensor_id" not in record.estimates[0].to_quantity()
    assert record.call_class_sample("remote_frontier_model").value == 1
    assert record.call_class_sample("deterministic").value == 0

    measurement_ids = {item.measurement_id for item in session.measurements}
    assert f"meas:{record.observation_id}" in measurement_ids
    bound = session.measurements[0]
    assert bound.span.span_id == call.span_id
    assert call.span_id in bound.source_span_ids


def test_openai_shaped_response_reports_nested_cached_and_reasoning_tokens() -> None:
    task = _task_span()
    call = _call_span(task)
    response = {
        "id": "chatcmpl-01JASEH011OPENAI",
        "span_id": call.span_id,
        "task_id": "ASEH-011",
        "model": "grok-4.6",
        "model_class": "remote_frontier_model",
        "system_fingerprint": "fp_aseh011rev",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }
    record = map_provider_response_to_span(response, admitted_spans=(task, call))
    assert record.is_admitted
    assert record.model_revision == "fp_aseh011rev"
    assert record.request_ids == ("chatcmpl-01JASEH011OPENAI",)
    assert record.require_sample("provider_native_input_tokens").value == 50
    assert record.require_sample("provider_native_output_tokens").value == 12
    assert record.require_sample("provider_native_cached_input_tokens").value == 7
    assert record.require_sample("provider_native_reasoning_tokens").value == 3
    _assert_unavailable_not_zero(record.require_sample("provider_reported_charge_microusd"))
    assert not record.estimates
    assert "token_based_estimated_charge_microusd" in record.unavailable_fields()


def test_absent_usage_and_cost_fields_remain_unavailable_not_zero() -> None:
    task = _task_span()
    call = _call_span(task)
    record = map_provider_response_to_span(
        {
            "span_id": call.span_id,
            "task_id": "ASEH-011",
            "model_class": "remote_frontier_model",
            "usage": {"prompt_tokens": 9, "completion_tokens": 2},
        },
        admitted_spans=(task, call),
    )
    assert record.is_admitted
    assert record.require_sample("provider_native_input_tokens").status is SampleStatus.MEASURED
    for name in (
        "provider_native_cached_input_tokens",
        "provider_native_reasoning_tokens",
        "provider_reported_charge_microusd",
    ):
        _assert_unavailable_not_zero(record.require_sample(name))
        assert record.require_sample(name).reason_code == UnavailableReason.PROVIDER_OMITTED.value
    assert not record.request_ids
    assert "safe_provider_request_ids" in record.unavailable_fields()

    fragment = record.to_model_use_fragment()
    assert fragment["cached_input_tokens"]["truth_state"] == "unavailable"
    assert "value" not in fragment["cached_input_tokens"]
    assert "count" not in fragment["cached_input_tokens"]
    cost = record.to_cost_fragment()
    assert cost["provider_reported_charge"]["truth_state"] == "unavailable"
    assert "value" not in cost["provider_reported_charge"]
    assert cost["token_based_estimated_charge"]["truth_state"] == "unavailable"


def test_reported_zero_cached_tokens_are_measured_with_sensor() -> None:
    task = _task_span()
    call = _call_span(task)
    record = map_provider_response_to_span(
        {
            "span_id": call.span_id,
            "usage": {
                "input_tokens": 4,
                "output_tokens": 1,
                "cached_input_tokens": 0,
            },
            "model_class": "local_small_model",
        },
        admitted_spans=(task, call),
    )
    cached = record.require_sample("provider_native_cached_input_tokens")
    assert cached.status is SampleStatus.MEASURED
    assert cached.value == 0
    assert cached.sensor_id
    _assert_unavailable_not_zero(record.require_sample("provider_native_reasoning_tokens"))


def test_task_id_uniquely_binds_when_span_id_is_omitted() -> None:
    task = _task_span()
    record = map_provider_response_to_span(
        {
            "task_id": "ASEH-011",
            "usage": {"input_tokens": 3, "output_tokens": 1},
            "model_class": "remote_standard_model",
        },
        admitted_spans=(task,),
    )
    assert record.is_admitted
    assert record.span_id == task.span_id
    assert record.model_revision == "grok-4.6-2026-08-19"


def test_unbound_usage_is_quarantined_and_not_recorded_on_the_span() -> None:
    task = _task_span()
    session = BenchmarkTelemetrySession(task)
    records = session.record_provider_responses(
        (
            {
                "span_id": "span:never-admitted",
                "task_id": "ASEH-011",
                "usage": {"input_tokens": 99, "output_tokens": 99},
            },
        )
    )
    assert records[0].disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        records[0].quarantine_reason
        == ProviderUsageQuarantineReason.UNBOUND_USAGE.value
    )
    assert session.measurements == ()
    for name in PROVIDER_USAGE_SAMPLE_NAMES:
        _assert_unavailable_not_zero(records[0].require_sample(name))
    fragment = records[0].to_model_use_fragment()
    assert fragment["input_tokens"]["truth_state"] == "unavailable"
    assert "value" not in fragment["input_tokens"]
    with pytest.raises(BenchmarkTelemetryError, match="cannot be admitted"):
        records[0].to_resource_measurement()


def test_missing_causal_identity_is_quarantined() -> None:
    task = _task_span()
    missing = map_provider_response_to_span(
        {"usage": {"input_tokens": 1, "output_tokens": 1}},
        admitted_spans=(task,),
    )
    assert missing.quarantine_reason == (
        ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY.value
    )
    mismatched = map_provider_response_to_span(
        {
            "span_id": task.span_id,
            "task_id": "ASEH-999",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        admitted_spans=(task,),
    )
    assert mismatched.quarantine_reason == (
        ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY.value
    )


def test_credential_leakage_quarantines_and_does_not_persist_secrets() -> None:
    task = _task_span()
    record = map_provider_response_to_span(
        {
            "span_id": task.span_id,
            "task_id": "ASEH-011",
            "api_key": "sk-abcdefghijklmnopqrstuvwxyz123456",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        },
        admitted_spans=(task,),
    )
    assert record.disposition is ProviderUsageDisposition.QUARANTINED
    assert (
        record.quarantine_reason
        == ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE.value
    )
    encoded = record.to_json()
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in encoded
    assert "api_key" not in encoded
    assert record.request_ids == ()

    credential_id = map_provider_response_to_span(
        {
            "span_id": task.span_id,
            "request_id": "sk-abcdefghijklmnopqrstuvwxyz123456",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        },
        admitted_spans=(task,),
    )
    assert credential_id.quarantine_reason == (
        ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE.value
    )
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in credential_id.to_json()


def test_unsafe_non_credential_request_ids_are_omitted_not_measured_zero() -> None:
    task = _task_span()
    record = map_provider_response_to_span(
        {
            "span_id": task.span_id,
            "request_id": "too short",
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "model_class": "remote_frontier_model",
        },
        admitted_spans=(task,),
    )
    assert record.is_admitted
    assert record.request_ids == ()
    fragment = record.to_model_use_fragment()
    assert fragment["safe_provider_request_ids"]["truth_state"] == "unavailable"
    assert "values" not in fragment["safe_provider_request_ids"]
    assert extract_safe_request_id("too short") is None
    assert extract_safe_request_id("req_01SAFEOPAQUEID") == "req_01SAFEOPAQUEID"
    with pytest.raises(BenchmarkTelemetryError, match="credential"):
        extract_safe_request_id("sk-abcdefghijklmnopqrstuvwxyz123456")


def test_estimated_as_measured_is_quarantined() -> None:
    task = _task_span()
    labeled = map_provider_response_to_span(
        {
            "span_id": task.span_id,
            "usage": {
                "input_tokens": 4,
                "output_tokens": 1,
                "source": "estimated",
                "truth_state": "measured",
                "sensor_id": "sensor:lie",
            },
        },
        admitted_spans=(task,),
    )
    assert labeled.quarantine_reason == (
        ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED.value
    )
    estimate_as_measured = map_provider_response_to_span(
        {
            "span_id": task.span_id,
            "usage": {"input_tokens": 4, "output_tokens": 1},
            "estimate": {
                "value": 12,
                "unit": "microusd",
                "estimator_id": "price",
                "method": "token_price_snapshot",
                "truth_state": "measured",
                "sensor_id": "sensor:lie",
            },
        },
        admitted_spans=(task,),
    )
    assert estimate_as_measured.quarantine_reason == (
        ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED.value
    )


def test_labeled_estimate_cannot_carry_a_sensor() -> None:
    with pytest.raises(BenchmarkTelemetryError, match="labeled measured"):
        LabeledEstimate.from_dict(
            {
                "metric_name": "token_based_estimated_charge_microusd",
                "truth_state": "measured",
                "value": 1,
                "unit": "microusd",
                "estimator_id": "price",
                "method": "token_price_snapshot",
                "price_snapshot_identity": "unavailable",
                "sensor_id": "sensor:nope",
            }
        )


def test_aggregate_does_not_fill_omitted_fields_with_zero() -> None:
    task = _task_span()
    first = _call_span(task, span_id="span:provider-call-1")
    second = _call_span(task, span_id="span:provider-call-2")
    records = map_provider_responses(
        (
            {
                "span_id": first.span_id,
                "model_class": "remote_frontier_model",
                "request_id": "req_01JASEH011CALLA",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cached_input_tokens": 4,
                    "reasoning_tokens": 1,
                },
                "charge": {"reported_microusd": 100},
            },
            {
                "span_id": second.span_id,
                "model_class": "remote_frontier_model",
                "request_id": "req_01JASEH011CALLB",
                "usage": {"input_tokens": 6, "output_tokens": 3},
            },
        ),
        admitted_spans=(task, first, second),
    )
    # Per-call omitted cached tokens stay unavailable on that record.
    _assert_unavailable_not_zero(
        records[1].require_sample("provider_native_cached_input_tokens")
    )

    # Rebind both calls onto the task span identity for a task-level rollup
    # by mapping through copies that share the task span_id.
    task_records = map_provider_responses(
        (
            {
                "span_id": task.span_id,
                "model_class": "remote_frontier_model",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cached_input_tokens": 4,
                },
                "charge": {"reported_microusd": 100},
            },
            {
                "span_id": task.span_id,
                "model_class": "local_small_model",
                "usage": {"input_tokens": 6, "output_tokens": 3},
            },
        ),
        admitted_spans=(task,),
    )
    rolled = aggregate_provider_usage_for_span(task_records, task)
    assert rolled.require_sample("provider_native_input_tokens").value == 16
    assert rolled.require_sample("provider_native_output_tokens").value == 5
    assert rolled.require_sample("model_call_count").value == 2
    _assert_unavailable_not_zero(
        rolled.require_sample("provider_native_cached_input_tokens")
    )
    _assert_unavailable_not_zero(
        rolled.require_sample("provider_native_reasoning_tokens")
    )
    _assert_unavailable_not_zero(
        rolled.require_sample("provider_reported_charge_microusd")
    )
    assert rolled.call_class_sample("remote_frontier_model").value == 1
    assert rolled.call_class_sample("local_small_model").value == 1
    assert rolled.call_class_sample("human").value == 0


def test_empty_aggregate_stays_unavailable_not_zero() -> None:
    task = _task_span()
    rolled = aggregate_provider_usage_for_span((), task)
    assert rolled.is_admitted
    for name in PROVIDER_USAGE_SAMPLE_NAMES:
        _assert_unavailable_not_zero(rolled.require_sample(name))
    for class_name in MODEL_CLASSES:
        _assert_unavailable_not_zero(rolled.call_class_sample(class_name))


def test_admitted_record_round_trips_and_feeds_efficiency_receipt() -> None:
    task = _task_span()
    call = _call_span(task)
    record = map_provider_response_to_span(
        _complete_response(), admitted_spans=(task, call)
    )
    restored = ProviderUsageRecord.from_dict(record.to_record())
    assert restored.content_id == record.content_id
    assert restored.INTERFACE == PROVIDER_USAGE_RECORD_INTERFACE
    assert restored.request_ids == record.request_ids

    receipt = build_task_efficiency_receipt(
        task_id="ASEH-011",
        task_cid=TASK_CID,
        objective_id="ASEH-G020",
        objective_revision=OBJECTIVE_REVISION,
        repository_commit=COMMIT,
        repository_tree=TREE,
        policy_identity=POLICY,
        provider_id=record.provider_id,
        model_id=record.model_id,
        model_class=record.model_class,
        model_revision=observed_identity(record.model_revision),
        final_task_outcome="succeeded",
        patch_disposition="accepted",
        population_kind="hermetic_development",
        live=False,
        model_use=record.to_model_use_fragment(),
        cost=record.to_cost_fragment(),
    )
    assert tuple(receipt.model_use) == MODEL_USE_FIELDS
    assert receipt.model_use["input_tokens"]["truth_state"] == "measured"
    assert receipt.model_use["input_tokens"]["value"] == 120
    assert receipt.model_use["cached_input_tokens"]["value"] == 16
    assert receipt.model_use["reasoning_tokens"]["value"] == 8
    assert receipt.model_use["number_of_calls"]["value"] == 1
    assert receipt.model_use["safe_provider_request_ids"]["values"] == [
        "req_01JASEH011SAFEID"
    ]
    assert receipt.model_use["provider_reported_usage"]["truth_state"] == "observed"
    assert receipt.cost["provider_reported_charge"]["truth_state"] == "measured"
    assert receipt.cost["token_based_estimated_charge"]["truth_state"] == "estimated"
    assert receipt.cost["token_based_estimated_charge"]["method"] == "token_price_snapshot"
    assert receipt.identity["provider_model"]["model_revision"]["value"] == (
        "grok-4.6-2026-08-19"
    )
    assert receipt.evidence_state["estimated_as_measured"] is False
    assert receipt.evidence_state["unavailable_as_zero"] is False
    assert "model_use.input_tokens" not in receipt.explicit_unavailable_fields
    assert "cost.local_compute_units" in collect_unavailable_fields(receipt.to_dict())
    assert set(receipt.cost) >= set(COST_FIELDS)


def test_unknown_fields_floats_and_negative_tokens_fail_closed() -> None:
    task = _task_span()
    with pytest.raises(BenchmarkTelemetryError, match="unknown fields"):
        ProviderUsageRecord.from_dict(
            {
                **map_provider_response_to_span(
                    _complete_response(), admitted_spans=(task, _call_span(task))
                ).to_record(),
                "unexpected": 1,
            }
        )
    with pytest.raises(BenchmarkTelemetryError, match="integer"):
        map_provider_response_to_span(
            {
                "span_id": task.span_id,
                "usage": {"input_tokens": 1.5, "output_tokens": 1},
            },
            admitted_spans=(task,),
        )
    with pytest.raises(BenchmarkTelemetryError, match="integer microusd"):
        map_provider_response_to_span(
            {
                "span_id": task.span_id,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "cost_usd": 0.0034,
            },
            admitted_spans=(task,),
        )
    with pytest.raises(BenchmarkTelemetryError, match="between"):
        map_provider_response_to_span(
            {
                "span_id": task.span_id,
                "usage": {"input_tokens": -1, "output_tokens": 1},
            },
            admitted_spans=(task,),
        )


def test_required_sample_names_are_always_present() -> None:
    task = _task_span()
    record = map_provider_response_to_span(
        {"span_id": task.span_id, "model_class": "human"},
        admitted_spans=(task,),
    )
    assert tuple(item.metric_name for item in record.samples) == PROVIDER_USAGE_SAMPLE_NAMES
    assert {name for name, _sample in record.calls_by_model_class} == set(MODEL_CLASSES)
