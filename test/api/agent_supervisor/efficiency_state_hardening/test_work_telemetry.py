"""ASEH-012: work/compute telemetry on admitted causal spans."""

from __future__ import annotations

import os

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import (
    AttributionRole,
    BenchmarkCausalSpan,
    BenchmarkHardwareProfile,
    BenchmarkProviderBinding,
    BenchmarkTelemetryError,
    BenchmarkTelemetrySession,
    COMPUTE_FIELD_NAMES,
    OperationObservation,
    OperationTruthState,
    PATCH_DISPOSITIONS,
    PatchDisposition,
    SampleStatus,
    SpanKind,
    TASK_OUTCOMES,
    TaskOutcome,
    TelemetrySample,
    UNIT_BYTES,
    UNIT_COUNT,
    UNIT_MICROUSD,
    UNIT_SECONDS_MILLIONTHS,
    UnavailableReason,
    WORK_FIELD_NAMES,
    WORK_METRIC_NAMES,
    WORK_OPERATION_FIELD_NAMES,
    WorkTelemetryDisposition,
    WorkTelemetryQuarantineReason,
    WorkTelemetryRecord,
    map_work_and_compute_to_admitted_span,
    project_compute_from_sensors,
    project_work_telemetry_samples,
    sample_process_tree_resources,
)


VERIFIER_CID = "baguqeerahqmsf7qnxxghlgiwvbiijqiy2gylj5coif6bjdqoxnzklolqfqzq"
SECOND_CID = "baguqeeracuaruumlrlfhii3zanabo75m6sv23teuo4kkutkshlrldmz6723a"


def _provider() -> BenchmarkProviderBinding:
    return BenchmarkProviderBinding(
        provider_id="grok_cli",
        model_id="grok-4.6",
        model_revision="grok-4.6-2026-08-15",
        tokenizer_id="tokenizer:grok-native",
        endpoint_id="endpoint:grok/v1",
        max_context_tokens=128_000,
    )


def _hardware(*, accelerator: bool = False) -> BenchmarkHardwareProfile:
    return BenchmarkHardwareProfile(
        profile_id="hw:aseh-012",
        hostname_alias="host-aseh-012",
        cpu_model_id="cpu:test-x86",
        cpu_count=4,
        memory_bytes=8 * 1024**3,
        accelerator_present=accelerator,
        accelerator_model_id="gpu:test" if accelerator else "",
        accelerator_count=1 if accelerator else 0,
        platform="linux",
    )


def _task_span(**overrides: object) -> BenchmarkCausalSpan:
    fields = dict(
        span_id="span:task-aseh-012",
        kind=SpanKind.TASK,
        run_id="run:aseh-012",
        case_id="case:work-telemetry",
        arm_id="arm:candidate",
        task_id="ASEH-012",
        attempt=1,
        process_id=f"pid:{os.getpid()}",
        role=AttributionRole.WORKER,
        provider=_provider(),
        hardware=_hardware(),
        started_at_mono_ns=1_000_000_000,
        finished_at_mono_ns=4_000_000_000,
        monotonic_clock=True,
    )
    fields.update(overrides)
    return BenchmarkCausalSpan(**fields)  # type: ignore[arg-type]


def _session(
    span: BenchmarkCausalSpan | None = None,
) -> tuple[BenchmarkTelemetrySession, BenchmarkCausalSpan]:
    root = BenchmarkCausalSpan(
        span_id="span:run-aseh-012",
        kind=SpanKind.RUN,
        run_id="run:aseh-012",
        case_id="case:work-telemetry",
        arm_id="arm:candidate",
        task_id="ASEH-012",
        attempt=0,
        process_id="pid:root",
        role=AttributionRole.ROOT,
        provider=_provider(),
        hardware=_hardware(),
        started_at_mono_ns=1,
        finished_at_mono_ns=9_000_000_000,
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


def _measured(name: str, value: int, *, unit: str, sensor: str) -> dict[str, object]:
    return {
        "truth_state": "measured",
        "value": value,
        "unit": unit,
        "sensor_id": sensor,
    }


def _overhead(value: int = 250) -> dict[str, object]:
    return _measured(
        "audit_and_verification_overhead",
        value,
        unit=UNIT_MICROUSD,
        sensor="sensor:audit-runner",
    )


def _observed(operation_id: str, *, count: int, observer: str = "validation-runner") -> dict[str, object]:
    return {
        "truth_state": "observed",
        "operation_id": operation_id,
        "observer_id": observer,
        "count": count,
    }


def _verified(
    operation_id: str,
    *,
    count: int,
    verifier_id: str = "pytest-junit",
    cid: str = VERIFIER_CID,
    observer: str = "validation-runner",
) -> dict[str, object]:
    return {
        "truth_state": "verified",
        "operation_id": operation_id,
        "observer_id": observer,
        "count": count,
        "verifier_id": verifier_id,
        "verifier_receipt_cid": cid,
    }


def _attempted(operation_id: str, *, attempt_count: int = 1) -> dict[str, object]:
    return {
        "truth_state": "attempted",
        "operation_id": operation_id,
        "attempt_count": attempt_count,
    }


def _complete_work(**overrides: object) -> dict[str, object]:
    work: dict[str, object] = {
        "tests_selected": _observed("pytest-select", count=4),
        "tests_executed": _verified("pytest", count=4),
        "full_suite_tests": _observed("pytest-full", count=12),
        "type_static_schema_checks": _observed("ruff-mypy", count=2),
        "proof_obligations_selected": _observed("lean-select", count=3),
        "proof_obligations_executed": _attempted("lean-kernel", attempt_count=1),
        "proof_receipts_reused": _observed("proof-cache", count=1),
        "retries": _observed("provider-retry", count=1),
        "rescue_attempts": _observed("rescue", count=0),
        "merge_conflicts": _observed("merge", count=0),
        "manual_recovery": _observed("recovery", count=0),
        "human_interventions": _observed("human", count=0, observer="operator"),
        "validation_result": _verified(
            "validation",
            count=1,
            verifier_id="pytest",
            cid=VERIFIER_CID,
        ),
        "final_task_outcome": TaskOutcome.SUCCEEDED.value,
        "patch_disposition": PatchDisposition.ACCEPTED.value,
    }
    work.update(overrides)
    return work


def _admit_verifiers(session: BenchmarkTelemetrySession) -> None:
    session.admit_verifier("pytest-junit", VERIFIER_CID)
    session.admit_verifier("pytest", VERIFIER_CID)


def test_causal_spans_cover_every_work_and_compute_field() -> None:
    session, span = _session()
    _admit_verifiers(session)
    check = span.child(
        span_id="span:check-pytest",
        kind=SpanKind.CHECK,
        role=AttributionRole.ORACLE,
        process_id="pid:pytest",
        started_at_mono_ns=span.started_at_mono_ns + 10,
        finished_at_mono_ns=span.started_at_mono_ns + 1_500_000_000,
    )
    proof = span.child(
        span_id="span:proof-lean",
        kind=SpanKind.PROOF,
        role=AttributionRole.ORACLE,
        process_id="pid:lean",
        started_at_mono_ns=span.started_at_mono_ns + 20,
        finished_at_mono_ns=span.started_at_mono_ns + 400_000_000,
    )
    retry = span.child(
        span_id="span:retry-1",
        kind=SpanKind.RETRY,
        role=AttributionRole.RETRY,
        process_id="pid:retry",
        started_at_mono_ns=span.started_at_mono_ns + 30,
        finished_at_mono_ns=span.started_at_mono_ns + 80_000_000,
    )
    human = span.child(
        span_id="span:human-1",
        kind=SpanKind.HUMAN,
        role=AttributionRole.HUMAN,
        process_id="pid:human",
        started_at_mono_ns=span.started_at_mono_ns + 40,
        finished_at_mono_ns=span.started_at_mono_ns + 50_000_000,
    )
    validation = span.child(
        span_id="span:validation-1",
        kind=SpanKind.VALIDATION,
        role=AttributionRole.VERIFIER,
        process_id="pid:validate",
        started_at_mono_ns=span.started_at_mono_ns + 50,
        finished_at_mono_ns=span.finished_at_mono_ns - 10,
    )
    for child in (check, proof, retry, human, validation):
        session.register_span(child)

    process_samples = sample_process_tree_resources(
        os.getpid(),
        wall_seconds_millionths=span.duration_seconds_millionths,
    )
    record = session.record_work_and_compute(
        {
            "work": _complete_work(),
            "audit_and_verification_overhead": _overhead(),
        },
        span=span,
        record_id="work:complete",
        process_samples=process_samples,
    )
    assert record.admitted
    assert record.disposition is WorkTelemetryDisposition.ADMITTED
    assert record.span_id == span.span_id
    assert record.task_id == "ASEH-012"
    assert set(record.compute_map()) == set(COMPUTE_FIELD_NAMES)
    assert set(record.operation_map()) == set(WORK_OPERATION_FIELD_NAMES)
    coverage = record.covering_span_map()
    required = set(WORK_METRIC_NAMES) | {"final_task_outcome", "patch_disposition"}
    assert required <= set(coverage)
    for field_name in required:
        assert coverage[field_name]
    assert coverage["tests_executed"] == check.span_id
    assert coverage["proof_obligations_executed"] == proof.span_id
    assert coverage["retries"] == retry.span_id
    assert coverage["human_interventions"] == human.span_id
    assert coverage["validation_result"] == validation.span_id
    availability = record.explicit_availability()
    for field_name in COMPUTE_FIELD_NAMES:
        assert availability[field_name] in {
            SampleStatus.MEASURED.value,
            SampleStatus.ESTIMATED.value,
            SampleStatus.UNAVAILABLE.value,
        }
    for field_name in WORK_OPERATION_FIELD_NAMES:
        assert availability[field_name] in {
            OperationTruthState.ATTEMPTED.value,
            OperationTruthState.OBSERVED.value,
            OperationTruthState.VERIFIED.value,
            OperationTruthState.UNAVAILABLE.value,
            OperationTruthState.SIMULATED.value,
        }
    assert availability["audit_and_verification_overhead"] == SampleStatus.MEASURED.value
    assert record.audit_and_verification_overhead.value == 250
    verified = [
        item
        for item in record.operations
        if item.truth_state is OperationTruthState.VERIFIED
    ]
    assert verified
    assert {item.verifier_id for item in verified} <= {
        pair[0] for pair in record.admitted_verifier_ids
    }
    for item in verified:
        assert (item.verifier_id, item.verifier_receipt_cid) in record.admitted_verifier_ids
    compute = record.to_compute_fields()
    assert set(compute) == set(COMPUTE_FIELD_NAMES)
    work = record.to_work_fields()
    assert set(work) == set(WORK_FIELD_NAMES)
    assert work["final_task_outcome"] == "succeeded"
    assert work["patch_disposition"] == "accepted"
    assert work["tests_executed"]["truth_state"] == "verified"
    assert work["proof_obligations_executed"]["truth_state"] == "attempted"
    terminal = record.to_terminal_fields()
    assert terminal["terminalized"] is True
    assert terminal["single_terminalization"] is True
    samples = project_work_telemetry_samples(record)
    assert set(COMPUTE_FIELD_NAMES) <= set(samples)
    assert "audit_and_verification_overhead" in samples
    measurement = record.to_resource_measurement()
    assert measurement.span.span_id == span.span_id
    assert check.span_id in measurement.source_span_ids
    receipt = session.seal_receipt()
    bound = next(
        item
        for item in receipt.measurements
        if item.measurement_id == measurement.measurement_id
    )
    assert bound.require_sample("wall_clock_duration").status is SampleStatus.MEASURED
    assert bound.require_sample("gpu_seconds").status is SampleStatus.UNAVAILABLE
    assert "value" not in bound.require_sample("gpu_seconds").to_envelope()
    certificates = receipt.certify_all()
    assert certificates and all(item.certified for item in certificates)


def test_absent_compute_and_work_fields_remain_unavailable_not_zero() -> None:
    session, span = _session()
    record = session.record_work_and_compute(
        {
            "work": {
                "final_task_outcome": "failed",
                "patch_disposition": "rejected",
            },
            "audit_and_verification_overhead": {
                "truth_state": "unavailable",
                "reason_code": "not_reported",
            },
        },
        span=span,
        record_id="work:absent",
    )
    assert record.admitted
    for sample in record.compute:
        if sample.status is SampleStatus.UNAVAILABLE:
            envelope = sample.to_envelope()
            assert "value" not in envelope
            quantity = sample.to_quantity_envelope()
            assert quantity["truth_state"] == "unavailable"
            assert "value" not in quantity
            assert "count" not in quantity
    for operation in record.operations:
        assert operation.truth_state is OperationTruthState.UNAVAILABLE
        envelope = operation.to_operation_envelope()
        assert envelope["truth_state"] == "unavailable"
        assert "count" not in envelope
        assert "value" not in envelope
        assert envelope.get("count", "missing") != 0
    assert "compute.gpu_seconds" in record.unavailable_fields()
    assert "work.tests_executed" in record.unavailable_fields()
    assert record.compute_map()["wall_clock_duration"].status is SampleStatus.MEASURED
    assert record.compute_map()["gpu_seconds"].reason_code == (
        UnavailableReason.HARDWARE_ABSENT.value
    )


def test_explicit_zero_work_counts_are_observed_not_unavailable() -> None:
    session, span = _session()
    record = session.record_work_and_compute(
        {
            "work": _complete_work(
                retries=_observed("provider-retry", count=0),
                rescue_attempts=_observed("rescue", count=0),
                tests_executed=_observed("pytest", count=0),
                validation_result=_observed("validation", count=0),
            ),
            "audit_and_verification_overhead": _overhead(0),
        },
        span=span,
        record_id="work:zeros",
    )
    retries = record.operation_map()["retries"]
    assert retries.truth_state is OperationTruthState.OBSERVED
    assert retries.count == 0
    assert retries.observer_id
    assert record.audit_and_verification_overhead.status is SampleStatus.MEASURED
    assert record.audit_and_verification_overhead.value == 0
    assert record.audit_and_verification_overhead.sensor_id


def test_process_tree_and_child_spans_project_compute_fields() -> None:
    session, span = _session()
    check = span.child(
        span_id="span:check-time",
        kind=SpanKind.CHECK,
        process_id="pid:check",
        started_at_mono_ns=span.started_at_mono_ns + 1,
        finished_at_mono_ns=span.started_at_mono_ns + 2_000_000,
    )
    proof = span.child(
        span_id="span:proof-time",
        kind=SpanKind.PROOF,
        process_id="pid:proof",
        started_at_mono_ns=span.started_at_mono_ns + 2,
        finished_at_mono_ns=span.started_at_mono_ns + 5_000_000,
    )
    session.register_span(check)
    session.register_span(proof)
    process_samples = sample_process_tree_resources(
        os.getpid(),
        wall_seconds_millionths=span.duration_seconds_millionths,
    )
    projected = project_compute_from_sensors(
        span,
        process_samples=process_samples,
        session=session,
    )
    assert set(COMPUTE_FIELD_NAMES) <= set(projected)
    cpu = projected["cpu_seconds"]
    if cpu.status is SampleStatus.MEASURED:
        assert cpu.unit == UNIT_SECONDS_MILLIONTHS
        assert cpu.sensor_id
    else:
        assert cpu.status is SampleStatus.UNAVAILABLE
        assert "value" not in cpu.to_envelope()
    assert projected["wall_clock_duration"].status is SampleStatus.MEASURED
    assert projected["wall_clock_duration"].value == span.duration_seconds_millionths
    assert projected["test_execution_time"].status is SampleStatus.MEASURED
    assert projected["test_execution_time"].value == check.duration_seconds_millionths
    assert projected["prover_execution_time"].value == proof.duration_seconds_millionths
    if projected["peak_memory"].status is SampleStatus.MEASURED:
        assert projected["peak_memory"].unit == UNIT_BYTES
    if projected["process_count"].status is SampleStatus.MEASURED:
        assert projected["process_count"].unit == UNIT_COUNT
        assert projected["process_count"].value >= 1


def test_unbound_work_is_quarantined() -> None:
    span = _task_span()
    record = map_work_and_compute_to_admitted_span(
        {
            "work": _complete_work(),
            "audit_and_verification_overhead": _overhead(),
        },
        span,
        record_id="work:unbound",
    )
    assert record.disposition is WorkTelemetryDisposition.QUARANTINED
    assert record.quarantine_reason == WorkTelemetryQuarantineReason.UNBOUND_WORK.value
    assert record.compute_map()["cpu_seconds"].status is SampleStatus.UNAVAILABLE
    assert record.compute_map()["cpu_seconds"].reason_code == (
        UnavailableReason.NOT_ADMITTED.value
    )
    with pytest.raises(BenchmarkTelemetryError, match="cannot seal"):
        record.to_resource_measurement()


def test_missing_causal_task_identity_is_quarantined() -> None:
    session, _task = _session()
    anonymous = _task_span(
        span_id="span:anonymous-work",
        task_id="",
        process_id="pid:anonymous-work",
        parent_span_id="span:run-aseh-012",
        ancestry=("span:run-aseh-012",),
    )
    session.register_span(anonymous)
    record = map_work_and_compute_to_admitted_span(
        {
            "work": {"final_task_outcome": "failed", "patch_disposition": "rejected"},
            "audit_and_verification_overhead": _overhead(),
        },
        anonymous,
        session=session,
        record_id="work:anonymous",
    )
    assert record.disposition is WorkTelemetryDisposition.QUARANTINED
    assert record.quarantine_reason == (
        WorkTelemetryQuarantineReason.MISSING_CAUSAL_IDENTITY.value
    )
    none_record = map_work_and_compute_to_admitted_span(
        None,
        None,
        session=session,
        record_id="work:none",
    )
    assert none_record.quarantine_reason == (
        WorkTelemetryQuarantineReason.MISSING_CAUSAL_IDENTITY.value
    )


def test_attempted_as_observed_fails_closed() -> None:
    session, span = _session()
    with pytest.raises(BenchmarkTelemetryError, match="attempted"):
        session.record_work_and_compute(
            {
                "work": _complete_work(
                    proof_obligations_executed={
                        "truth_state": "observed",
                        "operation_id": "lean-kernel",
                        "attempt_count": 2,
                    }
                ),
                "audit_and_verification_overhead": _overhead(),
            },
            span=span,
            record_id="work:attempted-as-observed",
        )
    with pytest.raises(BenchmarkTelemetryError, match="attempted"):
        OperationObservation(
            field_name="retries",
            truth_state=OperationTruthState.OBSERVED,
            operation_id="retry",
            observer_id="runner",
            count=1,
            attempt_count=1,
        )


def test_observed_as_verified_without_admitted_verifier_fails_closed() -> None:
    session, span = _session()
    with pytest.raises(BenchmarkTelemetryError, match="verified"):
        session.record_work_and_compute(
            {
                "work": _complete_work(
                    tests_executed={
                        "truth_state": "verified",
                        "operation_id": "pytest",
                        "observer_id": "validation-runner",
                        "count": 4,
                    }
                ),
                "audit_and_verification_overhead": _overhead(),
            },
            span=span,
            record_id="work:observed-as-verified",
        )
    session.admit_verifier("pytest-junit", VERIFIER_CID)
    with pytest.raises(BenchmarkTelemetryError, match="admitted verifier"):
        session.record_work_and_compute(
            {
                "work": _complete_work(),
                "audit_and_verification_overhead": _overhead(),
            },
            span=span,
            record_id="work:unadmitted-verifier",
        )
    with pytest.raises(BenchmarkTelemetryError, match="verified"):
        OperationObservation.observed(
            "tests_executed",
            "pytest",
            observer_id="runner",
            count=1,
        ).__class__(
            field_name="tests_executed",
            truth_state=OperationTruthState.OBSERVED,
            operation_id="pytest",
            observer_id="runner",
            count=1,
            verifier_id="pytest",
            verifier_receipt_cid=VERIFIER_CID,
        )


def test_missing_overhead_fails_closed() -> None:
    session, span = _session()
    session.admit_verifier("pytest-junit", VERIFIER_CID)
    session.admit_verifier("pytest", VERIFIER_CID)
    with pytest.raises(BenchmarkTelemetryError, match="overhead"):
        session.record_work_and_compute(
            {"work": _complete_work()},
            span=span,
            record_id="work:missing-overhead",
        )
    with pytest.raises(BenchmarkTelemetryError, match="overhead"):
        session.record_work_and_compute(
            {
                "work": _complete_work(),
                "audit_and_verification_overhead": {
                    "truth_state": "unavailable",
                    "reason_code": "not_reported",
                },
            },
            span=span,
            record_id="work:unavailable-overhead",
        )


def test_duplicate_terminal_accounting_fails_closed() -> None:
    session, span = _session()
    session.admit_verifier("pytest-junit", VERIFIER_CID)
    session.admit_verifier("pytest", VERIFIER_CID)
    first = session.record_work_and_compute(
        {
            "work": _complete_work(),
            "audit_and_verification_overhead": _overhead(),
        },
        span=span,
        record_id="work:terminal-1",
    )
    assert first.terminalized is True
    assert first.single_terminalization is True
    with pytest.raises(BenchmarkTelemetryError, match="duplicate terminal"):
        session.record_work_and_compute(
            {
                "work": _complete_work(),
                "audit_and_verification_overhead": _overhead(),
            },
            span=span,
            record_id="work:terminal-2",
        )
    with pytest.raises(BenchmarkTelemetryError, match="duplicate terminal"):
        session.record_work_and_compute(
            {
                "work": {
                    "final_task_outcome": "failed",
                    "patch_disposition": "rejected",
                    "single_terminalization": False,
                },
                "audit_and_verification_overhead": {
                    "truth_state": "unavailable",
                    "reason_code": "not_reported",
                },
            },
            span=span,
            record_id="work:not-single",
        )


def test_invalid_bounds_fail_closed() -> None:
    session, span = _session()
    session.admit_verifier("pytest-junit", VERIFIER_CID)
    session.admit_verifier("pytest", VERIFIER_CID)
    with pytest.raises(BenchmarkTelemetryError, match="bound|exceeds"):
        session.record_work_and_compute(
            {
                "work": _complete_work(
                    tests_selected=_observed("pytest-select", count=2),
                    tests_executed=_verified("pytest", count=5),
                ),
                "audit_and_verification_overhead": _overhead(),
            },
            span=span,
            record_id="work:executed-gt-selected",
        )
    with pytest.raises(BenchmarkTelemetryError):
        session.record_work_and_compute(
            {
                "work": _complete_work(
                    retries=_observed("provider-retry", count=-1),
                ),
                "audit_and_verification_overhead": _overhead(),
            },
            span=span,
            record_id="work:negative",
        )
    with pytest.raises(BenchmarkTelemetryError):
        TelemetrySample.measured(
            "cpu_seconds",
            True,  # type: ignore[arg-type]
            unit=UNIT_SECONDS_MILLIONTHS,
            sensor_id="sensor:bool",
        )
    with pytest.raises(BenchmarkTelemetryError):
        session.record_work_and_compute(
            {
                "work": _complete_work(),
                "compute": {
                    "concurrency": _measured(
                        "concurrency",
                        10**18 + 1,
                        unit=UNIT_COUNT,
                        sensor="sensor:too-big",
                    )
                },
                "audit_and_verification_overhead": _overhead(),
            },
            span=span,
            record_id="work:too-big",
        )


def test_unknown_fields_and_round_trip() -> None:
    session, span = _session()
    _admit_verifiers(session)
    record = session.record_work_and_compute(
        {
            "work": _complete_work(),
            "audit_and_verification_overhead": _overhead(),
            "compute": {
                "concurrency": _measured(
                    "concurrency", 1, unit=UNIT_COUNT, sensor="sensor:lane"
                )
            },
        },
        span=span,
        record_id="work:round-trip",
    )
    restored = WorkTelemetryRecord.from_dict(record.to_record())
    assert restored.content_id == record.content_id
    assert restored.to_work_fields()["tests_executed"]["verifier_receipt_cid"] == (
        VERIFIER_CID
    )
    payload = record.to_record()
    payload["unexpected"] = 1
    with pytest.raises(BenchmarkTelemetryError, match="unknown fields"):
        WorkTelemetryRecord.from_dict(payload)
    other_session, other_span = _session()
    _admit_verifiers(other_session)
    with pytest.raises(BenchmarkTelemetryError, match="unknown fields"):
        other_session.record_work_and_compute(
            {
                "work": {**_complete_work(), "surprise": 1},
                "audit_and_verification_overhead": _overhead(),
            },
            span=other_span,
            record_id="work:unknown-work",
        )


def test_closed_outcome_and_disposition_values() -> None:
    assert "succeeded" in TASK_OUTCOMES
    assert "accepted" in PATCH_DISPOSITIONS
    session, span = _session()
    with pytest.raises(BenchmarkTelemetryError, match="final_task_outcome"):
        session.record_work_and_compute(
            {
                "work": {
                    "final_task_outcome": "done",
                    "patch_disposition": "accepted",
                },
                "audit_and_verification_overhead": {
                    "truth_state": "unavailable",
                    "reason_code": "not_reported",
                },
            },
            span=span,
            record_id="work:bad-outcome",
        )


def test_verified_operation_round_trip_preserves_verifier_linkage() -> None:
    operation = OperationObservation.verified(
        "validation_result",
        "validation",
        observer_id="validation-runner",
        count=1,
        verifier_id="pytest",
        verifier_receipt_cid=SECOND_CID,
        covering_span_id="span:validation-1",
    )
    restored = OperationObservation.from_dict(operation.to_record())
    assert restored.content_id == operation.content_id
    assert restored.verifier_receipt_cid == SECOND_CID
    envelope = restored.to_operation_envelope()
    assert envelope["truth_state"] == "verified"
    assert envelope["verifier_id"] == "pytest"
    assert "reason_code" not in envelope


def test_unavailable_operation_cannot_encode_numeric_zero() -> None:
    with pytest.raises(BenchmarkTelemetryError, match="numeric"):
        OperationObservation.from_dict(
            {
                "field_name": "tests_executed",
                "truth_state": "unavailable",
                "reason_code": "not_reported",
                "count": 0,
            }
        )
    with pytest.raises(BenchmarkTelemetryError, match="numeric"):
        TelemetrySample(
            metric_name="cpu_seconds",
            status=SampleStatus.UNAVAILABLE,
            sensor_id="sensor:x",
            reason_code=UnavailableReason.NOT_REPORTED.value,
            value=0,
            unit=UNIT_SECONDS_MILLIONTHS,
        )
