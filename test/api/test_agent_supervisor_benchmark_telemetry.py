"""PDR-071: causal-span benchmark telemetry regressions."""

from __future__ import annotations

import os

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import (
    BENCHMARK_CAUSAL_SPAN_INTERFACE,
    BENCHMARK_RESOURCE_MEASUREMENT_INTERFACE,
    AttributionRole,
    BenchmarkCausalSpan,
    BenchmarkHardwareProfile,
    BenchmarkProviderBinding,
    BenchmarkResourceMeasurement,
    BenchmarkTelemetryError,
    BenchmarkTelemetrySession,
    SampleStatus,
    SpanKind,
    TelemetrySample,
    UnavailableReason,
    build_resource_measurement,
    build_span_joined_measurement,
    certify_measurement_from_source_spans,
    mono_ns,
    project_scheduler_clock_samples,
    project_token_ledger_samples,
    reject_self_certified_counters,
    sample_energy_optional,
    sample_gpu_resources,
    sample_network_bytes,
    sample_process_tree_resources,
    sample_rusage_self,
    sample_self_process_tree_resources,
    seconds_to_millionths,
)
from ipfs_accelerate_py.agent_supervisor.runtime.scheduler_metrics import (
    BENCHMARK_SPAN_CLOCK_SCHEMA,
    build_scheduler_snapshot,
    join_scheduler_snapshot_to_benchmark_span,
    project_snapshot_metrics_for_span,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_token_ledger import (
    CacheDecision,
    ProviderModelEnvelope,
    ProviderTokenUsage,
    SupervisorTokenLedger,
    TerminalCriterionAttribution,
    TerminalDisposition,
    TokenAttribution,
    UsageSource,
    ValidationResult,
    bind_attribution_to_span,
    provider_native_token_totals_for_span,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_contracts import (
    AuthorityClass,
    ResultBinding,
    SemanticDependencyIdentity,
    StageEvent,
    StageEventKind,
)


def _hardware(*, accelerator: bool = False) -> BenchmarkHardwareProfile:
    return BenchmarkHardwareProfile(
        profile_id="hw:test-host",
        hostname_alias="host-alias-1",
        cpu_model_id="cpu:test-x86",
        cpu_count=4,
        memory_bytes=8 * 1024**3,
        accelerator_present=accelerator,
        accelerator_model_id="gpu:test" if accelerator else "",
        accelerator_count=1 if accelerator else 0,
        platform="linux",
    )


def _provider() -> BenchmarkProviderBinding:
    return BenchmarkProviderBinding(
        provider_id="provider:example",
        model_id="model:reasoner",
        model_revision="model:reasoner@2026-07",
        tokenizer_id="tokenizer:provider-native",
        endpoint_id="endpoint:example/v1",
        max_context_tokens=16_384,
    )


def _root_span(**overrides: object) -> BenchmarkCausalSpan:
    started = mono_ns()
    finished = started + 2_000_000_000
    fields = dict(
        span_id="span:run-1",
        kind=SpanKind.RUN,
        run_id="run:benchmark-1",
        case_id="case:holdout-1",
        arm_id="arm:mainline",
        task_id="PDR-071",
        attempt=1,
        process_id=f"pid:{os.getpid()}",
        role=AttributionRole.ROOT,
        provider=_provider(),
        hardware=_hardware(),
        started_at_mono_ns=started,
        finished_at_mono_ns=finished,
        monotonic_clock=True,
    )
    fields.update(overrides)
    return BenchmarkCausalSpan(**fields)  # type: ignore[arg-type]


def _binding() -> ResultBinding:
    return ResultBinding(
        repository_id="repository:supervisor",
        tree_id="tree:telemetry",
        objective_id="PDR-G080",
        objective_revision="objective:g080@1",
        task_id="PDR-071",
        task_revision="PDR-071@1",
        policy_id="policy:token-ledger",
        policy_revision="policy:token-ledger@1",
        producer_id="producer:telemetry",
        producer_revision="producer:telemetry@1",
        capability_id="capability:provider-accounting",
        capability_revision="capability:provider-accounting@1",
        environment_id="environment:test",
        environment_revision="environment:test@1",
        semantic_dependencies=(
            SemanticDependencyIdentity(
                namespace="repository",
                key="source-tree",
                revision="tree:telemetry",
                digest="sha256:" + "b" * 64,
            ),
        ),
    )


def _event(
    binding: ResultBinding,
    *,
    stage: str,
    attempt: int,
    kind: StageEventKind,
) -> StageEvent:
    return StageEvent(
        binding=binding,
        stage=stage,
        attempt=attempt,
        sequence=0,
        kind=kind,
        authority=AuthorityClass.VALIDATION,
        occurred_at=f"2026-08-03T12:00:{attempt:02d}.000000Z",
        reason_code="" if kind is StageEventKind.COMPLETED else "ended",
    )


def _ledger_with_cancel() -> SupervisorTokenLedger:
    binding = _binding()
    envelope = ProviderModelEnvelope(
        provider_id="provider:example",
        model_id="model:reasoner",
        model_revision="model:reasoner@2026-07",
        tokenizer_id="tokenizer:provider-native",
        envelope_revision="envelope:2026-07",
        max_context_tokens=16_384,
    )
    ok = _event(binding, stage="inference", attempt=1, kind=StageEventKind.COMPLETED)
    cancelled = _event(
        binding, stage="inference", attempt=2, kind=StageEventKind.CANCELLED
    )
    ok_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=ok.event_id,
        criterion_id="criterion:main",
        disposition=TerminalDisposition.ACCEPTED,
        validation_result=ValidationResult.PASSED,
        evidence_gain=1,
    )
    cancel_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=cancelled.event_id,
        criterion_id="criterion:retry",
        disposition=TerminalDisposition.ABANDONED,
        validation_result=ValidationResult.NOT_RUN,
        reason_code="operator-cancel",
    )
    ok_usage = ProviderTokenUsage(
        measurement_id="request:ok",
        envelope=envelope,
        source=UsageSource.PROVIDER_NATIVE,
        input_tokens=40,
        output_tokens=10,
        cost_microunits=100,
    )
    cancel_usage = ProviderTokenUsage(
        measurement_id="request:cancel",
        envelope=envelope,
        source=UsageSource.PROVIDER_NATIVE,
        input_tokens=20,
        output_tokens=0,
        retry_tokens=20,
        failed_attempt_tokens=20,
        cancelled_tokens=20,
        cost_microunits=50,
    )
    return SupervisorTokenLedger(
        binding=binding,
        lifecycle_events=(ok, cancelled),
        terminal_attributions=(ok_terminal, cancel_terminal),
        attributions=(
            TokenAttribution(
                binding=binding,
                event_id=ok.event_id,
                stage=ok.stage,
                attempt=1,
                context_id="context:ok",
                cache_decision=CacheDecision.MISS,
                validation_result=ValidationResult.PASSED,
                terminal_attribution_id=ok_terminal.terminal_attribution_id,
                usage=ok_usage,
                span_id="span:run-1",
            ),
            TokenAttribution(
                binding=binding,
                event_id=cancelled.event_id,
                stage=cancelled.stage,
                attempt=2,
                context_id="context:cancel",
                cache_decision=CacheDecision.MISS,
                validation_result=ValidationResult.NOT_RUN,
                terminal_attribution_id=cancel_terminal.terminal_attribution_id,
                usage=cancel_usage,
                span_id="span:run-1",
            ),
        ),
    )


def test_interfaces_and_measured_unavailable_envelope() -> None:
    assert BenchmarkCausalSpan.INTERFACE == BENCHMARK_CAUSAL_SPAN_INTERFACE
    assert (
        BenchmarkResourceMeasurement.INTERFACE
        == BENCHMARK_RESOURCE_MEASUREMENT_INTERFACE
    )

    measured = TelemetrySample.measured(
        "peak_rss_bytes",
        0,
        unit="bytes",
        sensor_id="sensor:rss",
    )
    assert measured.status is SampleStatus.MEASURED
    assert measured.value == 0
    assert measured.to_envelope()["status"] == "measured"
    assert measured.to_envelope()["sensor_id"] == "sensor:rss"

    unavailable = TelemetrySample.unavailable(
        "peak_vram_bytes",
        UnavailableReason.HARDWARE_ABSENT,
    )
    assert unavailable.status is SampleStatus.UNAVAILABLE
    assert "value" not in unavailable.to_envelope()
    assert unavailable.to_envelope()["reason_code"] == "hardware-absent"

    with pytest.raises(BenchmarkTelemetryError):
        TelemetrySample(
            metric_name="broken",
            status=SampleStatus.UNAVAILABLE,
            sensor_id="sensor:x",
            reason_code="hardware-absent",
            value=0,
            unit="bytes",
        )


def test_unavailable_never_encodes_numeric_zero_and_round_trips() -> None:
    sample = TelemetrySample.unavailable(
        "network_rx_bytes",
        UnavailableReason.PERMISSION_DENIED,
        sensor_id="sensor:net",
    )
    restored = TelemetrySample.from_dict(sample.to_record())
    assert restored.content_id == sample.content_id
    assert restored.status is SampleStatus.UNAVAILABLE
    assert restored.value == 0
    assert restored.unit == ""


def test_causal_span_ancestry_and_child_binding() -> None:
    root = _root_span()
    child = root.child(
        span_id="span:attempt-1",
        kind=SpanKind.ATTEMPT,
        role=AttributionRole.RETRY,
        attempt=2,
        process_id="pid:child",
        started_at_mono_ns=root.started_at_mono_ns + 10,
        finished_at_mono_ns=root.finished_at_mono_ns - 10,
    )
    assert child.parent_span_id == root.span_id
    assert child.ancestry == (root.span_id,)
    assert child.provider is not None
    assert child.provider.tokenizer_id == "tokenizer:provider-native"
    assert child.hardware is not None
    assert child.hardware.profile_id == "hw:test-host"
    assert child.duration_ns > 0

    restored = BenchmarkCausalSpan.from_dict(root.to_record())
    assert restored.content_id == root.content_id
    assert restored.INTERFACE == "BenchmarkCausalSpan@1"

    with pytest.raises(BenchmarkTelemetryError):
        BenchmarkCausalSpan(
            span_id="span:bad",
            kind=SpanKind.TASK,
            run_id="run:1",
            case_id="case:1",
            arm_id="arm:1",
            task_id="t",
            attempt=1,
            process_id="p",
            parent_span_id="span:missing-parent",
            ancestry=(),
        )


def test_exactly_once_process_and_measurement_attribution() -> None:
    root = _root_span()
    session = BenchmarkTelemetrySession(root)
    daemon = root.child(
        span_id="span:daemon",
        kind=SpanKind.DAEMON,
        role=AttributionRole.DAEMON_CHILD,
        process_id="pid:daemon-1",
    )
    session.register_span(daemon)
    session.attribute_process("pid:daemon-1", daemon.span_id)

    with pytest.raises(BenchmarkTelemetryError, match="already attributed"):
        session.attribute_process("pid:daemon-1", root.span_id)

    cancel = root.child(
        span_id="span:cancel",
        kind=SpanKind.CANCEL,
        role=AttributionRole.CANCELLED,
        process_id="pid:cancel-1",
    )
    session.register_span(cancel)
    measurement = build_resource_measurement(
        measurement_id="meas:cancel",
        span=cancel,
        samples={
            "user_cpu_seconds": TelemetrySample.measured(
                "user_cpu_seconds",
                1_000_000,
                unit="seconds_millionths",
                sensor_id="sensor:cpu",
            )
        },
        attributed_process_ids=("pid:cancel-1",),
    )
    session.record_measurement(measurement)
    with pytest.raises(BenchmarkTelemetryError, match="already attributed"):
        session.attribute_measurement("meas:cancel", root.span_id)

    receipt = session.seal_receipt()
    assert receipt.root_span_id == root.span_id
    assert len(receipt.spans) == 3
    certificates = receipt.certify_all()
    assert certificates and certificates[0].certified is True


def test_serialized_counters_cannot_self_certify_without_span_replay() -> None:
    root = _root_span()
    measurement = build_resource_measurement(
        measurement_id="meas:counters",
        span=root,
        samples={
            "peak_rss_bytes": TelemetrySample.measured(
                "peak_rss_bytes",
                4096,
                unit="bytes",
                sensor_id="sensor:rss",
            )
        },
    )
    with pytest.raises(BenchmarkTelemetryError, match="cannot self-certify"):
        reject_self_certified_counters(
            serialized_counters={"peak_rss_bytes": 4096},
            source_spans=None,
        )

    uncertified = certify_measurement_from_source_spans(measurement, ())
    assert uncertified.certified is False

    certified = certify_measurement_from_source_spans(measurement, (root,))
    assert certified.certified is True
    assert certified.measurement_content_id == measurement.content_id


def test_process_tree_and_rusage_sensors_produce_measured_or_unavailable() -> None:
    samples = sample_self_process_tree_resources(
        wall_seconds_millionths=2_000_000,
        artifact_bytes_before=100,
        artifact_bytes_after=250,
    )
    assert samples["disk_artifact_growth_bytes"].status is SampleStatus.MEASURED
    assert samples["disk_artifact_growth_bytes"].value == 150
    # CPU/RSS may be measured on Linux or unavailable under restricted /proc.
    for name in (
        "user_cpu_seconds",
        "system_cpu_seconds",
        "total_cpu_seconds",
        "peak_rss_bytes",
        "memory_gib_seconds",
        "read_bytes",
        "write_bytes",
        "peak_process_count",
    ):
        assert samples[name].status in (
            SampleStatus.MEASURED,
            SampleStatus.UNAVAILABLE,
        )
        if samples[name].status is SampleStatus.UNAVAILABLE:
            assert samples[name].reason_code
            assert samples[name].unit == ""

    rusage = sample_rusage_self(wall_seconds_millionths=1_000_000)
    assert rusage["total_cpu_seconds"].status in (
        SampleStatus.MEASURED,
        SampleStatus.UNAVAILABLE,
    )

    missing = sample_process_tree_resources(
        2**30,  # extremely unlikely live PID
        wall_seconds_millionths=1_000_000,
    )
    assert missing["user_cpu_seconds"].status is SampleStatus.UNAVAILABLE
    assert missing["user_cpu_seconds"].reason_code in {
        UnavailableReason.SENSOR_ABSENT.value,
        UnavailableReason.PERMISSION_DENIED.value,
    }


def test_gpu_network_energy_unavailable_when_absent() -> None:
    gpu = sample_gpu_resources(accelerator_present=False)
    for name in (
        "gpu_utilization_time_weighted_ratio",
        "peak_vram_bytes",
        "gpu_seconds",
        "gpu_energy_joules_optional",
    ):
        assert gpu[name].status is SampleStatus.UNAVAILABLE
        assert gpu[name].reason_code == "hardware-absent"
        assert "value" not in gpu[name].to_envelope()

    network = sample_network_bytes()
    assert network["network_rx_bytes"].status is SampleStatus.UNAVAILABLE
    assert network["network_tx_bytes"].status is SampleStatus.UNAVAILABLE

    measured_net = sample_network_bytes(rx_bytes=10, tx_bytes=20)
    assert measured_net["network_rx_bytes"].value == 10
    assert measured_net["network_tx_bytes"].value == 20

    energy = sample_energy_optional(None)
    assert energy.status is SampleStatus.UNAVAILABLE
    assert energy.metric_name == "energy_joules_optional"


def test_scheduler_metrics_join_by_causal_span() -> None:
    events = [
        {
            "type": "task_ready",
            "timestamp": "2026-01-01T00:00:00Z",
            "goal_cid": "goal:g",
            "subgoal_cid": "subgoal:s",
            "task_cid": "task:PDR-071",
            "lane_id": "lane:1",
            "provider_id": "provider:example",
        },
        {
            "type": "implementation_started",
            "timestamp": "2026-01-01T00:00:10Z",
            "goal_cid": "goal:g",
            "subgoal_cid": "subgoal:s",
            "task_cid": "task:PDR-071",
            "lane_id": "lane:1",
            "provider_id": "provider:example",
            "attempt": 1,
        },
        {
            "type": "implementation_finished",
            "timestamp": "2026-01-01T00:00:40Z",
            "goal_cid": "goal:g",
            "subgoal_cid": "subgoal:s",
            "task_cid": "task:PDR-071",
            "lane_id": "lane:1",
            "provider_id": "provider:example",
            "returncode": 0,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "results": [
                    {
                        "started_at": "2026-01-01T00:00:25Z",
                        "finished_at": "2026-01-01T00:00:35Z",
                    }
                ],
            },
        },
        {
            "type": "merge_candidate_enqueued",
            "timestamp": "2026-01-01T00:00:42Z",
            "goal_cid": "goal:g",
            "subgoal_cid": "subgoal:s",
            "task_cid": "task:PDR-071",
            "lane_id": "lane:1",
            "provider_id": "provider:example",
        },
        {
            "type": "merge_started",
            "timestamp": "2026-01-01T00:00:50Z",
            "goal_cid": "goal:g",
            "subgoal_cid": "subgoal:s",
            "task_cid": "task:PDR-071",
            "lane_id": "lane:1",
            "provider_id": "provider:example",
        },
        {
            "type": "merge_finished",
            "timestamp": "2026-01-01T00:01:00Z",
            "goal_cid": "goal:g",
            "subgoal_cid": "subgoal:s",
            "task_cid": "task:PDR-071",
            "lane_id": "lane:1",
            "provider_id": "provider:example",
            "merged": True,
            "returncode": 0,
        },
    ]
    snapshot = build_scheduler_snapshot(events, now="2026-01-01T00:02:00Z")
    span = _root_span(task_id="PDR-071")

    projection = join_scheduler_snapshot_to_benchmark_span(snapshot, span)
    assert projection["schema"] == BENCHMARK_SPAN_CLOCK_SCHEMA
    assert projection["span_id"] == span.span_id
    assert projection["queue_wait_seconds"] == 10.0
    assert projection["merge_wait_seconds"] == 8.0
    assert projection["missing_dimensions_are_null"] is True
    assert projection["capacity_admission_unchanged"] is True

    samples = project_scheduler_clock_samples(
        snapshot,
        span,
        concurrency_one_makespan_seconds_millionths=seconds_to_millionths(60),
        accepted_criteria=1,
    )
    assert samples["end_to_end_makespan_seconds"].status is SampleStatus.MEASURED
    assert samples["queue_latency_p50_seconds"].status is SampleStatus.MEASURED
    assert samples["merge_conflict_serialization_seconds"].status is SampleStatus.MEASURED
    assert samples["critical_path_seconds"].status is SampleStatus.MEASURED

    empty = project_snapshot_metrics_for_span(
        {"metrics": [], "phase_counts": {}},
        span_id="span:empty",
        task_id="missing",
    )
    assert empty["queue_wait_seconds"] is None
    assert empty["end_to_end_makespan_seconds"] is None


def test_token_ledger_projection_includes_cancelled_and_bindings() -> None:
    span = _root_span()
    ledger = _ledger_with_cancel()
    assert ledger.report.cancelled_tokens == 20
    assert ledger.report.retry_tokens == 20

    totals = provider_native_token_totals_for_span(ledger, span.span_id)
    assert totals["provider_native_input_tokens"] == 60
    assert totals["provider_native_output_tokens"] == 10
    assert totals["provider_native_cancelled_tokens"] == 20
    assert totals["provider_native_retry_tokens"] == 20
    assert totals["model_call_count"] == 2

    samples = project_token_ledger_samples(ledger, span, provider_called=True)
    assert samples["provider_native_input_tokens"].value == 60
    assert samples["provider_native_cancelled_tokens"].value == 20
    assert samples["model_call_count"].value == 2
    assert samples["tokenizer_identity"].status is SampleStatus.MEASURED

    omitted = project_token_ledger_samples(
        ledger, span, provider_called=False
    )
    assert (
        omitted["provider_native_input_tokens"].status
        is SampleStatus.UNAVAILABLE
    )
    assert (
        omitted["provider_native_input_tokens"].reason_code
        == "provider-omitted"
    )


def test_span_joined_measurement_covers_required_dimensions() -> None:
    span = _root_span(hardware=_hardware(accelerator=False))
    ledger = _ledger_with_cancel()
    process = sample_rusage_self(
        wall_seconds_millionths=span.duration_seconds_millionths or 1_000_000
    )
    gpu = sample_gpu_resources(accelerator_present=False)
    network = sample_network_bytes(rx_bytes=0, tx_bytes=0)
    energy = sample_energy_optional(None)

    measurement = build_span_joined_measurement(
        measurement_id="meas:joined",
        span=span,
        token_ledger=ledger,
        process_samples=process,
        gpu_samples=gpu,
        network_samples=network,
        energy_sample=energy,
        provider_called=True,
        accepted_criteria=1,
        attributed_process_ids=(span.process_id,),
    )
    assert measurement.INTERFACE == "BenchmarkResourceMeasurement@1"
    assert measurement.require_sample(
        "provider_native_cancelled_tokens"
    ).value == 20
    assert measurement.require_sample("peak_vram_bytes").status is (
        SampleStatus.UNAVAILABLE
    )
    assert measurement.require_sample("energy_joules_optional").status is (
        SampleStatus.UNAVAILABLE
    )
    # Measured zero network bytes still carry a sensor receipt.
    assert measurement.require_sample("network_rx_bytes").status is (
        SampleStatus.MEASURED
    )
    assert measurement.require_sample("network_rx_bytes").value == 0
    assert measurement.require_sample("network_rx_bytes").sensor_id

    restored = BenchmarkResourceMeasurement.from_dict(measurement.to_record())
    assert restored.content_id == measurement.content_id
    cert = certify_measurement_from_source_spans(restored, (span,))
    assert cert.certified is True


def test_bind_attribution_to_span_is_idempotent_and_exclusive() -> None:
    ledger = _ledger_with_cancel()
    first = ledger.attributions[0]
    bound = bind_attribution_to_span(first, "span:run-1")
    assert bound.span_id == "span:run-1"
    again = bind_attribution_to_span(bound, "span:run-1")
    assert again.content_id == bound.content_id
    with pytest.raises(Exception):
        bind_attribution_to_span(bound, "span:other")
