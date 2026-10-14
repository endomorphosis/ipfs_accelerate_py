"""Bounded stub-provider stress run; no network, model, solver, or target execution.

Pressure is injected as telemetry, rather than imposed on the host. The real
batch queue and resource admission owner are exercised with two provider worker
threads plus their coordinator. Only aggregate measurements are exported.
The standalone CLI first reserves a live-host admission lease for 256 MiB and
two worker slots, and defers cleanly under pressure. That local preflight does
not provide a cross-process reservation or hard process containment.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
    ProviderBatchStatus,
    ResourceSchedulerBatchAdmission,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProviderCapacity,
    ResourcePolicy,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


def run_benchmark(request_count: int = 32) -> dict[str, object]:
    if isinstance(request_count, bool) or not isinstance(request_count, int) or not 8 <= request_count <= 128:
        raise ValueError("request_count must be an integer between 8 and 128")
    started = time.monotonic()
    deadline = started + 20
    lock = threading.RLock()
    phase = "cpu"
    observed = {name: threading.Event() for name in ("cpu", "memory", "provider")}
    phase_ms: dict[str, int] = {}
    first_running = threading.Event()
    two_running = threading.Event()
    release_dispatch = threading.Event()
    active_callbacks = 0
    peak_callbacks = 0
    peak_leases = 0
    physical_calls = 0
    dispatched: set[str] = set()
    denial_counts: dict[str, int] = {}
    resource = ResourceScheduler(ResourcePolicy(max_lanes=2, max_model_concurrency=2))

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise AssertionError("stub campaign exceeded its 20 second deadline")
        return value

    def host():
        with lock:
            current = phase
        return HostResourceSnapshot(
            cpu_percent=95 if current == "cpu" else 10,
            memory_percent=95 if current == "memory" else 10,
            disk_percent=10, memory_total_bytes=8 * 1024**3,
            memory_available_bytes=4 * 1024**3, disk_total_bytes=8 * 1024**3,
            disk_available_bytes=4 * 1024**3, worker_limit=2,
            available_worker_capacity=2, capabilities=("cpu",),
            resource_classes=("cpu-small", "llm-proof-draft"),
        )

    def providers(_provider_id):
        with lock:
            current = phase
        return (ProviderCapacity(
            provider_id="stub", max_concurrency=2, quota_remaining=1000,
            token_budget_remaining=100_000, context_window_tokens=1024,
            retry_after_ms=100 if current == "provider" else 0,
        ),)

    adapter = ResourceSchedulerBatchAdmission(
        resource, host_supplier=host, provider_supplier=providers, memory_bytes=1024**2,
    )

    def admission(key, members, capacity):
        nonlocal peak_leases
        # This executes on the coordinator thread, where metadata suppression
        # must be installed explicitly instead of relying on inherited context.
        with public_replay_without_metadata():
            grant = adapter(key, members, capacity)
        with lock:
            peak_leases = max(peak_leases, len(resource.active_leases))
            if not grant.admitted:
                denial_counts[grant.reason] = denial_counts.get(grant.reason, 0) + 1
                for name, reason in (("cpu", "host_cpu_high_watermark"),
                                     ("memory", "host_memory_high_watermark"),
                                     ("provider", "provider_backoff")):
                    if grant.reason == reason:
                        observed[name].set()
        return grant

    def dispatch(members):
        nonlocal active_callbacks, peak_callbacks, physical_calls
        ids = {member.request_id for member in members}
        with lock:
            assert phase == "recovered", "a provider callback escaped backpressure"
            assert not ids.intersection({"unsafe-seal", "unsafe-trace", "cancel-queued"})
            assert not dispatched.intersection(ids), "duplicate physical execution"
            dispatched.update(ids)
            active_callbacks += 1
            peak_callbacks = max(peak_callbacks, active_callbacks)
            physical_calls += 1
            if "cancel-running" in ids:
                first_running.set()
            if active_callbacks == 2:
                two_running.set()
        try:
            assert release_dispatch.wait(remaining()), "stub dispatch gate did not reopen"
            return ["accepted" for _ in members]
        finally:
            with lock:
                active_callbacks -= 1

    def request(identity, **claims):
        return ProviderBatchRequest(
            request_id=identity, payload={"fixture": identity}, provider_id="stub",
            route="admission-benchmark", model="stub", operation="generate",
            context_limit=128, token_budget=8, timeout_ms=19_000,
            provenance={"stage": "inference", "input_sealed": True, **claims},
        )

    scheduler = ProviderBatchScheduler(
        dispatch, admission=admission,
        config=ProviderBatchSchedulerConfig(
            max_batch_size=4, min_batch_size=4, batch_window_ms=0,
            max_queue_size=128, max_parallel_batches=2, provider_limits={"stub": 2},
            admission_retry_ms=2, fallback_on_dispatch_error=False,
        ),
    )
    try:
        running_cancel = scheduler.submit(request("cancel-running"))
        valid = [scheduler.submit(request(f"valid-{index}")) for index in range(request_count - 4)]
        unsafe = [
            scheduler.submit(request("unsafe-seal", input_sealed=False)),
            scheduler.submit(request("unsafe-trace", trace_identity="old", expected_trace_identity="new")),
        ]
        queued_cancel = scheduler.submit(request("cancel-queued"))
        assert scheduler.cancel("cancel-queued")
        assert queued_cancel.cancelled()

        for name, next_phase in (("cpu", "memory"), ("memory", "provider"), ("provider", "recovered")):
            phase_started = time.monotonic()
            assert observed[name].wait(remaining()), f"{name} pressure was not observed"
            with lock:
                assert physical_calls == 0 and not resource.active_leases
                phase_ms[name] = max(0, round((time.monotonic() - phase_started) * 1000))
                phase = next_phase

        assert first_running.wait(remaining())
        assert two_running.wait(remaining()), "campaign did not exercise both provider slots"
        assert scheduler.cancel("cancel-running")
        with lock:
            leases_during_cancel = len(resource.active_leases)
            assert leases_during_cancel > 0, "running cancellation prematurely released capacity"
        try:
            running_cancel.result(timeout=remaining())
        except CancelledError:
            pass
        else:
            raise AssertionError("cancelled future unexpectedly returned a result")
        release_dispatch.set()
        valid_results = [future.result(timeout=remaining()) for future in valid]
        unsafe_results = [future.result(timeout=remaining()) for future in unsafe]
        assert all(result.status is ProviderBatchStatus.SUCCEEDED for result in valid_results)
        assert all(result.status is ProviderBatchStatus.FAILED for result in unsafe_results)
        assert "unsealed_data" in unsafe_results[0].error
        assert "stale_trace" in unsafe_results[1].error
        assert scheduler.flush(remaining())
    finally:
        release_dispatch.set()
        scheduler.shutdown(cancel_pending=True)

    metrics = scheduler.metrics()
    assert not resource.active_leases and metrics.active_batches == 0 and metrics.queued_requests == 0
    assert peak_callbacks <= 2 and peak_leases <= 2
    assert metrics.admission_errors == metrics.capacity_errors == 0
    assert len(dispatched) == request_count - 3
    assert metrics.cancelled_requests == 2 and metrics.failed_requests == 2
    elapsed_ms = max(1, round((time.monotonic() - started) * 1000))
    return {
        "schema": "provider-resource-admission-stub-benchmark@1",
        "scope": "real queue and admission owner; authored stub callbacks and injected telemetry",
        "request_count": request_count, "succeeded": len(valid_results), "rejected_unsafe": 2,
        "cancelled_queued": 1, "cancelled_running": 1, "physical_calls": physical_calls,
        "physical_members": len(dispatched), "duplicate_executions": metrics.duplicate_executions,
        "peak_provider_callbacks": peak_callbacks, "peak_resource_leases": peak_leases,
        "leases_held_during_running_cancellation": leases_during_cancel,
        "remaining_leases": len(resource.active_leases), "remaining_batches": metrics.active_batches,
        "remaining_queue": metrics.queued_requests, "pressure_observations_ms": phase_ms,
        "admission_denials": dict(sorted(denial_counts.items())),
        "elapsed_ms": elapsed_ms, "completed_requests_per_second_milli": request_count * 1_000_000 // elapsed_ms,
        "live_provider_calls": 0, "target_executions": 0, "solver_executions": 0,
        "claims_provider_token_savings": False, "hard_process_containment_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    live_owner = ResourceScheduler(ResourcePolicy(max_lanes=2))
    with public_replay_without_metadata():
        decision, lease = live_owner.acquire(LaneResourceRequirements(
            lane_id="provider-admission-stub-benchmark", stage="analysis",
            memory_bytes=256 * 1024**2, process_slots=2,
        ))
    if lease is None:
        report = {
            "schema": "provider-resource-admission-stub-benchmark@1",
            "status": "deferred", "reason": decision.reason,
            "live_host_preflight": "refused", "live_provider_calls": 0,
            "target_executions": 0, "solver_executions": 0,
        }
        exit_code = 2
    else:
        try:
            report = run_benchmark(args.requests)
            report.update(status="completed", live_host_preflight="admitted",
                          live_host_reserved_memory_bytes=256 * 1024**2,
                          live_host_reserved_process_slots=2)
            exit_code = 0
        finally:
            live_owner.release(lease)
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
