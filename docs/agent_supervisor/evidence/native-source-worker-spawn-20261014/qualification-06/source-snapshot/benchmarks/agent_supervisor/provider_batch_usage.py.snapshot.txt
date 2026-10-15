"""Bounded physical usage benchmark with two queues and authored callbacks.

The standalone CLI first acquires a live-host lease. The campaign then uses an
isolated in-memory endpoint ledger and controlled host telemetry. No provider,
network, model, solver, container, or target program is invoked. Aggregate JSON
reports declared token ceilings; it makes no claim about measured provider bills.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchCapacity,
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
    ProviderBatchStatus,
    ResourceSchedulerBatchAdmission,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_usage import PhysicalBatchUsageAdmission
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProviderCapacity,
    ResourcePolicy,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.ledger import UsageLedger
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitWindow,
    Quantity,
    UsageDimension,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import InMemoryUsageLedgerStore, SystemClock


SCHEMA = "provider-batch-usage-stub-benchmark@1"


class _CountingCoordinator(UsageCoordinator):
    def __init__(self, store):
        super().__init__(store)
        self.reserve_calls = 0
        self._counter_lock = threading.Lock()

    def reserve(self, *args, **kwargs):
        with self._counter_lock:
            self.reserve_calls += 1
        return super().reserve(*args, **kwargs)


def run_benchmark(request_count: int = 32) -> dict[str, object]:
    if isinstance(request_count, bool) or not isinstance(request_count, int) or not 8 <= request_count <= 128:
        raise ValueError("request_count must be an integer between 8 and 128")
    started = time.monotonic()
    deadline = started + 20
    lock = threading.RLock()
    quota_backoff_seen = threading.Event()
    first_running = threading.Event()
    two_running = threading.Event()
    release_callbacks = threading.Event()
    active_callbacks = peak_callbacks = peak_host_leases = physical_calls = 0
    physical_members: set[str] = set()
    denials: dict[str, int] = {}
    store = InMemoryUsageLedgerStore(clock=SystemClock())
    coordinator = _CountingCoordinator(store)
    scope = stable_id("scope", "physical-batch-usage-benchmark")
    limits = []
    for dimension, ceiling in (
        (UsageDimension.CONCURRENT_REQUESTS, 2),
        (UsageDimension.REQUESTS, 1_000),
        (UsageDimension.BATCH_ITEMS, 1_000),
        (UsageDimension.INPUT_TOKENS, 100_000),
        (UsageDimension.OUTPUT_TOKENS, 100_000),
        (UsageDimension.TOTAL_TOKENS, 100_000),
    ):
        limits.append(UsageLimit(
            scope_id=scope, dimension=dimension, ceiling=Quantity.finite(ceiling),
            window=LimitWindow(kind=(
                WindowKind.CONCURRENT if dimension is UsageDimension.CONCURRENT_REQUESTS else WindowKind.LIFETIME
            )),
        ))
    coordinator.configure_limits(scope, limits)
    external = coordinator.reserve(
        scope, UsageVector.of(concurrent_requests=2), request_id="outside-work",
        idempotency_key="outside-work", owner_id="benchmark-outside", ttl_ms=60_000,
    )
    assert external.granted
    # Host admission permits four; the shared endpoint quota must enforce two
    # even though each independent queue also permits two active batches.
    resource = ResourceScheduler(ResourcePolicy(max_lanes=4, max_model_concurrency=4))
    controlled_host = HostResourceSnapshot(
        cpu_percent=10, memory_percent=10, disk_percent=10,
        memory_total_bytes=8 * 1024**3, memory_available_bytes=4 * 1024**3,
        disk_total_bytes=8 * 1024**3, disk_available_bytes=4 * 1024**3,
        worker_limit=4, available_worker_capacity=4, capabilities=("cpu",),
        resource_classes=("cpu-small", "llm-proof-draft"),
    )
    host_adapter = ResourceSchedulerBatchAdmission(
        resource, host_supplier=lambda: controlled_host,
        provider_supplier=lambda _provider: (ProviderCapacity(
            provider_id="stub", max_concurrency=4, quota_remaining=1_000,
            token_budget_remaining=100_000, context_window_tokens=1_024,
        ),),
        memory_bytes=1024**2,
    )
    usage_adapter = PhysicalBatchUsageAdmission(
        coordinator, scope_supplier=lambda _provider: scope,
        max_batch_tokens=100_000, reservation_ttl_ms=60_000,
        shared_overhead_tokens=3, host_admission=host_adapter,
    )

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise AssertionError("physical usage campaign exceeded 20 seconds")
        return value

    def admission(key, requests, capacity):
        nonlocal peak_host_leases
        with public_replay_without_metadata():
            grant = usage_adapter(key, requests, capacity)
        with lock:
            peak_host_leases = max(peak_host_leases, len(resource.active_leases))
            if not grant.admitted:
                denials[grant.reason] = denials.get(grant.reason, 0) + 1
                if grant.reason == "usage_reservation_backoff":
                    quota_backoff_seen.set()
        return grant

    def dispatch(requests):
        nonlocal active_callbacks, peak_callbacks, physical_calls
        fingerprints = {request.execution_fingerprint for request in requests}
        with lock:
            assert not physical_members.intersection(fingerprints), "duplicate physical execution"
            assert all(request.request_id != "cancel-queued" for request in requests)
            physical_members.update(fingerprints)
            physical_calls += 1
            active_callbacks += 1
            peak_callbacks = max(peak_callbacks, active_callbacks)
            if any(request.request_id == "cancel-running" for request in requests):
                first_running.set()
            if active_callbacks == 2:
                two_running.set()
            assert active_callbacks <= 2, "shared endpoint concurrency exceeded"
        try:
            assert release_callbacks.wait(remaining()), "callback gate did not reopen"
            if requests[0].route == "fault":
                raise RuntimeError("authored unknown provider outcome")
            return ["accepted" for _ in requests]
        finally:
            with lock:
                active_callbacks -= 1

    def request(identity, payload, *, route="usage-benchmark"):
        return ProviderBatchRequest(
            request_id=identity, payload={"fixture": payload}, provider_id="stub",
            route=route, model="stub", operation="generate", usage_scope_id=scope,
            context_limit=128, token_budget=8, timeout_ms=19_000,
            provenance={"stage": "inference", "input_sealed": True},
        )

    schedulers = [ProviderBatchScheduler(
        dispatch, admission=admission,
        # Ensure the cancellation anchor owns one first call before the other
        # independent queue competes for the shared endpoint's remaining slot.
        capacity_supplier=lambda _provider, index=index: ProviderBatchCapacity(
            provider_id="stub", healthy=True, max_batch_size=4,
            max_concurrent_batches=2,
            available_concurrent_batches=(0 if index == 1 and not first_running.is_set() else 2),
        ),
        config=ProviderBatchSchedulerConfig(
            max_batch_size=4, min_batch_size=4, batch_window_ms=0,
            max_queue_size=128, max_parallel_batches=2, provider_limits={"stub": 2},
            admission_retry_ms=5, fallback_on_dispatch_error=True,
        ),
        fallback=lambda _request: (_ for _ in ()).throw(AssertionError("unaccounted fallback invoked")),
    ) for index in range(2)]
    campaign_error: Exception | None = None
    phase = "quota_pressure"
    pressure_reserve_attempts = held_during_cancel = 0
    results = []
    try:
        running_cancel = schedulers[0].submit(request("cancel-running", "shared-anchor"))
        valid = [schedulers[(index // 2) % 2].submit(request(f"valid-{index}", f"pair-{index // 2}"))
                 for index in range(request_count - 4)]
        failed = schedulers[1].submit(request("failure", "failure", route="fault"))
        queued_cancel = schedulers[0].submit(request("cancel-queued", "unused"))
        assert schedulers[0].cancel("cancel-queued")
        assert quota_backoff_seen.wait(remaining()), "quota pressure never applied backoff"
        with lock:
            assert physical_calls == 0
        pressure_reserve_attempts = coordinator.reserve_calls - 1
        assert pressure_reserve_attempts <= 2, "quota denial generated a reservation storm"
        coordinator.cancel(external.reservation_id)
        phase = "quota_recovery"
        assert first_running.wait(remaining()), "first callback did not start after quota recovery"
        phase = "concurrent_callbacks"
        assert two_running.wait(remaining()), "second callback did not start within shared quota"
        sibling = schedulers[0].submit(request("late-sibling", "shared-anchor"))
        assert schedulers[0].cancel("cancel-running")
        held_during_cancel = sum(
            record["state"] == "held"
            for record in UsageLedger(store.read()).iter_reservation_records(scope)
        )
        assert held_during_cancel == 2
        phase = "settlement"
        release_callbacks.set()
        results = [future.result(timeout=remaining()) for future in (*valid, sibling)]
        failure = failed.result(timeout=remaining())
        assert all(result.status is ProviderBatchStatus.SUCCEEDED for result in results)
        assert failure.status is ProviderBatchStatus.FAILED
        assert failure.physical_usage["settlement_basis"] == "conservative_live_ceiling"
        for future in (running_cancel, queued_cancel):
            try:
                future.result(timeout=remaining())
            except CancelledError:
                pass
            else:
                raise AssertionError("cancelled logical request returned an output")
        assert all(scheduler.flush(remaining()) for scheduler in schedulers)
    except Exception as exc:
        campaign_error = exc
    finally:
        release_callbacks.set()
        for scheduler in schedulers:
            scheduler.shutdown(cancel_pending=True)
        coordinator.cancel(external.reservation_id)

    records = list(UsageLedger(store.read()).iter_reservation_records(scope))
    batch_records = [record for record in records if record["reservation_id"] != external.reservation_id]
    metrics = [scheduler.metrics() for scheduler in schedulers]
    expected_members = 2 + (request_count - 3) // 2
    settled_tokens = sum(record["charged_amounts"].get("total_tokens", 0) for record in batch_records)
    if campaign_error is None:
        phase = "validation"
        try:
            assert len(physical_members) == expected_members, "physical member count differs"
            assert len(batch_records) == physical_calls, "reservation count differs from physical calls"
            assert all(record["state"] == "committed" for record in batch_records), "usage settlement incomplete"
            assert not resource.active_leases, "host lease remains"
            assert all(item.active_batches == item.queued_requests == 0 for item in metrics), "queue cleanup incomplete"
            assert settled_tokens == expected_members * 136 + physical_calls * 3, "declared ceiling accounting differs"
            assert sum(item.cancelled_requests for item in metrics) == 2, "cancellation count differs"
            assert sum(item.failed_requests for item in metrics) == 1, "failure count differs"
            assert sum(item.admission_errors + item.capacity_errors for item in metrics) == 0, "admission lifecycle errors"
        except Exception as exc:
            campaign_error = exc
    elapsed_ms = max(1, round((time.monotonic() - started) * 1000))
    report = {
        "schema": SCHEMA,
        "status": "failed" if campaign_error else "completed",
        "scope": "two real queues and isolated host/usage owners; authored callbacks",
        "request_count": request_count,
        "succeeded": sum(item.succeeded_requests for item in metrics),
        "failed": sum(item.failed_requests for item in metrics),
        "cancelled": sum(item.cancelled_requests for item in metrics),
        "timed_out": sum(item.timed_out_requests for item in metrics),
        "scheduler_count": 2, "configured_parallel_batches_per_scheduler": 2,
        "shared_endpoint_concurrency_limit": 2, "peak_provider_callbacks": peak_callbacks,
        "physical_calls": physical_calls, "physical_members": len(physical_members),
        "singleflight_hits": sum(item.singleflight_hits for item in metrics),
        "duplicate_executions": sum(item.duplicate_executions for item in metrics),
        "pressure_reserve_attempts": pressure_reserve_attempts,
        "total_reserve_attempts": coordinator.reserve_calls,
        "endpoint_ledger_revision": store.read()["revision"],
        "held_reservations_during_running_cancellation": held_during_cancel,
        "remaining_usage_holds": sum(record["state"] in ("held", "pending") for record in records),
        "remaining_host_leases": len(resource.active_leases), "peak_host_leases": peak_host_leases,
        "remaining_batches": sum(item.active_batches for item in metrics),
        "remaining_queue": sum(item.queued_requests for item in metrics),
        "admission_denials": dict(sorted(denials.items())),
        "settled_declared_token_ceilings": settled_tokens,
        "elapsed_ms": elapsed_ms,
        "live_provider_calls": 0, "target_executions": 0, "solver_executions": 0,
        "claims_provider_token_savings": False, "hard_process_containment_tested": False,
        "cross_process_store_tested": False,
    }
    if campaign_error is not None:
        report.update(
            failed_phase=phase, error_class=type(campaign_error).__name__,
            reason=str(campaign_error), first_callback_observed=first_running.is_set(),
            two_callbacks_observed=two_running.is_set(),
            admission_errors=sum(item.admission_errors for item in metrics),
            capacity_errors=sum(item.capacity_errors for item in metrics),
            reservation_states={state: sum(record["state"] == state for record in records)
                                for state in sorted({record["state"] for record in records})},
        )
    else:
        report.update(cancelled_queued=1, cancelled_running=1)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    live_owner = ResourceScheduler(ResourcePolicy(max_lanes=4))
    with public_replay_without_metadata():
        decision, lease = live_owner.acquire(LaneResourceRequirements(
            lane_id="provider-batch-usage-stub-benchmark", stage="analysis",
            memory_bytes=256 * 1024**2, process_slots=4,
        ))
    if lease is None:
        report = {
            "schema": SCHEMA, "status": "deferred", "reason": decision.reason,
            "live_host_preflight": "refused", "live_provider_calls": 0,
            "target_executions": 0, "solver_executions": 0,
        }
        exit_code = 2
    else:
        try:
            report = run_benchmark(args.requests)
            report.update(live_host_preflight="admitted",
                          live_host_reserved_memory_bytes=256 * 1024**2,
                          live_host_reserved_process_slots=4)
            exit_code = 0 if report["status"] == "completed" else 1
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
