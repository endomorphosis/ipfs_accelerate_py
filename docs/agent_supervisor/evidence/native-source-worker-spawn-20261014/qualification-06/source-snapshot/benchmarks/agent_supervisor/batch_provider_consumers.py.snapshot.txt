"""Bounded batch-consumer benchmark with authored native transport and real quota.

The CLI takes a live supervisor ResourceScheduler lease, then exercises the
production task consumer, native scheduler, physical admission, and an isolated
durable SystemClock ledger. Paired logical requests share a physical charge;
completed work has no output cache and charges again when submitted later.
No model, solver, child process, target program, or network provider executes.
Reported tokens are conservative declared ceilings, not measured provider bills.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time

from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import _call_text_provider
from ipfs_accelerate_py.agent_supervisor.runtime.batch_provider_usage import (
    BatchProviderUsageBinding, summarize_batch_usage,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchRequest, ProviderBatchScheduler, ProviderBatchSchedulerConfig,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_usage import PhysicalBatchUsageAdmission
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import ConsumerId
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    LaneResourceRequirements, ResourcePolicy, ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import LlmRouterInvocation
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope, LimitWindow, ProtocolKind, Quantity, UsageDimension,
    UsageLimit, WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import DurableUsageLedgerStore, SystemClock


SCHEMA = "batch-provider-consumers-authored-benchmark@1"
PROVIDER = "authored-native-batch"
MODEL = "authored-model"
ROUTE = "authored-task-proposal"
OPERATION = "task.proposal.v1"
INPUT_BOUND = 128
OUTPUT_BOUND = 16
SHARED_OVERHEAD = 3
CONSUMER = ConsumerId.TASK_PROPOSAL_ROUTER


class _CountingCoordinator(UsageCoordinator):
    def __init__(self, store):
        super().__init__(store, writer_id="batch-consumer-benchmark", fence=1)
        self.reserve_calls = 0
        self._counter_lock = threading.Lock()

    def reserve(self, *args, **kwargs):
        with self._counter_lock:
            self.reserve_calls += 1
        return super().reserve(*args, **kwargs)


class _ObservedScheduler(ProviderBatchScheduler):
    def __init__(self, *args, second_submitted, **kwargs):
        self._observed_submissions = 0
        self._submission_lock = threading.Lock()
        self._second_submitted = second_submitted
        super().__init__(*args, **kwargs)

    def submit(self, request):
        result = super().submit(request)
        with self._submission_lock:
            self._observed_submissions += 1
            if self._observed_submissions % 2 == 0:
                self._second_submitted.set()
        return result


def _expected_units(physical_calls):
    return {
        "requests": physical_calls,
        "batch_items": physical_calls,
        "input_tokens": physical_calls * (INPUT_BOUND + SHARED_OVERHEAD),
        "output_tokens": physical_calls * OUTPUT_BOUND,
        "total_tokens": physical_calls * (INPUT_BOUND + SHARED_OVERHEAD + OUTPUT_BOUND),
    }


def run_benchmark(request_count: int = 12) -> dict[str, object]:
    if type(request_count) is not int or not 2 <= request_count <= 32 or request_count % 2:
        raise ValueError("request_count must be an even integer between 2 and 32")
    started_at = time.monotonic()
    deadline = started_at + 30
    pairs = request_count // 2
    callback_started, second_submitted, finish_callback = (threading.Event() for _ in range(3))
    counter_lock = threading.Lock()
    callback_count = active_callbacks = peak_callbacks = 0
    phase = "setup"
    error = None
    before_replay = after_replay = summary = None
    quota_status = "not_attempted"
    pressure_reserve_attempts = 0
    attributions = []

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise AssertionError("batch consumer campaign exceeded 30 seconds")
        return seconds

    scope = EndpointUsageScope(
        provider_id=stable_id("provider", "batch-consumer-benchmark"),
        protocol=ProtocolKind.HTTPS, operation="text.generate",
        deployment_id=stable_id("deployment", "authored-native-batch"),
        credential_pseudonym=stable_id("credential", "authored-batch-benchmark"),
    )
    with tempfile.TemporaryDirectory(prefix="batch-provider-consumers-") as directory:
        store = DurableUsageLedgerStore(
            Path(directory) / "ledger.json", clock=SystemClock(),
            writer_id="batch-consumer-benchmark", fence=1,
        )
        coordinator = _CountingCoordinator(store)
        values = {
            UsageDimension.CONCURRENT_REQUESTS: 1,
            UsageDimension.REQUESTS: pairs + 1,
            UsageDimension.BATCH_ITEMS: 100,
            UsageDimension.INPUT_TOKENS: 100_000,
            UsageDimension.OUTPUT_TOKENS: 100_000,
            UsageDimension.TOTAL_TOKENS: 200_000,
        }
        coordinator.configure_limits(scope.scope_id, [UsageLimit(
            scope_id=scope.scope_id, dimension=dimension, ceiling=Quantity.finite(amount),
            window=LimitWindow(kind=(WindowKind.CONCURRENT
                if dimension is UsageDimension.CONCURRENT_REQUESTS else WindowKind.LIFETIME)),
        ) for dimension, amount in values.items()])
        admission = PhysicalBatchUsageAdmission(
            coordinator, scope_supplier=lambda _provider: scope.scope_id,
            max_batch_tokens=100_000, reservation_ttl_ms=15_000,
            shared_overhead_tokens=SHARED_OVERHEAD,
            retry_initial_ms=100, retry_max_ms=1_000, require_hard_limits=True,
        )

        def dispatch(requests):
            nonlocal callback_count, active_callbacks, peak_callbacks
            with counter_lock:
                callback_count += 1
                active_callbacks += 1
                peak_callbacks = max(peak_callbacks, active_callbacks)
            try:
                assert len(requests) == 1, "duplicate subscribers reached physical dispatch"
                member = requests[0]
                assert (member.provider_id, member.model, member.usage_scope_id) == (
                    PROVIDER, MODEL, scope.scope_id,
                )
                assert member.token_budget == OUTPUT_BOUND
                holds = [record for record in store.read()["reservations"].values()
                         if record["state"] in ("held", "pending")]
                assert len(holds) == 1 and holds[0]["dispatched"]
                callback_started.set()
                # This authored callback has a five-second bound, below the
                # consumer's six-second declared physical callback lifetime.
                assert finish_callback.wait(min(5, remaining())), "late subscriber did not join"
                return ["authored proposal"]
            finally:
                with counter_lock:
                    active_callbacks -= 1

        scheduler = _ObservedScheduler(
            dispatch, admission=admission, second_submitted=second_submitted,
            config=ProviderBatchSchedulerConfig(
                max_batch_size=8, max_parallel_batches=1, max_queue_size=32,
                batch_window_ms=0, admission_retry_ms=5,
            ),
        )
        callers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="batch-consumer-benchmark")
        try:
            binding = BatchProviderUsageBinding(
                scheduler=scheduler, admission=admission, scope=scope,
                provider_id=PROVIDER, model=MODEL, route=ROUTE, operation=OPERATION,
                fence=1, single_physical_request=True,
            )
            invocation = LlmRouterInvocation(
                repo_root=Path(directory), provider=PROVIDER, model_name=MODEL,
                timeout_seconds=6, timeout_grace_seconds=0, max_new_tokens=OUTPUT_BOUND,
                allow_local_fallback=False, allow_cross_provider_fallback=False,
            )

            def call(pair):
                with public_replay_without_metadata():
                    text, result = _call_text_provider(
                        f"Plan authored task {pair}.", invocation, scheduler=None,
                        route=ROUTE, operation=OPERATION, context_limit=INPUT_BOUND,
                        response_contract="AuthoredProposal/v1",
                        provenance={"task_id": f"task:authored-{pair}"},
                        batch_usage_binding=binding,
                    )
                assert text == "authored proposal" and result is not None
                return result

            phase = "late_duplicate_coalescing"
            for pair in range(pairs):
                remaining()
                callback_started.clear()
                second_submitted.clear()
                finish_callback.clear()
                first = callers.submit(call, pair)
                assert callback_started.wait(min(5, remaining())), "first callback did not start"
                second = callers.submit(call, pair)
                assert second_submitted.wait(min(5, remaining())), "second logical request was not submitted"
                finish_callback.set()
                first_result = first.result(timeout=remaining())
                second_result = second.result(timeout=remaining())
                assert first_result.request_id != second_result.request_id
                assert first_result.execution_id == second_result.execution_id
                assert first_result.physical_usage == second_result.physical_usage
                for result in (first_result, second_result):
                    attribution = binding.attribution_for(result.request_id)
                    assert attribution is not None
                    attributions.append(attribution)
                assert scheduler.flush(remaining()), "pair did not release physical worker"
            assert callback_count == pairs
            before_replay = summarize_batch_usage(attributions)
            assert before_replay["logical_requests"] == request_count
            assert before_replay["physical_calls"] == pairs
            assert before_replay["settled"] == _expected_units(pairs)
            after_replay = summarize_batch_usage([*attributions, *attributions, *attributions[:2]])
            assert after_replay == before_replay, "attribution replay duplicated physical usage"

            phase = "completed_identical_request"
            finish_callback.set()
            repeated = callers.submit(call, 0).result(timeout=remaining())
            repeated_attribution = binding.attribution_for(repeated.request_id)
            assert repeated_attribution is not None
            assert repeated_attribution.physical_charge_id not in {
                item.physical_charge_id for item in attributions
            }
            attributions.append(repeated_attribution)
            assert scheduler.flush(remaining())
            summary = summarize_batch_usage(attributions)
            assert summary["logical_requests"] == request_count + 1
            assert summary["physical_calls"] == pairs + 1
            assert summary["settled"] == _expected_units(pairs + 1)
            assert callback_count == pairs + 1

            phase = "quota_exhaustion"
            attempts_before_pressure = coordinator.reserve_calls
            refused = binding.execute(ProviderBatchRequest(
                request_id="batch-consumer-quota-probe", payload="Plan exhausted authored task.",
                provider_id=PROVIDER, model=MODEL, route=ROUTE, operation=OPERATION,
                context_limit=INPUT_BOUND, token_budget=OUTPUT_BOUND, timeout_ms=180,
                generation_settings={"allow_local_fallback": False,
                    "allow_cross_provider_fallback": False, "disable_model_retry": True,
                    "max_new_tokens": OUTPUT_BOUND},
            ), consumer_id=CONSUMER)
            quota_status = refused.status.value
            pressure_reserve_attempts = coordinator.reserve_calls - attempts_before_pressure
            assert not refused.successful and not refused.physical_usage
            assert 1 <= pressure_reserve_attempts <= 3, "quota refusal caused a reservation storm"
            assert callback_count == pairs + 1
            assert scheduler.flush(remaining())
            phase = "cleanup_validation"
        except Exception as exc:
            error = exc
        finally:
            finish_callback.set()
            scheduler.shutdown(wait=True, cancel_pending=True)
            callers.shutdown(wait=True, cancel_futures=True)

        metrics = scheduler.metrics()
        document = store.read()
        records = tuple(document["reservations"].values())
        remaining_holds = sum(record["state"] in ("held", "pending") for record in records)
        charges = {name: sum(record.get("charged_amounts", {}).get(name, 0) for record in records)
                   for name in _expected_units(0)}
        if error is None:
            try:
                remaining()
                assert len(records) == pairs + 1 and all(record["state"] == "committed" for record in records)
                assert charges == _expected_units(pairs + 1)
                assert remaining_holds == metrics.active_batches == metrics.queued_requests == 0
                assert peak_callbacks == 1 and metrics.singleflight_hits == pairs
            except Exception as exc:
                error = exc
        store.close()
        report = {
            "schema": SCHEMA, "status": "failed" if error else "completed",
            "paired_logical_requests": request_count, "paired_physical_calls": pairs,
            "physical_callbacks_observed": callback_count, "peak_physical_callbacks": peak_callbacks,
            "caller_threads": 2, "configured_parallel_batches": 1,
            "singleflight_hits": metrics.singleflight_hits,
            "initial_usage_summary": before_replay,
            "replayed_attribution_summary": after_replay,
            "after_completed_resubmission_summary": summary,
            "quota_refusal_status": quota_status,
            "pressure_reserve_attempts": pressure_reserve_attempts,
            "total_reserve_attempts": coordinator.reserve_calls,
            "remaining_usage_holds": remaining_holds,
            "remaining_batches": metrics.active_batches, "remaining_queue": metrics.queued_requests,
            "settled_estimated_units": charges, "clock": "SystemClock",
            "store": "isolated durable CAS ledger", "ledger_revision": document["revision"],
            "elapsed_ms": round((time.monotonic() - started_at) * 1000),
            "campaign_deadline_ms": 30_000,
            "live_provider_calls": 0, "solver_executions": 0, "target_executions": 0,
            "claims_measured_provider_billing": False, "claims_provider_token_savings": False,
            "cross_process_store_tested": False, "hard_process_containment_tested": False,
        }
        if error is not None:
            report.update(failed_phase=phase, error_class=type(error).__name__, reason=str(error),
                          reservation_states={state: sum(item["state"] == state for item in records)
                                              for state in sorted({item["state"] for item in records})})
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    host = ResourceScheduler(ResourcePolicy(max_lanes=1))
    with public_replay_without_metadata():
        decision, lease = host.acquire(LaneResourceRequirements(
            lane_id="batch-provider-consumer-benchmark", stage="analysis",
            memory_bytes=256 * 1024**2, process_slots=1,
        ))
    if lease is None:
        report = {"schema": SCHEMA, "status": "deferred", "reason": decision.reason,
                  "live_host_preflight": "refused", "live_provider_calls": 0}
        code = 2
    else:
        try:
            with public_replay_without_metadata():
                report = run_benchmark(args.requests)
            report.update(live_host_preflight="admitted", live_host_reserved_process_slots=1,
                          live_host_reserved_memory_bytes=256 * 1024**2)
            code = 0 if report["status"] == "completed" else 1
        finally:
            with public_replay_without_metadata():
                host.release(lease)
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
