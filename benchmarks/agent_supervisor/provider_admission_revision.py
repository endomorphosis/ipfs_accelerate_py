"""Bounded real-clock gateway qualification with an isolated durable ledger.

Only authored callbacks execute. The CLI checks live local host admission before
running; this is not a provider-billing or cross-process throughput benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time

from ipfs_accelerate_py.agent_supervisor.runtime.provider_execution import (
    ProviderExecutionGateway, ProviderExecutionPhase,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import (
    ConsumerId, ProviderUsageMigrationError, build_consumer_call_context,
    dispatch_migrated_provider_call,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    LaneResourceRequirements, ResourcePolicy, ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitWindow, Quantity, UsageDimension, UsageLimit, UsageVector, WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import DurableUsageLedgerStore, SystemClock


SCHEMA = "provider-admission-revision-stub-benchmark@1"


def run_benchmark(request_count: int = 16) -> dict[str, object]:
    if type(request_count) is not int or not 2 <= request_count <= 64:
        raise ValueError("request_count must be an integer between 2 and 64")
    started = time.monotonic()
    callbacks = 0
    replay_count = 0
    restart_denials = 0
    scope = stable_id("scope", "admission-revision-benchmark")

    def invoke():
        nonlocal callbacks
        callbacks += 1
        return "authored response"

    with tempfile.TemporaryDirectory(prefix="provider-admission-revision-") as directory:
        ledger_path = Path(directory) / "ledger.json"
        store = DurableUsageLedgerStore(ledger_path, clock=SystemClock())
        owner = UsageCoordinator(store)
        limits = {"requests": request_count + 1, "input_tokens": 100_000, "output_tokens": 100_000}
        owner.configure_limits(scope, [UsageLimit(
            scope_id=scope, dimension=UsageDimension(name), ceiling=Quantity.finite(amount),
            window=LimitWindow(kind=WindowKind.LIFETIME),
        ) for name, amount in limits.items()])
        gateway = ProviderExecutionGateway(coordinator=owner)

        def context(identity, view=None, *, legacy=False):
            view = view or owner.admission_snapshot(scope)
            return build_consumer_call_context(
                consumer_id=ConsumerId.TASK_PROPOSAL_ROUTER, provider_id="authored-provider",
                endpoint_scope_id=scope, usage_revision=view.snapshot.usage_revision,
                admission_revision="" if legacy else view.admission_revision,
                request_id=identity, idempotency_key="idem:" + identity, fence_id="1",
                estimated_input_tokens=64, estimated_output_tokens=16,
            )

        def dispatch(call_context, selected_gateway=gateway):
            return dispatch_migrated_provider_call(
                context=call_context, invoke=invoke, mode="enforce", gateway=selected_gateway,
            )

        def expect_stale(call_context):
            try:
                dispatch(call_context)
            except ProviderUsageMigrationError as exc:
                assert "stale_coordination" in exc.reason_codes
            else:
                raise AssertionError("stale precondition must refuse the call")

        initial = owner.admission_snapshot(scope)
        later = owner.admission_snapshot(scope)
        assert initial.admission_revision == later.admission_revision
        assert initial.snapshot.usage_revision != later.snapshot.usage_revision
        expect_stale(context("legacy-observation-control", initial, legacy=True))
        assert callbacks == 0

        for index in range(request_count):
            assert time.monotonic() - started < 20, "bounded campaign deadline exceeded"
            call_context = context(f"success-{index}")
            # A read at a later real-clock instant must not stale the state pin.
            assert owner.admission_snapshot(scope).admission_revision == call_context.admission_revision
            result = dispatch(call_context)
            assert result.execution_result.phase is ProviderExecutionPhase.SETTLED
            assert result.text == "authored response"
            assert result.usage_receipt.scope.admission_revision == call_context.admission_revision
            assert result.usage_receipt.usage_revision == call_context.usage_revision
            repeated = dispatch(call_context)
            assert repeated.execution_result.phase is ProviderExecutionPhase.REPLAYED
            assert repeated.text == ""  # Operational replay does not retain generated text.
            replay_count += 1
            if index % 4 == 0:
                restarted = ProviderExecutionGateway(coordinator=UsageCoordinator(
                    DurableUsageLedgerStore(ledger_path, clock=SystemClock())))
                result = dispatch(call_context, restarted)
                assert result.execution_result.phase is ProviderExecutionPhase.DENIED
                assert "reservation_replay_requires_reconciliation" in result.reason_codes
                restart_denials += 1
        assert callbacks == request_count

        stale_context = context("changed-state-control")
        owner.configure_caller_budget(scope, UsageVector.of(**limits))
        expect_stale(stale_context)
        assert callbacks == request_count
        fresh = owner.admission_snapshot(scope)
        recovered = dispatch(replace(stale_context,
            usage_revision=fresh.snapshot.usage_revision, admission_revision=fresh.admission_revision))
        assert recovered.execution_result.phase is ProviderExecutionPhase.SETTLED
        assert callbacks == request_count + 1
        exhausted = dispatch(context("exhausted-control"))
        assert exhausted.execution_result.phase is ProviderExecutionPhase.DENIED
        assert "limit_exhausted" in exhausted.reason_codes
        assert callbacks == request_count + 1

        document = store.read()
        records = tuple(document["reservations"].values())
        charges = {name: sum(record.get("charged_amounts", {}).get(name, 0) for record in records)
                   for name in limits}
        assert charges == {"requests": request_count + 1,
            "input_tokens": (request_count + 1) * 64, "output_tokens": (request_count + 1) * 16}
        remaining_holds = sum(record["state"] in ("held", "pending") for record in records)
        assert remaining_holds == 0
        ledger_bytes = ledger_path.stat().st_size
        revision = document["revision"]

    return {
        "schema": SCHEMA, "status": "completed", "clock": "SystemClock",
        "store": "isolated durable CAS ledger", "request_count": request_count,
        "succeeded": request_count + 1, "same_gateway_replays": replay_count,
        "restart_reinvocation_denials": restart_denials,
        "stale_mutation_denials": 1, "capacity_denials": 1,
        "legacy_observation_pin_denials": 1, "recovered_after_refresh": True,
        "physical_stub_callbacks": callbacks, "duplicate_callbacks": 0,
        "remaining_usage_holds": remaining_holds, "settled_estimated_units": charges,
        "ledger_revision": revision, "ledger_bytes": ledger_bytes,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "live_provider_calls": 0, "target_executions": 0, "solver_executions": 0,
        "claims_provider_token_savings": False, "cross_process_store_tested": False,
        "hard_process_containment_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    host = ResourceScheduler(ResourcePolicy(max_lanes=1))
    with public_replay_without_metadata():
        decision, lease = host.acquire(LaneResourceRequirements(
            lane_id="provider-admission-revision-benchmark", stage="analysis",
            memory_bytes=256 * 1024**2, process_slots=1,
        ))
    if lease is None:
        report = {"schema": SCHEMA, "status": "deferred", "reason": decision.reason,
            "live_host_preflight": "refused", "live_provider_calls": 0}
        exit_code = 2
    else:
        try:
            report = run_benchmark(args.requests)
            report.update(live_host_preflight="admitted", live_host_reserved_process_slots=1,
                live_host_reserved_memory_bytes=256 * 1024**2)
            exit_code = 0
        finally:
            host.release(lease)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
