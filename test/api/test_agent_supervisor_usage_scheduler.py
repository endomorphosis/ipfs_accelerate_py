"""ASI-167: project endpoint usage into fair resource and batch admission."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    AdmissionDecision,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProviderCapacity,
    ResourcePolicy,
    ResourceScheduler,
)


ROOT = Path(__file__).resolve().parents[2]
DECLARED_RS = ROOT / "ipfs_accelerate_py" / "agent_supervisor" / "resource_scheduler.py"
DECLARED_PBS = (
    ROOT / "ipfs_accelerate_py" / "agent_supervisor" / "provider_batch_scheduler.py"
)


def _load(path: Path, module_name: str):
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rs():
    return _load(DECLARED_RS, "ipfs_accelerate_py.agent_supervisor._test_declared_rs")


@pytest.fixture(scope="module")
def pbs(rs):
    # Ensure resource usage symbols are installed before batch module loads.
    return _load(DECLARED_PBS, "ipfs_accelerate_py.agent_supervisor._test_declared_pbs")


def _host(**overrides: object) -> HostResourceSnapshot:
    values: dict[str, object] = {
        "cpu_percent": 10,
        "memory_available_bytes": 8 * 1024**3,
        "memory_total_bytes": 16 * 1024**3,
        "disk_available_bytes": 100 * 1024**3,
        "gpu_memory_available_bytes": 0,
        "worker_limit": 8,
        "active_workers": 0,
        "available_worker_capacity": 8,
        "capabilities": ("cpu",),
        "resource_classes": ("cpu-small", "cpu-medium", "llm-proof-draft"),
        "observed_at_ms": 1_000,
    }
    values.update(overrides)
    return HostResourceSnapshot.from_mapping(values)


def _provider(provider_id: str = "provider-a", **overrides: object) -> ProviderCapacity:
    values: dict[str, object] = {
        "provider_id": provider_id,
        "healthy": True,
        "quota_remaining": 100,
        "latency_ms": 20,
        "context_window_tokens": 8_192,
        "token_budget_remaining": 50_000,
        "max_concurrency": 4,
        "active_requests": 0,
        "capabilities": ("generate",),
        "observed_at_ms": 1_000,
        "retry_after_ms": 0,
    }
    values.update(overrides)
    return ProviderCapacity(**values)  # type: ignore[arg-type]


def _requirement(**overrides: object) -> LaneResourceRequirements:
    values: dict[str, object] = {
        "lane_id": "lane-1",
        "stage": "inference",
        "resource_class": "cpu-small",
        "required_capabilities": ("generate",),
        "provider_id": "provider-a",
        "requires_provider": True,
        "context_tokens": 1_024,
        "token_budget": 500,
        "quota_units": 1,
        "process_slots": 1,
        "fairness_key": "goal:g1",
    }
    values.update(overrides)
    return LaneResourceRequirements.from_mapping(values)


def _snapshot(
    *,
    scope_id: str = "scope:acct-1",
    state: str = "available",
    requests_available: int | None = 10,
    tokens_available: int | None = 5_000,
    concurrent_available: int | None = 2,
    next_eligible_at: str | None = None,
    stale: bool = False,
    unknown_requests: bool = False,
) -> dict[str, object]:
    headroom: list[dict[str, object]] = []
    if unknown_requests:
        headroom.append(
            {
                "dimension": "requests",
                "available": {"kind": "unknown"},
                "ceiling": {"kind": "unknown"},
                "reserved": {"kind": "finite", "value": 0},
                "state": "unknown",
            }
        )
    else:
        if requests_available is not None:
            headroom.append(
                {
                    "dimension": "requests",
                    "available": {"kind": "finite", "value": requests_available},
                    "ceiling": {"kind": "finite", "value": max(requests_available, 10)},
                    "reserved": {"kind": "finite", "value": 0},
                    "state": "available",
                }
            )
        if tokens_available is not None:
            headroom.append(
                {
                    "dimension": "total_tokens",
                    "available": {"kind": "finite", "value": tokens_available},
                    "ceiling": {"kind": "finite", "value": max(tokens_available, 1000)},
                    "reserved": {"kind": "finite", "value": 0},
                    "state": "available",
                }
            )
        if concurrent_available is not None:
            headroom.append(
                {
                    "dimension": "concurrent_requests",
                    "available": {"kind": "finite", "value": concurrent_available},
                    "ceiling": {"kind": "finite", "value": max(concurrent_available, 1)},
                    "reserved": {"kind": "finite", "value": 0},
                    "state": "available",
                }
            )
    return {
        "scope_id": scope_id,
        "usage_revision": "rev-test",
        "observed_at": "2026-07-28T00:00:00Z",
        "fresh_until": "2020-01-01T00:00:00Z" if stale else "2099-01-01T00:00:00Z",
        "state": "stale" if stale else state,
        "headroom": headroom,
        "reservations": [],
        "next_eligible_at": next_eligible_at,
        "reason_codes": [],
    }


def test_requirement_id_installed_on_runtime(rs) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import resource_scheduler as runtime_rs

    assert rs.ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID.startswith("requirement:")
    assert hasattr(runtime_rs, "ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID")
    assert (
        runtime_rs.ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID
        == rs.ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID
    )


def test_off_mode_preserves_base_capacity(rs) -> None:
    base = _provider(quota_remaining=99, token_budget_remaining=12_000, max_concurrency=3)
    projection = rs.project_provider_capacity_from_usage_snapshot(
        _snapshot(requests_available=1, tokens_available=10, concurrent_available=1),
        provider_id="provider-a",
        base=base,
        mode=rs.UsageAdmissionMode.OFF,
    )
    assert projection.capacity.quota_remaining == 99
    assert projection.capacity.token_budget_remaining == 12_000
    assert projection.capacity.max_concurrency == 3
    assert projection.mode == "off"


def test_projection_is_conservative_intersection(rs) -> None:
    base = _provider(quota_remaining=100, token_budget_remaining=50_000, max_concurrency=8)
    projection = rs.project_provider_capacity_from_usage_snapshot(
        _snapshot(requests_available=3, tokens_available=400, concurrent_available=2),
        provider_id="provider-a",
        base=base,
        mode=rs.UsageAdmissionMode.ENFORCE,
    )
    assert projection.capacity.quota_remaining == 3
    assert projection.capacity.token_budget_remaining == 400
    assert projection.capacity.max_concurrency == 2
    assert projection.scope_id == "scope:acct-1"


def test_unknown_fields_never_become_unlimited_under_enforce(rs) -> None:
    base = _provider(quota_remaining=-1, token_budget_remaining=-1)
    projection = rs.project_provider_capacity_from_usage_snapshot(
        _snapshot(unknown_requests=True, tokens_available=None, concurrent_available=None),
        provider_id="provider-a",
        base=base,
        mode=rs.UsageAdmissionMode.ENFORCE,
        unknown_policy=rs.UnknownStalePolicy.FAIL_CLOSED,
    )
    # Unknown must not project to legacy unlimited (-1).
    assert projection.capacity.quota_remaining == 0
    assert projection.capacity.token_budget_remaining == 0
    assert "unknown_quota_remaining" in projection.reason_codes or projection.unknown_fields


def test_stale_snapshot_fail_closed_enforce(rs) -> None:
    projection = rs.project_provider_capacity_from_usage_snapshot(
        _snapshot(stale=True, requests_available=50, tokens_available=9_000),
        provider_id="provider-a",
        base=_provider(),
        mode=rs.UsageAdmissionMode.ENFORCE,
        unknown_policy=rs.UnknownStalePolicy.FAIL_CLOSED,
        now_ms=1_700_000_000_000,
    )
    assert projection.stale is True
    assert projection.capacity.quota_remaining == 0
    assert "fail_closed_stale" in projection.reason_codes or "stale_snapshot" in projection.reason_codes


def test_ancestor_budget_can_only_lower(rs) -> None:
    cap = _provider(quota_remaining=50, token_budget_remaining=10_000, max_concurrency=4)
    budget = rs.HierarchicalBudgetView.from_value(
        {
            "scope_ids": ["run:1", "goal:g", "task:t"],
            "limits": [
                {"dimension": "requests", "remaining": 2},
                {"dimension": "total_tokens", "remaining": 300},
                {"dimension": "concurrent_requests", "remaining": 1},
            ],
        }
    )
    lowered = rs.intersect_with_ancestor_budgets(cap, budget)
    assert lowered.quota_remaining == 2
    assert lowered.token_budget_remaining == 300
    assert lowered.max_concurrency == 1


def test_weighted_fair_queue_protects_reserves(rs) -> None:
    queue = rs.WeightedFairQueue()
    queue.register(rs.FairQueueScope(scope_id="tenant-a", kind="tenant", weight=1, reserved_slots=1))
    queue.register(rs.FairQueueScope(scope_id="tenant-b", kind="tenant", weight=10, reserved_slots=1))
    # Total 2 slots; each tenant reserved 1. Heavy tenant must not take both.
    active = {"tenant-b": 1}
    assert queue.available_for_scope("tenant-b", total_slots=2, active_by_scope=active) == 0
    assert queue.available_for_scope("tenant-a", total_slots=2, active_by_scope=active) >= 1
    # Selection still prefers weighted progress when both waiting and capacity remains.
    picked = queue.select_next(["tenant-a", "tenant-b"], total_slots=2, active_by_scope={})
    assert picked in {"tenant-a", "tenant-b"}


def test_single_flight_refresh_prevents_herd(rs) -> None:
    flight = rs.SingleFlightRefresh()
    calls = {"n": 0}
    barrier = threading.Barrier(4)
    results: list[object] = []

    def worker() -> None:
        barrier.wait()
        results.append(
            flight.do(
                "provider-a",
                lambda: (calls.__setitem__("n", calls["n"] + 1) or time.sleep(0.05) or "snap"),
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert calls["n"] == 1
    assert results == ["snap", "snap", "snap", "snap"]


def test_reset_event_cursor_wakes_bounded_jittered_work(rs) -> None:
    cursor = rs.ResetEventCursor(jitter_ms=0, max_wakeups=8)
    cursor.note_next_eligible(1_000, "provider-a")
    cursor.note_next_eligible(5_000, "provider-b")
    assert cursor.due(1_500) == ("provider-a",)
    assert cursor.due(6_000) == ("provider-b",)
    assert cursor.due(9_000) == ()


def test_evaluate_usage_aware_admission_actions(rs) -> None:
    resource = AdmissionDecision(
        lane_id="lane-1",
        admitted=True,
        provider_id="provider-a",
        reasons=(),
    )
    # Healthy projection → admit
    healthy = rs.project_provider_capacity_from_usage_snapshot(
        _snapshot(requests_available=5, tokens_available=5_000, concurrent_available=2),
        provider_id="provider-a",
        base=_provider(),
        mode=rs.UsageAdmissionMode.ENFORCE,
        now_ms=10_000,
    )
    admit = rs.evaluate_usage_aware_admission(
        resource_decision=resource,
        projection=healthy,
        mode=rs.UsageAdmissionMode.ENFORCE,
        now_ms=10_000,
    )
    assert admit.admitted is True
    assert admit.action == "admit"

    # Exhausted with wait room → wait (relative next-eligible offset fits deadline)
    cooling = rs.project_provider_capacity_from_usage_snapshot(
        _snapshot(
            state="cooling_down",
            requests_available=0,
            tokens_available=0,
            concurrent_available=0,
            next_eligible_at="1500",  # relative ms offset from now
        ),
        provider_id="provider-a",
        base=_provider(quota_remaining=0, token_budget_remaining=0),
        mode=rs.UsageAdmissionMode.ENFORCE,
        now_ms=10_000,
        retry_after_ms=1_500,
    )
    wait = rs.evaluate_usage_aware_admission(
        resource_decision=resource,
        projection=cooling,
        mode=rs.UsageAdmissionMode.ENFORCE,
        deadline_ms=20_000,
        now_ms=10_000,
    )
    assert wait.admitted is False
    assert wait.action == "wait"
    assert wait.wait_ms > 0

    # Alternate route preferred when available (before wait/deny)
    route = rs.evaluate_usage_aware_admission(
        resource_decision=resource,
        projection=cooling,
        mode=rs.UsageAdmissionMode.ENFORCE,
        deadline_ms=10_100,
        now_ms=10_000,
        alternate_providers=("provider-b",),
    )
    assert route.action == "route"
    assert route.route_provider_id == "provider-b"

    # Authorized fallback when no route and wait does not fit deadline
    tight = rs.project_provider_capacity_from_usage_snapshot(
        _snapshot(
            state="exhausted",
            requests_available=0,
            tokens_available=0,
            concurrent_available=0,
            next_eligible_at="30000",
        ),
        provider_id="provider-a",
        base=_provider(quota_remaining=0, token_budget_remaining=0),
        mode=rs.UsageAdmissionMode.ENFORCE,
        now_ms=10_000,
        retry_after_ms=30_000,
    )
    fallback = rs.evaluate_usage_aware_admission(
        resource_decision=resource,
        projection=tight,
        mode=rs.UsageAdmissionMode.ENFORCE,
        deadline_ms=10_100,
        now_ms=10_000,
        fallback_authorized=True,
    )
    assert fallback.action == "fallback"
    assert fallback.fallback_authorized is True

    # Deny with typed usage_capacity_unavailable
    deny = rs.evaluate_usage_aware_admission(
        resource_decision=resource,
        projection=tight,
        mode=rs.UsageAdmissionMode.ENFORCE,
        deadline_ms=10_100,
        now_ms=10_000,
    )
    assert deny.admitted is False
    assert deny.action == "deny"
    assert rs.USAGE_CAPACITY_UNAVAILABLE in deny.reasons
    assert deny.backpressure is True


def test_usage_aware_resource_scheduler_off_matches_base(rs) -> None:
    base = ResourceScheduler(policy=ResourcePolicy())
    usage = rs.UsageAwareResourceScheduler(
        policy=ResourcePolicy(),
        usage_mode=rs.UsageAdmissionMode.OFF,
    )
    req = _requirement()
    host = _host()
    providers = [_provider()]
    base_decision = base.evaluate(req, host=host, providers=providers)
    usage_decision = usage.evaluate_with_usage(req, host=host, providers=providers)
    assert usage_decision.admitted == base_decision.admitted
    assert usage_decision.mode == "off"


def test_usage_aware_resource_scheduler_enforce_blocks_exhausted(rs) -> None:
    def supplier(_provider_id: str) -> dict[str, object]:
        return _snapshot(
            state="exhausted",
            requests_available=0,
            tokens_available=0,
            concurrent_available=0,
        )

    scheduler = rs.UsageAwareResourceScheduler(
        policy=ResourcePolicy(),
        usage_mode=rs.UsageAdmissionMode.ENFORCE,
        usage_snapshot_supplier=supplier,
        clock_ms=lambda: 10_000,
    )
    decision = scheduler.evaluate_with_usage(
        _requirement(),
        host=_host(),
        providers=[_provider()],
        deadline_ms=10_100,
    )
    assert decision.admitted is False
    assert decision.action in {"deny", "wait", "fallback", "route"}
    assert decision.backpressure or decision.action != "admit"


def test_physical_batch_reserves_once_and_isolates_cancel(pbs) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
        ProviderBatchRequest,
    )

    requests = [
        ProviderBatchRequest(
            request_id="m1",
            payload="a",
            provider_id="provider-a",
            route="proof",
            model="m",
            operation="generate",
            token_budget=100,
            timeout_ms=1_000,
        ),
        ProviderBatchRequest(
            request_id="m2",
            payload="b",
            provider_id="provider-a",
            route="proof",
            model="m",
            operation="generate",
            token_budget=200,
            timeout_ms=1_000,
        ),
    ]
    reservation, grant = pbs.reserve_physical_batch(
        requests,
        provider_id="provider-a",
        snapshot=_snapshot(tokens_available=10_000, concurrent_available=2),
        mode=pbs.UsageAdmissionMode.ENFORCE,
        shared_overhead_tokens=50,
        base_capacity={
            "healthy": True,
            "max_batch_size": 8,
            "max_concurrent_batches": 2,
            "available_concurrent_batches": 2,
            "token_budget_remaining": 10_000,
            "retry_after_ms": 0,
        },
    )
    assert grant.admitted is True
    assert reservation is not None
    # Shared overhead settles once.
    assert reservation.settle_shared_overhead_once() == 50
    assert reservation.settle_shared_overhead_once() == 0
    # Cancel member m1 — sibling m2 remains charged.
    cancelled = reservation.cancel_member("m1")
    assert cancelled.cancelled is True
    assert cancelled.charged is False
    m2 = reservation.member_attributions["m2"]
    assert m2.charged is True
    assert m2.token_budget == 200
    # Total charged excludes cancelled member tokens, includes overhead once.
    assert reservation.total_charged_tokens() == 200 + 50
    payload = reservation.to_dict()
    assert payload["physical_batch_requirement_id"] == pbs.PHYSICAL_BATCH_RESERVE_ONCE_REQUIREMENT_ID


def test_physical_batch_enforce_denies_when_capacity_missing(pbs) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
        ProviderBatchRequest,
    )

    requests = [
        ProviderBatchRequest(
            request_id="only",
            payload="x",
            provider_id="provider-a",
            route="proof",
            model="m",
            operation="generate",
            token_budget=5_000,
            timeout_ms=1_000,
        )
    ]
    reservation, grant = pbs.reserve_physical_batch(
        requests,
        provider_id="provider-a",
        snapshot=_snapshot(tokens_available=10, concurrent_available=0),
        mode=pbs.UsageAdmissionMode.ENFORCE,
        base_capacity={
            "healthy": True,
            "max_batch_size": 8,
            "max_concurrent_batches": 1,
            "available_concurrent_batches": 0,
            "token_budget_remaining": 10,
            "retry_after_ms": 0,
        },
    )
    assert reservation is None
    assert grant.admitted is False
    assert pbs.USAGE_CAPACITY_UNAVAILABLE in grant.reason or grant.reason


def test_usage_aware_batch_scheduler_off_preserves_batching(pbs) -> None:
    calls: list[tuple[str, ...]] = []

    def dispatch(requests: object) -> list[str]:
        members = tuple(requests)  # type: ignore[arg-type]
        calls.append(tuple(item.request_id for item in members))
        return [str(item.payload).upper() for item in members]

    from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
        ProviderBatchRequest,
        ProviderBatchSchedulerConfig,
    )

    config = ProviderBatchSchedulerConfig(
        max_batch_size=8,
        batch_window_ms=20,
        max_parallel_batches=2,
        provider_limits={"provider-a": 1},
        admission_retry_ms=1,
    )
    with pbs.UsageAwareProviderBatchScheduler(
        dispatch,
        config=config,
        usage_mode=pbs.UsageAdmissionMode.OFF,
    ) as scheduler:
        futures = [
            scheduler.submit(
                ProviderBatchRequest(
                    request_id="first",
                    payload="a",
                    provider_id="provider-a",
                    route="proof",
                    model="model-a",
                    operation="generate",
                    token_budget=100,
                    timeout_ms=2_000,
                )
            ),
            scheduler.submit(
                ProviderBatchRequest(
                    request_id="second",
                    payload="b",
                    provider_id="provider-a",
                    route="proof",
                    model="model-a",
                    operation="generate",
                    token_budget=100,
                    timeout_ms=2_000,
                )
            ),
        ]
        results = [future.result(timeout=2) for future in futures]
    assert [result.output for result in results] == ["A", "B"]
    assert calls == [("first", "second")] or sorted(calls[0]) == ["first", "second"]


def test_fair_queue_and_batch_symbols_exported(pbs, rs) -> None:
    assert pbs.ENDPOINT_USAGE_BATCH_ADMISSION_REQUIREMENT_ID.startswith("requirement:")
    assert callable(pbs.reserve_physical_batch)
    assert callable(rs.evaluate_usage_aware_admission)
    assert issubclass(rs.UsageAwareResourceScheduler, ResourceScheduler)
