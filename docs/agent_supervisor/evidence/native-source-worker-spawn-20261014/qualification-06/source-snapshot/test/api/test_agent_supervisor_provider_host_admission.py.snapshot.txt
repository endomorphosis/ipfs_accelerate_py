"""Default-on provider host admission against an isolated real native owner.

The shared ledger is real; host pressure and provider callbacks are authored
controls. These tests do not run a target repository or a model provider.
"""
import builtins
from concurrent.futures import CancelledError
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.provider_host_admission import (
    ProviderHostAdmission, ProviderHostBudget,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchAdmissionGrant, ProviderBatchCapacity, ProviderBatchRequest,
    ProviderBatchSchedulerConfig,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot, ResourcePolicy, ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.proof.leanstral_goal_development import (
    build_leanstral_goal_development_batch_dispatch, create_leanstral_goal_development_batch_scheduler,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLane, ResourceSchedulerConfig,
)


@pytest.fixture
def owners(tmp_path):
    pressure = {"stall": 0}
    native = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "authored-provider-host.json",
        proof_resource_sampler=lambda: ProofHostResources(8, 8192, 8192,
            memory_stall_percent=pressure["stall"]),
        total_cpu_slots=4, total_memory_mb=4096, total_child_process_slots=4,
        lane_reservations={"validation": 1}, poll_interval_seconds=.005,
        proof_backoff_seconds=.001))

    def sample(*args, **kwargs):
        return HostResourceSnapshot(memory_total_bytes=8*1024**3,
            memory_available_bytes=8*1024**3, disk_total_bytes=16*1024**3,
            disk_available_bytes=16*1024**3, worker_limit=4, available_worker_capacity=4)

    local = ResourceScheduler(ResourcePolicy(max_lanes=4, require_provider_telemetry=False),
        host_sampler=sample)
    admission = ProviderHostAdmission(local_scheduler=local, native_scheduler=native, path=tmp_path)
    yield admission, native, local, pressure
    assert local.active_leases == ()
    snapshot = native.snapshot()
    assert snapshot["active_lease_count"] == snapshot["waiting_request_count"] == 0


def _request(name="one", **kwargs):
    return ProviderBatchRequest(request_id=name, payload="authored:" + name,
        provider_id="authored-provider", timeout_ms=3000, **kwargs)


def _grant(admission, request=None):
    request = request or _request()
    return admission(request.batch_key, (request,), ProviderBatchCapacity(provider_id=request.provider_id))


def _factory(host_owner, callback, **kwargs):
    return create_leanstral_goal_development_batch_scheduler(
        scheduler_config=ProviderBatchSchedulerConfig(batch_window_ms=0,
            max_parallel_batches=2, admission_retry_ms=5, fallback_on_dispatch_error=False),
        llm_generate_batch=callback, host_admission=host_owner, **kwargs)


@pytest.mark.parametrize("values", [dict(memory_mb=0), dict(cpu_slots=True),
    dict(process_slots=2), dict(max_parallel=257), dict(memory_mb=65537)])
def test_host_budget_requires_finite_positive_cpu_envelope(values):
    with pytest.raises(ValueError):
        ProviderHostBudget(**values)


def test_default_factory_admits_shared_host_before_callback_and_releases(owners):
    admission, native, local, _ = owners
    seen = []

    def callback(prompts, **kwargs):
        snapshot = native.snapshot()
        seen.append((snapshot["active_root_lease_count"], len(local.active_leases)))
        return ["authored reply" for _ in prompts]

    with _factory(admission, callback) as scheduler:
        result = scheduler.execute(_request())
    assert result.successful and result.output == "authored reply"
    assert seen == [(1, 1)]


def test_factory_constructs_default_host_owner_without_explicit_admission(owners, monkeypatch):
    _, native, local, _ = owners
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as native_module

    monkeypatch.setattr(native_module, "get_global_resource_scheduler", lambda: native)
    original_init = ResourceScheduler.__init__
    created = []

    def controlled_init(self, policy=None, **kwargs):
        kwargs["host_sampler"] = local.host_sampler
        original_init(self, policy, **kwargs)
        created.append(self)

    monkeypatch.setattr(ResourceScheduler, "__init__", controlled_init)

    def callback(prompts, **kwargs):
        assert native.snapshot()["active_lease_count"] == 1
        assert len(created) == 1 and len(created[0].active_leases) == 1
        return ["authored" for _ in prompts]

    with create_leanstral_goal_development_batch_scheduler(llm_generate_batch=callback) as scheduler:
        assert scheduler.execute(_request()).successful
    assert created[0].active_leases == ()


def test_native_pressure_refuses_nonblocking_then_recovers(owners):
    admission, native, local, pressure = owners
    pressure["stall"] = 90
    refused = _grant(admission)
    assert not refused.admitted and refused.retryable
    assert refused.reason == "native_host_pressure_or_capacity"
    assert local.active_leases == ()
    assert native.snapshot()["waiting_request_count"] == 0
    pressure["stall"] = 0
    time.sleep(.01)
    accepted = _grant(admission)
    assert accepted.admitted
    accepted.release_once()
    accepted.release_once()


def test_delegated_parent_reserves_one_root_and_one_child(owners):
    admission, native, local, _ = owners
    parent = native.acquire(ResourceLane.ORCHESTRATION, cpu_slots=2, memory_mb=512,
        child_process_slots=2, timeout=0)
    try:
        child_admission = ProviderHostAdmission(local_scheduler=local,
            native_scheduler=native, parent_lease=parent)
        grant = _grant(child_admission)
        assert grant.admitted
        try:
            snapshot = native.snapshot()
            assert snapshot["active_root_lease_count"] == 1
            assert snapshot["active_child_lease_count"] == 1
            assert grant.lease.parent_lease_id == parent.lease_id
        finally:
            grant.release_once()
        assert native.snapshot()["active_root_lease_count"] == 1
    finally:
        parent.release()


@pytest.mark.parametrize("raise_error", [False, True])
def test_custom_admission_refusal_or_error_releases_both_owners(owners, raise_error):
    admission, native, local, _ = owners

    def custom(*args):
        assert native.snapshot()["active_lease_count"] == len(local.active_leases) == 1
        if raise_error:
            raise RuntimeError("authored custom refusal")
        return ProviderBatchAdmissionGrant(admitted=False, reason="authored refusal", retryable=False)

    combined = admission.compose(custom)
    if raise_error:
        with pytest.raises(RuntimeError, match="authored custom refusal"):
            _grant(combined)
    else:
        assert not _grant(combined).admitted
    assert native.snapshot()["active_lease_count"] == 0


def test_custom_lifecycle_hooks_remain_inside_default_host_grant(owners):
    admission, native, local, _ = owners
    events = []

    def observe(name):
        assert native.snapshot()["active_lease_count"] == 1
        events.append(name)

    def custom(*args):
        return ProviderBatchAdmissionGrant(admitted=True,
            before_dispatch=lambda requests: observe("before"),
            after_dispatch=lambda requests, results, error: observe("after"),
            release=lambda: observe("custom_release"))

    def callback(prompts, **kwargs):
        observe("callback")
        return ["authored" for _ in prompts]

    with _factory(admission, callback, admission=custom) as scheduler:
        assert scheduler.execute(_request()).successful
    assert events == ["before", "callback", "after", "custom_release"]


def test_cancelled_subscriber_retains_native_lease_until_callback_drains(owners):
    admission, native, local, _ = owners
    entered, finish = threading.Event(), threading.Event()

    def callback(prompts, **kwargs):
        entered.set()
        assert finish.wait(5)
        return ["authored" for _ in prompts]

    scheduler = _factory(admission, callback)
    try:
        future = scheduler.submit(_request())
        assert entered.wait(5)
        assert scheduler.cancel("one")
        with pytest.raises(CancelledError):
            future.result(timeout=5)
        assert future.cancelled()
        assert native.snapshot()["active_lease_count"] == len(local.active_leases) == 1
    finally:
        finish.set()
        scheduler.shutdown(wait=True)


def test_flush_and_shutdown_wait_until_admission_release_finishes(owners):
    admission, native, local, _ = owners
    release_entered, release_allowed = threading.Event(), threading.Event()
    flush_started, shutdown_started = threading.Event(), threading.Event()
    flush_done, shutdown_done = threading.Event(), threading.Event()
    flushed = []

    def release():
        release_entered.set()
        assert release_allowed.wait(5)

    scheduler = _factory(admission,
        lambda prompts, **kwargs: ["authored" for _ in prompts],
        admission=lambda *args: ProviderBatchAdmissionGrant(admitted=True, release=release))

    def flush():
        flush_started.set()
        flushed.append(scheduler.flush(timeout=5))
        flush_done.set()

    def shutdown():
        shutdown_started.set()
        scheduler.shutdown(wait=True)
        shutdown_done.set()

    threads = [threading.Thread(target=flush), threading.Thread(target=shutdown)]
    try:
        assert scheduler.execute(_request()).successful
        assert release_entered.wait(5)
        for thread in threads:
            thread.start()
        assert flush_started.wait(5) and shutdown_started.wait(5)
        assert not flush_done.wait(.05)
        assert not shutdown_done.wait(.05)
        assert native.snapshot()["active_lease_count"] == len(local.active_leases) == 1
    finally:
        release_allowed.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
        scheduler.shutdown(wait=True)
    assert flush_done.is_set() and shutdown_done.is_set()
    assert flushed == [True]
    assert local.active_leases == ()


@pytest.mark.parametrize("revoke", ["cancel", "release"])
def test_native_revocation_before_dispatch_prevents_callback_and_custom_hook(owners, revoke):
    admission, native, _, _ = owners
    calls = []

    def custom(*args):
        lease_row, = native.active_leases()
        if revoke == "cancel":
            native.cancel(lease_row["lease_id"])
        else:
            native.release(lease_row["lease_id"])
        return ProviderBatchAdmissionGrant(admitted=True,
            before_dispatch=lambda _: calls.append("custom_hook"))

    with _factory(admission, lambda *args, **kwargs: calls.append("callback"),
            admission=custom) as scheduler:
        result = scheduler.execute(_request())
    assert not result.successful
    assert calls == []


def test_trusted_process_envelope_is_reflected_in_local_and_native_reservations(owners):
    _, native, local, _ = owners
    admission = ProviderHostAdmission(budget=ProviderHostBudget(cpu_slots=2, process_slots=2),
        local_scheduler=local, native_scheduler=native)
    grant = _grant(admission)
    assert grant.admitted
    try:
        assert local.active_leases[0].requirement.process_slots == 2
        assert grant.lease.child_process_slots == 2
    finally:
        grant.release_once()


def test_callback_failure_releases_host_lease(owners):
    admission, native, _, _ = owners

    def fail(*args, **kwargs):
        assert native.snapshot()["active_lease_count"] == 1
        raise RuntimeError("authored provider failure")

    with _factory(admission, fail) as scheduler:
        assert not scheduler.execute(_request()).successful


def test_missing_native_dependency_refuses_without_local_reservation(owners, monkeypatch):
    admission, _, local, _ = owners
    original = builtins.__import__

    def import_control(name, *args, **kwargs):
        if name == "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler":
            raise ModuleNotFoundError("authored absent native owner")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_control)
    refused = _grant(admission)
    assert not refused.admitted and not refused.retryable
    assert refused.reason == "native_host_admission_unavailable"
    assert local.active_leases == ()


def test_disabled_native_safety_and_oversize_demands_refuse(owners):
    admission, native, _, _ = owners
    native.config.proof_safety_enabled = False
    refused = _grant(admission)
    assert not refused.admitted and refused.reason == "native_host_safety_required"
    native.config.proof_safety_enabled = True
    refused = _grant(admission, _request(provenance={"memory_bytes": 512*1024**2}))
    assert not refused.admitted and refused.reason == "native_host_demand_exceeds_budget"


def test_explicit_host_opt_out_preserves_custom_admission_without_native_use(owners):
    _, native, _, _ = owners
    calls = []

    def custom(*args):
        calls.append("custom")
        assert native.snapshot()["active_lease_count"] == 0
        return True

    with create_leanstral_goal_development_batch_scheduler(
        host_admission_enabled=False, admission=custom,
        llm_generate_batch=lambda prompts, **kwargs: ["authored" for _ in prompts],
    ) as scheduler:
        assert scheduler.execute(_request()).successful
    assert calls == ["custom"]


@pytest.mark.parametrize("slots", [1, 2])
def test_default_router_scalar_pool_is_bounded_by_admitted_host_slots(owners, monkeypatch, slots):
    """Run the real router batch pool with authored scalar callbacks only."""
    from ipfs_accelerate_py import llm_router

    _, native, local, _ = owners
    admission = ProviderHostAdmission(budget=ProviderHostBudget(cpu_slots=slots,
        process_slots=slots), local_scheduler=local, native_scheduler=native)
    monkeypatch.setenv("IPFS_ACCELERATE_LLM_ROUTER_BATCH_WORKERS", "128")
    monkeypatch.setattr(llm_router, "get_default_router_deps", lambda: object())
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    lock = threading.Lock()
    counts = {"active": 0, "peak": 0, "calls": 0}
    limits = []
    original_worker_count = llm_router._batch_worker_count

    def worker_count(**kwargs):
        limits.append(kwargs["max_workers"])
        return original_worker_count(**kwargs)

    def scalar(prompt, **kwargs):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            counts["calls"] += 1
        try:
            assert native.snapshot()["active_root_lease_count"] == 1
            time.sleep(.01)
            return "authored reply:" + prompt
        finally:
            with lock:
                counts["active"] -= 1

    monkeypatch.setattr(llm_router, "_batch_worker_count", worker_count)
    monkeypatch.setattr(llm_router, "generate_text", scalar)
    with create_leanstral_goal_development_batch_scheduler(host_admission=admission,
        scheduler_config=ProviderBatchSchedulerConfig(max_parallel_batches=1,
            max_batch_size=4, min_batch_size=4, batch_window_ms=30,
            fallback_on_dispatch_error=False)) as scheduler:
        futures = [scheduler.submit(_request(str(index))) for index in range(4)]
        results = [future.result(timeout=5) for future in futures]
    assert all(result.successful for result in results)
    assert counts["calls"] == 4 and counts["active"] == 0
    assert 1 <= counts["peak"] <= slots
    assert limits and set(limits) == {slots}


def test_router_worker_setting_does_not_change_custom_callback_contract():
    seen = []

    def callback(prompts, **kwargs):
        seen.append(kwargs)
        return ["authored" for _ in prompts]

    dispatch = build_leanstral_goal_development_batch_dispatch(
        llm_generate_batch=callback, router_max_workers=1)
    assert dispatch((_request(),)) == ("authored",)
    assert "max_workers" not in seen[0]
