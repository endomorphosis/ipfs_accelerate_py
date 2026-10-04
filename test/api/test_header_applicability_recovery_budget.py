"""Bounded replay recovery, with an isolated native reservation experiment.

Fast controls use a simulated clock. The long control uses real elapsed time,
native resource leases, captured authored source and Z3. Its telemetry is
authored; it neither induces nor establishes recovery from host memory pressure.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from test.api.test_header_intent_applicability import case  # noqa: F401
from test.integration.test_admitted_benchmark_runtime import admitted  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import header_intent_applicability as owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


PROFILE_ENV = "IPFS_DATASETS_PROOF_RESOURCE_PROFILE"


@pytest.fixture
def clock(monkeypatch):
    value = [1000.]
    # Do not change the shared time module used by leases or other threads.
    monkeypatch.setattr(owner, "time", SimpleNamespace(monotonic=lambda: value[0]))
    return value


def test_extended_replay_requires_explicit_scope_not_environment_alone(monkeypatch, clock):
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert owner.applicability_replay_timeout() == 45.
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    assert owner.applicability_replay_timeout() == 45.
    with owner.applicability_budget(120.):
        assert owner.applicability_replay_timeout() == 45.
    with owner.local_benchmark_applicability_budget(deadline_monotonic=1100.):
        assert owner.applicability_replay_timeout() == 120.
        assert owner.applicability_replay_timeout(7.) == 7.
    assert owner.applicability_replay_timeout() == 45.


@pytest.mark.parametrize("profile", [None, "", "local-benchmark@2"])
def test_local_scope_rejects_missing_or_unknown_profile(monkeypatch, clock, profile):
    if profile is None:
        monkeypatch.delenv(PROFILE_ENV, raising=False)
    else:
        monkeypatch.setenv(PROFILE_ENV, profile)
    with pytest.raises(ValueError):
        with owner.local_benchmark_applicability_budget(deadline_monotonic=1100.):
            pytest.fail("unselected local profile accepted")
    assert owner.applicability_replay_timeout() == 45.


@pytest.mark.parametrize("deadline", [True, "1100", float("nan"), float("inf"), 999., 1000., 1901.])
def test_local_scope_rejects_invalid_absolute_deadlines(monkeypatch, clock, deadline):
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    with pytest.raises((ValueError, TimeoutError)):
        with owner.local_benchmark_applicability_budget(deadline_monotonic=deadline):
            pytest.fail("invalid deadline accepted")


@pytest.mark.parametrize("extended,limit", [
    (False, 45.001), (True, 120.001), (False, True), (True, True),
    (False, 0), (True, -1), (False, float("nan")), (True, float("inf")), (True, "20"),
])
def test_replay_rejects_invalid_or_oversized_explicit_limit(monkeypatch, clock, extended, limit):
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    scope = (owner.local_benchmark_applicability_budget(deadline_monotonic=1120.)
             if extended else owner.applicability_budget(120.))
    with scope, pytest.raises(ValueError):
        owner.applicability_replay_timeout(limit)


def test_nested_scope_cannot_renew_expired_parent_and_restores_scope(monkeypatch, clock):
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    with owner.applicability_budget(5.):
        with owner.local_benchmark_applicability_budget(deadline_monotonic=1120.):
            clock[0] = 1004.
            with owner.applicability_budget(120.):
                owner.require_applicability_budget()
                clock[0] = 1006.
                with pytest.raises(TimeoutError):
                    owner.require_applicability_budget()
    owner.require_applicability_budget()
    assert owner.applicability_replay_timeout() == 45.


def test_captured_scope_crosses_thread_without_extending_deadline(monkeypatch, clock):
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    with owner.local_benchmark_applicability_budget(deadline_monotonic=1020.):
        scope = owner.capture_applicability_budget()
    assert scope is not None
    clock[0] = 1010.
    observed, errors = [], []

    def worker():
        try:
            assert owner.applicability_replay_timeout() == 45.
            with owner.resume_applicability_budget(scope, deadline_monotonic=1015.):
                observed.append(owner.applicability_replay_timeout())
                clock[0] = 1016.
                with pytest.raises(TimeoutError):
                    owner.require_applicability_budget()
            observed.append(owner.applicability_replay_timeout())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2.)
    assert not thread.is_alive() and not errors
    assert observed == [120., 45.]
    clock[0] = 1021.
    with pytest.raises(TimeoutError):
        with owner.resume_applicability_budget(scope):
            owner.require_applicability_budget()
    assert owner.applicability_replay_timeout() == 45.


@pytest.mark.parametrize("extended,explicit,parent,expected", [
    (False, None, 120., 45.), (False, 7., 120., 7.),
    (True, None, 120., 120.), (True, 7., 120., 7.), (True, None, 4., 4.),
])
def test_checked_replay_passes_selected_and_remaining_budget_to_native_owner(
        case, monkeypatch, clock, extended, explicit, parent, expected):
    from ipfs_datasets_py.logic.software_contracts import codebase_header_context as captured
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    observed = []

    class BoundaryReached(Exception):
        pass

    @contextmanager
    def capture(**kwargs):
        observed.append(kwargs["timeout_seconds"])
        raise BoundaryReached
        yield  # pragma: no cover

    monkeypatch.setattr(captured, "_operation", capture)
    scope = (owner.local_benchmark_applicability_budget(deadline_monotonic=1000. + parent)
             if extended else owner.applicability_budget(parent))
    kwargs = {} if explicit is None else {"timeout_seconds": explicit}
    with scope, pytest.raises(BoundaryReached):
        owner.checked_applicability(case.selection, nomination=case.nomination,
            manifest=case.manifest, **kwargs)
    assert observed == [expected]


@pytest.mark.parametrize("extended,expected", [(False, 45.), (True, 55.)])
def test_materialization_scope_keeps_selected_cap_and_parent_deadline(case, monkeypatch, clock, extended, expected):
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    observed = []

    def transaction(**kwargs):
        # This boundary control does not claim actual materialization or proof.
        observed.append(owner._DEADLINE.get() - clock[0])
        return {"boundary_reached": True}

    monkeypatch.setattr(local, "_materialize_local_transaction", transaction)
    admission = {"manifest": case.manifest}
    scope = (owner.local_benchmark_applicability_budget(deadline_monotonic=1055.)
             if extended else owner.applicability_budget(55.))
    with scope:
        assert local.materialize_local_benchmark_plan(admission=admission, intent=object()) == {"boundary_reached": True}
    assert observed == [expected]


def _local_scheduler(case, monkeypatch, *, memory_stall=0., available_memory=16384):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
    from ipfs_datasets_py.logic.software_contracts import codebase_resources
    path = case.root / "replay-resources.json"
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    monkeypatch.setenv("IPFS_DATASETS_RESOURCE_SCHEDULER_PATH", str(path))
    config = resources.ResourceSchedulerConfig.for_proof_host(
        state_path=path, proof_resource_profile="local-benchmark@1",
        proof_resource_sampler=lambda: ProofHostResources(8, 16384, available_memory,
            memory_stall_percent=memory_stall),
        proof_memory_stall_percent=10., proof_recovery_enabled=True, proof_recovery_grants=1,
        lane_reservations={}, auto_renew_leases=False, poll_interval_seconds=.05)
    scheduler = resources.GlobalResourceScheduler(config)
    monkeypatch.setattr(codebase_resources, "get_global_resource_scheduler", lambda: scheduler)
    return scheduler, resources


@pytest.mark.parametrize("reason", ["pressure", "headroom", "cancelled"])
def test_extended_replay_still_refuses_pressure_headroom_and_cancellation(case, monkeypatch, reason):
    from ipfs_datasets_py.logic.security_ir import bounded_header_checker
    scheduler, resources = _local_scheduler(case, monkeypatch,
        memory_stall=10.1 if reason == "pressure" else 0.,
        available_memory=256 if reason == "headroom" else 16384)
    monkeypatch.setattr(bounded_header_checker, "bounded_header_runner",
        lambda *args, **kwargs: pytest.fail("refused admission launched a solver"))
    cancellation = threading.Event()
    if reason == "cancelled":
        cancellation.set()
    expected = resources.LeaseCancelledError if reason == "cancelled" else resources.LeaseTimeoutError
    with owner.local_benchmark_applicability_budget(deadline_monotonic=time.monotonic() + 2.):
        with pytest.raises(expected) as caught:
            owner.checked_applicability(case.selection, nomination=case.nomination,
                manifest=case.manifest, scheduler=scheduler, cancel_event=cancellation,
                timeout_seconds=.5)
    if reason != "cancelled":
        observation = caught.value.admission_observation
        assert observation is not None and observation["terminal"] == "timeout"
        assert observation["last_sample"]["reason"] == (
            "proof_memory_stall" if reason == "pressure" else "proof_memory_headroom")
        assert observation["last_sample"]["thresholds"]["memory_stall_percent"] == 10.
    snapshot = scheduler.snapshot()
    assert snapshot["active_lease_count"] == snapshot["waiting_request_count"] == 0


@pytest.fixture
def runtime_scope_observations(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as native
    create = native.AdmittedBenchmarkRuntime.create.__func__
    verify = native.verify_local_benchmark_admission
    observations = []

    def observed_verify(*args, **kwargs):
        scope = owner.capture_applicability_budget()
        observations.append(dict(cap=owner.applicability_replay_timeout(), scope=scope))
        return verify(*args, **kwargs)

    def scoped_create(cls, *args, **kwargs):
        # Enter before the real constructor's first admission replay.
        with owner.local_benchmark_applicability_budget(deadline_monotonic=time.monotonic() + 150.):
            return create(cls, *args, **kwargs)

    monkeypatch.setattr(native, "verify_local_benchmark_admission", observed_verify)
    monkeypatch.setattr(native.AdmittedBenchmarkRuntime, "create", classmethod(scoped_create))
    return observations


@pytest.mark.parametrize("admitted", ["extended-startup"], indirect=True)
def test_native_runtime_keeps_scope_and_stop(runtime_scope_observations, admitted, monkeypatch, record_property):
    """Actual native startup/custody; the fixture's task is model-free."""
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import LifecycleOrchestrator
    runtime, _native_owner, _prepared = admitted
    assert runtime_scope_observations
    assert all(row["cap"] == 120. and row["scope"] is not None for row in runtime_scope_observations)
    assert runtime._replay_work_scope is not None
    assert owner.applicability_replay_timeout() == 45.  # Constructor caller scope has exited.
    original_verify = runtime._verify
    observations = []
    lifecycle_deadlines = []
    original_start_new = LifecycleOrchestrator._start_new

    def observed_start_new(orchestrator, state, profile, deadline):
        lifecycle_deadlines.append((deadline, runtime._replay_start_deadline))
        return original_start_new(orchestrator, state, profile, deadline)

    monkeypatch.setattr(LifecycleOrchestrator, "_start_new", observed_start_new)

    def observed_verify(*args, **kwargs):
        if not observations:
            # Actual elapsed pre-request verification must reduce START's
            # effective bound instead of receiving another full 120 seconds.
            time.sleep(2.1)
        scope = owner.capture_applicability_budget()
        observations.append(dict(cap=owner.applicability_replay_timeout(), scope=scope,
            start_deadline=runtime._replay_start_deadline,
            bootstrap_thread=threading.current_thread() is runtime._bootstrap_thread))
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(runtime, "_verify", observed_verify)
    started = runtime.start()
    assert started.succeeded
    assert 2000 <= started.bounds.timeout_ms <= 118000
    assert observations and any(row["bootstrap_thread"] for row in observations)
    assert all(row["cap"] == 120. and row["scope"] is not None for row in observations)
    starting = [row for row in observations if row["start_deadline"] is not None]
    assert starting and any(row["bootstrap_thread"] for row in starting)
    assert all(row["scope"].deadline_monotonic <= row["start_deadline"] for row in starting)
    assert lifecycle_deadlines and all(deadline <= original for deadline, original in lifecycle_deadlines)
    assert runtime._replay_start_deadline is None
    startup = runtime.startup_diagnostics()
    assert {row["phase"] for row in startup["observations"]} == {
        "control_validation", "launch_validation", "bootstrap_validation"}
    assert all(row["status"] == "completed" for row in startup["observations"])
    assert runtime.request(Operation.STOP).bounds.timeout_ms == 20_000
    with owner.applicability_budget(.001):
        time.sleep(.02)
        with pytest.raises(TimeoutError):
            owner.require_applicability_budget()
        stopped = runtime.stop()
    assert stopped.succeeded
    assert not runtime.process.snapshot(runtime.profile).members
    assert all(child.poll() is not None for child in runtime._children)
    record_property("native_runtime_replay_scope", json.dumps(dict(
        actual_native_start=True, actual_native_bootstrap=True, actual_native_stop=True,
        constructor_scope_captured_before_verification=True,
        bootstrap_thread_scope_observed=True, default_replay_cap_outside_scope=45.,
        selected_replay_cap=120., stop_timeout_ms=20_000,
        actual_pre_request_delay_seconds=2.1, start_request_timeout_ms=started.bounds.timeout_ms,
        lifecycle_uses_original_start_deadline=True,
        expired_caller_scope_did_not_block_stop=True, all_child_processes_reaped=True,
        header_solver_run_in_this_control=False, model_or_provider_calls=0), sort_keys=True))


@pytest.mark.parametrize("remaining", [1.999, 2., 7.])
def test_start_effective_bound_refuses_or_uses_original_remaining_time(monkeypatch, clock, remaining):
    """Simulated clock control of signed versus effective request bounds."""
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as native
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    with owner.local_benchmark_applicability_budget(deadline_monotonic=1150.):
        scope = owner.capture_applicability_budget()
    runtime = native.AdmittedBenchmarkRuntime.__new__(native.AdmittedBenchmarkRuntime)
    runtime.timeout_ms = 20_000
    runtime.start_timeout_ms = 120_000
    runtime.manifest = {"start_timeout_ms": 120_000}
    runtime._replay_work_scope = scope
    runtime._replay_lifetime_deadline = 1200.
    runtime._replay_start_deadline = clock[0] + remaining
    assert runtime._configured_operation_timeout_ms(Operation.START) == 120_000
    assert runtime._operation_timeout_ms(Operation.STOP) == 20_000
    if remaining < 2.:
        with pytest.raises(TimeoutError):
            runtime._operation_timeout_ms(Operation.START)
    else:
        assert runtime._operation_timeout_ms(Operation.START) == int(remaining * 1000)


@pytest.mark.parametrize("remaining", [None, -1., 1.999])
def test_refused_start_handler_does_not_invoke_lifecycle_owner(monkeypatch, clock, remaining):
    """Boundary-only control: no saga, process or proof is constructed."""
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as native
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
    monkeypatch.setenv(PROFILE_ENV, "local-benchmark@1")
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    with owner.local_benchmark_applicability_budget(deadline_monotonic=1150.):
        scope = owner.capture_applicability_budget()
    runtime = native.AdmittedBenchmarkRuntime.__new__(native.AdmittedBenchmarkRuntime)
    runtime.timeout_ms, runtime.start_timeout_ms = 20_000, 120_000
    runtime.manifest = {"start_timeout_ms": 120_000}
    runtime._replay_work_scope = scope
    runtime._replay_lifetime_deadline = 1200.
    runtime._replay_start_deadline = None if remaining is None else clock[0] + remaining
    invoked = []
    runtime.orchestrator = lambda request: invoked.append(request)
    with pytest.raises(TimeoutError):
        runtime._bounded_lifecycle_response(SimpleNamespace(operation=Operation.START))
    assert invoked == []


def test_real_captured_replay_recovers_after_45_seconds_without_renewal(case, monkeypatch, record_property):
    """Actual wait and seven bounded Z3 processes; no induced host pressure."""
    from ipfs_datasets_py.logic.security_ir import bounded_header_checker
    scheduler, resources = _local_scheduler(case, monkeypatch)
    process_observations, admission_waits, child_admission_waits, release_errors = [], [], [], []

    class ObservedRunner(bounded_header_checker.BoundedToolRunner):
        def run(self, request, **kwargs):
            invoked = time.monotonic()
            result = super().run(request, **kwargs)
            process_observations.append(dict(
                invoked=invoked, timeout_seconds=request.limits.timeout_seconds,
                elapsed_seconds=result.elapsed_seconds, pid=result.pid,
                returncode=result.returncode, workspace_cleaned=result.workspace_cleaned))
            return result

    monkeypatch.setattr(bounded_header_checker, "BoundedToolRunner", ObservedRunner)
    acquire = scheduler.acquire

    def observe_acquire(**kwargs):
        target = child_admission_waits if kwargs.get("parent_lease") is not None else admission_waits
        target.append(kwargs["timeout"])
        return acquire(**kwargs)

    blocker = acquire(lane=resources.ResourceLane.SNAPSHOT_EVALUATION,
        cpu_slots=scheduler.config.total_cpu_slots, memory_mb=512,
        child_process_slots=1, timeout=1.)
    monkeypatch.setattr(scheduler, "acquire", observe_acquire)
    started = time.monotonic()

    def release():
        try:
            blocker.release()
        except Exception as error:
            release_errors.append(type(error).__name__)

    timer = threading.Timer(46., release)
    timer.start()
    try:
        with owner.local_benchmark_applicability_budget(deadline_monotonic=started + 55.):
            evidence = owner.checked_applicability(case.selection, nomination=case.nomination,
                manifest=case.manifest, scheduler=scheduler)
            recovered_at = time.monotonic() - started
            assert 45. < recovered_at < 55.
            assert evidence["solver_calls"] == 6 and evidence["exact_candidate_replayed"]
            assert all(evidence[key] is False for key in owner.FALSE)
            assert {row["solver_answer"] for row in evidence["checked_obligations"]} == {"sat", "unsat"}
            assert len(process_observations) == 7
            assert process_observations[0]["invoked"] - started > 45.
            assert all(0 < row["timeout_seconds"] <= 5. for row in process_observations)
            assert all(row["returncode"] == 0 and row["workspace_cleaned"] for row in process_observations)
            with acquire(lane=resources.ResourceLane.SNAPSHOT_EVALUATION,
                    cpu_slots=scheduler.config.total_cpu_slots, memory_mb=512,
                    child_process_slots=1, timeout=1.):
                with pytest.raises(resources.LeaseTimeoutError):
                    owner.checked_applicability(case.selection, nomination=case.nomination,
                        manifest=case.manifest, scheduler=scheduler)
            expired_at = time.monotonic() - started
            assert 54.5 <= expired_at < 59.
            assert len(process_observations) == 7
    finally:
        timer.cancel()
        timer.join(timeout=2.)
        blocker.release()
    assert not timer.is_alive() and not release_errors
    assert len(admission_waits) == 2
    assert 45. < admission_waits[0] <= 55. and 0 < admission_waits[1] < 10.
    assert len(child_admission_waits) == 6
    assert all(0 < value <= 90. for value in admission_waits + child_admission_waits)
    snapshot = scheduler.snapshot()
    assert snapshot["active_lease_count"] == snapshot["waiting_request_count"] == 0
    for row in process_observations:
        assert type(row["pid"]) is int and not Path(f"/proc/{row['pid']}").exists()
    record_property("controlled_captured_replay_contention", json.dumps(dict(
        real_scheduler=True, real_z3=True, authored_captured_source=True, simulated_clock=False,
        host_pressure_recovery_claimed=False, scheduled_release_seconds=46.,
        recovery_elapsed_seconds=recovered_at, enclosing_deadline_seconds=55.,
        expired_after_seconds=expired_at, admission_wait_limits_seconds=admission_waits,
        child_admission_wait_limits_seconds=child_admission_waits,
        first_process_after_seconds=process_observations[0]["invoked"] - started,
        query_execution_limits_seconds=[row["timeout_seconds"] for row in process_observations],
        solver_calls=evidence["solver_calls"], observed_processes_gone=len(process_observations),
        remaining_leases=0, remaining_waiters=0), sort_keys=True))
