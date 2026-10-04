"""Closed native Lean checks and heterogeneous shared-owner execution."""
from dataclasses import replace
import shutil
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import datasets_kernel_tasks as tasks
from ipfs_accelerate_py.agent_supervisor.proof.datasets_hammer_tasks import make_datasets_smt_task
from ipfs_accelerate_py.agent_supervisor.proof.datasets_prover_resources import open_datasets_prover_lease
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    BundleProverSupervisor, ExecutionStatus, MultiProverResourceBudget, MultiProverResourceLease,
    ProverTaskExecutor,
)
from ipfs_datasets_py.logic.hammers import semantic_routing as routing
from ipfs_datasets_py.logic.backends.kernel import lean
from ipfs_datasets_py.logic.backends import process
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig,
)


def budget():
    return MultiProverResourceBudget(cpu_slots=2, process_slots=2, thread_slots=2,
        memory_bytes=2048 * 1024**2, max_portfolio_width=2)


@pytest.fixture
def native(tmp_path):
    assert shutil.which("lean"), "native qualification requires installed Lean"
    healthy = ProofHostResources(8, 8192, 8192)
    pressure = [healthy]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "leases.json", proof_resource_sampler=lambda: pressure[0],
        total_cpu_slots=4, total_memory_mb=4096, total_child_process_slots=4,
        lane_reservations={}, proof_backoff_seconds=.03, poll_interval_seconds=.005))
    yield scheduler, pressure, healthy
    deadline = time.monotonic() + 3
    while scheduler.snapshot()["active_lease_count"] and time.monotonic() < deadline:
        time.sleep(.005)
    assert scheduler.snapshot()["active_lease_count"] == 0
    assert scheduler.snapshot()["waiting_request_count"] == 0


def smt_target(task_id):
    parsed = routing.modal.parse_modal("p and not p", routing.modal.profile_k())
    replay = routing.modal.parse_modal(parsed.printed, routing.modal.profile_k())
    return dict(request_id=task_id, source_construct="authored-mixed-fixture",
        logic_family="propositional", ast_format="shared_logic", printed=parsed.printed,
        native_ast=replay.root.to_dict(), solver_names=("z3",))


def execute(scheduler, task):
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        return ProverTaskExecutor(lease).execute(task)


@pytest.mark.parametrize("offset", [0, 7, 65535])
def test_native_exact_closed_theorem_with_kernel_receipt_and_no_source_authority(native, offset):
    scheduler, _, _ = native
    task = tasks.make_datasets_lean_nat_task(task_id=f"lean-{offset}", offset=offset)
    result = execute(scheduler, task)
    assert result.status is ExecutionStatus.SUCCEEDED, result.to_dict()
    payload = result.result
    assert payload["kernel_accepted"] and payload["kernel_authority"] and payload["bindings_match"]
    assert payload["statement"] == f"forall n : Nat, n + {offset} = {offset} + n"
    assert payload["domain"] == "Lean.Nat"
    receipt = payload["kernel_receipt"]
    assert receipt["accepted"] and receipt["request_digest"] == payload["request_digest"]
    assert receipt["axiom_report"]["axioms"] == []
    assert receipt["axiom_report"]["contains_sorry_ax"] is False
    assert payload["typed_result"]["authority"] == "theorem"
    assert all(payload[key] is False for key in ("source_semantics_verified", "behavior_authority",
        "execution_authority", "completion_authority"))
    assert not task.deterministic and not result.cache_bypassed_execution


def test_native_smt_and_lean_bundle_share_one_root_and_obey_dependencies(native, monkeypatch):
    scheduler, _, _ = native
    assert shutil.which("z3"), "native qualification requires Z3"
    captured = []
    original = process.SubprocessExecutor.execute
    def observe(self, invocation, cancellation=None):
        state = scheduler.snapshot()
        captured.append((state["active_root_lease_count"], state["active_child_lease_count"], invocation))
        return original(self, invocation, cancellation)
    monkeypatch.setattr(process.SubprocessExecutor, "execute", observe)
    first = tasks.make_datasets_lean_nat_task(task_id="native-lean", offset=5)
    smt = make_datasets_smt_task(target=smt_target("native-smt"), expected_verdict="unsat")
    after = tasks.make_datasets_lean_nat_task(task_id="after-both", offset=8,
        dependencies=("native-lean", "native-smt"))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        result = BundleProverSupervisor(lease).execute((first, smt, after))
    assert result.successful_task_ids == ("native-lean", "native-smt", "after-both"), result.to_dict()
    by_id = {item.task_id: item for item in result.receipts}
    assert by_id["after-both"].started_at_ms >= max(by_id["native-lean"].finished_at_ms,
                                                  by_id["native-smt"].finished_at_ms)
    assert by_id["native-lean"].result["kernel_authority"]
    assert not by_id["native-smt"].result["proof_authority"]
    assert len(captured) == 4  # one bounded version probe and one kernel run per Lean task
    assert all(roots == 1 and children >= 1 for roots, children, _ in captured)
    for _, _, invocation in captured:
        assert invocation.limits.memory_bytes == 4096 * 1024**2
        assert invocation.limits.resident_memory_bytes == 512 * 1024**2
        assert invocation.limits.cpu_seconds > 0
        assert invocation.environment["PROVER_MAX_THREADS"] == "1"
        if "--json" in invocation.argv:
            assert "-j1" in invocation.argv


@pytest.mark.parametrize("values", [
    {"offset": True}, {"offset": -1}, {"offset": 65536}, {"offset": "1"},
    {"task_id": ""}, {"task_id": " x "}, {"task_id": "x" * 1025},
    {"timeout_seconds": 0}, {"timeout_seconds": float("inf")}, {"timeout_seconds": True},
    {"memory_mb": 0}, {"memory_mb": 4097}, {"memory_mb": True},
    {"dependencies": "one"}, {"dependencies": ["x"] * 65},
])
def test_invalid_closed_profile_inputs_fail_before_execution(values):
    with pytest.raises(ValueError):
        tasks.make_datasets_lean_nat_task(**{**dict(task_id="safe", offset=1), **values})


def test_standalone_supervisor_cannot_run_native_task_without_shared_authority():
    with MultiProverResourceLease(budget()) as lease:
        receipt = ProverTaskExecutor(lease).execute(tasks.make_datasets_lean_nat_task(task_id="unbridged", offset=1))
    assert receipt.status is ExecutionStatus.FAILED
    assert "datasets-backed" in receipt.diagnostics


def test_missing_lean_preserves_unavailability_and_blocks_dependent(native, monkeypatch):
    scheduler, _, _ = native
    monkeypatch.setattr(tasks.shutil, "which", lambda _: None)
    first = tasks.make_datasets_lean_nat_task(task_id="missing", offset=1)
    after = tasks.make_datasets_lean_nat_task(task_id="after", offset=2, dependencies=("missing",))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        result = BundleProverSupervisor(lease).execute((first, after))
    assert result.receipts[0].status is ExecutionStatus.FAILED
    assert result.receipts[0].result["status"] == "unavailable"
    assert not result.receipts[0].result["kernel_authority"]
    assert result.receipts[1].status is ExecutionStatus.BLOCKED


def test_forged_result_dictionary_never_becomes_kernel_authority(native, monkeypatch):
    scheduler, _, _ = native
    monkeypatch.setattr(lean.LeanKernelBackend, "run", lambda *a, **k: {
        "accepted": True, "status": "proved", "kernel_authority": True})
    result = execute(scheduler, tasks.make_datasets_lean_nat_task(task_id="forged", offset=3))
    assert result.status is ExecutionStatus.FAILED
    assert result.reasons == ("lean_binding_mismatch",)
    assert result.result["kernel_authority"] is False


def test_previous_real_accepted_outcome_is_rejected_for_a_different_request(native, monkeypatch):
    scheduler, _, _ = native
    original = lean.LeanKernelBackend.run
    previous = []
    def store_then_replay(self, request, **kwargs):
        if previous:
            return previous[0]
        result = original(self, request, **kwargs)
        previous.append(result)
        return result
    monkeypatch.setattr(lean.LeanKernelBackend, "run", store_then_replay)
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        executor = ProverTaskExecutor(lease)
        good = executor.execute(tasks.make_datasets_lean_nat_task(task_id="first", offset=3))
        stale = executor.execute(tasks.make_datasets_lean_nat_task(task_id="second", offset=9))
    assert good.status is ExecutionStatus.SUCCEEDED, good.to_dict()
    assert stale.status is ExecutionStatus.FAILED
    assert stale.reasons == ("lean_binding_mismatch",)
    assert stale.result["kernel_authority"] is False


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_pressure_wait_stops_without_any_native_probe_or_kernel_launch(native, monkeypatch, stop):
    scheduler, pressure, healthy = native
    launches = []
    monkeypatch.setattr(process.SubprocessExecutor, "execute", lambda *a, **k: launches.append(a))
    cancel = threading.Event()
    receipts = []
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(), admission_timeout_seconds=2) as lease:
        pressure[0] = replace(healthy, available_memory_mb=10)
        task = tasks.make_datasets_lean_nat_task(task_id="waiting", offset=1,
            timeout_seconds=.1 if stop == "deadline" else 2)
        thread = threading.Thread(target=lambda: receipts.append(ProverTaskExecutor(lease).execute(task, cancellation=cancel)))
        thread.start()
        if stop == "cancel":
            deadline = time.monotonic() + 1
            while not scheduler.snapshot()["waiting_request_count"] and time.monotonic() < deadline:
                time.sleep(.005)
            cancel.set()
        thread.join(2)
        assert not thread.is_alive()
    assert not launches
    assert receipts[0].status is (ExecutionStatus.CANCELLED if stop == "cancel" else ExecutionStatus.TIMED_OUT)


def test_rebound_accepted_packet_without_fresh_kernel_execution_fails(native, monkeypatch):
    scheduler, _, _ = native
    from ipfs_datasets_py.logic.ir_core.claims import FrozenMap
    original = lean.LeanKernelBackend.run
    previous = []
    def forged_rebound(self, request, **kwargs):
        if not previous:
            observed = original(self, request, **kwargs)
            previous.append(observed)
            return observed
        old = previous[0]
        binding = replace(old.source_binding, request_digest=request.digest)
        receipt = replace(old.receipt, request_digest=request.digest, source_binding=binding)
        metadata = old.result.metadata.to_dict()
        metadata.update(kernel_receipt=receipt.to_dict(), source_binding=binding.to_dict())
        witness = old.result.witness.to_dict()
        witness["receipt_id"] = receipt.receipt_id
        result = replace(old.result, result_id=f"result:lean:{request.digest[:24]}",
            bounds=request.bounds, metadata=FrozenMap(metadata), witness=FrozenMap(witness))
        return replace(old, request_digest=request.digest, source_binding=binding,
                       receipt=receipt, result=result)
    monkeypatch.setattr(lean.LeanKernelBackend, "run", forged_rebound)
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        executor = ProverTaskExecutor(lease)
        first = executor.execute(tasks.make_datasets_lean_nat_task(task_id="first", offset=4))
        rebound = executor.execute(tasks.make_datasets_lean_nat_task(task_id="rebound", offset=4))
    assert first.status is ExecutionStatus.SUCCEEDED, first.to_dict()
    assert rebound.status is ExecutionStatus.FAILED
    assert rebound.result["bindings_match"]  # Identity checks alone would accept this packet.
    assert rebound.result["native_checks"] == 0
    assert not rebound.result["kernel_authority"]
    assert rebound.reasons == ("lean_kernel_not_accepted",)


def test_cancellation_of_live_native_kernel_reaps_work_before_releasing_owner(native, monkeypatch):
    scheduler, _, _ = native
    original = process.SubprocessExecutor.execute
    started, finished, cancel = threading.Event(), threading.Event(), threading.Event()
    pids, results = [], []
    def instrument(self, invocation, cancellation=None):
        if "--json" not in invocation.argv:
            return original(self, invocation, cancellation)
        original_popen = self._popen
        def observed(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            pids.append(child.pid)
            started.set()
            return child
        self._popen = observed
        try:
            return original(self, invocation, cancellation)
        finally:
            finished.set()
    monkeypatch.setattr(process.SubprocessExecutor, "execute", instrument)
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        thread = threading.Thread(target=lambda: results.append(ProverTaskExecutor(lease).execute(
            tasks.make_datasets_lean_nat_task(task_id="live-cancel", offset=65535), cancellation=cancel)))
        thread.start()
        try:
            assert started.wait(5)
            assert scheduler.snapshot()["active_child_lease_count"] >= 1
            cancel.set()
            thread.join(2)
            assert not thread.is_alive()
            assert results[0].status is ExecutionStatus.CANCELLED
            assert not results[0].result
            assert finished.wait(2)
        finally:
            cancel.set()
            thread.join(3)
    from pathlib import Path
    assert pids and all(not Path(f"/proc/{pid}").exists() for pid in pids)


def test_repacked_task_cannot_underreserve_the_native_memory_envelope(native, monkeypatch):
    scheduler, _, _ = native
    launched = []
    monkeypatch.setattr(process.SubprocessExecutor, "execute", lambda *a, **k: launched.append(a))
    task = tasks.make_datasets_lean_nat_task(task_id="underreserved", offset=1)
    task = replace(task, resources=replace(task.resources, memory_bytes=128 * 1024**2))
    receipt = execute(scheduler, task)
    assert receipt.status is ExecutionStatus.FAILED
    assert receipt.reasons == ("lean_resource_envelope",)
    assert not launched


@pytest.mark.parametrize("mutation", ["witness", "backend", "result_id", "source_binding", "capability", "witness_bool"])
def test_real_native_result_with_contradictory_typed_metadata_is_rejected(native, monkeypatch, mutation):
    scheduler, _, _ = native
    from ipfs_datasets_py.logic.ir_core.claims import FrozenMap
    original = lean.LeanKernelBackend.run
    def mutate(self, request, **kwargs):
        observed = original(self, request, **kwargs)
        assert observed.receipt.accepted
        result = observed.result
        if mutation == "witness":
            value = result.witness.to_dict()
            value.update(theorem_name="unrelated_theorem", theorem_digest="f" * 64)
            result = replace(result, witness=FrozenMap(value))
        elif mutation == "witness_bool":
            value = result.witness.to_dict()
            value["axiom_report"]["contains_sorry_ax"] = 0
            result = replace(result, witness=FrozenMap(value))
        elif mutation == "backend":
            result = replace(result, backend_id="unrelated_backend")
        elif mutation == "result_id":
            result = replace(result, result_id="result:other")
        else:
            value = result.metadata.to_dict()
            value[mutation] = {"unrelated": True}
            result = replace(result, metadata=FrozenMap(value))
        return replace(observed, result=result)
    monkeypatch.setattr(lean.LeanKernelBackend, "run", mutate)
    receipt = execute(scheduler, tasks.make_datasets_lean_nat_task(task_id="mutated", offset=7))
    assert receipt.status is ExecutionStatus.FAILED
    assert receipt.result["native_checks"] == 1 and receipt.result["native_output_accepted"]
    assert receipt.result["bindings_match"] is False
    assert receipt.result["kernel_authority"] is False
    assert receipt.reasons == ("lean_binding_mismatch",)
