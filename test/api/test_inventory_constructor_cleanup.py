"""Actual owner/transport custody after refused inventory construction.

The 300-source local signatures, native tasks, Quack RPC, scheduler lease,
bootstrap listener/thread, run coordinator and lifecycle STOP are real. Scan
receiving, execution material, installed launcher and isolated UID cleanup are
explicit host adapters. No model registry, training or worker Popen is used.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLane, ResourceSchedulerConfig,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as entry
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution as candidate_boundary
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import local_completion_bridge as completion
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from test.api.test_codebase_inventory_evidence_admission import inventory_case  # noqa: F401


@pytest.fixture
def native_constructor_case(inventory_case, tmp_path, monkeypatch):
    case = inventory_case
    materialized = local.materialize_local_benchmark_plan(admission=case["admission"], intent=case["intent"])
    database = case["intent"].database_path
    case["intent"].close()
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json", lane_reservations={}, auto_renew_leases=False))
    lease = scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=64,
                              child_process_slots=1, timeout=3)
    scan = case["context"]["scan"]
    root = resume.CodebaseScanResumeRoot.from_dict(scan["root_cid"], scan["root_record"])
    completed = resume.CodebaseScanResumeCompletion.from_dict(scan["completion_cid"], scan["completion_record"])
    observed = {"controlled_scan_root": root.artifact_cid, "controlled_completion": completed.artifact_cid}
    binding = {"schema": candidate_boundary.SCHEMA,
        "argv": ["/opt/ipfs-supervisor/bin/validation-worker"], "files": {},
        "owner_uid": os.geteuid(), "worker_uid": 1001,
        "namespaces": {name: os.readlink("/proc/self/ns/" + name) for name in ("pid", "mnt", "net")},
        "completion_authority": False}
    monkeypatch.setattr(candidate_boundary, "bind_candidate_runner", lambda argv: deepcopy(binding))
    monkeypatch.setattr(candidate_boundary, "verify_candidate_runner", lambda value: value)
    monkeypatch.setattr(candidate_boundary, "_root_file",
                        lambda path: hashlib.sha256(str(path).encode()).hexdigest())
    popen_calls, cleanup_calls, rpc_calls = [], [], []
    def forbidden_popen(*args, **kwargs):
        popen_calls.append(args)
        raise AssertionError("constructor control cannot launch a worker")
    # Git uses Popen internally. Intercept only the supervisor module's literal
    # birth via a module-local subprocess proxy instead of patching global Git.
    actual_run = subprocess.run
    def host_cleanup(command, *args, **kwargs):
        if list(command[:5]) == ["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--"]:
            assert command[-1] == "--cleanup"
            cleanup_calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return actual_run(command, *args, **kwargs)
    monkeypatch.setattr(entry, "subprocess", SimpleNamespace(
        run=host_cleanup, Popen=forbidden_popen, STDOUT=subprocess.STDOUT, PIPE=subprocess.PIPE,
        DEVNULL=subprocess.DEVNULL,
        TimeoutExpired=subprocess.TimeoutExpired, CalledProcessError=subprocess.CalledProcessError))
    with open_existing_native_owner(database=database, checkout=case["repository"], state_dir=tmp_path / "owner",
            repository_id=case["manifest"]["payload"]["repository_cid"],
            execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE for task in case["tasks"]}) as native:
        owner = execution._Owner(root, completed, object(), case["repository"], object(),
                                 None, None, threading.Event(), 120.0, 1024)
        scope = execution.FrozenInventoryExecutionScope(execution._SEAL, owner=owner,
            admission=local._plain(case["admission"]), candidate={}, server=native.server,
            source=native.source, lease=lease, output=tmp_path)
        with native.server._lock:
            physical = execution._physical_native(native.server._connection)
        scope._material = json.dumps({"payload": {"profile": execution.PROFILE,
            "native_population": {**physical, "selected_task_cids": [case["tasks"][0].task_cid]},
            "current_inventory": observed,
            "candidate": {"argv": ["/opt/ipfs-supervisor/bin/owner-worker"],
                          "implementation_command": "authored-no-launch-fixture"}}}).encode()
        @contextmanager
        def receiving(**arguments):
            assert arguments["parent_lease"] is lease
            yield observed, lambda: observed
        monkeypatch.setattr(execution.inventory, "_current_inventory_admission_operation", receiving)
        monkeypatch.setattr(scope, "_current_inventory", lambda: observed)
        def native_prelaunch():
            scope._active()
            assert not native.server._lock._is_owned()
            page, ready = native.source.list_tasks(limit=17), native.source.ready_tasks(limit=17)
            assert {row.task_cid for row in page.tasks} == set(materialized["task_cids"])
            assert [row.task_cid for row in ready.tasks] == [case["tasks"][0].task_cid]
            rpc_calls.append((page.revision, ready.revision))
        monkeypatch.setattr(scope, "require_prelaunch_current", native_prelaunch)
        def detached_native_rows():
            with native.server._lock:
                assert execution._physical_native(native.server._connection) == physical
        monkeypatch.setattr(scope, "_detached_fence", detached_native_rows)
        with execution._LOCK:
            execution._ACTIVE[lease.lease_id] = scope
        result = SimpleNamespace(scope=scope, native=native, case=case, scheduler=scheduler,
            lease=lease, popen_calls=popen_calls, cleanup_calls=cleanup_calls, rpc_calls=rpc_calls,
            binding=binding, output=tmp_path)
        try:
            yield result
        finally:
            if not lease.released:
                runtime = scope._runtime
                if runtime is not None and hasattr(runtime, "service"):
                    runtime.stop()
                    runtime.close()
                scope._finish()
            scope._renew_thread.join(timeout=2)
            assert scheduler.snapshot()["active_lease_count"] == 0


def create(case):
    return entry.AdmittedBenchmarkRuntime.create(case.output / "launch", admission=case.case["admission"],
        server=case.native.server, source=case.native.source, implement=True,
        implementation_command="authored-no-launch-fixture", candidate_runner_argv=case.binding["argv"],
        inventory_execution_scope=case.scope)


def assert_closed(case):
    runtime = case.scope._runtime
    assert runtime._context_refresh_stopped() and case.scope._cleaned
    assert runtime._children == [] and case.popen_calls == []
    assert runtime._listener.fileno() == -1 and not runtime._bootstrap_thread.is_alive()
    assert runtime._bootstrap_stop.is_set() and not runtime.coordinator.is_open
    assert len(case.cleanup_calls) == 1 and len(case.rpc_calls) >= 2
    gateway = case.native.server._command_gateway
    assert gateway._local_task_validation_handler is None and gateway._local_task_validation_binding is None
    # Inspect the actual closed native run lease through a fresh read-only owner.
    import duckdb
    connection = duckdb.connect(str(runtime.state / "coordination.duckdb"), read_only=True)
    try:
        rows = connection.execute("SELECT state, resource_id FROM fenced_leases").fetchall()
        assert rows == [("released", runtime.run_id)]
    finally:
        connection.close()
    case.scope._finish()
    assert case.lease.released and case.scheduler.snapshot()["active_lease_count"] == 0


def test_ordinary_inventory_constructor_keeps_exact_completion_stop_and_close_custody(native_constructor_case):
    case = native_constructor_case
    runtime = create(case)
    assert not case.scope._constructor_cleanup_pending and not case.scope._spawned
    assert type(runtime.completion_service) is completion.OwnerLocalCompletionService
    assert runtime._listener.fileno() >= 0 and runtime._bootstrap_thread.is_alive()
    assert runtime.stop().succeeded
    runtime.close()
    assert runtime.completion_service.retired
    assert_closed(case)


@pytest.mark.parametrize("optimized", [True, False])
def test_late_inventory_completion_callback_failure_closes_real_native_transports(
        native_constructor_case, monkeypatch, optimized):
    case = native_constructor_case
    if not optimized:
        body = case.scope._owner.root.to_dict()
        body["optimized"] = False
        case.scope._owner = replace(case.scope._owner,
            root=resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(body), body))
    def refused(**arguments):
        raise ValueError("controlled late completion callback refused")
    monkeypatch.setattr(completion, "bind_owner_local_completion_service", refused)
    with pytest.raises(ValueError, match="controlled late completion callback refused"):
        create(case)
    assert not getattr(case.scope._runtime, "_construction_cleanup_failed", False)
    assert_closed(case)


def test_post_install_service_constructor_fault_rolls_back_only_exact_native_handler(
        native_constructor_case, monkeypatch):
    case = native_constructor_case
    gateway = case.native.server._command_gateway
    seen = []
    original_unbind = gateway.unbind_local_task_validation_handler
    def exact_unbind(handler, binding):
        assert handler is gateway._local_task_validation_handler
        assert binding is gateway._local_task_validation_binding
        seen.append((handler, binding))
        return original_unbind(handler, binding)
    monkeypatch.setattr(gateway, "unbind_local_task_validation_handler", exact_unbind)
    def refused(self, seal, **arguments):
        assert arguments["handler"] is gateway._local_task_validation_handler
        assert arguments["binding"] is gateway._local_task_validation_binding
        raise ValueError("controlled service constructor refused after install")
    monkeypatch.setattr(completion.OwnerLocalCompletionService, "__init__", refused)
    with pytest.raises(ValueError, match="controlled service constructor refused after install"):
        create(case)
    assert len(seen) == 1 and getattr(case.scope._runtime, "completion_service", None) is None
    assert_closed(case)


def test_exact_unbind_failure_retains_live_lease_until_native_retry_cleanup(
        native_constructor_case, monkeypatch):
    case = native_constructor_case
    gateway = case.native.server._command_gateway
    actual_init, actual_unbind = completion.OwnerLocalCompletionService.__init__, gateway.unbind_local_task_validation_handler
    state = {"first": True, "refuse": True, "unbind": []}
    def fail_first(self, seal, **arguments):
        if state["first"]:
            state["first"] = False
            raise ValueError("controlled post-install constructor fault")
        return actual_init(self, seal, **arguments)
    def unbind(handler, binding):
        state["unbind"].append((handler, binding))
        if state["refuse"]:
            return False
        return actual_unbind(handler, binding)
    monkeypatch.setattr(completion.OwnerLocalCompletionService, "__init__", fail_first)
    monkeypatch.setattr(gateway, "unbind_local_task_validation_handler", unbind)
    with pytest.raises(RuntimeError, match="inventory constructor cleanup unproven"):
        create(case)
    runtime = case.scope._runtime
    original_handler, original_binding = gateway._local_task_validation_handler, gateway._local_task_validation_binding
    assert original_handler is not None and original_binding is not None
    assert all(pair == (original_handler, original_binding) for pair in state["unbind"])
    assert runtime._context_refresh_stopped() and runtime._construction_cleanup_failed
    assert not case.lease.released and case.scope._renew_thread.is_alive()
    assert runtime._listener.fileno() >= 0 and runtime._bootstrap_thread.is_alive()
    assert runtime.coordinator.is_open and case.popen_calls == []
    original_pending = runtime._pending_completion_binding
    runtime._pending_completion_binding = SimpleNamespace(close=lambda _runtime: pytest.fail("foreign cleanup invoked"))
    with pytest.raises(RuntimeError, match="exact native binding custody"):
        runtime.close()
    runtime._pending_completion_binding = original_pending
    assert not case.lease.released
    # A mutable diagnostic cannot falsely authorize lease release.
    runtime._construction_cleanup_failed = False
    with pytest.raises(execution.InventoryExecutionError, match="still owns"):
        case.scope._finish()
    assert not case.lease.released and case.scope._renew_thread.is_alive()
    # A foreign callback is never detached by the retained exact token.
    foreign = lambda **arguments: None
    with gateway._grants_lock:
        gateway._local_task_validation_handler = foreign
    state["refuse"] = False
    with pytest.raises(local.LocalPlanningError, match="foreign handler"):
        runtime.close()
    assert gateway._local_task_validation_handler is foreign and not case.lease.released
    with gateway._grants_lock:
        gateway._local_task_validation_handler = original_handler
    runtime.close()
    assert runtime.completion_service.retired
    assert_closed(case)
