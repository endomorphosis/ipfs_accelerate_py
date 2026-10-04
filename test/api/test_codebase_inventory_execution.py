"""Launch boundaries with real resource leases and native SQL, without workers.

Public-artifact and signed-admission protocol cases use explicit small inputs;
they do not qualify model freshness or a container launch. The native resume
qualification separately exercises those owners and the isolated worker.
"""
import hashlib
import json
import os
from types import SimpleNamespace
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as runtime_module
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig, ResourceLane,
)


def test_private_scope_cannot_be_constructed_from_an_envelope():
    with pytest.raises(execution.InventoryExecutionError, match="native reservation"):
        execution.FrozenInventoryExecutionScope(object(), owner=None, admission={}, candidate={},
                                               server=None, source=None, lease=None, output=None)


def test_runtime_cannot_combine_finite_and_inventory_authority(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        runtime_module.AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=None, server=None,
            source=None, finite_execution_scope=object(), inventory_execution_scope=object())
    assert not (tmp_path / "launch").exists()


def test_inventory_manifest_cannot_use_the_generic_runtime_path(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "verify_local_benchmark_admission", lambda *args, **kwargs: {
        "manifest": {"schema": local.INVENTORY_MANIFEST_SCHEMA}})
    with pytest.raises(ValueError, match="exact native inventory execution scope"):
        runtime_module.AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission={}, server=None, source=None)
    assert not (tmp_path / "launch").exists()


@pytest.mark.parametrize("scope", [{"payload": {}}, object()])
def test_serialized_or_foreign_inventory_scope_cannot_authorize_launch(tmp_path, monkeypatch, scope):
    monkeypatch.setattr(runtime_module, "verify_local_benchmark_admission", lambda *args, **kwargs: {
        "manifest": {"schema": local.INVENTORY_MANIFEST_SCHEMA}, "graph": SimpleNamespace(tasks=())})
    with pytest.raises(ValueError, match="exact native inventory execution scope"):
        runtime_module.AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission={}, server=None,
                                                     source=None, inventory_execution_scope=scope)
    assert not (tmp_path / "launch").exists()


def candidate_case():
    descriptor = {"artifact": "/repository/.runtime/router-public-instruction/pinned.json",
                  "sha256": "ab" * 32, "task_cid": "selected"}
    candidate = {"public_instruction": descriptor, "task_revision": 3, "argv": [
        "/opt/ipfs-supervisor/bin/owner-worker", "--model", "inventory-authored-fixture",
        "--purpose", "coding", "--public-instruction-artifact", descriptor["artifact"],
        "--public-instruction-sha256", descriptor["sha256"],
        "--public-instruction-task-cid", "selected", "--timeout", "90"]}
    population = {"selected_task_cids": ["selected"],
                  "tasks": [{"task_cid": "selected", "task_alias": "RUNTIME", "revision": 3}]}
    admission = {"manifest": {"payload": {"repository": "/repository"}}}
    return candidate, descriptor, population, admission


def test_closed_worker_command_retains_exact_public_artifact_and_ready_revision(monkeypatch):
    candidate, descriptor, population, admission = candidate_case()
    monkeypatch.setattr(execution, "_public_artifact", lambda *args: descriptor)
    result = execution._candidate(candidate, admission, population)
    assert result["public_instruction"] is descriptor
    assert result["task_revision"] == 3
    import shlex
    assert shlex.split(result["implementation_command"]) == candidate["argv"]


@pytest.mark.parametrize("change", [
    "bool_revision", "stale_revision", "foreign_launcher", "missing_artifact", "duplicate_artifact",
    "foreign_sha", "foreign_task", "unknown_option", "odd_options", "doctor_route", "preflight_route",
    "planning_purpose", "bool_timeout", "zero_timeout", "unbounded_timeout", "leading_zero_timeout",
    "foreign_repository", "duplicate_model", "unsupported_effort",
])
def test_worker_options_cannot_rebind_or_broaden_signed_inventory_route(monkeypatch, change):
    candidate, descriptor, population, admission = candidate_case()
    monkeypatch.setattr(execution, "_public_artifact", lambda *args: descriptor)
    argv = candidate["argv"]
    if change == "bool_revision":
        candidate["task_revision"] = True
    elif change == "stale_revision":
        candidate["task_revision"] = 2
    elif change == "foreign_launcher":
        argv[0] = "/tmp/owner-worker"
    elif change == "missing_artifact":
        at = argv.index("--public-instruction-artifact")
        del argv[at:at + 2]
    elif change == "duplicate_artifact":
        argv.extend(["--public-instruction-artifact", descriptor["artifact"]])
    elif change == "foreign_sha":
        argv[argv.index("--public-instruction-sha256") + 1] = "cd" * 32
    elif change == "foreign_task":
        argv[argv.index("--public-instruction-task-cid") + 1] = "foreign"
    elif change == "unknown_option":
        argv.extend(["--extra-authority", "yes"])
    elif change == "odd_options":
        argv.append("--model")
    elif change == "doctor_route":
        argv.extend(["--doctor-task-cid", "foreign"])
    elif change == "preflight_route":
        argv.append("--preflight")
    elif change == "planning_purpose":
        argv[argv.index("--purpose") + 1] = "planning"
    elif change in {"bool_timeout", "zero_timeout", "unbounded_timeout", "leading_zero_timeout"}:
        argv[argv.index("--timeout") + 1] = {
            "bool_timeout": "True", "zero_timeout": "0", "unbounded_timeout": "301", "leading_zero_timeout": "090"
        }[change]
    elif change == "foreign_repository":
        argv.extend(["--semantic-repository", "/foreign"])
    elif change == "duplicate_model":
        argv.extend(["--model", "different"])
    elif change == "unsupported_effort":
        argv.extend(["--reasoning-effort", "unbounded"])
    with pytest.raises(execution.InventoryExecutionError):
        execution._candidate(candidate, admission, population)


@pytest.mark.parametrize("change", [
    {"cpu_slots": True}, {"cpu_slots": 3}, {"execution_memory_mb": True}, {"execution_memory_mb": 4095},
    {"child_process_slots": True}, {"child_process_slots": 7}, {"admission_timeout_seconds": True},
    {"admission_timeout_seconds": float("nan")}, {"timeout_seconds": True}, {"timeout_seconds": 601},
    {"memory_mb": True}, {"memory_mb": 1023},
])
def test_invalid_resource_envelopes_refuse_before_owners_or_output(tmp_path, change):
    with pytest.raises(execution.InventoryExecutionError, match="envelope"):
        with execution.reserve_inventory_execution(root=None, completion=None, index=None, repository=tmp_path,
                registry=None, admission={}, candidate={}, server=None, source=None,
                output=tmp_path / "refused", **change):
            pytest.fail("invalid envelope admitted")
    assert not (tmp_path / "refused").exists()


def test_default_scheduler_without_proof_host_safety_cannot_grant_inventory_launch(tmp_path):
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig(state_path=tmp_path / "resources.json"))
    with pytest.raises(execution.InventoryExecutionError, match="genuine host proof safety"):
        with execution.reserve_inventory_execution(root=None, completion=None, index=None, repository=tmp_path,
                registry=None, admission={}, candidate={}, server=None, source=None,
                output=tmp_path / "refused", scheduler=scheduler):
            pytest.fail("unsafe scheduler admitted")
    assert scheduler.snapshot()["active_lease_count"] == 0


@pytest.fixture
def leased_scope(tmp_path):
    # Actual host sampling and scheduler state; no substituted resource sampler.
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json", lane_reservations={}, auto_renew_leases=False))
    cancel = threading.Event()
    lease = scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=64,
                              child_process_slots=1, timeout=3)
    owner = execution._Owner(None, None, object(), tmp_path, object(), None, None, cancel, 120, 1024)
    server = SimpleNamespace(_lock=threading.RLock(), _connection=object(),
                             identity=SimpleNamespace(to_dict=lambda: {"owner": "original"}))
    source = SimpleNamespace(execution_route_policy=SimpleNamespace(to_dict=lambda: {"route": "original"}))
    scope = execution.FrozenInventoryExecutionScope(execution._SEAL, owner=owner, admission={}, candidate={},
                                                  server=server, source=source, lease=lease, output=tmp_path)
    scope._material = json.dumps({"payload": {"native_population": {"selected_task_cids": ["selected"]}}}).encode()
    scope._launcher = {"path": "/opt/ipfs-supervisor/bin/owner-worker", "sha256": "ab" * 32}
    with execution._LOCK:
        execution._ACTIVE[lease.lease_id] = scope
    try:
        yield scope, cancel, scheduler
    finally:
        scope._runtime = None
        scope._finish()
        scope._renew_thread.join(timeout=2)
        assert scheduler.snapshot()["active_lease_count"] == 0


def runtime_case(scope, *, cleaned):
    runtime = SimpleNamespace(manifest={"inventory_execution_scope": scope.to_dict(),
                                      "inventory_worker_launcher": scope.worker_launcher_binding},
                              server=scope._server, source=scope._source, admission=scope._admission,
                              _context_refresh_stopped=lambda: cleaned)
    scope._runtime = runtime
    return runtime


def test_real_lease_and_private_active_identity_are_both_required(leased_scope):
    scope, cancel, scheduler = leased_scope
    scope._active()
    assert scope.parent_lease.owner_pid > 0 and scheduler.snapshot()["active_lease_count"] == 1
    detached = scope.to_dict()
    detached["payload"]["native_population"]["selected_task_cids"].clear()
    assert scope.selected_task_cids == ("selected",)
    with execution._LOCK:
        execution._ACTIVE.pop(scope.parent_lease.lease_id)
    try:
        with pytest.raises(execution.InventoryExecutionError, match="exact active"):
            scope._active()
    finally:
        with execution._LOCK:
            execution._ACTIVE[scope.parent_lease.lease_id] = scope


def test_cancelled_real_lease_refuses_dispatch_but_allows_cleanup(leased_scope):
    scope, cancel, _scheduler = leased_scope
    runtime = runtime_case(scope, cleaned=True)
    scope.note_spawned(runtime)
    cancel.set()
    with pytest.raises(execution.InventoryExecutionError, match="cancelled"):
        scope._active()
    scope.finish_runtime(runtime)
    scope.require_close(runtime)
    assert scope._cleaned and not scope.parent_lease.released


def test_failed_cleanup_never_releases_a_real_envelope(leased_scope):
    scope, _cancel, scheduler = leased_scope
    runtime = runtime_case(scope, cleaned=False)
    scope.note_spawned(runtime)
    runtime.stop = lambda: None
    with pytest.raises(execution.InventoryExecutionError, match="STOP and isolated UID cleanup"):
        scope._finish()
    assert not scope.parent_lease.released and scheduler.snapshot()["active_lease_count"] == 1
    with pytest.raises(execution.InventoryExecutionError, match="successful STOP"):
        scope.require_close(runtime)


def test_signed_runtime_material_cannot_be_changed_even_during_stop(leased_scope):
    scope, _cancel, _scheduler = leased_scope
    runtime = runtime_case(scope, cleaned=True)
    runtime.manifest["inventory_execution_scope"]["payload"]["native_population"]["selected_task_cids"] = ["foreign"]
    with pytest.raises(execution.InventoryExecutionError, match="signed launch"):
        scope.finish_runtime(runtime)
    assert not scope._cleaned


def test_process_birth_records_cleanup_duty_before_later_cancellation(leased_scope):
    scope, cancel, _scheduler = leased_scope
    runtime = runtime_case(scope, cleaned=True)
    cancel.set()
    scope.note_spawned(runtime)
    assert scope._spawned
    with pytest.raises(execution.InventoryExecutionError, match="already launched"):
        scope.note_spawned(runtime)
    scope.finish_runtime(runtime)


def prelaunch_case(scope, monkeypatch):
    population = {"selected_task_cids": ["selected"], "tasks": [], "completion_rows": {}}
    observed = {"root_cid": "root", "completion_cid": "completion", "model": "model"}
    candidate = {"public_instruction": {}, "argv": [], "task_revision": 3}
    payload = {"implementation": {}, "native_population": population, "current_inventory": observed,
               "candidate": candidate}
    scope._material = json.dumps({"payload": payload}).encode()
    monkeypatch.setattr(execution, "_pins", lambda: {})
    monkeypatch.setattr(execution, "_native_population", lambda *args: population)
    monkeypatch.setattr(execution, "_candidate", lambda *args: candidate)
    monkeypatch.setattr(local, "verify_local_benchmark_admission", lambda *args, **kwargs: {"profile": object()})
    monkeypatch.setattr(local, "_verify_signature", lambda envelope, profile: envelope["payload"])
    return observed


def test_final_owner_spawn_hook_replays_receiving_gate_and_detached_rows(leased_scope, monkeypatch):
    scope, _cancel, _scheduler = leased_scope
    observed = prelaunch_case(scope, monkeypatch)
    runtime = runtime_case(scope, cleaned=False)
    calls = []
    monkeypatch.setattr(scope, "_current_inventory", lambda: calls.append("current") or observed)
    monkeypatch.setattr(scope, "_detached_fence", lambda: calls.append("detached"))
    scope.prepare_spawn_fence(runtime)
    with scope._server._lock:
        scope.require_spawn_fence(runtime)
    assert calls == ["current", "detached", "current", "detached"]
    assert not scope._spawned


def test_model_or_completion_drift_after_signer_callback_refuses_owner_spawn(leased_scope, monkeypatch):
    scope, _cancel, _scheduler = leased_scope
    observed = prelaunch_case(scope, monkeypatch)
    runtime = runtime_case(scope, cleaned=False)
    calls = []
    def current():
        calls.append("current")
        return observed if len(calls) == 1 else {**observed, "model": "successor-model"}
    monkeypatch.setattr(scope, "_current_inventory", current)
    monkeypatch.setattr(scope, "_detached_fence", lambda: calls.append("detached"))
    scope.prepare_spawn_fence(runtime)
    with pytest.raises(execution.InventoryExecutionError, match="during prelaunch callbacks"):
        with scope._server._lock:
            scope.require_spawn_fence(runtime)
    assert calls == ["current", "detached", "current"] and not scope._spawned and not scope.parent_lease.released


def test_full_native_population_change_refuses_owner_spawn_before_second_receiving_read(leased_scope, monkeypatch):
    scope, _cancel, _scheduler = leased_scope
    observed = prelaunch_case(scope, monkeypatch)
    runtime = runtime_case(scope, cleaned=False)
    monkeypatch.setattr(scope, "_current_inventory", lambda: observed)
    monkeypatch.setattr(execution, "_native_population", lambda *args: {"selected_task_cids": ["foreign"]})
    with pytest.raises(execution.InventoryExecutionError, match="full native task population changed"):
        scope.prepare_spawn_fence(runtime)
    assert not scope._spawned and not scope.parent_lease.released


def test_foreign_runtime_cannot_trigger_final_receiving_gate(leased_scope, monkeypatch):
    scope, _cancel, _scheduler = leased_scope
    runtime_case(scope, cleaned=False)
    monkeypatch.setattr(scope, "require_prelaunch_current", lambda: pytest.fail("foreign runtime reached live owner"))
    with pytest.raises(execution.InventoryExecutionError, match="exact unlaunched"):
        scope.require_spawn_fence(object())


def test_final_detached_closure_never_replays_public_planning_callbacks(leased_scope, monkeypatch, tmp_path):
    scope, _cancel, _scheduler = leased_scope
    identity, route = {"owner": "original"}, {"route": "original"}
    population = {"tasks": [], "completion_rows": {}, "authority_rows": {"plans": []}, "owner_identity": identity,
                  "execution_route_policy": route}
    descriptor = {"artifact": "pinned", "sha256": "ab" * 32}
    scope._owner = SimpleNamespace(repository=tmp_path)
    scope._server = SimpleNamespace(_lock=threading.RLock(), _connection=object(),
                                   identity=SimpleNamespace(to_dict=lambda: dict(identity)))
    scope._source = SimpleNamespace(execution_route_policy=SimpleNamespace(to_dict=lambda: dict(route)))
    scope._material = json.dumps({"payload": {"native_population": population,
                                             "candidate": {"public_instruction": descriptor}}}).encode()
    del scope._launcher
    calls = []
    monkeypatch.setattr(execution, "_physical_native", lambda connection: calls.append("sql") or
                        {name: population[name] for name in ("tasks", "completion_rows", "authority_rows")})
    monkeypatch.setattr(execution, "_public_artifact_bytes", lambda given, repository:
                        calls.append("bytes") or b"pinned")
    monkeypatch.setattr(execution, "_public_artifact", lambda *args:
                        pytest.fail("final detached closure called signed graph/planning replay"))
    scope._detached_fence()
    assert calls == ["sql", "bytes"]


@pytest.mark.parametrize("change", ["none", "bytes", "mode", "symlink", "hardlink", "oversized"])
def test_final_public_artifact_byte_guard_checks_real_files(tmp_path, change):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as public
    repository = tmp_path / "repository"
    directory = repository / public.DIRECTORY
    directory.mkdir(parents=True)
    path = directory / "artifact.json"
    raw = b"exact prepared bytes\n"
    path.write_bytes(raw)
    path.chmod(0o444)
    descriptor = {"artifact": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "task_cid": "selected",
        "context_cid": "context", "manifest_cid": "manifest", "source_path": "requirements.txt",
        "source_sha256": "ab" * 32, "source_bytes": 1, "completion_authority": False,
        "scope_expansion_authority": False, "inventory_context_cid": "worker-context",
        "codebase_inventory_context_cid": "full-context"}
    if change == "none":
        assert execution._public_artifact_bytes(descriptor, repository) == raw
        return
    if change == "bytes":
        path.chmod(0o600)
        path.write_bytes(b"changed prepared bytes\n")
        path.chmod(0o444)
    elif change == "mode":
        path.chmod(0o644)
    elif change == "symlink":
        target = directory / "target.json"
        path.rename(target)
        path.symlink_to(target)
    elif change == "hardlink":
        os.link(path, directory / "other.json")
    elif change == "oversized":
        path.chmod(0o600)
        with path.open("wb") as stream:
            stream.truncate(public.MAX_INVENTORY_BYTES + 1)
        path.chmod(0o444)
    with pytest.raises((execution.InventoryExecutionError, OSError)):
        execution._public_artifact_bytes(descriptor, repository)


def test_detached_native_sql_preserves_full_relation_rows_and_detects_edits(tmp_path):
    spec = {"outputs": [{"path": "answer.py", "effect": "modify", "media_type": "text/x-python"}],
            "acceptance": [{"criterion_key": "answer", "criterion": "Public answer passes", "validation_keys": ["check"]}],
            "validations": [{"validation_key": "check", "argv": ["python", "test_answer.py"], "cwd": ".",
                             "expected_exit_codes": [0], "policy_cid": content_identity({"policy": "public"})}]}
    cid = content_identity({"task": "actual-sql"})
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        goal = content_identity({"goal": "actual-sql"})
        intent.upsert_goal(goal_cid=goal, goal_alias="SQL-GOAL", title="public check")
        intent.upsert_task(task_cid=cid, task_alias="SQL-TASK", goal_cid=goal, body={"title": "public check"},
                           identity={}, dependencies=[], **spec)
        with intent._connection(write=True) as connection:
            captured = execution._physical_native(connection)
            row = captured["tasks"][0]
            assert {key: row[key] for key in spec} == execution._relations(spec)
            connection.execute("UPDATE task_validations SET argv_json = ? WHERE task_cid = ?",
                               [json.dumps(["python", "foreign.py"]), cid])
            changed = execution._physical_native(connection)
            assert not execution._same(changed, captured)
            assert changed["tasks"][0]["validations"][0]["argv"] == ["python", "foreign.py"]


def test_canonical_execution_material_distinguishes_boolean_and_integer():
    assert not execution._same({"revision": True}, {"revision": 1})
