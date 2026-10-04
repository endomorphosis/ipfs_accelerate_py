"""Reservation protocol controls with genuine, unchanged proof-safe leases.

The source/model receiver and signed full-graph callbacks are explicit inert
doubles. Real files exercise byte custody. These tests perform no Git, SQL
owner work, fits, inference or worker launch and do not qualify native
freshness; the full signed container qualification exercises those owners.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_successor_dispatch_admission as joined
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as public
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLane, ResourceSchedulerConfig,
)
from test.api.test_codebase_successor_dispatch_context import authored_successor_context


@pytest.fixture
def reservation_case(tmp_path, monkeypatch):
    # Keep the real host sampler, safety settings and admission backoff. This
    # small genuine lease supports an inert protocol scope, not a native run.
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json", lane_reservations={}, auto_renew_leases=False))
    cancel = threading.Event()
    lease = scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=64,
                              child_process_slots=1, timeout=3)
    context = authored_successor_context()
    refs = context["inventory"]["scan"]
    root = resume.CodebaseScanResumeRoot.from_dict(refs["root_cid"], refs["root_record"])
    completion = resume.CodebaseScanResumeCompletion.from_dict(refs["completion_cid"], refs["completion_record"])
    selection = successor.CodebaseSuccessorScanRecord.from_dict(
        context["selection"]["artifact_cid"], context["selection"]["value"])
    repository, output = tmp_path / "repository", tmp_path / "output"
    repository.mkdir()
    output.mkdir(mode=0o700)
    owner = execution._Owner(root, completion, object(), repository, object(), None, None,
                             cancel, 120, 1024, selection)
    identity, route = {"owner": "original"}, {"route": "original"}
    server = SimpleNamespace(_lock=threading.RLock(), _connection=object(),
                             identity=SimpleNamespace(to_dict=lambda: deepcopy(identity)))
    source = SimpleNamespace(execution_route_policy=SimpleNamespace(to_dict=lambda: deepcopy(route)))
    public_path = repository / public.DIRECTORY / "inert.json"
    public_path.parent.mkdir(parents=True, mode=0o755)
    public_raw = b"owner-pinned-inert-public-instruction"
    public_path.write_bytes(public_raw)
    public_path.chmod(0o444)
    descriptor = {"artifact": str(public_path), "sha256": hashlib.sha256(public_raw).hexdigest(),
        "task_cid": "selected", "context_cid": "inert-context", "manifest_cid": "inert-manifest",
        "source_path": "keep.py", "source_sha256": "ab" * 32, "source_bytes": 1,
        "completion_authority": False, "scope_expansion_authority": False,
        "inventory_context_cid": "inert-inventory", "codebase_inventory_context_cid": "inert-full-inventory",
        "codebase_successor_context_cid": "inert-successor", "successor_selection_cid": selection.artifact_cid,
        "source_delta_cid": context["source_delta"]["artifact_cid"]}
    admission = {"manifest": {"payload": {"schema": local.SUCCESSOR_MANIFEST_SCHEMA}},
                 "graph": {"all_tasks": ["prerequisite", "selected"]}, "receipt": {"all_pending": [1, 2]}}
    candidate = {"public_instruction": descriptor, "task_revision": 3, "argv": ["inert"]}
    scope = execution.FrozenInventoryExecutionScope(execution._SEAL, owner=owner, admission=admission,
        candidate=candidate, server=server, source=source, lease=lease, output=output)
    observed = {"head": {"generation": 2}, "all_tasks": ["prerequisite", "selected"],
                "current_facts": [], "removed_task_cids": []}
    population = {"tasks": [{"task_cid": "selected", "plan_cid": "inert-plan", "revision": 3}],
                  "completion_rows": {"prerequisite": ["genuine-binding-protocol-control"]},
                  "authority_rows": {"plans": []}, "selected_task_cids": ["selected"],
                  "owner_identity": deepcopy(identity), "execution_route_policy": deepcopy(route)}
    binding = dict(deepcopy(candidate), implementation_command="inert literal worker command")
    payload = {"current_inventory": deepcopy(observed), "native_population": deepcopy(population),
               "candidate": deepcopy(binding), "implementation": execution._pins()}
    scope._material = canonical_dag_json_bytes({"payload": payload, "binding": {"inert": True}})
    launcher = tmp_path / "inert-launcher"
    launcher.write_bytes(b"original launcher")
    scope._launcher = {"path": str(launcher), "sha256": hashlib.sha256(launcher.read_bytes()).hexdigest()}
    physical = {key: deepcopy(population[key]) for key in ("tasks", "completion_rows", "authority_rows")}
    native = {name: "unchanged" for name in ("source", "model", "registry", "producer", "capture")}
    calls = []
    control = SimpleNamespace(callback=None, close_hook=None, exit_hook=None)
    case = SimpleNamespace(scope=scope, scheduler=scheduler, cancel=cancel, observed=observed,
        population=population, binding=binding, identity=identity, route=route, physical=physical, native=native,
        calls=calls, control=control, artifact=output / "execution-scope.json", launcher=launcher,
        public_path=public_path)

    def initial_current(**arguments):
        assert arguments["selection"] is selection and arguments["root"] is owner.root
        assert arguments["completion"] is completion and arguments["index"] is owner.index
        assert arguments["registry"] is owner.registry and arguments["parent_lease"] is lease
        assert arguments["timeout_seconds"] == 120 and arguments["memory_mb"] == 1024
        calls.append("public-full-pair")
        return deepcopy(observed)

    @contextmanager
    def receiver(**arguments):
        assert arguments["selection"] is selection and arguments["root"] is owner.root
        assert arguments["completion"] is completion and arguments["index"] is owner.index
        assert arguments["registry"] is owner.registry and arguments["parent_lease"] is lease
        assert arguments["cancel_event"] is cancel and not lease.released
        assert arguments["timeout_seconds"] == 120 and arguments["memory_mb"] == 1024
        calls.append("paired-entry")
        before, used = deepcopy(native), False

        def close():
            nonlocal used
            assert not used and not lease.released
            used = True
            calls.append("paired-close")
            if scope._reservation_preparation_used:
                assert case.artifact.read_bytes() == scope._material
                assert case.artifact.stat().st_mode & 0o777 == 0o400
            if control.close_hook is not None:
                control.close_hook(case)
            if native != before:
                raise ValueError("inert closing source/model/registry/producer/capture fence changed")
            if cancel.is_set():
                raise ValueError("inert closing cancellation fence")
            return deepcopy(observed)

        try:
            yield deepcopy(observed), close
            assert used
            calls.append("paired-resource-exit")
            if control.exit_hook is not None:
                control.exit_hook(case)
        finally:
            calls.append("paired-dispose")

    def native_population(*arguments):
        calls.append("full-native-contract-callback")
        return deepcopy(population)

    def candidate_binding(*arguments):
        calls.append("full-public-graph-callback")
        return deepcopy(binding)

    def verify_admission(*arguments, **keywords):
        assert keywords == {"initial": True}
        calls.append("full-signed-graph-callback")
        if control.callback is not None:
            control.callback(case)
        return {"profile": object()}

    def signature(envelope, profile):
        calls.append("scope-signature-callback")
        return envelope["payload"]

    monkeypatch.setattr(joined, "verify_current_successor_admission", initial_current)
    monkeypatch.setattr(joined, "_current_successor_admission_operation", receiver)
    monkeypatch.setattr(execution, "_native_population", native_population)
    monkeypatch.setattr(execution, "_candidate", candidate_binding)
    monkeypatch.setattr(execution, "_physical_native", lambda connection: deepcopy(physical))
    monkeypatch.setattr(local, "verify_local_benchmark_admission", verify_admission)
    monkeypatch.setattr(local, "_verify_signature", signature)
    monkeypatch.setattr(candidate_execution, "_root_file",
                        lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    with execution._LOCK:
        execution._ACTIVE[lease.lease_id] = scope
    try:
        yield case
    finally:
        scope._runtime = None
        scope._finish()
        scope._renew_thread.join(timeout=2)
        assert scheduler.snapshot()["active_lease_count"] == scheduler.snapshot()["waiting_request_count"] == 0


def prepare(case):
    # The production reservation performs this unchanged initial full receive
    # before signing its material and preparing the on-disk scope artifact.
    assert case.scope._current_inventory() == case.observed
    case.scope._prepare_reservation_artifact(case.artifact)


def test_default_reservation_uses_two_full_pairs_and_closes_after_both_graph_callbacks(reservation_case):
    case = reservation_case
    prepare(case)
    assert case.calls.count("public-full-pair") == 1
    assert case.calls.count("paired-entry") == case.calls.count("paired-close") == 1
    for callback in ("full-native-contract-callback", "full-public-graph-callback",
                     "full-signed-graph-callback", "scope-signature-callback"):
        assert case.calls.count(callback) == 2
    assert case.calls.index("paired-close") > max(
        index for index, name in enumerate(case.calls) if name.endswith("callback"))
    assert case.artifact.read_bytes() == case.scope._material
    assert case.scope._receiving is None and case.scope._runtime is None
    assert case.scope._reservation_receiving_used and not case.scope._spawned
    assert not case.scope.parent_lease.released


def test_optout_retains_all_five_public_full_pairs(reservation_case):
    case = reservation_case
    value = case.scope._owner.root.to_dict()
    value["optimized"] = False
    object.__setattr__(case.scope._owner, "root",
                       resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(value), value))
    prepare(case)
    assert case.calls.count("public-full-pair") == 5
    assert "paired-entry" not in case.calls and not case.scope._reservation_receiving_used
    assert case.calls.count("full-signed-graph-callback") == 2


@pytest.mark.parametrize("part", ["source", "model", "registry", "producer", "capture"])
def test_mandatory_full_close_detects_drift_after_artifact_write(reservation_case, part):
    case = reservation_case

    def mutate_after_write(current):
        if current.artifact.exists():
            current.native[part] = "changed during final signed graph callback"

    case.control.callback = mutate_after_write
    with pytest.raises(ValueError, match="closing source/model"):
        prepare(case)
    assert case.calls.count("paired-close") == 1 and case.scope._receiving is None
    assert not case.scope._spawned


def test_cancellation_after_artifact_write_refuses_before_reservation_yield(reservation_case):
    case = reservation_case
    case.control.callback = lambda current: current.cancel.set() if current.artifact.exists() else None
    with pytest.raises(execution.InventoryExecutionError, match="cancelled"):
        prepare(case)
    assert case.artifact.exists() and "paired-close" not in case.calls and not case.scope._spawned


@pytest.mark.parametrize("phase", ["close", "resource_exit"])
@pytest.mark.parametrize("part", ["native_task", "native_completion", "native_authority", "public_file", "launcher",
                                  "scope_file", "scope_mode", "scope_symlink", "scope_hardlink"])
def test_late_native_and_file_mutations_refuse_after_callbacks(reservation_case, phase, part):
    case = reservation_case

    def mutate(current):
        if part == "native_task":
            current.physical["tasks"][0]["revision"] += 1
        elif part == "native_completion":
            current.physical["completion_rows"]["prerequisite"].append("foreign")
        elif part == "native_authority":
            current.physical["authority_rows"]["plans"].append(["foreign"])
        elif part == "public_file":
            current.public_path.chmod(0o644)
            current.public_path.write_bytes(b"changed public bytes")
            current.public_path.chmod(0o444)
        elif part == "launcher":
            current.launcher.write_bytes(b"changed launcher")
        elif part == "scope_file":
            current.artifact.chmod(0o600)
            current.artifact.write_bytes(b"changed scope bytes")
            current.artifact.chmod(0o400)
        elif part == "scope_mode":
            current.artifact.chmod(0o600)
        elif part == "scope_symlink":
            other = current.artifact.with_name("other.json")
            current.artifact.rename(other)
            current.artifact.symlink_to(other)
        else:
            os.link(current.artifact, current.artifact.with_name("alias.json"))

    setattr(case.control, "close_hook" if phase == "close" else "exit_hook", mutate)
    with pytest.raises((execution.InventoryExecutionError, OSError)):
        prepare(case)
    assert "paired-close" in case.calls and case.scope._receiving is None and not case.scope._spawned


@pytest.mark.parametrize("part", ["admission", "candidate", "material", "owner", "connection", "identity", "route",
                                  "seal", "pid", "thread", "output", "native_admission", "runtime"])
def test_sealed_reservation_inputs_cannot_change_during_graph_callbacks(reservation_case, part):
    case = reservation_case

    def mutate(current):
        scope = current.scope
        if part == "admission":
            scope._admission["graph"]["all_tasks"].pop()
        elif part == "candidate":
            scope._candidate_input["task_revision"] += 1
        elif part == "material":
            scope._material += b" "
        elif part == "owner":
            object.__setattr__(scope._owner, "index", object())
        elif part == "connection":
            scope._server._connection = object()
        elif part == "identity":
            current.identity["owner"] = "foreign"
        elif part == "route":
            current.route["route"] = "foreign"
        elif part in {"seal", "pid", "thread"}:
            setattr(scope._receiving, {"seal": "seal", "pid": "pid", "thread": "thread_id"}[part],
                    object() if part == "seal" else -1)
        elif part == "output":
            scope._output = scope._output.with_name("foreign-output")
        elif part == "native_admission":
            scope._native_admission = {"foreign": True}
        else:
            scope._runtime = object()

    case.control.callback = mutate
    with pytest.raises((execution.InventoryExecutionError, OSError)):
        prepare(case)
    assert case.scope._receiving is None and not case.scope._spawned


@pytest.mark.parametrize("part", ["admission", "candidate", "material", "owner", "owner_object", "connection",
                                  "identity", "route", "output", "cancel", "runtime", "native_admission",
                                  "native_admission_bytes", "native_admission_ref", "seal", "pid", "thread",
                                  "operation_runtime", "operation_native_admission", "nested_exit",
                                  "reservation_reuse", "preparation_reuse"])
def test_resource_exit_cannot_replace_reservation_bindings(reservation_case, part):
    case = reservation_case

    def mutate(current):
        scope = current.scope
        if part == "admission":
            scope._admission["receipt"]["all_pending"].pop()
        elif part == "candidate":
            scope._candidate_input["task_revision"] += 1
        elif part == "material":
            scope._material += b" "
        elif part == "owner":
            object.__setattr__(scope._owner, "registry", object())
        elif part == "owner_object":
            scope._owner = replace(scope._owner)
        elif part == "connection":
            scope._server._connection = object()
        elif part == "identity":
            current.identity["owner"] = "foreign"
        elif part == "route":
            current.route["route"] = "foreign"
        elif part == "output":
            scope._output = scope._output.with_name("foreign-output")
        elif part == "runtime":
            scope._runtime = object()
        elif part == "native_admission":
            scope._native_admission = {"foreign": True}
        elif part == "native_admission_bytes":
            scope._native_admission_bytes = b"foreign"
        elif part == "native_admission_ref":
            scope._native_admission_ref = object()
        elif part in {"seal", "pid", "thread"}:
            setattr(current.closed_operation, {"seal": "seal", "pid": "pid", "thread": "thread_id"}[part],
                    object() if part == "seal" else -1)
        elif part == "operation_runtime":
            current.closed_operation.runtime = object()
        elif part == "operation_native_admission":
            current.closed_operation.native_admission = object()
        elif part == "nested_exit":
            with scope._receiving_operation(purpose="create"):
                pytest.fail("nested resource-exit receiver entered")
        elif part == "reservation_reuse":
            scope._reservation_receiving_used = False
        elif part == "preparation_reuse":
            scope._reservation_preparation_used = False
        else:
            current.cancel.set()

    case.control.close_hook = lambda current: setattr(current, "closed_operation", current.scope._receiving)
    case.control.exit_hook = mutate
    with pytest.raises(execution.InventoryExecutionError):
        prepare(case)
    assert case.scope._receiving is None and not case.scope._spawned


@pytest.mark.parametrize("phase", ["close", "resource_exit"])
def test_late_runtime_producer_change_refuses_after_full_close(reservation_case, monkeypatch, phase):
    case = reservation_case
    pins = execution._pins()
    setattr(case.control, "close_hook" if phase == "close" else "exit_hook",
            lambda current: monkeypatch.setattr(execution, "_pins", lambda: {**pins, "late": "changed"}))
    with pytest.raises(execution.InventoryExecutionError, match="producer bytes"):
        prepare(case)
    assert "paired-close" in case.calls and not case.scope._spawned


@pytest.mark.parametrize("failure", ["unused", "double_close", "nested", "thread", "released"])
def test_reservation_receiver_is_owned_consumed_once_and_not_nested(reservation_case, failure):
    case, scope = reservation_case, reservation_case.scope
    with pytest.raises(execution.InventoryExecutionError):
        with scope._receiving_operation(purpose="reserve") as operation:
            if failure == "unused":
                pass
            elif failure == "double_close":
                scope._close_receiving_operation(operation)
                scope._close_receiving_operation(operation)
            elif failure == "nested":
                with scope._receiving_operation(purpose="create"):
                    pytest.fail("nested reservation receiver entered")
            elif failure == "released":
                scope.parent_lease.release()
                scope._close_receiving_operation(operation)
            else:
                errors = []

                def foreign_thread():
                    try:
                        scope.require_prelaunch_current()
                    except execution.InventoryExecutionError as error:
                        errors.append(error)

                thread = threading.Thread(target=foreign_thread)
                thread.start()
                thread.join(timeout=2)
                assert not thread.is_alive() and len(errors) == 1
                raise errors[0]
    assert scope._receiving is None and not scope._spawned


def test_successful_reservation_preparation_and_receiver_cannot_be_reused(reservation_case):
    case = reservation_case
    prepare(case)
    with pytest.raises(execution.InventoryExecutionError, match="preparation cannot be reused"):
        case.scope._prepare_reservation_artifact(case.artifact)
    with pytest.raises(execution.InventoryExecutionError, match="already used"):
        with case.scope._receiving_operation(purpose="reserve"):
            pytest.fail("reservation receiver reused")
    # Construction still needs its own fresh receiver; no reservation result
    # is treated as a reusable model/source currentness token.
    with case.scope._receiving_operation(purpose="create") as operation:
        case.scope._close_receiving_operation(operation)
    assert case.calls.count("paired-entry") == case.calls.count("paired-close") == 2


def test_exception_after_scope_write_never_consumes_or_replays_closing_receiver(reservation_case):
    case = reservation_case

    def fail_after_write(current):
        if current.artifact.exists():
            raise RuntimeError("inert graph callback failed after write")

    case.control.callback = fail_after_write
    with pytest.raises(RuntimeError, match="after write"):
        prepare(case)
    assert case.artifact.exists() and "paired-close" not in case.calls
    assert case.scope._receiving is None and not case.scope._spawned
