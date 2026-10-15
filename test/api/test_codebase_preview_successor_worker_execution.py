"""Controlled @7 launch boundaries with genuine signatures and a small lease.

The signed child admission, preview inputs, public artifact and shared 64 MB
resource lease are genuine. Native task population, detached row observations
and source/model currentness use explicit fixture doubles. These controls
exercise custody and callback ordering; they launch no worker, qualify no
numerical head and establish no full native worker or production qualification.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as runtime_module
from ipfs_accelerate_py.agent_supervisor.runtime import (
    codebase_inventory_execution as execution,
    codebase_preview_successor_worker_admission as joined,
    codebase_successor_dispatch_context as successor_context,
    local_planning_admission as local,
    router_public_instruction as public,
)
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes
from test.api.test_codebase_inventory_execution import leased_scope, runtime_case  # noqa: F401
from test.api.test_codebase_preview_successor_worker_admission import (
    _current_arguments, _prepare, _verify,
    bridge_case, prohibit_private_source_model_and_numerical_work, successor_case, worker_case,  # noqa: F401
)
from test.api.test_codebase_preview_successor_bridge import _different_root
from test.api.test_plan_create_semantic_input_identity import _cid


@pytest.fixture
def execution_case(worker_case, leased_scope, monkeypatch):
    fixture = worker_case
    scope, cancel, scheduler = leased_scope
    case = fixture["case"]
    owner = execution._Owner(case["root"], case["completion"], case["index"], case["repository"],
        case["registry"], None, None, cancel, 120, 1024, case["selection"], fixture["bridge_inputs"])
    scope._owner = owner
    scope._preview_bridge_inputs_capture = joined.capture_preview_successor_bridge_inputs(owner.preview_bridge_inputs)
    scope._admission = deepcopy(fixture["worker_admission"])
    descriptor = _prepare(fixture)
    selected = case["tasks"][0].task_cid
    population = {"selected_task_cids": [selected],
        "tasks": [{"task_cid": task.task_cid, "task_alias": task.task_key, "revision": 1}
                  for task in fixture["graph"].tasks],
        "completion_rows": {}, "authority_rows": {"plans": []},
        "owner_identity": scope._server.identity.to_dict(),
        "execution_route_policy": scope._source.execution_route_policy.to_dict()}
    scope._candidate_input = {"public_instruction": descriptor, "argv": [
        "/opt/ipfs-supervisor/bin/owner-worker",
        "--public-instruction-artifact", descriptor["artifact"],
        "--public-instruction-sha256", descriptor["sha256"],
        "--public-instruction-task-cid", selected], "task_revision": 1}
    candidate = execution._candidate(scope._candidate_input, scope._admission, population)
    observed = _verify(fixture)
    payload = {"schema": execution.PREVIEW_SUCCESSOR_SCHEMA,
        "profile": execution.PREVIEW_SUCCESSOR_PROFILE, "implementation": {},
        "native_population": population, "candidate": candidate, "current_inventory": observed}
    scope._material = canonical_dag_json_bytes(local._signed(payload, fixture["worker_manifest"]["payload"]))
    events = []

    def population_double(*arguments):
        assert not scope._server._lock._is_owned()
        events.append("population")
        return deepcopy(population)

    def detached_double():
        events.append("detached")

    monkeypatch.setattr(execution, "_native_population", population_double)
    monkeypatch.setattr(execution, "_pins", lambda: {})
    monkeypatch.setattr(scope, "_detached_fence", detached_double)
    # Replace only this module's namespace; Git's genuine subprocess runner
    # remains available for independent signed admission replay.
    monkeypatch.setattr(runtime_module, "subprocess", SimpleNamespace(
        Popen=lambda *args, **kwargs: pytest.fail("controlled preview worker test attempted Popen"),
        STDOUT=-2))
    fixture["case"]["observations"].clear()
    fixture["state"]["events"].clear()
    return {"fixture": fixture, "scope": scope, "events": events,
            "population": population, "candidate": candidate, "scheduler": scheduler}


def _mutate(fixture, target):
    if target == "root":
        fixture["state"]["roots"] = _different_root(fixture["state"]["roots"], "policy_root")
    elif target == "checker":
        fixture["state"]["checker"] = _cid("controlled-worker-checker-drift")
    elif target == "material":
        fixture["materials"].extra["authored_metadata"] = "changed after callbacks"
    else:
        raise AssertionError(target)


def _runtime(case):
    return runtime_case(case["scope"], cleaned=True)


def test_preview_profile_always_uses_paired_receiving_even_when_inference_is_opted_out(execution_case):
    scope = execution_case["scope"]
    scope._owner = replace(scope._owner, root=SimpleNamespace(to_dict=lambda: {"optimized": False}))
    assert scope._uses_paired_receiving() is True


def test_scope_current_check_reobserves_the_signed_companion_and_full_population(execution_case):
    case = execution_case
    observed = case["scope"]._current_inventory()
    assert observed == case["scope"].to_dict()["payload"]["current_inventory"]
    assert observed["administrator_task_cids"] == sorted(task.task_cid for task in case["fixture"]["graph"].tasks)
    assert observed["worker_profile_integrated"] is True
    assert observed["current_facts"] == observed["removed_task_cids"] == []
    assert all(value is False for value in observed["authority"].values())


def test_constructor_prepares_callbacks_unlocked_then_consumes_native_closing_fence(execution_case):
    case, scope = execution_case, execution_case["scope"]
    with scope._receiving_operation(purpose="create") as operation:
        runtime = _runtime(case)
        operation.runtime = runtime
        scope._finish_creation_receiving(runtime)
        assert operation.closed
        events = list(case["fixture"]["state"]["events"])
        closes = case["fixture"]["case"]["observations"].count("close")
    assert case["fixture"]["state"]["events"] == events
    assert case["fixture"]["case"]["observations"].count("close") == closes
    assert not scope._spawned


def test_reservation_writes_scope_before_unlocked_preparation_and_locked_native_close(execution_case, monkeypatch):
    case, scope = execution_case, execution_case["scope"]
    real_prepare = joined._PreviewSuccessorWorkerClosingFence.prepare
    real_close = joined._PreviewSuccessorWorkerClosingFence.__call__
    ordering = []

    def prepare(fence):
        assert not scope._server._lock._is_owned()
        assert (scope._output / "execution-scope.json").read_bytes() == scope._material
        ordering.append("prepare")
        return real_prepare(fence)

    def close(fence):
        assert scope._server._lock._is_owned()
        ordering.append("close")
        return real_close(fence)

    monkeypatch.setattr(joined._PreviewSuccessorWorkerClosingFence, "prepare", prepare)
    monkeypatch.setattr(joined._PreviewSuccessorWorkerClosingFence, "__call__", close)
    scope._prepare_reservation_artifact(scope._output / "execution-scope.json")
    assert ordering == ["prepare", "close"]
    assert scope._reservation_receiving_used and scope._reservation_preparation_used
    assert not scope._spawned and not scope.parent_lease.released


def test_start_preparation_finishes_all_callbacks_before_locked_native_close(execution_case, monkeypatch):
    case, scope = execution_case, execution_case["scope"]
    runtime = _runtime(case)
    fixture = case["fixture"]
    fixture["state"]["roots_callback"] = lambda: pytest.fail("roots observer ran under lock") if (
        scope._server._lock._is_owned()) else None
    fixture["state"]["checker_callback"] = lambda: pytest.fail("checker observer ran under lock") if (
        scope._server._lock._is_owned()) else None
    with scope._receiving_operation(purpose="start", runtime=runtime) as operation:
        scope.prepare_spawn_fence(runtime)
        events = list(fixture["state"]["events"])
        closes = fixture["case"]["observations"].count("close")
        with scope._server._lock:
            scope.require_spawn_fence(runtime)
        assert operation.closed
        assert fixture["state"]["events"] == events
        assert fixture["case"]["observations"].count("close") == closes + 1
    assert fixture["state"]["events"] == events
    assert not scope._spawned and scope._spawn_preparation is None


@pytest.mark.parametrize("target", ("root", "checker", "material"))
def test_drift_after_unlocked_native_population_callbacks_refuses_start_preparation(execution_case, monkeypatch, target):
    case, scope = execution_case, execution_case["scope"]
    runtime = _runtime(case)
    with pytest.raises(ValueError):
        with scope._receiving_operation(purpose="start", runtime=runtime):
            def mutate_after_population(*arguments):
                _mutate(case["fixture"], target)
                return deepcopy(case["population"])
            monkeypatch.setattr(execution, "_native_population", mutate_after_population)
            scope.prepare_spawn_fence(runtime)
    assert not scope._spawned and scope._spawn_preparation is None
    assert not scope.parent_lease.released


def test_source_edit_after_unlocked_preparation_refuses_the_final_native_start_close(execution_case):
    case, scope = execution_case, execution_case["scope"]
    fixture = case["fixture"]
    runtime = _runtime(case)
    with pytest.raises(ValueError):
        with scope._receiving_operation(purpose="start", runtime=runtime):
            scope.prepare_spawn_fence(runtime)
            (fixture["case"]["repository"] / "input.py").write_bytes(b"VALUE = 8765\n")

            def independent_source_close():
                declared = fixture["manifest"]["payload"]["sources"]
                if local._sources(fixture["case"]["repository"], sorted(declared)) != declared:
                    raise ValueError("controlled final source fence detected edit")

            fixture["case"]["close_callback"] = independent_source_close
            with scope._server._lock:
                scope.require_spawn_fence(runtime)
    assert not scope._spawned and not scope.parent_lease.released


def test_material_edit_between_preparation_and_native_close_refuses_start(execution_case):
    case, scope = execution_case, execution_case["scope"]
    runtime = _runtime(case)
    with pytest.raises(ValueError):
        with scope._receiving_operation(purpose="start", runtime=runtime):
            scope.prepare_spawn_fence(runtime)
            _mutate(case["fixture"], "material")
            with scope._server._lock:
                scope.require_spawn_fence(runtime)
    assert not scope._spawned and scope._spawn_preparation is None


def test_replacing_private_capture_cannot_hide_mutated_preview_material_during_operation(execution_case):
    case, scope = execution_case, execution_case["scope"]
    with pytest.raises(execution.InventoryExecutionError, match="inputs changed"):
        with scope._receiving_operation(purpose="create") as operation:
            _mutate(case["fixture"], "material")
            scope._preview_bridge_inputs_capture = joined.capture_preview_successor_bridge_inputs(
                scope._owner.preview_bridge_inputs)
            scope._operation_guard(operation)
    assert not scope._spawned


def test_close_preparation_requires_unlocked_owner_and_cannot_be_reused(execution_case):
    scope = execution_case["scope"]
    with scope._receiving_operation(purpose="create") as operation:
        with scope._server._lock:
            with pytest.raises(execution.InventoryExecutionError, match="precede the native owner lock"):
                scope._prepare_receiving_close(operation)
        scope._prepare_receiving_close(operation)
        with pytest.raises(ValueError, match="already consumed"):
            scope._prepare_receiving_close(operation)
        with scope._server._lock:
            scope._close_receiving_operation(operation)
        with pytest.raises(execution.InventoryExecutionError, match="already closed"):
            scope._close_receiving_operation(operation)


def test_callback_cannot_rebind_private_closing_functions_to_skip_native_close(execution_case):
    scope = execution_case["scope"]
    with scope._receiving_operation(purpose="create") as operation:
        with pytest.raises(ValueError, match="cannot be rebound"):
            operation.close_current._prepare = lambda: None
        with pytest.raises(ValueError, match="cannot be rebound"):
            operation.close_current._close = lambda: operation.observed
        scope._prepare_receiving_close(operation)
        with scope._server._lock:
            scope._close_receiving_operation(operation)
        assert operation.closed


def test_final_start_close_refuses_a_missing_unlocked_preparation(execution_case):
    scope = execution_case["scope"]
    runtime = _runtime(execution_case)
    with pytest.raises(ValueError, match="closing preparation"):
        with scope._receiving_operation(purpose="start", runtime=runtime) as operation:
            with scope._server._lock:
                scope._close_receiving_operation(operation)
    assert not scope._spawned


@pytest.mark.parametrize("operation", ("reserve", "create", "start"))
def test_each_boundary_requires_a_successful_closing_fence(execution_case, operation):
    scope = execution_case["scope"]
    runtime = _runtime(execution_case) if operation == "start" else None
    with pytest.raises(ValueError, match="closing fence"):
        with scope._receiving_operation(purpose=operation, runtime=runtime):
            pass
    assert not scope._spawned


def test_constructor_boundary_reobserves_root_drift_after_previous_reservation(execution_case):
    case, scope = execution_case, execution_case["scope"]
    scope._prepare_reservation_artifact(scope._output / "execution-scope.json")
    _mutate(case["fixture"], "root")
    with pytest.raises(ValueError):
        with scope._receiving_operation(purpose="create"):
            pytest.fail("stale preview roots reached constructor body")
    assert not scope._spawned


@pytest.mark.parametrize("target", ("missing", "unexpected"))
def test_preview_inputs_profile_mismatch_refuses_before_scheduler_acquisition(worker_case, leased_scope, tmp_path, monkeypatch, target):
    scope, _cancel, scheduler = leased_scope
    admission = worker_case["worker_admission"] if target == "missing" else worker_case["admission"]
    inputs = None if target == "missing" else worker_case["bridge_inputs"]
    monkeypatch.setattr(scheduler, "acquire", lambda *args, **kwargs: pytest.fail("invalid preview profile acquired lease"))
    destination = tmp_path / ("refused-" + target)
    arguments = {name: value for name, value in _current_arguments(worker_case).items() if name != "selection"}
    with pytest.raises(execution.InventoryExecutionError, match="exact live bridge inputs"):
        with execution.reserve_inventory_execution(**arguments, admission=admission,
                preview_bridge_inputs=inputs, successor_selection=worker_case["case"]["selection"],
                candidate={}, server=scope._server, source=scope._source, output=destination, scheduler=scheduler):
            pytest.fail("invalid preview profile entered reservation")
    assert not destination.exists()


def test_invalid_typed_preview_inputs_refuse_before_scheduler_acquisition(worker_case, leased_scope, tmp_path, monkeypatch):
    scope, _cancel, scheduler = leased_scope
    monkeypatch.setattr(scheduler, "acquire", lambda *args, **kwargs: pytest.fail("invalid preview inputs acquired lease"))
    arguments = {name: value for name, value in _current_arguments(worker_case).items() if name != "selection"}
    with pytest.raises(ValueError, match="exact private preview"):
        with execution.reserve_inventory_execution(**arguments, admission=worker_case["worker_admission"],
                preview_bridge_inputs={}, successor_selection=worker_case["case"]["selection"],
                candidate={}, server=scope._server, source=scope._source,
                output=tmp_path / "invalid-inputs", scheduler=scheduler):
            pytest.fail("invalid preview inputs entered reservation")


def test_legacy_worker_builder_and_public_instruction_downgrade_reject_child_admission(worker_case):
    fixture = worker_case
    context = fixture["case"]["context"]
    with pytest.raises(ValueError, match="explicit signed successor manifest"):
        successor_context.build_successor_worker_context(manifest_envelope=fixture["worker_manifest"],
            graph=fixture["worker_admission"]["graph"], receipt=fixture["worker_admission"]["receipt"],
            task_cid=fixture["case"]["tasks"][0].task_cid,
            owner_identity=fixture["worker_manifest"]["binding"]["identity"],
            owner_profile_id=fixture["worker_manifest"]["binding"]["profile_id"],
            inventory_context=context["inventory"], successor_context=context)
    descriptor = _prepare(fixture)
    payload = json.loads(Path(descriptor["artifact"]).read_bytes())
    payload["schema"] = public.SUCCESSOR_SCHEMA
    with pytest.raises(ValueError, match="explicit signed public instruction profile"):
        public._selection(payload)


def test_signed_preview_admission_cannot_take_generic_runtime_path(worker_case, tmp_path):
    destination = tmp_path / "refused-generic-launch"
    with pytest.raises(ValueError, match="exact native inventory execution scope"):
        runtime_module.AdmittedBenchmarkRuntime._create(destination,
            admission=worker_case["worker_admission"], server=object(), source=object())
    assert not destination.exists()


def test_changed_preview_roots_and_cancellation_do_not_prevent_stop_cleanup(execution_case):
    case, scope = execution_case, execution_case["scope"]
    runtime = _runtime(case)
    scope.note_spawned(runtime)
    _mutate(case["fixture"], "root")
    scope._owner.cancel_event.set()
    scope.finish_runtime(runtime)
    scope.require_close(runtime)
    assert scope._cleaned and not scope.parent_lease.released


def test_preview_material_and_owner_capture_are_bound_to_receiving_operation_identity(execution_case):
    case, scope = execution_case, execution_case["scope"]
    capture = scope._preview_bridge_inputs_capture
    encoded = execution._preview_inputs_custody(capture)
    assert encoded["type"] == "tuple"
    assert encoded["items"][0]["items"][0]["value"] == id(scope._owner.preview_bridge_inputs)
    assert encoded["items"][1]["items"][0]["sha256"] == hashlib.sha256(capture[1][0]).hexdigest()
    with pytest.raises(execution.InventoryExecutionError, match="unsupported.*custody"):
        execution._preview_inputs_custody(object())
