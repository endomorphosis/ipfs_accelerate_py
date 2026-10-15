"""Signed full-task admission custody through a successful resource exit.

The signer, graph and pending task replay are genuine. The current-owner scope
is an explicit protocol double; these controls establish no source/model
freshness, scheduling, numerical execution or worker-launch authority.
"""
from contextlib import contextmanager
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    codebase_inventory_evidence_admission as inventory,
    codebase_inventory_evidence_worker_context as inventory_context,
    codebase_successor_dispatch_admission as successor,
    codebase_successor_dispatch_context as successor_context,
    local_planning_admission as local,
)
from test.api.test_codebase_inventory_evidence_admission import inventory_case  # noqa: F401
from test.api.test_codebase_successor_dispatch_admission import successor_case  # noqa: F401


@pytest.fixture(params=("inventory", "successor"))
def operation_case(request, monkeypatch):
    route = request.param
    case = request.getfixturevalue(route + "_case")
    module = inventory if route == "inventory" else successor
    if route == "inventory":
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume

        scan = case["context"]["scan"]
        arguments = {
            "root": resume.CodebaseScanResumeRoot.from_dict(scan["root_cid"], scan["root_record"]),
            "completion": resume.CodebaseScanResumeCompletion.from_dict(
                scan["completion_cid"], scan["completion_record"]),
            "index": object(), "registry": object(), "repository": case["repository"],
        }
        declaration = inventory_context.inventory_declaration(case["context"])
        operation = module._current_inventory_admission_operation
        verify = module.verify_current_inventory_admission
    else:
        arguments = {name: case[name] for name in (
            "selection", "root", "completion", "index", "registry", "repository")}
        declaration = successor_context.successor_declaration(case["context"])
        operation = module._current_successor_admission_operation
        verify = module.verify_current_successor_admission
    state = {"close_callback": None, "exit_callback": None, "close_error": None,
             "observations": [], "detached_admissions": [], "receipts": []}

    @contextmanager
    def controlled_owner_scope(**options):
        assert all(options[name] is value for name, value in arguments.items())
        state["observations"].append("entry")

        def current():
            state["observations"].append("close")
            if state["close_error"] is not None:
                raise state["close_error"]
            if state["close_callback"] is not None:
                state["close_callback"]()

        if route == "inventory":
            yield deepcopy(declaration), current
        else:
            yield deepcopy(case["context"]), deepcopy(declaration), current
        state["observations"].append("resource_exit")
        if state["exit_callback"] is not None:
            state["exit_callback"]()

    original_verify = local.verify_local_benchmark_admission

    def observed_native_verification(admission, **options):
        result = original_verify(admission, **options)
        state["detached_admissions"].append(admission)
        return result

    original_wire = module._wire
    receipt_schema = "supervisor-current-" + route + "-admission-verification@1"

    def observed_wire(value, **options):
        if type(value) is dict and value.get("schema") == receipt_schema:
            if not state["receipts"] or state["receipts"][-1] is not value:
                state["receipts"].append(value)
        return original_wire(value, **options)

    monkeypatch.setattr(module, "_current_scope", controlled_owner_scope)
    monkeypatch.setattr(module, "_wire", observed_wire)
    monkeypatch.setattr(local, "verify_local_benchmark_admission", observed_native_verification)
    return {"case": case, "state": state, "arguments": {**arguments, "admission": case["admission"]},
            "operation": operation, "verify": verify}


def _mutate_target(fixture, target):
    state = fixture["state"]
    if target == "receipt":
        state["receipts"][-1]["administrator_task_cids"].pop()
    else:
        state["detached_admissions"][-1]["graph"]["tasks"].pop()


def test_complete_signed_task_population_survives_close_and_resource_exit(operation_case):
    fixture = operation_case
    caller = deepcopy(fixture["case"]["admission"])
    with fixture["operation"](**fixture["arguments"]) as (receipt, close):
        assert close() is receipt
        expected = sorted(task.task_cid for task in fixture["case"]["tasks"])
        assert receipt["administrator_task_cids"] == expected
        assert receipt["current_facts"] == receipt["removed_task_cids"] == []
        assert receipt["runtime_requirements_preserved"] is True
        assert all(value is False for value in receipt["authority"].values())
    assert fixture["state"]["observations"] == ["entry", "close", "resource_exit"]
    assert fixture["case"]["admission"] == caller


@pytest.mark.parametrize("phase", ("inside_closing", "after_closing", "resource_exit"))
@pytest.mark.parametrize("target", ("receipt", "detached_admission"))
def test_late_receipt_or_signed_population_mutation_refuses_success(operation_case, phase, target):
    fixture = operation_case
    caller = deepcopy(fixture["case"]["admission"])
    mutate = lambda: _mutate_target(fixture, target)
    if phase == "inside_closing":
        fixture["state"]["close_callback"] = mutate
    elif phase == "resource_exit":
        fixture["state"]["exit_callback"] = mutate
    with pytest.raises(ValueError, match="changed"):
        with fixture["operation"](**fixture["arguments"]) as (_, close):
            close()
            if phase == "after_closing":
                mutate()
    assert fixture["case"]["admission"] == caller


@pytest.mark.parametrize("target", ("receipt", "detached_admission"))
def test_public_verification_withholds_late_resource_exit_mutation(operation_case, target):
    fixture = operation_case
    fixture["state"]["exit_callback"] = lambda: _mutate_target(fixture, target)
    with pytest.raises(ValueError, match="resource closure"):
        fixture["verify"](**fixture["arguments"])
    assert fixture["state"]["observations"] == ["entry", "close", "resource_exit"]


def test_private_operation_cannot_succeed_without_closing(operation_case):
    fixture = operation_case
    with pytest.raises(ValueError, match="requires a successful closing fence"):
        with fixture["operation"](**fixture["arguments"]):
            pass
    assert fixture["state"]["observations"] == ["entry"]


def test_private_operation_closing_callback_is_one_use(operation_case):
    fixture = operation_case
    with fixture["operation"](**fixture["arguments"]) as (receipt, close):
        assert close() is receipt
        with pytest.raises(ValueError, match="already closed"):
            close()
    assert fixture["state"]["observations"] == ["entry", "close", "resource_exit"]


def test_failed_body_preserves_primary_error_without_success_fence(operation_case):
    fixture = operation_case
    error = RuntimeError("controlled operation body failure")
    with pytest.raises(RuntimeError) as caught:
        with fixture["operation"](**fixture["arguments"]):
            raise error
    assert caught.value is error
    assert fixture["state"]["observations"] == ["entry"]


def test_failed_current_owner_close_preserves_primary_error(operation_case):
    fixture = operation_case
    error = RuntimeError("controlled current owner close failure")
    fixture["state"]["close_error"] = error
    with pytest.raises(RuntimeError) as caught:
        with fixture["operation"](**fixture["arguments"]) as (_, close):
            close()
    assert caught.value is error
    assert fixture["state"]["observations"] == ["entry", "close"]


def test_private_receipt_is_detached_from_caller_admission(operation_case):
    fixture = operation_case
    caller = fixture["case"]["admission"]
    original = deepcopy(caller)
    with fixture["operation"](**fixture["arguments"]) as (_, close):
        caller["graph"]["tasks"].pop()
        assert len(fixture["state"]["detached_admissions"][-1]["graph"]["tasks"]) == 2
        close()
    caller.clear()
    caller.update(original)
    assert fixture["state"]["observations"] == ["entry", "close", "resource_exit"]
