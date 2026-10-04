"""Real native reservations and compiler forecasts; no execution permission.

Most cases isolate capacity mechanics using the previously qualified pure
finite service fixture, rooted in a freshly captured native Git repository.
The forward integration case also uses the actual Python/Lean finite matcher.
"""
from copy import deepcopy
from dataclasses import replace
import threading
import time

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_capacity as capacity_module
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_plan_preview as adapter
from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_capacity import (
    FINITE_CAPACITY_PROFILE, CapacityBoundFiniteIntegerPlanCreateService,
    FiniteIntegerCapacityError, reserve_finite_integer_capacity,
)
from ipfs_accelerate_py.agent_supervisor.planning.parallel_plan_compiler import (
    ParallelPlanCompiler, ParallelPlanCompilationRequest,
)
from ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context import structural_codebase_context
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import freeze_plan_create_input_snapshot
from test.api.test_finite_integer_codebase import (
    finite_match, finite_prepared, finite_tools,
)
from test.api.test_finite_integer_plan_preview import preview_arguments
from test.api.test_finite_integer_plan_service import service_case


@pytest.fixture
def capacity_case(finite_prepared, finite_tools):
    arguments = preview_arguments(finite_prepared, finite_tools)
    request, owner = arguments["request"], arguments["owner"]
    _, materials, operations = service_case()
    root = request.roots.dirty_worktree_root
    intent = replace(materials.intent, current_root_id=root)
    facts = tuple(replace(fact, current_root_id=root) for fact in materials.current_facts)
    match = {**materials.extra["finite_match"], "typed_intent": intent.to_dict(),
             "current_facts": [fact.to_dict() for fact in facts], "current_root_id": root}
    with structural_codebase_context(owner.index, owner.repository, repository_id=request.repository_id,
            expected_head=owner.expected_head, scheduler=owner.scheduler, memory_mb=owner.memory_mb) as context:
        context_record, context_cid = context.to_dict(), context.cid
    materials = replace(materials, intent=intent, current_facts=facts,
        frozen_goal=replace(materials.frozen_goal, repository_tree_id=root),
        scan={"scan_cid": context_cid, "structural_codebase": context_record},
        extra={**materials.extra, "finite_match": match, "structural_codebase": context_record,
               "structural_codebase_context_cid": context_cid,
               "root_observation_profile": "repository-live-root-observation@1"})
    return owner, request, materials, operations, arguments


def bound_materials(materials, capacity):
    return replace(materials, extra={**materials.extra, "finite_capacity_profile": FINITE_CAPACITY_PROFILE,
                                    "finite_capacity_binding": capacity.material_binding})


def observer(owner, materials):
    def observe(request):
        with structural_codebase_context(owner.index, owner.repository, repository_id=request.repository_id,
                expected_head=owner.expected_head, scheduler=owner.scheduler if owner.parent_lease is None else None,
                parent_lease=owner.parent_lease,
                memory_mb=owner.memory_mb) as context:
            assert context.to_dict() == materials.extra["structural_codebase"]
        return request.roots
    return observe


def service_for(owner, materials, operations, capacity):
    return CapacityBoundFiniteIntegerPlanCreateService(root_observer=observer(owner, materials),
        operation_bindings=operations, capacity=capacity)


def no_float_or_key(value):
    if type(value) is dict:
        assert "lease_key" not in value
        for child in value.values():
            no_float_or_key(child)
    elif type(value) is list:
        for child in value:
            no_float_or_key(child)
    else:
        assert type(value) in {str, int, bool, type(None)}


def test_native_reserved_forecast_replays_and_keeps_reviewed_meanings(capacity_case):
    owner, request, materials, operations, _ = capacity_case
    original_snapshot = freeze_plan_create_input_snapshot(request, materials=materials)
    with reserve_finite_integer_capacity(owner=owner, request=request, materials=materials) as capacity:
        binding = capacity.material_binding
        assert FINITE_CAPACITY_PROFILE.endswith("@2")
        assert binding["freshness_policy"]["max_age_ms"] == 30_000
        assert binding["freshness_policy_cid"] == cid_for_structured(binding["freshness_policy"])
        assert binding["material_snapshot"] == original_snapshot.to_dict()
        no_float_or_key(binding)
        detached = capacity.material_binding
        detached["reservation"]["memory_mb"] = 1
        assert capacity.material_binding == binding
        bound = bound_materials(materials, capacity)
        service = service_for(owner, bound, operations, capacity)
        receipt = service.preview_create(request, materials=bound)
        stages = {stage.stage.value: stage for stage in receipt.stage_results}
        assert stages["critique"].passed and stages["parallel_plan"].passed
        assert not stages["admission"].passed and receipt.verdict.value == "review_only"
        assert service.execution_plan.admitted and service.execution_plan.resource_feasibility.freshness_proved
        assert service.execution_plan.resource_feasibility.available_host["memory_bytes"] == owner.memory_mb * 1024**2
        replay = ParallelPlanCompiler().replay(service.execution_plan)
        assert replay.plan_id == service.execution_plan.plan_id
        record = service.capacity_compilation_request
        assert record["capacity_snapshot"]["max_age_ms"] == 30_000
        assert record["capacity_snapshot"]["freshness_policy_cid"] == binding["freshness_policy_cid"]
        assert record["capacity_snapshot"]["candidate_plan_cid"] == cid_for_structured(service.candidate_plan)
        assert record["capacity_snapshot"]["bound_material_snapshot_cid"] == receipt.input_snapshot_cid
        assert record["tasks"] == [{**task, "resource_contract": capacity.task_resource_profile}
                                   for task in service.candidate_plan["tasks"]]
        assert all("resource_contract" not in task for task in service.candidate_plan["tasks"])
        assert service.candidate_plan["effects"][0]["operation"] == "update"
        assert binding["execution_authority"] is binding["production_admitted"] is False
        no_float_or_key(record)
        assert service.preview_create(request, materials=bound) == receipt
    assert owner.scheduler.snapshot()["active_lease_count"] == 0
    with pytest.raises(FiniteIntegerCapacityError, match="released"):
        capacity.require_current(request, bound)


def test_actual_finite_python_lean_match_also_has_real_resource_feasibility(capacity_case):
    owner, request, _, _, arguments = capacity_case
    prepared = {"index": owner.index, "repository": owner.repository, "repository_id": request.repository_id,
        "expected_head": owner.expected_head, "scheduler": owner.scheduler, "output": arguments["output"]}
    match = finite_match(prepared, arguments["tool_policy"])
    with structural_codebase_context(owner.index, owner.repository, repository_id=request.repository_id,
            expected_head=owner.expected_head, scheduler=owner.scheduler, memory_mb=owner.memory_mb) as context:
        materials, operations = adapter._materials(match, arguments["operation_catalog"], context)
    with reserve_finite_integer_capacity(owner=owner, request=request, materials=materials) as capacity:
        bound = bound_materials(materials, capacity)
        service = service_for(owner, bound, operations, capacity)
        receipt = service.preview_create(request, materials=bound)
        assert service.critique.accepted and service.execution_plan.admitted
        assert len(service.obligation_graph.facts) == 1
        assert service.candidate_plan["tasks"][0]["task_id"] == "task:finite:offset"
        assert service.critic_evidence["finite_match"] == match
        assert receipt.verdict.value == "review_only"


@pytest.mark.parametrize("kind", ["released", "cancelled", "expired", "owner", "identity"])
def test_native_reservation_state_cannot_be_replaced_by_an_inert_binding(capacity_case, kind):
    owner, request, materials, _, _ = capacity_case
    with pytest.raises(FiniteIntegerCapacityError):
        with reserve_finite_integer_capacity(owner=owner, request=request, materials=materials) as capacity:
            bound = bound_materials(materials, capacity)
            if kind == "released":
                capacity.lease.release()
            elif kind == "cancelled":
                capacity.lease.cancel()
            else:
                with owner.scheduler._locked_state() as state:
                    row = state["leases"][capacity.lease.lease_id]
                    if kind == "expired":
                        row["expires_at"] = time.time() - 1
                    elif kind == "owner":
                        row["owner_birth_marker"] = "different-native-birth-marker"
                    else:
                        row["lease_key"] = "different-native-key"
            try:
                capacity.require_current(request, bound)
            finally:
                if kind == "identity":
                    with owner.scheduler._locked_state() as state:
                        state["leases"][capacity.lease.lease_id]["lease_key"] = capacity.lease.lease_key


@pytest.mark.parametrize("kind", ["memory", "pressure", "unknown", "backoff", "configuration"])
def test_independent_current_pressure_and_configuration_are_required(capacity_case, monkeypatch, kind):
    owner, request, materials, _, _ = capacity_case
    with pytest.raises(FiniteIntegerCapacityError):
        with reserve_finite_integer_capacity(owner=owner, request=request, materials=materials) as capacity:
            bound = bound_materials(materials, capacity)
            if kind == "configuration":
                owner.scheduler.config.proof_memory_headroom_mb += 1
            elif kind == "backoff":
                with owner.scheduler._locked_state() as state:
                    state["proof_backoff"] = {"until": time.time() + 10, "reason": "native-pressure-control"}
            elif kind == "unknown":
                def unavailable():
                    raise OSError("probe unavailable")
                monkeypatch.setattr(capacity_module, "collect_proof_host_resources", unavailable)
            else:
                measured = capacity_module.collect_proof_host_resources()
                hostile = replace(measured, available_memory_mb=0) if kind == "memory" else (
                    replace(measured, memory_stall_percent=100.0))
                monkeypatch.setattr(capacity_module, "collect_proof_host_resources", lambda: hostile)
            try:
                capacity.require_current(request, bound)
            finally:
                if kind == "configuration":
                    owner.scheduler.config.proof_memory_headroom_mb -= 1


@pytest.mark.parametrize("kind", ["basis", "request", "binding", "head"])
def test_complete_materials_request_binding_and_current_head_are_rechecked(capacity_case, kind):
    owner, request, materials, _, _ = capacity_case
    with pytest.raises(FiniteIntegerCapacityError):
        with reserve_finite_integer_capacity(owner=owner, request=request, materials=materials) as capacity:
            bound = bound_materials(materials, capacity)
            altered = request
            if kind == "basis":
                bound.extra["finite_match"]["scope"] = "changed"
            elif kind == "binding":
                bound.extra["finite_capacity_binding"]["reservation"]["cpu_slots"] = 999
            elif kind == "request":
                altered = replace(request, budget=replace(request.budget, max_tasks=request.budget.max_tasks + 1))
            else:
                (owner.repository / "calc.py").write_text("def increment(n: int) -> int:\n    return n + 2\n")
                owner.index.prepare_current(owner.repository, repository_id=request.repository_id,
                    operation_id="superseding-capacity-head", expected_head=owner.expected_head, scheduler=owner.scheduler)
            capacity.require_current(altered, bound)


def test_provided_parent_is_authenticated_and_capacity_keeps_source_as_sibling(capacity_case):
    owner, request, materials, operations, _ = capacity_case
    with owner.scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=3, memory_mb=owner.memory_mb * 3,
                                 child_process_slots=2, timeout=1) as parent:
        owned = replace(owner, scheduler=None, parent_lease=parent)
        with reserve_finite_integer_capacity(owner=owned, request=request, materials=materials) as capacity:
            assert capacity.lease.parent_lease_id == parent.lease_id
            bound = bound_materials(materials, capacity)
            service = service_for(owned, bound, operations, capacity)
            service.preview_create(request, materials=bound)
            assert service.execution_plan.admitted
            assert capacity.material_binding["reservation"]["parent_lease_id"] == parent.lease_id
        assert not parent.released and not parent.cancelled


@pytest.mark.parametrize("kind", ["identifier", "wrong_key", "wrong_scheduler", "insufficient",
                                  "insufficient_process", "cancelled"])
def test_parent_authority_and_overlap_budget_cannot_be_bypassed(capacity_case, kind, tmp_path):
    owner, request, materials, _, _ = capacity_case
    with owner.scheduler.acquire(ResourceLane.ORCHESTRATION,
            cpu_slots=1 if kind == "insufficient" else 3,
            memory_mb=owner.memory_mb if kind == "insufficient" else owner.memory_mb * 3,
            child_process_slots=1 if kind in {"insufficient", "insufficient_process"} else 2, timeout=1) as parent:
        control = parent
        if kind == "identifier":
            control = parent.lease_id
        elif kind == "wrong_key":
            control = replace(parent.token, lease_key="wrong-key")
        elif kind == "wrong_scheduler":
            control = replace(parent.token, state_path=str(tmp_path / "different-scheduler.json"))
        elif kind == "cancelled":
            parent.cancel()
        owned = replace(owner, parent_lease=control)
        with pytest.raises(FiniteIntegerCapacityError):
            with reserve_finite_integer_capacity(owner=owned, request=request, materials=materials):
                pytest.fail("invalid parent acquired capacity outside its budget")
        assert owner.scheduler.snapshot()["waiting_request_count"] == 0


def test_authenticated_parent_token_uses_an_actual_sibling_source_lease(capacity_case):
    owner, request, materials, operations, _ = capacity_case
    with owner.scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=3, memory_mb=owner.memory_mb * 3,
                                 child_process_slots=2, timeout=1) as parent:
        owned = replace(owner, parent_lease=parent.token)
        source_owner = replace(owner, scheduler=None, parent_lease=parent)
        with reserve_finite_integer_capacity(owner=owned, request=request, materials=materials) as capacity:
            bound = bound_materials(materials, capacity)
            service = CapacityBoundFiniteIntegerPlanCreateService(root_observer=observer(source_owner, bound),
                operation_bindings=operations, capacity=capacity)
            service.preview_create(request, materials=bound)
            assert service.execution_plan.admitted
            rows = owner.scheduler.active_leases()
            assert {row["lease_id"] for row in rows} == {parent.lease_id, capacity.lease.lease_id}
            assert not parent.cancelled


@pytest.mark.parametrize("kind", ["expired", "cancelled", "released"])
def test_parent_loss_after_reservation_revokes_the_forecast(capacity_case, kind):
    owner, request, materials, _, _ = capacity_case
    with owner.scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=3, memory_mb=owner.memory_mb * 3,
                                 child_process_slots=2, timeout=1) as parent:
        owned = replace(owner, parent_lease=parent)
        with pytest.raises(FiniteIntegerCapacityError):
            with reserve_finite_integer_capacity(owner=owned, request=request, materials=materials) as capacity:
                bound = bound_materials(materials, capacity)
                if kind == "expired":
                    with owner.scheduler._locked_state() as state:
                        state["leases"][parent.lease_id]["expires_at"] = time.time() - 1
                elif kind == "cancelled":
                    parent.cancel()
                else:
                    parent.release()
                capacity.require_current(request, bound)
        assert owner.scheduler.snapshot()["waiting_request_count"] == 0


def test_post_compile_pressure_and_stale_cached_forecast_cannot_publish(capacity_case, monkeypatch):
    owner, request, materials, operations, _ = capacity_case
    with reserve_finite_integer_capacity(owner=owner, request=request, materials=materials) as capacity:
        bound = bound_materials(materials, capacity)
        service = service_for(owner, bound, operations, capacity)
        service.preview_create(request, materials=bound)
        original_forecast = deepcopy(capacity._forecast_observation)
        request_body = deepcopy(capacity.last_compilation_request)
        request_body.pop("schema")
        expired_request = ParallelPlanCompilationRequest(**{
            **request_body, "current_time_ms": original_forecast["fresh_until_ms"]})
        rejected = ParallelPlanCompiler().compile(expired_request)
        assert not rejected.admitted and "stale_capacity" in [issue.code.value for issue in rejected.issues]
        # Move only this helper's test clock to the retained declared expiry;
        # native scheduler lease state, measured telemetry and bytes stay real.
        with monkeypatch.context() as expiry_clock:
            expiry_clock.setattr(capacity_module, "_now_ms", lambda: original_forecast["fresh_until_ms"])
            with pytest.raises(FiniteIntegerCapacityError, match="stale"):
                service.preview_create(request, materials=bound)
        assert capacity._forecast_observation == original_forecast

    original = ParallelPlanCompiler.compile
    injected = False
    def compile_then_pressure(self, *args, **kwargs):
        nonlocal injected
        plan = original(self, *args, **kwargs)
        if not injected:
            injected = True
            measured = capacity_module.collect_proof_host_resources()
            monkeypatch.setattr(capacity_module, "collect_proof_host_resources",
                                lambda: replace(measured, available_memory_mb=0))
        return plan
    monkeypatch.setattr(ParallelPlanCompiler, "compile", compile_then_pressure)
    with pytest.raises(FiniteIntegerCapacityError, match="headroom"):
        with reserve_finite_integer_capacity(owner=owner, request=request, materials=materials) as capacity:
            bound = bound_materials(materials, capacity)
            service = service_for(owner, bound, operations, capacity)
            service.preview_create(request, materials=bound)
    assert service._preview_by_key == {}


def test_resource_shadow_and_reviewed_effect_changes_are_rejected(capacity_case):
    owner, request, materials, operations, _ = capacity_case
    with reserve_finite_integer_capacity(owner=owner, request=request, materials=materials) as capacity:
        bound = bound_materials(materials, capacity)
        service = service_for(owner, bound, operations, capacity)
        service.preview_create(request, materials=bound)
        shadow = deepcopy(service.candidate_plan)
        shadow["tasks"][0]["memory_bytes"] = 0
        shadow["plan_id"] = cid_for_structured({key: value for key, value in shadow.items() if key != "plan_id"})
        with pytest.raises(FiniteIntegerCapacityError, match="shadow"):
            capacity.compile(request=request, materials=bound, candidate_plan=shadow)
        effect = deepcopy(service.candidate_plan)
        effect["effects"][0]["operation"] = "delete"
        with pytest.raises(FiniteIntegerCapacityError, match="complete selected"):
            service._stage_parallel_plan(request, bound, effect)


def test_cancelled_owner_and_unsigned_capacity_maps_are_not_live_controls(capacity_case):
    owner, request, materials, operations, _ = capacity_case
    with pytest.raises(FiniteIntegerCapacityError, match="privately acquired"):
        CapacityBoundFiniteIntegerPlanCreateService(root_observer=lambda bound: bound.roots,
            operation_bindings=operations, capacity={"cpu_slots": 999, "memory_bytes": 99999999999})
    cancel = threading.Event()
    cancel.set()
    cancelled_owner = replace(owner, cancel_event=cancel)
    with pytest.raises(Exception, match="cancel"):
        with reserve_finite_integer_capacity(owner=cancelled_owner, request=request, materials=materials):
            pytest.fail("cancelled owner acquired live capacity")
    assert owner.scheduler.snapshot()["active_lease_count"] == 0
