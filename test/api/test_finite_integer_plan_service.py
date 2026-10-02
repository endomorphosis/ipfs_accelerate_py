"""Pure service contracts; live observation provenance is tested by the owner API."""
from copy import deepcopy
from dataclasses import replace

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_service import (
    FINITE_SERVICE_PROFILE, FiniteIntegerPlanCreateService, FiniteIntegerPlanServiceError,
)
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority, FactTruth, ObservedFact, ProducerRule, TaskCandidate, TypedIntent,
    TypedPredicate, obligation_id_for_predicate, obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import SymbolicCandidatePortfolio
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    PlanCreateMaterials, PlanCreateStaleRootError,
)
from test.api.test_plan_create_semantic_input_identity import _cid, _request


def service_case(*, complete=False):
    request = replace(_request(), scope_paths=("calc.py",),
                      budget=replace(_request().budget, max_model_calls=0))
    root = request.roots.dirty_worktree_root
    predicates = (TypedPredicate("finite-predicate:type", "reviewed_finite_type", "closed:increment"),
                  TypedPredicate("finite-predicate:offset", "reviewed_finite_offset", "closed:increment"))
    requirements = {"clause:type": predicates[0].predicate_id, "clause:offset": predicates[1].predicate_id}
    intent = TypedIntent("intent:finite-service", predicates, ("intent-source:complete",),
                        current_root_id=root, metadata={"requirement_predicate_ids": requirements})
    facts = [ObservedFact("fact:type", predicates[0], FactTruth.TRUE,
        FactAuthority.BOUNDED_OBSERVATION, ("observation:authored",), current_root_id=root)]
    if complete:
        facts.append(ObservedFact("fact:offset", predicates[1], FactTruth.TRUE,
            FactAuthority.BOUNDED_OBSERVATION, ("observation:authored",), current_root_id=root))
    producers, tasks, operations = [], [], {}
    for index, (requirement, predicate) in enumerate(zip(("clause:type", "clause:offset"), predicates)):
        task_id, producer_id, review = f"task:finite:{index}", f"producer:finite:{index}", f"review:operation:{index}"
        closed = obligation_id_for_predicate(predicate.predicate_id) if index == 0 or complete else (
            obligation_id_for_producer(producer_id, predicate.predicate_id))
        producers.append(ProducerRule(producer_id, (predicate.predicate_id,),
            provenance_refs=intent.source_refs + (review,), task_candidate_ids=(task_id,)))
        tasks.append(TaskCandidate(task_id, (closed,), producer_id=producer_id,
                                   provenance_refs=intent.source_refs + (review,)))
        operations[task_id] = {"requirement_id": requirement, "task_id": task_id,
            "producer_id": producer_id, "operation": "update", "path": "calc.py",
            "function_name": "increment", "parameter": "n", "review_ref": review,
            "predicate_id": predicate.predicate_id, "closes_obligation_ids": [closed], "depends_on": []}
    policy = EvidenceAwarePlanPolicy(acceptance_criteria=intent.goal_predicate_ids,
        evidence_terms=intent.source_refs, supported_semantics=("python-integer-offset-finite@1",),
        allowed_scopes=("scope:calc.py",), available_resource_classes=("cpu",),
        require_validation=True, require_proof=False)
    match = {"profile": "python-integer-offset-finite@1", "typed_intent": intent.to_dict(),
             "current_facts": [fact.to_dict() for fact in facts], "current_root_id": root,
             "scope": "pure_service_contract_fixture"}
    materials = PlanCreateMaterials(intent=intent, current_facts=tuple(facts),
        producers=tuple(producers), task_candidates=tuple(tasks),
        frozen_goal=FrozenPlanningGoal("goal:finite-service", _cid("finite-goal"), root, policy),
        candidate_context={"domain": "finite-service-contract-fixture", "repository_paths": ["calc.py"],
            "task_metadata": {task.candidate_id: {"predicted_files": ["calc.py"],
                "predicted_symbols": ["increment"], "scope_ids": ["scope:calc.py"],
                "resource_classes": ["cpu"]} for task in tasks}},
        extra={"finite_service_profile": FINITE_SERVICE_PROFILE,
               "finite_operation_bindings": deepcopy(operations), "finite_match": match})
    return request, materials, operations


def test_actual_service_projects_only_selected_reviewed_task_and_binds_full_plan():
    request, materials, operations = service_case()
    observed = []
    service = FiniteIntegerPlanCreateService(
        root_observer=lambda value: observed.append(value.request_cid) or value.roots,
        operation_bindings=operations)
    receipt = service.preview_create(request, materials=materials)
    stages = {stage.stage.value: stage for stage in receipt.stage_results}
    assert stages["obligation"].passed and stages["candidate"].passed
    assert stages["critique"].passed
    # The real base compiler retains its missing-capacity debt. This profile
    # deliberately grants no execution authority or synthetic capacity receipt.
    assert not stages["parallel_plan"].passed
    assert "parallel:stale_capacity" in stages["parallel_plan"].blockers
    assert receipt.verdict.value == "review_only" and not stages["admission"].passed
    assert isinstance(service.portfolio, SymbolicCandidatePortfolio)
    assert service.planner_status == "selected"
    assert len(service.obligation_graph.root_obligation_ids) == 2
    assert len(service.obligation_graph.facts) == 1
    plan = service.candidate_plan
    assert [task["task_id"] for task in plan["tasks"]] == ["task:finite:1"]
    task = plan["tasks"][0]
    row = operations[task["task_id"]]
    assert task["closes_obligation_ids"] == row["closes_obligation_ids"]
    assert task["depends_on"] == row["depends_on"] and task["outputs"] == [row["path"]]
    effect = plan["effects"][0]
    assert effect["task_id"] == task["task_id"] and effect["target_id"] == "calc.py"
    assert effect["predicate_id"] == row["predicate_id"] and effect["operation"] == row["operation"]
    assert plan["expected_effect_ids"] == [effect["effect_id"]]
    assert plan["plan_id"] == cid_for_structured({key: value for key, value in plan.items() if key != "plan_id"})
    assert service.critique.accepted and not service.critique.truncated
    assert service.critic_evidence["finite_match"] == materials.extra["finite_match"]
    assert service.critic_evidence["operation_bindings"] == operations
    assert len(observed) >= 3 and receipt.read_only and receipt.wrote_effects == ()
    assert service.require_live_root_observation is True and service.receipt_store is None


def test_complete_roots_retain_facts_and_generate_distinct_no_work_artifacts():
    request, materials, operations = service_case(complete=True)
    service = FiniteIntegerPlanCreateService(root_observer=lambda value: value.roots,
                                           operation_bindings=operations)
    receipt = service.preview_create(request, materials=materials)
    stages = {stage.stage.value: stage for stage in receipt.stage_results}
    assert all(stages[name].passed for name in ("obligation", "candidate", "critique", "parallel_plan"))
    assert service.planner_status == "already_complete_in_finite_domain"
    assert not isinstance(service.portfolio, SymbolicCandidatePortfolio)
    assert len(service.obligation_graph.root_obligation_ids) == len(service.obligation_graph.facts) == 2
    assert service.candidate_plan["tasks"] == service.candidate_plan["effects"] == []
    assert service.critique.accepted
    assert service.execution_plan["status"] == "no_execution_requested"
    assert service.execution_plan["task_ids"] == [] and service.execution_plan["admitted"] is False
    assert receipt.verdict.value == "review_only"


def test_constructor_detaches_operation_map_and_cache_restores_matching_diagnostics():
    request, materials, operations = service_case()
    service = FiniteIntegerPlanCreateService(root_observer=lambda value: value.roots,
                                           operation_bindings=operations)
    operations["task:finite:1"]["path"] = "attacker.py"
    first = service.preview_create(request, materials=materials)
    plan = deepcopy(service.candidate_plan)
    service.candidate_plan["effects"][0]["operation"] = "delete"
    service.critic_evidence["finite_match"]["scope"] = "altered"
    second = service.preview_create(request, materials=materials)
    assert first == second and service.candidate_plan == plan
    assert service.critic_evidence["finite_match"] == materials.extra["finite_match"]
    assert service.operation_bindings["task:finite:1"]["path"] == "calc.py"


@pytest.mark.parametrize("member", ["model_provider", "obligation_graph", "evidence_bundle",
                                    "admission_materials", "parallel_tasks", "current_roots", "query_plan"])
def test_stage_and_model_injections_reject_before_live_observation(member):
    request, materials, operations = service_case()
    setattr(materials, member, request.roots if member == "current_roots" else {"claimed": True})
    calls = []
    service = FiniteIntegerPlanCreateService(root_observer=lambda value: calls.append(value) or value.roots,
                                           operation_bindings=operations)
    with pytest.raises(FiniteIntegerPlanServiceError, match="injected stages"):
        service.preview_create(request, materials=materials)
    assert calls == [] and service._preview_by_key == {}


def test_operation_binding_change_cannot_reuse_frozen_preview():
    request, materials, operations = service_case()
    service = FiniteIntegerPlanCreateService(root_observer=lambda value: value.roots,
                                           operation_bindings=operations)
    service.preview_create(request, materials=materials)
    materials.extra["finite_operation_bindings"]["task:finite:1"]["path"] = "other.py"
    with pytest.raises(FiniteIntegerPlanServiceError, match="exact frozen"):
        service.preview_create(request, materials=materials)


def test_altered_task_effect_semantics_reject_before_critic_or_cache():
    request, materials, operations = service_case()

    class AlteredEffect(FiniteIntegerPlanCreateService):
        def _candidate_plan_projection(self, request, portfolio, obligation):
            plan = super()._candidate_plan_projection(request, portfolio, obligation)
            plan["effects"][0]["operation"] = "delete"
            return plan

    service = AlteredEffect(root_observer=lambda value: value.roots, operation_bindings=operations)
    with pytest.raises(FiniteIntegerPlanServiceError, match="effect semantics"):
        service.preview_create(request, materials=materials)
    assert service.critique is None and service._preview_by_key == {}


def test_graph_closure_cannot_differ_from_reviewed_operation_map():
    request, materials, operations = service_case()
    operations["task:finite:1"]["closes_obligation_ids"] = ["obligation:invented"]
    materials.extra["finite_operation_bindings"] = deepcopy(operations)
    service = FiniteIntegerPlanCreateService(root_observer=lambda value: value.roots,
                                           operation_bindings=operations)
    with pytest.raises(FiniteIntegerPlanServiceError, match="closure"):
        service.preview_create(request, materials=materials)
    assert service._preview_by_key == {}


def test_live_root_change_before_admission_never_enters_cache():
    request, materials, operations = service_case()
    calls = []

    def observe(value):
        calls.append(value)
        return value.roots if len(calls) == 1 else replace(value.roots, policy_root=_cid("changed-policy"))

    service = FiniteIntegerPlanCreateService(root_observer=observe, operation_bindings=operations)
    with pytest.raises(PlanCreateStaleRootError):
        service.preview_create(request, materials=materials)
    assert service._preview_by_key == {} and len(calls) == 2


def test_extra_zero_premise_logical_producer_cannot_replace_the_residual_repair():
    request, materials, operations = service_case()
    residual = operations["task:finite:1"]["predicate_id"]
    materials.producers = (*materials.producers, ProducerRule(
        "producer:undeclared-logical", (residual,), executable=False))
    service = FiniteIntegerPlanCreateService(root_observer=lambda value: value.roots,
                                           operation_bindings=operations)
    with pytest.raises(FiniteIntegerPlanServiceError, match="complete finite graph/task population"):
        service.preview_create(request, materials=materials)
    assert service._preview_by_key == {} and service.portfolio is None


@pytest.mark.parametrize("change", ["premise", "task", "assumption", "proof", "validation"])
def test_reviewed_producer_cannot_acquire_hidden_semantics(change):
    request, materials, operations = service_case()
    producer = materials.producers[1]
    if change == "premise":
        hidden = TypedPredicate("predicate:hidden", "reviewed_hidden_premise", "closed:hidden")
        materials.predicates = (hidden,)
        producer = replace(producer, required_predicate_ids=(hidden.predicate_id,))
    elif change == "task":
        producer = replace(producer, task_candidate_ids=tuple(sorted(("task:undeclared", *producer.task_candidate_ids))))
    elif change == "assumption":
        producer = replace(producer, assumption_refs=("assumption:undeclared",))
    elif change == "proof":
        producer = replace(producer, proof_requirement_refs=("proof:undeclared",))
    else:
        producer = replace(producer, validation_requirement_refs=("validation:undeclared",))
    materials.producers = (materials.producers[0], producer)
    service = FiniteIntegerPlanCreateService(root_observer=lambda value: value.roots,
                                           operation_bindings=operations)
    reason = "complete finite graph/task population" if change == "task" else "reviewed operation differs"
    with pytest.raises(FiniteIntegerPlanServiceError, match=reason):
        service.preview_create(request, materials=materials)
    assert service._preview_by_key == {} and service.portfolio is None
