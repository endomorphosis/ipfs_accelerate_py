from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_logic_vocabulary import (
    DCECOperator,
    DCECVocabulary,
    EvidenceEdge,
    EvidenceNode,
    FiniteTrace,
    Formula,
    FrameLogicProjection,
    FrameProjectionConfig,
    LOGIC_VOCABULARY_VERSION,
    ReviewedPredicate,
    TDFOLProperty,
    TDFOLVocabulary,
    TermSort,
    TraceFact,
    TraceStep,
    atom,
    constant,
    evaluate_formula,
    project_frame_logic,
    subgoal_satisfied,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_contracts import (
    Actor,
    ActorKind,
    ContractValidationError,
    Effect,
    EffectOperation,
    EventKind,
    EvidenceRequirement,
    EvidenceRequirementKind,
    Fluent,
    FluentValueType,
    FormalWorkPlan,
    Goal,
    Norm,
    NormKind,
    PlanAssurance,
    PlanConformanceLevel,
    PlanConsistencyLevel,
    PlanEvent,
    PlanTask,
    Precondition,
    RefinementMode,
    Subgoal,
    TemporalConstraint,
    TemporalConstraintKind,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    content_identity,
)


def _formulas():
    ready = atom(
        ReviewedPredicate.TASK_READY,
        constant(TermSort.TASK, "task:implement"),
    )
    done = atom(
        ReviewedPredicate.TASK_COMPLETED,
        constant(TermSort.TASK, "task:implement"),
    )
    goal_done = atom(
        ReviewedPredicate.GOAL_SATISFIED,
        constant(TermSort.GOAL, "goal:formal-plan"),
    )
    return ready, done, goal_done


def _plan(*, task_dependencies=()):
    ready, done, goal_done = _formulas()
    subgoal_done = TDFOLVocabulary.subgoal_satisfaction("subgoal:contract", 4)
    requirement = EvidenceRequirement(
        requirement_id="evidence:tests",
        kind=EvidenceRequirementKind.TEST,
        subject_ids=("task:implement", "goal:formal-plan"),
        source_scope_ids=("tree:base", "symbol:FormalWorkPlan"),
        freshness_seconds=3600,
        fallback_check_ids=("pytest:formal-planning",),
    )
    precondition = Precondition(
        precondition_id="pre:ready",
        formula_id=ready.formula_id,
        task_id="task:implement",
        event_id="event:execute",
    )
    effect = Effect(
        effect_id="effect:done",
        operation=EffectOperation.ASSIGN,
        task_id="task:implement",
        event_id="event:execute",
        fluent_id="fluent:done",
        value=True,
    )
    return FormalWorkPlan(
        vocabulary_profile_id="supervisor-reviewed-dcec-tdfol",
        vocabulary_version=LOGIC_VOCABULARY_VERSION,
        source_ids=("source:taskboard", "source:objective"),
        repository_tree_id="tree:base",
        trace_bound=4,
        actors=(
            Actor(
                actor_id="actor:supervisor",
                kind=ActorKind.SUPERVISOR,
                capabilities=("delegate", "merge"),
            ),
            Actor(
                actor_id="actor:codex",
                kind=ActorKind.AGENT,
                capabilities=("implement",),
            ),
        ),
        goals=(
            Goal(
                goal_id="goal:formal-plan",
                owner_actor_id="actor:supervisor",
                satisfaction_formula_id=goal_done.formula_id,
                evidence_requirement_ids=("evidence:tests",),
            ),
        ),
        subgoals=(
            Subgoal(
                subgoal_id="subgoal:contract",
                goal_id="goal:formal-plan",
                satisfaction_formula_id=subgoal_done.formula_id,
                evidence_requirement_ids=("evidence:tests",),
            ),
        ),
        tasks=(
            PlanTask(
                task_id="task:implement",
                goal_id="goal:formal-plan",
                subgoal_id="subgoal:contract",
                actor_ids=("actor:codex",),
                depends_on=task_dependencies,
                precondition_ids=("pre:ready",),
                effect_ids=("effect:done",),
                event_ids=("event:execute",),
                evidence_requirement_ids=("evidence:tests",),
            ),
        ),
        events=(
            PlanEvent(
                event_id="event:execute",
                kind=EventKind.EXECUTED,
                actor_id="actor:codex",
                task_id="task:implement",
                logical_time=2,
                provenance_ids=("receipt:lease",),
            ),
        ),
        fluents=(
            Fluent(
                fluent_id="fluent:done",
                value_type=FluentValueType.BOOLEAN,
                initial_value=False,
            ),
        ),
        preconditions=(precondition,),
        effects=(effect,),
        norms=(
            Norm(
                norm_id="norm:execute",
                kind=NormKind.OBLIGATION,
                bearer_actor_id="actor:codex",
                issuer_actor_id="actor:supervisor",
                action_id="task:implement",
                activation_formula_id=ready.formula_id,
                valid_from=1,
                valid_until=4,
            ),
        ),
        temporal_constraints=(
            TemporalConstraint(
                constraint_id="temporal:deadline",
                kind=TemporalConstraintKind.DEADLINE,
                subject_ids=("task:implement",),
                formula_id=TDFOLVocabulary.deadline("task:implement", 4).formula_id,
                lower_bound=0,
                upper_bound=4,
            ),
        ),
        evidence_requirements=(requirement,),
        formulas=(
            ready,
            done,
            goal_done,
            subgoal_done,
            TDFOLVocabulary.deadline("task:implement", 4),
        ),
    )


def test_formal_work_plan_records_every_required_collection_and_round_trips():
    plan = _plan()
    payload = plan.to_dict()

    assert plan.plan_id.startswith("baguqeera")
    assert FormalWorkPlan.from_dict(payload) == plan
    assert FormalWorkPlan.from_json(plan.to_json()) == plan
    assert json.loads(plan.to_json())["schema"].endswith("formal-work-plan@1")
    assert set(
        (
            "actors",
            "goals",
            "subgoals",
            "tasks",
            "events",
            "fluents",
            "preconditions",
            "effects",
            "norms",
            "temporal_constraints",
            "evidence_requirements",
            "formulas",
        )
    ).issubset(payload)


def test_plan_identity_is_deterministic_for_set_semantic_collections():
    plan = _plan()
    payload = plan.to_dict()
    payload["actors"] = list(reversed(payload["actors"]))
    payload["formulas"] = list(reversed(payload["formulas"]))
    payload["source_ids"] = list(reversed(payload["source_ids"]))

    reordered = FormalWorkPlan.from_dict(payload)
    assert reordered.to_json() == plan.to_json()
    assert reordered.plan_id == plan.plan_id
    assert _plan().plan_id == plan.plan_id


def test_plan_rejects_unknown_references_cycles_and_out_of_bound_events():
    with pytest.raises(ContractValidationError, match="unknown dependency"):
        _plan(task_dependencies=("task:missing",))

    payload = _plan().to_dict()
    payload["events"][0]["logical_time"] = 5
    with pytest.raises(ContractValidationError, match="outside trace_bound"):
        FormalWorkPlan.from_dict(payload)

    payload = _plan().to_dict()
    payload["goals"][0]["owner_actor_id"] = "actor:invented"
    with pytest.raises(ContractValidationError, match="unknown id"):
        FormalWorkPlan.from_dict(payload)


def test_subgoal_satisfaction_is_typed_versioned_and_deterministic():
    atom_formula = subgoal_satisfied("subgoal:contract")
    bounded = TDFOLVocabulary.subgoal_satisfaction("subgoal:contract", 4)

    assert atom_formula.predicate is ReviewedPredicate.SUBGOAL_SATISFIED
    assert atom_formula.terms[0].sort is TermSort.SUBGOAL
    assert atom_formula.profile_version == LOGIC_VOCABULARY_VERSION
    assert bounded.operands == (atom_formula,)
    assert Formula.from_dict(atom_formula.to_dict()).formula_id == atom_formula.formula_id
    assert subgoal_satisfied("subgoal:contract").formula_id == atom_formula.formula_id

    fact = TraceFact(
        ReviewedPredicate.SUBGOAL_SATISFIED,
        (constant(TermSort.SUBGOAL, "subgoal:contract"),),
    )
    trace = FiniteTrace(
        source_plan_id="plan:typed-subgoal",
        bound=1,
        steps=(TraceStep(0), TraceStep(1, facts=(fact,))),
    )
    assert evaluate_formula(
        TDFOLVocabulary.subgoal_satisfaction("subgoal:contract", 1), trace
    )


def test_revision_one_formula_identity_is_preserved_and_cannot_use_new_predicate():
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/reviewed-formula@1",
        "vocabulary_version": 1,
        "profile_id": "supervisor-reviewed",
        "profile_version": 1,
        "operator": "atom",
        "predicate": "goal_satisfied",
        "terms": [
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/logic-term@1",
                "vocabulary_version": 1,
                "sort": "goal",
                "kind": "constant",
                "value": "goal:legacy",
            }
        ],
        "operands": [],
        "lower_bound": None,
        "upper_bound": None,
    }
    restored = Formula.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.formula_id == content_identity(payload)

    payload["predicate"] = "subgoal_satisfied"
    payload["terms"][0]["sort"] = "subgoal"
    with pytest.raises(ContractValidationError, match="requires reviewed vocabulary"):
        Formula.from_dict(payload)


def test_subgoal_refinement_records_parent_and_safe_default_without_root_mutation():
    plan = _plan()
    root_formula_id = plan.goals[0].satisfaction_formula_id
    child_formula = TDFOLVocabulary.subgoal_satisfaction("subgoal:tests", 4)
    payload = plan.to_dict()
    payload["subgoals"].append(
        Subgoal(
            subgoal_id="subgoal:tests",
            goal_id="goal:formal-plan",
            parent_id="subgoal:contract",
            refinement_mode=RefinementMode.EQUIVALENT,
            satisfaction_formula_id=child_formula.formula_id,
            depends_on=("subgoal:contract",),
            evidence_requirement_ids=("evidence:tests",),
        ).to_dict()
    )
    payload["formulas"].append(child_formula.to_dict())

    refined = FormalWorkPlan.from_dict(payload)
    direct = next(item for item in refined.subgoals if item.subgoal_id == "subgoal:contract")
    child = next(item for item in refined.subgoals if item.subgoal_id == "subgoal:tests")
    assert direct.parent_id == direct.goal_id
    assert direct.refinement_mode is RefinementMode.SUFFICIENT
    assert child.parent_id == direct.subgoal_id
    assert child.refinement_mode is RefinementMode.EQUIVALENT
    assert refined.goals[0].satisfaction_formula_id == root_formula_id
    assert FormalWorkPlan.from_dict(refined.to_dict()) == refined


def test_legacy_plan_without_subgoals_preserves_shape_and_identity():
    payload = _plan().to_dict()
    payload["subgoals"] = []
    payload["tasks"][0]["subgoal_id"] = ""
    payload.pop("formulas")
    payload["vocabulary_version"] = 1

    legacy = FormalWorkPlan.from_dict(payload)
    assert legacy.subgoals == ()
    assert legacy.formulas == ()
    assert "formulas" not in legacy.to_dict()
    assert FormalWorkPlan.from_dict(legacy.to_dict()).plan_id == legacy.plan_id

    typed = _plan().to_dict()
    typed["vocabulary_version"] = 1
    with pytest.raises(ContractValidationError, match="typed subgoals require"):
        FormalWorkPlan.from_dict(typed)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("operator", "model_invented_operator", "operator must be one of"),
        ("predicate", "model_invented_predicate", "predicate must be one of"),
    ),
)
def test_plan_rejects_unreviewed_formula_operators_and_predicates(
    field, value, match
):
    payload = _plan().to_dict()
    formula = next(item for item in payload["formulas"] if item["operator"] == "atom")
    formula[field] = value
    with pytest.raises(ContractValidationError, match=match):
        FormalWorkPlan.from_dict(payload)


def test_plan_rejects_unknown_sorts_and_formula_references():
    payload = _plan().to_dict()
    formula = next(
        item
        for item in payload["formulas"]
        if item["operator"] == "atom" and item["terms"]
    )
    formula["terms"][0]["sort"] = "model_invented_sort"
    with pytest.raises(ContractValidationError, match="sort must be one of"):
        FormalWorkPlan.from_dict(payload)

    payload = _plan().to_dict()
    payload["subgoals"][0]["satisfaction_formula_id"] = "formula:unreviewed"
    with pytest.raises(ContractValidationError, match="unknown satisfaction formula"):
        FormalWorkPlan.from_dict(payload)


def test_plan_rejects_wrong_subgoal_formula_self_links_and_graph_cycles():
    payload = _plan().to_dict()
    payload["subgoals"][0]["satisfaction_formula_id"] = _formulas()[1].formula_id
    with pytest.raises(ContractValidationError, match="subgoal_satisfied"):
        FormalWorkPlan.from_dict(payload)

    with pytest.raises(ContractValidationError, match="own parent"):
        Subgoal(
            subgoal_id="subgoal:self",
            goal_id="goal:formal-plan",
            parent_id="subgoal:self",
            satisfaction_formula_id=subgoal_satisfied("subgoal:self").formula_id,
        )
    with pytest.raises(ContractValidationError, match="depend on itself"):
        Subgoal(
            subgoal_id="subgoal:self",
            goal_id="goal:formal-plan",
            satisfaction_formula_id=subgoal_satisfied("subgoal:self").formula_id,
            depends_on=("subgoal:self",),
        )

    child_formula = subgoal_satisfied("subgoal:child")
    payload = _plan().to_dict()
    payload["subgoals"][0]["parent_id"] = "subgoal:child"
    payload["subgoals"].append(
        Subgoal(
            subgoal_id="subgoal:child",
            goal_id="goal:formal-plan",
            parent_id="subgoal:contract",
            satisfaction_formula_id=child_formula.formula_id,
        ).to_dict()
    )
    payload["formulas"].append(child_formula.to_dict())
    with pytest.raises(ContractValidationError, match="acyclic"):
        FormalWorkPlan.from_dict(payload)


def test_plan_rejects_unknown_and_cross_root_subgoal_parents():
    payload = _plan().to_dict()
    payload["subgoals"][0]["parent_id"] = "subgoal:missing"
    with pytest.raises(ContractValidationError, match="unknown id"):
        FormalWorkPlan.from_dict(payload)

    payload = _plan().to_dict()
    payload["goals"].append(
        Goal(
            goal_id="goal:other",
            owner_actor_id="actor:supervisor",
            satisfaction_formula_id=_formulas()[2].formula_id,
        ).to_dict()
    )
    payload["subgoals"][0]["parent_id"] = "goal:other"
    with pytest.raises(ContractValidationError, match="does not match goal_id"):
        FormalWorkPlan.from_dict(payload)

def test_plan_rejects_conflicting_effects_for_one_transition():
    payload = _plan().to_dict()
    payload["effects"].append(
        Effect(
            effect_id="effect:not-done",
            operation=EffectOperation.ASSIGN,
            task_id="task:implement",
            event_id="event:execute",
            fluent_id="fluent:done",
            value=False,
        ).to_dict()
    )
    payload["tasks"][0]["effect_ids"].append("effect:not-done")
    with pytest.raises(ContractValidationError, match="conflicting effects"):
        FormalWorkPlan.from_dict(payload)


def test_dcec_exposes_only_reviewed_required_modalities_and_round_trips():
    ready, _, _ = _formulas()
    formulas = (
        DCECVocabulary.belief("actor:codex", ready, 0),
        DCECVocabulary.knowledge("actor:codex", ready, 0),
        DCECVocabulary.intention("actor:codex", ready, 1),
        DCECVocabulary.obligation("actor:codex", ready, 4),
        DCECVocabulary.permission("actor:codex", ready, 1),
        DCECVocabulary.prohibition("actor:codex", ready, 1),
        DCECVocabulary.delegation(
            "actor:supervisor", "actor:codex", "task:implement", 1
        ),
        DCECVocabulary.execution_event(
            "actor:codex", "task:implement", "event:execute", 2
        ),
    )
    assert {formula.operator.value for formula in formulas} == {
        item.value for item in DCECOperator
    }
    assert all(Formula.from_dict(item.to_dict()) == item for item in formulas)

    with pytest.raises(ContractValidationError, match="one of"):
        atom(
            "predicate-derived-from-model-prose",
            constant(TermSort.SYMBOL, "anything"),
        )
    with pytest.raises(ContractValidationError, match="expects term sorts"):
        atom(
            ReviewedPredicate.TASK_READY,
            constant(TermSort.ACTOR, "actor:codex"),
        )


def _trace():
    return FiniteTrace(
        source_plan_id=_plan().plan_id,
        bound=3,
        steps=(
            TraceStep(
                0,
                facts=(
                    TraceFact(
                        ReviewedPredicate.SAFE_STATE,
                        (constant(TermSort.SYMBOL, "leases"),),
                    ),
                ),
            ),
            TraceStep(
                1,
                facts=(
                    TraceFact(
                        ReviewedPredicate.TASK_COMPLETED,
                        (constant(TermSort.TASK, "task:prepare"),),
                    ),
                    TraceFact(
                        ReviewedPredicate.SAFE_STATE,
                        (constant(TermSort.SYMBOL, "leases"),),
                    ),
                ),
            ),
            TraceStep(
                2,
                facts=(
                    TraceFact(
                        ReviewedPredicate.TASK_STARTED,
                        (constant(TermSort.TASK, "task:implement"),),
                    ),
                    TraceFact(
                        ReviewedPredicate.TASK_COMPLETED,
                        (constant(TermSort.TASK, "task:implement"),),
                    ),
                    TraceFact(
                        ReviewedPredicate.SAFE_STATE,
                        (constant(TermSort.SYMBOL, "leases"),),
                    ),
                ),
            ),
            TraceStep(
                3,
                facts=(
                    TraceFact(
                        ReviewedPredicate.GOAL_SATISFIED,
                        (constant(TermSort.GOAL, "goal:formal-plan"),),
                    ),
                    TraceFact(
                        ReviewedPredicate.SAFE_STATE,
                        (constant(TermSort.SYMBOL, "leases"),),
                    ),
                ),
            ),
        ),
    )


def test_tdfol_models_all_required_properties_over_a_finite_trace():
    trace = _trace()
    safe = atom(
        ReviewedPredicate.SAFE_STATE,
        constant(TermSort.SYMBOL, "leases"),
    )
    completed = atom(
        ReviewedPredicate.TASK_COMPLETED,
        constant(TermSort.TASK, "task:implement"),
    )
    formulas = {
        TDFOLProperty.DEPENDENCY_ORDER: TDFOLVocabulary.dependency_order(
            "task:prepare", "task:implement"
        ),
        TDFOLProperty.DEADLINE: TDFOLVocabulary.deadline("task:implement", 2),
        TDFOLProperty.LIVENESS: TDFOLVocabulary.liveness(completed, 3),
        TDFOLProperty.SAFETY: TDFOLVocabulary.safety(safe, 3),
        TDFOLProperty.GOAL_SATISFACTION: TDFOLVocabulary.goal_satisfaction(
            "goal:formal-plan", 3
        ),
    }

    assert all(evaluate_formula(formula, trace) for formula in formulas.values())
    assert {formula.operator.value for formula in formulas.values()} == {
        item.value for item in TDFOLProperty
    }
    with pytest.raises(ContractValidationError, match="finite upper_bound"):
        Formula(operator="liveness", operands=(completed,))


def test_finite_trace_rejects_gaps_and_never_claims_unbounded_liveness():
    with pytest.raises(ContractValidationError, match="contiguous"):
        FiniteTrace(
            source_plan_id="plan:x",
            bound=2,
            steps=(TraceStep(0), TraceStep(2)),
        )
    with pytest.raises(ContractValidationError, match="outside"):
        evaluate_formula(
            TDFOLVocabulary.goal_satisfaction("goal:x", 2),
            FiniteTrace(source_plan_id="plan:x", bound=0, steps=(TraceStep(0),)),
            index=1,
        )


def test_frame_projection_is_versioned_bounded_and_not_a_code_proof():
    trace = _trace()
    nodes = tuple(EvidenceNode("e:%d" % index) for index in range(5))
    edges = tuple(
        EvidenceEdge("e:%d" % index, "e:%d" % (index + 1), "supports")
        for index in range(4)
    )
    projection = project_frame_logic(
        trace,
        evidence_nodes=nodes,
        evidence_edges=edges,
        seed_ids=("e:0",),
        config=FrameProjectionConfig(
            max_worlds=2, max_hops=1, max_nodes=2, max_edges=1
        ),
    )

    assert len(projection.worlds) == 2
    assert len(projection.accessibility_relations) == 1
    assert len(projection.evidence_nodes) == 2
    assert len(projection.evidence_edges) == 1
    assert projection.truncated
    assert projection.proves_code is False
    assert projection.code_assurance is AssuranceLevel.UNVERIFIED
    assert FrameLogicProjection.from_dict(projection.to_dict()) == projection

    forged = projection.to_dict()
    forged["proves_code"] = True
    forged["code_assurance"] = "kernel_verified"
    with pytest.raises(ContractValidationError, match="never prove code"):
        FrameLogicProjection.from_dict(forged)


def test_plan_consistency_conformance_and_code_assurance_are_separate():
    plan = _plan()
    assurance = PlanAssurance(
        plan_id=plan.plan_id,
        consistency=PlanConsistencyLevel.BOUNDED_CONSISTENT,
        conformance=PlanConformanceLevel.BOUNDED_CONFORMANT,
        generated_code_assurance=AssuranceLevel.UNVERIFIED,
        consistency_receipt_ids=("receipt:plan-check",),
        conformance_receipt_ids=("receipt:trace-check",),
        bounds={"trace_steps": 4, "worlds": 4},
    )
    assert assurance.plan_consistency is PlanConsistencyLevel.BOUNDED_CONSISTENT
    assert assurance.plan_conformance is PlanConformanceLevel.BOUNDED_CONFORMANT
    assert assurance.code_assurance is AssuranceLevel.UNVERIFIED
    assert PlanAssurance.from_dict(assurance.to_dict()) == assurance
    independently_proved = PlanAssurance(
        plan_id=plan.plan_id,
        consistency=PlanConsistencyLevel.BOUNDED_CONSISTENT,
        generated_code_assurance=AssuranceLevel.KERNEL_VERIFIED,
        consistency_receipt_ids=("receipt:plan-check",),
        code_proof_receipt_ids=("receipt:code-kernel",),
    )
    assert independently_proved.code_assurance is AssuranceLevel.KERNEL_VERIFIED

    with pytest.raises(ContractValidationError, match="independent code-proof receipt"):
        PlanAssurance(
            plan_id=plan.plan_id,
            consistency=PlanConsistencyLevel.KERNEL_VERIFIED,
            conformance=PlanConformanceLevel.ATTESTED,
            generated_code_assurance=AssuranceLevel.KERNEL_VERIFIED,
            consistency_receipt_ids=("receipt:plan-kernel",),
            conformance_receipt_ids=("receipt:trace-attestation",),
        )
    with pytest.raises(ContractValidationError, match="cannot be promoted"):
        PlanAssurance(
            plan_id=plan.plan_id,
            consistency=PlanConsistencyLevel.KERNEL_VERIFIED,
            generated_code_assurance=AssuranceLevel.KERNEL_VERIFIED,
            consistency_receipt_ids=("receipt:plan-kernel",),
            code_proof_receipt_ids=("receipt:plan-kernel",),
        )


def test_canonical_contracts_reject_floats_and_forged_identities():
    with pytest.raises(ContractValidationError, match="floats"):
        Actor(actor_id="actor:x", metadata={"confidence": 0.9})

    payload = _plan().to_dict()
    payload["content_id"] = "baguqeeraforged"
    with pytest.raises(ContractValidationError, match="identity"):
        FormalWorkPlan.from_dict(payload)
