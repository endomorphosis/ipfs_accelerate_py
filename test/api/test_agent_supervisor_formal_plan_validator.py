from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.formal_logic_vocabulary import (
    DCEC,
    ReviewedPredicate,
    TDFOL,
    TermSort,
    atom,
    constant,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_compiler import (
    CompilationStatus,
    compile_formal_plan,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_validator import (
    CancellationToken,
    FindingDisposition,
    FormalPlanValidator,
    PlanCheckEvidence,
    PlanCheckKind,
    PlanFindingCode,
    PlanValidationOutcome,
    PlanValidationResult,
    PlanValidationStatus,
    ValidationBounds,
    validate_formal_plan,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_contracts import (
    EffectOperation,
    EventKind,
    RefinementMode,
    Norm,
    NormKind,
    PlanConsistencyLevel,
    PlanEvent,
    Subgoal,
)


def _source() -> dict[str, object]:
    return {
        "repository_tree_id": "tree:formal-plan",
        "objectives": [
            {
                "goal_id": "G",
                "goal_cid": "goal:cid",
                "owner_actor_id": "supervisor",
                "acceptance_criteria": ["all required work is complete"],
            }
        ],
        "tasks": [
            {
                "task_id": "A",
                "task_cid": "task:a",
                "goal_id": "G",
                "actor_id": "agent:a",
                "resource_needs": ["cpu"],
                "acceptance_criteria": ["test A"],
                "validation_commands": ["pytest test_a.py"],
                "lease": {
                    "lease_cid": "lease:a",
                    "holder_id": "agent:a",
                    "fencing_token": 7,
                },
            },
            {
                "task_id": "B",
                "task_cid": "task:b",
                "goal_id": "G",
                "actor_id": "agent:b",
                "depends_on": ["A"],
                "acceptance_criteria": ["test B"],
                "validation_commands": ["pytest test_b.py"],
                "deadline": 8,
            },
        ],
        "ast": [
            {
                "symbol_cid": "symbol:a",
                "tree_cid": "tree:formal-plan",
                "task_cid": "task:a",
            },
            {
                "symbol_cid": "symbol:b",
                "tree_cid": "tree:formal-plan",
                "task_cid": "task:b",
            },
        ],
        "policies": [
            {
                "policy_cid": "policy:plan",
                "minimum_code_assurance": "candidate",
                "fallback_check_ids": ["policy-review"],
            }
        ],
    }


def _compiled():
    result = compile_formal_plan(_source())
    assert result.status is CompilationStatus.COMPILED
    assert result.plan is not None
    return result.plan, result.formulas


def _replace_event(plan, event_id: str, **updates):
    events = tuple(
        replace(item, **updates) if item.event_id == event_id else item
        for item in plan.events
    )
    return replace(plan, events=events)


def _with_subgoals(plan, formulas, *, equivalent_second: bool = False):
    first_formula = TDFOL.subgoal_satisfaction("subgoal:a", plan.trace_bound)
    second_formula = TDFOL.subgoal_satisfaction("subgoal:b", plan.trace_bound)
    tasks = tuple(
        replace(
            item,
            subgoal_id="subgoal:a" if item.task_id == "task:a" else "subgoal:b",
        )
        for item in plan.tasks
    )
    first_task = next(item for item in tasks if item.task_id == "task:a")
    second_task = next(item for item in tasks if item.task_id == "task:b")
    subgoals = (
        Subgoal(
            "subgoal:a",
            "goal:cid",
            first_formula.formula_id,
            evidence_requirement_ids=first_task.evidence_requirement_ids,
            metadata={"source_ids": ["source:subgoal:a"]},
        ),
        Subgoal(
            "subgoal:b",
            "goal:cid",
            second_formula.formula_id,
            refinement_mode=(
                RefinementMode.EQUIVALENT
                if equivalent_second
                else RefinementMode.SUFFICIENT
            ),
            depends_on=("subgoal:a",),
            evidence_requirement_ids=second_task.evidence_requirement_ids,
            metadata={"source_ids": ["source:subgoal:b"]},
        ),
    )
    all_formulas = {
        item.formula_id: item
        for item in (*formulas, first_formula, second_formula)
    }
    return (
        replace(
            plan,
            subgoals=subgoals,
            tasks=tasks,
            formulas=tuple(all_formulas.values()),
        ),
        tuple(all_formulas.values()),
    )


def test_consistent_compiled_plan_checks_every_required_property() -> None:
    plan, formulas = _compiled()

    first = validate_formal_plan(plan, formulas)
    second = validate_formal_plan(plan.to_record(), formulas)
    automatic = validate_formal_plan(plan)

    assert first.status is PlanValidationStatus.CONSISTENT
    assert automatic.status is PlanValidationStatus.CONSISTENT
    assert automatic.formula_ids == first.formula_ids
    assert first.outcome is PlanValidationOutcome.CONSISTENT
    assert first.consistency_level is PlanConsistencyLevel.BOUNDED_CONSISTENT
    assert first.plan_check_only
    assert first.findings == ()
    assert first.validation_id == second.validation_id
    assert first.to_json() == second.to_json()
    assert (
        PlanValidationResult.from_dict(first.to_record()).validation_id
        == first.validation_id
    )
    assert PlanValidationResult.from_json(first.to_json()).to_dict() == first.to_dict()
    assert set(first.checks_performed) >= {
        PlanCheckKind.DEPENDENCY_READINESS,
        PlanCheckKind.ACTOR_AUTHORITY,
        PlanCheckKind.UNIQUE_LEASE,
        PlanCheckKind.FENCING,
        PlanCheckKind.REQUIRED_EVIDENCE,
        PlanCheckKind.LEGAL_TRANSITION,
        PlanCheckKind.EVENTUAL_TERMINAL,
        PlanCheckKind.FORBIDDEN_MERGE_STATE,
        PlanCheckKind.DEONTIC_CONSISTENCY,
        PlanCheckKind.TEMPORAL_FORMULA,
    }
    assert first.assumptions
    assert {item.kind for item in first.assumptions} >= {
        "finite_domain",
        "finite_trace",
        "event_completeness",
        "inertia",
        "plan_only",
        "evidence_realisability",
    }
    assert first.bounds.plan_trace_bound == plan.trace_bound
    assert first.bounds.effective_trace_bound == plan.trace_bound
    assert first.bounds.domain_sizes["tasks"] == 2
    assert first.bounds.search_nodes_explored > 0


def test_deontic_contradiction_is_not_a_countermodel() -> None:
    plan, formulas = _compiled()
    obligation = next(
        item
        for item in plan.norms
        if item.action_id == "task:a" and item.kind is NormKind.OBLIGATION
    )
    prohibition = Norm(
        norm_id="norm:prohibit:a",
        kind=NormKind.PROHIBITION,
        bearer_actor_id=obligation.bearer_actor_id,
        issuer_actor_id="supervisor",
        action_id=obligation.action_id,
        valid_from=obligation.valid_from,
        valid_until=obligation.valid_until,
    )

    result = validate_formal_plan(
        replace(plan, norms=(*plan.norms, prohibition)), formulas
    )

    assert result.status is PlanValidationStatus.INCONSISTENT
    assert result.outcome is PlanValidationOutcome.CONTRADICTION
    assert result.countermodel is None
    assert result.consistency_level is PlanConsistencyLevel.COUNTEREXAMPLE
    assert PlanFindingCode.CONFLICTING_NORMS in {item.code for item in result.findings}
    assert all(
        item.disposition is FindingDisposition.CONTRADICTION for item in result.findings
    )


def test_dependency_and_authority_violations_return_bounded_countermodels() -> None:
    plan, formulas = _compiled()
    dependency_completion = "task:a:event:completed"
    dependency_late = _replace_event(plan, dependency_completion, logical_time=6)

    dependency = validate_formal_plan(dependency_late, formulas)
    assert dependency.status is PlanValidationStatus.VIOLATED
    assert dependency.outcome is PlanValidationOutcome.COUNTERMODEL
    assert dependency.countermodel is not None
    assert (
        PlanValidationResult.from_dict(dependency.to_record()).countermodel
        == dependency.countermodel
    )
    assert PlanFindingCode.DEPENDENCY_NOT_READY in {
        item.code for item in dependency.findings
    }

    unauthorized = _replace_event(plan, "task:b:event:started", actor_id="supervisor")
    authority = validate_formal_plan(unauthorized, formulas)
    assert authority.outcome is PlanValidationOutcome.COUNTERMODEL
    assert PlanFindingCode.ACTOR_NOT_ASSIGNED in {
        item.code for item in authority.findings
    }


def test_unique_lease_and_fencing_are_independently_checked() -> None:
    plan, formulas = _compiled()
    event = next(
        item for item in plan.events if item.event_id == "task:a:event:started"
    )
    multiple = _replace_event(
        plan,
        event.event_id,
        metadata={
            **event.metadata,
            "lease_ids": ["lease:a", "lease:other"],
            "fencing_tokens": ["7", "8"],
        },
    )
    duplicate_result = validate_formal_plan(multiple, formulas)
    assert duplicate_result.outcome is PlanValidationOutcome.CONTRADICTION
    assert PlanFindingCode.MULTIPLE_ACTIVE_LEASES in {
        item.code for item in duplicate_result.findings
    }

    stale = _replace_event(
        plan,
        event.event_id,
        metadata={
            **event.metadata,
            "current_fencing_token": 8,
            "mutation_fencing_token": 7,
        },
    )
    stale_result = validate_formal_plan(stale, formulas)
    assert stale_result.outcome is PlanValidationOutcome.COUNTERMODEL
    assert PlanFindingCode.STALE_FENCING_TOKEN in {
        item.code for item in stale_result.findings
    }


def test_missing_evidence_has_a_concrete_countermodel() -> None:
    plan, formulas = _compiled()
    task = next(item for item in plan.tasks if item.task_id == "task:b")
    requirement_id = task.evidence_requirement_ids[0]
    requirements = tuple(
        replace(
            item,
            fallback_check_ids=(),
            source_scope_ids=(),
            metadata={**item.metadata, "available": False},
        )
        if item.requirement_id == requirement_id
        else item
        for item in plan.evidence_requirements
    )

    result = validate_formal_plan(
        replace(plan, evidence_requirements=requirements), formulas
    )

    assert result.outcome is PlanValidationOutcome.COUNTERMODEL
    finding = next(
        item
        for item in result.findings
        if item.code is PlanFindingCode.REQUIRED_EVIDENCE_UNAVAILABLE
    )
    assert requirement_id in finding.subject_ids
    assert result.countermodel is not None
    assert result.countermodel.states[-1].logical_time == finding.logical_time


def test_illegal_transition_missing_terminal_and_forbidden_merge_are_witnessed() -> (
    None
):
    plan, formulas = _compiled()
    illegal = _replace_event(plan, "task:a:event:assigned", kind=EventKind.STARTED)
    illegal_result = validate_formal_plan(illegal, formulas)
    assert PlanFindingCode.ILLEGAL_TRANSITION in {
        item.code for item in illegal_result.findings
    }

    nonterminal = _replace_event(
        plan, "task:b:event:completed", kind=EventKind.EVIDENCE_PRODUCED
    )
    terminal_result = validate_formal_plan(nonterminal, formulas)
    assert PlanFindingCode.TERMINAL_OUTCOME_MISSING in {
        item.code for item in terminal_result.findings
    }

    completion_effect = next(
        item
        for item in plan.effects
        if item.task_id == "task:b" and item.operation is EffectOperation.ASSIGN
    )
    effects = tuple(
        replace(item, value="merged_with_conflicts")
        if item.effect_id == completion_effect.effect_id
        else item
        for item in plan.effects
    )
    merge_result = validate_formal_plan(replace(plan, effects=effects), formulas)
    assert PlanFindingCode.FORBIDDEN_MERGE_STATE in {
        item.code for item in merge_result.findings
    }


def test_unsupported_operator_timeout_cancellation_and_incomplete_are_distinct() -> (
    None
):
    plan, formulas = _compiled()
    ready = atom(
        ReviewedPredicate.TASK_READY,
        constant(TermSort.TASK, "task:a"),
    )
    unsupported = DCEC.belief("agent:a", ready, 0)
    unsupported_result = validate_formal_plan(plan, (*formulas, unsupported))
    assert unsupported_result.status is PlanValidationStatus.UNSUPPORTED
    assert unsupported_result.outcome is PlanValidationOutcome.UNSUPPORTED_OPERATOR

    timed_out = validate_formal_plan(
        plan, formulas, bounds=ValidationBounds(timeout_ms=0)
    )
    assert timed_out.status is PlanValidationStatus.TIMED_OUT
    assert timed_out.outcome is PlanValidationOutcome.TIMEOUT

    token = CancellationToken()
    token.cancel()

    def must_not_be_consumed():
        raise AssertionError("cancelled validation consumed formula input")
        yield  # pragma: no cover

    cancelled = validate_formal_plan(
        plan, must_not_be_consumed(), cancellation_token=token
    )
    assert cancelled.status is PlanValidationStatus.CANCELLED
    assert cancelled.outcome is PlanValidationOutcome.CANCELLED

    incomplete_search = validate_formal_plan(
        plan, formulas, bounds=ValidationBounds(max_search_nodes=1)
    )
    assert incomplete_search.status is PlanValidationStatus.INCOMPLETE
    assert incomplete_search.outcome is PlanValidationOutcome.INCOMPLETE_SEARCH
    assert PlanFindingCode.SEARCH_BOUND_EXHAUSTED in {
        item.code for item in incomplete_search.findings
    }

    incomplete_horizon = validate_formal_plan(
        plan,
        formulas,
        bounds=ValidationBounds(max_trace_steps=2),
    )
    assert incomplete_horizon.status is PlanValidationStatus.INCOMPLETE
    assert incomplete_horizon.outcome is PlanValidationOutcome.INCOMPLETE_SEARCH
    assert incomplete_horizon.bounds.truncated_dimensions == ("trace_steps",)


def test_domain_resource_exhaustion_is_not_incomplete_search() -> None:
    plan, formulas = _compiled()

    result = validate_formal_plan(plan, formulas, bounds=ValidationBounds(max_tasks=1))

    assert result.status is PlanValidationStatus.RESOURCE_EXHAUSTED
    assert result.outcome is PlanValidationOutcome.RESOURCE_EXHAUSTED
    assert PlanFindingCode.DOMAIN_BOUND_EXCEEDED in {
        item.code for item in result.findings
    }

    provider_result = validate_formal_plan(
        plan,
        formulas,
        bounds=ValidationBounds(max_provider_evidence=0),
        proof_evidence=(PlanCheckEvidence(evidence_id="native", backend_id="tdfol"),),
    )
    assert provider_result.outcome is PlanValidationOutcome.RESOURCE_EXHAUSTED


def test_native_success_never_promotes_without_exact_accepted_reconstruction() -> None:
    plan, formulas = _compiled()
    baseline = validate_formal_plan(plan, formulas)
    native = PlanCheckEvidence(
        evidence_id="native:dcec",
        backend_id="dcec",
        backend_role="dcec",
        accepted=True,
        plan_id=baseline.plan_id,
        formula_ids=baseline.formula_ids,
        assumption_ids=baseline.assumption_ids,
        bounds_id=baseline.bounds.bounds_id,
    )

    native_result = validate_formal_plan(plan, formulas, proof_evidence=(native,))

    assert native_result.consistency_level is PlanConsistencyLevel.BOUNDED_CONSISTENT
    assert native_result.plan_check_only
    assert native_result.evidence[0].native_plan_check_only

    wrong_kernel = replace(
        native,
        evidence_id="kernel:wrong",
        backend_id="lean",
        backend_role="kernel",
        reconstructed=True,
        formula_ids=baseline.formula_ids[:-1],
    )
    wrong_result = validate_formal_plan(plan, formulas, proof_evidence=(wrong_kernel,))
    assert wrong_result.consistency_level is PlanConsistencyLevel.BOUNDED_CONSISTENT

    exact_kernel = replace(
        native,
        evidence_id="kernel:exact",
        backend_id="lean",
        backend_role="kernel",
        reconstructed=True,
    )
    kernel_result = validate_formal_plan(plan, formulas, proof_evidence=(exact_kernel,))
    assert kernel_result.consistency_level is PlanConsistencyLevel.KERNEL_VERIFIED
    assert not kernel_result.plan_check_only


def test_fake_clock_and_bounds_make_timeout_deterministic() -> None:
    plan, formulas = _compiled()
    ticks = iter((0.0, 0.002, 0.004, 0.006))
    validator = FormalPlanValidator(
        ValidationBounds(timeout_ms=1),
        clock=lambda: next(ticks, 0.006),
    )

    result = validator.validate(plan, formulas)

    assert result.outcome is PlanValidationOutcome.TIMEOUT
    assert result.validation_id == validator.validate(plan, formulas).validation_id


def test_subgoal_hierarchy_has_deterministic_witnesses_and_legacy_compatibility():
    legacy, formulas = _compiled()
    legacy_result = validate_formal_plan(legacy, formulas)
    plan, formulas = _with_subgoals(legacy, formulas)

    result = validate_formal_plan(plan, formulas)
    automatic = validate_formal_plan(plan)

    assert legacy_result.status is PlanValidationStatus.CONSISTENT
    assert result.status is automatic.status is PlanValidationStatus.CONSISTENT
    assert result.bounds.domain_sizes["subgoals"] == 2
    assert PlanCheckKind.SUBGOAL_WITNESS in result.checks_performed
    assert PlanCheckKind.PARENT_REFINEMENT in result.checks_performed


def test_parent_refinement_and_subgoal_liveness_have_bounded_countermodels():
    plan, formulas = _compiled()
    plan, formulas = _with_subgoals(plan, formulas)
    false_parent = TDFOL.goal_satisfaction(
        "goal:without-a-trace-witness", plan.trace_bound
    )
    parent_violation = replace(
        plan,
        goals=(
            replace(
                plan.goals[0],
                satisfaction_formula_id=false_parent.formula_id,
            ),
        ),
        formulas=(*plan.formulas, false_parent),
    )

    result = validate_formal_plan(
        parent_violation, (*formulas, false_parent)
    )

    assert result.outcome is PlanValidationOutcome.COUNTERMODEL
    assert PlanFindingCode.PARENT_REFINEMENT_VIOLATED in {
        item.code for item in result.findings
    }

    equivalent, equivalent_formulas = _with_subgoals(
        *_compiled(), equivalent_second=True
    )
    without_child_work = replace(
        equivalent,
        tasks=tuple(
            replace(item, subgoal_id="")
            if item.task_id == "task:b"
            else item
            for item in equivalent.tasks
        ),
    )
    liveness = validate_formal_plan(without_child_work, equivalent_formulas)

    assert liveness.outcome is PlanValidationOutcome.COUNTERMODEL
    assert {
        PlanFindingCode.SUBGOAL_NOT_SATISFIED,
        PlanFindingCode.EQUIVALENT_REFINEMENT_VIOLATED,
    }.issubset({item.code for item in liveness.findings})


def test_subgoal_dependency_readiness_returns_a_concrete_countermodel():
    plan, formulas = _compiled()
    plan, formulas = _with_subgoals(plan, formulas)
    first_completion = next(
        item
        for item in plan.events
        if item.task_id == "task:a" and item.kind is EventKind.COMPLETED
    )
    plan = _replace_event(
        plan, first_completion.event_id, logical_time=plan.trace_bound
    )

    result = validate_formal_plan(plan, formulas)

    assert result.outcome is PlanValidationOutcome.COUNTERMODEL
    assert PlanFindingCode.SUBGOAL_DEPENDENCY_NOT_READY in {
        item.code for item in result.findings
    }
    assert result.countermodel is not None
    assert result.countermodel.states[-1].satisfied_subgoal_ids == ()


def test_stale_and_circular_subgoal_evidence_fail_closed():
    plan, formulas = _compiled()
    plan, formulas = _with_subgoals(plan, formulas)
    first_task = next(item for item in plan.tasks if item.task_id == "task:a")
    requirement_id = first_task.evidence_requirement_ids[0]
    requirement = next(
        item
        for item in plan.evidence_requirements
        if item.requirement_id == requirement_id
    )
    completed = next(
        item
        for item in plan.events
        if item.task_id == first_task.task_id and item.kind is EventKind.COMPLETED
    )
    producer = PlanEvent(
        event_id="task:a:event:evidence",
        kind=EventKind.EVIDENCE_PRODUCED,
        actor_id=completed.actor_id,
        task_id=first_task.task_id,
        logical_time=completed.logical_time - 1,
        provenance_ids=(requirement_id,),
        metadata={"requirement_ids": [requirement_id]},
    )
    tasks = tuple(
        replace(item, event_ids=(*item.event_ids, producer.event_id))
        if item.task_id == first_task.task_id
        else item
        for item in plan.tasks
    )
    requirements = tuple(
        replace(item, freshness_seconds=0)
        if item.requirement_id == requirement_id
        else item
        for item in plan.evidence_requirements
    )
    stale = validate_formal_plan(
        replace(
            plan,
            tasks=tasks,
            events=(*plan.events, producer),
            evidence_requirements=requirements,
        ),
        formulas,
    )
    assert PlanFindingCode.STALE_EVIDENCE in {
        item.code for item in stale.findings
    }

    circular_requirements = tuple(
        replace(
            item,
            fallback_check_ids=(requirement_id,),
            source_scope_ids=(),
            metadata={**item.metadata, "available": True},
        )
        if item.requirement_id == requirement_id
        else item
        for item in plan.evidence_requirements
    )
    circular = validate_formal_plan(
        replace(plan, evidence_requirements=circular_requirements), formulas
    )
    assert circular.outcome is PlanValidationOutcome.COUNTERMODEL
    assert PlanFindingCode.CIRCULAR_EVIDENCE in {
        item.code for item in circular.findings
    }


def test_scope_only_and_future_dated_evidence_cannot_witness_a_subgoal():
    plan, formulas = _with_subgoals(*_compiled())
    first_task = next(item for item in plan.tasks if item.task_id == "task:a")
    requirement_id = first_task.evidence_requirement_ids[0]
    requirements = tuple(
        replace(item, fallback_check_ids=())
        if item.requirement_id == requirement_id
        else item
        for item in plan.evidence_requirements
    )

    scope_only = validate_formal_plan(
        replace(plan, evidence_requirements=requirements), formulas
    )

    assert PlanFindingCode.REQUIRED_EVIDENCE_UNAVAILABLE in {
        item.code for item in scope_only.findings
    }

    completion = next(
        item
        for item in plan.events
        if item.task_id == first_task.task_id and item.kind is EventKind.COMPLETED
    )
    future_dated = validate_formal_plan(
        replace(
            plan,
            evidence_requirements=requirements,
            metadata={
                **plan.metadata,
                "available_evidence_ids": [requirement_id],
                "evidence_observed_at": {
                    requirement_id: completion.logical_time + 1
                },
            },
        ),
        formulas,
    )

    assert PlanFindingCode.STALE_EVIDENCE in {
        item.code for item in future_dated.findings
    }


def test_subgoal_bound_and_timeout_remain_distinct_from_assurance():
    plan, formulas = _compiled()
    plan, formulas = _with_subgoals(plan, formulas)

    exhausted = validate_formal_plan(
        plan, formulas, bounds=ValidationBounds(max_subgoals=1)
    )
    timed_out = validate_formal_plan(
        plan, formulas, bounds=ValidationBounds(timeout_ms=0)
    )

    assert exhausted.outcome is PlanValidationOutcome.RESOURCE_EXHAUSTED
    assert timed_out.outcome is PlanValidationOutcome.TIMEOUT
    assert timed_out.consistency_level is PlanConsistencyLevel.INCONCLUSIVE
