"""Contract tests for bounded AND/OR obligation graph compilation (PDR-022)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    AssumptionBinding,
    AssumptionStatus,
    CompilationBounds,
    FactAuthority,
    FactTruth,
    InvalidationSelector,
    InvalidationSelectorKind,
    ObligationCompilationError,
    ObligationGraph,
    ObligationGraphCompiler,
    ObligationGraphDecision,
    ObligationIssueKind,
    ObligationNodeKind,
    ObligationStatus,
    ObservedFact,
    PredicatePolarity,
    ProducerRule,
    RefinementKind,
    SemanticSupport,
    TaskCandidate,
    TypedIntent,
    TypedPredicate,
    compile_obligation_graph,
    obligation_id_for_predicate,
    obligation_id_for_producer,
)


def _predicate(
    predicate_id: str,
    *,
    subject: str | None = None,
    polarity: PredicatePolarity = PredicatePolarity.POSITIVE,
    support: SemanticSupport = SemanticSupport.REVIEWED,
    assumptions: tuple[str, ...] = (),
    invalidators: tuple[InvalidationSelector, ...] = (),
    property_id: str = "",
) -> TypedPredicate:
    return TypedPredicate(
        predicate_id=predicate_id,
        predicate_type="behavior_state",
        subject_ref=subject or predicate_id,
        polarity=polarity,
        support=support,
        property_id=property_id,
        provenance_refs=(f"contract:{predicate_id}",),
        assumption_refs=assumptions,
        invalidation_selectors=invalidators,
        proof_requirement_refs=(f"proof:{predicate_id}",),
        validation_requirement_refs=(f"validation:{predicate_id}",),
    )


def _intent(*predicates: TypedPredicate) -> TypedIntent:
    return TypedIntent(
        intent_id="intent:test",
        desired_predicates=predicates,
        source_refs=("intent-source:test",),
        current_root_id="tree:current",
    )


def _fact(
    predicate: TypedPredicate,
    *,
    fact_id: str | None = None,
    truth: FactTruth = FactTruth.TRUE,
    authority: FactAuthority = FactAuthority.CURRENT_ROOT_FACT,
    root: str = "tree:current",
) -> ObservedFact:
    return ObservedFact(
        fact_id=fact_id or f"fact:{predicate.predicate_id}:{truth.value}",
        predicate=predicate,
        truth=truth,
        authority=authority,
        provenance_refs=(f"evidence:{predicate.predicate_id}:{truth.value}",),
        current_root_id=root,
    )


def _candidate(producer: ProducerRule, effect_id: str) -> TaskCandidate:
    return TaskCandidate(
        candidate_id=f"task:{producer.producer_id}",
        closes_obligation_ids=(
            obligation_id_for_producer(producer.producer_id, effect_id),
        ),
        producer_id=producer.producer_id,
        provenance_refs=(f"proposal:{producer.producer_id}",),
    )


def test_authoritative_current_fact_discharges_root_without_inventing_work() -> None:
    goal = _predicate("goal:available")

    graph = compile_obligation_graph(_intent(goal), (_fact(goal),))

    assert graph.ready
    assert graph.complete
    assert graph.node(obligation_id_for_predicate(goal.predicate_id)).status is (
        ObligationStatus.DISCHARGED
    )
    assert graph.task_candidates == ()
    assert graph.uncovered_leaf_obligation_ids == ()


def test_backward_chaining_distinguishes_or_strategies_from_and_requirements() -> None:
    goal = _predicate("goal:release")
    built = _predicate("premise:built")
    tested = _predicate("premise:tested")
    direct = ProducerRule(
        producer_id="producer:direct",
        effect_predicate_ids=(goal.predicate_id,),
        required_predicate_ids=(built.predicate_id, tested.predicate_id),
        provenance_refs=("operator:direct",),
        proof_requirement_refs=("proof:direct",),
    )
    fallback = ProducerRule(
        producer_id="producer:fallback",
        effect_predicate_ids=(goal.predicate_id,),
        required_predicate_ids=(tested.predicate_id,),
        provenance_refs=("operator:fallback",),
        validation_requirement_refs=("validation:fallback",),
    )

    graph = compile_obligation_graph(
        _intent(goal),
        (_fact(built), _fact(tested)),
        (direct, fallback),
        predicates=(built, tested),
        task_candidates=(
            _candidate(direct, goal.predicate_id),
            _candidate(fallback, goal.predicate_id),
        ),
    )

    root = obligation_id_for_predicate(goal.predicate_id)
    root_refinement = graph.refinements_for(root)
    assert len(root_refinement) == 1
    assert root_refinement[0].kind is RefinementKind.OR
    assert set(root_refinement[0].child_obligation_ids) == {
        obligation_id_for_producer(direct.producer_id, goal.predicate_id),
        obligation_id_for_producer(fallback.producer_id, goal.predicate_id),
    }
    for producer in (direct, fallback):
        producer_id = obligation_id_for_producer(
            producer.producer_id, goal.predicate_id
        )
        refinement = graph.refinements_for(producer_id)
        assert len(refinement) == 1
        assert refinement[0].kind is RefinementKind.AND
        assert graph.node(producer_id).provenance_refs
    assert graph.ready
    assert not graph.complete
    assert graph.uncovered_leaf_obligation_ids == ()


def test_uncovered_leaf_blocks_candidate_generation() -> None:
    graph = compile_obligation_graph(_intent(_predicate("goal:uncovered")))

    assert graph.planning_blocked
    assert graph.uncovered_leaf_obligation_ids == (
        obligation_id_for_predicate("goal:uncovered"),
    )
    assert graph.issues_of_kind(ObligationIssueKind.UNCOVERED_LEAF)


def test_every_task_candidate_must_close_an_existing_named_leaf() -> None:
    goal = _predicate("goal:target")
    producer = ProducerRule(
        producer_id="producer:target",
        effect_predicate_ids=(goal.predicate_id,),
        provenance_refs=("operator:target",),
    )
    invalid = TaskCandidate(
        candidate_id="task:invalid",
        # The root is refined and therefore is not an executable leaf.
        closes_obligation_ids=(obligation_id_for_predicate(goal.predicate_id),),
    )

    graph = compile_obligation_graph(
        _intent(goal),
        producers=(producer,),
        task_candidates=(invalid,),
    )

    assert graph.planning_blocked
    assert graph.issues_of_kind(ObligationIssueKind.INVALID_TASK_CLOSURE)
    assert graph.issues_of_kind(ObligationIssueKind.UNCOVERED_LEAF)
    with pytest.raises(ObligationCompilationError):
        TaskCandidate(candidate_id="task:no-closure", closes_obligation_ids=())


def test_cycles_are_retained_as_explicit_blockers() -> None:
    first = _predicate("predicate:first")
    second = _predicate("predicate:second")
    first_from_second = ProducerRule(
        producer_id="producer:first-from-second",
        effect_predicate_ids=(first.predicate_id,),
        required_predicate_ids=(second.predicate_id,),
        provenance_refs=("operator:first",),
    )
    second_from_first = ProducerRule(
        producer_id="producer:second-from-first",
        effect_predicate_ids=(second.predicate_id,),
        required_predicate_ids=(first.predicate_id,),
        provenance_refs=("operator:second",),
    )

    graph = compile_obligation_graph(
        _intent(first),
        producers=(first_from_second, second_from_first),
        predicates=(second,),
    )

    assert graph.decision is ObligationGraphDecision.BLOCKED
    cycles = graph.issues_of_kind(ObligationIssueKind.CYCLE)
    assert len(cycles) == 1
    assert set(cycles[0].predicate_ids) == {
        first.predicate_id,
        second.predicate_id,
    }


def test_contradictory_facts_and_desired_predicates_fail_closed() -> None:
    positive = _predicate("predicate:enabled", subject="feature")
    negative = _predicate(
        "predicate:not-enabled",
        subject="feature",
        polarity=PredicatePolarity.NEGATIVE,
    )
    graph = compile_obligation_graph(
        _intent(positive, negative),
        (_fact(positive), _fact(negative)),
    )

    assert graph.planning_blocked
    contradictions = graph.issues_of_kind(ObligationIssueKind.CONTRADICTION)
    assert contradictions
    assert all(item.severity.value == "error" for item in contradictions)

    # A false observation of a positive atom is the same contradiction as a
    # true observation of its negative counterpart.
    false_fact_graph = compile_obligation_graph(
        _intent(positive),
        (
            _fact(positive, fact_id="fact:true"),
            _fact(positive, fact_id="fact:false", truth=FactTruth.FALSE),
        ),
    )
    assert false_fact_graph.issues_of_kind(ObligationIssueKind.CONTRADICTION)


def test_inconsistent_or_refuted_producer_premises_are_detected() -> None:
    goal = _predicate("goal:repair")
    required = _predicate("premise:safe", subject="mutation")
    forbidden = _predicate(
        "premise:not-safe",
        subject="mutation",
        polarity=PredicatePolarity.NEGATIVE,
    )
    producer = ProducerRule(
        producer_id="producer:inconsistent",
        effect_predicate_ids=(goal.predicate_id,),
        required_predicate_ids=(required.predicate_id, forbidden.predicate_id),
        provenance_refs=("operator:inconsistent",),
    )

    graph = compile_obligation_graph(
        _intent(goal),
        producers=(producer,),
        predicates=(required, forbidden),
    )

    issues = graph.issues_of_kind(ObligationIssueKind.INCONSISTENT_PREMISE)
    assert issues
    producer_node = graph.node(
        obligation_id_for_producer(producer.producer_id, goal.predicate_id)
    )
    assert producer_node.status is ObligationStatus.BLOCKED


@pytest.mark.parametrize(
    "support", [SemanticSupport.UNSUPPORTED, SemanticSupport.UNKNOWN]
)
def test_unsupported_semantics_remain_review_unknown(
    support: SemanticSupport,
) -> None:
    goal = _predicate("goal:opaque", support=support)

    graph = compile_obligation_graph(_intent(goal))

    assert graph.decision is ObligationGraphDecision.REVIEW_REQUIRED
    assert graph.review_required
    assert not graph.complete
    assert graph.node(obligation_id_for_predicate(goal.predicate_id)).status is (
        ObligationStatus.REVIEW
    )
    assert graph.issues_of_kind(ObligationIssueKind.UNSUPPORTED_SEMANTICS)
    # Unknown semantics are never converted into a generated implementation
    # task or misreported as an ordinary uncovered leaf.
    assert graph.task_candidates == ()
    assert not graph.issues_of_kind(ObligationIssueKind.UNCOVERED_LEAF)


def test_assumptions_bind_provenance_and_exact_invalidation_selectors() -> None:
    selector = InvalidationSelector(
        selector_id="selector:policy",
        kind=InvalidationSelectorKind.POLICY,
        value_ref="policy:v1",
        provenance_refs=("policy-receipt:v1",),
    )
    assumption = AssumptionBinding(
        assumption_id="assumption:policy-stable",
        statement_ref="statement:policy-stable",
        provenance_refs=("review:assumption",),
        invalidation_selectors=(selector,),
    )
    goal = _predicate(
        "goal:authorized",
        assumptions=(assumption.assumption_id,),
    )
    producer = ProducerRule(
        producer_id="producer:authorized",
        effect_predicate_ids=(goal.predicate_id,),
        provenance_refs=("operator:authorized",),
        task_candidate_ids=("task:authorized",),
    )

    graph = compile_obligation_graph(
        _intent(goal),
        producers=(producer,),
        assumptions=(assumption,),
    )

    assert graph.ready
    root = graph.node(obligation_id_for_predicate(goal.predicate_id))
    assert root.assumption_refs == (assumption.assumption_id,)
    assert root.invalidation_selector_ids == (selector.selector_id,)
    assert graph.invalidated_by({"policy_ids": ["policy:v1"]})
    assert graph.invalidated_by({"policy_ids": ["policy:v2"]}) == ()

    invalid_graph = compile_obligation_graph(
        _intent(goal),
        producers=(producer,),
        assumptions=(replace(assumption, status=AssumptionStatus.INVALID),),
    )
    assert invalid_graph.planning_blocked
    assert invalid_graph.issues_of_kind(ObligationIssueKind.INVALID_ASSUMPTION)


def test_stale_and_nomination_only_facts_cannot_discharge_semantics() -> None:
    goal = _predicate("goal:current")
    graph = compile_obligation_graph(
        _intent(goal),
        (
            _fact(
                goal,
                fact_id="fact:stale",
                root="tree:stale",
            ),
            _fact(
                goal,
                fact_id="fact:nomination",
                authority=FactAuthority.NOMINATION_ONLY,
            ),
        ),
    )

    assert graph.review_required
    assert not graph.complete
    assert graph.issues_of_kind(ObligationIssueKind.STALE_FACT)
    assert graph.issues_of_kind(ObligationIssueKind.NON_AUTHORITATIVE_FACT)


def test_depth_bound_stops_backward_chaining_as_review_not_success() -> None:
    goal = _predicate("goal:bounded")
    premise = _predicate("premise:beyond-bound")
    producer = ProducerRule(
        producer_id="producer:bounded",
        effect_predicate_ids=(goal.predicate_id,),
        required_predicate_ids=(premise.predicate_id,),
        provenance_refs=("operator:bounded",),
        task_candidate_ids=("task:bounded",),
    )

    graph = ObligationGraphCompiler(
        bounds=CompilationBounds(max_depth=1, max_nodes=32)
    ).compile(
        _intent(goal),
        producers=(producer,),
        predicates=(premise,),
    )

    assert graph.review_required
    assert graph.issues_of_kind(ObligationIssueKind.BOUND_EXCEEDED)
    assert not graph.complete


def test_upstream_query_and_evidence_incompleteness_are_graph_blockers() -> None:
    goal = _predicate("goal:upstream")
    graph = compile_obligation_graph(
        _intent(goal),
        task_candidates=(
            TaskCandidate(
                candidate_id="task:upstream",
                closes_obligation_ids=(obligation_id_for_predicate(goal.predicate_id),),
            ),
        ),
        query_plan={"decision": "blocked", "plan_id": "query-plan:1"},
        evidence_bundle={"decision": "rejected", "bundle_id": "bundle:1"},
    )

    assert graph.planning_blocked
    assert graph.issues_of_kind(ObligationIssueKind.INCOMPLETE_QUERY_PLAN)
    assert graph.issues_of_kind(ObligationIssueKind.INCOMPLETE_EVIDENCE)

    ready = compile_obligation_graph(
        _intent(goal),
        task_candidates=(
            TaskCandidate(
                candidate_id="task:upstream",
                closes_obligation_ids=(obligation_id_for_predicate(goal.predicate_id),),
            ),
        ),
        query_plan={"decision": "ready"},
        evidence_bundle={"decision": "ready"},
    )
    assert ready.ready


def test_property_catalog_unknown_property_remains_review() -> None:
    class Catalog:
        @staticmethod
        def get(property_id: str) -> object | None:
            return object() if property_id == "property:reviewed" else None

    goal = _predicate("goal:catalog", property_id="property:not-reviewed")
    graph = ObligationGraphCompiler(property_catalog=Catalog()).compile(
        _intent(goal)
    )

    assert graph.review_required
    assert graph.node(obligation_id_for_predicate(goal.predicate_id)).status is (
        ObligationStatus.REVIEW
    )
    assert graph.issues_of_kind(ObligationIssueKind.UNSUPPORTED_SEMANTICS)


def test_logic_goal_adapter_preserves_unsupported_and_discharged_status() -> None:
    unsupported = {
        "goal_id": "logic:opaque",
        "positive_statement_ref": "statement:opaque",
        "unsupported_facets": [{"facet_id": "facet:native"}],
        "disposition": "open",
        "proof_status": "unproved",
        "content_id": "logic-goal:opaque",
    }
    graph = compile_obligation_graph(logic_goals=(unsupported,))
    assert graph.review_required
    assert graph.issues_of_kind(ObligationIssueKind.UNSUPPORTED_SEMANTICS)

    discharged = {
        "goal_id": "logic:proved",
        "positive_statement_ref": "statement:proved",
        "unsupported_facets": [],
        "disposition": "discharged",
        "proof_status": "kernel_verified",
        "content_id": "logic-goal:proved",
    }
    proved = compile_obligation_graph(logic_goals=(discharged,))
    assert proved.ready
    assert proved.complete


def test_formal_work_plan_projection_creates_goal_and_task_producer_obligation() -> (
    None
):
    plan = {
        "content_id": "formal-plan:test",
        "repository_tree_id": "tree:current",
        "goals": [
            {
                "goal_id": "formal-goal:release",
                "satisfaction_formula_id": "formula:released",
                "evidence_requirement_ids": ["validation:release"],
            }
        ],
        "subgoals": [],
        "tasks": [
            {
                "task_id": "formal-task:release",
                "goal_id": "formal-goal:release",
                "subgoal_id": "",
                "precondition_ids": [],
                "evidence_requirement_ids": ["validation:release"],
            }
        ],
        "preconditions": [],
        "evidence_requirements": [
            {"requirement_id": "validation:release"}
        ],
    }

    graph = compile_obligation_graph(formal_work_plan=plan)

    assert graph.ready
    assert {item.kind for item in graph.nodes} == {
        ObligationNodeKind.GOAL,
        ObligationNodeKind.PRODUCER,
    }
    assert tuple(item.candidate_id for item in graph.task_candidates) == (
        "formal-task:release",
    )
    producer = next(
        item for item in graph.nodes if item.kind is ObligationNodeKind.PRODUCER
    )
    assert producer.validation_requirement_refs == ("validation:release",)


def test_graph_round_trip_is_deterministic_and_tamper_evident() -> None:
    goal = _predicate("goal:round-trip")
    graph = compile_obligation_graph(
        _intent(goal),
        (_fact(goal),),
    )

    round_trip = ObligationGraph.from_json(graph.to_json())
    assert round_trip == graph
    assert round_trip.graph_id == graph.graph_id
    assert round_trip.to_dict() == graph.to_dict()

    payload = graph.to_dict()
    payload["current_root_id"] = "tree:tampered"
    with pytest.raises(ObligationCompilationError, match="graph_id"):
        ObligationGraph.from_dict(payload)
