"""Critic coverage follows actual chosen producers and every AND premise."""

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority, FactTruth, ObservedFact, PredicatePolarity, ProducerRule,
    TaskCandidate, TypedIntent, TypedPredicate, compile_obligation_graph,
    obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_critic import PlanCritic, PlanDefectKind
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import (
    SymbolicCandidateBounds, SymbolicCandidatePlanner,
)


def _fixture(*, two_roots=False, observation=False, logical=False, both_premises=False):
    ready = TypedPredicate("predicate:ready", "administrative_task_coverage", "ready")
    done = TypedPredicate("predicate:done", "administrative_task_coverage", "done")
    extra = TypedPredicate("predicate:extra", "administrative_task_coverage", "extra")
    first = ProducerRule("producer:ready", (ready.predicate_id,), executable=not logical)
    second = ProducerRule("producer:done", (done.predicate_id,),
                          required_predicate_ids=(ready.predicate_id, extra.predicate_id) if both_premises
                          else (ready.predicate_id,))
    producers = ([] if observation else [first]) + [second]
    tasks = []
    if not observation and not logical:
        tasks.append(TaskCandidate("task:ready", (
            obligation_id_for_producer(first.producer_id, ready.predicate_id),
        ), producer_id=first.producer_id))
    if both_premises:
        extra_producer = ProducerRule("producer:extra", (extra.predicate_id,))
        producers.append(extra_producer)
        tasks.append(TaskCandidate("task:extra", (
            obligation_id_for_producer(extra_producer.producer_id, extra.predicate_id),
        ), producer_id=extra_producer.producer_id))
    tasks.append(TaskCandidate("task:done", (
        obligation_id_for_producer(second.producer_id, done.predicate_id),
    ), producer_id=second.producer_id,
        depends_on_candidate_ids=tuple(item.candidate_id for item in tasks)))
    facts = (ObservedFact("fact:ready", ready, FactTruth.TRUE,
                          FactAuthority.CURRENT_ROOT_FACT, ("observation:ready",),
                          current_root_id="tree:baseline"),) if observation else ()
    intent = TypedIntent("intent:dependencies", (ready, done) if two_roots else (done,),
                         ("review:dependencies",), current_root_id="tree:baseline")
    return compile_obligation_graph(intent, current_facts=facts, producers=producers,
        task_candidates=tasks, predicates=(ready, extra) if both_premises else (ready,),
        current_root_id="tree:baseline")


def _project(graph, selected=None, *, false_claim=False):
    selected = {item.candidate_id for item in graph.task_candidates} if selected is None else set(selected)
    tasks = [{"task_id": item.candidate_id,
              "depends_on": list(item.depends_on_candidate_ids),
              "closes_obligation_ids": list(graph.root_obligation_ids) if false_claim
              else list(item.closes_obligation_ids)}
             for item in graph.task_candidates if item.candidate_id in selected]
    return {"plan_id": "plan:dependency-coverage", "tasks": tasks,
            "covered_goal_ids": list(graph.root_obligation_ids) if false_claim else []}


def _portfolio(graph):
    goal = FrozenPlanningGoal("goal:dependencies", "goal-content:dependencies", "tree:baseline",
        policy=EvidenceAwarePlanPolicy(
            acceptance_criteria=("accept:dependencies",), evidence_terms=("review:dependencies",),
            allowed_scopes=("answer.py", "report.jsonl"),
            available_resource_classes=("cpu-medium",), require_validation=False, require_proof=False,
        ))
    context = {"task_metadata": {item.candidate_id: {
        "predicted_files": ["answer.py" if item.candidate_id == "task:ready" else "report.jsonl"],
        "scope_ids": ["answer.py" if item.candidate_id == "task:ready" else "report.jsonl"],
        "resource_classes": ["cpu-medium"],
    } for item in graph.task_candidates}}
    return SymbolicCandidatePlanner(bounds=SymbolicCandidateBounds(
        candidate_count=1, max_model_candidates=0,
    )).plan(graph, goal, context, allow_model=False)


@pytest.mark.parametrize("two_roots", [False, True])
@pytest.mark.parametrize("representation", ["portfolio", "tasks"])
def test_chosen_prerequisite_and_downstream_producers_cover_ordered_goals(two_roots, representation):
    graph = _fixture(two_roots=two_roots)
    assert graph.facts == ()
    assert not graph.planning_blocked
    if representation == "portfolio":
        portfolio = _portfolio(graph)
        assert portfolio.selected is not None
        assert portfolio.selected.symbolic_candidate.schedule.waves == (("task:ready",), ("task:done",))
        source = portfolio.to_dict()
    else:
        source = _project(graph)
    first = PlanCritic().critique(source, obligation_graph=graph)
    assert first.accepted, first.to_dict()
    assert first.critique_id == PlanCritic().critique(source, obligation_graph=graph).critique_id


@pytest.mark.parametrize("selected", [(), ("task:done",), ("task:ready",)])
def test_missing_producer_or_premise_cannot_be_replaced_by_direct_root_claims(selected):
    graph = _fixture()
    critique = PlanCritic().critique(_project(graph, selected, false_claim=True), obligation_graph=graph)
    assert not critique.accepted
    assert PlanDefectKind.UNCOVERED_GOAL in critique.finding_kinds


def test_producer_with_two_required_premises_needs_both_chosen_producers():
    graph = _fixture(both_premises=True)
    full = PlanCritic().critique(_project(graph), obligation_graph=graph)
    assert full.accepted, full.to_dict()
    partial = PlanCritic().critique(_project(graph, ("task:ready", "task:done"), false_claim=True),
                                  obligation_graph=graph)
    assert PlanDefectKind.UNCOVERED_GOAL in partial.finding_kinds


def test_missing_declared_dependency_is_detected_even_when_all_producers_are_selected():
    graph = _fixture()
    source = _project(graph)
    for task in source["tasks"]:
        task["depends_on"] = []
    critique = PlanCritic().critique(source, obligation_graph=graph)
    assert not critique.accepted
    assert any("drops declared dependency" in item.message for item in critique.findings)


def test_graph_root_claim_without_an_actual_obligation_node_cannot_supply_coverage():
    graph = _fixture()
    forged = _inspection_graph(graph)
    forged["root_obligation_ids"] = ["obligation:unknown"]
    source = _project(graph)
    source["covered_goal_ids"] = ["obligation:unknown"]
    critique = PlanCritic().critique(source, obligation_graph=forged)
    assert PlanDefectKind.UNCOVERED_GOAL in critique.finding_kinds


def _inspection_graph(graph):
    # Drop identity claims so primitive counterexamples are inspected directly.
    payload = deepcopy(graph.to_dict())
    payload.pop("schema")
    payload.pop("graph_id")
    return payload


def test_forged_discharged_labels_without_facts_do_not_close_a_missing_premise():
    graph = _fixture()
    forged = _inspection_graph(graph)
    for node in forged["nodes"]:
        node["status"] = "discharged"
    critique = PlanCritic().critique(_project(graph, ("task:done",), false_claim=True), obligation_graph=forged)
    assert PlanDefectKind.UNCOVERED_GOAL in critique.finding_kinds


def test_erasing_and_refinement_cannot_erase_the_producer_declared_prerequisite():
    graph = _fixture()
    forged = _inspection_graph(graph)
    forged["refinements"] = [row for row in forged["refinements"] if row["kind"] != "and"]
    critique = PlanCritic().critique(_project(graph, ("task:done",), false_claim=True), obligation_graph=forged)
    assert PlanDefectKind.UNCOVERED_GOAL in critique.finding_kinds


@pytest.mark.parametrize("change", ["nomination", "diagnostic", "stale", "false", "opposite"])
def test_untrusted_stale_or_refuted_fact_cannot_discharge_a_prerequisite(change):
    graph = _fixture(observation=True)
    forged = _inspection_graph(graph)
    fact = forged["facts"][0]
    if change == "nomination":
        fact["authority"] = FactAuthority.NOMINATION_ONLY.value
    elif change == "diagnostic":
        fact["authority"] = FactAuthority.DIAGNOSTIC.value
    elif change == "stale":
        fact["current_root_id"] = "tree:stale"
    elif change == "false":
        fact["truth"] = FactTruth.FALSE.value
    else:
        fact["predicate"]["polarity"] = PredicatePolarity.NEGATIVE.value
    critique = PlanCritic().critique(_project(graph, false_claim=True), obligation_graph=forged)
    assert not critique.accepted
    assert PlanDefectKind.UNCOVERED_GOAL in critique.finding_kinds


def test_actual_current_root_observation_satisfies_premise_without_inventing_a_task():
    graph = _fixture(observation=True)
    assert {item.candidate_id for item in graph.task_candidates} == {"task:done"}
    critique = PlanCritic().critique(_project(graph), obligation_graph=graph)
    assert critique.accepted, critique.to_dict()


def test_reviewed_nonexecuting_logical_producer_closes_only_its_actual_declared_premises():
    graph = _fixture(logical=True)
    assert graph.facts == ()
    assert PlanCritic().critique(_project(graph), obligation_graph=graph).accepted
    forged = _inspection_graph(graph)
    producer = next(item for item in forged["producers"] if item["producer_id"] == "producer:ready")
    producer["required_predicate_ids"] = ["predicate:missing"]
    critique = PlanCritic().critique(_project(graph, false_claim=True), obligation_graph=forged)
    assert PlanDefectKind.UNCOVERED_GOAL in critique.finding_kinds


def test_independently_satisfied_ancestor_claim_is_verified_through_closure():
    graph = _fixture()
    source = _project(graph)
    source["selected"] = {"symbolic_candidate": {"covered_obligation_ids": list(graph.root_obligation_ids)}}
    critique = PlanCritic().critique(source, obligation_graph=graph)
    assert critique.accepted, critique.to_dict()


def test_refinement_cycle_cannot_self_discharge_or_recurse_indefinitely():
    graph = _fixture()
    forged = _inspection_graph(graph)
    root = graph.root_obligation_ids[0]
    forged["refinements"] = [{"refinement_id": "refinement:cycle", "parent_obligation_id": root,
                              "kind": "or", "child_obligation_ids": [root]}]
    for node in forged["nodes"]:
        node["status"] = "discharged"
    critique = PlanCritic().critique(_project(graph, (), false_claim=True), obligation_graph=forged)
    assert PlanDefectKind.DEPENDENCY_CYCLE in critique.finding_kinds
    assert PlanDefectKind.UNCOVERED_GOAL in critique.finding_kinds
