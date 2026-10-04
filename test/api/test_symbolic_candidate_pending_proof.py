"""Candidate planning preserves pending proof and internal task ordering."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import (
    FrozenPlanningGoal,
)
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    ProducerRule,
    TaskCandidate,
    TypedIntent,
    TypedPredicate,
    compile_obligation_graph,
    obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import (
    EvidenceAwarePlanPolicy,
    PlanBranch,
    evaluate_evidence_aware_plans,
)
from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import (
    PartialOrderSchedule,
    SymbolicCandidateBounds,
    SymbolicCandidatePlanner,
    SymbolicCandidatePlanningError,
    SymbolicCandidatePortfolio,
)


VALIDATION_COMMAND = json.dumps(
    ["python", "-m", "pytest", "test_answer.py", "-q"], separators=(",", ":")
)


def _portfolio(
    *, require_proof: bool, proof_refs: tuple[str, ...] = (), ordered: bool = False
):
    predicate = TypedPredicate(
        predicate_id="goal:answer",
        predicate_type="output_effect",
        subject_ref="answer.py",
        object_ref="modify",
        provenance_refs=("signed-operation:answer",),
        proof_requirement_refs=proof_refs,
        validation_requirement_refs=("public-answer",),
    )
    producer = ProducerRule(
        producer_id="operation:answer",
        effect_predicate_ids=(predicate.predicate_id,),
        provenance_refs=("signed-operation:answer",),
        proof_requirement_refs=proof_refs,
        validation_requirement_refs=("public-answer",),
        required_predicate_ids=("goal:prepare",) if ordered else (),
    )
    task = TaskCandidate(
        candidate_id="task:answer",
        producer_id=producer.producer_id,
        closes_obligation_ids=(
            obligation_id_for_producer(producer.producer_id, predicate.predicate_id),
        ),
        provenance_refs=("signed-operation:answer",),
        depends_on_candidate_ids=("task:prepare",) if ordered else (),
    )
    preparation = TypedPredicate(
        predicate_id="goal:prepare",
        predicate_type="output_effect",
        subject_ref="prepare.py",
        object_ref="modify",
        provenance_refs=("signed-operation:prepare",),
        validation_requirement_refs=("public-answer",),
    )
    preparation_producer = ProducerRule(
        producer_id="operation:prepare",
        effect_predicate_ids=(preparation.predicate_id,),
        provenance_refs=("signed-operation:prepare",),
    )
    preparation_task = TaskCandidate(
        candidate_id="task:prepare",
        producer_id=preparation_producer.producer_id,
        closes_obligation_ids=(
            obligation_id_for_producer(
                preparation_producer.producer_id, preparation.predicate_id
            ),
        ),
        provenance_refs=("signed-operation:prepare",),
    )
    graph = compile_obligation_graph(
        TypedIntent(
            intent_id="intent:answer",
            desired_predicates=(predicate,),
            source_refs=("source:instruction",),
            current_root_id="tree:baseline",
        ),
        current_facts=(),
        producers=(producer, preparation_producer) if ordered else (producer,),
        task_candidates=(task, preparation_task) if ordered else (task,),
        predicates=(preparation,) if ordered else (),
    )
    goal = FrozenPlanningGoal(
        goal_id="goal:answer",
        goal_content_id="goal-content:answer",
        repository_tree_id="tree:baseline",
        policy=EvidenceAwarePlanPolicy(
            acceptance_criteria=("criterion:answer",),
            evidence_terms=("public-answer",),
            allowed_scopes=("answer.py", "prepare.py") if ordered else ("answer.py",),
            available_resource_classes=("cpu",),
            require_validation=True,
            require_proof=require_proof,
        ),
    )
    context = {
        "repository_paths": ["answer.py", "test_answer.py"],
        "task_metadata": {
            "task:answer": {
                "predicted_files": ["answer.py"],
                "predicted_symbols": ["answer"],
                "scope_ids": ["answer.py"],
                "resource_classes": ["cpu"],
                "validation_commands": [VALIDATION_COMMAND],
            }
        },
    }
    if ordered:
        context["repository_paths"].append("prepare.py")
        context["task_metadata"]["task:prepare"] = {
            "predicted_files": ["prepare.py"],
            "predicted_symbols": ["prepare"],
            "scope_ids": ["prepare.py"],
            "resource_classes": ["cpu"],
            "validation_commands": [VALIDATION_COMMAND],
        }
    return SymbolicCandidatePlanner(
        bounds=SymbolicCandidateBounds(candidate_count=1, max_model_candidates=0)
    ).plan(graph, goal, context, allow_model=False)


def test_candidate_policy_selects_pending_plan_without_fabricated_proof() -> None:
    portfolio = _portfolio(require_proof=False)

    assert portfolio.selected is not None
    branch = portfolio.selected.symbolic_candidate.plan.branch
    assert branch.validation_proof == ()
    assert branch.validation_commands == (VALIDATION_COMMAND,)
    assert not portfolio.provider_usage.attempted
    assert "proof:independent-admission-required" not in portfolio.to_json()
    assert "validation:independent-admission-required" not in portfolio.to_json()
    assert SymbolicCandidatePortfolio.from_json(portfolio.to_json()) == portfolio


def test_proof_policy_rejects_plan_with_no_proof_obligation() -> None:
    portfolio = _portfolio(require_proof=True)

    assert portfolio.selected is None
    assert portfolio.baseline.disposition == "rejected"
    assert portfolio.baseline.symbolic_candidate.plan.branch.validation_proof == ()
    assert not portfolio.baseline.symbolic_candidate.plan.proof_feasible
    assert any("validation_and_proof" in item for item in portfolio.baseline.reason_codes)
    assert "proof:independent-admission-required" not in portfolio.to_json()


def test_required_proof_gate_cannot_be_bypassed_with_feasibility_flag() -> None:
    portfolio = _portfolio(require_proof=False)
    candidate = replace(portfolio.baseline.symbolic_candidate.plan, proof_feasible=True)
    policy = replace(portfolio.request.frozen_goal.policy, require_proof=True)

    evaluated = evaluate_evidence_aware_plans((candidate,), policy=policy)

    assert evaluated.selected is None
    assert len(evaluated.rejected) == 1
    assert "required proof is not feasible" in evaluated.rejected[0].rationale


def test_real_proof_obligation_refs_are_retained() -> None:
    portfolio = _portfolio(require_proof=True, proof_refs=("proof:answer-behavior",))

    assert portfolio.selected is not None
    assert portfolio.selected.symbolic_candidate.plan.branch.validation_proof == (
        "proof:answer-behavior",
    )
    assert "proof:independent-admission-required" not in portfolio.to_json()


def test_plan_branch_round_trips_empty_refs_but_rejects_empty_ref_strings() -> None:
    branch = _portfolio(require_proof=False).baseline.symbolic_candidate.plan.branch

    assert PlanBranch.from_dict(branch.to_dict()).validation_proof == ()
    assert PlanBranch.from_dict(branch.to_dict()) == branch
    with pytest.raises(ValueError, match="validation_proof"):
        replace(branch, validation_proof=("",))


def test_internal_predecessor_is_scheduled_without_becoming_an_observed_fact() -> None:
    portfolio = _portfolio(require_proof=False, ordered=True)

    assert portfolio.selected is not None
    record = portfolio.selected.symbolic_candidate
    assert record.schedule.waves == (("task:prepare",), ("task:answer",))
    assert record.schedule.dependency_edges == (("task:prepare", "task:answer"),)
    assert record.plan.branch.dependencies == ("task:prepare",)
    assert record.plan.dependencies == record.plan.critical_path == ()
    assert portfolio.request.obligation_graph.facts == ()
    assert portfolio.request.frozen_goal.policy.satisfied_dependencies == ()
    assert SymbolicCandidatePortfolio.from_json(portfolio.to_json()) == portfolio


def test_internal_dependency_declaration_cannot_be_dropped() -> None:
    record = _portfolio(require_proof=False, ordered=True).baseline.symbolic_candidate

    with pytest.raises(SymbolicCandidatePlanningError, match="branch dependencies"):
        replace(record, plan=replace(record.plan, branch=replace(record.plan.branch, dependencies=())))
    with pytest.raises(SymbolicCandidatePlanningError, match="externally observed"):
        replace(record, plan=replace(record.plan, dependencies=("task:prepare",)))


@pytest.mark.parametrize("waves", [
    (("task:answer",), ("task:prepare",)),
    (("task:answer", "task:prepare"),),
])
def test_internal_predecessor_must_appear_in_an_earlier_wave(waves) -> None:
    with pytest.raises(SymbolicCandidatePlanningError, match="violates a dependency"):
        PartialOrderSchedule(waves, (("task:prepare", "task:answer"),))


def test_forged_schedule_cannot_erase_the_frozen_candidate_dependency() -> None:
    portfolio = _portfolio(require_proof=False, ordered=True)
    snapshot = portfolio.baseline
    record = snapshot.symbolic_candidate
    forged_plan = replace(record.plan, branch=replace(record.plan.branch, dependencies=()))
    forged_record = replace(
        record,
        plan=forged_plan,
        schedule=PartialOrderSchedule((("task:answer", "task:prepare"),), ()),
    )
    forged_snapshot = replace(snapshot, symbolic_candidate=forged_record)

    with pytest.raises(SymbolicCandidatePlanningError, match="frozen task dependency"):
        replace(portfolio, snapshots=(forged_snapshot,))


def test_model_nomination_includes_the_checked_predecessor_closure() -> None:
    request = _portfolio(require_proof=False, ordered=True).request

    def provider(frozen_request):
        return {"candidates": [{
            "request_id": frozen_request.request_id,
            "task_candidate_ids": ["task:answer"],
        }]}

    portfolio = SymbolicCandidatePlanner(
        bounds=SymbolicCandidateBounds(candidate_count=2, max_model_candidates=1)
    ).plan(
        request.obligation_graph,
        request.frozen_goal,
        request.context,
        model_provider=provider,
    )

    # Closing the nominated task's predecessors makes it the same plan as
    # the baseline, so it is deduplicated after closure rather than admitted
    # as an incomplete alternative.
    assert len(portfolio.snapshots) == 1
    assert portfolio.provider_usage.attempted
    assert portfolio.provider_usage.reason_code == "no_novel_bounded_candidate"
    record = portfolio.baseline.symbolic_candidate
    assert record.task_candidate_ids == ("task:answer", "task:prepare")
    assert record.schedule.waves == (("task:prepare",), ("task:answer",))
    assert record.plan.dependencies == ()
    assert portfolio.request.frozen_goal.policy.satisfied_dependencies == ()
