from __future__ import annotations

import json
from itertools import repeat

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AutonomyContractError,
    DecisionQuestion,
    DecisionQuestionType,
    QuestionDisposition,
    RiskClass,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.decision_graph import (
    DecisionGraphChangeKind,
    DecisionGraphController,
    DecisionGraphError,
    compile_decision_graph,
    graph_is_complete,
    invalidate_evidence,
    semantic_question_id,
)


def _question(
    *,
    criterion: str,
    question_type: DecisionQuestionType = DecisionQuestionType.WHICH_TEST_IS_REQUIRED,
    alternatives: tuple[str, ...] = ("selected", "not_required"),
    required: tuple[str, ...] = (),
    known: tuple[str, ...] = (),
    contradictory: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    uncertainty: int = 5_000,
    disposition: QuestionDisposition = QuestionDisposition.UNRESOLVED,
    answer: str = "",
    mandatory: bool = True,
    risk: RiskClass = RiskClass.R1_READ_ONLY,
    deadline: int = 1_000,
) -> DecisionQuestion:
    return DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=(criterion,),
        question_type=question_type,
        current_alternatives=alternatives,
        required_evidence_ids=required,
        known_evidence_ids=known,
        contradictory_evidence_ids=contradictory,
        residual_uncertainty_bp=uncertainty,
        decision_deadline_ms=deadline,
        risk_if_incorrect=risk,
        risk_if_left_unresolved=risk,
        possible_resolution_action_ids=("action:static",),
        dependency_question_ids=dependencies,
        terminal_decision_rule="select only from current alternatives",
        mandatory=mandatory,
        disposition=disposition,
        terminal_answer=answer,
    )


def _compile(*questions: DecisionQuestion):
    return compile_decision_graph(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:one",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=questions,
    )


def _by_semantic(graph):
    return {semantic_question_id(question): question for question in graph.questions}


def test_compile_semantically_deduplicates_and_is_order_stable() -> None:
    first = _question(
        criterion="AC-1",
        alternatives=("selected", "not_required"),
        known=("evidence:first",),
        risk=RiskClass.R1_READ_ONLY,
        deadline=2_000,
    )
    duplicate = _question(
        criterion="AC-2",
        alternatives=("not_required", "selected"),
        required=("evidence:required",),
        risk=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
        deadline=500,
    )

    forward = _compile(first, duplicate)
    reverse = _compile(duplicate, first)

    assert forward.graph_id == reverse.graph_id
    assert len(forward.questions) == 1
    merged = forward.questions[0]
    assert merged.acceptance_criterion_ids == ("AC-1", "AC-2")
    assert merged.required_evidence_ids == ("evidence:required",)
    assert merged.known_evidence_ids == ("evidence:first",)
    assert merged.risk_if_incorrect is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION
    assert merged.decision_deadline_ms == 500


def test_resolution_requires_named_evidence_and_allows_terminal_completion() -> None:
    question = _question(criterion="AC-1", required=("evidence:test",))
    controller = DecisionGraphController(_compile(question))

    with pytest.raises(DecisionGraphError, match="attributable evidence"):
        controller.resolve(
            semantic_question_id(question),
            terminal_answer="selected",
            evidence_ids=(),
        )
    with pytest.raises(DecisionGraphError, match="bounded alternatives"):
        controller.resolve(
            semantic_question_id(question),
            terminal_answer="invented",
            evidence_ids=("evidence:test",),
        )

    receipt = controller.resolve(
        semantic_question_id(question),
        terminal_answer="selected",
        evidence_ids=("evidence:test",),
    )

    assert receipt.change_kind is DecisionGraphChangeKind.QUESTION_RESOLVED
    assert receipt.before_graph_id != receipt.after_graph_id
    assert controller.complete
    assert graph_is_complete(controller.graph)
    assert controller.unresolved_questions == ()


def test_snapshot_is_canonical_content_bound_and_restart_safe() -> None:
    controller = DecisionGraphController(_compile(_question(criterion="AC-1")))
    snapshot = controller.snapshot_json()
    recovered = DecisionGraphController.from_snapshot(snapshot)

    assert recovered.graph.graph_id == controller.graph.graph_id
    assert recovered.snapshot_json() == snapshot
    immutable_snapshot = recovered.snapshot()
    with pytest.raises(TypeError):
        immutable_snapshot["questions"][0]["objective_id"] = "forged"

    forged = json.loads(snapshot)
    forged["tree_id"] = "tree:forged"
    with pytest.raises(AutonomyContractError, match="identity does not match"):
        DecisionGraphController.from_snapshot(json.dumps(forged))


def test_invalidation_changes_only_evidence_consumers_and_dependency_suffix() -> None:
    root = _question(
        criterion="AC-root",
        required=("evidence:root",),
        known=("evidence:root",),
        uncertainty=0,
        disposition=QuestionDisposition.RESOLVED,
        answer="selected",
    )
    dependent = _question(
        criterion="AC-dependent",
        question_type=DecisionQuestionType.WHICH_PROOF_OBLIGATION_APPLIES,
        required=("evidence:dependent",),
        known=("evidence:dependent",),
        dependencies=(root.question_id,),
        uncertainty=0,
        disposition=QuestionDisposition.RESOLVED,
        answer="selected",
    )
    independent = _question(
        criterion="AC-independent",
        question_type=DecisionQuestionType.WHETHER_CACHE_IS_REUSABLE,
        required=("evidence:independent",),
        known=("evidence:independent",),
        uncertainty=0,
        disposition=QuestionDisposition.RESOLVED,
        answer="selected",
    )
    graph = _compile(root, dependent, independent)
    semantics = {
        "root": semantic_question_id(root),
        "dependent": semantic_question_id(dependent),
        "independent": semantic_question_id(independent),
    }

    transition = invalidate_evidence(graph, ("evidence:root",), new_tree_id="tree:two")
    states = _by_semantic(transition.graph)

    assert transition.graph.tree_id == "tree:two"
    assert transition.graph.graph_revision == graph.graph_revision + 1
    assert states[semantics["root"]].disposition is QuestionDisposition.INVALIDATED
    assert states[semantics["dependent"]].disposition is QuestionDisposition.INVALIDATED
    assert states[semantics["independent"]].disposition is QuestionDisposition.RESOLVED
    assert set(transition.receipt.invalidated_question_ids) == {
        semantics["root"],
        semantics["dependent"],
    }
    assert transition.receipt.preserved_question_ids == (semantics["independent"],)


def test_contradiction_invalidates_only_the_dependent_cone_and_is_recorded() -> None:
    root = _question(
        criterion="AC-root",
        known=("evidence:old",),
        uncertainty=0,
        disposition=QuestionDisposition.RESOLVED,
        answer="selected",
    )
    dependent = _question(
        criterion="AC-dependent",
        question_type=DecisionQuestionType.WHETHER_REPLAN_IS_REQUIRED,
        dependencies=(root.question_id,),
    )
    independent = _question(
        criterion="AC-independent",
        question_type=DecisionQuestionType.WHETHER_FAILURE_IS_FLAKY,
    )
    controller = DecisionGraphController(_compile(root, dependent, independent))

    receipt = controller.record_evidence(
        semantic_question_id(root),
        evidence_ids=("evidence:counterexample",),
        contradictory=True,
    )
    states = _by_semantic(controller.graph)

    assert receipt.change_kind is DecisionGraphChangeKind.CONTRADICTION_RECORDED
    assert (
        "evidence:counterexample"
        in states[semantic_question_id(root)].contradictory_evidence_ids
    )
    assert (
        states[semantic_question_id(dependent)].disposition
        is QuestionDisposition.INVALIDATED
    )
    assert (
        states[semantic_question_id(independent)].disposition
        is QuestionDisposition.UNRESOLVED
    )


def test_replayed_evidence_is_a_no_op_without_revision_or_write_churn() -> None:
    question = _question(criterion="AC-1", known=("evidence:one",))
    controller = DecisionGraphController(_compile(question))
    before = controller.graph

    receipt = controller.record_evidence(
        semantic_question_id(question), evidence_ids=("evidence:one",)
    )

    assert receipt.change_kind is DecisionGraphChangeKind.NO_OP
    assert receipt.before_graph_id == receipt.after_graph_id
    assert receipt.affected_question_ids == ()
    assert controller.graph is before


def test_optional_dependency_of_mandatory_question_must_also_be_terminal() -> None:
    optional = _question(
        criterion="AC-helper",
        mandatory=False,
    )
    mandatory = _question(
        criterion="AC-main",
        question_type=DecisionQuestionType.WHETHER_CONTEXT_IS_SUFFICIENT,
        dependencies=(optional.question_id,),
        known=("evidence:main",),
        uncertainty=0,
        disposition=QuestionDisposition.RESOLVED,
        answer="selected",
    )
    controller = DecisionGraphController(_compile(optional, mandatory))

    assert not controller.complete
    controller.resolve(
        semantic_question_id(optional),
        terminal_answer="selected",
        evidence_ids=("evidence:helper",),
    )
    assert controller.complete


def test_compile_facade_accepts_input_version_alias_and_empty_graph_is_not_complete() -> (
    None
):
    question = _question(criterion="AC-1")
    controller = DecisionGraphController.compile(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:one",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=(item for item in (question,)),
    )

    controller.resolve(
        question.question_id,
        terminal_answer="selected",
        evidence_ids=("evidence:one",),
    )
    assert controller.complete
    assert not graph_is_complete(_compile())


def test_compile_rejects_unknown_and_deduplication_self_dependencies() -> None:
    unknown = _question(criterion="AC-1", dependencies=("missing:question",))
    with pytest.raises(DecisionGraphError, match="unknown input question"):
        _compile(unknown)

    first = _question(criterion="AC-1")
    duplicate_with_dependency = _question(
        criterion="AC-2", dependencies=(first.question_id,)
    )
    with pytest.raises(DecisionGraphError, match="depend on itself"):
        _compile(first, duplicate_with_dependency)


def test_compile_rejects_unbounded_question_iterable() -> None:
    question = _question(criterion="AC-1")
    with pytest.raises(DecisionGraphError, match="too many questions"):
        DecisionGraphController.compile(
            repository_id="repo:ipfs-accelerate",
            tree_id="tree:one",
            objective_id="APMC-G000",
            objective_revision="revision:one",
            questions=repeat(question),
        )
