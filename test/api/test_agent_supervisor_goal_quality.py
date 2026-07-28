from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.goal_quality import (
    GOAL_GRAMMAR_REQUIREMENT_ID,
    AcceptanceCriterion,
    DebtSeverity,
    EvidenceAuthority,
    EvidenceProducer,
    FreshnessPolicy,
    FrozenRootIdentity,
    GoalAdmissionError,
    GoalDebt,
    GoalDebtCode,
    GoalQualityError,
    GoalQualityPolicy,
    GoalQualityReport,
    GoalScope,
    RefinementBudget,
    RepairKind,
    ResourceEnvelope,
    TypedGoal,
    UncertaintyDisposition,
    UncertaintyItem,
    UnsupportedSemantic,
    ValidationRule,
    assert_frozen_root,
    canonical_goal_bytes,
    goal_from_objective_markdown,
    lint_goal,
    lint_objective_markdown,
    project_objective_markdown,
    score_goal,
    validate_goal,
)


def _goal(*, reverse: bool = False) -> TypedGoal:
    criteria = (
        AcceptanceCriterion(
            criterion_id="criterion:test",
            statement="The targeted test command exits with status zero.",
            evidence_producer_ids=("producer:pytest",),
            validation_rule_ids=("validation:pytest",),
            depends_on_criterion_ids=("criterion:contract",),
            completion_signal="exit_code == 0",
        ),
        AcceptanceCriterion(
            criterion_id="criterion:contract",
            statement="The canonical contract round-trips without loss.",
            evidence_producer_ids=("producer:pytest",),
            validation_rule_ids=("validation:pytest",),
            completion_signal="restored == original",
        ),
    )
    producers = (
        EvidenceProducer(
            producer_id="producer:pytest",
            kind="test_runner",
            output_schema="schema:pytest-receipt@1",
            authority=EvidenceAuthority.VALIDATION,
            capability_id="capability:python-pytest",
            independent=True,
        ),
    )
    rules = (
        ValidationRule(
            rule_id="validation:pytest",
            command="python -m pytest test/api/test_agent_supervisor_goal_quality.py -q",
            producer_id="producer:pytest",
            criterion_ids=("criterion:test", "criterion:contract"),
            hermetic=True,
        ),
    )
    return TypedGoal(
        goal_id="ASI-G230",
        root=FrozenRootIdentity("ASI-G200", "objective:root-v1"),
        outcome="Admit only bounded, verifiable typed goals.",
        scope=GoalScope(
            include=(
                "test/api/test_agent_supervisor_goal_quality.py",
                "ipfs_accelerate_py/agent_supervisor/goal_quality.py",
            ),
            exclude=("docs/architecture",),
            dependency_goal_ids=("ASI-G200",),
        ),
        assumptions=("ASI-093 contracts remain available.",),
        non_goals=("Grant mutation or completion authority.",),
        acceptance_criteria=tuple(reversed(criteria)) if reverse else criteria,
        evidence_producers=producers,
        validation_rules=rules,
        freshness=FreshnessPolicy(max_age_seconds=900),
        resources=ResourceEnvelope(
            max_wall_seconds=120,
            max_tokens=16_384,
            max_cost_microunits=1_000_000,
            max_artifacts=16,
            max_parallelism=2,
            max_scope_items=8,
        ),
        uncertainties=(
            UncertaintyItem(
                uncertainty_id="uncertainty:reviewed-none",
                statement="No unresolved uncertainty remains after validation.",
                disposition=UncertaintyDisposition.MITIGATED,
                impact="none",
                resolution="Reopen when a validation dependency changes.",
            ),
        ),
        unsupported_semantics=(
            UnsupportedSemantic(
                semantic_id="semantic:reviewed-none",
                statement="No unsupported semantic is used for admission.",
                fallback="Fail closed on unknown semantics.",
            ),
        ),
        refinement_budget=RefinementBudget(
            max_rounds=2,
            max_children=4,
            max_depth=3,
            max_debt_items=16,
            max_tokens=8_192,
        ),
    )


def test_complete_grammar_is_canonical_immutable_and_round_trips() -> None:
    goal = _goal()
    assert GOAL_GRAMMAR_REQUIREMENT_ID.isdecimal()
    assert lint_goal(goal).accepted
    assert lint_goal(goal).debt == ()
    assert score_goal(goal) == 1_000_000
    assert validate_goal(goal).goal_content_id == goal.content_id

    encoded = goal.to_json()
    assert encoded.encode() == canonical_goal_bytes(goal)
    assert TypedGoal.from_json(encoded) == goal
    assert TypedGoal.from_dict(goal.to_record()) == goal
    assert GoalQualityReport.from_dict(lint_goal(goal).to_record()) == lint_goal(goal)

    with pytest.raises(FrozenInstanceError):
        goal.outcome = "forged"  # type: ignore[misc]


def test_canonical_serialization_and_scoring_ignore_set_like_input_order() -> None:
    first = _goal()
    second = _goal(reverse=True)

    assert second == first
    assert second.content_id == first.content_id
    assert second.to_json() == first.to_json()
    assert lint_goal(second).to_json() == lint_goal(first).to_json()


def test_closed_decoders_reject_unknown_fields_and_forged_projections() -> None:
    goal = _goal()
    unknown = goal.to_record()
    unknown["grant_completion"] = True
    with pytest.raises(GoalQualityError, match="unsupported fields"):
        TypedGoal.from_dict(unknown)

    forged_goal = copy.deepcopy(goal.to_record())
    forged_goal["outcome"] = "A different outcome."
    with pytest.raises(GoalQualityError, match="identity"):
        TypedGoal.from_dict(forged_goal)

    report = lint_goal(goal)
    forged_report = copy.deepcopy(report.to_record())
    forged_report["accepted"] = False
    with pytest.raises(GoalQualityError, match="forged"):
        GoalQualityReport.from_dict(forged_report)

    incomplete = lint_goal(
        TypedGoal(
            goal_id="goal:incomplete",
            root=FrozenRootIdentity("goal:root", "root:v1"),
        )
    )
    compensated = copy.deepcopy(incomplete.to_dict())
    compensated["score_millionths"] = 1_000_000
    with pytest.raises(GoalQualityError, match="deterministic"):
        GoalQualityReport.from_dict(compensated)
    downgraded = copy.deepcopy(incomplete.debt[0].to_dict())
    downgraded["severity"] = DebtSeverity.WARNING.value
    with pytest.raises(GoalQualityError, match="severity"):
        GoalDebt.from_dict(downgraded)


def test_linter_emits_repairable_typed_debt_for_every_required_dimension() -> None:
    empty = TypedGoal(
        goal_id="goal:empty",
        root=FrozenRootIdentity("goal:root", "root:v1"),
    )
    report = lint_goal(empty)
    expected = {
        GoalDebtCode.MISSING_OUTCOME,
        GoalDebtCode.MISSING_SCOPE,
        GoalDebtCode.MISSING_ASSUMPTIONS,
        GoalDebtCode.MISSING_NON_GOALS,
        GoalDebtCode.MISSING_ACCEPTANCE,
        GoalDebtCode.MISSING_EVIDENCE_PRODUCER,
        GoalDebtCode.MISSING_VALIDATION,
        GoalDebtCode.MISSING_FRESHNESS,
        GoalDebtCode.MISSING_RESOURCE_ENVELOPE,
        GoalDebtCode.MISSING_UNCERTAINTY,
        GoalDebtCode.MISSING_UNSUPPORTED_SEMANTICS,
        GoalDebtCode.MISSING_REFINEMENT_BUDGET,
    }
    assert expected.issubset(set(report.debt_codes))
    assert not report.accepted
    assert 0 <= report.score_millionths < 1_000_000
    assert all(
        isinstance(item, GoalDebt)
        and item.path
        and item.message
        and item.repair
        and isinstance(item.repair_kind, RepairKind)
        for item in report.debt
    )
    assert any(item.severity is DebtSeverity.WARNING for item in report.debt)
    with pytest.raises(GoalAdmissionError) as rejected:
        validate_goal(empty)
    assert rejected.value.report == report


def test_adversarial_goal_is_rejected_without_score_compensation() -> None:
    adversarial = replace(
        _goal(),
        scope=GoalScope(
            include=("**", "src", "src/private"),
            exclude=("src/private",),
            dependency_goal_ids=("goal:missing",),
        ),
        acceptance_criteria=(
            AcceptanceCriterion(
                criterion_id="criterion:a",
                statement="Produce a sufficiently good result as needed.",
                evidence_producer_ids=("producer:model",),
                validation_rule_ids=("validation:missing",),
                depends_on_criterion_ids=("criterion:b",),
            ),
            AcceptanceCriterion(
                criterion_id="criterion:b",
                statement="Criterion B holds.",
                evidence_producer_ids=("producer:missing",),
                validation_rule_ids=("validation:pytest",),
                depends_on_criterion_ids=("criterion:a", "criterion:missing"),
            ),
        ),
        evidence_producers=(
            EvidenceProducer(
                producer_id="producer:model",
                kind="llm",
                output_schema="",
                authority=EvidenceAuthority.COMPLETION_GATE,
            ),
        ),
        validation_rules=(
            ValidationRule(
                rule_id="validation:pytest",
                command="",
                producer_id="producer:missing",
                criterion_ids=("criterion:b", "criterion:missing"),
            ),
        ),
    )
    report = lint_goal(
        adversarial,
        policy=GoalQualityPolicy(
            max_scope_items=1,
            max_acceptance_criteria=1,
            max_dependencies=1,
            max_total_breadth=3,
        ),
        known_goal_ids=("ASI-G230",),
    )
    assert {
        GoalDebtCode.CIRCULAR_ACCEPTANCE,
        GoalDebtCode.UNBOUNDED_SCOPE,
        GoalDebtCode.CONFLICTING_SCOPE,
        GoalDebtCode.HIDDEN_AUTHORITY,
        GoalDebtCode.UNVERIFIABLE_EVIDENCE,
        GoalDebtCode.ORPHAN_DEPENDENCY,
        GoalDebtCode.AMBIGUOUS_COMPLETION,
        GoalDebtCode.EXCESSIVE_BREADTH,
    }.issubset(set(report.debt_codes))
    assert not report.accepted
    with pytest.raises(GoalAdmissionError):
        validate_goal(adversarial, known_goal_ids=("ASI-G230",))


def test_declared_completion_gate_is_explicit_but_never_inferred() -> None:
    gate = EvidenceProducer(
        producer_id="producer:gate",
        kind="reviewed_completion_gate",
        output_schema="schema:completion-gate@1",
        authority=EvidenceAuthority.COMPLETION_GATE,
    )
    criterion = AcceptanceCriterion(
        criterion_id="criterion:gate",
        statement="The reviewed completion gate returns passed.",
        evidence_producer_ids=("producer:gate",),
        validation_rule_ids=("validation:gate",),
        completion_signal="gate.passed is true",
    )
    rule = ValidationRule(
        rule_id="validation:gate",
        command="verify-completion-gate",
        producer_id="producer:gate",
        criterion_ids=("criterion:gate",),
    )
    undeclared = replace(
        _goal(),
        acceptance_criteria=(criterion,),
        evidence_producers=(gate,),
        validation_rules=(rule,),
    )
    assert GoalDebtCode.HIDDEN_AUTHORITY in lint_goal(undeclared).debt_codes

    declared = replace(
        undeclared, authorized_completion_producer_ids=("producer:gate",)
    )
    assert GoalDebtCode.HIDDEN_AUTHORITY not in lint_goal(declared).debt_codes


def test_unresolved_uncertainty_and_unsupported_semantics_remain_typed_debt() -> None:
    uncertain = replace(
        _goal(),
        uncertainties=(
            UncertaintyItem(
                uncertainty_id="uncertainty:open",
                statement="Provider capability may change.",
                disposition=UncertaintyDisposition.OPEN,
            ),
        ),
        unsupported_semantics=(
            UnsupportedSemantic(
                semantic_id="semantic:temporal",
                statement="Temporal implication is not supported.",
            ),
        ),
    )
    report = lint_goal(uncertain)
    assert GoalDebtCode.UNCERTAINTY_DEBT in report.debt_codes
    assert GoalDebtCode.UNSUPPORTED_SEMANTICS in report.debt_codes
    assert any(
        item.code is GoalDebtCode.UNCERTAINTY_DEBT
        and item.severity is DebtSeverity.WARNING
        for item in report.debt
    )
    assert not report.accepted

    blocking = replace(
        uncertain,
        unsupported_semantics=_goal().unsupported_semantics,
        uncertainties=(
            UncertaintyItem(
                uncertainty_id="uncertainty:blocking",
                statement="The required capability is unavailable.",
                disposition=UncertaintyDisposition.BLOCKING,
            ),
        ),
    )
    assert any(
        item.code is GoalDebtCode.UNCERTAINTY_DEBT
        and item.severity is DebtSeverity.ERROR
        for item in lint_goal(blocking).debt
    )


def test_frozen_root_identity_survives_child_refinement_and_rejects_substitution() -> None:
    parent = _goal()
    child = replace(parent, goal_id="ASI-G230-child", outcome="Refine one criterion.")
    assert_frozen_root(parent, child)
    assert child.root is parent.root

    substituted = replace(
        child, root=FrozenRootIdentity("ASI-G200", "objective:forged")
    )
    with pytest.raises(GoalQualityError, match="frozen root"):
        assert_frozen_root(parent, substituted)


def test_current_objective_markdown_projects_conservatively_with_stable_root() -> None:
    markdown = """# Objective Heap

## ASI-G200 Supervisor generation two
- Goal: Build the generation-two supervisor
- Outputs: ipfs_accelerate_py/agent_supervisor
- Acceptance: The supervisor is bounded

## ASI-G230 Constraint-based planning and responsive goals
- Parent: ASI-G200
- Depends on: ASI-G210, ASI-G220
- Goal: Compile high-quality typed goals into bounded alternatives
- Outputs: ipfs_accelerate_py/agent_supervisor/goal_quality.py, test/api/test_agent_supervisor_goal_quality.py
- Evidence: goal_quality.GOAL_GRAMMAR_REQUIREMENT_ID
- Validation: python -m pytest test/api/test_agent_supervisor_goal_quality.py -q
- Acceptance: Goals are linted deterministically; unknown authority is rejected
"""
    projected = project_objective_markdown(markdown)
    goal = goal_from_objective_markdown(markdown, "ASI-G230")

    assert len(projected) == 2
    assert goal.root.goal_id == "ASI-G200"
    assert goal.scope.dependency_goal_ids == ("ASI-G210", "ASI-G220")
    assert len(goal.acceptance_criteria) == 2
    assert goal.evidence_producers[0].authority is EvidenceAuthority.DIAGNOSTIC
    assert goal.validation_rules[0].command.startswith("python -m pytest")
    reports = lint_objective_markdown(markdown)
    child_report = next(item for item in reports if item.goal_id == "ASI-G230")
    assert {
        GoalDebtCode.MISSING_ASSUMPTIONS,
        GoalDebtCode.MISSING_RESOURCE_ENVELOPE,
        GoalDebtCode.MISSING_REFINEMENT_BUDGET,
        GoalDebtCode.AMBIGUOUS_COMPLETION,
    }.issubset(set(child_report.debt_codes))

    changed_child = markdown.replace(
        "unknown authority is rejected",
        "unknown authority and orphan dependencies are rejected",
    )
    assert (
        goal_from_objective_markdown(changed_child, "ASI-G230").root
        == goal.root
    )
    changed_root = markdown.replace(
        "Build the generation-two supervisor",
        "Build and operate the generation-two supervisor",
    )
    assert (
        goal_from_objective_markdown(changed_root, "ASI-G230").root
        != goal.root
    )


def test_projection_rejects_parent_cycles_and_missing_selected_goal() -> None:
    cycle = """## goal:a A
- Parent: goal:b
- Goal: A

## goal:b B
- Parent: goal:a
- Goal: B
"""
    with pytest.raises(GoalQualityError, match="parent cycle"):
        project_objective_markdown(cycle)
    with pytest.raises(GoalQualityError, match="does not contain"):
        project_objective_markdown("## goal:a A\n- Goal: A\n", goal_id="goal:b")
