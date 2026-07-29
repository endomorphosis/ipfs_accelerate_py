"""Datasets-contract evidence for lossless typed objective admission (DSCON-G055)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.goal_quality import (
    ARTIFACT_RECEIPT_OUTPUT_SCHEMA,
    DebtSeverity,
    EvidenceAuthority,
    GoalDebtCode,
    ObjectiveTypedGoals,
    PYTEST_RECEIPT_OUTPUT_SCHEMA,
    goal_from_objective_markdown,
    lint_objective_markdown,
    lint_objective_typed_goals,
    migrate_objective_markdown,
    project_objective_markdown,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    objective_heap_content_id,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    ObjectiveLaunchQualitySummary,
    build_objective_launch_quality_summary,
    build_objective_typed_goals,
    load_objective_goal_quality_report,
    load_objective_typed_goals,
    write_objective_goal_quality_report,
    write_objective_typed_goals,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OBJECTIVE_PATH = REPO_ROOT / "docs" / "planning" / "DATASETS_CONTRACT_ANALYSIS_OBJECTIVES.md"
GOAL_QUALITY_PATH = (
    REPO_ROOT
    / "data"
    / "datasets_contract_analysis"
    / "agent_supervisor"
    / "goal-quality.json"
)


def _objective_text() -> str:
    return OBJECTIVE_PATH.read_text(encoding="utf-8")


def test_legacy_structural_projection_reports_typed_quality_debt() -> None:
    """Default projection stays on the documented structural legacy path."""

    text = _objective_text()
    goals = project_objective_markdown(text)
    reports = lint_objective_markdown(text)

    assert goals
    assert len(reports) == len(goals)
    assert all(not report.accepted for report in reports)
    debt_codes = {debt.code for report in reports for debt in report.debt}
    assert GoalDebtCode.AMBIGUOUS_COMPLETION in debt_codes
    assert GoalDebtCode.UNVERIFIABLE_EVIDENCE in debt_codes

    sample = goal_from_objective_markdown(text, "DSCON-G055")
    assert sample.evidence_producers
    assert all(not producer.output_schema for producer in sample.evidence_producers)
    assert all(not criterion.completion_signal for criterion in sample.acceptance_criteria)
    assert sample.evidence_producers[0].authority is EvidenceAuthority.DIAGNOSTIC


def test_structured_markdown_fields_are_preserved_not_dropped() -> None:
    markdown = """
## goal:root Root
- Goal: Root outcome
- Outputs: src/root.py
- Acceptance: Root holds

## goal:child Child goal
- Parent: goal:root
- Depends on: goal:root
- Goal: Preserve producer schema and completion signal
- Outputs: src/child.py, test/test_child.py
- Evidence Producers JSON: [{"producer_id":"producer:pytest","kind":"test_runner","output_schema":"schema:pytest-receipt@1","authority":"validation","independent":true}]
- Completion Signals JSON: ["pytest exit code is 0 and receipt binds criterion:1"]
- Assumptions: Contracts remain reviewed.
- Non Goals: Do not invent completion authority.
- Freshness Horizon Seconds: 900
- Resource Envelope JSON: {"max_wall_seconds":120,"max_tokens":4096,"max_cost_microunits":1000,"max_artifacts":4,"max_parallelism":1,"max_scope_items":4}
- Refinement Budget JSON: {"max_rounds":2,"max_children":2,"max_depth":2,"max_debt_items":8,"max_tokens":2048}
- Uncertainties JSON: [{"uncertainty_id":"uncertainty:reviewed-none","statement":"None remain.","disposition":"mitigated","impact":"none","resolution":"Reopen on schema change."}]
- Unsupported Semantics JSON: [{"semantic_id":"semantic:reviewed-none","statement":"None used.","fallback":"Fail closed."}]
- Validation: python -m pytest test/test_child.py -q
- Acceptance: The child contract round-trips
"""
    goal = goal_from_objective_markdown(markdown, "goal:child")
    assert goal.evidence_producers[0].kind == "test_runner"
    assert goal.evidence_producers[0].output_schema == "schema:pytest-receipt@1"
    assert goal.evidence_producers[0].authority is EvidenceAuthority.VALIDATION
    assert goal.acceptance_criteria[0].completion_signal.startswith("pytest exit code")
    assert goal.freshness.max_age_seconds == 900
    assert goal.resources.bounded
    assert goal.refinement_budget.bounded
    assert goal.uncertainties[0].disposition.value == "mitigated"
    assert goal.unsupported_semantics[0].fallback
    child = next(
        item for item in lint_objective_markdown(markdown) if item.goal_id == "goal:child"
    )
    assert child.accepted
    assert not any(item.severity is DebtSeverity.ERROR for item in child.debt)


def test_current_heap_migrates_to_lossless_typed_sidecar_without_error_debt() -> None:
    text = _objective_text()
    heap_id = objective_heap_content_id(text)
    document = migrate_objective_markdown(text)

    assert document.objective_heap_id == heap_id
    assert document.goals
    restored = ObjectiveTypedGoals.from_dict(document.to_dict())
    assert restored == document
    assert restored.content_id == document.content_id

    reports = lint_objective_typed_goals(document)
    assert len(reports) == len(document.goals)
    assert all(report.accepted for report in reports)
    assert not any(
        debt.severity is DebtSeverity.ERROR
        for report in reports
        for debt in report.debt
    )

    for goal in document.goals:
        assert goal.assumptions
        assert goal.non_goals
        assert goal.freshness.max_age_seconds > 0
        assert goal.resources.bounded
        assert goal.refinement_budget.bounded
        assert goal.uncertainties
        assert goal.unsupported_semantics
        assert all(item.fallback for item in goal.unsupported_semantics)
        assert goal.acceptance_criteria
        for criterion in goal.acceptance_criteria:
            assert criterion.completion_signal
            assert criterion.evidence_producer_ids
            assert criterion.validation_rule_ids
        for producer in goal.evidence_producers:
            assert producer.kind
            assert producer.output_schema
            # Migration never invents completion-gate authority.
            assert producer.authority is not EvidenceAuthority.COMPLETION_GATE


def test_typed_overlay_and_tracker_sidecar_bind_exact_heap_cid(
    tmp_path: Path,
) -> None:
    text = _objective_text()
    heap_id = objective_heap_content_id(text)
    objective_path = tmp_path / "objectives.md"
    sidecar_path = tmp_path / "typed-goals.json"
    objective_path.write_text(text, encoding="utf-8")

    written = write_objective_typed_goals(objective_path, sidecar_path)
    loaded = load_objective_typed_goals(sidecar_path, objective_path=objective_path)
    assert written == loaded
    assert loaded.objective_heap_id == heap_id

    overlaid = project_objective_markdown(text, typed_overlay=loaded)
    assert {goal.goal_id for goal in overlaid} == {goal.goal_id for goal in loaded.goals}
    assert all(
        report.accepted
        for report in lint_objective_markdown(text, typed_overlay=loaded)
    )

    objective_path.write_text(text + "\n<!-- changed -->\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        load_objective_typed_goals(sidecar_path, objective_path=objective_path)


def test_launcher_summary_defaults_to_structural_legacy_and_reports_debt() -> None:
    text = _objective_text()
    summary = build_objective_launch_quality_summary(text)

    assert summary.admission_path == "structural_legacy"
    assert summary.typed_admission_claimed is False
    assert summary.legacy_structure_accepted is True
    assert summary.objective_heap_id == objective_heap_content_id(text)
    assert summary.strict_typed_rejected > 0
    assert summary.strict_typed_debt
    assert summary.strict_typed_error_debt
    assert GoalDebtCode.AMBIGUOUS_COMPLETION.value in summary.strict_typed_debt

    restored = ObjectiveLaunchQualitySummary.from_dict(summary.to_dict())
    assert restored == summary

    claimed = build_objective_launch_quality_summary(
        text, claim_typed_admission=True
    )
    assert claimed.admission_path == "typed_sidecar"
    assert claimed.typed_admission_claimed is True
    assert claimed.strict_typed_rejected == 0
    assert claimed.strict_typed_error_debt == {}
    assert claimed.typed_sidecar_content_id == build_objective_typed_goals(text).content_id


def test_checked_in_goal_quality_report_is_bound_to_current_heap() -> None:
    text = _objective_text()
    heap_id = objective_heap_content_id(text)
    assert GOAL_QUALITY_PATH.is_file(), f"missing evidence artifact: {GOAL_QUALITY_PATH}"

    report = load_objective_goal_quality_report(
        GOAL_QUALITY_PATH, objective_path=OBJECTIVE_PATH
    )
    assert report.objective_heap_id == heap_id
    assert report.quality_records
    assert report.debt_records

    # Rewrite is deterministic and heap-bound.
    rebuilt = write_objective_goal_quality_report(OBJECTIVE_PATH, GOAL_QUALITY_PATH)
    assert rebuilt.objective_heap_id == heap_id
    assert rebuilt.content_id == report.content_id
    payload = json.loads(GOAL_QUALITY_PATH.read_text(encoding="utf-8"))
    assert payload["objective_heap_id"] == heap_id
    assert payload["content_id"] == report.content_id


def test_reviewed_producer_classification_covers_evidence_kinds() -> None:
    text = _objective_text()
    goal = goal_from_objective_markdown(text, "DSCON-G055", lossless=True)
    producer_ids = {item.producer_id for item in goal.evidence_producers}
    assert "data/datasets_contract_analysis/agent_supervisor/goal-quality.json" in producer_ids
    assert (
        "ipfs_accelerate_py/test/api/test_agent_supervisor_datasets_contract_goal_quality.py"
        in producer_ids
    )
    by_id = {item.producer_id: item for item in goal.evidence_producers}
    artifact = by_id[
        "data/datasets_contract_analysis/agent_supervisor/goal-quality.json"
    ]
    test_producer = by_id[
        "ipfs_accelerate_py/test/api/test_agent_supervisor_datasets_contract_goal_quality.py"
    ]
    assert artifact.kind == "artifact_receipt"
    assert artifact.output_schema == ARTIFACT_RECEIPT_OUTPUT_SCHEMA
    assert test_producer.kind == "test_runner"
    assert test_producer.output_schema == PYTEST_RECEIPT_OUTPUT_SCHEMA
    assert test_producer.authority is EvidenceAuthority.VALIDATION
