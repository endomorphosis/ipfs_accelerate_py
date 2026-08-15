"""VGO-053: bounded, resumable GUI improvement loop tests.

Acceptance coverage:

* every phase is explicit and resumable by a stable run ID
* acceptance requires a measurable target-metric gain plus hard gates
* accessibility, security, and confirmation regressions reject
* missing evidence rejects or reviews
* opaque / ambiguous proposals escalate to human review
* bounded attempts and whole-app rewrites reject
* rejected or reviewed runs never mutate the canonical branch
* every terminal decision writes a receipt
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.improvement_loop import (
    GUI_IMPROVEMENT_DECISION_INTERFACE,
    GUI_IMPROVEMENT_RUN_INTERFACE,
    PHASE_ORDER,
    VERIFIED_GUI_OPTIMIZER_INTERFACE,
    GuiImprovementLoopError,
    ImprovementDecisionKind,
    ImprovementReasonCode,
    VerifiedGuiOptimizer,
    default_verified_gui_optimizer,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.proposal import (
    DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE,
    GuiPatchProposer,
    ProposalReasonCode,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.run_journal import (
    JournalPhase,
    RunStatus,
    default_run_journal,
)
from ipfs_datasets_py.logic.gui_optimizer.models import GuiImprovementReceipt
from ipfs_datasets_py.logic.gui_optimizer.schema import GUI_IMPROVEMENT_RECEIPT_INTERFACE

IN_SCOPE = "swissknife/web/js/apps/agent-supervisor.js"
SOURCE = (
    "const deprecatedTitle = title;\n"
    "<label>Goal</label>\n"
    "export const GoalForm = () => null;\n"
)
REVISION = "b" * 40
MODULE_PATH = Path(__file__).resolve().parents[2] / (
    "ipfs_accelerate_py/agent_supervisor/gui_optimizer/improvement_loop.py"
)


def _source(**overrides: Any) -> dict[str, Any]:
    payload = {
        "path": IN_SCOPE,
        "content": SOURCE,
        "component_id": "comp:goal-form",
        "editable": True,
    }
    payload.update(overrides)
    return payload


def _pack(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pack_id": "pack:agent-supervisor-goal",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective": "Repair the goal form label.",
        "raw_sources": [_source()],
        "analysis_classification": "exact",
        "verification_status": "unverified",
        "escalation_conditions": [],
        "formal_invariant_failures": [],
        "acceptance_criteria": ["crit:accessible-name"],
    }
    payload.update(overrides)
    return payload


def _transform(**overrides: Any) -> dict[str, Any]:
    payload = {
        "kind": "label",
        "path": IN_SCOPE,
        "find": "<label>Goal</label>",
        "replace": '<label for="goal">Goal</label>',
        "expected_count": 1,
        "interface": DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE,
        "schema_version": (
            "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
            "deterministic-transformation@1"
        ),
    }
    payload.update(overrides)
    return payload


def _evidence(**overrides: Any) -> dict[str, Any]:
    payload = {
        "visual_receipt_ids": ["visual:goal-form"],
        "accessibility_receipt_ids": ["a11y:goal-form"],
        "interaction_receipt_ids": ["interaction:goal-form"],
        "constraint_receipt_ids": ["constraint:goal-form"],
        "invalidation_plan_id": "invalidate:label-form",
        "context_pack_id": "pack:agent-supervisor-goal",
    }
    payload.update(overrides)
    return payload


def _request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "run:agent-supervisor-label",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective_id": "objective:accessible-name",
        "objective": "Ensure the goal form has an accessible name.",
        "source_revision": REVISION,
        "canonical_branch": "main",
        "canonical_revision": REVISION,
        "canonical_porcelain": "",
        "attempt": 1,
        "context_pack": _pack(),
        "transformations": [_transform()],
        "intended_file_paths": [IN_SCOPE],
        "intended_component_ids": ["comp:goal-form"],
        "acceptance_criteria": ["crit:accessible-name"],
        "expected_test_ids": ["test:goal-form-a11y"],
        "expected_screenshot_ids": ["screenshot:keyboard-desktop"],
        "state_effect_ids": ["state:ready"],
        "analysis_classification": "exact",
        "route_kind": "deterministic_transform",
        "baseline": {"violations": ["missing-name"]},
        "baseline_metrics": {
            "accessible_name_coverage": 0.4,
            "critical_accessibility_violations": 2,
        },
        "candidate_metrics": {
            "accessible_name_coverage": 1.0,
            "critical_accessibility_violations": 0,
        },
        "objective_metric_id": "accessible_name_coverage",
        "impact": {"affected_component_ids": ["comp:goal-form"]},
        "invalidation": {
            "plan_id": "invalidate:label-form",
            "fallback_triggered": False,
        },
        "application": {
            "applied": True,
            "promoted": False,
            "disposition": "applied",
            "reason_codes": ["applied"],
        },
        "check_execution": {
            "acceptance_blocked": False,
            "executed_check_ids": ["check:direct-tests"],
            "failed_required_check_ids": [],
            "fallback_applied": False,
        },
        "evidence": _evidence(),
        "hard_gates": {
            "accessibility_regression": False,
            "security_regression": False,
            "confirmation_regression": False,
        },
    }
    payload.update(overrides)
    return payload


def _improve(tmp_path: Path, **overrides: Any):
    optimizer = default_verified_gui_optimizer(tmp_path / "runtime")
    return optimizer, optimizer.improve(_request(**overrides))


def test_optimizer_exports_declared_interfaces(tmp_path: Path) -> None:
    optimizer = default_verified_gui_optimizer(tmp_path / "runtime")
    assert optimizer.interface == VERIFIED_GUI_OPTIMIZER_INTERFACE
    assert VERIFIED_GUI_OPTIMIZER_INTERFACE == "VerifiedGuiOptimizer@1"
    assert GUI_IMPROVEMENT_RUN_INTERFACE == "GuiImprovementRun@1"
    assert GUI_IMPROVEMENT_DECISION_INTERFACE == "GuiImprovementDecision@1"
    assert tuple(phase.value for phase in PHASE_ORDER) == (
        "baseline",
        "select_objective",
        "impact",
        "context_pack",
        "proposal",
        "isolated_worktree",
        "rescan",
        "invalidation",
        "affected_checks",
        "fallback",
        "compare",
        "decision",
        "receipt",
    )


def test_module_does_not_import_model_routing() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    forbidden = {
        "llm_router",
        "model_router",
        "model_routing",
        "semantic_index",
        "semantic_capsule",
        "proof_cache",
    }
    assert imported.isdisjoint(forbidden)
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "llm_router" not in source
    assert "model_routing" not in source


def test_acceptance_requires_metric_gain_and_hard_gates(tmp_path: Path) -> None:
    _optimizer, run = _improve(tmp_path)
    assert run.decision.kind is ImprovementDecisionKind.ACCEPT
    assert run.decision.measurable_improvement is True
    assert run.decision.hard_gates_passed is True
    assert run.promoted is False
    assert run.canonical_mutated is False
    assert run.status is RunStatus.COMPLETED
    assert run.receipt is not None
    decoded = GuiImprovementReceipt.from_dict(dict(run.receipt))
    assert decoded.interface == GUI_IMPROVEMENT_RECEIPT_INTERFACE
    assert decoded.decision.value == "accept"
    assert set(PHASE_ORDER) <= {JournalPhase(item) for item in run.phases}


def test_proposal_escalation_requires_human_review(tmp_path: Path) -> None:
    proposer = GuiPatchProposer()
    optimizer = VerifiedGuiOptimizer(
        journal=default_run_journal(tmp_path / "runtime"),
        proposer=proposer,
    )
    pack = {
        "acceptance_criteria": ["crit:accessible-name"],
        "analysis_classification": "opaque",
        "application_id": "app:agent-supervisor",
        "objective": "Guess a better visual hierarchy.",
        "pack_id": "pack:agent-supervisor-goal",
        "raw_sources": [
            {
                "component_id": "comp:goal-form",
                "content": "export const GoalForm = () => null;\n",
                "editable": True,
                "path": IN_SCOPE,
            }
        ],
        "screen_id": "screen:agent-supervisor",
    }
    run = optimizer.improve(
        _request(
            run_id="run:visual-hierarchy-review",
            context_pack=pack,
            analysis_classification="opaque",
            objective="Guess a better visual hierarchy.",
            transformations=[],
            candidate_metrics={
                "accessible_name_coverage": 0.4,
                "critical_accessibility_violations": 2,
            },
        )
    )
    assert run.decision.kind is ImprovementDecisionKind.HUMAN_REVIEW
    assert run.decision.requires_human_review is True
    assert ImprovementReasonCode.PROPOSAL_ESCALATED.value in run.decision.reason_codes
    assert ProposalReasonCode.OPAQUE_CONTEXT.value in run.decision.reason_codes
    assert run.receipt is not None
    assert run.receipt["decision"] == "human_review"
    assert run.promoted is False
    assert run.canonical_mutated is False
    decoded = GuiImprovementReceipt.from_dict(dict(run.receipt))
    assert decoded.decision.value == "human_review"


def test_accessibility_regression_rejects_despite_metric_gain(tmp_path: Path) -> None:
    _optimizer, run = _improve(
        tmp_path,
        run_id="run:a11y-regression",
        hard_gates={"accessibility_regression": True},
    )
    assert run.decision.kind is ImprovementDecisionKind.REJECT
    assert ImprovementReasonCode.ACCESSIBILITY_REGRESSION.value in run.decision.reason_codes
    assert run.receipt is not None
    assert run.receipt["decision"] == "reject"
    assert run.canonical_mutated is False


def test_no_measurable_improvement_rejects(tmp_path: Path) -> None:
    _optimizer, run = _improve(
        tmp_path,
        run_id="run:no-gain",
        candidate_metrics={
            "accessible_name_coverage": 0.4,
            "critical_accessibility_violations": 2,
        },
    )
    assert run.decision.kind is ImprovementDecisionKind.REJECT
    assert (
        ImprovementReasonCode.NO_MEASURABLE_IMPROVEMENT.value
        in run.decision.reason_codes
    )


def test_missing_evidence_rejects_or_reviews(tmp_path: Path) -> None:
    _optimizer, rejected = _improve(
        tmp_path,
        run_id="run:missing-evidence",
        evidence={
            "visual_receipt_ids": [],
            "accessibility_receipt_ids": [],
            "interaction_receipt_ids": [],
            "constraint_receipt_ids": [],
        },
    )
    assert rejected.decision.kind is ImprovementDecisionKind.REJECT
    assert rejected.decision.missing_evidence is True
    assert ImprovementReasonCode.MISSING_EVIDENCE.value in rejected.decision.reason_codes

    _optimizer, reviewed = _improve(
        tmp_path,
        run_id="run:missing-evidence-opaque",
        analysis_classification="heuristic",
        evidence={
            "visual_receipt_ids": [],
            "accessibility_receipt_ids": ["a11y:goal-form"],
            "interaction_receipt_ids": ["interaction:goal-form"],
            "constraint_receipt_ids": ["constraint:goal-form"],
            "invalidation_plan_id": "invalidate:label-form",
            "context_pack_id": "pack:agent-supervisor-goal",
        },
    )
    assert reviewed.decision.kind is ImprovementDecisionKind.HUMAN_REVIEW
    assert reviewed.decision.missing_evidence is True


def test_required_check_failure_blocks_acceptance(tmp_path: Path) -> None:
    _optimizer, run = _improve(
        tmp_path,
        run_id="run:check-failed",
        check_execution={
            "acceptance_blocked": True,
            "executed_check_ids": ["check:direct-tests"],
            "failed_required_check_ids": ["check:direct-tests"],
            "fallback_applied": True,
        },
    )
    assert run.decision.kind is ImprovementDecisionKind.REJECT
    assert ImprovementReasonCode.REQUIRED_CHECK_FAILED.value in run.decision.reason_codes


def test_attempt_budget_and_rewrite_reject(tmp_path: Path) -> None:
    _optimizer, exhausted = _improve(
        tmp_path,
        run_id="run:attempts-exhausted",
        attempt=4,
        max_attempts=3,
    )
    assert exhausted.decision.kind is ImprovementDecisionKind.REJECT
    assert (
        ImprovementReasonCode.ATTEMPT_BUDGET_EXHAUSTED.value
        in exhausted.decision.reason_codes
    )

    _optimizer, rewrite = _improve(
        tmp_path,
        run_id="run:whole-app",
        objective="Perform a whole-app aesthetic rewrite of the dashboard.",
    )
    assert rewrite.decision.kind is ImprovementDecisionKind.REJECT
    assert ImprovementReasonCode.WHOLE_APP_REWRITE.value in rewrite.decision.reason_codes


def test_too_many_objectives_reject(tmp_path: Path) -> None:
    _optimizer, run = _improve(
        tmp_path,
        run_id="run:too-many-objectives",
        objective_ids=[
            "objective:accessible-name",
            "objective:focus-order",
            "objective:empty-state",
        ],
    )
    assert run.decision.kind is ImprovementDecisionKind.REJECT
    assert ImprovementReasonCode.TOO_MANY_OBJECTIVES.value in run.decision.reason_codes


def test_automatic_canonical_merge_is_forbidden(tmp_path: Path) -> None:
    _optimizer, run = _improve(
        tmp_path,
        run_id="run:no-merge",
        application={
            "applied": True,
            "promoted": True,
            "disposition": "applied",
        },
    )
    assert run.decision.kind is ImprovementDecisionKind.REJECT
    assert (
        ImprovementReasonCode.CANONICAL_MERGE_FORBIDDEN.value
        in run.decision.reason_codes
    )
    assert run.promoted is False
    assert run.canonical_mutated is False
    assert run.canonical_revision == REVISION


def test_phases_are_explicit_and_resumable(tmp_path: Path) -> None:
    optimizer = default_verified_gui_optimizer(tmp_path / "runtime")
    halted = optimizer.improve(
        _request(run_id="run:resume-label", halt_after_phase="context_pack")
    )
    assert halted.decision.kind is ImprovementDecisionKind.PENDING
    assert halted.status is RunStatus.INTERRUPTED
    assert halted.receipt is None
    assert "context_pack" in halted.phases
    assert "decision" not in halted.phases

    resumed = optimizer.improve(_request(run_id="run:resume-label", resume=True))
    assert resumed.run_id == "run:resume-label"
    assert resumed.decision.kind is ImprovementDecisionKind.ACCEPT
    assert resumed.status is RunStatus.COMPLETED
    assert resumed.receipt is not None
    assert "proposal" in resumed.phases
    assert "decision" in resumed.phases
    assert "receipt" in resumed.phases


def test_completed_rerun_returns_same_terminal_receipt(tmp_path: Path) -> None:
    optimizer = default_verified_gui_optimizer(tmp_path / "runtime")
    first = optimizer.improve(_request(run_id="run:idempotent"))
    second = optimizer.improve(_request(run_id="run:idempotent"))
    assert first.terminal_receipt_cid == second.terminal_receipt_cid
    assert first.receipt == second.receipt
    assert second.status is RunStatus.COMPLETED


def test_every_terminal_decision_has_a_receipt(tmp_path: Path) -> None:
    cases = [
        {},
        {
            "run_id": "run:review-receipt",
            "analysis_classification": "opaque",
            "context_pack": _pack(analysis_classification="opaque"),
            "transformations": [],
        },
        {
            "run_id": "run:reject-receipt",
            "candidate_metrics": {
                "accessible_name_coverage": 0.4,
                "critical_accessibility_violations": 2,
            },
        },
    ]
    for overrides in cases:
        _optimizer, run = _improve(tmp_path, **overrides)
        assert run.receipt is not None
        assert run.receipt["interface"] == GUI_IMPROVEMENT_RECEIPT_INTERFACE
        assert run.terminal_receipt_cid
        GuiImprovementReceipt.from_dict(dict(run.receipt))


def test_closed_wire_rejects_unknown_null_and_wrong_container(tmp_path: Path) -> None:
    optimizer = default_verified_gui_optimizer(tmp_path / "runtime")
    with pytest.raises(GuiImprovementLoopError) as unknown:
        optimizer.improve(_request(vendor="x"))
    assert unknown.value.reason_code == ImprovementReasonCode.UNKNOWN_FIELD.value
    with pytest.raises(GuiImprovementLoopError) as null_field:
        payload = _request()
        payload["objective"] = None
        optimizer.improve(payload)
    assert "null" in str(null_field.value)
    with pytest.raises(GuiImprovementLoopError) as tuple_ids:
        optimizer.improve(_request(objective_ids=("objective:accessible-name",)))
    assert (
        tuple_ids.value.reason_code
        == ImprovementReasonCode.INVALID_COLLECTION_TYPE.value
    )


def test_pixel_change_alone_does_not_accept(tmp_path: Path) -> None:
    _optimizer, run = _improve(
        tmp_path,
        run_id="run:pixel-only",
        objective_metric_id="accessible_name_coverage",
        baseline_metrics={
            "accessible_name_coverage": 0.4,
            "pixel_diff_percent": 12.0,
        },
        candidate_metrics={
            "accessible_name_coverage": 0.4,
            "pixel_diff_percent": 1.0,
        },
    )
    assert run.decision.kind is ImprovementDecisionKind.REJECT
    assert ImprovementReasonCode.PIXEL_CHANGE_ONLY.value in run.decision.reason_codes
