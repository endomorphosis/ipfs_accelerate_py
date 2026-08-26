"""EAAEF-074: compile goals to work plans and admit exactly one candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.planning.external_goal_contract import (
    ExternalGoalContract,
    GoalContractError,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_work_plan import (
    ExternalWorkPlan,
    WorkPlanError,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_admission import (
    SCORE_AXES,
    PlanAdmissionError,
    admit_plan,
    rank_plans,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

WRITE_SCOPE = "src/"
PLAN_ADMISSION_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "ipfs_accelerate_py/agent_supervisor/planning/plan_admission.py"
)
RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "planning.json"
)
ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-qualification-artifact@1"
)
PRODUCER_ARGV = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/integration/test_external_agent_plan_compilation.py",
)
RECEIPT_FIELDS = {
    "artifact_cid",
    "cases_exercised",
    "content_addressed_plan_id",
    "current_proofs_verified",
    "evidence_mode",
    "independent_review_invoked",
    "live_plan_execution_invoked",
    "live_runtime_invoked",
    "offline_selected_candidate_id",
    "producer_argv",
    "producer_source_cid",
    "production_qualification_claimed",
    "qualification_scope",
    "qualification_status",
    "schema",
    "task_completion_claimed",
    "task_id",
}


def _producer_source_cid() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_receipt(payload: dict[str, object]) -> None:
    assert set(payload) == RECEIPT_FIELDS
    assert payload["schema"] == ARTIFACT_SCHEMA
    assert payload["task_id"] == "EAAEF-074"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "offline_plan_compilation_contract_only"
    assert payload["qualification_status"] == "not_live_qualified"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_runtime_invoked"] is False
    assert payload["live_plan_execution_invoked"] is False
    assert payload["independent_review_invoked"] is False
    assert payload["current_proofs_verified"] is False
    assert payload["producer_argv"] == list(PRODUCER_ARGV)
    assert payload["producer_source_cid"] == _producer_source_cid()
    unsealed = dict(payload)
    artifact_cid = unsealed.pop("artifact_cid")
    assert artifact_cid == content_identity(unsealed)


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    sealed = {
        **payload,
        "producer_argv": list(PRODUCER_ARGV),
        "producer_source_cid": _producer_source_cid(),
    }
    sealed["artifact_cid"] = content_identity(sealed)
    _validate_receipt(sealed)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sealed


def _objective(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "objective_id": "EAAEF-G080",
        "desired_outcomes": ("cover goal", "admit one plan"),
        "prohibited_outcomes": ("self_approve", "hidden_chain_of_thought"),
        "write_scope": (WRITE_SCOPE,),
        "authority_ceiling": "preview_only",
        "verification_requirements": ("focused pytest",),
        "proof_requirements": ("content identity",),
        "review_requirements": ("independent supervisor",),
        "completion_evidence": ("test receipt", "plan identity"),
        "timeout_seconds": 7200,
        "cpu_millicores": 4000,
        "ram_mib": 8192,
    }
    payload.update(overrides)
    return payload


def _goal(**overrides: object) -> ExternalGoalContract:
    return ExternalGoalContract.compile(_objective(**overrides))


def _narrow_plan() -> ExternalWorkPlan:
    return ExternalWorkPlan.decompose(
        _goal(),
        (
            {
                "task_id": "task-cover",
                "covers": ("cover goal", "admit one plan"),
                "write_scope": ("src/plan.py",),
                "depends_on": (),
                "timeout_seconds": 60,
                "cpu_millicores": 1000,
                "ram_mib": 1024,
            },
        ),
    )


def _wide_plan() -> ExternalWorkPlan:
    return ExternalWorkPlan.decompose(
        _goal(),
        (
            {
                "task_id": "task-a",
                "covers": ("cover goal",),
                "write_scope": ("src/a.py",),
                "depends_on": (),
                "timeout_seconds": 600,
                "cpu_millicores": 2000,
                "ram_mib": 2048,
            },
            {
                "task_id": "task-b",
                "covers": ("admit one plan",),
                "write_scope": ("src/b.py",),
                "depends_on": ("task-a",),
                "timeout_seconds": 600,
                "cpu_millicores": 2000,
                "ram_mib": 2048,
            },
        ),
    )


def test_compile_goals_and_admit_exactly_one_plan() -> None:
    goal = _goal()
    assert set(goal.desired_outcomes) == {"cover goal", "admit one plan"}
    narrow = _narrow_plan()
    wide = _wide_plan()
    assert {outcome for task in narrow.tasks for outcome in task.covers} >= set(
        goal.desired_outcomes
    )
    admission = admit_plan(
        (
            {"candidate_id": "narrow", "plan": narrow, "model_cost": 1},
            {"candidate_id": "wide", "plan": wide, "model_cost": 50},
        )
    )
    assert admission.admitted_id == "narrow"
    assert admission.admitted.content_id == narrow.content_id
    assert len(admission.ranked) == 2
    again = admit_plan(
        (
            {"candidate_id": "wide", "plan": wide, "model_cost": 50},
            {"candidate_id": "narrow", "plan": narrow, "model_cost": 1},
        )
    )
    assert again.admitted_id == "narrow"
    assert again.admitted.content_id == admission.admitted.content_id
    ranked = rank_plans(
        (
            {"candidate_id": "narrow", "plan": narrow, "model_cost": 1},
            {"candidate_id": "wide", "plan": wide, "model_cost": 50},
        )
    )
    assert ranked[0].candidate_id == "narrow"


def test_empty_candidates_and_omitted_coverage_fail_closed() -> None:
    with pytest.raises(PlanAdmissionError, match="candidates"):
        admit_plan(())
    with pytest.raises(PlanAdmissionError, match="candidates"):
        rank_plans([])
    with pytest.raises(WorkPlanError, match="coverage"):
        ExternalWorkPlan.decompose(
            _goal(),
            (
                {
                    "task_id": "partial",
                    "covers": ("cover goal",),
                    "write_scope": ("src/plan.py",),
                    "timeout_seconds": 60,
                    "cpu_millicores": 1000,
                    "ram_mib": 1024,
                },
            ),
        )
    with pytest.raises(GoalContractError, match="evidence"):
        ExternalGoalContract.compile(_objective(completion_evidence=()))


def test_ranking_uses_integer_axes_without_random_or_wall_clock() -> None:
    source = PLAN_ADMISSION_SOURCE.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "no random" in lowered
    assert "wall-clock" in lowered
    assert "wall-clock value participates" in lowered
    for axis in SCORE_AXES:
        assert axis in {
            "critical_path",
            "safe_width",
            "model_proof_cost",
            "resources",
            "merge_risk",
            "uncertainty",
            "prior_success",
            "cache_locality",
        }
    assert "random" not in SCORE_AXES
    assert "wall_clock" not in SCORE_AXES
    assert "now_ms" not in SCORE_AXES
    assert "timestamp" not in SCORE_AXES


def test_write_offline_plan_compilation_receipt() -> None:
    narrow = _narrow_plan()
    wide = _wide_plan()
    decision = admit_plan(
        (
            {"candidate_id": "wide", "plan": wide, "model_cost": 50},
            {"candidate_id": "narrow", "plan": narrow, "model_cost": 1},
        )
    )
    assert decision.admitted_id == "narrow"
    assert decision.admitted.content_id == narrow.content_id
    with pytest.raises(PlanAdmissionError, match="candidates"):
        admit_plan(())
    with pytest.raises(WorkPlanError, match="coverage"):
        ExternalWorkPlan.decompose(
            _goal(),
            (
                {
                    "task_id": "partial",
                    "covers": ("cover goal",),
                    "write_scope": ("src/plan.py",),
                    "timeout_seconds": 60,
                    "cpu_millicores": 1000,
                    "ram_mib": 1024,
                },
            ),
        )

    receipt = _write_receipt(
        {
            "schema": ARTIFACT_SCHEMA,
            "task_id": "EAAEF-074",
            "evidence_mode": "contract_fail_closed",
            "qualification_scope": "offline_plan_compilation_contract_only",
            "qualification_status": "not_live_qualified",
            "task_completion_claimed": False,
            "production_qualification_claimed": False,
            "live_runtime_invoked": False,
            "live_plan_execution_invoked": False,
            "independent_review_invoked": False,
            "current_proofs_verified": False,
            "offline_selected_candidate_id": "narrow",
            "content_addressed_plan_id": narrow.content_id,
            "cases_exercised": [
                "deterministic_candidate_ranking",
                "empty_candidate_refusal",
                "omitted_goal_coverage_refusal",
            ],
        }
    )
    _validate_receipt(receipt)
