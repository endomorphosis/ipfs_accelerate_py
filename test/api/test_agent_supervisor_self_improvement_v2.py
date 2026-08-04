"""Regression coverage for generation-2 self-improvement surfaces used by PDR-080.

Covers the live-epoch residual bridge and successor policy helpers added so the
daemon lifecycle can feed Planner/Doctor epoch residuals into generation-2
admission without routing through test-only ``run_self_improvement_epoch``.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement_v2 import (
    MAX_V2_SUCCESSOR_GOALS,
    MAX_V2_SUCCESSOR_TASKS,
    PLANNER_DOCTOR_LIVE_EPOCH_MAX_GOALS,
    PLANNER_DOCTOR_LIVE_EPOCH_MAX_TASKS,
    PLANNER_DOCTOR_LIVE_EPOCH_REQUIREMENT_ID,
    V2ResidualKind,
    V2ResidualSignal,
    V2SelfEvaluationError,
    V2SuccessorGenerationPolicy,
    generate_v2_successor_goals,
    planner_doctor_epoch_successor_policy,
    v2_residuals_from_planner_doctor_epoch,
)


def _residual(index: int = 0) -> V2ResidualSignal:
    slug = f"epoch-gap-{index}"
    return V2ResidualSignal(
        residual_id=f"residual:{slug}",
        kind=V2ResidualKind.BENCHMARK_RESIDUAL,
        title=f"Close {slug}",
        detail=f"Live paired epoch reported residual {slug} under shadow mode.",
        acceptance_criteria=(f"Oracle closes {slug}",),
        evidence_ids=(f"evidence:{slug}",),
        predicted_files=(
            "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py",
            f"test/api/test_{slug.replace('-', '_')}.py",
        ),
        predicted_symbols=(f"repair_{slug.replace('-', '_')}",),
        validation_commands=(
            "python -m pytest test/api/test_agent_supervisor_planner_doctor_epoch.py -q",
        ),
    )


def test_live_epoch_constants_align_with_hard_maxima() -> None:
    assert PLANNER_DOCTOR_LIVE_EPOCH_MAX_GOALS == MAX_V2_SUCCESSOR_GOALS == 8
    assert PLANNER_DOCTOR_LIVE_EPOCH_MAX_TASKS == MAX_V2_SUCCESSOR_TASKS == 24
    assert PLANNER_DOCTOR_LIVE_EPOCH_REQUIREMENT_ID.startswith("pdr-080")


def test_v2_residuals_from_planner_doctor_epoch_binds_source() -> None:
    residual = _residual()
    assert residual.source_receipt_id == ""
    normalized = v2_residuals_from_planner_doctor_epoch(
        (residual,),
        epoch_id="sha256:" + ("ab" * 32),
        stop_reason="no_admitted_improvement",
    )
    assert len(normalized) == 1
    assert normalized[0].source_receipt_id.startswith("epoch:")
    assert "no_admitted_improvement" in normalized[0].source_receipt_id
    assert normalized[0].residual_id == residual.residual_id


def test_v2_residuals_preserve_existing_source_receipt() -> None:
    residual = V2ResidualSignal(
        residual_id="residual:kept",
        kind=V2ResidualKind.REGRESSION,
        title="Keep source",
        detail="Existing source receipt must not be overwritten by the bridge.",
        acceptance_criteria=("Keep source lineage",),
        evidence_ids=("evidence:kept",),
        predicted_files=("ipfs_accelerate_py/agent_supervisor/x.py",),
        predicted_symbols=("keep_source",),
        validation_commands=("python -m pytest -q",),
        source_receipt_id="receipt:original",
    )
    normalized = v2_residuals_from_planner_doctor_epoch(
        (residual,),
        epoch_id="epoch-1",
        stop_reason="completed",
    )
    assert normalized[0].source_receipt_id == "receipt:original"


def test_v2_residuals_accept_mapping_payloads() -> None:
    payload = _residual().to_dict()
    payload.pop("source_receipt_id", None)
    normalized = v2_residuals_from_planner_doctor_epoch(
        (payload,),
        epoch_id="epoch-map",
        stop_reason="safety_regression",
    )
    assert normalized[0].kind is V2ResidualKind.BENCHMARK_RESIDUAL
    assert normalized[0].source_receipt_id.startswith("epoch:epoch-map:")


def test_v2_residuals_reject_empty_epoch_id() -> None:
    with pytest.raises(V2SelfEvaluationError):
        v2_residuals_from_planner_doctor_epoch(
            (_residual(),),
            epoch_id="",
            stop_reason="completed",
        )


def test_planner_doctor_epoch_successor_policy_is_finite() -> None:
    policy = planner_doctor_epoch_successor_policy()
    assert isinstance(policy, V2SuccessorGenerationPolicy)
    assert policy.max_goals == MAX_V2_SUCCESSOR_GOALS
    assert policy.max_tasks == MAX_V2_SUCCESSOR_TASKS
    assert policy.max_goals <= 8
    assert policy.max_tasks <= 24


def test_planner_doctor_epoch_successor_policy_cannot_enlarge() -> None:
    with pytest.raises(V2SelfEvaluationError):
        planner_doctor_epoch_successor_policy(max_goals=9)
    with pytest.raises(V2SelfEvaluationError):
        planner_doctor_epoch_successor_policy(max_tasks=25)


def test_epoch_residuals_generate_bounded_successors() -> None:
    residuals = v2_residuals_from_planner_doctor_epoch(
        tuple(_residual(i) for i in range(3)),
        epoch_id="epoch-successors",
        stop_reason="no_admitted_improvement",
    )
    policy = planner_doctor_epoch_successor_policy(max_goals=3, max_tasks=6)
    admission = generate_v2_successor_goals(
        residuals,
        policy=policy,
        current_open_work=0,
    )
    assert admission.generated_goal_count <= 3
    assert admission.generated_task_count <= 6
    # At least one residual should admit or the rejection ledger is non-empty.
    assert (
        admission.generated_goal_count >= 1
        or len(admission.rejected) >= 1
    )


def test_exports_include_live_epoch_bridge() -> None:
    from ipfs_accelerate_py.agent_supervisor.self_improvement import (
        self_improvement_v2 as module,
    )

    assert "v2_residuals_from_planner_doctor_epoch" in module.__all__
    assert "planner_doctor_epoch_successor_policy" in module.__all__
    assert "PLANNER_DOCTOR_LIVE_EPOCH_REQUIREMENT_ID" in module.__all__
