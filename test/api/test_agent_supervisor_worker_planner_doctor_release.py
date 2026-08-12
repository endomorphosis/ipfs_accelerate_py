"""WPD-070: terminal Worker Planner–Doctor release gate tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.worker_planner_doctor_release import (
    WORKER_PLANNER_DOCTOR_RELEASE_INTERFACE,
    ReleaseVerdict,
    SafetyFloors,
    evaluate_release,
)


def test_interface_identity() -> None:
    assert WORKER_PLANNER_DOCTOR_RELEASE_INTERFACE == "WorkerPlannerDoctorRelease@1"


def test_release_blocks_synthetic_promotion_when_tree_healthy() -> None:
    receipt = evaluate_release(synthetic_only=True)
    assert receipt.safety_floors.all_zero is True
    assert receipt.modules_missing == ()
    assert receipt.benchmark_provider_call_reduction > 0
    assert receipt.promotion_allowed is False
    assert receipt.verdict is ReleaseVerdict.BLOCKED_SYNTHETIC
    assert "synthetic_only_blocks_promotion" in receipt.reason_codes


def test_release_fails_on_nonzero_safety_floor() -> None:
    receipt = evaluate_release(
        safety_floors=SafetyFloors(unauthorized_provider_calls=1),
        synthetic_only=True,
    )
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert receipt.promotion_allowed is False
    assert "safety_floor_nonzero" in receipt.reason_codes


def test_release_fails_on_missing_required_module() -> None:
    receipt = evaluate_release(
        synthetic_only=True,
        require_modules=(
            "ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_kernel",
            "ipfs_accelerate_py.agent_supervisor.does_not_exist_module",
        ),
    )
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert "required_module_missing" in receipt.reason_codes
    assert any("does_not_exist" in m for m in receipt.modules_missing)


def test_non_synthetic_promotes_when_all_gates_pass() -> None:
    receipt = evaluate_release(synthetic_only=False)
    assert receipt.safety_floors.all_zero is True
    assert receipt.modules_missing == ()
    assert receipt.benchmark_provider_call_reduction > 0
    assert receipt.promotion_allowed is True
    assert receipt.verdict is ReleaseVerdict.PASS


def test_receipt_is_body_free() -> None:
    receipt = evaluate_release(synthetic_only=True)
    payload = receipt.to_dict()
    for forbidden in ("source", "prompt", "transcript", "api_key"):
        assert forbidden not in payload
