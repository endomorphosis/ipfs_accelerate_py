"""Contracts for epic S host-gated admission-evidence tasks.

These tests are host-controlled and must pass without live supervisor launch.
Missing signed artifacts are represented as typed absences, not xfail/skip.
"""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    plan_automatic_implementation_rescue,
)

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
RECEIPT_DIR = CAMPAIGN / "receipts" / "host_admission"
BOARD = CAMPAIGN / "task_board.json"
HOST_EVIDENCE_IDS = [f"EAAEF-{number}" for number in range(180, 192)]


def _board() -> dict:
    return json.loads(BOARD.read_text(encoding="utf-8"))


def _tasks() -> dict[str, dict]:
    return {task["stable_task_id"]: task for task in _board()["tasks"]}


def _receipt_contract(name: str) -> None:
    path = RECEIPT_DIR / name
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload.get("process_started") is not True
    assert payload.get("supervisor_process_started") is not True


def test_inventory_host_evidence_is_bootstrap_ready_frontier() -> None:
    board = _board()
    tasks = _tasks()
    for task_id in HOST_EVIDENCE_IDS:
        task = tasks[task_id]
        assert task["initial_population"] is True
        assert task["is_schedulable"] is True
        assert task["epic"] == "S"
        assert task["resource_request"]["supervisor_processes"] == 0
        assert task["resource_request"]["provider_concurrency"] == 0
    ready = [
        task_id
        for task_id, task in tasks.items()
        if task["status"] == "todo"
        and task["is_schedulable"]
        and not task["dependencies"]
    ]
    assert ready == ["EAAEF-180", "EAAEF-181", "EAAEF-182", "EAAEF-183"]
    assert "EAAEF-191" in tasks["EAAEF-000"]["dependencies"]
    assert board["goals"]
    assert any(goal["goal_id"] == "EAAEF-G190" for goal in board["goals"])


def test_inventory_classifies_ingest_failures_as_host_bootstrap_recovery() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "error": "ControlPlaneIdentityError: output path is not a safe identifier",
        }
    )
    assert plan.action is AutoRescueAction.HOST_BOOTSTRAP_RECOVERY
    assert plan.max_provider_rescue_passes == 0


def test_principals_receipt_contract() -> None:
    _receipt_contract("runtime_principals.json")


def test_duckdb_quack_receipt_contract() -> None:
    _receipt_contract("duckdb_quack_155.json")


def test_engine_mode_receipt_contract() -> None:
    _receipt_contract("engine_mode.json")


def test_provider_authorization_receipt_contract() -> None:
    _receipt_contract("provider_authorization.json")


def test_worker_image_receipt_contract() -> None:
    _receipt_contract("worker_image.json")


def test_container_profile_receipt_contract() -> None:
    _receipt_contract("container_profile.json")


def test_worker_network_receipt_contract() -> None:
    _receipt_contract("worker_network.json")


def test_command_fabric_receipt_contract() -> None:
    _receipt_contract("command_fabric_endpoints.json")


def test_native_lane_receipt_contract() -> None:
    _receipt_contract("native_lane_dispatcher.json")


def test_plan_r2_receipt_contract() -> None:
    _receipt_contract("plan_r2_remote_owner.json")


def test_admission_bundle_receipt_contract() -> None:
    _receipt_contract("admission_bundle.json")
    assert _tasks()["EAAEF-191"]["completion_mode"] == "manual"
    assert _tasks()["EAAEF-183"]["completion_mode"] == "manual"
    assert _tasks()["EAAEF-184"]["completion_mode"] == "manual"
