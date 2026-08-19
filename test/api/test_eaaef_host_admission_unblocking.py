"""Contracts for epic S host-gated admission-evidence tasks.

These tests are host-controlled and must pass without live supervisor launch.
Missing signed artifacts are represented as typed receipts, not xfail/skip.
"""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
    BUNDLE_SCHEMA,
    RECEIPT_DIR,
    RECEIPT_FILES,
    RECEIPT_SCHEMA,
    classify_blocker,
    closing_task_ids,
    collect_host_admission_receipts,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    plan_automatic_implementation_rescue,
)

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
BOARD = CAMPAIGN / "task_board.json"
HOST_EVIDENCE_IDS = [f"EAAEF-{number}" for number in range(180, 192)]


def _board() -> dict:
    return json.loads(BOARD.read_text(encoding="utf-8"))


def _tasks() -> dict[str, dict]:
    return {task["stable_task_id"]: task for task in _board()["tasks"]}


def _receipt(name: str) -> dict:
    path = RECEIPT_DIR / name
    assert path.is_file(), f"missing host-admission receipt {name}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload.get("process_started") is not True
    assert payload.get("supervisor_process_started") is not True
    assert payload.get("self_signed") is not True
    assert payload.get("receipt_cid", "").startswith("sha256:")
    return payload


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
    owned = tasks["EAAEF-180"]["owned_files"]
    assert "ipfs_accelerate_py/agent_supervisor/validation/eaaef_host_admission.py" in owned
    assert "scripts/collect_eaaef_host_admission_receipts.py" in owned


def test_inventory_classifies_ingest_failures_as_host_bootstrap_recovery() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "error": "ControlPlaneIdentityError: output path is not a safe identifier",
        }
    )
    assert plan.action is AutoRescueAction.HOST_BOOTSTRAP_RECOVERY
    assert plan.max_provider_rescue_passes == 0
    assert classify_blocker("output path is not a safe identifier") == "auto_recoverable"
    assert classify_blocker("eaaef_scoped_provider_authorization_missing") == (
        "host_gated_external_authority"
    )
    assert "EAAEF-184" in closing_task_ids("eaaef_scoped_provider_authorization_missing")


def test_inventory_receipt_classifies_launch_plan_blockers() -> None:
    payload = _receipt("blocker_inventory.json")
    assert payload["schema"] == RECEIPT_SCHEMA
    assert payload["task_id"] == "EAAEF-180"
    items = payload["evidence"]["items"]
    assert items
    classes = {item["class"] for item in items}
    assert "host_gated_external_authority" in classes
    assert all(item["closing_task_ids"] for item in items)


def test_principals_receipt_contract() -> None:
    payload = _receipt("runtime_principals.json")
    principals = payload["evidence"]["principals"]
    dids = [item["did"] for item in principals]
    roles = [item["role"] for item in principals]
    assert roles == ["worker", "provider", "quack_owner"]
    assert len(set(dids)) == 3
    assert all(did.startswith("did:key:z") for did in dids)
    assert payload["evidence"]["secret_material_exported"] is False
    assert payload["evidence"]["admitted_authority"] is False
    dumped = json.dumps(payload)
    assert "BEGIN PRIVATE" not in dumped
    assert "PKCS8" not in dumped
    assert "private_key_pkcs8" not in dumped


def test_duckdb_quack_receipt_contract() -> None:
    payload = _receipt("duckdb_quack_155.json")
    evidence = payload["evidence"]
    assert evidence["required_duckdb"] == "1.5.5"
    assert evidence["required_quack"] == "1.5.5+core"
    assert evidence["network_install_attempted"] is False
    if evidence["observed_duckdb"] != "1.5.5":
        assert evidence["silent_substitution_refused"] is True
        assert payload["decision"] == "typed_missing"
    elif payload["decision"] == "admitted":
        assert evidence["under_approved_import_root"] is True
        assert evidence["quack_probe"]["passes_health_check"] is True
        assert evidence["quack_probe"]["extension"]["installed_from"] == "core"


def test_engine_mode_receipt_contract() -> None:
    payload = _receipt("engine_mode.json")
    evidence = payload["evidence"]
    assert evidence["docker_socket_mounted"] is False
    assert evidence["supervisor_started"] is False
    if evidence.get("rootless") is True and payload["decision"] == "admitted":
        assert evidence.get("host_docker_socket_used") is not True
        assert str(evidence.get("docker_host") or "") != "unix:///var/run/docker.sock"
    else:
        fallback = evidence["fallback_package"]
        assert fallback["signed"] is False
        assert fallback["docker_socket_mount"] == "prohibited"
        assert fallback["independent_security_review_required"] is True
        assert payload["decision"] == "typed_missing"


def test_provider_authorization_receipt_contract() -> None:
    payload = _receipt("provider_authorization.json")
    evidence = payload["evidence"]
    assert evidence["self_signed_rejected"] is True
    assert evidence["supervisor_signed"] is False
    assert evidence["configured_board_launch"] is False
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert evidence["reviewer_provider"] == "local_operator"
        assert evidence["route_id"] == (
            "agent-supervisor-eaaef-v1-grok46-terra56-high-auth-or-hard-quota-v1"
        )
        assert str(evidence.get("artifact_path") or "").endswith(".json")
        assert str(evidence.get("authorization_id") or "").startswith("sha256:")
    else:
        assert payload["decision"] == "typed_missing"
        assert evidence["independent_signature_present"] is False


def test_worker_image_receipt_contract() -> None:
    payload = _receipt("worker_image.json")
    assert payload["decision"] == "typed_missing"
    assert payload["evidence"]["live_dispatch_claimed"] is False


def test_container_profile_receipt_contract() -> None:
    payload = _receipt("container_profile.json")
    assert payload["decision"] == "typed_missing"


def test_worker_network_receipt_contract() -> None:
    payload = _receipt("worker_network.json")
    assert payload["decision"] == "typed_missing"
    assert payload["evidence"]["required_lanes"] == 5


def test_command_fabric_receipt_contract() -> None:
    payload = _receipt("command_fabric_endpoints.json")
    assert payload["decision"] == "typed_missing"
    assert payload["evidence"]["implemented_unqualified_fail_closed_admitted"] is False


def test_native_lane_receipt_contract() -> None:
    payload = _receipt("native_lane_dispatcher.json")
    assert payload["decision"] == "typed_missing"


def test_plan_r2_receipt_contract() -> None:
    payload = _receipt("plan_r2_remote_owner.json")
    assert payload["decision"] == "typed_missing"
    assert payload["evidence"]["r1_evidence_promotes_r2"] is False


def test_admission_bundle_receipt_contract() -> None:
    payload = _receipt("admission_bundle.json")
    assert payload["schema"] == BUNDLE_SCHEMA
    assert payload["decision"] == "no_go"
    assert payload["evidence"]["prospective_supervisor_signature_rejected"] is True
    assert payload["evidence"]["independent_operator_signature"] == ""
    assert payload["evidence"]["independent_security_reviewer_signature"] == ""
    child_cids = payload["evidence"]["child_receipt_cids"]
    for task_id, filename in RECEIPT_FILES.items():
        if task_id == "EAAEF-191":
            continue
        child = _receipt(filename)
        assert child_cids[task_id] == child["receipt_cid"]
    assert _tasks()["EAAEF-191"]["completion_mode"] == "manual"
    assert _tasks()["EAAEF-183"]["completion_mode"] == "auto"
    assert _tasks()["EAAEF-184"]["completion_mode"] == "auto"


def test_collector_refuses_live_launch_allowed_plan() -> None:
    try:
        collect_host_admission_receipts(
            launch_plan={"allowed": True, "process_started": False, "blockers": []}
        )
    except RuntimeError as exc:
        assert "live-launch-allowed" in str(exc)
    else:
        raise AssertionError("collector accepted a live-launch-allowed plan")
