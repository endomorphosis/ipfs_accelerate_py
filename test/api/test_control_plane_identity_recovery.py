"""Concurrent identity-recovery planning uses DuckDB/Quack, not DuckLake."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.validation.control_plane_identity_recovery import (
    CAS_TASK_STATUS_SQL,
    CLAIM_IDENTITY_RECOVERY_SQL,
    OWNER_IDENTITY_RECOVERY_SQL,
    IdentityRecoveryAction,
    identity_control_projection,
    overlay_completed_count,
    plan_control_plane_identity_recovery,
    restore_overlay_cas_parameters,
    snapshot_overlay_alias_status,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    plan_automatic_implementation_rescue,
)


def test_plan_waits_when_another_supervisor_holds_the_claim() -> None:
    plan = plan_control_plane_identity_recovery(
        source_head="a" * 40,
        materialization_source_head="b" * 40,
        overlay_statuses={"EAAEF-000": "completed"},
        recovery_held=True,
    )
    assert plan.action is IdentityRecoveryAction.WAIT_FOR_HOLDER
    assert plan.ducklake_current_authority is False
    assert plan.to_record()["process_started"] is False


def test_plan_preserves_overlay_when_source_drifted() -> None:
    plan = plan_control_plane_identity_recovery(
        source_head="a" * 40,
        materialization_source_head="b" * 40,
        overlay_statuses={"EAAEF-000": "completed", "EAAEF-001": "todo"},
    )
    assert plan.action is IdentityRecoveryAction.PRESERVE_AND_REBIND
    assert plan.overlay_completed == 1


def test_plan_diagnoses_when_identities_already_match() -> None:
    plan = plan_control_plane_identity_recovery(
        source_head="a" * 40,
        materialization_source_head="a" * 40,
        overlay_statuses={"EAAEF-000": "completed"},
    )
    assert plan.action is IdentityRecoveryAction.DIAGNOSE_ADMISSION


def test_plan_refuses_ducklake_as_current_authority() -> None:
    plan = plan_control_plane_identity_recovery(
        source_head="a" * 40,
        materialization_source_head="b" * 40,
        ducklake_current_authority=True,
    )
    assert plan.action is IdentityRecoveryAction.NONE
    assert plan.reason == "ducklake_is_not_current_authority"


def test_identity_projection_ignores_live_status() -> None:
    live = {
        "tasks": [
            {"task_cid": "sha256:" + "a" * 64, "status": "completed", "revision": 4}
        ],
        "projection_root": "sha256:" + "b" * 64,
        "intent_snapshot": {"task_count": 1, "projection_cid": "baguqeera" + "1" * 50},
        "exact_relations": {"tasks": {"status": "completed"}},
    }
    receipt = {
        "tasks": [{"task_cid": "sha256:" + "a" * 64, "status": "todo", "revision": 1}],
        "projection_root": "sha256:" + "c" * 64,
        "intent_snapshot": {"task_count": 1, "projection_cid": "baguqeera" + "2" * 50},
        "exact_relations": {"tasks": {"status": "todo"}},
    }
    assert identity_control_projection(live) == identity_control_projection(receipt)


def test_restore_parameters_use_closed_cas_template() -> None:
    parameters = restore_overlay_cas_parameters(
        live_rows=[
            {
                "task_alias": "EAAEF-000",
                "task_cid": "cid-0",
                "status": "todo",
                "revision": 1,
            },
            {
                "task_alias": "EAAEF-001",
                "task_cid": "cid-1",
                "status": "completed",
                "revision": 2,
            },
        ],
        overlay_statuses={"EAAEF-000": "completed", "EAAEF-001": "completed"},
        updated_at="2026-08-22T00:00:00Z",
    )
    assert parameters == [
        ("completed", 2, "2026-08-22T00:00:00Z", "cid-0", 1),
    ]


def test_owner_sql_is_closed_and_includes_claim() -> None:
    assert CLAIM_IDENTITY_RECOVERY_SQL in OWNER_IDENTITY_RECOVERY_SQL
    assert "identity_recovery" in CLAIM_IDENTITY_RECOVERY_SQL
    assert "RETURNING action_id" in CLAIM_IDENTITY_RECOVERY_SQL
    assert CAS_TASK_STATUS_SQL not in OWNER_IDENTITY_RECOVERY_SQL


def test_auto_rescue_routes_source_drift_to_host_bootstrap_recovery() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "error": "bootstrap receipt population_cid differs from current source",
        }
    )
    assert plan.action is AutoRescueAction.HOST_BOOTSTRAP_RECOVERY
    assert plan.max_provider_rescue_passes == 0


def test_snapshot_reads_alias_status_projection(tmp_path) -> None:
    path = tmp_path / "task-status-projection.json"
    path.write_text(
        '{"schema":"x","statuses":{"EAAEF-000":"completed","EAAEF-001":"todo"}}\n',
        encoding="utf-8",
    )
    statuses = snapshot_overlay_alias_status(path)
    assert overlay_completed_count(statuses) == 1
    assert statuses["EAAEF-000"] == "completed"
