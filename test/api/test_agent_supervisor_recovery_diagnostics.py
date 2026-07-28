from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    IncidentKind,
    prompt_workflow_cid,
)
from ipfs_accelerate_py.agent_supervisor.rescue.recovery_diagnostics import (
    RecoveryDiagnosticError,
    RecoveryDiagnosticLimits,
    RecoveryEvidenceKind,
    diagnose_supervisor_incident,
)


def _identity(tmp_path: Path) -> dict[str, str]:
    return {
        "repository_root": str(tmp_path.resolve()),
        "state_root": str((tmp_path / "state").resolve()),
        "repository_root_cid": prompt_workflow_cid({"repository": "current"}),
        "policy_root": prompt_workflow_cid({"policy": "current"}),
        "run_cid": prompt_workflow_cid({"run": "current"}),
    }


def test_stale_projection_is_not_misdiagnosed_as_a_live_fault(
    tmp_path: Path,
) -> None:
    diagnosis = diagnose_supervisor_incident(
        **_identity(tmp_path),
        status={"state": "running", "projection_stale": True},
        health={"healthy": True},
        process={"lane_id": "lane-1", "state": "running"},
        heartbeat={"state": "current"},
        observed_at_ms=100,
    )

    assert diagnosis.kind is IncidentKind.STALE_PROJECTION
    assert diagnosis.stale_projection
    assert not diagnosis.live_fault
    assert diagnosis.reason_codes == ("projection_stale_live_state_healthy",)
    assert diagnosis.incident.health["live_fault"] is False


@pytest.mark.parametrize(
    ("slot", "evidence", "expected"),
    [
        ("process", {"failed": True}, IncidentKind.LANE_FAILURE),
        ("heartbeat", {"stale": True}, IncidentKind.STALE_HEARTBEAT),
        ("event", {"cursor_stale": True}, IncidentKind.STALE_HEARTBEAT),
        ("lease", {"expired": True}, IncidentKind.STALE_LEASE),
        ("lock", {"orphaned": True}, IncidentKind.ORPHANED_LOCK),
        ("attempt", {"consumed": True}, IncidentKind.CONSUMED_ATTEMPT),
        ("task_source", {"digest_mismatch": True}, IncidentKind.CORRUPT_TASK_SOURCE),
        ("worktree", {"dirty": True}, IncidentKind.DIRTY_WORKTREE),
        ("merge", {"status": "failed"}, IncidentKind.MERGE_FAILURE),
        ("provider", {"unavailable": True}, IncidentKind.PROVIDER_UNAVAILABLE),
        ("validation", {"status": "failed"}, IncidentKind.VALIDATION_FAILURE),
        ("disk", {"full": True}, IncidentKind.RESOURCE_EXHAUSTION),
    ],
)
def test_classifies_typed_live_fault_evidence(
    tmp_path: Path,
    slot: str,
    evidence: dict[str, object],
    expected: IncidentKind,
) -> None:
    diagnosis = diagnose_supervisor_incident(
        **_identity(tmp_path),
        **{slot: {"lane_id": "lane-1", **evidence}},
    )
    assert diagnosis.kind is expected
    assert diagnosis.live_fault


def test_incident_cid_is_semantic_and_prior_actions_are_identity_bearing(
    tmp_path: Path,
) -> None:
    common = {
        **_identity(tmp_path),
        "heartbeat": {
            "lane_id": "lane-1",
            "stale": True,
            "age_ms": 50_000,
            "updated_at_ms": 10,
        },
    }
    first = diagnose_supervisor_incident(
        **common,
        observed_at_ms=100,
        prior_actions=(
            {
                "operation": "retry",
                "outcome": "failed",
                "finished_at_ms": 90,
            },
        ),
    )
    later = diagnose_supervisor_incident(
        **{
            **common,
            "heartbeat": {
                **common["heartbeat"],
                "age_ms": 80_000,
                "updated_at_ms": 20,
            },
        },
        observed_at_ms=200,
        prior_actions=(
            {
                "operation": "retry",
                "outcome": "failed",
                "finished_at_ms": 190,
            },
        ),
    )
    changed_action = diagnose_supervisor_incident(
        **common,
        observed_at_ms=300,
        prior_actions=(
            {"operation": "restart_lane", "outcome": "failed"},
        ),
    )

    assert first.incident_cid == later.incident_cid
    assert first.incident.failure_fingerprint == later.incident.failure_fingerprint
    assert changed_action.incident_cid != first.incident_cid


def test_diagnostics_are_bounded_and_redact_secret_bearing_fields(
    tmp_path: Path,
) -> None:
    sensitive_key = "_".join(("api", "key"))
    diagnosis = diagnose_supervisor_incident(
        **_identity(tmp_path),
        provider={
            "provider_id": "provider-1",
            "unavailable": True,
            sensitive_key: "must-not-enter-evidence",
        },
    )
    provider = diagnosis.evidence_for(RecoveryEvidenceKind.PROVIDER)[0]
    assert sensitive_key not in provider.value
    assert "must-not-enter-evidence" not in str(provider.to_dict())

    with pytest.raises(RecoveryDiagnosticError, match="evidence exceeds bound"):
        diagnose_supervisor_incident(
            **_identity(tmp_path),
            process=[{"failed": True}, {"failed": True}],
            limits=RecoveryDiagnosticLimits(max_evidence_items=1),
        )


def test_split_brain_and_corruption_take_precedence_over_retryable_faults(
    tmp_path: Path,
) -> None:
    split = diagnose_supervisor_incident(
        **_identity(tmp_path),
        process={"lane_id": "lane-1", "live_owner_count": 2},
        task={"task_id": "task-1", "failed": True},
    )
    corrupt = diagnose_supervisor_incident(
        **_identity(tmp_path),
        task_source={"task_id": "task-1", "corrupt": True},
        validation={"failed": True},
    )
    assert split.kind is IncidentKind.SPLIT_BRAIN
    assert corrupt.kind is IncidentKind.CORRUPT_TASK_SOURCE


def test_common_watchdog_shapes_distinguish_live_fault_from_stale_status(
    tmp_path: Path,
) -> None:
    dead_process = diagnose_supervisor_incident(
        **_identity(tmp_path),
        process={"lane_id": "lane-1", "alive": False},
    )
    unhealthy = diagnose_supervisor_incident(
        **_identity(tmp_path),
        health={"lane_id": "lane-1", "healthy": False},
    )
    stale_status = diagnose_supervisor_incident(
        **_identity(tmp_path),
        status={"lane_id": "lane-1", "state": "failed"},
        process={"lane_id": "lane-1", "alive": True},
        health={"lane_id": "lane-1", "healthy": True},
        heartbeat={"lane_id": "lane-1", "state": "current"},
    )

    assert dead_process.kind is IncidentKind.LANE_FAILURE
    assert unhealthy.kind is IncidentKind.LANE_FAILURE
    assert stale_status.kind is IncidentKind.STALE_PROJECTION
    assert stale_status.stale_projection
    assert not stale_status.live_fault


def test_cid_bearing_evidence_is_deeply_immutable_and_aggregate_bounded(
    tmp_path: Path,
) -> None:
    diagnosis = diagnose_supervisor_incident(
        **_identity(tmp_path),
        process={
            "lane_id": "lane-1",
            "failed": True,
            "nested": {"alive": False},
        },
    )
    nested = diagnosis.evidence_for(RecoveryEvidenceKind.PROCESS)[0].value[
        "nested"
    ]
    with pytest.raises(TypeError):
        nested["alive"] = True

    with pytest.raises(
        RecoveryDiagnosticError,
        match="aggregate diagnostic evidence",
    ):
        diagnose_supervisor_incident(
            **_identity(tmp_path),
            process=[
                {"lane_id": "lane-1", "failed": True, "detail": "x" * 180},
                {"lane_id": "lane-2", "failed": True, "detail": "y" * 180},
            ],
            limits=RecoveryDiagnosticLimits(max_serialized_bytes=400),
        )
