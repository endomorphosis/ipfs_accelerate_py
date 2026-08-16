"""Focused WorldSnapshotBuilder@1 admission checks."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_builder import (
    WorldSnapshotAdmissionError,
    build_world_snapshot,
    current_authority_fixture,
    observe_ducklake_projection,
)


def test_current_authorities_admit_schedulable_snapshot() -> None:
    first = build_world_snapshot(current_authority_fixture())
    second = build_world_snapshot(current_authority_fixture())
    assert first["schedulable"] is True
    assert first["snapshot"]["snapshot_cid"] == second["snapshot"]["snapshot_cid"]
    assert first["ducklake_projection"]["authoritative"] is False
    assert set(first["component_status"]) >= {
        "repository",
        "accepted_plan_root",
        "task_population",
        "datasets_semantic_state_root",
        "policy_root",
    }


def test_stale_unavailable_quarantine_fail_closed() -> None:
    for name, status in (
        ("accepted_plan_root", "stale"),
        ("datasets_semantic_state_root", "unavailable"),
        ("policy_root", "quarantined"),
    ):
        authorities = current_authority_fixture()
        authorities[name] = {**authorities[name], "status": status}
        if status == "unavailable":
            authorities[name]["cid"] = ""
        result = build_world_snapshot(authorities)
        assert result["schedulable"] is False
        assert any(name in reason for reason in result["unschedulable_reasons"])


def test_required_authority_disagreement_is_unschedulable() -> None:
    authorities = current_authority_fixture()
    authorities["task_population"] = {
        **authorities["task_population"],
        "plan_cid": "sha256:" + ("ab" * 32),
    }
    result = build_world_snapshot(authorities)
    assert result["schedulable"] is False
    assert result["component_status"]["task_population"] == "inconsistent"


def test_optional_projection_cannot_grant_or_revoke_authority() -> None:
    authorities = current_authority_fixture()
    healthy = build_world_snapshot(
        authorities,
        ducklake_receipt={
            "receipt_cid": "sha256:" + ("cd" * 32),
            "status": "current",
            "valid": True,
        },
    )
    absent = build_world_snapshot(authorities, ducklake_receipt=None)
    outage = build_world_snapshot(
        authorities,
        ducklake_receipt={"valid": False, "tampered": True, "receipt_cid": ""},
    )
    assert healthy["schedulable"] is True
    assert absent["schedulable"] is True
    assert outage["schedulable"] is True
    assert healthy["ducklake_projection"]["authoritative"] is False
    assert absent["ducklake_projection"]["authoritative"] is False
    observed = observe_ducklake_projection(None)
    assert observed["authoritative"] is False
    assert observed["status"] == "unavailable"

    stale_plan = copy.deepcopy(authorities)
    stale_plan["accepted_plan_root"]["status"] = "stale"
    still_blocked = build_world_snapshot(
        stale_plan,
        ducklake_receipt={
            "receipt_cid": "sha256:" + ("ef" * 32),
            "status": "current",
            "valid": True,
        },
    )
    assert still_blocked["schedulable"] is False


def test_implicit_lookup_rejected() -> None:
    authorities = current_authority_fixture()
    authorities["claims"] = {**authorities["claims"], "implicit_lookup": True}
    with pytest.raises(WorldSnapshotAdmissionError, match="implicit"):
        build_world_snapshot(authorities)
