"""Focused SupervisorWorldSnapshot@1 contract checks."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_contracts import (
    COMPONENT_OWNERS,
    INTERFACE,
    SCHEMA,
    WorldSnapshotContractError,
    compute_snapshot_cid,
    example_current_snapshot,
    field_ownership_table,
    mutable_snapshot,
    parse_world_snapshot,
)


def test_schema_roundtrip_and_deterministic_identity() -> None:
    first = example_current_snapshot()
    second = parse_world_snapshot(mutable_snapshot(first))
    third = example_current_snapshot()
    assert first["schema"] == SCHEMA
    assert first["interface"] == INTERFACE
    assert first["snapshot_cid"] == second["snapshot_cid"] == third["snapshot_cid"]
    assert first["snapshot_cid"] == compute_snapshot_cid(first)
    assert first["snapshot_cid"].startswith("sha256:")


def test_unknown_and_forbidden_fields_rejected() -> None:
    payload = mutable_snapshot(example_current_snapshot())
    payload["prompt"] = "do not embed"
    with pytest.raises(WorldSnapshotContractError, match="forbidden"):
        parse_world_snapshot(payload)
    payload = mutable_snapshot(example_current_snapshot())
    payload["unexpected"] = "nope"
    with pytest.raises(WorldSnapshotContractError, match="unknown fields"):
        parse_world_snapshot(payload)


def test_malformed_cid_status_epoch_and_floats_rejected() -> None:
    payload = mutable_snapshot(example_current_snapshot())
    components = copy.deepcopy(payload["components"])
    components["policy_root"] = {
        "status": "current",
        "cid": "not-a-cid",
        "owner": COMPONENT_OWNERS["policy_root"],
    }
    payload["components"] = components
    payload.pop("snapshot_cid", None)
    with pytest.raises(WorldSnapshotContractError, match="malformed CID"):
        parse_world_snapshot(payload)

    payload = mutable_snapshot(example_current_snapshot())
    components = copy.deepcopy(payload["components"])
    components["policy_root"] = {
        "status": "maybe",
        "cid": components["policy_root"]["cid"],
        "owner": COMPONENT_OWNERS["policy_root"],
    }
    payload["components"] = components
    payload.pop("snapshot_cid", None)
    with pytest.raises(WorldSnapshotContractError, match="malformed status"):
        parse_world_snapshot(payload)

    payload = mutable_snapshot(example_current_snapshot())
    payload["coordination_epoch"] = -1
    payload.pop("snapshot_cid", None)
    with pytest.raises(WorldSnapshotContractError, match="coordination_epoch"):
        parse_world_snapshot(payload)

    payload = mutable_snapshot(example_current_snapshot())
    payload["fencing_epoch"] = 1.5
    payload.pop("snapshot_cid", None)
    with pytest.raises(WorldSnapshotContractError, match="float"):
        parse_world_snapshot(payload)


def test_repository_mismatch_and_operational_datasets_data_rejected() -> None:
    payload = mutable_snapshot(example_current_snapshot())
    components = copy.deepcopy(payload["components"])
    components["repository_tree"] = {
        **components["repository_tree"],
        "owner": "ipfs_datasets_py",
    }
    payload["components"] = components
    payload.pop("snapshot_cid", None)
    with pytest.raises(WorldSnapshotContractError, match="mismatch"):
        parse_world_snapshot(payload)

    payload = mutable_snapshot(example_current_snapshot())
    components = copy.deepcopy(payload["components"])
    components["datasets_semantic_state_root"] = {
        **components["datasets_semantic_state_root"],
        "claim_id": "claim:forbidden",
    }
    payload["components"] = components
    payload.pop("snapshot_cid", None)
    with pytest.raises(WorldSnapshotContractError, match="unknown fields"):
        parse_world_snapshot(payload)


def test_field_ownership_table_covers_required_roots() -> None:
    table = field_ownership_table()
    fields = {row["field"] for row in table}
    assert "datasets_semantic_state_root" in fields
    assert "event_cursor" in fields
    assert "coordination_epoch" in fields
    datasets = [
        row for row in table if row["authority"] == "datasets-semantic"
    ]
    assert datasets
    assert all(row["owner"] == "ipfs_datasets_py" for row in datasets)
    ducklake = next(
        row for row in table if row["field"] == "ducklake_projection_health"
    )
    assert ducklake["authority"] == "optional-non-authoritative-projection"
