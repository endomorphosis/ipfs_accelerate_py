"""Semantic world-model records live in DuckDB and project to DuckLake."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
    ProgramWorldDatabase,
    ProgramWorldDatabaseError,
    persist_program_world_record,
    records_for_decision,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_service import (
    ProgramWorldService,
)


def test_refuses_extra_gate_control_duckdb(tmp_path) -> None:
    path = tmp_path / "control.duckdb"
    store = ProgramWorldDatabase(path)
    with pytest.raises(ProgramWorldDatabaseError, match="control.duckdb"):
        store.persist({"task_id": "SAWM-039", "operation": "reuse"})


def test_persist_and_query_for_decision(tmp_path) -> None:
    store = ProgramWorldDatabase(tmp_path / "world_model.duckdb")
    stored = store.persist(
        {
            "task_id": "SAWM-016",
            "board": "sawm",
            "operation": "reuse",
            "verdict": "reject",
            "proposal_only": True,
            "completion_authority": False,
        }
    )
    assert stored["stored"] is True
    assert stored["completion_authority"] is False
    query = store.records_for_decision(task_id="SAWM-016", operation="reuse")
    assert query["n"] == 1
    assert query["records"][0]["task_id"] == "SAWM-016"
    assert query["completion_authority"] is False
    assert query["decision_authority"] is False
    assert query["ducklake_authoritative"] is False


def test_cannot_persist_admitted_completion(tmp_path) -> None:
    store = ProgramWorldDatabase(tmp_path / "world_model.duckdb")
    with pytest.raises(ProgramWorldDatabaseError, match="admitted"):
        store.persist({"task_id": "SAWM-039", "completion_authority": True})


def test_program_world_service_persists_when_configured(tmp_path, monkeypatch) -> None:
    path = tmp_path / "world_model.duckdb"
    monkeypatch.setenv("IPFS_ACCELERATE_PROGRAM_WORLD_DUCKDB", str(path))
    result = ProgramWorldService().operation("reuse", {"task_id": "SAWM-016"})
    assert result["completion_authority"] is False
    query = records_for_decision(task_id="SAWM-016", operation="reuse")
    assert query["n"] >= 1
    assert query["decision_authority"] is False


def test_unconfigured_store_is_skip(monkeypatch) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_PROGRAM_WORLD_DUCKDB", raising=False)
    skipped = persist_program_world_record({"task_id": "SAWM-039", "operation": "status"})
    assert skipped["status"] == "skip"
    empty = records_for_decision(task_id="SAWM-039")
    assert empty["n"] == 0
    assert empty["decision_authority"] is False


def test_ducklake_projection_is_observational(tmp_path) -> None:
    store = ProgramWorldDatabase(
        tmp_path / "world_model.duckdb",
        ducklake_root=tmp_path / "world_model_ducklake",
    )
    store.persist(
        {
            "task_id": "SAWM-039",
            "board": "sawm",
            "operation": "status",
            "completion_authority": False,
        }
    )
    projected = store.project_ducklake()
    assert projected["completion_authority"] is False
    assert projected["authoritative"] is False
    assert projected["status"] in {"projected", "unavailable"}
    if projected["status"] == "projected":
        assert projected["stored_records"] >= 1
