"""Supervisor meta-index links catalogs behind DuckLake without extra-gate attach."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
    SupervisorMetaIndex,
    SupervisorMetaIndexError,
    compose_for_subject,
    orchestration_view,
    register_taskboard,
)


def test_refuses_extra_gate_control_duckdb(tmp_path) -> None:
    index = SupervisorMetaIndex(tmp_path / "control.duckdb")
    with pytest.raises(SupervisorMetaIndexError, match="control.duckdb"):
        index.register_catalog(kind="ast", locator_ref="ast.duckdb")


def test_links_catalogs_and_composes_without_attaching_taskboard(tmp_path) -> None:
    index = SupervisorMetaIndex(tmp_path / "meta_index.duckdb")
    kinds = (
        "filesystem_mtime",
        "ast",
        "bm25",
        "knowledge_graph",
        "vector",
        "proof_cache",
        "proof_certificate",
        "world_model",
        "capsule",
        "metadata",
    )
    catalogs = {
        kind: index.register_catalog(kind=kind, locator_ref=f"{kind}.duckdb", tree_id="tree:1")
        for kind in kinds
    }
    board = index.register_catalog(
        kind="taskboard",
        locator_ref="/boards/sawm/control.duckdb",
        exclusive_owner="ipfs-taskboard-sawm-supervisor.service",
        repository_id="sawm",
        tree_id="tree:1",
    )
    assert board["attach_permitted"] is False
    for kind, catalog in catalogs.items():
        index.link_identity(
            subject_kind="path",
            subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
            catalog_id=catalog["catalog_id"],
            record_kind=kind,
            record_ref=f"{kind}:record:1",
            freshness_mtime_ns=1,
            capsule_cid="capsule:cli",
        )
    composed = index.compose_for_subject(
        subject_kind="path",
        subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
    )
    assert composed["n"] >= len(kinds)
    assert composed["completion_authority"] is False
    assert composed["event_driven_qualified"] is True
    assert all(item["attach_permitted"] is True for item in composed["linked"])
    view = index.orchestration_view(tree_id="tree:1")
    assert "taskboard" in view["kinds"]
    assert view["extra_gate_attached"] is False
    assert view["decision_authority"] is False
    board_row = next(item for item in view["catalogs"] if item["kind"] == "taskboard")
    assert board_row["attach_permitted"] is False


def test_unconfigured_meta_index_is_skip(monkeypatch) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    skipped = register_taskboard(
        board_id="sawm",
        exclusive_owner="ipfs-taskboard-sawm-supervisor.service",
    )
    assert skipped["status"] == "skip"
    empty = compose_for_subject(subject_kind="task_id", subject_ref="SAWM-039")
    assert empty["n"] == 0
    assert empty["event_driven_qualified"] is True
    view = orchestration_view()
    assert view["extra_gate_attached"] is False


def test_ducklake_projection_is_observational(tmp_path) -> None:
    index = SupervisorMetaIndex(
        tmp_path / "meta_index.duckdb",
        ducklake_root=tmp_path / "meta_index_ducklake",
    )
    index.register_catalog(kind="ast", locator_ref="ast.duckdb")
    projected = index.project_ducklake()
    assert projected["completion_authority"] is False
    assert projected["authoritative"] is False
    assert projected["status"] in {"projected", "unavailable"}
    if projected["status"] == "projected":
        assert projected["stored_catalogs"] >= 1
