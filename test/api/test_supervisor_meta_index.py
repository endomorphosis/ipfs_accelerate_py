"""Supervisor meta-index links catalogs behind DuckLake without extra-gate attach."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
    SupervisorMetaIndex,
    SupervisorMetaIndexError,
    compose_for_subject,
    compose_semantic_work,
    mirror_capsule_record,
    mirror_vector_index,
    mirror_work_record,
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


def test_bind_supervisor_catalogs_and_observe_path(tmp_path) -> None:
    index = SupervisorMetaIndex(tmp_path / "meta_index.duckdb")
    bound = index.bind_supervisor_catalogs(
        tree_id="tree:work",
        locators={
            "ast": str(tmp_path / "ast.duckdb"),
            "bm25": str(tmp_path / "bm25.duckdb"),
            "vector": str(tmp_path / "vector.duckdb"),
            "knowledge_graph": str(tmp_path / "kg.duckdb"),
            "world_model": str(tmp_path / "world_model.duckdb"),
            "proof_cache": str(tmp_path / "proof.duckdb"),
            "proof_certificate": str(tmp_path / "proof.duckdb"),
        },
    )
    assert bound["extra_gate_attached"] is False
    assert bound["event_driven_qualified"] is True
    kinds = {item["catalog_id"].split(":")[1] for item in bound["catalogs"]}
    assert "ast" in kinds
    assert "taskboard" in kinds
    observed = index.observe_path(
        "ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
        mtime_ns=42,
        extra_kinds=("ast", "bm25", "vector", "world_model"),
        capsule_cid="capsule:cli",
    )
    assert observed["n"] >= 2
    work = index.compose_semantic_work(
        subject_kind="path",
        subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
        tree_id="tree:work",
    )
    assert work["capsule_composition"] is True
    assert work["extra_gate_attached"] is False
    assert "ast" in work["formal_surfaces"] or "world_model" in work["formal_surfaces"]
    assert work["completion_authority"] is False


def test_mirror_capsule_and_vector_into_composition(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    capsule = mirror_capsule_record(
        capsule_cid="capsule:cli",
        node_id="node:cli",
        tree_id="tree:work",
        dependency_cids=("capsule:dep",),
    )
    assert capsule["completion_authority"] is False
    assert capsule["n"] >= 1
    vector = mirror_vector_index(
        tree_id="tree:work",
        index_id="vector:1",
        paths=("ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",),
    )
    assert vector["n"] >= 1
    work = compose_semantic_work(
        subject_kind="path",
        subject_ref="ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",
        tree_id="tree:work",
    )
    assert work["capsule_composition"] is True
    assert work["extra_gate_attached"] is False
    assert "vector" in work["formal_surfaces"] or "vector" in work["kinds"]


def test_mirror_plan_synthesis_and_bm25_work_records(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(tmp_path / "meta_index.duckdb"))
    plan = mirror_work_record(
        catalog_kind="metadata",
        record_kind="plan",
        record_ref="plan:1",
        subject_kind="record_cid",
        subject_ref="plan-content:1",
    )
    synthesis = mirror_work_record(
        catalog_kind="metadata",
        record_kind="synthesis",
        record_ref="repair:1",
        tree_id="tree:work",
        paths=("ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py",),
    )
    bm25 = mirror_work_record(
        catalog_kind="bm25",
        record_kind="bm25",
        record_ref="bm25:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    analysis = mirror_work_record(
        catalog_kind="metadata",
        record_kind="static_analysis",
        record_ref="ast:1",
        tree_id="tree:work",
        subject_kind="tree_id",
        subject_ref="tree:work",
    )
    for item in (plan, synthesis, bm25, analysis):
        assert item["completion_authority"] is False
        assert item["n"] >= 1
    work = compose_semantic_work(
        subject_kind="tree_id",
        subject_ref="tree:work",
        tree_id="tree:work",
    )
    assert work["extra_gate_attached"] is False
    assert work["event_driven_qualified"] is True


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
