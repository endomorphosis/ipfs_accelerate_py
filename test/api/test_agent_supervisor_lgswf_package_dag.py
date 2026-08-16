"""Hermetic checks for the LGSWF-002 package DAG and interface freeze."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs/architecture/logic_governed_semantic_work_fabric_inventory"
REQUIRED = (
    "StateRepository",
    "QuackStateRepository",
    "QuackStateClient",
    "QuackStateServer",
    "StateServerIdentity",
    "ControlPlaneStoreIdentity",
    "StateCommand",
    "DatabaseProgramConfig",
    "ipfs_datasets_py.duckdb_control.quack_security scoped server authorization",
)


def _load(name: str) -> dict:
    payload = json.loads((INVENTORY / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_package_dag_records_nodes_and_has_no_upward_imports() -> None:
    dag = _load("package_dag.json")
    assert dag["schema"] == "lgswf/package-dag@1"
    ids = {node["id"] for node in dag["nodes"]}
    assert "ipfs_datasets_py.logic" in ids
    assert "ipfs_accelerate_py.agent_supervisor.task_sources" in ids
    assert dag["sccs"] == []
    assert dag["upward_imports"] == []


def test_authority_map_keeps_datasets_semantic_and_accelerator_operational() -> None:
    authority = _load("authority_map.json")
    assert authority["semantic_authority"] == "ipfs_datasets_py"
    assert authority["operational_authority"] == "ipfs_accelerate_py"
    assert authority["ducklake_authority"] is False


def test_interface_freeze_seals_required_named_contracts() -> None:
    freeze = _load("interface_freeze.json")
    names = [item["name"] for item in freeze["interfaces"]]
    assert names == list(REQUIRED)
    assert freeze["undefined_count"] == 0
    assert all(item["canonical"] and not item["duplicate"] for item in freeze["interfaces"])
    assert freeze["control_plane_split"]["direct_shared_duckdb_file_access"] is False
    assert freeze["control_plane_split"]["embedded_one_writer_bootstrap"] is True
    for item in freeze["interfaces"]:
        assert (ROOT / item["path"]).is_file(), item["path"]
        assert item["defined"] is True
