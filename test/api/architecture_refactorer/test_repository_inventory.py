"""Hermetic PCAR-001 repository, entrypoint, and store inventory tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INV = ROOT / "docs/architecture/architecture_refactorer_inventory"
SUPERVISOR = ROOT / "ipfs_accelerate_py" / "agent_supervisor"


def _load(name: str) -> dict:
    raw = (INV / name).read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert raw == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return payload


def test_required_root_coverage() -> None:
    inventory = _load("current_repository_inventory.json")
    assert inventory["schema"].endswith("current-repository-inventory@1")
    assert inventory["authority"] is False
    assert inventory["task_id"] == "PCAR-001"
    assert inventory["required_root"] == "ipfs_accelerate_py/agent_supervisor"
    assert (ROOT / inventory["required_root"]).is_dir()
    paths = {item["path"] for item in inventory["packages"]}
    assert "ipfs_accelerate_py/agent_supervisor/todo_daemon" in paths
    assert "ipfs_accelerate_py/agent_supervisor/control" in paths
    assert "ipfs_accelerate_py/agent_supervisor/architecture_refactorer" in paths
    for item in inventory["packages"]:
        assert item["kind"] == "package"
        assert item["reachability"] in {"production", "test", "compatibility", "simulation"}
        assert (ROOT / item["path"]).exists()
    for item in inventory["overlapping_paths"]:
        present = (ROOT / item["path"]).exists()
        assert item["present"] is present
        if not present:
            assert item["uncertainty"]


def test_entrypoint_coverage() -> None:
    entrypoints = _load("current_entrypoints.json")
    assert entrypoints["schema"].endswith("current-entrypoints@1")
    names = {item["name"] for item in entrypoints["entrypoints"]}
    assert "ipfs-accelerate-agent-implementation-supervisor" in names
    assert "run_agent_supervisor_architecture_refactorer" in names
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for item in entrypoints["entrypoints"]:
        if item["kind"] == "console_script":
            assert item["name"] in pyproject
        else:
            assert (ROOT / item["target"]).is_file()


def test_store_inventory_and_uncertainty_retained() -> None:
    stores = _load("current_state_stores.json")
    assert stores["schema"].endswith("current-state-stores@1")
    kinds = [item["kind"] for item in stores["stores"]]
    required = json.loads(
        (INV / "state_store_baseline.json").read_text(encoding="utf-8")
    )["required_kinds"]
    assert kinds == required
    closed = set(
        json.loads((INV / "state_store_baseline.json").read_text(encoding="utf-8"))[
            "closed_dispositions"
        ]
    )
    for item in stores["stores"]:
        assert item["disposition"] in closed
        if item["disposition"] == "unknown":
            assert item["uncertainty"]
