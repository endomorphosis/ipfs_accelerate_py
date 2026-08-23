"""Hermetic PCAR-001 repository, entrypoint, and store inventory tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import AGENT_SUPERVISOR_DOMAIN_PACKAGES

ROOT = Path(__file__).resolve().parents[3]
INV = ROOT / "docs/architecture/architecture_refactorer_inventory"
SUPERVISOR = ROOT / "ipfs_accelerate_py" / "agent_supervisor"
PYPROJECT = ROOT / "pyproject.toml"
SEALED_COMMIT = "a2d1529934197dc64fe18cfbaec9dc7daf438703"
REACHABILITY = {"production", "test", "compatibility", "simulation"}
REQUIRED_OVERLAPPING_KINDS = {
    "accelerator",
    "provider",
    "hardware",
    "endpoint",
    "compatibility",
}
INITIAL_CONCERNS = {
    "content identity",
    "operation identity",
    "provider capability",
    "provider selection",
    "execution result",
    "task identity",
    "objective identity",
    "policy decision",
    "authorization",
    "confirmation",
    "lease and fencing",
    "state persistence",
    "proof verification",
    "test evidence",
    "completion evidence",
    "release qualification",
}


def _load(name: str) -> dict:
    raw = (INV / name).read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert raw == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return payload


def _python_files(package: Path) -> list[Path]:
    return sorted(
        path
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    )


def _first_level_packages() -> set[str]:
    paths: set[str] = set()
    for child in SUPERVISOR.iterdir():
        if not child.is_dir() or child.name.startswith(".") or child.name == "__pycache__":
            continue
        if (child / "__init__.py").is_file() or any(child.glob("*.py")):
            paths.add(child.relative_to(ROOT).as_posix())
    return paths


def _residual_root_modules() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in SUPERVISOR.glob("*.py")
        if path.name != "__init__.py"
    }


def _assert_span(span: dict, *, required: bool = True) -> None:
    if span is None:
        assert required is False
        return
    assert isinstance(span["path"], str) and span["path"]
    assert isinstance(span["start_line"], int)
    assert isinstance(span["end_line"], int)
    assert span["start_line"] >= 1
    assert span["end_line"] >= span["start_line"]
    path = ROOT / span["path"]
    assert path.is_file(), span["path"]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    assert span["end_line"] <= len(lines), span


def _pyproject_console_scripts() -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"\[project\.scripts\]\n(.*?)(?:\n\[|\Z)", text, re.S)
    assert block is not None
    names: list[str] = []
    for line in block.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        names.append(stripped.split("=", 1)[0].strip())
    return names


def test_required_root_coverage() -> None:
    inventory = _load("current_repository_inventory.json")
    assert inventory["schema"].endswith("current-repository-inventory@1")
    assert inventory["authority"] is False
    assert inventory["task_id"] == "PCAR-001"
    assert inventory["required_root"] == "ipfs_accelerate_py/agent_supervisor"
    assert (ROOT / inventory["required_root"]).is_dir()
    assert inventory["repository_tree"] == f"{SEALED_COMMIT}^{{tree}}"
    assert inventory["inspection"]["method"] == "static_source_inventory"
    assert inventory["inspection"]["nonclaim"]

    paths = {item["path"] for item in inventory["packages"]}
    assert "ipfs_accelerate_py/agent_supervisor/todo_daemon" in paths
    assert "ipfs_accelerate_py/agent_supervisor/control" in paths
    assert "ipfs_accelerate_py/agent_supervisor/architecture_refactorer" in paths
    assert "ipfs_accelerate_py/agent_supervisor/entrypoints" in paths
    live_packages = _first_level_packages()
    assert paths == live_packages
    for name in AGENT_SUPERVISOR_DOMAIN_PACKAGES:
        assert f"ipfs_accelerate_py/agent_supervisor/{name}" in paths

    for item in inventory["packages"]:
        assert item["kind"] == "package"
        assert item["reachability"] in REACHABILITY
        package_path = ROOT / item["path"]
        assert package_path.exists()
        assert item["python_file_count"] == len(_python_files(package_path))
        live_samples = [
            path.relative_to(ROOT).as_posix()
            for path in _python_files(package_path)[:8]
        ]
        assert item["sample_files"] == live_samples
        _assert_span(item["source_span"])
        assert isinstance(item["authority_candidates"], list)

    residual_paths = {item["path"] for item in inventory["residual_root_modules"]}
    assert residual_paths == _residual_root_modules()
    for item in inventory["residual_root_modules"]:
        assert item["kind"] == "module"
        assert item["reachability"] in REACHABILITY
        assert (ROOT / item["path"]).is_file()
        _assert_span(item["source_span"])
        assert item["uncertainty"]

    overlapping_kinds = {item["kind"] for item in inventory["overlapping_paths"]}
    assert REQUIRED_OVERLAPPING_KINDS <= overlapping_kinds
    assert {"fixture", "simulation", "test"} <= overlapping_kinds
    absent_overlapping = [item for item in inventory["overlapping_paths"] if not item["present"]]
    assert absent_overlapping
    for item in inventory["overlapping_paths"]:
        present = (ROOT / item["path"]).exists()
        assert item["present"] is present
        assert item["reachability"] in REACHABILITY
        if present:
            _assert_span(item["source_span"])
        else:
            assert item["source_span"] is None
            assert item["uncertainty"]
    hardware_paths = {
        item["path"]
        for item in inventory["overlapping_paths"]
        if item["kind"] == "hardware"
    }
    if (ROOT / "common/hardware_detection.py").is_file():
        assert "common/hardware_detection.py" in hardware_paths
    concerns = {item["concern"] for item in inventory["authority_candidates"]}
    assert INITIAL_CONCERNS <= concerns
    for item in inventory["authority_candidates"]:
        assert item["present"] is (ROOT / item["path"]).exists()
        _assert_span(item["source_span"])
        assert item["confidence"] in {"exact", "conservative", "heuristic", "opaque"}
        if item["confidence"] != "exact":
            assert item["uncertainty"]

    assert inventory["dynamic_loading"]
    for item in inventory["dynamic_loading"]:
        assert item["present"] is True
        assert (ROOT / item["path"]).is_file()
        _assert_span(item["source_span"])
        assert item["uncertainty"]
        assert item["mechanism"]


def test_entrypoint_coverage() -> None:
    entrypoints = _load("current_entrypoints.json")
    assert entrypoints["schema"].endswith("current-entrypoints@1")
    assert entrypoints["authority"] is False
    assert entrypoints["task_id"] == "PCAR-001"
    names = {item["name"] for item in entrypoints["entrypoints"]}
    assert "ipfs-accelerate-agent-implementation-supervisor" in names
    assert "run_agent_supervisor_architecture_refactorer" in names
    assert "ipfs-accelerate agent" in names
    assert "agent_supervisor" in names
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    declared_scripts = _pyproject_console_scripts()
    console_names = [
        item["name"]
        for item in entrypoints["entrypoints"]
        if item["kind"] == "console_script"
    ]
    assert console_names == declared_scripts
    for item in entrypoints["entrypoints"]:
        _assert_span(item["source_span"])
        if item["kind"] == "console_script":
            assert item["name"] in pyproject
            assert ":" in item["target"]
        else:
            assert (ROOT / item["target"]).is_file()
        assert item["reachability"] in REACHABILITY
    mcp_category = next(
        item for item in entrypoints["entrypoints"] if item["name"] == "agent_supervisor"
    )
    assert mcp_category["kind"] == "mcp_category"
    assert mcp_category["uncertainty"]


def test_store_inventory_and_uncertainty_retained() -> None:
    stores = _load("current_state_stores.json")
    assert stores["schema"].endswith("current-state-stores@1")
    assert stores["authority"] is False
    assert stores["task_id"] == "PCAR-001"
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
        _assert_span(item["source_span"])
        if item["disposition"] == "unknown":
            assert item["uncertainty"]
        if item["kind"] == "DuckDB tables":
            assert item["disposition"] == "authoritative"
            assert item["uncertainty"]
            assert item["tables"]
        if item["kind"] == "Markdown task boards":
            assert item["disposition"] == "materialized_projection"
            assert (ROOT / item["path"]).is_file()
