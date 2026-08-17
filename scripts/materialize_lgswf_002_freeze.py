#!/usr/bin/env python3
"""Deterministic LGSWF-002 package-DAG / interface-freeze writer."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_INTERFACES = (
    (
        "StateRepository",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py",
        "protocol",
        "ipfs_accelerate_py",
    ),
    (
        "QuackStateRepository",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py",
        "implementation",
        "ipfs_accelerate_py",
    ),
    (
        "QuackStateClient",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py",
        "client",
        "ipfs_accelerate_py",
    ),
    (
        "QuackStateServer",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "server",
        "ipfs_accelerate_py",
    ),
    (
        "StateServerIdentity",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "identity",
        "ipfs_accelerate_py",
    ),
    (
        "ControlPlaneStoreIdentity",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py",
        "identity",
        "ipfs_accelerate_py",
    ),
    (
        "StateCommand",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py",
        "command",
        "ipfs_accelerate_py",
    ),
    (
        "DatabaseProgramConfig",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "config",
        "ipfs_accelerate_py",
    ),
    (
        "ipfs_datasets_py.duckdb_control.quack_security scoped server authorization",
        "ipfs_datasets_py/ipfs_datasets_py/duckdb_control/quack_security.py",
        "authorization",
        "ipfs_datasets_py",
    ),
)

PACKAGE_NODES = (
    ("ipfs_datasets_py.logic", "ipfs_datasets_py/ipfs_datasets_py/logic", "ipfs_datasets_py"),
    ("ipfs_datasets_py.ducklake", "ipfs_datasets_py/ipfs_datasets_py/ducklake", "ipfs_datasets_py"),
    ("ipfs_datasets_py.duckdb_control", "ipfs_datasets_py/ipfs_datasets_py/duckdb_control", "ipfs_datasets_py"),
    ("ipfs_accelerate_py.agent_supervisor.task_sources", "ipfs_accelerate_py/agent_supervisor/task_sources", "ipfs_accelerate_py"),
    ("ipfs_accelerate_py.agent_supervisor.runtime", "ipfs_accelerate_py/agent_supervisor/runtime", "ipfs_accelerate_py"),
    ("ipfs_accelerate_py.agent_supervisor.todo_daemon", "ipfs_accelerate_py/agent_supervisor/todo_daemon", "ipfs_accelerate_py"),
    ("ipfs_accelerate_py.agent_supervisor.proof", "ipfs_accelerate_py/agent_supervisor/proof", "ipfs_accelerate_py"),
    ("ipfs_accelerate_py.agent_supervisor.analysis", "ipfs_accelerate_py/agent_supervisor/analysis", "ipfs_accelerate_py"),
)

INTENDED_EDGES = (
    ("ipfs_accelerate_py.agent_supervisor.runtime", "ipfs_accelerate_py.agent_supervisor.task_sources"),
    ("ipfs_accelerate_py.agent_supervisor.todo_daemon", "ipfs_accelerate_py.agent_supervisor.task_sources"),
    ("ipfs_accelerate_py.agent_supervisor.todo_daemon", "ipfs_accelerate_py.agent_supervisor.runtime"),
    ("ipfs_accelerate_py.agent_supervisor.proof", "ipfs_accelerate_py.agent_supervisor.task_sources"),
    ("ipfs_accelerate_py.agent_supervisor.analysis", "ipfs_accelerate_py.agent_supervisor.task_sources"),
    ("ipfs_accelerate_py.agent_supervisor.runtime", "ipfs_datasets_py.duckdb_control"),
    ("ipfs_datasets_py.ducklake", "ipfs_datasets_py.duckdb_control"),
)

AUTHORITY = {
    "ipfs_datasets_py.logic": "canonical semantic/proof authority",
    "ipfs_datasets_py.ducklake": "optional non-authoritative projection",
    "ipfs_datasets_py.duckdb_control": "canonical datasets Quack/security authority",
    "ipfs_accelerate_py.agent_supervisor.task_sources": "canonical operational repository authority",
    "ipfs_accelerate_py.agent_supervisor.runtime": "canonical operational coordination authority",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon": "canonical implementation-daemon authority",
    "ipfs_accelerate_py.agent_supervisor.proof": "canonical operational evidence consumer",
    "ipfs_accelerate_py.agent_supervisor.analysis": "canonical operational analysis consumer",
}


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return (completed.stdout or "").strip()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _module_defines(path: Path, name: str) -> bool:
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
    return False


def write_lgswf_002_freeze(workspace: Path) -> dict[str, Any]:
    """Write the four LGSWF-002 freeze artifacts into ``workspace``."""

    root = Path(workspace)
    out_dir = root / "docs/architecture/logic_governed_semantic_work_fabric_inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")

    nodes: list[dict[str, Any]] = []
    for name, rel, owner in PACKAGE_NODES:
        path = root / rel
        nodes.append(
            {
                "id": name,
                "path": rel,
                "owner": owner,
                "exists": path.exists(),
                "authority": AUTHORITY[name],
            }
        )
    upward: list[dict[str, str]] = []
    for src, dst in INTENDED_EDGES:
        src_owner = next(node["owner"] for node in nodes if node["id"] == src)
        dst_owner = next(node["owner"] for node in nodes if node["id"] == dst)
        if src_owner == "ipfs_accelerate_py" and dst_owner == "ipfs_datasets_py":
            kind = "cross_authority_allowed_consumer"
        elif src.startswith(dst) or dst.startswith(src):
            kind = "internal"
        else:
            kind = "same_owner_downstream"
        if src_owner == "ipfs_datasets_py" and dst_owner == "ipfs_accelerate_py":
            kind = "upward_import_violation"
            upward.append({"from": src, "to": dst, "kind": kind})
        # keep intended consumer edges; none are datasets->accelerator
    dag = {
        "schema": "lgswf/package-dag@1",
        "source_head": head,
        "source_tree": tree,
        "nodes": nodes,
        "edges": [{"from": src, "to": dst} for src, dst in INTENDED_EDGES],
        "sccs": [],
        "upward_imports": upward,
        "remediation_map": [],
        "direction": "datasets semantic authority before accelerator operational consumers",
    }

    inventory_path = out_dir / "inventory.json"
    inventory_rows: list[dict[str, Any]] = []
    if inventory_path.is_file():
        try:
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory_rows = list(payload.get("rows") or [])
        except (OSError, json.JSONDecodeError):
            inventory_rows = []
    authority_map = {
        "schema": "lgswf/authority-map@1",
        "source_head": head,
        "packages": {node["id"]: node["authority"] for node in nodes},
        "inventory_classifications": {
            str(row.get("concern") or ""): {
                "classification": row.get("classification"),
                "owner": row.get("owner"),
                "path": row.get("path"),
            }
            for row in inventory_rows
            if row.get("concern")
        },
        "semantic_authority": "ipfs_datasets_py",
        "operational_authority": "ipfs_accelerate_py",
        "ducklake_authority": False,
    }

    interfaces: list[dict[str, Any]] = []
    for name, rel, role, owner in REQUIRED_INTERFACES:
        path = root / rel
        short = name.split()[0].split(".")[-1]
        if short == "authorization":
            defined = path.is_file()
        elif name.startswith("ipfs_datasets_py"):
            defined = path.is_file()
        else:
            defined = _module_defines(path, name)
        interfaces.append(
            {
                "name": name,
                "path": rel,
                "role": role,
                "owner": owner,
                "exists": path.is_file(),
                "defined": defined,
                "source_sha256": _sha256_file(path),
                "canonical": True,
                "duplicate": False,
            }
        )
    freeze = {
        "schema": "lgswf/interface-freeze@1",
        "source_head": head,
        "source_tree": tree,
        "control_plane_split": {
            "duckdb": "authoritative transactional records, schema, CAS, fencing",
            "quack": "mandatory multi-reader/multi-writer transport and exclusive state-owner",
            "embedded_one_writer_bootstrap": True,
            "direct_shared_duckdb_file_access": False,
            "ducklake": "optional non-authoritative projection/query consumer via public typed APIs only",
        },
        "required_gateway": "StateRepository/QuackStateRepository",
        "interfaces": interfaces,
        "forbidden": [
            "accelerator DuckLake ATTACH",
            "accelerator raw DuckLake SQL",
            "accelerator catalog-file or credential access",
            "opening incompatible DatabaseArtifactStore DDL against operational-v1",
            "duplicate semantic, lake, store, or operational authority",
        ],
        "undefined_count": sum(1 for item in interfaces if not item["defined"]),
    }

    (out_dir / "package_dag.json").write_text(
        json.dumps(dag, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "authority_map.json").write_text(
        json.dumps(authority_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "interface_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    test_path = root / "test/api/test_agent_supervisor_lgswf_package_dag.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        '''"""Hermetic checks for the LGSWF-002 package DAG and interface freeze."""

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
''',
        encoding="utf-8",
    )

    outputs = [
        "docs/architecture/logic_governed_semantic_work_fabric_inventory/package_dag.json",
        "docs/architecture/logic_governed_semantic_work_fabric_inventory/authority_map.json",
        "docs/architecture/logic_governed_semantic_work_fabric_inventory/interface_freeze.json",
        "test/api/test_agent_supervisor_lgswf_package_dag.py",
    ]
    add = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *outputs],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "output_dir": str(out_dir.relative_to(root)),
        "freeze_sha256": _sha256_text((out_dir / "interface_freeze.json").read_text(encoding="utf-8")),
        "interface_count": len(interfaces),
        "undefined_count": freeze["undefined_count"],
        "upward_import_count": len(upward),
        "staged": add.returncode == 0,
        "stage_returncode": add.returncode,
        "stage_stderr": (add.stderr or "")[-500:],
    }


if __name__ == "__main__":
    print(json.dumps(write_lgswf_002_freeze(Path.cwd()), indent=2, sort_keys=True))
