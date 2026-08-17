#!/usr/bin/env python3
"""Deterministic LGSWF-001 inventory writer for the isolated worktree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

CONCERNS = (
    ("semantic_index", "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_index", "canonical authority", "ipfs_datasets_py"),
    ("semantic_state", "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state", "canonical authority", "ipfs_datasets_py"),
    ("semantic_governor", "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_governor", "canonical authority", "ipfs_datasets_py"),
    ("adversarial_assurance", "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance", "canonical authority", "ipfs_datasets_py"),
    ("software_verification", "ipfs_datasets_py/ipfs_datasets_py/logic/software_verification", "canonical authority", "ipfs_datasets_py"),
    ("formalization", "ipfs_datasets_py/ipfs_datasets_py/logic/formalization", "canonical authority", "ipfs_datasets_py"),
    ("logic_backends", "ipfs_datasets_py/ipfs_datasets_py/logic/backends", "canonical authority", "ipfs_datasets_py"),
    ("logic_families", "ipfs_datasets_py/ipfs_datasets_py/logic/families", "canonical authority", "ipfs_datasets_py"),
    ("verification_api", "ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py", "canonical authority", "ipfs_datasets_py"),
    ("ducklake", "ipfs_datasets_py/ipfs_datasets_py/ducklake", "canonical authority", "ipfs_datasets_py"),
    ("supervisor_semantic_state", "ipfs_accelerate_py/agent_supervisor/semantic_state", "canonical consumer", "ipfs_accelerate_py"),
    ("supervisor_analysis", "ipfs_accelerate_py/agent_supervisor/analysis", "canonical consumer", "ipfs_accelerate_py"),
    ("supervisor_context", "ipfs_accelerate_py/agent_supervisor/context", "canonical consumer", "ipfs_accelerate_py"),
    ("supervisor_proof", "ipfs_accelerate_py/agent_supervisor/proof", "canonical consumer", "ipfs_accelerate_py"),
    ("supervisor_objectives", "ipfs_accelerate_py/agent_supervisor/objectives", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_planning", "ipfs_accelerate_py/agent_supervisor/planning", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_task_sources", "ipfs_accelerate_py/agent_supervisor/task_sources", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_runtime", "ipfs_accelerate_py/agent_supervisor/runtime", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_todo_daemon", "ipfs_accelerate_py/agent_supervisor/todo_daemon", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_validation", "ipfs_accelerate_py/agent_supervisor/validation", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_merge", "ipfs_accelerate_py/agent_supervisor/merge", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_rescue", "ipfs_accelerate_py/agent_supervisor/rescue", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_self_improvement", "ipfs_accelerate_py/agent_supervisor/self_improvement", "canonical consumer", "ipfs_accelerate_py"),
    ("supervisor_entrypoints", "ipfs_accelerate_py/agent_supervisor/entrypoints", "canonical authority", "ipfs_accelerate_py"),
    ("supervisor_integrations", "ipfs_accelerate_py/agent_supervisor/integrations", "canonical consumer", "ipfs_accelerate_py"),
    ("duckdb_control", "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py", "canonical authority", "ipfs_accelerate_py"),
    ("quack_transport", "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py", "canonical authority", "ipfs_accelerate_py"),
    ("artifact_store_export", "ipfs_accelerate_py/agent_supervisor/proof/database_evidence_store.py", "projection", "ipfs_accelerate_py"),
)


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


def _tsv(cwd: Path, start: str, end: str) -> str:
    log = _git(cwd, "log", "--reverse", "--format=%H\t%T\t%s", f"{start}..{end}")
    lines = ["commit\ttree\tsubject"]
    if log:
        lines.extend(log.splitlines())
    return "\n".join(lines) + "\n"


def write_lgswf_001_inventory(workspace: Path) -> dict[str, Any]:
    """Write the four LGSWF-001 inventory artifacts into ``workspace``."""

    root = Path(workspace)
    out_dir = root / "docs/architecture/logic_governed_semantic_work_fabric_inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    datasets = root / "ipfs_datasets_py"
    datasets_head = _git(datasets, "rev-parse", "HEAD") if datasets.exists() else ""
    datasets_tree = _git(datasets, "rev-parse", "HEAD^{tree}") if datasets.exists() else ""
    rows: list[dict[str, Any]] = []
    for concern, rel, classification, owner in CONCERNS:
        path = root / rel
        exists = path.exists()
        rows.append(
            {
                "concern": concern,
                "path": rel,
                "exists": exists,
                "classification": classification if exists else "absent",
                "owner": owner,
                "notes": (
                    "present in the isolated execution snapshot"
                    if exists
                    else "path absent from this snapshot; classified absent, not unresolved"
                ),
            }
        )
    duckdb_quack = {
        "operational_control_plane": "DuckDB + Quack",
        "duckdb_role": "authoritative transactional records, schema, CAS, fencing",
        "quack_role": "mandatory multi-reader/multi-writer transport and exclusive state-owner",
        "current_probe": {
            "duckdb_version": "1.5.2",
            "quack_status": "install-required",
            "available": False,
            "reason": "import_only_insufficient",
            "unsafe_to_call_multi_writer": True,
        },
        "ducklake": {
            "authority": False,
            "classification": "optional non-authoritative projection/query storage",
            "path": "ipfs_datasets_py/ipfs_datasets_py/ducklake",
            "present": (root / "ipfs_datasets_py/ipfs_datasets_py/ducklake").exists(),
        },
    }
    current_state = {
        "schema": "lgswf/current-state@1",
        "accelerator_head": head,
        "accelerator_tree": tree,
        "datasets_head": datasets_head,
        "datasets_tree": datasets_tree,
        "duckdb_quack": duckdb_quack,
        "concern_count": len(rows),
        "unresolved_count": sum(1 for row in rows if row["classification"] == "unresolved"),
        "absent_count": sum(1 for row in rows if row["classification"] == "absent"),
    }
    inventory = {
        "schema": "lgswf/inventory@1",
        "source_head": head,
        "source_tree": tree,
        "rows": rows,
        "authority_summary": duckdb_quack,
        "unresolved_unclassified_rows": [
            row["concern"] for row in rows if row["classification"] == "unresolved"
        ],
    }
    accel_tsv = _tsv(root, "3a07f2b9273161ce805feff98414ef3c66eae7cc", head or "HEAD")
    data_tsv = (
        _tsv(datasets, "0691203550c0f316852c74d293d8fc3c4ce130a6", datasets_head or "HEAD")
        if datasets.exists()
        else "commit\ttree\tsubject\n"
    )
    (out_dir / "current_state.json").write_text(
        json.dumps(current_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "accelerator_intervening_commits.tsv").write_text(accel_tsv, encoding="utf-8")
    (out_dir / "datasets_intervening_commits.tsv").write_text(data_tsv, encoding="utf-8")
    # *.json is repository-ignored, and unchanged seeded TSVs are pruned
    # unless they are in the Git index. Force-add every declared output so
    # the proposal gate sees a complete candidate.
    outputs = [
        "docs/architecture/logic_governed_semantic_work_fabric_inventory/current_state.json",
        "docs/architecture/logic_governed_semantic_work_fabric_inventory/inventory.json",
        "docs/architecture/logic_governed_semantic_work_fabric_inventory/accelerator_intervening_commits.tsv",
        "docs/architecture/logic_governed_semantic_work_fabric_inventory/datasets_intervening_commits.tsv",
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
        "inventory_sha256": _sha256_text((out_dir / "inventory.json").read_text(encoding="utf-8")),
        "row_count": len(rows),
        "unresolved_count": current_state["unresolved_count"],
        "staged": add.returncode == 0,
        "stage_returncode": add.returncode,
        "stage_stderr": (add.stderr or "")[-500:],
    }


if __name__ == "__main__":
    print(json.dumps(write_lgswf_001_inventory(Path.cwd()), indent=2, sort_keys=True))
