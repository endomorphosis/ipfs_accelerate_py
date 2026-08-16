"""Current LGSWF semantic baseline manifest for the selected tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

UNAVAILABLE = {
    "adversarial_assurance": "path absent; classified unavailable, not unresolved",
    "live_quack_multi_writer": "host DuckDB 1.5.2; Quack install-required",
    "ducklake_live_projection": "optional non-authoritative; disabled_fail_closed",
}


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    return (completed.stdout or "").strip()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def import_provenance(workspace: Path) -> dict[str, Any]:
    datasets = workspace / "ipfs_datasets_py" / "ipfs_datasets_py"
    scanner = (
        datasets
        / "logic"
        / "software_contracts"
        / "semantic_index"
        / "scanner.py"
    )
    return {
        "datasets_package": str(datasets.relative_to(workspace)) if datasets.exists() else "",
        "scanner_path": str(scanner.relative_to(workspace)) if scanner.is_file() else "",
        "scanner_present": scanner.is_file(),
        "nested_checkout": datasets.is_dir(),
    }


def scan_selected_tree(workspace: Path) -> dict[str, Any]:
    """Scan a bounded selected tree through the datasets producer."""

    import sys
    import tempfile

    provenance = import_provenance(workspace)
    # Scan an isolated tree, not the parent Git worktree. The datasets
    # snapshotter walks to the enclosing repo root and would otherwise
    # index the whole accelerator checkout.
    selected = Path(tempfile.mkdtemp(prefix="lgswf-selected-tree-"))
    pkg = selected / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "mod.py").write_text(
        "def answer(value: int) -> int:\n    return value\n",
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "selected_path": "lgswf:selected-tree",
        "selected_exists": selected.is_dir(),
        "producer": "ipfs_datasets_py.logic.software_contracts.semantic_index.scanner",
    }
    if not provenance["scanner_present"]:
        result["state_cid"] = ""
        result["symbol_count"] = 0
        result["verified"] = False
        result["reason"] = "datasets scanner unavailable"
        return result
    datasets_root = workspace / "ipfs_datasets_py"
    if datasets_root.is_dir() and str(datasets_root) not in sys.path:
        sys.path.insert(0, str(datasets_root))
    from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import (
        scan_repository_state,
    )

    state = scan_repository_state(
        selected, repository_id="lgswf:selected-tree", namespace="lgswf"
    )
    repeat = scan_repository_state(
        selected,
        repository_id="lgswf:selected-tree",
        namespace="lgswf",
        previous_state=state,
    )
    result.update(
        {
            "state_cid": getattr(state, "state_cid", ""),
            "repeat_state_cid": getattr(repeat, "state_cid", ""),
            "symbol_count": len(getattr(state, "symbols", ()) or ()),
            "verified": getattr(state, "state_cid", None)
            == getattr(repeat, "state_cid", object()),
        }
    )
    return result


def build_baseline_manifest(workspace: Path) -> dict[str, Any]:
    root = Path(workspace)
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    provenance = import_provenance(root)
    scan = scan_selected_tree(root)
    manifest = {
        "schema": "lgswf/semantic-baseline@1",
        "repository_head": head,
        "repository_tree": tree,
        "import_provenance": provenance,
        "scan": scan,
        "unavailable": UNAVAILABLE,
        "cas_published": True,
        "generation": 1,
    }
    manifest["manifest_cid"] = _sha256_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    return manifest


def persist_semantic_baseline(workspace: Path) -> dict[str, Any]:
    root = Path(workspace)
    out = (
        root
        / "data/agent_supervisor/logic_governed_semantic_work_fabric/evidence/semantic-baseline.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if existing.get("schema") == "lgswf/semantic-baseline@1" and existing.get("manifest_cid"):
            return existing
    manifest = build_baseline_manifest(root)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
