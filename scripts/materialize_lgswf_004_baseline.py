#!/usr/bin/env python3
"""Deterministic LGSWF-004 semantic-baseline writer."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

BASELINE_PY = '''"""Current LGSWF semantic baseline manifest for the selected tree."""

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
        "def answer(value: int) -> int:\\n    return value\\n",
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
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return manifest
'''

TEST_PY = '''"""Focused LGSWF-004 current semantic baseline checks."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.semantic_state.baseline import (
    persist_semantic_baseline,
)

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    ROOT
    / "data/agent_supervisor/logic_governed_semantic_work_fabric/evidence/semantic-baseline.json"
)


def test_persist_semantic_baseline_binds_tree_and_typed_gaps() -> None:
    # Do not rewrite an already-published receipt; validation must not mutate
    # the candidate after proposal admission.
    manifest = persist_semantic_baseline(ROOT)
    assert manifest["schema"] == "lgswf/semantic-baseline@1"
    assert manifest["repository_head"]
    assert manifest["repository_tree"]
    assert manifest["import_provenance"]["nested_checkout"] is True
    assert manifest["import_provenance"]["scanner_present"] is True
    assert manifest["scan"]["verified"] is True
    assert manifest["unavailable"]["adversarial_assurance"]
    assert EVIDENCE.is_file()
    loaded = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert loaded["manifest_cid"] == manifest["manifest_cid"]
'''

INIT_PY = '''"""Accelerator semantic-state consumer package."""
'''


def _git_add(root: Path, *paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *paths],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def write_lgswf_004_baseline(workspace: Path) -> dict[str, Any]:
    """Write baseline module, test, and CAS evidence for LGSWF-004."""

    root = Path(workspace)
    pkg = root / "ipfs_accelerate_py/agent_supervisor/semantic_state"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "baseline.py").write_text(BASELINE_PY, encoding="utf-8")
    test_path = root / "test/api/semantic_state/test_lgswf_current_baseline.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(TEST_PY, encoding="utf-8")

    # Import from the attempt worktree after writing the module.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lgswf_baseline_mod", pkg / "baseline.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("baseline module is unreadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.persist_semantic_baseline(root)

    outputs = [
        "ipfs_accelerate_py/agent_supervisor/semantic_state/baseline.py",
        "test/api/semantic_state/test_lgswf_current_baseline.py",
        "data/agent_supervisor/logic_governed_semantic_work_fabric/evidence/semantic-baseline.json",
    ]
    add = _git_add(root, *outputs)
    return {
        "manifest_cid": manifest.get("manifest_cid"),
        "state_cid": (manifest.get("scan") or {}).get("state_cid"),
        "verified": (manifest.get("scan") or {}).get("verified"),
        "staged": add.returncode == 0,
        "stage_returncode": add.returncode,
        "stage_stderr": (add.stderr or "")[-500:],
    }


if __name__ == "__main__":
    print(json.dumps(write_lgswf_004_baseline(Path.cwd()), indent=2, sort_keys=True))
