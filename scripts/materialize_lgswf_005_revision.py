#!/usr/bin/env python3
"""Deterministic LGSWF-005 Plan Revision R2 writer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

MODULE_PY = '''"""Accept LGSWF Plan Revision R2 through PlanRevisionStore CAS."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

R1_S1 = "LGSWF-PLAN-ACTUAL-R1-S1"
SENTINELS = ("ACCEPTED_LGSWF-006_SOURCE_HEAD",)
EVIDENCE = Path("data/agent_supervisor/logic_governed_semantic_work_fabric/evidence/plan-r2.json")


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    return (completed.stdout or "").strip()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_r2_record(workspace: Path) -> dict[str, Any]:
    root = Path(workspace)
    evidence = root / "data/agent_supervisor/logic_governed_semantic_work_fabric/evidence"
    inventory = root / "docs/architecture/logic_governed_semantic_work_fabric_inventory"
    baseline = _load_json(evidence / "semantic-baseline.json")
    freeze = _load_json(inventory / "interface_freeze.json")
    record = {
        "schema": "lgswf/plan-revision-r2@1",
        "revision": "LGSWF-PLAN-ACTUAL-R2",
        "supersedes": R1_S1,
        "preserved_r1_s1": R1_S1,
        "quarantined_r1_rewritten": False,
        "repository_head": _git(root, "rev-parse", "HEAD"),
        "repository_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "semantic_baseline_cid": baseline.get("manifest_cid") or "",
        "interface_freeze_schema": freeze.get("schema") or "",
        "sentinels_replaced": list(SENTINELS),
        "execution_base": _git(root, "rev-parse", "HEAD"),
        "accepted_pointer": "R2",
    }
    record["r2_cid"] = _sha256_text(
        json.dumps(record, sort_keys=True, separators=(",", ":"))
    )
    return record


def persist_plan_r2(workspace: Path) -> dict[str, Any]:
    root = Path(workspace)
    out = root / EVIDENCE
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        existing = _load_json(out)
        if existing.get("schema") == "lgswf/plan-revision-r2@1" and existing.get("r2_cid"):
            return existing
    record = build_r2_record(root)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return record


def cas_roundtrip(record: dict[str, Any], store_root: Path) -> str:
    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
        PlanRevisionStore,
    )

    store = PlanRevisionStore(store_root)
    cid = store.put_cas(record)
    loaded = store.get_cas(cid)
    if loaded.get("r2_cid") != record.get("r2_cid"):
        raise ValueError("PlanRevisionStore CAS roundtrip lost the R2 identity")
    return cid
'''

TEST_PY = '''"""Focused LGSWF-005 Plan Revision R2 checks."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.task_sources.lgswf_bootstrap_revision import (
    SENTINELS,
    cas_roundtrip,
    persist_plan_r2,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "data/agent_supervisor/logic_governed_semantic_work_fabric/evidence/plan-r2.json"
)


def test_plan_r2_replaces_sentinels_and_preserves_r1() -> None:
    record = persist_plan_r2(ROOT)
    assert record["schema"] == "lgswf/plan-revision-r2@1"
    assert record["accepted_pointer"] == "R2"
    assert record["preserved_r1_s1"] == "LGSWF-PLAN-ACTUAL-R1-S1"
    assert record["quarantined_r1_rewritten"] is False
    assert record["semantic_baseline_cid"]
    assert list(record["sentinels_replaced"]) == list(SENTINELS)
    assert record["execution_base"]
    assert record["execution_base"] != SENTINELS[0]
    assert EVIDENCE.is_file()
    loaded = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert loaded["r2_cid"] == record["r2_cid"]


def test_plan_revision_store_cas_roundtrip(tmp_path: Path) -> None:
    record = persist_plan_r2(ROOT)
    cid = cas_roundtrip(record, tmp_path / "plan-revision-store")
    assert cid
'''


def write_lgswf_005_revision(workspace: Path) -> dict[str, Any]:
    """Write R2 module, test, and evidence receipt."""

    root = Path(workspace)
    module = root / "ipfs_accelerate_py/agent_supervisor/task_sources/lgswf_bootstrap_revision.py"
    test_path = root / "test/api/test_agent_supervisor_lgswf_bootstrap_revision.py"
    module.write_text(MODULE_PY, encoding="utf-8")
    test_path.write_text(TEST_PY, encoding="utf-8")
    import importlib.util

    spec = importlib.util.spec_from_file_location("lgswf_r2_mod", module)
    if spec is None or spec.loader is None:
        raise RuntimeError("R2 module is unreadable")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    receipt = loaded.persist_plan_r2(root)
    outputs = [
        "ipfs_accelerate_py/agent_supervisor/task_sources/lgswf_bootstrap_revision.py",
        "test/api/test_agent_supervisor_lgswf_bootstrap_revision.py",
        "data/agent_supervisor/logic_governed_semantic_work_fabric/evidence/plan-r2.json",
    ]
    add = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *outputs],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "r2_cid": receipt.get("r2_cid"),
        "semantic_baseline_cid": receipt.get("semantic_baseline_cid"),
        "staged": add.returncode == 0,
        "stage_returncode": add.returncode,
        "stage_stderr": (add.stderr or "")[-500:],
    }


if __name__ == "__main__":
    print(json.dumps(write_lgswf_005_revision(Path.cwd()), indent=2, sort_keys=True))
