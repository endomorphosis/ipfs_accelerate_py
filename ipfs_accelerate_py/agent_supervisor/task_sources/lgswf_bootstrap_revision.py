"""Accept LGSWF Plan Revision R2 through PlanRevisionStore CAS."""

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
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
