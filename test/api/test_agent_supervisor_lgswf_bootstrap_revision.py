"""Focused LGSWF-005 Plan Revision R2 checks."""

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
