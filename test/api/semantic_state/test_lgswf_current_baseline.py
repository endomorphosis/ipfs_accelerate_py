"""Focused LGSWF-004 current semantic baseline checks."""

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
