"""Exact-tree prerequisite inventory contract tests for PCPC-000/001."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "docs/architecture/procedure_compiler_inventory"
START_COMMIT = "bbf7f68799072c2b81f7d96eac91f2df3c4b3952"
START_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
ALLOWED = {"available", "available_with_caveats", "incompatible", "stale", "missing"}


def _object(name: str) -> dict[str, object]:
    value = json.loads((INVENTORY / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_baseline_binds_exact_committed_tree_and_releases() -> None:
    baseline = _object("baseline.json")
    repository = baseline["repository"]
    assert isinstance(repository, dict)
    assert repository["commit"] == START_COMMIT
    assert repository["tree"] == START_TREE
    assert repository["origin_main_commit"] == START_COMMIT
    assert baseline["package"] == {"name": "ipfs_accelerate_py", "version": "0.0.45"}
    assert baseline["sibling_gitlinks"] == {
        "ipfs_accelerate_py/mcplusplus": "5ac0ab162f420264fd224073a5df3f2d7c054ae3",
        "ipfs_datasets_py": "480a1666f144ad606fcb3cacb66e59775f28d0d1",
        "ipfs_kit_py": "2564aea1ae35061f2165872aff91e8a40801ab7e",
    }
    assert baseline["excluded_sources"] == [
        "planning_document_status",
        "receipt_shaped_unadmitted_json",
        "task_board_status",
        "uncommitted_working_tree_overlays",
    ]
    assert baseline["test_evidence"]


def test_prerequisites_have_one_closed_honest_disposition() -> None:
    inventory = _object("prerequisites.json")
    rows = inventory["dispositions"]
    assert isinstance(rows, list)
    by_name = {str(row["authority"]): row for row in rows}
    assert len(by_name) == len(rows)
    assert {str(row["status"]) for row in rows} <= ALLOWED
    assert inventory["baseline_commit"] == START_COMMIT
    assert inventory["baseline_tree"] == START_TREE
    assert by_name["AdaptivePlanner"]["status"] == "incompatible"
    assert (
        by_name["AdaptivePlanner"]["blocker"]
        == "adaptive_planner_import_missing_committed_mcp_contract_catalog"
    )
    assert by_name["AutonomousMetaController"]["status"] == "missing"
    assert by_name["AdversarialAssuranceEngine"]["status"] == "available_with_caveats"
    assert by_name["IncrementalProofSealer"]["status"] == "available_with_caveats"


def test_inventory_does_not_claim_runtime_completion() -> None:
    for name in ("baseline.json", "prerequisites.json"):
        text = (INVENTORY / name).read_text(encoding="utf-8").lower()
        assert "production ready" not in text
        assert "board complete" not in text
        assert "all tasks done" not in text
