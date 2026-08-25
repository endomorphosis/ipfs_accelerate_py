"""The full EAAEF board is cataloged in embedded DuckDB without Plan R2 launch."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
CATALOG = CAMPAIGN / "receipts" / "host_admission" / "held_board_catalog.json"


def test_held_board_catalog_covers_the_full_board_without_live_launch() -> None:
    board = json.loads((CAMPAIGN / "task_board.json").read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert catalog["future_tasks_materialized"] is False
    assert catalog["plan_r2_applied"] is False
    assert catalog["configured_board_launch"] is False
    assert catalog["live_launch_allowed"] is False
    assert catalog["process_started"] is False
    assert catalog["task_count_after"] == len(board["tasks"])
    assert catalog["inserted_count"] == 94

    # This is a durable catalog-receipt test, not an operational database
    # health check.  The control DB and generation cursor are intentionally
    # ignored runtime state, so opening their historical paths would either
    # couple the test to one host or create an empty DuckDB in a clean clone.
    board_ids = {task["stable_task_id"] for task in board["tasks"]}
    held_ids = {
        task["stable_task_id"]
        for task in board["tasks"]
        if task["status"] == "blocked"
    }
    existing_ids = board_ids - held_ids
    inserted_ids = set(catalog["inserted_task_ids"])
    skipped_ids = set(catalog["skipped_existing_task_ids"])

    assert inserted_ids == held_ids
    assert skipped_ids == existing_ids
    assert inserted_ids.isdisjoint(skipped_ids)
    assert inserted_ids | skipped_ids == board_ids
    assert catalog["inserted_count"] == len(inserted_ids)
    assert catalog["materialize_task_count"] == len(inserted_ids)
    assert sum(catalog["status_counts"].values()) == len(board_ids)
    assert catalog["status_counts"]["blocked"] == len(held_ids)
