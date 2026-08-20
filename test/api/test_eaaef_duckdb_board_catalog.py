"""The full EAAEF board is cataloged in embedded DuckDB without Plan R2 launch."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
CURSOR = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "generation-cursor.json"
)
CATALOG = CAMPAIGN / "receipts" / "host_admission" / "held_board_catalog.json"


def _control_db() -> Path:
    generation = "eaaef-run-v14"
    if CURSOR.is_file():
        generation = str(
            json.loads(CURSOR.read_text(encoding="utf-8")).get("active_generation")
            or generation
        )
    number = generation.rsplit("-v", 1)[-1]
    return (
        ROOT
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / f"run-v{number}"
        / "control.duckdb"
    )


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
    control = _control_db()
    with DatabaseTaskSource(control, install_schema=False) as source:
        tasks = source.list_tasks(limit=1000).tasks
        aliases = {item.task_alias for item in tasks}
        expected = {task["stable_task_id"] for task in board["tasks"]}
        assert aliases == expected
        assert len(tasks) == 116
        held = [
            item
            for item in tasks
            if item.task_alias.startswith("EAAEF-")
            and int(item.task_alias.split("-")[-1]) not in set(range(0, 10)) | set(range(180, 192))
        ]
        assert held
        assert all(item.status == "blocked" for item in held)
        ready = [item.task_alias for item in source.ready_tasks(limit=1000).tasks]
        assert not any(
            alias not in {f"EAAEF-{n:03d}" for n in range(10)}
            and alias not in {f"EAAEF-{n}" for n in range(180, 192)}
            for alias in ready
        )
