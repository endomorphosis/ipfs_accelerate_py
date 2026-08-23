"""Repair, validate, and ingest supervisor taskboards into DuckDB/Quack."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_ingest import (
    TaskboardIngestError,
    compact_task_view,
    ingest_taskboard,
    load_taskboard,
    parse_json_taskboard,
    parse_markdown_field_board,
    repair_malformed_text,
    taskboard_context_view,
    validate_taskboard,
    write_repaired_text,
)


def _board() -> dict[str, object]:
    return {
        "board_namespace": "example-board-v1",
        "plan_revision": "PLAN-R1",
        "goals": [{"goal_id": "G-ROOT", "title": "Root"}],
        "tasks": [
            {
                "stable_task_id": "EX-001",
                "title": "First",
                "status": "todo",
                "subgoal_id": "G-ROOT",
                "objective": "Do the first thing with enough detail to truncate later.",
                "dependencies": [],
                "owned_files": ["a.py", "b.py"],
                "execution_validation": [
                    {"working_directory": ".", "argv": ["python3", "-m", "pytest", "-q"]}
                ],
            },
            {
                "stable_task_id": "EX-002",
                "title": "Second",
                "status": "blocked",
                "subgoal_id": "G-ROOT",
                "objective": "Depends on first",
                "dependencies": ["EX-001"],
                "owned_files": ["c.py"],
            },
        ],
    }


def test_repair_malformed_text_fixes_bom_quotes_and_markdown_spacing() -> None:
    raw = (
        "\ufeff##EX-001 Title\r\n-Status:\u00a0To Do\r\n- Depends on: EX-000\n"
    ).encode("utf-8")
    result = repair_malformed_text(raw, kind="markdown")
    kinds = {item.kind for item in result.actions}
    assert "strip_bom" in kinds
    assert "normalize_newlines" in kinds
    assert "heading_space" in kinds
    assert "bullet_space" in kinds
    assert "normalize_unicode" in kinds
    assert result.text.startswith("## EX-001 Title\n")
    assert "- Status: To Do" in result.text


def test_json_trailing_comma_and_duplicate_keys() -> None:
    repaired = repair_malformed_text(
        '{"tasks":[{"stable_task_id":"A","status":"todo",}],}\n',
        kind="json",
    )
    board = parse_json_taskboard(repaired.text)
    assert board["tasks"][0]["stable_task_id"] == "A"
    with pytest.raises(TaskboardIngestError, match="duplicate JSON key"):
        parse_json_taskboard('{"tasks":[], "tasks":[]}\n')


def test_markdown_field_board_normalizes_status_and_dependencies() -> None:
    text = "## EX-001 First\n- Stable task ID: EX-001\n- Status: In Progress\n- Depends on: EX-000, EX-002\n"
    board = parse_markdown_field_board(text)
    task = board["tasks"][0]
    assert task["status"] == "in_progress"
    assert task["dependencies"] == ["EX-000", "EX-002"]


def test_validate_rejects_unknown_dependency_and_duplicate_ids() -> None:
    board = _board()
    board["tasks"][1]["dependencies"] = ["MISSING"]
    errors = validate_taskboard(board)
    assert any("unknown dependency" in item for item in errors)
    board = _board()
    board["tasks"].append(dict(board["tasks"][0]))
    errors = validate_taskboard(board)
    assert any("duplicate task id" in item for item in errors)


def test_ingest_and_context_view_keep_the_board_out_of_raw_dumps(
    tmp_path: Path,
) -> None:
    board_path = tmp_path / "board.json"
    store = tmp_path / "control.duckdb"
    board_path.write_text(json.dumps(_board()), encoding="utf-8")
    result = ingest_taskboard(
        board_path=board_path,
        store_path=store,
        require_quack=False,
    )
    assert result["task_count"] == 2
    assert result["inserted_count"] == 2
    assert result["configured_board_launch"] is False
    assert result["quack"]["network_install_attempted"] is False
    again = ingest_taskboard(
        board_path=board_path,
        store_path=store,
        require_quack=False,
    )
    assert again["inserted_count"] == 0
    assert again["skipped_existing_count"] == 2
    context = taskboard_context_view(store, ready_only=True, max_bytes=2048)
    assert context["schema"].endswith("taskboard-context-view@1")
    assert context["task_count"] == 2
    assert context["byte_count"] <= 2048
    ids = {item["task_id"] for item in context["selected"]}
    assert "EX-001" in ids
    assert "EX-002" not in ids
    compact = compact_task_view(
        {
            "task_alias": "EX-001",
            "status": "todo",
            "body": _board()["tasks"][0],
        }
    )
    assert compact["owned_file_count"] == 2
    assert "owned_files" not in compact


def test_write_repaired_text_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "board.md"
    dest = tmp_path / "repaired.md"
    source.write_bytes(b"\xef\xbb\xbf##T1\n-Status: DONE\n")
    result = write_repaired_text(dest, source.read_bytes(), kind="markdown")
    assert dest.is_file()
    assert result.changed is True
    loaded = load_taskboard(dest)
    assert loaded["tasks"][0]["status"] == "completed"


def test_load_taskboard_rejects_control_characters(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"stable_task_id": "A\nB", "status": "todo", "dependencies": []}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TaskboardIngestError, match="newline"):
        load_taskboard(path)
