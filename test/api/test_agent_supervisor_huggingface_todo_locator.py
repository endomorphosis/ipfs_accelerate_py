from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.huggingface_todo_locator import (
    LOCATOR_SCHEMA,
    POINTER_SCHEMA,
    HuggingFaceTodoLocatorError,
    load_huggingface_todo_locator,
    locate_and_schedule_huggingface_todos,
    select_within_budget,
    time_management_from_records,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.persistent_task_queue import (
    PersistentTaskQueue,
)


BOARD = """# U.S. Code autoformal discrepancy board

## AFTD-001-deadbeef Preserve or replace dropped_clause for usc:us:5:552.span-1
- Status: todo
- Priority: P1
- Track: autoformal-discrepancy
- Goal id: AFTD-G010
- Board namespace: uscode-autoformal-todo-v1
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 8000
- Estimated validation seconds: 300
- Preserve: compiled clauses that already round-trip
- Replace: decompiler reconstruction for dropped clauses
- Failure mode: dropped_clause
- Match is not admit: true
- Formalized: false
- Admitted: false
- Acceptance: Do not mark formalized.

## AFTD-002-cafebabe Preserve or replace no_parser_elements for usc:us:18:1001.span-1
- Status: todo
- Priority: P0
- Track: autoformal-discrepancy
- Goal id: AFTD-G010
- Board namespace: uscode-autoformal-todo-v1
- Resource class: cpu-medium
- Token class: large
- Estimated tokens: 12000
- Estimated validation seconds: 600
- Preserve: compiled compiler rows
- Replace: parser atom inventory
- Failure mode: no_parser_elements
- Match is not admit: true
- Formalized: false
- Admitted: false
- Acceptance: Do not mark formalized.
"""


def _locator() -> dict:
    return {
        "schema": LOCATOR_SCHEMA,
        "dataset_repo_id": "justicedao/uscode-autoformal-todos",
        "revision": "main",
        "release_id": "rel-1",
        "board_path": "data/autoformal_todo/rel-1/board.md",
        "todos_path": "data/autoformal_todo/rel-1/todos.parquet",
        "board_namespace": "uscode-autoformal-todo-v1",
        "task_header_prefix": "## AFTD-",
        "task_prefix": "AFTD-",
        "goal_prefix": "AFTD-G",
        "time_management": {
            "task_count": 2,
            "total_estimated_tokens": 20000,
            "total_estimated_validation_seconds": 900,
        },
    }


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "hf-package"
    root.mkdir()
    (root / "board.md").write_text(BOARD, encoding="utf-8")
    (root / "locator.json").write_text(json.dumps(_locator()), encoding="utf-8")
    return root


def test_load_locator_rejects_jsonl_schema_and_missing_fields(tmp_path: Path) -> None:
    with pytest.raises(HuggingFaceTodoLocatorError, match="unsupported locator schema"):
        load_huggingface_todo_locator({"schema": "todos.jsonl"})
    payload = _locator()
    payload.pop("board_path")
    with pytest.raises(HuggingFaceTodoLocatorError, match="board_path"):
        load_huggingface_todo_locator(payload)
    package = _package(tmp_path)
    loaded = load_huggingface_todo_locator(package)
    assert loaded["dataset_repo_id"] == "justicedao/uscode-autoformal-todos"
    assert loaded["schema"] == LOCATOR_SCHEMA


def test_locate_and_schedule_indexes_time_management_without_jsonl(tmp_path: Path) -> None:
    package = _package(tmp_path)
    queue = PersistentTaskQueue()
    receipt = locate_and_schedule_huggingface_todos(
        package / "locator.json",
        repo_root=tmp_path,
        board_destination=tmp_path / "board.md",
        package_root=package,
        queue=queue,
        max_tokens=15000,
    )
    assert receipt["jsonl_written"] is False
    assert receipt["wrote_compiler"] is False
    assert receipt["task_count"] == 2
    assert receipt["time_management"]["total_estimated_tokens"] == 20000
    assert receipt["time_management"]["total_estimated_validation_seconds"] == 900
    assert receipt["scheduled_task_ids"] == ["AFTD-002-cafebabe"]
    assert receipt["enqueued_task_ids"] == ["AFTD-002-cafebabe"]
    assert any(entry.priority == "P0" for entry in queue.entries.values())
    assert not list(tmp_path.glob("*.jsonl"))


def test_release_pointer_resolves_local_package_for_scheduling(tmp_path: Path) -> None:
    package = _package(tmp_path)
    pointer = {
        "schema": POINTER_SCHEMA,
        "dataset_repo_id": "justicedao/uscode-autoformal-todos",
        "release_id": "rel-1",
        "revision": "main",
        "package_root": str(package),
        "locator_path": str(package / "locator.json"),
        "board_path": str(package / "board.md"),
        "todos_path": str(package / "todos.parquet"),
        "jsonl_written": False,
        "time_management": {"task_count": 2, "total_estimated_tokens": 20000},
    }
    pointer_path = tmp_path / "autoformal-todo-release-pointer.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    loaded = load_huggingface_todo_locator(pointer_path)
    assert loaded["schema"] == LOCATOR_SCHEMA
    receipt = locate_and_schedule_huggingface_todos(
        pointer_path,
        repo_root=tmp_path,
        board_destination=tmp_path / "from-pointer.md",
        queue=PersistentTaskQueue(),
    )
    assert receipt["task_count"] == 2
    assert receipt["jsonl_written"] is False
    assert (tmp_path / "from-pointer.md").is_file()


def test_locate_and_schedule_rejects_jsonl_board(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "board.md").write_text(BOARD + "\nSee todos.jsonl\n", encoding="utf-8")
    with pytest.raises(HuggingFaceTodoLocatorError, match="jsonl"):
        locate_and_schedule_huggingface_todos(
            package / "locator.json",
            repo_root=tmp_path,
            board_destination=tmp_path / "bad.md",
            package_root=package,
        )


def test_fetch_locator_uses_huggingface_paths_not_jsonl(tmp_path: Path) -> None:
    seen = {}

    def fetch(repo_id, revision, relative_path):
        seen["repo_id"] = repo_id
        seen["revision"] = revision
        seen["relative_path"] = relative_path
        return BOARD.encode("utf-8")

    receipt = locate_and_schedule_huggingface_todos(
        _locator(),
        repo_root=tmp_path,
        board_destination=tmp_path / "fetched.md",
        fetch=fetch,
    )
    assert seen["repo_id"] == "justicedao/uscode-autoformal-todos"
    assert seen["relative_path"].endswith("board.md")
    assert not seen["relative_path"].endswith(".jsonl")
    assert receipt["time_management"]["task_count"] == 2
    budgeted = select_within_budget(
        [],
        max_tokens=1,
    )
    assert budgeted == []
    records_budget = time_management_from_records([])
    assert records_budget["total_estimated_tokens"] == 0
