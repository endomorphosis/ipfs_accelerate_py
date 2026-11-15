"""Tests for analysis-driven parallel lane planning."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.planning.parallel_lane_planner import (
    parse_markdown_todo_tasks,
    plan_parallel_lanes,
    plan_parallel_lanes_from_todo_path,
)


SAMPLE_BOARD = """# Sample

## TASK-A Own module a

- Status: pending
- Is schedulable: true
- Goal id: G1
- Outputs: src/a.py, tests/test_a.py
- Predicted files: src/a.py, tests/test_a.py
- Depends on:
- Allow concurrent with: TASK-B
- Bundle: demo/a
- Parallel lane: demo-a
- Conflict policy: Own a only.

## TASK-B Own module b

- Status: pending
- Is schedulable: true
- Goal id: G1
- Outputs: src/b.py, tests/test_b.py
- Predicted files: src/b.py, tests/test_b.py
- Depends on:
- Allow concurrent with: TASK-A
- Bundle: demo/b
- Parallel lane: demo-b
- Conflict policy: Own b only.

## TASK-C Conflicts with A

- Status: pending
- Is schedulable: true
- Goal id: G1
- Outputs: src/a.py, tests/test_c.py
- Predicted files: src/a.py
- Depends on: TASK-A
- Bundle: demo/c
- Parallel lane: demo-c
- Conflict policy: Shares a.py with A.

## TASK-DONE Already finished

- Status: completed
- Is schedulable: true
- Goal id: G1
- Outputs: src/done.py
- Predicted files: src/done.py
"""


def test_parse_skips_completed_and_extracts_fields() -> None:
    tasks = parse_markdown_todo_tasks(SAMPLE_BOARD)
    ids = {t["task_id"] for t in tasks}
    assert ids == {"TASK-A", "TASK-B", "TASK-C"}
    by_id = {t["task_id"]: t for t in tasks}
    assert "src/a.py" in by_id["TASK-A"]["predicted_files"]
    assert by_id["TASK-C"]["depends_on"] == ["TASK-A"]


def test_conflicting_predicted_files_never_share_lane(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    tasks = parse_markdown_todo_tasks(SAMPLE_BOARD)
    plan = plan_parallel_lanes(tasks, max_lanes=4, repo_root=tmp_path, use_ast=True)
    assert plan.task_count == 3
    # A and C both touch src/a.py → different lanes
    assert plan.task_colors["TASK-A"] != plan.task_colors["TASK-C"]
    # A and B are disjoint and allow concurrent → may share or not; must not conflict
    lane_of = plan.task_colors
    # Reconstruct: no lane may contain both of a blocking pair A-C
    for lane_ids in plan.lane_slices.values():
        assert not ({"TASK-A", "TASK-C"} <= set(lane_ids))
    assert plan.blocking_edge_count >= 1
    assert plan.independent_width >= 1


def test_plan_from_todo_path(tmp_path: Path) -> None:
    board = tmp_path / "board.todo.md"
    board.write_text(SAMPLE_BOARD, encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("y=2\n", encoding="utf-8")
    plan = plan_parallel_lanes_from_todo_path(
        board, max_lanes=3, repo_root=tmp_path, use_ast=False
    )
    assert plan.task_count == 3
    assert sum(len(v) for v in plan.lane_slices.values()) == 3
    payload = plan.to_dict()
    assert payload["schema"].startswith("ipfs_accelerate_py/agent-supervisor/parallel-lane-plan")
    assert "lane_slices" in payload
