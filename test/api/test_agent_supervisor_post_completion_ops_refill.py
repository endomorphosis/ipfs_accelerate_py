"""Tests for deterministic post-completion ops refill."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.post_completion_ops_refill import (
    POST_COMPLETION_OPS_CATALOG_SCHEMA,
    expand_lane_slices_for_catalog,
    load_post_completion_ops_catalog,
    render_post_completion_task_block,
    seed_post_completion_ops,
    task_shard_index,
)


def _write_catalog(path: Path) -> None:
    payload = {
        "schema": POST_COMPLETION_OPS_CATALOG_SCHEMA,
        "program": "demo-program-v1",
        "task_prefix": "DEMO-",
        "trigger": "board_drained",
        "tasks": [
            {
                "task_id": "DEMO-165",
                "title": "Validate completion gate offline",
                "depends_on": ["DEMO-164"],
                "outputs": ["scripts/ops/validate.py"],
                "validation": "python scripts/ops/validate.py --offline",
                "acceptance": "Offline gate passes or lists explicit gaps.",
            },
            {
                "task_id": "DEMO-166",
                "title": "Assemble PR package without push",
                "depends_on": ["DEMO-165"],
                "outputs": ["docs/ops/pr_package.md"],
                "validation": "test -f docs/ops/pr_package.md",
                "acceptance": "PR package exists and no push occurs.",
            },
            {
                "task_id": "DEMO-167",
                "title": "Live canary with offline fallback",
                "depends_on": ["DEMO-165"],
                "outputs": ["scripts/ops/canary.py"],
                "validation": "python -m pytest tests/test_canary.py -q",
                "acceptance": "Canary defaults offline.",
            },
            {
                "task_id": "DEMO-168",
                "title": "Hub dry-run staging",
                "depends_on": ["DEMO-165"],
                "outputs": ["scripts/ops/hub_dry_run.py"],
                "validation": "python -m pytest tests/test_hub_dry_run.py -q",
                "acceptance": "Dry-run only.",
            },
            {
                "task_id": "DEMO-169",
                "title": "Operator handoff receipt",
                "depends_on": ["DEMO-166", "DEMO-167", "DEMO-168"],
                "outputs": ["docs/ops/handoff.md"],
                "validation": "test -f docs/ops/handoff.md",
                "acceptance": "Handoff receipt binds prior ops results.",
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _drained_board(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Demo board",
                "",
                "## DEMO-164 Completion gate",
                "",
                "- Status: completed",
                "- Is schedulable: true",
                "- Depends on:",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_catalog_and_shard_assignment(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    catalog = load_post_completion_ops_catalog(catalog_path)
    assert catalog.task_prefix == "DEMO-"
    assert len(catalog.tasks) == 5
    assert task_shard_index(catalog.tasks[0], shard_count=4) == 165 % 4
    block = render_post_completion_task_block(catalog.tasks[0], board_namespace="demo")
    assert block.startswith("## DEMO-165 ")
    assert "- Status: pending" in block
    slices = expand_lane_slices_for_catalog({}, catalog, shard_count=4)
    assert "DEMO-165" in slices[str(165 % 4)]
    assert "DEMO-168" in slices[str(168 % 4)]


def test_seed_on_drained_board_is_idempotent(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    todo_path = tmp_path / "todo.md"
    strategy_path = tmp_path / "strategy.json"
    config_path = tmp_path / "config.json"
    _write_catalog(catalog_path)
    _drained_board(todo_path)
    strategy_path.write_text("{}\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "shard_count": 4,
                "lane_slices": {
                    "0": ["DEMO-164"],
                    "1": [],
                    "2": [],
                    "3": [],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    catalog = load_post_completion_ops_catalog(catalog_path)

    first = seed_post_completion_ops(
        todo_path=todo_path,
        strategy_path=strategy_path,
        catalog=catalog,
        config_path=config_path,
        board_namespace="demo-program-v1",
        shard_count=4,
        shard_index=1,
    )
    assert first.reason == "seeded"
    assert set(first.seeded_task_ids) == {
        "DEMO-165",
        "DEMO-166",
        "DEMO-167",
        "DEMO-168",
        "DEMO-169",
    }
    assert first.board_updated is True
    assert first.config_updated is True
    assert "DEMO-165" in first.expanded_execution_slice_task_ids

    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## DEMO-165 " in todo_text
    assert "## DEMO-169 " in todo_text
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "DEMO-165" in config["lane_slices"][str(165 % 4)]
    assert "DEMO-164" in config["lane_slices"]["0"]

    second = seed_post_completion_ops(
        todo_path=todo_path,
        strategy_path=strategy_path,
        catalog=catalog,
        config_path=config_path,
        board_namespace="demo-program-v1",
        shard_count=4,
        shard_index=1,
    )
    assert second.reason == "already_seeded"
    assert second.seeded_task_ids == []
    assert second.board_updated is False


def test_seed_skips_when_board_not_drained(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    todo_path = tmp_path / "todo.md"
    strategy_path = tmp_path / "strategy.json"
    _write_catalog(catalog_path)
    todo_path.write_text(
        "\n".join(
            [
                "## DEMO-164 Completion gate",
                "",
                "- Status: pending",
                "- Is schedulable: true",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    strategy_path.write_text("{}\n", encoding="utf-8")
    catalog = load_post_completion_ops_catalog(catalog_path)
    result = seed_post_completion_ops(
        todo_path=todo_path,
        strategy_path=strategy_path,
        catalog=catalog,
        shard_count=4,
        shard_index=0,
    )
    assert result.reason == "board_not_drained"
    assert result.seeded_task_ids == []
    assert "## DEMO-165 " not in todo_path.read_text(encoding="utf-8")
