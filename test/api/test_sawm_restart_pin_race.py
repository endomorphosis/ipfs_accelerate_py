"""M70 restart checks must not mix task heads from different native snapshots."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)

from ipfs_accelerate_py.agent_supervisor.task_sources.restart_check_pin import (
    RestartCheckPinChanged,
)


@pytest.fixture
def native_m70(tmp_path, monkeypatch):
    # This isolated owner store must not discover a live campaign endpoint.
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_PREFER", "0")
    path = tmp_path / "control.duckdb"
    population = {"plan_root_cid": "plan:a", "taskboard": []}
    with open_intent_repository(path) as repo:
        for index in range(29):
            repo.upsert_goal(
                goal_cid=f"goal:{index}", goal_alias=f"G{index}", title="Goal"
            )
        repo.upsert_plan(plan_cid="plan:a", goal_cid="goal:0", plan_alias="P")
        for index in range(45):
            # Three predecessors give 129 edges; seven fourth predecessors
            # make the actual M70 dependency population of 136.
            count = 4 if 4 <= index <= 10 else min(index, 3)
            repo.upsert_task(
                task_cid=f"task:{index}",
                task_alias=f"A{index}",
                goal_cid=f"goal:{index % 29}",
                plan_cid="plan:a",
                dependencies=[
                    f"task:{previous}" for previous in range(index - count, index)
                ],
            )
            population["taskboard"].append(
                {
                    "task_id": f"A{index}",
                    "task_cid": f"task:{index}",
                    "goal_cid": f"goal:{index % 29}",
                }
            )
    source = DatabaseTaskSource(path, install_schema=False, plan_root_cid="plan:a")
    snapshot = source.snapshot()
    assert (
        snapshot.task_count,
        snapshot.goal_count,
        snapshot.dependency_count,
        snapshot.plan_count,
    ) == (45, 29, 136, 1)
    heads = {}
    for task in population["taskboard"]:
        row = source.intent.get_task(task["task_cid"])
        heads[task["task_id"]] = {
            key: row[key]
            for key in (
                "task_cid",
                "task_alias",
                "goal_cid",
                "status",
                "revision",
            )
        }
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "sawm_restart_race_test",
        root / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_M70_TARGET_EVENT_WATERMARK", snapshot.event_cursor)
    materializer = SimpleNamespace(
        _validated_m70_live_preflight_contract=lambda _: {
            "bind_store_report_to_live_event_digest": True
        },
        _M70_TARGET_PLAN_REVISION=1,
        MigrationRequired=module.OperatorError,
    )
    try:
        yield source, module, population, {"expected_task_heads": heads}, materializer
    finally:
        source.close()


def verify(case, expected_projection_cid):
    source, module, population, authority, materializer = case
    return module._verify_m70_live_head_task_projection(
        source,
        population,
        materializer,
        authority=authority,
        expected_projection_cid=expected_projection_cid,
    )


def test_progress_between_outer_snapshot_and_inner_check_requires_retry(native_m70):
    source, _, _, _, _ = native_m70
    source.compare_and_set_status("A0", 1, "in_progress")
    outer = source.snapshot()
    source.compare_and_set_status("A1", 1, "in_progress")
    current = source.snapshot()
    assert current.event_cursor == outer.event_cursor + 1
    assert current.projection_cid != outer.projection_cid
    with pytest.raises(RestartCheckPinChanged, match="M70 .*?(projection|observation)"):
        verify(native_m70, outer.projection_cid)


def test_stable_progress_uses_current_heads_and_preserves_anchor(native_m70):
    source, _, _, authority, _ = native_m70
    source.compare_and_set_status("A0", 1, "in_progress")
    outer = source.snapshot()
    statuses, revisions, _ = verify(native_m70, outer.projection_cid)
    assert statuses["A0"] == "in_progress" and revisions["A0"] == 2
    assert authority["expected_task_heads"]["A0"]["status"] == "ready"
    assert authority["expected_task_heads"]["A0"]["revision"] == 1
    assert source.snapshot().to_dict() == outer.to_dict()


def test_progress_after_head_reads_is_rejected_by_final_snapshot(
    native_m70, monkeypatch
):
    source, _, _, _, _ = native_m70
    source.compare_and_set_status("A0", 1, "in_progress")
    outer = source.snapshot()
    original_snapshot = source.snapshot
    calls = 0

    def changed_snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:
            source.compare_and_set_status("A1", 1, "in_progress")
        return original_snapshot()

    monkeypatch.setattr(source, "snapshot", changed_snapshot)
    with pytest.raises(
        RestartCheckPinChanged, match="restart observation changed during verification"
    ):
        verify(native_m70, outer.projection_cid)
    assert calls == 2
