from dataclasses import replace
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import bundle_supervisor
from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import (
    BundleLaneSpec,
    DynamicBundleScheduler,
)
from ipfs_accelerate_py.agent_supervisor.lease_coordination import TaskLeaseState


def _lane(repo: Path, shard: Path) -> BundleLaneSpec:
    return BundleLaneSpec(
        bundle_key="objective/test/cache",
        parallel_lane="objective/test/cache",
        todo_path=shard,
        state_dir=repo / "state" / "cache",
        worktree_root=repo / "worktrees" / "cache",
        state_prefix="cache",
        task_ids=["T-1"],
        conflict_policy="",
        command=[],
        log_path=repo / "logs" / "cache.log",
    )


def test_scheduler_reuses_plan_until_an_index_or_shard_revision_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    shard = repo / "bundle.todo.md"
    repo.mkdir()
    index.write_text("{}", encoding="utf-8")
    shard.write_text("## T-1 cached\n", encoding="utf-8")
    calls = 0

    def fake_plan_bundle_lanes(**_kwargs):
        nonlocal calls
        calls += 1
        return [_lane(repo, shard)]

    monkeypatch.setattr(bundle_supervisor, "plan_bundle_lanes", fake_plan_bundle_lanes)
    scheduler = DynamicBundleScheduler(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        max_lanes=1,
    )

    assert scheduler._plan()[0].task_ids == ["T-1"]
    assert scheduler._plan()[0].task_ids == ["T-1"]
    assert calls == 1

    shard.write_text("## T-1 changed\n", encoding="utf-8")
    scheduler._plan()
    assert calls == 2

    index.write_text('{"revision": 2}', encoding="utf-8")
    scheduler._plan()
    assert calls == 3


def test_external_worker_heartbeat_reapplies_fence_without_rebuilding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    shard = repo / "bundle.todo.md"
    external_state = repo / "serial-state.json"
    repo.mkdir()
    index.write_text("{}", encoding="utf-8")
    shard.write_text("## T-1 cached\n", encoding="utf-8")
    external_state.write_text(
        '{"active_task_id":"T-1","active_phase":"implementing"}',
        encoding="utf-8",
    )
    calls = 0

    def fake_plan_bundle_lanes(**_kwargs):
        nonlocal calls
        calls += 1
        return [
            replace(
                _lane(repo, shard),
                queue_payload={"tasks": [{"task_id": "T-1"}]},
                claimable=True,
            )
        ]

    monkeypatch.setattr(bundle_supervisor, "plan_bundle_lanes", fake_plan_bundle_lanes)
    scheduler = DynamicBundleScheduler(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        external_task_state_paths=(external_state,),
        max_lanes=1,
    )

    assert scheduler._plan()[0].claimable is False
    external_state.write_text(
        '{"active_task_id":"T-1","active_phase":"validating"}',
        encoding="utf-8",
    )
    assert scheduler._plan()[0].claimable is False
    external_state.write_text("{}", encoding="utf-8")
    assert scheduler._plan()[0].claimable is True
    assert calls == 1


def test_hot_path_projections_copy_mutable_roots_without_recursive_graph_copy(
    tmp_path: Path,
) -> None:
    nested_task = {"task_id": "T-1", "planning_graph": {"edges": ["a", "b"]}}
    queue_payload = {"tasks": [nested_task]}
    lane = replace(
        _lane(tmp_path, tmp_path / "bundle.todo.md"),
        queue_payload=queue_payload,
    )

    lane_payload = lane.to_dict()
    assert lane_payload["queue_payload"] == queue_payload
    assert lane_payload["queue_payload"] is not queue_payload
    lane_payload["queue_payload"]["new"] = True
    assert "new" not in queue_payload

    state = TaskLeaseState(
        task_cid="task",
        goal_cid="goal",
        subgoal_cid="subgoal",
        task_id="T-1",
        bundle=queue_payload,
        state="ready",
        lease_state=None,
        claim_cid=None,
        resolution_cid=None,
        claimant_did=None,
        logical_epoch=0,
        fencing_token=0,
        lease_expires_at_ms=None,
        attempt=0,
        max_attempts=3,
        release_reason=None,
        retry_not_before_ms=0,
        registered_at_ms=1,
        updated_at_ms=1,
    )
    state_payload = state.to_dict()
    assert state_payload["bundle"] == queue_payload
    assert state_payload["bundle"] is not queue_payload
    state_payload["bundle"]["new"] = True
    assert "new" not in queue_payload
