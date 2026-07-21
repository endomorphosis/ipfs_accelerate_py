from __future__ import annotations

import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import (
    launch_bundle_lanes,
    plan_bundle_lanes,
)
from ipfs_accelerate_py.agent_supervisor.lease_coordination import (
    LeaseConflictError,
    LeaseCoordinator,
    LeaseExpiredError,
    LeaseQueueBridge,
    StaleFencingTokenError,
    adapt_goal_bundle,
    profile_g_cid,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    build_bundle_task_payloads,
    submit_bundle_tasks,
)


def _bundle() -> dict[str, object]:
    return {
        "bundle_key": "objective/accelerator/leases",
        "parallel_lane": "accelerator-a",
        "todo_path": "tasks.todo.md",
        "source_todo": "all.todo.md",
        "tasks": [{"task_id": "SVD-085"}],
    }


def test_goal_bundle_adapter_emits_immutable_linked_profile_g_artifacts() -> None:
    adapted = adapt_goal_bundle(_bundle(), created_at_ms=1_783_872_000_000)

    assert adapted["goal"]["schema"] == "mcp++/profile-g/goal@1"
    assert adapted["subgoal"]["goal_cid"] == adapted["goal_cid"]
    assert adapted["selection"]["plan_branch_cid"] == adapted["plan_branch_cid"]
    assert adapted["task"]["subgoal_cid"] == adapted["subgoal_cid"]
    assert adapted["task"]["selection_cid"] == adapted["selection_cid"]
    assert adapted["task"]["tool"] == "codex.todo_bundle"
    for name in ("goal", "subgoal", "plan_branch", "selection", "task"):
        assert profile_g_cid(adapted[name]) in adapted["artifacts"]


def test_regenerated_bundle_keeps_one_canonical_lease_identity(tmp_path: Path) -> None:
    first = adapt_goal_bundle(_bundle(), created_at_ms=1_783_872_000_000)
    second = adapt_goal_bundle(_bundle(), created_at_ms=1_783_872_001_000)

    assert first["task_spec_cid"] != second["task_spec_cid"]
    assert first["canonical_task_cid"] == second["canonical_task_cid"]
    assert first["task"]["canonical_task_cid"] == first["canonical_task_cid"]

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        registered_first = coordinator.register_bundle(_bundle(), created_at_ms=1_783_872_000_000)
        grant = coordinator.claim(
            registered_first["task_cid"],
            "did:web:lane-a.example",
            requested_lease_ms=10_000,
        )
        registered_second = coordinator.register_bundle(_bundle(), created_at_ms=1_783_872_001_000)

        assert registered_first["task_spec_cid"] != registered_second["task_spec_cid"]
        assert registered_first["task_cid"] == registered_second["task_cid"] == grant.task_cid
        assert coordinator.claim(
            registered_second["task_cid"],
            "did:web:lane-a.example",
            requested_lease_ms=10_000,
        ).goal_cid == grant.goal_cid
        with pytest.raises(LeaseConflictError):
            coordinator.claim(
                registered_second["task_cid"],
                "did:web:lane-b.example",
                requested_lease_ms=10_000,
            )


def test_claim_renew_heartbeat_release_and_receipt_are_fenced(tmp_path: Path) -> None:
    now = 1_783_872_000_000
    with LeaseCoordinator(tmp_path / "leases.sqlite3", clock_ms=lambda: now) as coordinator:
        adapted = coordinator.register_bundle(_bundle(), created_at_ms=now)
        grant = coordinator.claim(adapted["task_cid"], "did:web:lane-a.example", requested_lease_ms=10_000)

        with pytest.raises(LeaseConflictError):
            coordinator.claim(adapted["task_cid"], "did:web:lane-b.example", requested_lease_ms=10_000)

        renewed = coordinator.renew(grant, requested_lease_ms=20_000, now_ms=now + 1_000)
        assert renewed.fencing_token == grant.fencing_token == 1
        assert renewed.lease_expires_at_ms == now + 21_000
        heartbeat = coordinator.heartbeat(renewed, capacity_millionths=750_000, now_ms=now + 2_000)
        assert heartbeat["fencing_token"] == 1
        assert heartbeat["capacity_millionths"] == 750_000

        result = coordinator.receipt(
            renewed,
            status="succeeded",
            output={"commit": "abc123"},
            started_at_ms=now,
            now_ms=now + 3_000,
        )
        assert result["goal_cid"] == adapted["goal_cid"]
        assert result["subgoal_cid"] == adapted["subgoal_cid"]
        assert result["receipt"]["claim_cid"] == grant.claim_cid
        assert result["receipt"]["fencing_token"] == 1
        assert coordinator.active_lease(adapted["task_cid"], now_ms=now + 3_001) is None
        with pytest.raises(LeaseExpiredError):
            coordinator.heartbeat(renewed, capacity_millionths=1, now_ms=now + 3_001)


def test_expired_lane_is_recovered_with_higher_epoch_and_token(tmp_path: Path) -> None:
    now = 1_783_872_000_000
    with LeaseCoordinator(tmp_path / "leases.sqlite3", clock_ms=lambda: now) as coordinator:
        adapted = coordinator.register_bundle(_bundle(), created_at_ms=now)
        old = coordinator.claim(adapted["task_cid"], "did:web:lane-a.example", requested_lease_ms=5_000)
        replacement = coordinator.claim(
            adapted["task_cid"],
            "did:web:lane-b.example",
            requested_lease_ms=5_000,
            now_ms=now + 5_001,
        )

        assert replacement.logical_epoch == old.logical_epoch + 1
        assert replacement.fencing_token == old.fencing_token + 1
        with pytest.raises((LeaseExpiredError, StaleFencingTokenError)):
            coordinator.renew(old, requested_lease_ms=5_000, now_ms=now + 5_002)
        with pytest.raises((LeaseExpiredError, StaleFencingTokenError)):
            coordinator.receipt(old, status="failed", failure_class="fenced", now_ms=now + 5_002)
        assert coordinator.active_lease(adapted["task_cid"], now_ms=now + 5_002) == replacement


def test_release_allows_a_fenced_takeover(tmp_path: Path) -> None:
    now = 1_783_872_000_000
    with LeaseCoordinator(tmp_path / "leases.sqlite3", clock_ms=lambda: now) as coordinator:
        adapted = coordinator.register_bundle(_bundle(), created_at_ms=now)
        old = coordinator.claim(adapted["task_cid"], "did:web:lane-a.example", requested_lease_ms=5_000)
        resolution_cid = coordinator.release(old, now_ms=now + 100)
        assert coordinator.get_artifact(resolution_cid)["outcome"] == "released"

        replacement = coordinator.claim(
            adapted["task_cid"], "did:web:lane-b.example", requested_lease_ms=5_000, now_ms=now + 101
        )
        assert replacement.fencing_token == old.fencing_token + 1
        with pytest.raises((LeaseExpiredError, StaleFencingTokenError)):
            coordinator.release(old, now_ms=now + 102)


def test_queue_bridge_carries_goal_task_adapters(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps({"source_todo": "tasks.todo.md", "bundles": {"objective/test": {
            "shard_path": "test.todo.md", "parallel_lane": "test", "tasks": [{"task_id": "T-1"}]
        }}}),
        encoding="utf-8",
    )
    payload = build_bundle_task_payloads(index)[0]
    assert payload["profile_g"]["task"]["subgoal_cid"] == payload["profile_g"]["subgoal_cid"]

    submitted: list[dict[str, object]] = []

    class Queue:
        def submit(self, **kwargs):
            submitted.append(kwargs)
            return "queue-id"

    assert submit_bundle_tasks(index, queue=Queue()) == ["queue-id"]
    assert submitted[0]["payload"]["profile_g"]["goal_cid"]
    assert submitted[0]["payload"]["profile_g"]["task_cid"]


def test_queue_bridge_requires_lease_and_links_terminal_receipt(tmp_path: Path) -> None:
    from types import SimpleNamespace

    payload = _bundle()
    payload["profile_g"] = adapt_goal_bundle(payload, created_at_ms=1_783_872_000_000)

    class Queue:
        released: list[str] = []
        completed: list[dict[str, object]] = []

        def claim_next(self, **_kwargs):
            return SimpleNamespace(task_id="queue-1", payload=payload)

        def release(self, *, reason, **_kwargs):
            self.released.append(reason)
            return True

        def complete(self, **kwargs):
            self.completed.append(kwargs)
            return True

    queue = Queue()
    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        bridge = LeaseQueueBridge(
            queue,
            coordinator,
            worker_id="worker-a",
            claimant_did="did:web:worker-a.example",
            lease_ms=5_000,
        )
        leased = bridge.claim_next(supported_task_types=["codex.todo_bundle"])
        assert leased is not None
        assert coordinator.active_lease(leased.grant.task_cid) == leased.grant
        receipt = bridge.complete(leased, status="succeeded", output={"commit": "abc"})

    assert receipt["goal_cid"] == payload["profile_g"]["goal_cid"]
    assert queue.completed[0]["status"] == "completed"
    assert queue.completed[0]["result"]["profile_g"]["receipt"]["fencing_token"] == 1


def test_bundle_launcher_runs_only_an_accepted_lease(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    index = repo / "index.json"
    index.write_text(json.dumps({"bundles": {"objective/test": {
        "shard_path": "test.todo.md", "parallel_lane": "test", "tasks": [{"task_id": "T-1"}]
    }}}), encoding="utf-8")
    lanes = plan_bundle_lanes(
        bundle_index_path=index, repo_root=repo, state_root=repo / "state",
        worktree_root=repo / "worktrees", log_dir=repo / "logs",
    )
    starts: list[list[str]] = []

    class Process:
        pid = 1234

    def fake_popen(command, **_kwargs):
        starts.append(list(command))
        return Process()

    monkeypatch.setattr("ipfs_accelerate_py.agent_supervisor.bundle_supervisor.subprocess.Popen", fake_popen)
    coordination = repo / "coordination.sqlite3"
    first = launch_bundle_lanes(lanes, repo_root=repo, coordination_path=coordination, claimant_did="did:web:lane-a.example")
    second = launch_bundle_lanes(lanes, repo_root=repo, coordination_path=coordination, claimant_did="did:web:lane-b.example")

    assert first[0]["accepted"] is True
    assert "ipfs_accelerate_py.agent_supervisor.leased_lane" in first[0]["command"]
    assert second[0]["accepted"] is False
    assert second[0]["code"] == "G_CLAIM_CONFLICT"
    assert len(starts) == 1
