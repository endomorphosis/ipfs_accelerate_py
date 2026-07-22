from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import DynamicBundleScheduler
from ipfs_accelerate_py.agent_supervisor.lease_coordination import LeaseCoordinator
from ipfs_accelerate_py.agent_supervisor.leased_lane import run_leased_lane_result

_MANIFEST_GRAPH_FIELDS = {
    "conflict_graph",
    "conflict_planning_decisions",
    "dependency_dag",
    "task_conflict_graph",
    "task_dependency_graph",
}


def _bundle(task_id: str, *, lane: str | None = None) -> dict[str, Any]:
    key = f"objective/test/{task_id.lower()}"
    return {
        "shard_path": f"bundles/{task_id.lower()}.todo.md",
        "parallel_lane": lane or key,
        "conflict_policy": "bundle-local edits only",
        "tasks": [{"task_id": task_id}],
    }


def _write_index(path: Path, *task_ids: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    f"objective/test/{task_id.lower()}": _bundle(task_id)
                    for task_id in task_ids
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


@dataclass
class _FakeProcess:
    pid: int
    alive: bool = True
    returncode: int | None = None

    def finish(self, returncode: int = 0) -> None:
        self.alive = False
        self.returncode = returncode


class _FakeLauncher:
    def __init__(self) -> None:
        self.starts: list[tuple[Any, Any, _FakeProcess]] = []

    def __call__(self, lane: Any, grant: Any) -> _FakeProcess:
        process = _FakeProcess(pid=10_000 + len(self.starts))
        self.starts.append((lane, grant, process))
        return process

    @staticmethod
    def alive(process: _FakeProcess) -> bool:
        return process.alive

    def process_for(self, task_id: str) -> _FakeProcess:
        return next(process for lane, _grant, process in self.starts if task_id in lane.task_ids)


def _scheduler(
    tmp_path: Path,
    index: Path,
    launcher: _FakeLauncher,
    *,
    claimant: str = "did:web:worker-a.example",
    max_lanes: int = 1,
    manifest_name: str = "manifest.json",
) -> DynamicBundleScheduler:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return DynamicBundleScheduler(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        manifest_path=repo / manifest_name,
        coordination_path=repo / "coordination.sqlite3",
        max_lanes=max_lanes,
        claimant_did=claimant,
        launcher=launcher,
        process_alive=launcher.alive,
        poll_interval=0,
    )


def _active_task_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        task_id
        for lane in manifest["lanes"]
        for task_id in lane.get("task_ids", [])
    }


def test_persistent_scheduler_discovers_new_and_refilled_work_without_restart(tmp_path: Path) -> None:
    """The same scheduler object must rescan its source after every drained lane."""

    repo = tmp_path / "repo"
    index = repo / "index.json"
    launcher = _FakeLauncher()
    _write_index(index, "T-1")
    scheduler = _scheduler(tmp_path, index, launcher)

    first = scheduler.reconcile_once()
    assert [lane.task_ids for lane, _grant, _process in launcher.starts] == [["T-1"]]
    assert first["counts"]["active"] == 1

    # This task did not exist when the scheduler started. Capacity prevents it
    # from launching until the current lane drains.
    _write_index(index, "T-2")
    second = scheduler.reconcile_once()
    assert len(launcher.starts) == 1
    assert second["counts"]["active"] == 1

    launcher.process_for("T-1").finish()
    third = scheduler.reconcile_once()
    assert [lane.task_ids for lane, _grant, _process in launcher.starts] == [["T-1"], ["T-2"]]
    assert _active_task_ids(third) == {"T-2"}

    # Refill again after the worker pool has already been running for multiple
    # reconciliation cycles. No new scheduler instance is constructed.
    _write_index(index, "T-3")
    launcher.process_for("T-2").finish()
    fourth = scheduler.reconcile_once()
    assert [lane.task_ids for lane, _grant, _process in launcher.starts] == [
        ["T-1"],
        ["T-2"],
        ["T-3"],
    ]
    assert _active_task_ids(fourth) == {"T-3"}


def test_pool_capacity_and_shared_leases_prevent_duplicate_execution(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1", "T-2", "T-3")
    first_launcher = _FakeLauncher()
    second_launcher = _FakeLauncher()
    first = _scheduler(tmp_path, index, first_launcher, max_lanes=2, manifest_name="first.json")
    second = _scheduler(
        tmp_path,
        index,
        second_launcher,
        claimant="did:web:worker-b.example",
        max_lanes=2,
        manifest_name="second.json",
    )

    first_manifest = first.reconcile_once()
    second_manifest = second.reconcile_once()

    # Each pool stays bounded, while the second worker is free to claim the one
    # conflict-safe task that the first pool did not have capacity to run.
    assert len(first_launcher.starts) == 2
    assert len(second_launcher.starts) == 1
    assert first_manifest["counts"]["active"] == 2
    assert second_manifest["counts"]["active"] == 1
    assert len(first_manifest["lanes"]) <= 2
    assert len(second_manifest["lanes"]) <= 2

    with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
        accepted = [task for task in coordinator.list_tasks() if task["state"] == "accepted"]
    assert len(accepted) == 3
    assert len({task["task_cid"] for task in accepted}) == len(accepted)

    # Reconciliation is idempotent while the accepted leases remain live.
    first.reconcile_once()
    second.reconcile_once()
    assert len(first_launcher.starts) == 2
    assert len(second_launcher.starts) == 1


def test_two_scheduler_processes_with_same_did_do_not_share_one_grant(tmp_path: Path) -> None:
    """A DID identifies authority, not a particular local executing process."""

    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1")
    owner_launcher = _FakeLauncher()
    peer_launcher = _FakeLauncher()
    owner = _scheduler(tmp_path, index, owner_launcher, manifest_name="owner.json")
    peer = _scheduler(tmp_path, index, peer_launcher, manifest_name="peer.json")

    owner.reconcile_once()
    peer_manifest = peer.reconcile_once()

    assert len(owner_launcher.starts) == 1
    assert not peer_launcher.starts
    assert peer_manifest["counts"]["active"] == 0
    assert peer_manifest["counts"]["blocked"] == 1


def test_settled_boards_release_capacity_without_starting_workers(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1")
    shard = repo / "bundles" / "t-1.todo.md"
    shard.parent.mkdir(parents=True)
    shard.write_text("- [x] Task checkbox-1: T-1 complete\n", encoding="utf-8")
    launcher = _FakeLauncher()
    scheduler = _scheduler(tmp_path, index, launcher)

    drained = scheduler.reconcile_once()

    assert not launcher.starts
    assert drained["counts"]["active"] == 0
    assert drained["counts"]["completed"] == 1

    # A board whose only remaining tasks are blocked relinquishes the slot and
    # consumes the bounded bundle attempt budget instead of pinning a daemon.
    _write_index(index, "T-2")
    blocked_shard = repo / "bundles" / "t-2.todo.md"
    blocked_shard.write_text("- [!] Task checkbox-1: T-2 blocked\n", encoding="utf-8")
    scheduler.reconcile_once()
    scheduler.reconcile_once()
    blocked = scheduler.reconcile_once()
    assert not launcher.starts
    assert blocked["counts"]["active"] == 0
    assert any(item["bundle_key"].endswith("t-2") for item in blocked["blocked"])


def test_authoritative_lane_state_releases_a_worker_with_no_ready_work(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1")
    launcher = _FakeLauncher()
    scheduler = _scheduler(tmp_path, index, launcher)

    initial = scheduler.reconcile_once()
    assert initial["counts"]["active"] == 1
    lane = launcher.starts[0][0]
    lane.state_dir.mkdir(parents=True, exist_ok=True)
    (lane.state_dir / f"{lane.state_prefix}_task_state.json").write_text(
        json.dumps(
            {
                "task_count": 2,
                "completed_count": 1,
                "blocked_count": 1,
                "waiting_count": 0,
                "implementation_in_progress": False,
                "active_task_id": "",
            }
        ),
        encoding="utf-8",
    )

    settled = scheduler.reconcile_once()

    assert settled["counts"]["active"] == 0
    # The first release remains retryable; repeated state-aware admission
    # consumes the bounded attempt budget without spawning another worker.
    for _ in range(3):
        settled = scheduler.reconcile_once()
    assert settled["counts"]["blocked"] == 1
    assert len(launcher.starts) == 1
    with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
        state = coordinator.task_state(launcher.starts[0][1].task_cid)
    assert state is not None and state["state"] == "blocked"


def test_drained_lane_releases_lease_for_conflict_safe_steal(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1")
    owner_launcher = _FakeLauncher()
    thief_launcher = _FakeLauncher()
    owner = _scheduler(tmp_path, index, owner_launcher, manifest_name="owner.json")
    thief = _scheduler(
        tmp_path,
        index,
        thief_launcher,
        claimant="did:web:worker-b.example",
        manifest_name="thief.json",
    )

    owner.reconcile_once()
    old_grant = owner_launcher.starts[0][1]
    blocked = thief.reconcile_once()
    assert blocked["counts"]["active"] == 0
    assert blocked["counts"]["blocked"] == 1
    assert not thief_launcher.starts

    # Removing the drained bundle before reaping it prevents the original
    # scheduler from immediately reclaiming its own released work.
    _write_index(index)
    owner_launcher.process_for("T-1").finish(returncode=75)
    drained = owner.reconcile_once()
    assert drained["counts"]["active"] == 0

    with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
        state = coordinator.task_state(old_grant.task_cid)
        assert state is not None
        assert state["state"] == "ready"

    _write_index(index, "T-1")
    stolen = thief.reconcile_once()
    replacement = thief_launcher.starts[0][1]
    assert replacement.task_cid == old_grant.task_cid
    assert replacement.claimant_did == "did:web:worker-b.example"
    assert replacement.fencing_token > old_grant.fencing_token
    assert _active_task_ids(stolen) == {"T-1"}


def test_manifest_is_an_authoritative_live_projection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    manifest_path = repo / "manifest.json"
    launcher = _FakeLauncher()
    _write_index(index, "T-1", "T-2")
    scheduler = _scheduler(tmp_path, index, launcher, manifest_name=manifest_path.name)

    initial = scheduler.reconcile_once()
    assert _active_task_ids(initial) == {"T-1"}
    assert initial["counts"]["active"] == len(initial["lanes"]) == 1
    assert initial["counts"]["ready"] == len(initial["ready"]) == 1

    _write_index(index, "T-2")
    launcher.process_for("T-1").finish()
    current = scheduler.reconcile_once()
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert persisted == current
    assert _active_task_ids(current) == {"T-2"}
    assert all("T-1" not in lane.get("task_ids", []) for lane in current["lanes"])
    assert current["counts"]["active"] == len(current["lanes"]) == 1
    assert current["counts"]["ready"] == len(current["ready"])
    assert current["counts"]["blocked"] == len(current["blocked"])
    assert current["counts"]["completed"] == len(current["completed"])
    assert "generated_at" in current

    # A terminal receipt changes the next projection in place; completed work
    # is not left behind in the active lane snapshot.
    _lane, grant, process = launcher.starts[-1]
    with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
        coordinator.receipt(grant, status="succeeded", output={"commit": "abc123"})
    process.finish()
    terminal = scheduler.reconcile_once()
    assert terminal["lanes"] == []
    assert terminal["counts"]["active"] == 0
    assert terminal["counts"]["completed"] == 1
    assert terminal["completed"][0]["task_cid"] == grant.task_cid


def test_manifest_references_full_planning_graphs_without_embedding_them(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    launcher = _FakeLauncher()
    _write_index(index, "T-1")
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    bundle = index_payload["bundles"]["objective/test/t-1"]
    bundle.update(
        {
            "objective_bundle_index": str(index),
            "task_dependency_graph": {"padding": "d" * 20_000},
            "dependency_dag": {"padding": "d" * 20_000},
            "task_conflict_graph": {"padding": "c" * 10_000},
            "conflict_graph": {"padding": "c" * 10_000},
            "conflict_planning_decisions": [{"padding": "p" * 5_000}],
        }
    )
    index.write_text(json.dumps(index_payload), encoding="utf-8")
    scheduler = _scheduler(tmp_path, index, launcher)

    manifest = scheduler.reconcile_once()

    task_bundle = manifest["tasks"][0]["bundle"]
    lane_bundle = manifest["lanes"][0]["queue_payload"]
    for projected in (task_bundle, lane_bundle):
        assert not _MANIFEST_GRAPH_FIELDS.intersection(projected)
        reference = projected["planning_evidence_ref"]
        assert reference["bundle_key"] == "objective/test/t-1"
        assert set(reference["omitted_fields"]) == _MANIFEST_GRAPH_FIELDS
    assert len(json.dumps(index_payload)) > 65_000
    assert len(json.dumps(manifest)) < 30_000


def test_leased_lane_publishes_terminal_and_blocked_projection(tmp_path: Path) -> None:
    coordination = tmp_path / "coordination.sqlite3"
    successful_bundle = {
        "bundle_key": "objective/test/success",
        "tasks": [{"task_id": "T-SUCCESS"}],
    }
    blocked_bundle = {
        "bundle_key": "objective/test/blocked",
        "tasks": [{"task_id": "T-BLOCKED"}],
        "max_attempts": 1,
    }
    with LeaseCoordinator(coordination) as coordinator:
        successful = coordinator.register_bundle(successful_bundle)
        successful_grant = coordinator.claim_ready(
            "did:web:worker.example",
            eligible_task_cids=(successful["task_cid"],),
            requested_lease_ms=5_000,
        )
        blocked = coordinator.register_bundle(blocked_bundle)

    assert successful_grant is not None
    successful_result = run_leased_lane_result(
        coordination_path=coordination,
        grant=successful_grant,
        command=(sys.executable, "-c", "raise SystemExit(0)"),
        lease_ms=5_000,
        heartbeat_interval=0.05,
    )
    assert successful_result.successful
    assert successful_result.receipt_cid

    with LeaseCoordinator(coordination) as coordinator:
        blocked_grant = coordinator.claim_ready(
            "did:web:worker.example",
            eligible_task_cids=(blocked["task_cid"],),
            requested_lease_ms=5_000,
        )
    assert blocked_grant is not None
    blocked_result = run_leased_lane_result(
        coordination_path=coordination,
        grant=blocked_grant,
        command=(sys.executable, "-c", "raise SystemExit(75)"),
        lease_ms=5_000,
        heartbeat_interval=0.05,
    )
    assert blocked_result.disposition == "blocked"

    with LeaseCoordinator(coordination) as coordinator:
        assert coordinator.task_state(successful_grant.task_cid)["state"] == "completed"
        blocked_state = coordinator.task_state(blocked_grant.task_cid)
        assert blocked_state["state"] == "blocked"
        assert coordinator.claim_ready(
            "did:web:another-worker.example",
            eligible_task_cids=(blocked_grant.task_cid,),
            requested_lease_ms=5_000,
        ) is None
