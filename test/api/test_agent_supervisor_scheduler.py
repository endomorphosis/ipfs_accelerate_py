from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Barrier
from typing import Any

from ipfs_accelerate_py.agent_supervisor import bundle_supervisor as bundle_supervisor_module
from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import DynamicBundleScheduler
from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import launch_bundle_lanes
from ipfs_accelerate_py.agent_supervisor.lease_coordination import LeaseCoordinator
from ipfs_accelerate_py.agent_supervisor.leased_lane import run_leased_lane_result
from ipfs_accelerate_py.agent_supervisor.resource_scheduler import HostResourceSnapshot
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive

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
    terminate_calls: int = 0
    wait_calls: int = 0
    kill_calls: int = 0

    def finish(self, returncode: int = 0) -> None:
        self.alive = False
        self.returncode = returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.finish(-signal.SIGTERM)

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.alive:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        return int(self.returncode or 0)

    def kill(self) -> None:
        self.kill_calls += 1
        self.finish(-signal.SIGKILL)


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


class _StubbornFakeProcess(_FakeProcess):
    def terminate(self) -> None:
        self.terminate_calls += 1


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

    def host_resource_source(
        _state_root: Path,
        *,
        active_workers: int,
        worker_limit: int,
        active_phase: str,
    ) -> HostResourceSnapshot:
        return HostResourceSnapshot(
            cpu_percent=10,
            memory_percent=10,
            disk_percent=10,
            active_phase=active_phase,
            active_workers=active_workers,
            worker_limit=worker_limit,
            available_worker_capacity=max(0, worker_limit - active_workers),
        )

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
        host_resource_source=host_resource_source,
        poll_interval=0,
        task_prefix="T-",
    )


def _active_task_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        task_id
        for lane in manifest["lanes"]
        for task_id in lane.get("task_ids", [])
    }


def test_terminate_handle_kills_and_reaps_an_unresponsive_wrapper() -> None:
    process = _StubbornFakeProcess(pid=10_000)

    DynamicBundleScheduler._terminate_handle(process, grace_seconds=0)

    assert process.terminate_calls == 1
    assert process.wait_calls == 2
    assert process.kill_calls == 1
    assert not process.alive


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
    assert launcher.process_for("T-1").wait_calls == 1
    assert launcher.process_for("T-1").terminate_calls == 0

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


def test_dependency_blocked_candidate_does_not_consume_admission_capacity(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1", "T-2")
    launcher = _FakeLauncher()
    scheduler = _scheduler(tmp_path, index, launcher, max_lanes=1)
    lanes = scheduler._plan()
    blocked = replace(
        lanes[0],
        queue_payload={
            **dict(lanes[0].queue_payload or {}),
            "dependency_task_cids": ["bmissing-prerequisite"],
        },
    )
    scheduler._plan = lambda: [blocked, lanes[1]]  # type: ignore[method-assign]

    manifest = scheduler.reconcile_once()

    assert [lane.task_ids for lane, _grant, _process in launcher.starts] == [["T-2"]]
    blocked_task = next(
        task for task in manifest["tasks"] if task["bundle_key"].endswith("t-1")
    )
    assert blocked_task["state"] == "blocked"
    assert blocked_task["blocked_reason"] == "dependency_not_ready"
    assert blocked_task["missing_dependency_task_cids"] == ["bmissing-prerequisite"]
    decision = next(
        item
        for item in manifest["scheduler_decisions"]
        if item["bundle_key"].endswith("t-1")
    )
    assert decision["reason"] == "snapshot_not_ready"


def test_planner_blocked_lane_ignores_stale_ready_metric_identity(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1")
    launcher = _FakeLauncher()
    scheduler = _scheduler(tmp_path, index, launcher)
    lane = replace(scheduler._plan()[0], claimable=False)
    scheduler._plan = lambda: [lane]  # type: ignore[method-assign]
    lane.state_dir.mkdir(parents=True, exist_ok=True)
    (lane.state_dir / f"{lane.state_prefix}_events.jsonl").write_text(
        json.dumps(
            {
                "type": "scheduler_lane_state",
                "timestamp": "2099-01-01T00:00:00Z",
                "phase": "ready",
                "goal_cid": "stale-goal-identity",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = scheduler.reconcile_once()

    assert not launcher.starts
    assert manifest["counts"]["ready"] == 0
    assert manifest["counts"]["blocked"] == 1
    assert manifest["scheduler_decisions"][0]["reason"] == "snapshot_not_ready"


def test_lease_race_backfills_the_admission_slot_in_the_same_cycle(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1", "T-2")
    launcher = _FakeLauncher()
    scheduler = _scheduler(tmp_path, index, launcher, max_lanes=1)
    original_claim_ready = LeaseCoordinator.claim_ready
    failed_once = False

    def lose_first_lease(self: LeaseCoordinator, *args: Any, **kwargs: Any):
        nonlocal failed_once
        if not failed_once and kwargs.get("eligible_task_cids"):
            failed_once = True
            return None
        return original_claim_ready(self, *args, **kwargs)

    monkeypatch.setattr(LeaseCoordinator, "claim_ready", lose_first_lease)

    manifest = scheduler.reconcile_once()

    assert failed_once is True
    assert [lane.task_ids for lane, _grant, _process in launcher.starts] == [["T-2"]]
    decisions = {
        item["bundle_key"]: item for item in manifest["scheduler_decisions"]
    }
    assert decisions["objective/test/t-1"]["reason"] == "lease_unavailable"
    assert decisions["objective/test/t-2"]["decision"] == "launched"
    assert manifest["resource_schedule"]["admitted_count"] == 1


def test_concurrent_serial_and_dynamic_supervisors_publish_one_canonical_result(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A serial launch racing the persistent pool must remain exactly once.

    This exercises the process boundary contract, not merely two calls on one
    coordinator instance: both supervisor entry paths open their own SQLite
    connection, rediscover/register the same generated bundle, and contend for
    its canonical lease.  The winning execution then publishes the only
    terminal receipt and a fresh scheduler process reconstructs the completed
    projection from durable state.
    """

    repo = tmp_path / "repo"
    index = repo / "index.json"
    generated_bundle = _bundle("T-GENERATED-1")
    generated_bundle["tasks"] = [
        {
            "task_id": "T-GENERATED-1",
            "goal_id": "G-GENERATED",
            "title": "Generate the canonical goal output",
            "status": "todo",
        },
        {
            "task_id": "T-GENERATED-2",
            "goal_id": "G-GENERATED",
            "title": "Validate the generated output",
            "status": "todo",
            "depends_on": ["T-GENERATED-1"],
        },
    ]
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {"objective/test/generated": generated_bundle},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    dynamic_launcher = _FakeLauncher()
    dynamic = _scheduler(
        tmp_path,
        index,
        dynamic_launcher,
        claimant="did:web:dynamic-supervisor.example",
        manifest_name="dynamic.json",
    )
    lane = dynamic._plan()[0]
    assert lane.queue_payload is not None
    # Bundle workers cannot independently refill their shard and manufacture
    # another copy of objective work while the serial supervisor owns refill.
    assert "--objective-refill-scan" not in lane.command
    assert "--codebase-refill-scan" not in lane.command

    coordination = repo / "coordination.sqlite3"
    with LeaseCoordinator(coordination) as coordinator:
        initially_registered = coordinator.register_bundle(lane.queue_payload)

    race = Barrier(2)
    original_claim = LeaseCoordinator.claim

    def synchronized_serial_claim(self: LeaseCoordinator, *args: Any, **kwargs: Any):
        race.wait(timeout=10)
        return original_claim(self, *args, **kwargs)

    monkeypatch.setattr(LeaseCoordinator, "claim", synchronized_serial_claim)
    original_resource_source = dynamic._host_resource_source

    def synchronized_dynamic_admission(*args: Any, **kwargs: Any) -> HostResourceSnapshot:
        race.wait(timeout=10)
        return original_resource_source(*args, **kwargs)

    dynamic._host_resource_source = synchronized_dynamic_admission

    serial_process = _FakeProcess(pid=20_000)
    serial_grants: list[Any] = []

    def fake_serial_spawn(serial_lane: Any, grant: Any, **_kwargs: Any):
        serial_grants.append(grant)
        return serial_process, list(serial_lane.command), serial_lane.state_dir / "serial.pid"

    monkeypatch.setattr(bundle_supervisor_module, "_spawn_accepted_lane", fake_serial_spawn)

    with ThreadPoolExecutor(max_workers=2) as executor:
        serial_future = executor.submit(
            launch_bundle_lanes,
            [lane],
            repo_root=repo,
            coordination_path=coordination,
            claimant_did="did:web:serial-supervisor.example",
        )
        dynamic_future = executor.submit(dynamic.reconcile_once)
        serial_result = serial_future.result(timeout=15)
        dynamic_manifest = dynamic_future.result(timeout=15)

    serial_accepted = bool(serial_result[0]["accepted"])
    dynamic_accepted = bool(dynamic_launcher.starts)
    assert serial_accepted + dynamic_accepted == 1
    assert len(serial_grants) == int(serial_accepted)
    assert dynamic_manifest["counts"]["active"] == int(dynamic_accepted)

    winning_grant = (
        serial_grants[0] if serial_accepted else dynamic_launcher.starts[0][1]
    )
    with LeaseCoordinator(coordination) as coordinator:
        receipt = coordinator.receipt(
            winning_grant,
            status="succeeded",
            output={"merge_commit": "abc123", "generated_goal_id": "G-GENERATED"},
        )
        receipts = coordinator.list_receipts(winning_grant.task_cid)
        tasks = coordinator.list_tasks()

    assert len(receipts) == 1
    assert receipts[0]["receipt_cid"] == receipt["receipt_cid"]
    assert len(tasks) == 1
    assert tasks[0]["task_cid"] == initially_registered["task_cid"]
    assert tasks[0]["goal_cid"] == receipt["goal_cid"]
    assert tasks[0]["subgoal_cid"] == receipt["subgoal_cid"]
    persisted_members = tasks[0]["bundle"]["tasks"]
    assert [item["task_id"] for item in persisted_members] == [
        "T-GENERATED-1",
        "T-GENERATED-2",
    ]
    assert {item["goal_id"] for item in persisted_members} == {"G-GENERATED"}
    assert persisted_members[1]["dependency_task_cids"] == [
        persisted_members[0]["canonical_task_cid"]
    ]

    if dynamic_accepted:
        dynamic_launcher.starts[0][2].finish()
    else:
        serial_process.finish()

    restarted_launcher = _FakeLauncher()
    restarted = _scheduler(
        tmp_path,
        index,
        restarted_launcher,
        claimant="did:web:restarted-supervisor.example",
        manifest_name="restarted.json",
    )
    restarted_manifest = restarted.reconcile_once()

    assert not restarted_launcher.starts
    assert restarted_manifest["counts"] == {
        "active": 0,
        "ready": 0,
        "blocked": 0,
        "completed": 1,
        "capacity": 1,
        "effective_capacity": 1,
    }
    completed = restarted_manifest["completed"][0]
    assert completed["task_cid"] == winning_grant.task_cid
    assert completed["goal_cid"] == receipt["goal_cid"]
    assert completed["subgoal_cid"] == receipt["subgoal_cid"]
    assert completed["bundle"]["tasks"][1]["dependency_task_cids"] == [
        completed["bundle"]["tasks"][0]["canonical_task_cid"]
    ]
    with LeaseCoordinator(coordination) as coordinator:
        assert len(coordinator.list_tasks()) == 1
        assert len(coordinator.list_receipts(winning_grant.task_cid)) == 1


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

    blocked_shard.write_text("- [ ] Task checkbox-1: T-2 reopened\n", encoding="utf-8")
    reopened = scheduler.reconcile_once()

    assert reopened["counts"]["active"] == 1
    assert launcher.starts[-1][0].task_ids == ["T-2"]
    assert launcher.starts[-1][1].attempt == 1


def test_explicit_terminal_status_uses_the_configured_task_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_index(index, "T-1")
    shard = repo / "bundles" / "t-1.todo.md"
    shard.parent.mkdir(parents=True)
    shard.write_text(
        "- [ ] Task checkbox-1: stale checkbox\n\n"
        "## T-1 Completed by a prior lane\n\n"
        "- Status: completed\n"
        "- Completion: manual\n",
        encoding="utf-8",
    )
    launcher = _FakeLauncher()
    scheduler = _scheduler(tmp_path, index, launcher)

    settled = scheduler.reconcile_once()

    assert not launcher.starts
    assert settled["counts"]["active"] == 0
    assert settled["counts"]["completed"] == 1


def test_stale_terminal_state_does_not_settle_a_refilled_bundle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    bundle = _bundle("T-1")
    bundle["tasks"] = [
        {"task_id": "T-1", "title": "Prior work", "paths": ["prior.py"]},
        {"task_id": "T-2", "title": "Refilled work", "paths": ["refilled.py"]},
    ]
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps({"source_todo": "tasks.todo.md", "bundles": {"objective/test/t-1": bundle}}),
        encoding="utf-8",
    )
    shard = repo / "bundles" / "t-1.todo.md"
    shard.parent.mkdir(parents=True)
    shard.write_text(
        "- [x] Task checkbox-1: T-1 complete\n\n"
        "## T-1 Prior work\n\n"
        "- Status: completed\n\n"
        "- [ ] Task checkbox-2: T-2 newly refilled\n\n"
        "## T-2 Refilled work\n\n"
        "- Status: todo\n",
        encoding="utf-8",
    )
    launcher = _FakeLauncher()
    scheduler = _scheduler(tmp_path, index, launcher)
    lane = scheduler._plan()[0]
    lane.state_dir.mkdir(parents=True, exist_ok=True)
    (lane.state_dir / f"{lane.state_prefix}_task_state.json").write_text(
        json.dumps(
            {
                "task_count": 1,
                "completed_count": 1,
                "blocked_count": 0,
                "waiting_count": 0,
                "implementation_in_progress": False,
                "active_task_id": "",
                "task_identities": {"T-1": {"display_task_id": "T-1"}},
            }
        ),
        encoding="utf-8",
    )

    active = scheduler.reconcile_once()

    assert len(launcher.starts) == 1
    assert active["counts"]["active"] == 1
    assert active["counts"]["completed"] == 0


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
    assert launcher.starts[0][2].terminate_calls == 1
    assert launcher.starts[0][2].wait_calls == 1
    assert launcher.starts[0][2].kill_calls == 0
    # The first release remains retryable; repeated state-aware admission
    # consumes the bounded attempt budget without spawning another worker.
    for _ in range(3):
        settled = scheduler.reconcile_once()
    assert settled["counts"]["blocked"] == 1
    assert len(launcher.starts) == 1
    with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
        state = coordinator.task_state(launcher.starts[0][1].task_cid)
    assert state is not None and state["state"] == "blocked"


def test_transitively_blocked_waiting_tasks_release_an_idle_lane(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    bundle = _bundle("T-1")
    bundle["tasks"] = [
        {"task_id": "T-1", "title": "Blocked root"},
        {"task_id": "T-2", "title": "First dependent", "depends_on": ["T-1"]},
        {"task_id": "T-3", "title": "Second dependent", "depends_on": ["T-2"]},
    ]
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps({"source_todo": "tasks.todo.md", "bundles": {"objective/test/t-1": bundle}}),
        encoding="utf-8",
    )
    launcher = _FakeLauncher()
    scheduler = _scheduler(tmp_path, index, launcher)

    initial = scheduler.reconcile_once()
    assert initial["counts"]["active"] == 1
    lane = launcher.starts[0][0]
    lane.todo_path.parent.mkdir(parents=True, exist_ok=True)
    lane.todo_path.write_text(
        "## T-1 Blocked root\n\n"
        "- Status: blocked\n"
        "- Depends on: none\n\n"
        "## T-2 First dependent\n\n"
        "- Status: todo\n"
        "- Depends on: T-1\n\n"
        "## T-3 Second dependent\n\n"
        "- Status: todo\n"
        "- Depends on: T-2\n",
        encoding="utf-8",
    )
    lane.state_dir.mkdir(parents=True, exist_ok=True)
    (lane.state_dir / f"{lane.state_prefix}_task_state.json").write_text(
        json.dumps(
            {
                "task_count": 3,
                "completed_count": 0,
                "ready_count": 0,
                "blocked_count": 1,
                "waiting_count": 2,
                "implementation_in_progress": False,
                "active_task_id": "",
                "task_statuses": {
                    "T-1": "blocked",
                    "T-2": "waiting",
                    "T-3": "waiting",
                },
                "task_identities": {
                    task_id: {"display_task_id": task_id}
                    for task_id in ("T-1", "T-2", "T-3")
                },
            }
        ),
        encoding="utf-8",
    )

    settled = scheduler.reconcile_once()

    assert settled["counts"]["active"] == 0
    assert launcher.starts[0][2].terminate_calls == 1
    assert launcher.starts[0][2].wait_calls == 1
    assert len(launcher.starts) == 1


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


def test_manifest_excludes_superseded_bundle_revisions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    launcher = _FakeLauncher()
    initial_bundle = _bundle("T-1")
    initial_bundle["tasks"][0].update({"title": "Old work", "outputs": ["old.py"]})
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps({"source_todo": "tasks.todo.md", "bundles": {"objective/test/t-1": initial_bundle}}),
        encoding="utf-8",
    )
    scheduler = _scheduler(tmp_path, index, launcher)

    first = scheduler.reconcile_once()
    old_task_cid = first["lanes"][0]["task_cid"]
    launcher.starts[0][2].finish()
    replacement = _bundle("T-1")
    replacement["tasks"][0].update({"title": "Replacement work", "outputs": ["new.py"]})
    index.write_text(
        json.dumps({"source_todo": "tasks.todo.md", "bundles": {"objective/test/t-1": replacement}}),
        encoding="utf-8",
    )

    current = scheduler.reconcile_once()

    assert len(current["tasks"]) == 1
    assert current["tasks"][0]["task_cid"] != old_task_cid
    assert all(item["task_cid"] != old_task_cid for item in current["ready"])
    with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
        assert len(coordinator.list_tasks()) == 2
        assert len(coordinator.list_tasks(task_cids={current["tasks"][0]["task_cid"]})) == 1


def test_live_bundle_revision_blocks_replacement_with_the_same_bundle_key(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    launcher = _FakeLauncher()
    initial_bundle = _bundle("T-1")
    initial_bundle["tasks"][0].update({"title": "Old work", "outputs": ["old.py"]})
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {"objective/test/t-1": initial_bundle},
            }
        ),
        encoding="utf-8",
    )
    scheduler = _scheduler(tmp_path, index, launcher, max_lanes=2)

    first = scheduler.reconcile_once()
    old_task_cid = first["lanes"][0]["task_cid"]
    replacement = _bundle("T-1")
    replacement["tasks"][0].update(
        {"title": "Replacement work", "outputs": ["new.py"]}
    )
    index.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {"objective/test/t-1": replacement},
            }
        ),
        encoding="utf-8",
    )

    current = scheduler.reconcile_once()

    assert len(launcher.starts) == 1
    assert current["counts"]["active"] == 1
    assert current["lanes"][0]["task_cid"] == old_task_cid
    decision = next(
        item
        for item in current["scheduler_decisions"]
        if item["task_cid"] != old_task_cid
    )
    assert decision["reason"] == "bundle_key_active"
    assert decision["blocking_task_cid"] == old_task_cid


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


def test_leased_lane_signal_terminates_detached_descendants(tmp_path: Path) -> None:
    coordination = tmp_path / "coordination.sqlite3"
    marker = tmp_path / "descendant.pid"
    bundle = {
        "bundle_key": "objective/test/process-tree",
        "tasks": [{"task_id": "T-PROCESS-TREE"}],
    }
    with LeaseCoordinator(coordination) as coordinator:
        registered = coordinator.register_bundle(bundle)
        grant = coordinator.claim_ready(
            "did:web:worker.example",
            eligible_task_cids=(registered["task_cid"],),
            requested_lease_ms=300_000,
        )
    assert grant is not None
    child_script = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "start_new_session=True);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(60)"
    )
    wrapper = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.leased_lane",
            "--coordination-path",
            str(coordination),
            "--grant-json",
            json.dumps(grant.to_dict()),
            "--lease-ms",
            "300000",
            "--heartbeat-interval",
            "30",
            "--",
            sys.executable,
            "-c",
            child_script,
            str(marker),
        ]
    )
    descendant_pid = 0
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        descendant_pid = int(marker.read_text(encoding="utf-8"))
        assert pid_alive(descendant_pid)

        wrapper.send_signal(signal.SIGTERM)
        wrapper.wait(timeout=15)
        deadline = time.monotonic() + 5
        while pid_alive(descendant_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pid_alive(descendant_pid)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=5)
        if descendant_pid and pid_alive(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)
