"""Focused regressions for implementation-provider write authority."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Barrier
from typing import Any

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    parse_task_file,
)

_ResourceClaimResult = tuple[
    list[tuple[Path, dict[str, Any]]],
    str,
    str,
    dict[str, Any] | None,
]


def _resource_claim_daemon(
    repo: Path,
    *,
    lane: str,
) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / lane / "task_state.json",
        strategy_path=repo / lane / "strategy.json",
        events_path=repo / lane / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## PCTDD-",
        worktree_submodule_paths=("external/ipfs_accelerate",),
    )


def _resource_claim_task(task_id: str, predicted_path: str) -> PortalTask:
    return PortalTask(
        task_id=task_id,
        title=f"Implement {task_id}",
        status="ready",
        completion="manual",
        priority="P1",
        track="runtime",
        outputs=[predicted_path],
        metadata={"predicted files": predicted_path},
    )


def _concurrent_resource_claims(
    first_daemon: PortalImplementationDaemon,
    first_task: PortalTask,
    second_daemon: PortalImplementationDaemon,
    second_task: PortalTask,
) -> list[_ResourceClaimResult]:
    barrier = Barrier(2)

    def acquire(
        daemon: PortalImplementationDaemon,
        task: PortalTask,
    ) -> _ResourceClaimResult:
        barrier.wait(timeout=5)
        return daemon._acquire_implementation_resource_claims(
            task,
            attempt=1,
            started_at="2026-08-29T00:00:00Z",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(acquire, first_daemon, first_task),
            executor.submit(acquire, second_daemon, second_task),
        )
        return [future.result(timeout=10) for future in futures]


def test_concurrent_parent_and_child_resource_claims_have_one_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        implementation_daemon_module,
        "process_command_line",
        lambda _pid: str(sys.argv[0]),
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    first_daemon = _resource_claim_daemon(repo, lane="lane-a")
    second_daemon = _resource_claim_daemon(repo, lane="lane-b")
    parent_path = "external/ipfs_accelerate/ipfs_accelerate_py/runtime"
    child_path = f"{parent_path}/child"
    first_task = _resource_claim_task("PCTDD-101", parent_path)
    second_task = _resource_claim_task("PCTDD-102", child_path)
    results: list[_ResourceClaimResult] = []

    try:
        results = _concurrent_resource_claims(
            first_daemon,
            first_task,
            second_daemon,
            second_task,
        )
        winners = [index for index, result in enumerate(results) if result[0]]
        assert len(winners) == 1
        loser = results[1 - winners[0]]
        assert loser[2] == "overlapping_claim_exists"
        assert loser[3] is not None
        assert loser[3]["resource_path"] in {parent_path, child_path}
    finally:
        for daemon, result in zip((first_daemon, second_daemon), results):
            daemon._release_implementation_resource_claims(result[0])
        first_daemon.close_event_runtime()
        second_daemon.close_event_runtime()


def test_concurrent_disjoint_sibling_resource_claims_both_win(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        implementation_daemon_module,
        "process_command_line",
        lambda _pid: str(sys.argv[0]),
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    first_daemon = _resource_claim_daemon(repo, lane="lane-a")
    second_daemon = _resource_claim_daemon(repo, lane="lane-b")
    base_path = "external/ipfs_accelerate/ipfs_accelerate_py/runtime"
    first_task = _resource_claim_task("PCTDD-103", f"{base_path}/alpha")
    second_task = _resource_claim_task("PCTDD-104", f"{base_path}/beta")
    results: list[_ResourceClaimResult] = []

    try:
        results = _concurrent_resource_claims(
            first_daemon,
            first_task,
            second_daemon,
            second_task,
        )
        assert all(result[0] for result in results)
        assert all(result[1:] == ("", "acquired", None) for result in results)
    finally:
        for daemon, result in zip((first_daemon, second_daemon), results):
            daemon._release_implementation_resource_claims(result[0])
        first_daemon.close_event_runtime()
        second_daemon.close_event_runtime()
