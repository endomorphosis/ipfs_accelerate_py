"""Real same-HEAD dependency drift and the existing active-attempt reload fence."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTaskState

DEPENDENCIES = (
    "ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
    "ipfs_accelerate_py/agent_supervisor/proof/code_proof_obligations.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "control-plane"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Reload Fixture")
    _git(repo, "config", "user.email", "reload@example.invalid")
    for relative in set(runtime.CONTROL_PLANE_SOURCE_PATHS) | set(DEPENDENCIES):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Initial control-plane source\n")
    (repo / "notes.txt").write_text("Initial unrelated source\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial control plane")
    return repo


@pytest.mark.parametrize("relative", DEPENDENCIES)
@pytest.mark.parametrize("delete", [False, True])
def test_critical_dependency_drift_is_detected_without_a_git_commit(repository, relative, delete):
    assert relative in runtime.CONTROL_PLANE_SOURCE_PATHS
    before = runtime._read_control_plane_source_snapshot(repository)
    path = repository / relative
    if delete:
        path.unlink()
    else:
        path.write_text("# Changed dependency at the same Git HEAD\n")
    after = runtime._read_control_plane_source_snapshot(repository)
    assert after["repository_revision"] == before["repository_revision"]
    assert after["control_plane_tree_id"] == before["control_plane_tree_id"]
    assert after["source_id"] != before["source_id"]
    assert runtime._control_plane_update_is_pending(before, after) is True
    row = next(row for row in after["sources"] if row["path"] == relative)
    assert row["available"] is not delete


def test_unrelated_dirty_file_does_not_change_the_watched_source_identity(repository):
    before = runtime._read_control_plane_source_snapshot(repository)
    (repository / "notes.txt").write_text("Uncommitted unrelated edit\n")
    after = runtime._read_control_plane_source_snapshot(repository)
    assert after["source_id"] == before["source_id"]
    assert runtime._control_plane_update_is_pending(before, after) is False
    _git(repository, "add", "notes.txt")
    _git(repository, "commit", "-q", "-m", "committed repository advance")
    committed = runtime._read_control_plane_source_snapshot(repository)
    assert committed["control_plane_tree_id"] == before["control_plane_tree_id"]
    assert runtime._control_plane_update_is_pending(before, committed) is True


@pytest.mark.parametrize("relative", DEPENDENCIES)
def test_real_dependency_drift_defers_active_work_then_requests_uncharged_reload(repository, tmp_path, relative):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "task-state.json"
    original = PortalTaskState(active_task_id="WATCH-001", active_phase="validating",
        implementation_in_progress=True, implementation_attempts={"WATCH-001": 2})
    original.save(state_path)
    supervisor = runtime.PortalImplementationSupervisor(runtime.PortalSupervisorConfig(
        todo_path=repository / "tasks.todo.md", state_path=state_path,
        strategy_path=state_dir / "strategy.json", events_path=state_dir / "events.jsonl",
        state_dir=state_dir, repo_root=repository,
    ))
    supervisor._control_plane_probe_root = repository
    supervisor._loaded_control_plane_source = runtime._read_control_plane_source_snapshot(repository)
    (repository / relative).write_text("# Reviewed dependency update\n")
    supervisor._last_control_plane_source_probe_monotonic = 0
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    child = SimpleNamespace(pid=os.getpid())

    active = supervisor._supervisor_loop_watchdog_decision(loop, child, {})

    assert active.action == "continue"
    assert loop.config.status_extra_fields["control_plane_reload_deferred"] is True
    assert loop.config.status_extra_fields["control_plane_reload_deferred_task_id"] == "WATCH-001"
    assert PortalTaskState.load(state_path).implementation_attempts == {"WATCH-001": 2}
    PortalTaskState(implementation_attempts={"WATCH-001": 2}).save(state_path)
    supervisor._last_control_plane_source_probe_monotonic = 0

    idle = supervisor._supervisor_loop_watchdog_decision(loop, child, {})

    assert idle.action == "stop"
    assert idle.status == runtime.CONTROL_PLANE_RELOAD_STATUS
    assert loop.config.status_extra_fields["control_plane_reload_attempt_budget_consumed"] is False
    assert loop.config.status_extra_fields["control_plane_reload_provider_invocation_consumed"] is False
    assert PortalTaskState.load(state_path).implementation_attempts == {"WATCH-001": 2}
