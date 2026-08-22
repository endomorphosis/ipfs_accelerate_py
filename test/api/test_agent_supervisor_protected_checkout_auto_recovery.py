"""Auto-recovery for leftover supervisor protected-checkout journals."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_lock_metadata,
    checkout_mutation_lock_path,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA,
    DatabasePortalExecutionBridge,
    is_protected_checkout_setup_block,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PORTAL_RETRY_DEFERRAL_SCHEMA,
    PortalImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _seed_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    todo_path = repo / "docs" / "generated.todo.md"
    todo_path.parent.mkdir()
    todo_path.write_text("# Generated board\n", encoding="utf-8")
    _git(repo, "add", "docs/generated.todo.md")
    _git(repo, "commit", "-m", "seed generated board")
    return repo, todo_path


def test_supervisor_releases_inactive_clean_leftover_recovery_journal(
    tmp_path: Path,
) -> None:
    repo, todo_path = _seed_repo(tmp_path)
    state_dir = tmp_path / "state"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "supervisor_task_state.json",
            strategy_path=state_dir / "supervisor_strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            implementation_protected_paths=("docs/generated.todo.md",),
        )
    )
    lock_path = checkout_mutation_lock_path(repo)
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="APMC-013",
        attempt=1,
        extra={"operation": "generated_board_update"},
    )
    metadata.update(
        {
            "pid": 999999999,
            "protected_recovery_required": True,
            "protected_recovery_owner": "implementation_supervisor",
            "protected_paths": ["docs/generated.todo.md"],
        }
    )
    lock_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    recovered = supervisor._recover_retained_generated_checkout_lease()

    assert recovered["recovered"] is True
    assert recovered["retained_lease"] is False
    assert recovered["reason"] == "stale_supervisor_protected_recovery_released"
    assert not lock_path.exists()


def test_portal_run_once_emits_setup_deferral_for_external_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, todo_path = _seed_repo(tmp_path)
    state_dir = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## APMC-",
        implementation_protected_paths=("docs/generated.todo.md",),
    )
    monkeypatch.setattr(
        daemon,
        "_recover_protected_checkout_mutation",
        lambda: {
            "required": True,
            "recovered": False,
            "blocked": True,
            "reason": "external_protected_checkout_recovery_required",
        },
    )

    result = daemon.run_once()

    assert result["blocked"] is True
    implementation = result["implementation_result"]
    assert implementation["deferral_schema"] == PORTAL_RETRY_DEFERRAL_SCHEMA
    assert implementation["deferral_schema"] == DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA
    assert implementation["deferred"] is True
    assert implementation["attempt_consumed"] is False
    assert implementation["failure_kind"] == "lifecycle_setup"
    assert implementation["provider_dispatched"] is False
    assert is_protected_checkout_setup_block(str(result["reason"]))
    assert DatabasePortalExecutionBridge._explicit_retryable_deferral(
        implementation
    )
    assert DatabasePortalExecutionBridge._terminal_failure(result) == (
        "external_protected_checkout_recovery_required"
    )
