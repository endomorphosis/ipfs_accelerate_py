from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import (
    persistent_task_queue as queue_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.persistent_task_queue import (
    PersistentTaskQueue,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ImplementationRetryDeferred,
    PortalTaskState,
    TodoImplementationDaemon,
    _codex_implementation_command,
)


def test_persistent_task_queue_deferral_is_durable_and_non_consuming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1_000.0}
    monkeypatch.setattr(queue_module.time, "time", lambda: clock["now"])
    queue_path = tmp_path / "task_queue.json"
    queue = PersistentTaskQueue.load(queue_path, save_interval=0)

    queue.record_selection("task:provider-unavailable")
    queue.defer(
        "task:provider-unavailable",
        300,
        reason="primary provider unavailable",
    )
    queue.save()

    restored = PersistentTaskQueue.load(queue_path)
    entry = restored.entries["task:provider-unavailable"]
    assert entry.attempt_count == 1
    assert entry.consecutive_failures == 0
    assert entry.selection_penalty == 0
    assert entry.cooldown_until == 1_300.0
    assert entry.notes == "primary provider unavailable"


def test_direct_codex_command_is_ephemeral_and_ignores_ambient_user_config(
    tmp_path: Path,
) -> None:
    command = _codex_implementation_command(
        codex="/usr/local/bin/codex",
        workspace_path=tmp_path,
        model_override="gpt-5.6-terra",
        reasoning_effort_override="medium",
    )

    assert command[:4] == [
        "/usr/local/bin/codex",
        "exec",
        "--ignore-user-config",
        "--ephemeral",
    ]
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert command[-1] == "-"
    assert "features.code_mode=false" not in command
    assert "features.code_mode_only=false" not in command
    assert "features.code_mode_host=false" not in command
    assert not any("grok" in part.lower() or "docker" in part.lower() for part in command)


@pytest.mark.parametrize("gate_admitted", [False, True])
def test_provider_unavailable_defers_before_dispatch_without_daemon_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_admitted: bool,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text(
        """# Tasks

## RETRY-001 Wait for the primary provider

- Status: todo
- Completion: manual
- Priority: P0
- Track: runtime
- Outputs: provider-effect.txt
- Acceptance: Provider-unavailable work remains pending without side effects.
""",
        encoding="utf-8",
    )
    # Provider dispatch requires a real registered worktree even when this
    # test deliberately stops at the provider readiness check.
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "add", "tasks.todo.md"], cwd=repo, check=True)
    subprocess.run([
        "git", "-c", "user.name=Retry Test", "-c", "user.email=retry@example.invalid",
        "commit", "-qm", "test board",
    ], cwd=repo, check=True)
    state_dir = repo / "state"
    events_path = state_dir / "events.jsonl"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=events_path,
        repo_root=repo,
        task_header_prefix="## RETRY-",
        implement=True,
        use_ephemeral_worktree=True,
        worktree_root=tmp_path / "worktrees",
    )
    provider_effects: list[str] = []
    readiness_calls: list[object] = []

    def provider_unavailable(_task: object) -> None:
        readiness_calls.append(_task)
        raise ImplementationRetryDeferred(
            "primary provider unavailable",
            backoff_seconds=300,
        )

    monkeypatch.setattr(
        daemon,
        "_require_primary_provider_readiness",
        provider_unavailable,
    )
    monkeypatch.setattr(
        daemon,
        "_build_implementation_prompt",
        lambda *_args: provider_effects.append("prompt") or "unexpected",
    )
    monkeypatch.setattr(
        daemon,
        "_prepare_residual_provider_handoff",
        lambda **_kwargs: pytest.fail("unavailable provider must not prepare a handoff"),
    )
    if gate_admitted:
        # Isolate readiness after the gate; this incomplete test result must
        # never reach the real handoff/dispatch authority boundary above.
        monkeypatch.setattr(
            daemon,
            "_evaluate_pre_implementation_provider_gate",
            lambda **_kwargs: {"skip_provider": False},
        )

    # A missing queue adapter used to raise AttributeError here, terminating
    # the daemon process and making the supervisor restart it.  Returning a
    # typed retry result proves the same daemon turn survives the condition.
    result = daemon.run_once()

    implementation = result["implementation_result"]
    expected_reason = (
        "primary_provider_unavailable"
        if gate_admitted
        else "pre_implementation_abstain_review_no_analytical_close"
    )
    assert implementation["deferred"] is True
    assert implementation["retryable"] is True
    assert implementation["reason"] == expected_reason
    assert len(readiness_calls) == int(gate_admitted)
    assert implementation["attempt_consumed"] is False
    assert implementation["provider_dispatched"] is False
    assert provider_effects == []
    assert not (repo / "provider-effect.txt").exists()

    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_attempts == {}
    assert state.implementation_attempts_by_cid == {}
    assert state.implementation_in_progress is False
    # Worktree setup is recorded, but its deferral does not spend an attempt.
    assert state.last_implementation_finished_at

    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("type") == "implementation_retry_deferred"
        and event.get("reason") == expected_reason
        for event in events
    )
    assert not any(
        event.get("type")
        in {"provider_invocation_committed", "provider_callback_started"}
        for event in events
    )
    terminal_events = [
        event for event in events if event.get("type") == "implementation_finished"
    ]
    assert terminal_events
    assert all(event.get("attempt_consumed") is False for event in terminal_events)
    assert all(event.get("provider_dispatched") is False for event in terminal_events)
    assert len(daemon.task_queue.entries) == 1
    [queue_entry] = daemon.task_queue.entries.values()
    assert queue_entry.is_cooled_down() is True
    assert queue_entry.notes == expected_reason
