from __future__ import annotations

import json
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
    assert not any("grok" in part.lower() or "docker" in part.lower() for part in command)


def test_provider_unavailable_defers_before_dispatch_without_daemon_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    )
    provider_effects: list[str] = []

    def provider_unavailable(_task: object) -> None:
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

    # A missing queue adapter used to raise AttributeError here, terminating
    # the daemon process and making the supervisor restart it.  Returning a
    # typed retry result proves the same daemon turn survives the condition.
    result = daemon.run_once()

    implementation = result["implementation_result"]
    assert implementation["skipped"] is True
    assert implementation["reason"] == "primary_provider_unavailable"
    assert implementation["attempt_consumed"] is False
    assert implementation["provider_dispatched"] is False
    assert provider_effects == []
    assert not (repo / "provider-effect.txt").exists()

    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_attempts == {}
    assert state.implementation_attempts_by_cid == {}
    assert state.implementation_in_progress is False
    assert state.last_implementation_finished_at == ""

    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("type") == "implementation_retry_deferred"
        and event.get("reason") == "primary_provider_unavailable"
        for event in events
    )
    assert not any(
        event.get("type")
        in {
            "implementation_started",
            "implementation_finished",
            "provider_invocation_committed",
        }
        for event in events
    )
    assert len(daemon.task_queue.entries) == 1
    [queue_entry] = daemon.task_queue.entries.values()
    assert queue_entry.is_cooled_down() is True
    assert queue_entry.notes == "primary provider unavailable"
