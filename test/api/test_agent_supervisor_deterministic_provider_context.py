"""Provider-independent context compilation for typed local tasks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def test_deterministic_context_compilation_is_provider_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Deterministic Test")
    _git(repo, "config", "user.email", "deterministic@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Deterministic tasks\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## SCA-",
    )
    task = PortalTask(
        task_id="SCA-DET-001",
        title="Materialize deterministic evidence",
        status="ready",
        completion="manual",
        priority="P0",
        track="static-analysis",
        outputs=["artifact.json"],
        validation=["true"],
        acceptance="Materialize the reviewed evidence without a model.",
        metadata={
            "Provider role": "deterministic-only",
            "LLM context budget bytes": "30000",
        },
    )
    monkeypatch.setenv(
        implementation_daemon_module.IMPLEMENTATION_PROVIDER_ENV,
        "unsupported-model-provider",
    )
    monkeypatch.setenv(
        implementation_daemon_module._GROK_CONTEXT_WINDOW_ENV,
        "invalid",
    )
    monkeypatch.setenv(
        implementation_daemon_module._CODEX_CONTEXT_WINDOW_ENV,
        "invalid",
    )
    monkeypatch.setattr(
        daemon,
        "_provider_capacity_latch_states",
        lambda: pytest.fail("typed local context consulted provider state"),
    )

    base_budget = daemon._base_implementation_context_budget()
    local_window = base_budget.total_token_window

    assert daemon._implementation_context_window(task) == local_window
    daemon._require_primary_provider_readiness(task)
    result = daemon._compile_implementation_context(task, attempt=1)
    resolution = result.receipt.budget_resolution
    assert resolution.provider_context_window == (
        30_000 + base_budget.reserved_output_tokens + base_budget.reserved_tool_tokens
    )
    assert resolution.effective_input_limit == 30_000
