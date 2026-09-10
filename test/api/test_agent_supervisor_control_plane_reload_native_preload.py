"""Control-plane supervisor reload must preload sealed DuckDB like the daemon."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    IMPLEMENTATION_SUPERVISOR_MODULE_SENTINEL,
    ORDINARY_IMPLEMENTATION_SUPERVISOR_BOOTSTRAP,
    TodoImplementationSupervisor,
    TodoSupervisorConfig,
)


def test_control_plane_reload_without_wrapper_preloads_sealed_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )
    calls: list[tuple[str, list[str]]] = []

    class ExecRequested(Exception):
        pass

    def fake_execv(executable: str, arguments: list[str]) -> None:
        calls.append((executable, list(arguments)))
        raise ExecRequested

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.os.execv",
        fake_execv,
    )
    monkeypatch.setattr(sys, "argv", ["implementation-supervisor", "--state-prefix", "lane-2"])

    with pytest.raises(ExecRequested):
        supervisor._reload_for_control_plane_update()

    assert calls == [
        (
            sys.executable,
            [
                sys.executable,
                "-c",
                ORDINARY_IMPLEMENTATION_SUPERVISOR_BOOTSTRAP,
                IMPLEMENTATION_SUPERVISOR_MODULE_SENTINEL,
                "--state-prefix",
                "lane-2",
            ],
        )
    ]
    assert "preload_sealed_native_dependency_from_environment" in (
        ORDINARY_IMPLEMENTATION_SUPERVISOR_BOOTSTRAP
    )


def test_control_plane_reload_keeps_configured_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    wrapper = repo / "bin" / "supervisor-wrapper.py"
    wrapper.parent.mkdir()
    wrapper.write_text("pass\n", encoding="utf-8")
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            supervisor_script_path=Path("bin/supervisor-wrapper.py"),
        )
    )
    calls: list[tuple[str, list[str]]] = []

    class ExecRequested(Exception):
        pass

    def fake_execv(executable: str, arguments: list[str]) -> None:
        calls.append((executable, list(arguments)))
        raise ExecRequested

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.os.execv",
        fake_execv,
    )
    monkeypatch.setattr(sys, "argv", ["implementation-supervisor", "--state-prefix", "lane-1"])

    with pytest.raises(ExecRequested):
        supervisor._reload_for_control_plane_update()

    assert calls == [
        (
            sys.executable,
            [
                sys.executable,
                str(wrapper.resolve()),
                "--state-prefix",
                "lane-1",
            ],
        )
    ]
