from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import ipfs_accelerate_py.llm_router as llm_router
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
    implementation_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def _daemon(root: Path) -> TodoImplementationDaemon:
    board = root / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=board,
        state_path=root / "state" / "task-state.json",
        strategy_path=root / "state" / "strategy.json",
        events_path=root / "state" / "events.jsonl",
        repo_root=root,
    )


def _clear_provider_overrides(monkeypatch) -> None:
    monkeypatch.delenv(
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV,
        raising=False,
    )
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ROUTE_ENABLED_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV,
        raising=False,
    )


def _prompt_task(**overrides) -> PortalTask:
    payload = {
        "task_id": "ASE-001",
        "title": "Implement a prompt-only supervisor entrypoint",
        "status": "ready",
        "completion": "manual",
        "priority": "high",
        "track": "entrypoints",
        "outputs": ["ipfs_accelerate_py/agent_supervisor/entrypoints/api.py"],
        "validation": ["python -m pytest test/api/test_prompt_entrypoint.py -q"],
        "acceptance": "A prompt can launch the inferred supervisor workflow.",
    }
    payload.update(overrides)
    return PortalTask(**payload)


def test_default_implementation_provider_prefers_grok(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_command",
        lambda *, workspace_path: [
            "/opt/providers/grok-runner",
            str(workspace_path),
        ],
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[:2] == [
        "/opt/providers/grok-runner",
        str(tmp_path.resolve()),
    ]
    fallback_index = command.index("--codex-fallback-command-json")
    fallback_command = json.loads(command[fallback_index + 1])
    assert fallback_command[:2] == ["/opt/providers/codex", "exec"]
    assert fallback_command[-1] == "-"


def test_ordinary_prompt_task_uses_grok_with_codex_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_command",
        lambda *, workspace_path: [
            "/opt/providers/grok-runner",
            str(workspace_path),
        ],
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    daemon = _daemon(tmp_path)
    task = _prompt_task()

    assert daemon._production_provider_route_enabled(task) is False
    command = daemon._build_implementation_command(tmp_path, task=task)

    assert command[:2] == [
        "/opt/providers/grok-runner",
        str(tmp_path.resolve()),
    ]
    fallback_index = command.index("--codex-fallback-command-json")
    fallback_command = json.loads(command[fallback_index + 1])
    assert fallback_command[:2] == ["/opt/providers/codex", "exec"]


def test_systemd_minimal_path_still_selects_user_local_grok(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    fake_home = tmp_path / "home"
    fake_grok = fake_home / ".local" / "bin" / "grok"
    fake_grok.parent.mkdir(parents=True)
    fake_grok.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_grok.chmod(0o700)
    auth_path = fake_home / ".grok" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv(
        "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    )
    for name in (
        "ipfs_accelerate_py_GROK_CLI_CMD",
        "IPFS_ACCELERATE_PY_GROK_CLI_CMD",
        "IPFS_DATASETS_PY_GROK_CLI_CMD",
        implementation_daemon._GROK_BIN_ENV,
        "GROK_CLI_CMD",
        "GROK_BIN",
        "GROK_HOME",
        "XAI_API_KEY",
        "ipfs_accelerate_py_XAI_API_KEY",
        "IPFS_ACCELERATE_PY_XAI_API_KEY",
        "IPFS_DATASETS_PY_XAI_API_KEY",
        "GROK_AUTH_PROVIDER_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    real_which = implementation_daemon.shutil.which
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: (
            "/usr/local/bin/codex"
            if name == "codex"
            else real_which(name)
        ),
    )
    daemon = _daemon(tmp_path)

    command = daemon._build_implementation_command(
        tmp_path,
        task=_prompt_task(),
    )

    assert command[0] == implementation_daemon.sys.executable
    assert command[1].endswith("grok_cli_runner.py")
    assert command[command.index("--grok-bin") + 1] == str(fake_grok)
    fallback_index = command.index("--codex-fallback-command-json")
    assert json.loads(command[fallback_index + 1])[:2] == [
        "/usr/local/bin/codex",
        "exec",
    ]


def test_default_implementation_provider_falls_back_to_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[:5] == [
        "/opt/providers/codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(tmp_path.resolve()),
    ]
    assert command[-1] == "-"


def test_unauthenticated_grok_binary_falls_back_to_codex_before_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        llm_router,
        "_grok_cli_auth_available",
        lambda: False,
    )
    monkeypatch.setattr(
        llm_router,
        "get_llm_provider",
        lambda _provider: (_ for _ in ()).throw(
            AssertionError("provider construction must follow authentication")
        ),
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[0:2] == ["/opt/providers/codex", "exec"]


def test_grok_provider_construction_failure_falls_back_to_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        llm_router,
        "_grok_cli_auth_available",
        lambda: True,
    )
    monkeypatch.setattr(
        llm_router,
        "get_llm_provider",
        lambda _provider: (_ for _ in ()).throw(
            RuntimeError("provider registry unavailable")
        ),
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[0:2] == ["/opt/providers/codex", "exec"]


def test_default_grok_runtime_failure_runs_codex_with_same_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    prompt = "repair the failed implementation"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        if len(calls) == 1:
            prompt_path = Path(argv[argv.index("--prompt-file") + 1])
            assert prompt_path.read_text(encoding="utf-8") == prompt
            return subprocess.CompletedProcess(argv, 23)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    returncode = grok_cli_runner.main(command[2:])

    assert returncode == 0
    assert len(calls) == 2
    assert calls[0][0][0] == "/opt/providers/grok"
    assert calls[1][0][:2] == ["/opt/providers/codex", "exec"]
    assert calls[1][1]["cwd"] == tmp_path.resolve()
    assert calls[1][1]["input"] == prompt
    assert calls[1][1]["text"] is True


def test_explicit_grok_runtime_failure_does_not_fall_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV,
        "grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 29)

    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        io.StringIO("use Grok or fail"),
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    assert "--codex-fallback-command-json" not in command
    assert grok_cli_runner.main(command[2:]) == 29
    assert len(calls) == 1
    assert calls[0][0] == "/opt/providers/grok"


def test_launch_defaults_do_not_override_grok_first_provider_inference() -> None:
    daemon_args = implementation_daemon.parse_args([])
    supervisor_args = implementation_supervisor.parse_args([])

    assert daemon_args.implementation_command == ""
    assert supervisor_args.implementation_command == ""
