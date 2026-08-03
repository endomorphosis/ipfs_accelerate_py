from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import ipfs_accelerate_py.llm_router as llm_router
import pytest
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
    monkeypatch.setenv(implementation_daemon._CODEX_MODEL_ENV, "wrong-model")
    monkeypatch.setenv(
        implementation_daemon._CODEX_REASONING_EFFORT_ENV,
        "xhigh",
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
    assert fallback_command[fallback_command.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback_command
    assert fallback_command[-1] == "-"


def test_grok_model_is_pinned_despite_model_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in (
        implementation_daemon._GROK_MODEL_ENV,
        "GROK_CLI_MODEL",
        "GROK_MODEL",
        "ipfs_accelerate_py_GROK_CLI_MODEL",
    ):
        monkeypatch.setenv(name, "grok-unreviewed")
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )

    command = implementation_daemon._grok_cli_command(
        workspace_path=tmp_path,
    )

    assert command[command.index("--model") + 1] == "grok-4.5"


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


def test_default_implementation_provider_fails_closed_without_grok(
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

    with pytest.raises(RuntimeError, match="automatic implementation route requires"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_codex_before_copilot_uses_local_default_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(implementation_daemon._CODEX_MODEL_ENV, raising=False)

    command = implementation_daemon._copilot_fallback_command(
        codex="/opt/providers/codex",
        copilot="/opt/providers/copilot",
        workspace_path=tmp_path,
    )

    # Positional arguments after the embedded shell program are stable inputs
    # consumed by that program; the Codex model is its fourth argument.
    assert command[4:8] == [
        "/opt/providers/codex",
        "/opt/providers/copilot",
        str(tmp_path),
        "gpt-5.6-sol",
    ]


def test_unauthenticated_grok_binary_fails_closed_before_dispatch(
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

    with pytest.raises(RuntimeError, match="automatic implementation route requires"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_grok_provider_construction_failure_fails_closed(
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

    with pytest.raises(RuntimeError, match="automatic implementation route requires"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_verified_grok_quota_exhaustion_runs_codex_with_same_prompt(
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
            kwargs["stderr"].write(b'{"reason":"usage_pool_exhausted"}\n')
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


@pytest.mark.parametrize(
    "reason",
    ["usage_pool_exhausted", "usage_limit_reached"],
)
@pytest.mark.parametrize(
    "field",
    ["code", "kind", "reason", "reason_code", "type"],
)
def test_exact_structured_grok_quota_reasons_allow_fallback(
    reason: str,
    field: str,
) -> None:
    raw = json.dumps({field: reason}).encode("utf-8")

    assert grok_cli_runner._stderr_allows_codex_fallback(
        raw,
        overflow=False,
    )


def test_exact_grok_build_402_balance_error_allows_fallback() -> None:
    raw = (
        "Internal error: "
        + json.dumps(
            {
                "message": (
                    "API error (status 402 Payment Required): "
                    "Grok Build usage balance exhausted"
                ),
                "http_status": 402,
            }
        )
    ).encode("utf-8")

    assert grok_cli_runner._stderr_allows_codex_fallback(
        raw,
        overflow=False,
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"reason":"rate_limit_reached"}',
        b'{"reason":"quota_exhausted"}',
        b'usage_pool_exhausted',
        b'authentication failed',
        b'network connection failed',
        b'request timed out',
        b'429',
        b'402',
        b'generic provider failure',
        b'log prefix {"reason":"usage_pool_exhausted"}',
        b'Internal error: {"reason":"usage_limit_reached"}',
    ],
)
def test_unreviewed_grok_failures_do_not_allow_fallback(raw: bytes) -> None:
    assert not grok_cli_runner._stderr_allows_codex_fallback(
        raw,
        overflow=False,
    )


def test_grok_stderr_overflow_disables_fallback() -> None:
    raw = b'{"reason":"usage_pool_exhausted"}'

    assert not grok_cli_runner._stderr_allows_codex_fallback(
        raw,
        overflow=True,
    )


def test_stdout_quota_text_cannot_trigger_codex_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fallback = implementation_daemon._codex_implementation_command(
        codex="/opt/providers/codex",
        workspace_path=tmp_path,
        quota_fallback=True,
    )
    command = [
        "--workspace",
        str(tmp_path),
        "--grok-bin",
        "/opt/providers/grok",
        "--model",
        "grok-unreviewed",
        "--codex-fallback-command-json",
        json.dumps(fallback),
    ]
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(
            argv,
            23,
            stdout=b'{"reason":"usage_pool_exhausted"}',
        )

    monkeypatch.setenv("GROK_CLI_MODEL", "grok-unreviewed-environment")
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("prompt"))
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    assert grok_cli_runner.main(command) == 23
    assert len(calls) == 1
    grok_argv, grok_kwargs = calls[0]
    assert grok_argv[grok_argv.index("--model") + 1] == "grok-4.5"
    grok_env = grok_kwargs["env"]
    assert isinstance(grok_env, dict)
    assert "GROK_CLI_MODEL" not in grok_env


@pytest.mark.parametrize(
    ("model", "reasoning"),
    [
        ("gpt-5.6-sol", 'model_reasoning_effort="medium"'),
        ("gpt-5.6-terra", 'model_reasoning_effort="high"'),
    ],
)
def test_runner_rejects_non_exact_codex_fallback(
    tmp_path: Path,
    model: str,
    reasoning: str,
) -> None:
    fallback = implementation_daemon._codex_implementation_command(
        codex="/opt/providers/codex",
        workspace_path=tmp_path,
        quota_fallback=True,
    )
    fallback[fallback.index("-m") + 1] = model
    reasoning_index = fallback.index('model_reasoning_effort="medium"')
    fallback[reasoning_index] = reasoning

    with pytest.raises(ValueError):
        grok_cli_runner._parse_codex_fallback_command(json.dumps(fallback))


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
