from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import time
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
    monkeypatch.setenv(implementation_daemon._GROK_MODEL_ENV, "grok-older")
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
        lambda *, workspace_path, model_override="": [
            "/opt/providers/grok-runner",
            str(workspace_path),
            "--model",
            model_override,
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
    assert command[command.index("--model") + 1] == "grok-4.5"
    assert "--codex-fallback-command-json" not in command


def test_ordinary_prompt_task_uses_pinned_grok_without_raw_fallback(
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
        lambda *, workspace_path, model_override="": [
            "/opt/providers/grok-runner",
            str(workspace_path),
            "--model",
            model_override,
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
    assert command[command.index("--model") + 1] == "grok-4.5"
    assert "--codex-fallback-command-json" not in command


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
        lambda name: ("/usr/local/bin/codex" if name == "codex" else real_which(name)),
    )
    daemon = _daemon(tmp_path)

    command = daemon._build_implementation_command(
        tmp_path,
        task=_prompt_task(),
    )

    assert command[0] == implementation_daemon.sys.executable
    assert command[1].endswith("grok_cli_runner.py")
    assert command[command.index("--grok-bin") + 1] == str(fake_grok)
    assert command[command.index("--model") + 1] == "grok-4.5"
    assert "--codex-fallback-command-json" not in command


def test_default_provider_does_not_predispatch_codex_when_grok_is_unavailable(
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

    with pytest.raises(RuntimeError, match="verified Grok quota-exhaustion"):
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


def test_unauthenticated_grok_binary_does_not_authorize_codex(
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

    with pytest.raises(RuntimeError, match="verified Grok quota-exhaustion"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_grok_provider_construction_failure_does_not_authorize_codex(
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

    with pytest.raises(RuntimeError, match="verified Grok quota-exhaustion"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_raw_grok_failure_never_dispatches_a_second_provider(
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
        prompt_path = Path(argv[argv.index("--prompt-file") + 1])
        assert prompt_path.read_text(encoding="utf-8") == prompt
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    returncode = grok_cli_runner.main(command[2:])

    assert returncode == 1
    assert len(calls) == 1
    assert calls[0][0][0] == "/opt/providers/grok"
    assert calls[0][0][calls[0][0].index("--model") + 1] == "grok-4.5"


def _native_grok_failure_event(error_type: str, message: str = "") -> str:
    return json.dumps(
        {
            "method": "_x.ai/session/update",
            "params": {
                "update": {
                    "sessionUpdate": "retry_state",
                    "type": "failed",
                    "error_type": error_type,
                    "message": message,
                }
            },
        }
    )


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        ("usage_pool_exhausted", "usage_pool_exhausted"),
        ("usage_limit_reached", "usage_limit_reached"),
        ("rate_limited", "rate_limited"),
        ("unauthorized", "unauthorized"),
        ("network_error", "network_error"),
    ],
)
def test_native_grok_failure_classifier_is_exact(
    error_type: str,
    expected: str,
) -> None:
    assert (
        grok_cli_runner._grok_failure_type_from_stream_event(
            _native_grok_failure_event(error_type)
        )
        == expected
    )


def test_legacy_native_402_balance_event_is_typed_quota() -> None:
    event = _native_grok_failure_event(
        "api",
        "API error (status 402 Payment Required): "
        "Grok Build usage balance exhausted",
    )
    assert (
        grok_cli_runner._grok_failure_type_from_stream_event(event)
        == grok_cli_runner.GROK_QUOTA_FAILURE_MARKER
    )


def test_documented_streaming_error_shape_is_typed_quota() -> None:
    event = json.dumps(
        {
            "type": "error",
            "message": (
                "API error (status 402 Payment Required): "
                "Grok Build usage balance exhausted"
            ),
        }
    )

    assert (
        grok_cli_runner._grok_failure_type_from_stream_event(event)
        == grok_cli_runner.GROK_QUOTA_FAILURE_MARKER
    )


def test_live_stream_capture_observes_documented_quota_event(
    capsys,
) -> None:
    event = json.dumps(
        {
            "type": "error",
            "message": (
                "API error (status 402 Payment Required): "
                "Grok Build usage balance exhausted"
            ),
        }
    )
    command = [
        grok_cli_runner.sys.executable,
        "-c",
        (
            "import sys; "
            f"print({event!r}); "
            "print('bounded diagnostic', file=sys.stderr); "
            "raise SystemExit(17)"
        ),
    ]

    returncode, failure_types = grok_cli_runner._run_grok_with_typed_failure_capture(
        command,
        env=grok_cli_runner.os.environ.copy(),
    )

    captured = capsys.readouterr()
    assert returncode == 17
    assert failure_types == {grok_cli_runner.GROK_QUOTA_FAILURE_MARKER}
    assert event in captured.out
    assert "bounded diagnostic" in captured.err


@pytest.mark.timeout(5)
def test_stream_capture_does_not_wait_for_detached_descendant_pipe_owner(
    tmp_path: Path,
    capsys,
) -> None:
    orphan_pid_path = tmp_path / "escaped-descendant.pid"
    child_script = (
        "import pathlib, subprocess, sys; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True); "
        f"pathlib.Path({str(orphan_pid_path)!r}).write_text(str(child.pid)); "
        "print('direct child complete'); "
        "raise SystemExit(23)"
    )
    started = time.monotonic()
    try:
        returncode, failure_types = (
            grok_cli_runner._run_grok_with_typed_failure_capture(
                [grok_cli_runner.sys.executable, "-c", child_script],
                env=grok_cli_runner.os.environ.copy(),
            )
        )
    finally:
        if orphan_pid_path.is_file():
            orphan_pid = int(orphan_pid_path.read_text(encoding="utf-8"))
            try:
                os.kill(orphan_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    elapsed = time.monotonic() - started
    captured = capsys.readouterr()
    assert returncode == 23
    assert failure_types == set()
    assert "direct child complete" in captured.out
    assert elapsed < 3.0


@pytest.mark.timeout(5)
def test_stream_capture_keeps_draining_when_destination_is_broken(
    monkeypatch,
) -> None:
    class BrokenDestination(io.StringIO):
        def write(self, value: str) -> int:
            raise BrokenPipeError("destination closed")

    event = json.dumps(
        {
            "type": "error",
            "message": (
                "API error (status 402 Payment Required): "
                "Grok Build usage balance exhausted"
            ),
        }
    )
    child_script = (
        "import sys; "
        f"print({event!r}); "
        "sys.stdout.write('x' * (256 * 1024)); "
        "sys.stdout.flush(); "
        "raise SystemExit(17)"
    )
    monkeypatch.setattr(grok_cli_runner.sys, "stdout", BrokenDestination())

    started = time.monotonic()
    returncode, failure_types = grok_cli_runner._run_grok_with_typed_failure_capture(
        [grok_cli_runner.sys.executable, "-c", child_script],
        env=grok_cli_runner.os.environ.copy(),
    )

    assert returncode == 17
    assert failure_types == {grok_cli_runner.GROK_QUOTA_FAILURE_MARKER}
    assert time.monotonic() - started < 3.0


def test_runner_rejects_deprecated_raw_fallback_before_provider_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("deprecated raw fallback must fail before dispatch")
        ),
    )

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            grok_cli_runner.sys.executable,
            "--model",
            grok_cli_runner.DEFAULT_GROK_MODEL,
            "--codex-fallback-command-json",
            json.dumps(["/opt/providers/codex", "exec", "-"]),
        ]
    )

    assert returncode == 2


@pytest.mark.parametrize(
    "event",
    [
        '{"error_type":"usage_pool_exhausted"}',
        json.dumps(
            {
                "method": "_x.ai/session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "type": "failed",
                        "error_type": "usage_pool_exhausted",
                    }
                },
            }
        ),
        _native_grok_failure_event("api", "quota exhausted"),
        _native_grok_failure_event("api", "HTTP 429 rate limit"),
        _native_grok_failure_event("usage_pool_exhausted"),
        _native_grok_failure_event("usage_limit_reached"),
        json.dumps({"type": "error", "message": "quota exhausted"}),
        json.dumps({"type": "error", "message": "Payment required"}),
        json.dumps(
            {
                "type": "error",
                "message": "HTTP 429: Grok Build usage balance exhausted",
            }
        ),
        json.dumps(
            {
                "type": "text",
                "data": json.dumps(
                    {
                        "type": "error",
                        "message": (
                            "API error (status 402 Payment Required): "
                            "Grok Build usage balance exhausted"
                        ),
                    }
                ),
            }
        ),
        "not-json",
    ],
)
def test_untrusted_or_inexact_quota_text_is_not_authoritative(event: str) -> None:
    assert grok_cli_runner._grok_failure_type_from_stream_event(event) not in (
        grok_cli_runner.GROK_QUOTA_ERROR_TYPES
    )


def test_explicit_grok_runtime_failure_does_not_fall_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setenv(implementation_daemon._GROK_MODEL_ENV, "grok-older")
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
    assert command[command.index("--model") + 1] == "grok-4.5"
    assert grok_cli_runner.main(command[2:]) == 29
    assert len(calls) == 1
    assert calls[0][0] == "/opt/providers/grok"


def test_launch_defaults_do_not_override_grok_first_provider_inference() -> None:
    daemon_args = implementation_daemon.parse_args([])
    supervisor_args = implementation_supervisor.parse_args([])

    assert daemon_args.implementation_command == ""
    assert supervisor_args.implementation_command == ""
