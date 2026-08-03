from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    GROK_CODEX_PROVIDER_ALIASES,
    IMPLEMENTATION_PROVIDER_ENV,
    TodoImplementationDaemon,
    _grok_cli_available,
    _provider_labels_from_implementation_command,
)

RUNNER_PATH = (
    Path(implementation_daemon_module.__file__).resolve().parents[1] / "provider_fallback_runner.py"
)


def _daemon(tmp_path: Path) -> TodoImplementationDaemon:
    todo_path = tmp_path / "todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
    )


def _json_command(command: list[str], flag: str) -> list[str]:
    return json.loads(command[command.index(flag) + 1])


def test_grok_codex_policy_builds_ordered_no_shell_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    grok_command = [
        sys.executable,
        "/provider/grok-runner.py",
        "--workspace",
        str(tmp_path),
    ]
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, "grok-codex")
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_command",
        lambda *, workspace_path: list(grok_command),
    )
    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda name: "/provider/codex" if name == "codex" else None,
    )

    command = daemon._build_implementation_command(tmp_path)

    assert command[:2] == [sys.executable, str(RUNNER_PATH)]
    assert "bash" not in command
    assert command[command.index("--primary-provider") + 1] == "grok"
    assert command[command.index("--fallback-provider") + 1] == "codex"
    assert _json_command(command, "--primary-command-json") == grok_command
    fallback = _json_command(command, "--fallback-command-json")
    assert fallback[:4] == [
        "/provider/codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
    ]
    assert fallback[4] == str(tmp_path)
    assert fallback[-1] == "-"
    assert set(_provider_labels_from_implementation_command(command)) == {
        "grok",
        "codex",
    }


@pytest.mark.parametrize("policy", sorted(GROK_CODEX_PROVIDER_ALIASES))
def test_grok_codex_policy_aliases_select_the_same_runner(
    policy: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, policy)
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_command",
        lambda *, workspace_path: ["/provider/grok", str(workspace_path)],
    )
    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda name: "/provider/codex" if name == "codex" else None,
    )

    command = daemon._build_implementation_command(tmp_path)

    assert command[:2] == [sys.executable, str(RUNNER_PATH)]


def test_grok_codex_policy_uses_direct_codex_when_grok_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, "grok-codex")
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "_copilot_has_auth",
        lambda: (_ for _ in ()).throw(AssertionError("grok-codex must not select Copilot")),
    )
    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda name: (
            "/provider/codex"
            if name == "codex"
            else "/provider/copilot"
            if name == "copilot"
            else None
        ),
    )

    command = daemon._build_implementation_command(tmp_path)

    assert command[:5] == [
        "/provider/codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(tmp_path),
    ]
    assert str(RUNNER_PATH) not in command
    assert "bash" not in command


def test_grok_codex_policy_fails_closed_without_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, "grok-codex")
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda _name: None,
    )

    with pytest.raises(RuntimeError, match="requires the Codex CLI"):
        daemon._build_implementation_command(tmp_path)


def test_grok_readiness_requires_binary_and_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_binary",
        lambda: "/provider/grok",
    )
    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: False)
    assert _grok_cli_available() is False

    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: True)
    assert _grok_cli_available() is True

    monkeypatch.setattr(implementation_daemon_module, "_grok_binary", lambda: None)
    assert _grok_cli_available() is False


def _write_fake_provider(path: Path) -> None:
    path.write_text(
        """\
from __future__ import annotations
import json
import os
import pathlib
import sys

name, record_path, returncode = sys.argv[1:]
prompt = sys.stdin.read()
pathlib.Path(record_path).write_text(
    json.dumps({"cwd": os.getcwd(), "prompt": prompt}),
    encoding="utf-8",
)
print(f"{name}-stream")
raise SystemExit(int(returncode))
""",
        encoding="utf-8",
    )


def _run_fallback_runner(
    *,
    workspace: Path,
    prompt: str,
    primary_command: list[str],
    fallback_command: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--workspace",
            str(workspace),
            "--primary-provider",
            "grok",
            "--fallback-provider",
            "codex",
            "--primary-command-json",
            json.dumps(primary_command),
            "--fallback-command-json",
            json.dumps(fallback_command),
        ],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )


def test_runner_does_not_invoke_fallback_after_primary_success(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "fake_provider.py"
    _write_fake_provider(worker)
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    prompt = "implement primary success\nwith exact stdin\n"

    result = _run_fallback_runner(
        workspace=tmp_path,
        prompt=prompt,
        primary_command=[
            sys.executable,
            str(worker),
            "grok",
            str(primary_record),
            "0",
        ],
        fallback_command=[
            sys.executable,
            str(worker),
            "codex",
            str(fallback_record),
            "0",
        ],
    )

    assert result.returncode == 0
    assert "grok-stream" in result.stdout
    assert "codex-stream" not in result.stdout
    assert json.loads(primary_record.read_text(encoding="utf-8")) == {
        "cwd": str(tmp_path),
        "prompt": prompt,
    }
    assert not fallback_record.exists()


@pytest.mark.parametrize("primary_launches", [True, False])
def test_runner_falls_back_with_identical_prompt_and_workspace(
    primary_launches: bool,
    tmp_path: Path,
) -> None:
    worker = tmp_path / "fake_provider.py"
    _write_fake_provider(worker)
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    prompt = "implement after primary failure\nwith exact stdin\n"
    primary_command = (
        [
            sys.executable,
            str(worker),
            "grok",
            str(primary_record),
            "19",
        ]
        if primary_launches
        else [str(tmp_path / "missing-grok")]
    )

    result = _run_fallback_runner(
        workspace=tmp_path,
        prompt=prompt,
        primary_command=primary_command,
        fallback_command=[
            sys.executable,
            str(worker),
            "codex",
            str(fallback_record),
            "0",
        ],
    )

    assert result.returncode == 0
    assert "codex-stream" in result.stdout
    assert "falling back to codex" in result.stderr
    fallback_payload = json.loads(fallback_record.read_text(encoding="utf-8"))
    assert fallback_payload == {
        "cwd": str(tmp_path),
        "prompt": prompt,
    }
    if primary_launches:
        assert json.loads(primary_record.read_text(encoding="utf-8")) == (fallback_payload)
    else:
        assert "grok provider could not launch" in result.stderr
        assert not primary_record.exists()
