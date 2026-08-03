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
    GROK_QUOTA_EXHAUSTED_FALLBACK_POLICY,
    IMPLEMENTATION_PROVIDER_ENV,
    PROVIDER_FALLBACK_POLICY_ENV,
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
    assert command[command.index("--fallback-policy") + 1] == "any_failure"
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
        lambda: (_ for _ in ()).throw(
            AssertionError("grok-codex must not select Copilot")
        ),
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


def test_grok_quota_only_policy_fails_closed_when_grok_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, "grok-codex")
    monkeypatch.setenv(
        PROVIDER_FALLBACK_POLICY_ENV,
        GROK_QUOTA_EXHAUSTED_FALLBACK_POLICY,
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda name: "/provider/codex" if name == "codex" else None,
    )

    with pytest.raises(
        RuntimeError,
        match="allowed only after confirmed Grok quota",
    ):
        daemon._build_implementation_command(tmp_path)


def test_grok_quota_only_policy_forwards_terra_model_and_medium_reasoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, "grok-codex")
    monkeypatch.setenv(
        PROVIDER_FALLBACK_POLICY_ENV,
        GROK_QUOTA_EXHAUSTED_FALLBACK_POLICY,
    )
    monkeypatch.setenv(
        implementation_daemon_module._CODEX_MODEL_ENV,
        "gpt-5.6-terra",
    )
    monkeypatch.setenv(
        implementation_daemon_module._CODEX_REASONING_EFFORT_ENV,
        "medium",
    )
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
    fallback = _json_command(command, "--fallback-command-json")

    assert command[command.index("--fallback-policy") + 1] == (
        GROK_QUOTA_EXHAUSTED_FALLBACK_POLICY
    )
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback


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

name, record_path, returncode, *messages = sys.argv[1:]
prompt = sys.stdin.read()
pathlib.Path(record_path).write_text(
    json.dumps({"cwd": os.getcwd(), "prompt": prompt}),
    encoding="utf-8",
)
print(f"{name}-stream")
for message in messages:
    if message.startswith("stdout::"):
        print(message.removeprefix("stdout::"))
    else:
        print(message, file=sys.stderr)
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
    fallback_policy: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
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
    ]
    if fallback_policy is not None:
        command.extend(["--fallback-policy", fallback_policy])
    return subprocess.run(
        command,
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


@pytest.mark.parametrize(
    "quota_message",
    (
        "Error: xAI API usage quota exhausted.",
        '{"error":{"type":"insufficient_quota","message":"no capacity"}}',
    ),
)
def test_quota_only_runner_falls_back_only_on_confirmed_grok_quota(
    quota_message: str,
    tmp_path: Path,
) -> None:
    worker = tmp_path / "fake_provider.py"
    _write_fake_provider(worker)
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"

    result = _run_fallback_runner(
        workspace=tmp_path,
        prompt="implement after quota exhaustion\n",
        primary_command=[
            sys.executable,
            str(worker),
            "grok",
            str(primary_record),
            "19",
            quota_message,
        ],
        fallback_command=[
            sys.executable,
            str(worker),
            "codex",
            str(fallback_record),
            "0",
        ],
        fallback_policy=GROK_QUOTA_EXHAUSTED_FALLBACK_POLICY,
    )

    assert result.returncode == 0
    assert fallback_record.is_file()
    assert "quota exhaustion confirmed" in result.stderr
    assert "falling back to codex" in result.stderr


@pytest.mark.parametrize(
    ("failure_message", "expected_kind"),
    (
        (
            "Error: authentication failed: invalid XAI_API_KEY",
            "authentication_failure",
        ),
        (
            "Error: authentication failed; xAI API quota exhausted.",
            "authentication_failure",
        ),
        ("Error: request timed out", "timeout"),
        ("Error: connection reset by peer", "transport_failure"),
        ("Error: rate limit exceeded", "generic_nonzero_exit"),
        ("Error: provider failed with status 500", "generic_nonzero_exit"),
        ('{"error":{"type":"quota_exhausted"}', "generic_nonzero_exit"),
        ("\ufffdError: xAI API quota exhausted.", "malformed_output"),
        ("pytest: 1 failed, 12 passed", "generic_nonzero_exit"),
        (
            "task failed: expected the string 'quota exhausted'",
            "generic_nonzero_exit",
        ),
    ),
)
def test_quota_only_runner_preserves_every_non_quota_grok_failure(
    failure_message: str,
    expected_kind: str,
    tmp_path: Path,
) -> None:
    worker = tmp_path / "fake_provider.py"
    _write_fake_provider(worker)
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"

    result = _run_fallback_runner(
        workspace=tmp_path,
        prompt="do not route ordinary failures to codex\n",
        primary_command=[
            sys.executable,
            str(worker),
            "grok",
            str(primary_record),
            "19",
            failure_message,
        ],
        fallback_command=[
            sys.executable,
            str(worker),
            "codex",
            str(fallback_record),
            "0",
        ],
        fallback_policy=GROK_QUOTA_EXHAUSTED_FALLBACK_POLICY,
    )

    assert result.returncode == 19
    assert not fallback_record.exists()
    assert "fallback suppressed by quota-only policy" in result.stderr
    assert expected_kind in result.stderr


def test_quota_only_runner_does_not_fallback_when_grok_cannot_launch(
    tmp_path: Path,
) -> None:
    fallback_record = tmp_path / "fallback.json"
    worker = tmp_path / "fake_provider.py"
    _write_fake_provider(worker)

    result = _run_fallback_runner(
        workspace=tmp_path,
        prompt="grok must launch first\n",
        primary_command=[str(tmp_path / "missing-grok")],
        fallback_command=[
            sys.executable,
            str(worker),
            "codex",
            str(fallback_record),
            "0",
        ],
        fallback_policy=GROK_QUOTA_EXHAUSTED_FALLBACK_POLICY,
    )

    assert result.returncode == 127
    assert not fallback_record.exists()
    assert "launch_failure" in result.stderr


def test_quota_only_runner_never_trusts_quota_json_from_task_stdout(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "fake_provider.py"
    _write_fake_provider(worker)
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"

    result = _run_fallback_runner(
        workspace=tmp_path,
        prompt="task output is not provider quota evidence\n",
        primary_command=[
            sys.executable,
            str(worker),
            "grok",
            str(primary_record),
            "19",
            'stdout::{"error":{"type":"quota_exhausted"}}',
            "ordinary Grok task failure",
        ],
        fallback_command=[
            sys.executable,
            str(worker),
            "codex",
            str(fallback_record),
            "0",
        ],
        fallback_policy=GROK_QUOTA_EXHAUSTED_FALLBACK_POLICY,
    )

    assert result.returncode == 19
    assert '"type":"quota_exhausted"' in result.stdout
    assert not fallback_record.exists()
    assert "generic_nonzero_exit" in result.stderr
