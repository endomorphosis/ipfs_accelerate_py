"""End-to-end supervisor subprocess coverage for the Grok CLI provider."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LlmRouterInvocation,
    call_llm_router,
)


def test_supervisor_child_routes_grok_through_datasets_router(monkeypatch, tmp_path) -> None:
    fake_grok = tmp_path / "grok"
    fake_grok.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
prompt_path = pathlib.Path(args[args.index("--prompt-file") + 1])
prompt = prompt_path.read_text(encoding="utf-8")
model = args[args.index("--model") + 1]
print(json.dumps({
    "text": f"supervisor:{model}:{prompt}",
    "stopReason": "EndTurn",
    "sessionId": "supervisor-session",
    "requestId": "supervisor-request",
}))
""",
        encoding="utf-8",
    )
    fake_grok.chmod(0o700)

    monkeypatch.setenv("IPFS_DATASETS_PY_GROK_CLI_CMD", str(fake_grok))
    monkeypatch.setenv("IPFS_DATASETS_PY_ROUTER_CACHE", "0")
    monkeypatch.setenv("IPFS_DATASETS_PY_ROUTER_RESPONSE_CACHE", "0")

    config = LlmRouterInvocation(
        repo_root=Path(__file__).resolve().parents[2],
        provider="grok",
        model_name="grok-4.5",
        allow_local_fallback=False,
        timeout_seconds=15,
        timeout_grace_seconds=2,
        max_new_tokens=16,
        python_executable=sys.executable,
        required_effective_providers=("grok_cli",),
    )

    assert call_llm_router("child-smoke", config) == "supervisor:grok-4.5:child-smoke"


def test_grok_agent_runner_forwards_resolved_launch_policy(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
        captured["prompt"] = prompt_path.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair the board"))
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--model",
            "grok-4.5",
            "--max-turns",
            "1234",
            "--permission-mode",
            "acceptEdits",
            "--mode",
            "agent",
        ]
    )

    assert result == 0
    assert captured["prompt"] == "repair the board"
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--model") + 1] == "grok-4.5"
    assert cmd[cmd.index("--max-turns") + 1] == "1234"
    assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
    assert cmd[cmd.index("--output-format") + 1] == "plain"
    assert "--always-approve" in cmd


@pytest.mark.parametrize(
    ("diagnostic", "creates_result"),
    [
        ("ERROR insufficient_quota: quota exhausted", True),
        ("429 Too Many Requests: rate limit exceeded", False),
        ("401 authentication required", False),
        ("malformed provider response", False),
        ("debug: user task says insufficient_quota must be handled", False),
        ("tool stderr: test fixture insufficient_quota", False),
    ],
)
def test_grok_agent_runner_emits_only_typed_hard_quota_result(
    monkeypatch,
    tmp_path,
    capsys,
    diagnostic,
    creates_result,
) -> None:
    fake_grok = tmp_path / "grok"
    fake_grok.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"print({diagnostic!r}, file=sys.stderr, flush=True)\n"
        "raise SystemExit(23)\n",
        encoding="utf-8",
    )
    fake_grok.chmod(0o700)
    result_path = tmp_path / "capacity.json"
    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        io.StringIO("secret implementation prompt"),
    )
    original_cwd = Path.cwd()
    try:
        returncode = grok_cli_runner.main(
            [
                "--workspace",
                str(tmp_path),
                "--grok-bin",
                str(fake_grok),
                "--model",
                "grok-4.5",
                "--capacity-result-path",
                str(result_path),
            ]
        )
    finally:
        os.chdir(original_cwd)

    assert returncode == (
        grok_cli_runner.HARD_QUOTA_EXIT_CODE if creates_result else 23
    )
    assert diagnostic in capsys.readouterr().err
    assert result_path.exists() is creates_result
    if creates_result:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["schema"] == grok_cli_runner.CAPACITY_RESULT_SCHEMA
        assert payload["provider"] == "grok_cli"
        assert payload["model"] == "grok-4.5"
        assert payload["reason"] == "provider_quota_exhausted"
        assert payload["reason_codes"] == [
            "capacity_unavailable",
            "quota_exhausted",
        ]
        assert payload["returncode"] == grok_cli_runner.HARD_QUOTA_EXIT_CODE
        assert payload["provider_returncode"] == 23
        assert "secret implementation prompt" not in json.dumps(payload)


def test_grok_agent_runner_reserves_hard_quota_exit_code(
    monkeypatch,
    tmp_path,
) -> None:
    fake_grok = tmp_path / "grok"
    fake_grok.write_text(
        "#!/usr/bin/env python3\n"
        "raise SystemExit(75)\n",
        encoding="utf-8",
    )
    fake_grok.chmod(0o700)
    result_path = tmp_path / "capacity.json"
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("prompt"))
    original_cwd = Path.cwd()
    try:
        returncode = grok_cli_runner.main(
            [
                "--workspace",
                str(tmp_path),
                "--grok-bin",
                str(fake_grok),
                "--model",
                "grok-4.5",
                "--capacity-result-path",
                str(result_path),
            ]
        )
    finally:
        os.chdir(original_cwd)

    assert returncode == 1
    assert not result_path.exists()


def test_grok_agent_runner_replaces_symlink_entry_not_its_target(
    monkeypatch,
    tmp_path,
) -> None:
    fake_grok = tmp_path / "grok"
    fake_grok.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('ERROR: quota exhausted', file=sys.stderr)\n"
        "raise SystemExit(9)\n",
        encoding="utf-8",
    )
    fake_grok.chmod(0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep-me", encoding="utf-8")
    result_path = tmp_path / "capacity.json"
    result_path.symlink_to(victim)
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("prompt"))
    original_cwd = Path.cwd()
    try:
        assert grok_cli_runner.main(
            [
                "--workspace",
                str(tmp_path),
                "--grok-bin",
                str(fake_grok),
                "--model",
                "grok-4.5",
                "--capacity-result-path",
                str(result_path),
            ]
        ) == grok_cli_runner.HARD_QUOTA_EXIT_CODE
    finally:
        os.chdir(original_cwd)

    assert victim.read_text(encoding="utf-8") == "keep-me"
    assert not result_path.is_symlink()
