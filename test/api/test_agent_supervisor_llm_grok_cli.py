"""End-to-end supervisor subprocess coverage for the Grok CLI provider."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

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

    def fake_run(cmd, *, env):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
        captured["prompt"] = prompt_path.read_text(encoding="utf-8")
        return 0, b"", 0, False

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair the board"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_bounded_stderr",
        fake_run,
    )

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


def test_grok_runner_bounds_child_stderr() -> None:
    emitted = grok_cli_runner.MAX_GROK_ERROR_BYTES + 4096
    returncode, retained, total, overflow = (
        grok_cli_runner._run_grok_with_bounded_stderr(
            [
                sys.executable,
                "-c",
                f"import sys; sys.stderr.buffer.write(b'x' * {emitted})",
            ],
            env=dict(os.environ),
        )
    )

    assert returncode == 0
    assert len(retained) == grok_cli_runner.MAX_GROK_ERROR_BYTES
    assert total == emitted
    assert overflow is True


def test_grok_runner_forbids_quota_receipt_for_oversized_stderr(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    quota_error = b"You've hit your usage limit."
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_bounded_stderr",
        lambda command, *, env: (
            grok_cli_runner.GROK_QUOTA_EXHAUSTED_EXIT_CODE,
            quota_error,
            grok_cli_runner.MAX_GROK_ERROR_BYTES + 1,
            True,
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--model",
            "grok-4.5",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "quota fallback forbidden" in captured.err
    assert grok_cli_runner.GROK_QUOTA_RECEIPT_SCHEMA not in captured.err
