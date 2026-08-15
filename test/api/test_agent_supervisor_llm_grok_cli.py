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

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self):
            return self.returncode

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
        captured["prompt"] = prompt_path.read_text(encoding="utf-8")
        return FakeProcess(cmd, **kwargs)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair the board"))
    monkeypatch.setattr(grok_cli_runner.subprocess, "Popen", fake_popen)

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
    assert cmd[cmd.index("--output-format") + 1] == "streaming-json"
    assert "--always-approve" in cmd


def test_grok_agent_runner_emits_private_body_free_failure_receipt(
    monkeypatch,
    tmp_path,
) -> None:
    class FailedProcess:
        stdout = io.StringIO("")
        stderr = io.StringIO(
            '{"error":{"message":"authentication failed",'
            '"api_key":"xai-private-sentinel-4427"},"http_status":401}\n'
        )

        @staticmethod
        def wait():
            return 19

    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(
        grok_cli_runner.TRUSTED_FAILURE_RECEIPT_FD_ENV,
        str(write_fd),
    )
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda cmd, **kwargs: FailedProcess(),
    )
    try:
        result = grok_cli_runner.main(
            [
                "--workspace",
                str(tmp_path),
                "--grok-bin",
                "/bin/true",
                "--mode",
                "agent",
            ]
        )
        receipt = os.read(read_fd, 4096).decode("utf-8")
    finally:
        os.close(read_fd)
        try:
            os.close(write_fd)
        except OSError:
            pass

    parsed = __import__(
        "ipfs_accelerate_py.llm_router",
        fromlist=["parse_agent_cli_failure_receipt"],
    ).parse_agent_cli_failure_receipt(receipt)
    assert result == 19
    assert parsed is not None
    assert parsed[0].kind.value == "authentication_failure"
    assert parsed[2].value == "no_activity"
    assert "xai-private-sentinel-4427" not in receipt
