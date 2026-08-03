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


def _run_supervised_fake_grok(
    monkeypatch,
    tmp_path,
    *,
    stream: bytes,
    returncode: int,
) -> tuple[int, bytes, dict[str, object]]:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(kwargs["env"])
            captured["stderr"] = kwargs["stderr"]
            self.stdout = io.BytesIO(stream)

        @staticmethod
        def wait(*_args, **_kwargs):
            return returncode

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda cmd, **kwargs: FakeProcess(cmd, **kwargs),
    )
    outer_command = grok_cli_runner.bind_grok_runner_command(
        [
            sys.executable,
            str(Path(grok_cli_runner.__file__).resolve()),
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--model",
            "grok-4.5",
            "--max-turns",
            "100000",
            "--mode",
            "agent",
        ]
    )
    receipt_read_fd, receipt_write_fd = os.pipe()
    monkeypatch.setenv(
        grok_cli_runner.GROK_TERMINAL_RECEIPT_FD_ENV,
        str(receipt_write_fd),
    )
    result = grok_cli_runner.main(outer_command[2:])
    receipt = os.read(
        receipt_read_fd,
        grok_cli_runner.GROK_TERMINAL_RECEIPT_MAX_BYTES,
    )
    os.close(receipt_read_fd)
    captured["outer_command"] = outer_command
    return result, receipt, captured


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


def test_grok_agent_runner_emits_command_bound_terminal_quota_envelope(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    stream = (
        b'{"type":"text","text":"resource exhausted; '
        b'too many requests; usage_pool_exhausted"}\n'
        b'{"type":"error","message":"usage_pool_exhausted"}'
    )

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(kwargs["env"])
            captured["stderr"] = kwargs["stderr"]
            captured["close_fds"] = kwargs["close_fds"]
            captured["pass_fds"] = kwargs.get("pass_fds")
            self.stdout = io.BytesIO(stream)

        @staticmethod
        def wait():
            return 23

    def fake_popen(cmd, **kwargs):
        return FakeProcess(cmd, **kwargs)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(grok_cli_runner.subprocess, "Popen", fake_popen)

    outer_command = grok_cli_runner.bind_grok_runner_command(
        [
            sys.executable,
            str(Path(grok_cli_runner.__file__).resolve()),
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--model",
            "grok-4.5",
        ]
    )
    receipt_read_fd, receipt_write_fd = os.pipe()
    monkeypatch.setenv(
        grok_cli_runner.GROK_TERMINAL_RECEIPT_FD_ENV,
        str(receipt_write_fd),
    )
    result = grok_cli_runner.main(outer_command[2:])
    receipt_bytes = os.read(
        receipt_read_fd,
        grok_cli_runner.GROK_TERMINAL_RECEIPT_MAX_BYTES,
    )
    os.close(receipt_read_fd)

    assert result == grok_cli_runner.GROK_QUOTA_EXHAUSTED_EXIT_CODE
    captured_output = capsys.readouterr()
    assert "resource exhausted" in captured_output.out
    command = captured["cmd"]
    assert isinstance(command, list)
    receipt = grok_cli_runner.parse_grok_terminal_quota_receipt(
        receipt_bytes,
        expected_runner_command=outer_command,
    )
    assert command != outer_command
    assert receipt["error_kind"] == "quota_exhausted"
    assert receipt["provider"] == "grok"
    assert receipt["model"] == "grok-4.5"
    assert receipt["inner_returncode"] == 23
    assert command[command.index("--output-format") + 1] == "streaming-json"
    assert captured["stderr"] is None
    assert captured["close_fds"] is True
    assert captured["pass_fds"] is None
    assert grok_cli_runner.GROK_TERMINAL_RECEIPT_FD_ENV not in captured["env"]


def test_grok_agent_runner_keeps_private_fd_and_env_out_of_descendants(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    fake_grok = tmp_path / "fake-grok"
    fake_grok.write_text(
        """#!/usr/bin/env python3
import json
import os
import subprocess
import sys

fd = int(os.environ["TEST_GROK_PRIVATE_FD"])
fd_path = f"/proc/self/fd/{fd}"
grandchild_code = (
    "import json,os;"
    "fd=int(os.environ['TEST_GROK_PRIVATE_FD']);"
    "print(json.dumps({"
    "'receipt_env': 'IPFS_ACCELERATE_GROK_TERMINAL_RECEIPT_FD' in os.environ,"
    "'fd_open': os.path.exists(f'/proc/self/fd/{fd}')"
    "}))"
)
grandchild = json.loads(
    subprocess.check_output([sys.executable, "-c", grandchild_code], text=True)
)
print(json.dumps({
    "type": "error",
    "code": "usage_pool_exhausted",
    "child_receipt_env": (
        "IPFS_ACCELERATE_GROK_TERMINAL_RECEIPT_FD" in os.environ
    ),
    "child_fd_open": os.path.exists(fd_path),
    "grandchild_receipt_env": grandchild["receipt_env"],
    "grandchild_fd_open": grandchild["fd_open"],
}), flush=True)
raise SystemExit(23)
""",
        encoding="utf-8",
    )
    fake_grok.chmod(0o755)
    outer_command = grok_cli_runner.bind_grok_runner_command(
        [
            sys.executable,
            str(Path(grok_cli_runner.__file__).resolve()),
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            str(fake_grok),
            "--model",
            "grok-4.5",
        ]
    )
    receipt_read_fd, receipt_write_fd = os.pipe()
    os.set_inheritable(receipt_write_fd, True)
    monkeypatch.setenv(
        grok_cli_runner.GROK_TERMINAL_RECEIPT_FD_ENV,
        str(receipt_write_fd),
    )
    monkeypatch.setenv("TEST_GROK_PRIVATE_FD", str(receipt_write_fd))
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))

    result = grok_cli_runner.main(outer_command[2:])
    receipt = os.read(
        receipt_read_fd,
        grok_cli_runner.GROK_TERMINAL_RECEIPT_MAX_BYTES,
    )
    os.close(receipt_read_fd)
    captured = capsys.readouterr().out

    assert result == grok_cli_runner.GROK_QUOTA_EXHAUSTED_EXIT_CODE
    assert grok_cli_runner.parse_grok_terminal_quota_receipt(
        receipt,
        expected_runner_command=outer_command,
    )
    assert '"child_receipt_env": false' in captured
    assert '"child_fd_open": false' in captured
    assert '"grandchild_receipt_env": false' in captured
    assert '"grandchild_fd_open": false' in captured


@pytest.mark.parametrize(
    ("stream", "returncode", "expected_returncode"),
    (
        (
            b'{"type":"text","error":{"type":"error",'
            b'"code":"usage_pool_exhausted"}}\n',
            23,
            23,
        ),
        (
            b'{malformed}\n'
            b'{"type":"error","code":"usage_pool_exhausted"}\n',
            23,
            23,
        ),
        (
            b'{"type":"error","code":"usage_pool_exhausted"}\n'
            b'{"type":"end"}\n',
            23,
            23,
        ),
        (
            b'{"type":"error","message":"HTTP 429 too many requests"}\n',
            23,
            23,
        ),
        (
            b'{"type":"error","code":"rate_limit",'
            b'"message":"usage_pool_exhausted"}\n',
            23,
            23,
        ),
        (
            b'{"type":"error","code":["usage_pool_exhausted"]}\n',
            23,
            23,
        ),
        (
            b'{"type":"error","code":"usage_limit_reached"}\n',
            0,
            0,
        ),
        (
            b'{"type":"end"}\n',
            grok_cli_runner.GROK_QUOTA_EXHAUSTED_EXIT_CODE,
            1,
        ),
    ),
)
def test_grok_agent_runner_rejects_untrusted_terminal_streams(
    monkeypatch,
    tmp_path,
    stream,
    returncode,
    expected_returncode,
) -> None:
    result, receipt, _captured = _run_supervised_fake_grok(
        monkeypatch,
        tmp_path,
        stream=stream,
        returncode=returncode,
    )

    assert result == expected_returncode
    assert receipt == b""


def test_grok_agent_runner_rejects_oversized_stream_before_typed_quota(
    monkeypatch,
    tmp_path,
) -> None:
    oversized = b"x" * (grok_cli_runner.GROK_STREAM_FRAME_MAX_BYTES + 1)
    stream = (
        oversized
        + b"\n"
        + b'{"type":"error","code":"usage_pool_exhausted"}\n'
    )

    result, receipt, _captured = _run_supervised_fake_grok(
        monkeypatch,
        tmp_path,
        stream=stream,
        returncode=23,
    )

    assert result == 23
    assert receipt == b""


def test_grok_agent_runner_accepts_exact_message_only_cli_quota_code(
    monkeypatch,
    tmp_path,
) -> None:
    result, receipt, captured = _run_supervised_fake_grok(
        monkeypatch,
        tmp_path,
        stream=b'{"type":"error","message":"usage_limit_reached"}\n',
        returncode=23,
    )

    assert result == grok_cli_runner.GROK_QUOTA_EXHAUSTED_EXIT_CODE
    parsed = grok_cli_runner.parse_grok_terminal_quota_receipt(
        receipt,
        expected_runner_command=captured["outer_command"],
    )
    assert parsed["quota_code"] == "usage_limit_reached"


@pytest.mark.parametrize(
    "message",
    (
        "terminal: usage_limit_reached",
        "not usage_limit_reached",
        "diagnostic mentions usage_pool_exhausted incidentally",
    ),
)
def test_grok_agent_runner_rejects_incidental_message_quota_tokens(
    monkeypatch,
    tmp_path,
    message,
) -> None:
    stream = json.dumps(
        {"type": "error", "message": message},
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"

    result, receipt, _captured = _run_supervised_fake_grok(
        monkeypatch,
        tmp_path,
        stream=stream,
        returncode=23,
    )

    assert result == 23
    assert receipt == b""


def test_grok_agent_runner_read_only_ambient_fd_preserves_direct_contract(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(
            cmd,
            grok_cli_runner.GROK_QUOTA_EXHAUSTED_EXIT_CODE,
        )

    receipt_read_fd, receipt_write_fd = os.pipe()
    monkeypatch.setenv(
        grok_cli_runner.GROK_TERMINAL_RECEIPT_FD_ENV,
        str(receipt_read_fd),
    )
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    try:
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
        os.fstat(receipt_read_fd)
    finally:
        os.close(receipt_read_fd)
        os.close(receipt_write_fd)

    assert result == grok_cli_runner.GROK_QUOTA_EXHAUSTED_EXIT_CODE
    command = captured["cmd"]
    assert isinstance(command, list)
    assert command[command.index("--output-format") + 1] == "plain"
