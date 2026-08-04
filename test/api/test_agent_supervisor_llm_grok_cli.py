"""End-to-end supervisor subprocess coverage for the Grok CLI provider."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LlmRouterInvocation,
    call_llm_router,
)


def test_grok_streaming_runner_forwards_output_and_keeps_bounded_tail(capsys) -> None:
    returncode, transcript = grok_cli_runner._run_grok_streaming(
        [
            "/bin/sh",
            "-c",
            "printf 'provider stdout'; printf 'provider stderr' >&2; exit 7",
        ],
        env=dict(os.environ),
    )

    captured = capsys.readouterr()
    assert returncode == 7
    assert captured.out == "provider stdout"
    assert captured.err == "provider stderr"
    assert "provider stdout" not in transcript
    assert "provider stderr" in transcript


def test_grok_streaming_runner_forwards_short_output_before_child_exit(
    monkeypatch,
) -> None:
    output_seen = threading.Event()
    result: dict[str, tuple[int, str]] = {}

    class _Buffer:
        def write(self, chunk: bytes) -> int:
            if b"provider-ready" in chunk:
                output_seen.set()
            return len(chunk)

        def flush(self) -> None:
            return None

    class _Target:
        buffer = _Buffer()

    monkeypatch.setattr(grok_cli_runner.sys, "stdout", _Target())

    def invoke() -> None:
        result["value"] = grok_cli_runner._run_grok_streaming(
            [
                sys.executable,
                "-u",
                "-c",
                (
                    "import sys, time; "
                    "sys.stdout.write('provider-ready\\n'); "
                    "sys.stdout.flush(); time.sleep(4)"
                ),
            ],
            env=dict(os.environ),
        )

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    assert output_seen.wait(timeout=3), "short output was buffered until EOF"
    assert worker.is_alive(), "child exited before incremental output was observed"
    worker.join(timeout=6)
    assert worker.is_alive() is False
    assert result["value"] == (0, "")


def test_supervisor_child_routes_grok_through_canonical_router(monkeypatch, tmp_path) -> None:
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
    hostile_pythonpath = tmp_path / "hostile-pythonpath"
    hostile_pythonpath.mkdir()
    sitecustomize_marker = tmp_path / "sitecustomize-executed"
    (hostile_pythonpath / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sitecustomize_marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(hostile_pythonpath))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "hostile-python-home"))
    hostile_script_root = tmp_path / "hostile-script-root"
    hostile_script_root.mkdir()
    script_root_marker = tmp_path / "script-root-imported"
    (hostile_script_root / "inspect.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(script_root_marker)!r}).write_text('executed')\n"
        "raise RuntimeError('hostile temporary script root executed')\n",
        encoding="utf-8",
    )
    # Force the private child program into a directory containing a hostile
    # stdlib shadow. Its implicit script path must not remain import authority.
    monkeypatch.setattr(tempfile, "tempdir", str(hostile_script_root))
    empty_child_cwd = tmp_path / "empty-child-cwd"
    empty_child_cwd.mkdir()
    (empty_child_cwd / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sitecustomize_marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    shadow_package = empty_child_cwd / "ipfs_accelerate_py"
    shadow_package.mkdir()
    (shadow_package / "__init__.py").write_text(
        "raise RuntimeError('hostile checkout package executed')\n",
        encoding="utf-8",
    )

    config = LlmRouterInvocation(
        repo_root=empty_child_cwd,
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
    assert sitecustomize_marker.exists() is False
    assert script_root_marker.exists() is False


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
        return 0, ""

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair the board"))
    monkeypatch.setattr(grok_cli_runner, "_run_grok_streaming", fake_run)

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
