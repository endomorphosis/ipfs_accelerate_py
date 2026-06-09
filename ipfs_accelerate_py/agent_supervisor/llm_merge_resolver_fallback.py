"""Codex/Copilot fallback command for LLM merge-conflict resolution."""

from __future__ import annotations

import os
import signal
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def llm_merge_resolver_fallback_command(
    *,
    python_executable: str = "python3",
) -> str:
    """Return a shell-safe command for the packaged fallback resolver."""

    return shlex.join(
        (
            python_executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback",
        )
    )


def _git_common_dir(workspace: Path) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "--git-common-dir"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _acquire_git_lock(workspace: Path):
    if os.environ.get("AGENT_RESOLVER_LOCK_BYPASS", "0") == "1":
        return None
    common_dir = _git_common_dir(workspace)
    if common_dir is None:
        return None
    try:
        import fcntl
    except ImportError:
        return None
    common_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (common_dir / "agent-llm-resolver.lock").open("w", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    return lock_handle


def _timeout_seconds(env_var: str, default: str = "60") -> float | None:
    raw_value = os.environ.get(env_var, default)
    if raw_value == "0":
        return None
    try:
        return float(raw_value)
    except ValueError:
        return float(default)


def _run_tool(
    command: Sequence[str],
    *,
    prompt: str,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_codex(prompt: str, workspace: Path) -> int | None:
    codex_bin = os.environ.get("CODEX_BIN", "").strip() or shutil.which("codex")
    if not codex_bin or os.environ.get("PREFER_COPILOT_MERGE_RESOLVER", "0") == "1":
        return None
    command = [
        codex_bin,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(workspace),
        "-",
    ]
    try:
        completed = _run_tool(
            command,
            prompt=prompt,
            timeout=_timeout_seconds("CODEX_MERGE_RESOLVER_TIMEOUT_SECONDS"),
        )
    except subprocess.TimeoutExpired:
        print("codex merge resolver timed out; falling back to copilot", file=sys.stderr)
        return 124
    if completed.returncode != 0:
        print(
            f"codex merge resolver failed with exit {completed.returncode}; falling back to copilot",
            file=sys.stderr,
        )
    return completed.returncode


def _copilot_has_auth() -> bool:
    if any(os.environ.get(name) for name in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")):
        return True
    gh = shutil.which("gh")
    if not gh:
        return False
    completed = subprocess.run([gh, "auth", "status"], text=True, capture_output=True, check=False)
    return completed.returncode == 0


def _run_copilot(prompt: str, workspace: Path) -> int:
    copilot_bin = os.environ.get("COPILOT_BIN", "").strip() or shutil.which("copilot")
    if not copilot_bin:
        print("no copilot fallback binary available for merge resolution", file=sys.stderr)
        return 127
    if not _copilot_has_auth():
        print("copilot fallback is not authenticated for merge resolution", file=sys.stderr)
        return 127
    try:
        completed = _run_tool(
            [
                copilot_bin,
                "-C",
                str(workspace),
                "--silent",
                "--allow-all-tools",
                "--allow-all-paths",
                "--no-ask-user",
                "--autopilot",
                "--prompt",
                prompt,
            ],
            prompt=prompt,
            timeout=_timeout_seconds("COPILOT_MERGE_RESOLVER_TIMEOUT_SECONDS"),
        )
    except subprocess.TimeoutExpired:
        print("copilot merge resolver timed out", file=sys.stderr)
        return 124
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Read a merge prompt from stdin and run Codex with Copilot fallback."""

    args = list(sys.argv[1:] if argv is None else argv)
    workspace = Path(
        args[0]
        if args
        else os.environ.get("IPFS_ACCELERATE_AGENT_MERGE_WORKSPACE", os.getcwd())
    )
    prompt = sys.stdin.read()
    lock_handle = _acquire_git_lock(workspace)
    try:
        codex_result = _run_codex(prompt, workspace)
        if codex_result == 0:
            return 0
        copilot_result = _run_copilot(prompt, workspace)
        if copilot_result == 127 and codex_result is not None:
            return codex_result
        return copilot_result
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
