"""Codex/Copilot fallback command for LLM merge-conflict resolution."""

from __future__ import annotations

import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


# Recursion depth guard: prevents infinite loops when Codex/Copilot
# invokes this resolver which in turn re-invokes Codex/Copilot.
_INVOCATION_DEPTH_ENV = "_AGENT_RESOLVER_INVOCATION_DEPTH"
_MAX_INVOCATION_DEPTH = int(os.environ.get("AGENT_RESOLVER_MAX_DEPTH", "3"))

# Lock acquisition timeout to prevent indefinite blocking
_LOCK_TIMEOUT_ENV = "AGENT_RESOLVER_LOCK_TIMEOUT_SECONDS"
_DEFAULT_LOCK_TIMEOUT_SECONDS = 120.0
_DEFAULT_CODEX_TIMEOUT_SECONDS = 900.0
_DEFAULT_COPILOT_TIMEOUT_SECONDS = 600.0
_ACTIVE_TOOL_PROCESS: subprocess.Popen[str] | None = None


def _terminate_active_tool(_signum: int, _frame: object) -> None:
    if _ACTIVE_TOOL_PROCESS is not None and _ACTIVE_TOOL_PROCESS.poll() is None:
        try:
            os.killpg(_ACTIVE_TOOL_PROCESS.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(143)


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
    """Acquire an exclusive git lock with a timeout to prevent deadlocks."""
    if os.environ.get("AGENT_RESOLVER_LOCK_BYPASS", "0") == "1":
        return None
    common_dir = _git_common_dir(workspace)
    if common_dir is None:
        return None
    try:
        import fcntl
    except ImportError:
        return None
    try:
        common_dir.mkdir(parents=True, exist_ok=True)
        lock_path = common_dir / "agent-llm-resolver.lock"
        lock_handle = lock_path.open("w", encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not open lock file: {exc}", file=sys.stderr)
        return None

    # Use non-blocking flock with a polling timeout
    timeout = float(os.environ.get(_LOCK_TIMEOUT_ENV, str(_DEFAULT_LOCK_TIMEOUT_SECONDS)))
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_handle
        except (IOError, OSError):
            if time.monotonic() >= deadline:
                print(
                    f"warning: lock acquisition timed out after {timeout}s; proceeding without lock",
                    file=sys.stderr,
                )
                lock_handle.close()
                return None
            time.sleep(0.5)


def _timeout_seconds(env_var: str, default: float) -> float | None:
    raw_value = os.environ.get(env_var, str(default))
    if raw_value == "0":
        return None
    try:
        return float(raw_value)
    except ValueError:
        return default


def _run_tool(
    command: Sequence[str],
    *,
    prompt: str,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    global _ACTIVE_TOOL_PROCESS
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    _ACTIVE_TOOL_PROCESS = process
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
    finally:
        if _ACTIVE_TOOL_PROCESS is process:
            _ACTIVE_TOOL_PROCESS = None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_codex(prompt: str, workspace: Path) -> int | None:
    codex_bin = os.environ.get("CODEX_BIN", "").strip() or shutil.which("codex")
    if not codex_bin or os.environ.get("PREFER_COPILOT_MERGE_RESOLVER", "0") == "1":
        return None
    codex_model = os.environ.get("IPFS_ACCELERATE_AGENT_CODEX_MODEL", "").strip()
    codex_reasoning = os.environ.get("IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT", "high").strip()
    command = [
        codex_bin,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(workspace),
    ]
    if codex_model:
        command.extend(["-m", codex_model])
    if codex_reasoning:
        command.extend(["-c", f'model_reasoning_effort="{codex_reasoning}"'])
    command.append("-")
    try:
        completed = _run_tool(
            command,
            prompt=prompt,
            timeout=_timeout_seconds(
                "CODEX_MERGE_RESOLVER_TIMEOUT_SECONDS",
                _DEFAULT_CODEX_TIMEOUT_SECONDS,
            ),
        )
    except subprocess.TimeoutExpired as exc:
        detail = str(exc.stderr or exc.output or "").strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        print(
            f"codex merge resolver timed out; falling back to copilot{suffix}",
            file=sys.stderr,
        )
        return 124
    if completed.returncode != 0:
        detail = str(completed.stderr or completed.stdout or "").strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        print(
            f"codex merge resolver failed with exit {completed.returncode}; "
            f"falling back to copilot{suffix}",
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
    copilot_model = os.environ.get("IPFS_ACCELERATE_AGENT_COPILOT_MODEL", "").strip()
    copilot_effort = os.environ.get("IPFS_ACCELERATE_AGENT_COPILOT_EFFORT", "high").strip()
    command = [
        copilot_bin,
        "-C",
        str(workspace),
        "--silent",
        "--allow-all-tools",
        "--allow-all-paths",
        "--no-ask-user",
        "--autopilot",
    ]
    if copilot_model:
        command.append(f"--model={copilot_model}")
    if copilot_effort:
        command.append(f"--effort={copilot_effort}")
    command.extend(["--prompt", prompt])
    try:
        completed = _run_tool(
            command,
            prompt=prompt,
            timeout=_timeout_seconds(
                "COPILOT_MERGE_RESOLVER_TIMEOUT_SECONDS",
                _DEFAULT_COPILOT_TIMEOUT_SECONDS,
            ),
        )
    except subprocess.TimeoutExpired as exc:
        detail = str(exc.stderr or exc.output or "").strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        print(f"copilot merge resolver timed out{suffix}", file=sys.stderr)
        return 124
    if completed.returncode != 0:
        detail = str(completed.stderr or completed.stdout or "").strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        print(
            f"copilot merge resolver failed with exit {completed.returncode}{suffix}",
            file=sys.stderr,
        )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Read a merge prompt from stdin and run Codex with Copilot fallback."""

    signal.signal(signal.SIGTERM, _terminate_active_tool)
    signal.signal(signal.SIGINT, _terminate_active_tool)

    # Recursion guard: prevent infinite loops when submodules invoke each other
    current_depth = int(os.environ.get(_INVOCATION_DEPTH_ENV, "0"))
    if current_depth >= _MAX_INVOCATION_DEPTH:
        print(
            f"error: resolver invocation depth {current_depth} exceeds maximum {_MAX_INVOCATION_DEPTH}; "
            f"aborting to prevent infinite recursion",
            file=sys.stderr,
        )
        return 2
    os.environ[_INVOCATION_DEPTH_ENV] = str(current_depth + 1)

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
        if not _copilot_has_auth():
            print("copilot fallback is not authenticated; returning codex result", file=sys.stderr)
            return codex_result if codex_result is not None else 127
        copilot_result = _run_copilot(prompt, workspace)
        if copilot_result == 127 and codex_result is not None:
            return codex_result
        return copilot_result
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
