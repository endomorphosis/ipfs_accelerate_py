"""Grok-first, quota-gated Codex command for merge-conflict resolution.

Compatibility entrypoint for LLM merge-conflict resolution. Exact Grok 4.5 is
the primary route; exact Codex ``gpt-5.6-terra`` at medium reasoning is allowed
only after a typed native Grok quota/balance exhaustion event from that same
invocation. Missing tools, authentication failures, timeouts, transient rate
limits, and every other failure fail closed. Copilot is not a route member.
The legacy module CLI remains available only to configurations that invoke it
explicitly.
"""

from __future__ import annotations

import errno
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

# Recursion depth guard: prevents infinite loops when a provider invokes this
# resolver which in turn re-invokes a provider.
_INVOCATION_DEPTH_ENV = "_AGENT_RESOLVER_INVOCATION_DEPTH"
_MAX_INVOCATION_DEPTH = int(os.environ.get("AGENT_RESOLVER_MAX_DEPTH", "3"))

# Lock acquisition timeout to prevent indefinite blocking
_LOCK_TIMEOUT_ENV = "AGENT_RESOLVER_LOCK_TIMEOUT_SECONDS"
_DEFAULT_LOCK_TIMEOUT_SECONDS = 120.0
_DEFAULT_GROK_TIMEOUT_SECONDS = 900.0
_DEFAULT_CODEX_TIMEOUT_SECONDS = 600.0
GROK_MERGE_RESOLVER_MODEL = "grok-4.5"
CODEX_MERGE_RESOLVER_MODEL = "gpt-5.6-terra"
CODEX_MERGE_RESOLVER_REASONING_EFFORT = "medium"
_LOCK_ACQUISITION_FAILURE_EXIT_CODE = 75
_MAX_TOOL_OUTPUT_BYTES = 256 * 1024
_TOOL_OUTPUT_CHUNK_BYTES = 16 * 1024
_ACTIVE_TOOL_PROCESS: subprocess.Popen[bytes] | None = None


class _GitLockAcquisitionError(RuntimeError):
    """Prevent a resolver from editing a checkout without serialization."""


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
    """Return the canonical shell-safe Grok/quota-only resolver command."""

    from ..grok_cli_runner import build_grok_quota_routed_agent_command

    return shlex.join(
        build_grok_quota_routed_agent_command(
            workspace=".",
            python_executable=python_executable,
        )
    )


def _git_common_dir(workspace: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--git-common-dir"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise _GitLockAcquisitionError(
            f"could not resolve the git common directory: {type(exc).__name__}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = str(result.stderr or result.stdout or "").strip()
        suffix = f": {detail[-500:]}" if detail else ""
        raise _GitLockAcquisitionError(
            f"git common-directory discovery failed with exit {result.returncode}{suffix}"
        )
    value = result.stdout.strip()
    if not value:
        raise _GitLockAcquisitionError("git common-directory discovery returned an empty path")
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _acquire_git_lock(workspace: Path):
    """Acquire the checkout lock or raise before any provider can mutate it.

    ``AGENT_RESOLVER_LOCK_BYPASS=1`` is retained as an explicit test harness
    escape hatch.  Every operational discovery, import, open, flock, and
    timeout failure raises instead of being conflated with that bypass.
    """

    if os.environ.get("AGENT_RESOLVER_LOCK_BYPASS", "0") == "1":
        return None
    try:
        import fcntl
    except ImportError as exc:
        raise _GitLockAcquisitionError(
            "the platform does not provide fcntl checkout locking"
        ) from exc

    common_dir = _git_common_dir(workspace)
    try:
        common_dir.mkdir(parents=True, exist_ok=True)
        lock_path = common_dir / "agent-llm-resolver.lock"
        lock_handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise _GitLockAcquisitionError(
            f"could not open the checkout lock: {type(exc).__name__}: {exc}"
        ) from exc

    # Use non-blocking flock with a polling timeout
    try:
        timeout = float(
            os.environ.get(_LOCK_TIMEOUT_ENV, str(_DEFAULT_LOCK_TIMEOUT_SECONDS))
        )
    except (TypeError, ValueError) as exc:
        lock_handle.close()
        raise _GitLockAcquisitionError(
            f"{_LOCK_TIMEOUT_ENV} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(timeout) or timeout < 0:
        lock_handle.close()
        raise _GitLockAcquisitionError(
            f"{_LOCK_TIMEOUT_ENV} must be a finite non-negative number"
        )
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_handle
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                lock_handle.close()
                raise _GitLockAcquisitionError(
                    f"checkout flock failed: {type(exc).__name__}: {exc}"
                ) from exc
            if time.monotonic() >= deadline:
                lock_handle.close()
                raise _GitLockAcquisitionError(
                    f"checkout lock acquisition timed out after {timeout}s"
                )
            time.sleep(min(0.5, max(0.001, deadline - time.monotonic())))


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
    """Run a provider with bounded, concurrently drained output channels."""

    global _ACTIVE_TOOL_PROCESS
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    _ACTIVE_TOOL_PROCESS = process
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}

    def drain(name: str, stream: object) -> None:
        try:
            while True:
                chunk = stream.read(_TOOL_OUTPUT_CHUNK_BYTES)
                if not chunk:
                    return
                remaining = _MAX_TOOL_OUTPUT_BYTES - len(buffers[name])
                if remaining > 0:
                    buffers[name].extend(chunk[:remaining])
                if len(chunk) > max(0, remaining):
                    truncated[name] = True
        except (OSError, ValueError):
            truncated[name] = True

    stdout_thread = threading.Thread(
        target=drain,
        args=("stdout", process.stdout),
        name="merge-resolver-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=("stderr", process.stderr),
        name="merge-resolver-stderr",
        daemon=True,
    )

    def write_prompt() -> None:
        if process.stdin is None:
            return
        try:
            process.stdin.write(prompt.encode("utf-8"))
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            return

    stdin_thread = threading.Thread(
        target=write_prompt,
        name="merge-resolver-stdin",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    stdin_thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    finally:
        stdin_thread.join(timeout=1)
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        for name, stream, thread in (
            ("stdout", process.stdout, stdout_thread),
            ("stderr", process.stderr, stderr_thread),
        ):
            if thread.is_alive():
                truncated[name] = True
            elif stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if process.stdin is not None and not stdin_thread.is_alive():
            try:
                process.stdin.close()
            except OSError:
                pass
        if _ACTIVE_TOOL_PROCESS is process:
            _ACTIVE_TOOL_PROCESS = None

    captured: dict[str, str] = {}
    for name in ("stdout", "stderr"):
        value = bytes(buffers[name]).decode("utf-8", errors="replace")
        if truncated[name]:
            value += (
                "\n[agent merge resolver "
                f"{name} truncated at {_MAX_TOOL_OUTPUT_BYTES} bytes]\n"
            )
        captured[name] = value
    if timed_out:
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=captured["stdout"],
            stderr=captured["stderr"],
        ) from None
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        captured["stdout"],
        captured["stderr"],
    )


def _grok_binary() -> str:
    configured = (
        os.environ.get("IPFS_ACCELERATE_AGENT_GROK_BIN", "").strip()
        or os.environ.get("GROK_BIN", "").strip()
    )
    if configured:
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(configured)
        if found:
            return found
    return shutil.which("grok") or ""


def _strict_grok_quota_exhaustion(text: str, *, returncode: int) -> bool:
    """Use the implementation daemon's trusted, command-bound classifier.

    The import is intentionally function-local.  ``implementation_daemon``
    imports :func:`llm_merge_resolver_fallback_command` while defining its
    defaults, so a module-level reverse import would create a cycle.  At this
    point the Grok child has already exited and normal module initialization is
    complete.
    """

    from ..todo_daemon.implementation_daemon import (
        PRIMARY_QUOTA_EXHAUSTED_FALLBACK_TRIGGER,
        QUOTA_OR_BALANCE_EXHAUSTED_FAILURE_KIND,
        classify_provider_capacity_failure,
    )

    classified = classify_provider_capacity_failure(
        text,
        provider_labels=("grok",),
        provider_returncode=returncode,
    )
    providers = {
        str(provider).strip().lower()
        for provider in classified.get("providers", ())
        if str(provider).strip()
    }
    return bool(
        classified.get("exhausted") is True
        and "grok" in providers
        and classified.get("fallback_eligible") is True
        and str(classified.get("capacity_failure_kind") or "").strip().lower()
        == QUOTA_OR_BALANCE_EXHAUSTED_FAILURE_KIND
        and str(classified.get("provider_attribution") or "").strip().lower()
        == "implementation_command"
        and str(classified.get("fallback_trigger") or "").strip().lower()
        == PRIMARY_QUOTA_EXHAUSTED_FALLBACK_TRIGGER
    )


def _run_grok(prompt: str, workspace: Path) -> tuple[int | None, bool]:
    """Run exact Grok 4.5 and return ``(returncode, codex_authorized)``."""

    grok_bin = _grok_binary()
    if not grok_bin:
        print("grok merge resolver is unavailable; Codex fallback is forbidden", file=sys.stderr)
        return None, False
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "--workspace",
        str(workspace),
        "--grok-bin",
        grok_bin,
        "--model",
        GROK_MERGE_RESOLVER_MODEL,
        "--mode",
        "agent",
    ]
    try:
        completed = _run_tool(
            command,
            prompt=prompt,
            timeout=_timeout_seconds(
                "GROK_MERGE_RESOLVER_TIMEOUT_SECONDS",
                _DEFAULT_GROK_TIMEOUT_SECONDS,
            ),
        )
    except subprocess.TimeoutExpired as exc:
        detail = str(exc.stderr or exc.output or "").strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        print(
            f"grok merge resolver timed out; Codex fallback is forbidden{suffix}",
            file=sys.stderr,
        )
        return 124, False
    if completed.returncode == 0:
        return 0, False

    detail = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if str(part or "").strip()
    )
    # Grok's ordinary/model response is stdout and can contain task-controlled
    # conflict text.  Only the direct error channel from this exact invocation
    # may provide quota evidence for Codex authorization.
    error_detail = str(completed.stderr or "").strip()
    codex_authorized = _strict_grok_quota_exhaustion(
        error_detail,
        returncode=int(completed.returncode),
    )
    suffix = f": {detail[-2000:]}" if detail else ""
    if codex_authorized:
        print(
            "grok merge resolver reported explicit quota/balance exhaustion; "
            f"authorizing Codex fallback{suffix}",
            file=sys.stderr,
        )
    else:
        print(
            f"grok merge resolver failed with exit {completed.returncode}; "
            f"Codex fallback is forbidden{suffix}",
            file=sys.stderr,
        )
    return completed.returncode, codex_authorized


def _run_codex(prompt: str, workspace: Path) -> int | None:
    """Run the exact reviewed Codex fallback binding."""

    codex_bin = os.environ.get("CODEX_BIN", "").strip() or shutil.which("codex")
    if not codex_bin:
        print("no Codex fallback binary is available", file=sys.stderr)
        return None
    command = [
        codex_bin,
        "exec",
        "--ignore-user-config",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(workspace),
        "-m",
        CODEX_MERGE_RESOLVER_MODEL,
        "-c",
        f'model_reasoning_effort="{CODEX_MERGE_RESOLVER_REASONING_EFFORT}"',
        "-",
    ]
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
            f"codex merge resolver timed out{suffix}",
            file=sys.stderr,
        )
        return 124
    if completed.returncode != 0:
        detail = str(completed.stderr or completed.stdout or "").strip()
        suffix = f": {detail[-2000:]}" if detail else ""
        print(
            f"codex merge resolver failed with exit {completed.returncode}{suffix}",
            file=sys.stderr,
        )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Run Grok, allowing exact Codex fallback only for proven Grok quota."""

    signal.signal(signal.SIGTERM, _terminate_active_tool)
    signal.signal(signal.SIGINT, _terminate_active_tool)

    # Recursion guard: prevent infinite loops when submodules invoke each other
    current_depth = int(os.environ.get(_INVOCATION_DEPTH_ENV, "0"))
    if current_depth >= _MAX_INVOCATION_DEPTH:
        print(
            f"error: resolver invocation depth {current_depth} exceeds maximum "
            f"{_MAX_INVOCATION_DEPTH}; "
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
    try:
        lock_handle = _acquire_git_lock(workspace)
    except _GitLockAcquisitionError as exc:
        print(
            f"error: merge resolver checkout lock unavailable: {exc}; "
            "no provider was invoked",
            file=sys.stderr,
        )
        return _LOCK_ACQUISITION_FAILURE_EXIT_CODE
    try:
        grok_result, codex_authorized = _run_grok(prompt, workspace)
        if grok_result == 0:
            return 0
        if not codex_authorized:
            return grok_result if grok_result is not None else 127
        codex_result = _run_codex(prompt, workspace)
        return codex_result if codex_result is not None else (grok_result or 127)
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
