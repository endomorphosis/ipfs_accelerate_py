"""Grok-first, quota-gated Codex command for merge-conflict resolution.

Compatibility entrypoint for LLM merge-conflict resolution. It retains the
checkout lock and recursion guard, then delegates the original stdin and
workspace to the canonical ``llm_router``-owned Grok runner. Exact Codex
``gpt-5.6-terra`` at medium reasoning is available only through that runner's
typed native quota route. This module does not classify provider failures or
execute a fallback provider itself.
"""

from __future__ import annotations

import errno
import math
import os
import shlex
import subprocess
import sys
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
GROK_MERGE_RESOLVER_MODEL = "grok-4.6"
CODEX_MERGE_RESOLVER_MODEL = "gpt-5.6-terra"
CODEX_MERGE_RESOLVER_REASONING_EFFORT = "medium"
_LOCK_ACQUISITION_FAILURE_EXIT_CODE = 75


class _GitLockAcquisitionError(RuntimeError):
    """Prevent a resolver from editing a checkout without serialization."""


def llm_merge_resolver_fallback_command(
    *,
    python_executable: str = "python3",
) -> str:
    """Return the lock-preserving compatibility entrypoint command."""

    return shlex.join(
        [
            str(python_executable),
            "-m",
            (
                "ipfs_accelerate_py.agent_supervisor.integrations."
                "llm_merge_resolver_fallback"
            ),
        ]
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
                ) from None
            time.sleep(min(0.5, max(0.001, deadline - time.monotonic())))


def _timeout_seconds(env_var: str, default: float) -> float | None:
    """Retain the legacy finite timeout parser for configuration consumers."""

    raw_value = os.environ.get(env_var, str(default))
    if raw_value == "0":
        return None
    try:
        return float(raw_value)
    except ValueError:
        return default


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical router-owned merge-resolution provider route."""

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
        from ..runtime import grok_cli_runner

        try:
            command = grok_cli_runner.build_grok_quota_routed_agent_command(
                workspace=workspace,
                python_executable=sys.executable,
                fallback_reasoning_effort=(
                    CODEX_MERGE_RESOLVER_REASONING_EFFORT
                ),
                enable_internal_legacy_preflight=True,
            )
        except (OSError, ValueError) as exc:
            print(
                f"canonical merge-resolver route is unavailable: {exc}",
                file=sys.stderr,
            )
            return 2
        return grok_cli_runner.main(command[3:])
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
