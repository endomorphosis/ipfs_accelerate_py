"""Bounded shared CLI process runner.

Provides synchronous, asynchronous, and streamed argv execution with:

- ``shell=False`` only (never interpolates dynamic data into a shell string)
- argv, environment, cwd, stdin, stdout, stderr, elapsed-time, and event bounds
- new process groups and descendant termination on timeout/cancellation
  (POSIX ``start_new_session`` / ``killpg``; Windows ``CREATE_NEW_PROCESS_GROUP``)
- distinct failure kinds: spawn, nonzero exit, timeout, cancellation,
  malformed output, and policy denial
- tracking of whether any output or side-effect event occurred
- redaction of prompts and secret-shaped environment values in diagnostics
- injected clocks and subprocess factories for deterministic tests

Importing this module starts no processes.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from .contracts import (
    MAX_ARGV_ITEM_CHARS,
    MAX_ARGV_ITEMS,
    MAX_EVENT_COUNT,
    MAX_EVENT_PAYLOAD_CHARS,
    MAX_PROMPT_CHARS,
    MAX_TEXT_CHARS,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    CLIEvent,
    CLIResult,
    EventKind,
    ExecutionMode,
)
from .errors import (
    BoundsExceededError,
    ContractValidationError,
    MalformedOutputError,
    NonzeroExitError,
    PolicyDeniedError,
    ProcessCancelledError,
    ProcessSpawnError,
    ProcessTimeoutError,
)

# ---------------------------------------------------------------------------
# Bounds and redaction defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_ENV_KEYS: int = 256
DEFAULT_MAX_ENV_KEY_CHARS: int = 256
DEFAULT_MAX_ENV_VALUE_CHARS: int = 65536
DEFAULT_MAX_STDIN_BYTES: int = MAX_PROMPT_CHARS
DEFAULT_MAX_STDOUT_BYTES: int = MAX_TEXT_CHARS
DEFAULT_MAX_STDERR_BYTES: int = MAX_TEXT_CHARS
DEFAULT_MAX_EVENTS: int = MAX_EVENT_COUNT
DEFAULT_TERM_GRACE_SECONDS: float = 0.5
DEFAULT_KILL_WAIT_SECONDS: float = 1.0
DEFAULT_POLL_INTERVAL_SECONDS: float = 0.02
DEFAULT_READ_CHUNK_BYTES: int = 65536

_SENSITIVE_ENV_MARKERS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "credential",
    "private_key",
    "access_key",
    "session_key",
    "cookie",
    "bearer",
)

REDACTED: str = "[redacted]"

Clock = Callable[[], float]
PopenFactory = Callable[..., Any]
CancelCheck = Callable[[], bool]


# ---------------------------------------------------------------------------
# Public configuration and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessBounds:
    """Hard limits enforced before and during process execution."""

    max_argv_items: int = MAX_ARGV_ITEMS
    max_argv_item_chars: int = MAX_ARGV_ITEM_CHARS
    max_env_keys: int = DEFAULT_MAX_ENV_KEYS
    max_env_key_chars: int = DEFAULT_MAX_ENV_KEY_CHARS
    max_env_value_chars: int = DEFAULT_MAX_ENV_VALUE_CHARS
    max_stdin_bytes: int = DEFAULT_MAX_STDIN_BYTES
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES
    max_elapsed_seconds: float = MAX_TIMEOUT_SECONDS
    max_event_count: int = DEFAULT_MAX_EVENTS
    term_grace_seconds: float = DEFAULT_TERM_GRACE_SECONDS
    kill_wait_seconds: float = DEFAULT_KILL_WAIT_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    read_chunk_bytes: int = DEFAULT_READ_CHUNK_BYTES

    def __post_init__(self) -> None:
        for name in (
            "max_argv_items",
            "max_argv_item_chars",
            "max_env_keys",
            "max_env_key_chars",
            "max_env_value_chars",
            "max_stdin_bytes",
            "max_stdout_bytes",
            "max_stderr_bytes",
            "max_event_count",
            "read_chunk_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ContractValidationError(f"{name} must be a positive integer")
        for name in (
            "max_elapsed_seconds",
            "term_grace_seconds",
            "kill_wait_seconds",
            "poll_interval_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ContractValidationError(f"{name} must be a positive number")


@dataclass(frozen=True)
class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def __bool__(self) -> bool:
        return self.is_cancelled()


@dataclass(frozen=True)
class ProcessSpec:
    """Validated argv-only process execution request.

    Environment overlay values of ``None`` mean *remove this key* from the
    inherited environment (when ``env_overlay=True``) or from the base map.
    """

    argv: Sequence[str]
    cwd: Optional[str] = None
    env: Optional[Mapping[str, Optional[str]]] = None
    env_overlay: bool = True
    stdin: Union[str, bytes, None] = None
    timeout_seconds: Optional[float] = None
    allowed_cwd_roots: Sequence[str] = ()
    check: bool = False
    text: bool = True
    encoding: str = "utf-8"
    decode_errors: str = "replace"
    side_effecting: bool = False
    mode: ExecutionMode = ExecutionMode.CHAT
    provider_name: Optional[str] = None
    model_name: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    cancel_token: Optional[CancellationToken] = None
    cancel_check: Optional[CancelCheck] = None


@dataclass(frozen=True)
class ProcessRunResult:
    """Bounded, redaction-safe outcome of a process run."""

    exit_code: Optional[int]
    stdout: str
    stderr: str
    elapsed_seconds: float
    ok: bool
    truncated_stdout: bool = False
    truncated_stderr: bool = False
    had_output: bool = False
    had_side_effect_event: bool = False
    process_started: bool = False
    cancelled: bool = False
    timed_out: bool = False
    pid: Optional[int] = None
    events: tuple[CLIEvent, ...] = ()
    argv_preview: tuple[str, ...] = ()
    cwd: Optional[str] = None
    env_keys: tuple[str, ...] = ()
    redacted_env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_seconds": self.elapsed_seconds,
            "ok": self.ok,
            "truncated_stdout": self.truncated_stdout,
            "truncated_stderr": self.truncated_stderr,
            "had_output": self.had_output,
            "had_side_effect_event": self.had_side_effect_event,
            "process_started": self.process_started,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "pid": self.pid,
            "events": [event.to_dict() for event in self.events],
            "argv_preview": list(self.argv_preview),
            "cwd": self.cwd,
            "env_keys": list(self.env_keys),
            "redacted_env": dict(self.redacted_env),
            "metadata": dict(self.metadata),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    def to_cli_result(self) -> CLIResult:
        """Map the process outcome onto the shared :class:`CLIResult` contract."""
        text = self.stdout
        if len(text) > MAX_TEXT_CHARS:
            text = text[: MAX_TEXT_CHARS - 3] + "..."
        error = None
        if not self.ok and self.error_message:
            from .errors import CLIErrorRecord, CLIRuntimeErrorCode

            code = CLIRuntimeErrorCode.INTERNAL
            if self.error_code:
                try:
                    code = CLIRuntimeErrorCode(self.error_code)
                except ValueError:
                    code = CLIRuntimeErrorCode.INTERNAL
            error = CLIErrorRecord(
                code=code,
                message=self.error_message,
                details={
                    "exit_code": "" if self.exit_code is None else str(self.exit_code),
                    "timed_out": str(self.timed_out),
                    "cancelled": str(self.cancelled),
                    "had_output": str(self.had_output),
                    "process_started": str(self.process_started),
                },
            )
        return CLIResult(
            text=text,
            ok=self.ok,
            mode=ExecutionMode.CHAT,
            provider_name=None,
            model_name=None,
            side_effecting=self.had_side_effect_event,
            cacheable=not self.had_side_effect_event,
            retryable=not self.had_side_effect_event and self.ok,
            streaming=False,
            truncated=self.truncated_stdout or self.truncated_stderr,
            cancelled=self.cancelled,
            exit_code=self.exit_code,
            elapsed_seconds=self.elapsed_seconds,
            events=self.events,
            error=error,
            metadata={
                **dict(self.metadata),
                "had_output": str(self.had_output),
                "process_started": str(self.process_started),
            },
            had_side_effect_event=self.had_side_effect_event,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_secret_env_key(key: str) -> bool:
    """Return True when an environment key looks secret-shaped."""
    lowered = str(key).strip().lower().replace("-", "_")
    if not lowered:
        return False
    return any(marker in lowered for marker in _SENSITIVE_ENV_MARKERS)


def redact_env_mapping(
    env: Mapping[str, Any] | None,
    *,
    max_value_chars: int = DEFAULT_MAX_ENV_VALUE_CHARS,
) -> dict[str, str]:
    """Return a diagnostics-safe copy of an environment mapping."""
    if env is None:
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in env.items():
        key = str(raw_key)
        if is_secret_env_key(key):
            out[key] = REDACTED
            continue
        if raw_value is None:
            out[key] = "[removed]"
            continue
        text = str(raw_value)
        if len(text) > max_value_chars:
            text = text[: max(0, max_value_chars - 3)] + "..."
        # Never echo prompt-like bulk stdin content through env diagnostics.
        if "prompt" in key.lower() or "stdin" in key.lower():
            out[key] = REDACTED
        else:
            out[key] = text
    return out


def redact_prompt(value: Any) -> str:
    """Always redact prompt/stdin material for diagnostics."""
    if value is None:
        return ""
    return REDACTED


def _clip(text: str, maximum: int) -> str:
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 3)] + "..."


def _normalize_argv(argv: Sequence[str], bounds: ProcessBounds) -> tuple[str, ...]:
    if not isinstance(argv, (list, tuple)):
        raise ContractValidationError("argv must be a sequence of strings")
    if not argv:
        raise PolicyDeniedError(
            "argv must not be empty",
            details={"reason": "empty_argv"},
        )
    if len(argv) > bounds.max_argv_items:
        raise BoundsExceededError(
            f"argv exceeds {bounds.max_argv_items} items",
            details={"length": len(argv), "maximum": bounds.max_argv_items},
        )
    normalized: list[str] = []
    for index, item in enumerate(argv):
        if not isinstance(item, str):
            raise ContractValidationError(
                f"argv[{index}] must be a string",
                details={"index": str(index), "type": type(item).__name__},
            )
        if "\x00" in item:
            raise PolicyDeniedError(
                "argv items must not contain NUL bytes",
                details={"index": str(index)},
            )
        if len(item) > bounds.max_argv_item_chars:
            raise BoundsExceededError(
                f"argv[{index}] exceeds {bounds.max_argv_item_chars} characters",
                details={
                    "index": str(index),
                    "length": len(item),
                    "maximum": bounds.max_argv_item_chars,
                },
            )
        normalized.append(item)
    # Reject accidental shell-string usage (single string that looks like a
    # full command with spaces is still one argv item — that is allowed and
    # deliberately *not* split). Callers must never pass a shell script as
    # the only means of interpretation; shell is always disabled.
    return tuple(normalized)


def _normalize_stdin(
    stdin: Union[str, bytes, None],
    bounds: ProcessBounds,
    *,
    encoding: str,
) -> Optional[bytes]:
    if stdin is None:
        return None
    if isinstance(stdin, str):
        if len(stdin) > MAX_PROMPT_CHARS:
            raise BoundsExceededError(
                f"stdin text exceeds {MAX_PROMPT_CHARS} characters",
                details={"length": len(stdin), "maximum": MAX_PROMPT_CHARS},
            )
        data = stdin.encode(encoding, errors="strict")
    elif isinstance(stdin, (bytes, bytearray, memoryview)):
        data = bytes(stdin)
    else:
        raise ContractValidationError("stdin must be str, bytes, or None")
    if len(data) > bounds.max_stdin_bytes:
        raise BoundsExceededError(
            f"stdin exceeds {bounds.max_stdin_bytes} bytes",
            details={"length": len(data), "maximum": bounds.max_stdin_bytes},
        )
    return data


def _resolve_cwd(
    cwd: Optional[str],
    allowed_roots: Sequence[str],
) -> Optional[str]:
    if cwd is None:
        return None
    if not isinstance(cwd, str) or not cwd.strip():
        raise ContractValidationError("cwd must be a non-empty string or None")
    try:
        path = Path(cwd).expanduser()
        if not path.is_absolute():
            # Relative paths are resolved against the current working directory
            # then validated against roots — never used raw for shell join.
            path = path.resolve(strict=False)
        else:
            path = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PolicyDeniedError(
            "cwd could not be resolved",
            details={"cwd": _clip(str(cwd), 256), "error": type(exc).__name__},
        ) from exc

    if not path.exists():
        raise PolicyDeniedError(
            "cwd does not exist",
            details={"cwd": _clip(str(path), 512)},
        )
    if not path.is_dir():
        raise PolicyDeniedError(
            "cwd is not a directory",
            details={"cwd": _clip(str(path), 512)},
        )

    if allowed_roots:
        resolved_roots: list[Path] = []
        for root in allowed_roots:
            if not isinstance(root, str) or not root.strip():
                raise ContractValidationError(
                    "allowed_cwd_roots entries must be non-empty strings"
                )
            try:
                resolved_roots.append(Path(root).expanduser().resolve(strict=False))
            except (OSError, RuntimeError) as exc:
                raise PolicyDeniedError(
                    "allowed_cwd_roots entry could not be resolved",
                    details={"root": _clip(str(root), 256), "error": type(exc).__name__},
                ) from exc
        if not any(_is_relative_to(path, root) for root in resolved_roots):
            raise PolicyDeniedError(
                "cwd escapes allowed roots",
                details={
                    "cwd": _clip(str(path), 512),
                    "reason": "cwd_escape",
                },
            )
    return str(path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _build_environment(
    env: Optional[Mapping[str, Optional[str]]],
    *,
    overlay: bool,
    bounds: ProcessBounds,
    base_env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    if overlay:
        result: dict[str, str] = dict(base_env if base_env is not None else os.environ)
    else:
        result = dict(base_env) if base_env is not None else {}

    if env is None:
        if len(result) > bounds.max_env_keys:
            raise BoundsExceededError(
                f"environment exceeds {bounds.max_env_keys} keys",
                details={"length": len(result), "maximum": bounds.max_env_keys},
            )
        return result

    if not isinstance(env, Mapping):
        raise ContractValidationError("env must be a mapping or None")
    if len(env) > bounds.max_env_keys:
        raise BoundsExceededError(
            f"environment overlay exceeds {bounds.max_env_keys} keys",
            details={"length": len(env), "maximum": bounds.max_env_keys},
        )

    for raw_key, raw_value in env.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ContractValidationError("environment keys must be non-empty strings")
        if "\x00" in raw_key:
            raise PolicyDeniedError("environment keys must not contain NUL bytes")
        if len(raw_key) > bounds.max_env_key_chars:
            raise BoundsExceededError(
                f"environment key exceeds {bounds.max_env_key_chars} characters",
                details={"length": len(raw_key), "maximum": bounds.max_env_key_chars},
            )
        if raw_value is None:
            result.pop(raw_key, None)
            continue
        if not isinstance(raw_value, str):
            raise ContractValidationError(
                f"environment value for {raw_key!r} must be str or None"
            )
        if "\x00" in raw_value:
            raise PolicyDeniedError(
                "environment values must not contain NUL bytes",
                details={"key": raw_key},
            )
        if len(raw_value) > bounds.max_env_value_chars:
            raise BoundsExceededError(
                f"environment value for {raw_key!r} exceeds "
                f"{bounds.max_env_value_chars} characters",
                details={
                    "key": raw_key,
                    "length": len(raw_value),
                    "maximum": bounds.max_env_value_chars,
                },
            )
        result[raw_key] = raw_value

    if len(result) > bounds.max_env_keys:
        raise BoundsExceededError(
            f"environment exceeds {bounds.max_env_keys} keys",
            details={"length": len(result), "maximum": bounds.max_env_keys},
        )
    return result


def _normalize_timeout(
    timeout_seconds: Optional[float],
    bounds: ProcessBounds,
) -> Optional[float]:
    if timeout_seconds is None:
        return None
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
        raise ContractValidationError("timeout_seconds must be a number or None")
    value = float(timeout_seconds)
    if value < MIN_TIMEOUT_SECONDS:
        raise ContractValidationError(
            f"timeout_seconds must be >= {MIN_TIMEOUT_SECONDS}"
        )
    if value > bounds.max_elapsed_seconds:
        raise BoundsExceededError(
            f"timeout_seconds exceeds {bounds.max_elapsed_seconds}",
            details={"timeout_seconds": value, "maximum": bounds.max_elapsed_seconds},
        )
    return value


def _decode_output(
    data: bytes,
    *,
    encoding: str,
    decode_errors: str,
    maximum: int,
) -> tuple[str, bool, bool]:
    """Return ``(text, truncated, malformed)``."""
    truncated = False
    if len(data) > maximum:
        data = data[:maximum]
        truncated = True
    malformed = False
    if decode_errors == "strict":
        try:
            text = data.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            malformed = True
            text = data.decode(encoding, errors="replace")
    else:
        text = data.decode(encoding, errors=decode_errors)
    if truncated and not text.endswith("..."):
        # Indicate truncation without growing beyond the character view much.
        text = text[: max(0, len(text) - 3)] + "..."
    return text, truncated, malformed


def _spawn_kwargs() -> dict[str, Any]:
    """Platform-specific process-group isolation flags."""
    kwargs: dict[str, Any] = {"shell": False}
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP allows CTRL_BREAK_EVENT targeting.
        create_new = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = create_new
    else:
        kwargs["start_new_session"] = True
    return kwargs


def terminate_process_tree(
    process: Any,
    *,
    grace_seconds: float = DEFAULT_TERM_GRACE_SECONDS,
    kill_wait_seconds: float = DEFAULT_KILL_WAIT_SECONDS,
    clock: Optional[Clock] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> bool:
    """Send TERM then KILL to the process group; return True if reaped.

    Works for POSIX process groups created with ``start_new_session=True`` and
    for Windows processes started with ``CREATE_NEW_PROCESS_GROUP``.
    """
    now = clock or time.monotonic
    nap = sleep or time.sleep

    if process is None:
        return True
    try:
        if process.poll() is not None:
            return True
    except Exception:
        return False

    pid = getattr(process, "pid", None)
    _send_term(process, pid)
    deadline = now() + max(0.0, grace_seconds)
    while now() < deadline:
        try:
            if process.poll() is not None:
                return True
        except Exception:
            break
        nap(min(0.05, max(0.0, deadline - now())))

    _send_kill(process, pid)
    kill_deadline = now() + max(0.0, kill_wait_seconds)
    while now() < kill_deadline:
        try:
            if process.poll() is not None:
                return True
        except Exception:
            break
        nap(min(0.05, max(0.0, kill_deadline - now())))
    try:
        process.wait(timeout=0.01)
    except Exception:
        pass
    try:
        return process.poll() is not None
    except Exception:
        return False


def _send_term(process: Any, pid: Optional[int]) -> None:
    try:
        if os.name == "nt":
            # Prefer CTRL_BREAK for process-group isolation when available.
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None and pid is not None:
                try:
                    process.send_signal(ctrl_break)
                    return
                except (OSError, ValueError, AttributeError):
                    pass
            process.terminate()
            return
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        process.terminate()
    except (ProcessLookupError, OSError):
        pass


def _send_kill(process: Any, pid: Optional[int]) -> None:
    try:
        if os.name == "nt":
            process.kill()
            return
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        process.kill()
    except (ProcessLookupError, OSError):
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ProcessRunner:
    """Bounded argv process executor with injectable clock and Popen factory."""

    def __init__(
        self,
        *,
        bounds: Optional[ProcessBounds] = None,
        clock: Optional[Clock] = None,
        sleep: Optional[Callable[[float], None]] = None,
        popen_factory: Optional[PopenFactory] = None,
        base_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.bounds = bounds or ProcessBounds()
        self._clock: Clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._popen: PopenFactory = popen_factory or subprocess.Popen
        self._base_env = dict(base_env) if base_env is not None else None

    # -- public API --------------------------------------------------------

    def run(
        self,
        argv: Sequence[str] | ProcessSpec,
        /,
        **kwargs: Any,
    ) -> ProcessRunResult:
        """Run *argv* synchronously and return a bounded result.

        Either pass a :class:`ProcessSpec` or ``argv`` plus keyword fields that
        match :class:`ProcessSpec` (except ``argv``).
        """
        spec = self._coerce_spec(argv, kwargs)
        return self._run_spec(spec)

    async def run_async(
        self,
        argv: Sequence[str] | ProcessSpec,
        /,
        **kwargs: Any,
    ) -> ProcessRunResult:
        """Async variant with the same semantics as :meth:`run`."""
        spec = self._coerce_spec(argv, kwargs)
        loop = asyncio.get_running_loop()
        # Use a worker thread so the injectable Popen factory remains shared
        # between sync and async paths (parity).
        return await loop.run_in_executor(None, self._run_spec, spec)

    def stream(
        self,
        argv: Sequence[str] | ProcessSpec,
        /,
        **kwargs: Any,
    ) -> Iterator[Union[CLIEvent, ProcessRunResult]]:
        """Yield progress events then a final :class:`ProcessRunResult`.

        Event emission is best-effort and bounded by ``max_event_count``.
        """
        spec = self._coerce_spec(argv, kwargs)
        yield from self._stream_spec(spec)

    async def stream_async(
        self,
        argv: Sequence[str] | ProcessSpec,
        /,
        **kwargs: Any,
    ):
        """Async generator yielding the same sequence as :meth:`stream`."""
        spec = self._coerce_spec(argv, kwargs)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _produce() -> None:
            try:
                for item in self._stream_spec(spec):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as exc:  # pragma: no cover - forwarded to consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        executor_task = loop.run_in_executor(None, _produce)
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await executor_task

    # -- internals ---------------------------------------------------------

    def _coerce_spec(
        self,
        argv: Sequence[str] | ProcessSpec,
        kwargs: dict[str, Any],
    ) -> ProcessSpec:
        if isinstance(argv, ProcessSpec):
            if kwargs:
                raise ContractValidationError(
                    "cannot pass both ProcessSpec and keyword overrides"
                )
            return argv
        return ProcessSpec(argv=argv, **kwargs)

    def _cancelled(self, spec: ProcessSpec) -> bool:
        if spec.cancel_token is not None and spec.cancel_token.is_cancelled():
            return True
        if spec.cancel_check is not None:
            try:
                if bool(spec.cancel_check()):
                    return True
            except Exception:
                return False
        return False

    def _prepare(
        self, spec: ProcessSpec
    ) -> tuple[
        tuple[str, ...],
        Optional[str],
        dict[str, str],
        Optional[bytes],
        Optional[float],
        dict[str, str],
        tuple[str, ...],
    ]:
        argv = _normalize_argv(spec.argv, self.bounds)
        cwd = _resolve_cwd(spec.cwd, spec.allowed_cwd_roots)
        env = _build_environment(
            spec.env,
            overlay=spec.env_overlay,
            bounds=self.bounds,
            base_env=self._base_env,
        )
        stdin_bytes = _normalize_stdin(
            spec.stdin, self.bounds, encoding=spec.encoding
        )
        timeout = _normalize_timeout(spec.timeout_seconds, self.bounds)
        redacted = redact_env_mapping(env, max_value_chars=128)
        # Overlay-only keys that were requested for removal appear as removed
        # only in the requested overlay view; surface requested keys for tests.
        if spec.env is not None:
            requested_view = {
                str(k): (None if v is None else str(v)) for k, v in spec.env.items()
            }
            redacted_requested = redact_env_mapping(
                requested_view, max_value_chars=128
            )
        else:
            redacted_requested = {}
        env_keys = tuple(sorted(env.keys()))
        return argv, cwd, env, stdin_bytes, timeout, redacted_requested or redacted, env_keys

    def _run_spec(self, spec: ProcessSpec) -> ProcessRunResult:
        items = list(self._stream_spec(spec))
        result = items[-1]
        if not isinstance(result, ProcessRunResult):
            raise RuntimeError("process stream did not produce a result")
        return result

    def _stream_spec(
        self, spec: ProcessSpec
    ) -> Iterator[Union[CLIEvent, ProcessRunResult]]:
        started = self._clock()
        events: list[CLIEvent] = []
        seq = 0

        def emit(kind: EventKind, message: str = "", **payload: str) -> CLIEvent:
            nonlocal seq
            event = CLIEvent(
                kind=kind,
                sequence=seq,
                message=_clip(message, MAX_EVENT_PAYLOAD_CHARS),
                payload={k: _clip(str(v), 256) for k, v in payload.items()},
                side_effecting=kind
                in (EventKind.TOOL_CALL, EventKind.TOOL_RESULT, EventKind.SIDE_EFFECT)
                or bool(spec.side_effecting and kind is EventKind.STARTED),
            )
            seq += 1
            if len(events) < self.bounds.max_event_count:
                events.append(event)
            return event

        try:
            argv, cwd, env, stdin_bytes, timeout, redacted_env, env_keys = (
                self._prepare(spec)
            )
        except (PolicyDeniedError, BoundsExceededError, ContractValidationError):
            raise

        if self._cancelled(spec):
            raise ProcessCancelledError(
                "process cancelled before spawn",
                details={"phase": "pre_spawn"},
            )

        yield emit(EventKind.STARTED, "process starting", argv0=argv[0])

        process: Any = None
        process_started = False
        pid: Optional[int] = None
        stdout_buf = bytearray()
        stderr_buf = bytearray()
        truncated_stdout = False
        truncated_stderr = False
        timed_out = False
        cancelled = False
        malformed = False
        exit_code: Optional[int] = None
        metadata_note = ""

        spawn_kwargs = _spawn_kwargs()
        # Never allow shell=True even if a custom factory ignores kwargs —
        # we still always pass shell=False explicitly.
        spawn_kwargs["shell"] = False

        try:
            try:
                process = self._popen(
                    list(argv),
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **spawn_kwargs,
                )
            except FileNotFoundError as exc:
                raise ProcessSpawnError(
                    "executable not found",
                    details={"argv0": argv[0], "error": type(exc).__name__},
                ) from exc
            except OSError as exc:
                raise ProcessSpawnError(
                    f"failed to spawn process: {type(exc).__name__}",
                    details={"argv0": argv[0], "errno": str(getattr(exc, "errno", ""))},
                ) from exc
            except TypeError as exc:
                # Factory signature mismatch — surface as spawn failure.
                raise ProcessSpawnError(
                    f"subprocess factory rejected arguments: {exc}",
                    details={"argv0": argv[0]},
                ) from exc

            process_started = True
            pid = getattr(process, "pid", None)

            # Write stdin on a helper thread so large payloads cannot deadlock
            # against a full pipe buffer before the child begins reading.
            stdin_done = threading.Event()
            stdin_error: list[BaseException] = []

            def _write_stdin() -> None:
                try:
                    if stdin_bytes is None or process.stdin is None:
                        return
                    view = memoryview(stdin_bytes)
                    offset = 0
                    chunk = self.bounds.read_chunk_bytes
                    while offset < len(view):
                        try:
                            written = process.stdin.write(view[offset : offset + chunk])
                        except BrokenPipeError:
                            return
                        if not written:
                            break
                        offset += written
                    try:
                        process.stdin.flush()
                    except BrokenPipeError:
                        return
                except BaseException as exc:  # pragma: no cover - surfaced via list
                    stdin_error.append(exc)
                finally:
                    try:
                        if process.stdin is not None:
                            process.stdin.close()
                    except Exception:
                        pass
                    stdin_done.set()

            if stdin_bytes is not None:
                threading.Thread(
                    target=_write_stdin,
                    name="cli-runtime-stdin",
                    daemon=True,
                ).start()
            else:
                stdin_done.set()

            # Concurrent readers for stdout/stderr to avoid pipe deadlock.
            stdout_done = threading.Event()
            stderr_done = threading.Event()

            def _read_stream(
                stream: Optional[BinaryIO],
                buffer: bytearray,
                done: threading.Event,
                *,
                is_stdout: bool,
            ) -> None:
                nonlocal truncated_stdout, truncated_stderr
                try:
                    if stream is None:
                        return
                    maximum = (
                        self.bounds.max_stdout_bytes
                        if is_stdout
                        else self.bounds.max_stderr_bytes
                    )
                    while True:
                        chunk = stream.read(self.bounds.read_chunk_bytes)
                        if not chunk:
                            break
                        remaining = maximum - len(buffer)
                        if remaining <= 0:
                            if is_stdout:
                                truncated_stdout = True
                            else:
                                truncated_stderr = True
                            # Drain without retaining so the child can exit.
                            continue
                        if len(chunk) > remaining:
                            buffer.extend(chunk[:remaining])
                            if is_stdout:
                                truncated_stdout = True
                            else:
                                truncated_stderr = True
                        else:
                            buffer.extend(chunk)
                except Exception:
                    pass
                finally:
                    done.set()

            stdout_thread = threading.Thread(
                target=_read_stream,
                args=(process.stdout, stdout_buf, stdout_done),
                kwargs={"is_stdout": True},
                name="cli-runtime-stdout",
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_read_stream,
                args=(process.stderr, stderr_buf, stderr_done),
                kwargs={"is_stdout": False},
                name="cli-runtime-stderr",
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            deadline = None if timeout is None else started + timeout
            while True:
                if self._cancelled(spec):
                    cancelled = True
                    terminate_process_tree(
                        process,
                        grace_seconds=self.bounds.term_grace_seconds,
                        kill_wait_seconds=self.bounds.kill_wait_seconds,
                        clock=self._clock,
                        sleep=self._sleep,
                    )
                    break
                if deadline is not None and self._clock() >= deadline:
                    timed_out = True
                    terminate_process_tree(
                        process,
                        grace_seconds=self.bounds.term_grace_seconds,
                        kill_wait_seconds=self.bounds.kill_wait_seconds,
                        clock=self._clock,
                        sleep=self._sleep,
                    )
                    break
                try:
                    code = process.poll()
                except Exception:
                    code = None
                if code is not None:
                    exit_code = int(code)
                    break
                remaining = self.bounds.poll_interval_seconds
                if deadline is not None:
                    remaining = min(remaining, max(0.0, deadline - self._clock()))
                self._sleep(remaining)

            # Ensure pipes are drained after exit/kill.
            stdin_done.wait(timeout=self.bounds.kill_wait_seconds + 1.0)
            stdout_done.wait(timeout=self.bounds.kill_wait_seconds + 1.0)
            stderr_done.wait(timeout=self.bounds.kill_wait_seconds + 1.0)
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if process.stderr is not None:
                    process.stderr.close()
            except Exception:
                pass
            try:
                if exit_code is None:
                    exit_code = process.poll()
                if exit_code is None:
                    process.wait(timeout=0.05)
                    exit_code = process.poll()
            except Exception:
                pass
            if stdin_error and not timed_out and not cancelled:
                # Non-fatal for most cases; record in metadata only.
                metadata_note = type(stdin_error[0]).__name__

        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        terminate_process_tree(
                            process,
                            grace_seconds=self.bounds.term_grace_seconds,
                            kill_wait_seconds=self.bounds.kill_wait_seconds,
                            clock=self._clock,
                            sleep=self._sleep,
                        )
                except Exception:
                    pass

        elapsed = max(0.0, self._clock() - started)
        stdout_text, trunc_out, mal_out = _decode_output(
            bytes(stdout_buf),
            encoding=spec.encoding,
            decode_errors=spec.decode_errors,
            maximum=self.bounds.max_stdout_bytes,
        )
        stderr_text, trunc_err, mal_err = _decode_output(
            bytes(stderr_buf),
            encoding=spec.encoding,
            decode_errors=spec.decode_errors,
            maximum=self.bounds.max_stderr_bytes,
        )
        truncated_stdout = truncated_stdout or trunc_out
        truncated_stderr = truncated_stderr or trunc_err
        malformed = mal_out or mal_err

        had_output = bool(stdout_buf or stderr_buf)
        if had_output and len(events) < self.bounds.max_event_count:
            events.append(
                CLIEvent(
                    kind=EventKind.TEXT_DELTA,
                    sequence=seq,
                    message=_clip(
                        f"captured stdout={len(stdout_buf)} stderr={len(stderr_buf)}",
                        MAX_EVENT_PAYLOAD_CHARS,
                    ),
                    payload={
                        "stdout_bytes": str(len(stdout_buf)),
                        "stderr_bytes": str(len(stderr_buf)),
                    },
                )
            )
            seq += 1

        had_side_effect = any(e.side_effecting for e in events) or (
            bool(spec.side_effecting) and process_started
        )
        if had_side_effect and process_started:
            # Explicit side-effect marker for consumers that key off events.
            if not any(e.kind is EventKind.SIDE_EFFECT for e in events):
                if len(events) < self.bounds.max_event_count:
                    events.append(
                        CLIEvent(
                            kind=EventKind.SIDE_EFFECT,
                            sequence=seq,
                            message="process started under side-effecting policy",
                            side_effecting=True,
                        )
                    )
                    seq += 1
                    had_side_effect = True

        metadata = {str(k): str(v) for k, v in (spec.metadata or {}).items()}
        metadata["had_output"] = str(had_output)
        metadata["process_started"] = str(process_started)
        if truncated_stdout:
            metadata["truncated_stdout"] = "true"
        if truncated_stderr:
            metadata["truncated_stderr"] = "true"
        if metadata_note:
            metadata["stdin_write_error"] = metadata_note

        # Failure classification (raise after constructing diagnostic context).
        error_code: Optional[str] = None
        error_message: Optional[str] = None
        ok = True

        def _result(**overrides: Any) -> ProcessRunResult:
            base = dict(
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                elapsed_seconds=elapsed,
                ok=ok,
                truncated_stdout=truncated_stdout,
                truncated_stderr=truncated_stderr,
                had_output=had_output,
                had_side_effect_event=had_side_effect,
                process_started=process_started,
                cancelled=cancelled,
                timed_out=timed_out,
                pid=pid,
                events=tuple(events[: self.bounds.max_event_count]),
                argv_preview=tuple(_clip(a, 128) for a in argv[:16]),
                cwd=cwd,
                env_keys=env_keys[: self.bounds.max_env_keys],
                redacted_env=redacted_env,
                metadata=metadata,
                error_code=error_code,
                error_message=error_message,
            )
            base.update(overrides)
            return ProcessRunResult(**base)

        def _flag(value: bool) -> str:
            return "true" if value else "false"

        failure_details = {
            "had_output": _flag(had_output),
            "process_started": _flag(process_started),
            "had_side_effect_event": _flag(had_side_effect),
            "elapsed_seconds": f"{elapsed:.6f}",
            "exit_code": "" if exit_code is None else str(exit_code),
            "argv0": argv[0],
        }

        if cancelled:
            ok = False
            error_code = "cancelled"
            error_message = "process cancelled"
            if len(events) < self.bounds.max_event_count:
                events.append(
                    CLIEvent(
                        kind=EventKind.CANCELLED,
                        sequence=seq,
                        message=error_message,
                    )
                )
            raise ProcessCancelledError(
                error_message,
                details=failure_details,
            )

        if timed_out:
            ok = False
            error_code = "timeout"
            error_message = "process exceeded timeout"
            if len(events) < self.bounds.max_event_count:
                events.append(
                    CLIEvent(
                        kind=EventKind.FAILED,
                        sequence=seq,
                        message=error_message,
                    )
                )
            raise ProcessTimeoutError(
                error_message,
                details={**failure_details, "timeout_seconds": str(timeout)},
            )

        if malformed:
            ok = False
            error_code = "malformed_output"
            error_message = "process output was not valid under strict decoding"
            if len(events) < self.bounds.max_event_count:
                events.append(
                    CLIEvent(
                        kind=EventKind.FAILED,
                        sequence=seq,
                        message=error_message,
                    )
                )
            raise MalformedOutputError(
                error_message,
                details=failure_details,
            )

        if exit_code is None:
            ok = False
            error_code = "spawn_failed"
            error_message = "process ended without an exit code"
            raise ProcessSpawnError(error_message, details=failure_details)

        if exit_code != 0:
            ok = False
            error_code = "nonzero_exit"
            error_message = f"process exited with status {exit_code}"
            if len(events) < self.bounds.max_event_count:
                events.append(
                    CLIEvent(
                        kind=EventKind.FAILED,
                        sequence=seq,
                        message=error_message,
                        payload={"exit_code": str(exit_code)},
                    )
                )
            if spec.check:
                raise NonzeroExitError(
                    error_message,
                    details={
                        **failure_details,
                        "stderr_chars": str(len(stderr_text)),
                    },
                )
            yield _result()
            return

        events.append(
            CLIEvent(
                kind=EventKind.COMPLETED,
                sequence=seq,
                message="process completed",
                payload={"exit_code": "0"},
            )
        )
        yield _result(ok=True, error_code=None, error_message=None)


# Module-level convenience helpers ------------------------------------------------

_default_runner = ProcessRunner()


def run_process(
    argv: Sequence[str] | ProcessSpec,
    /,
    **kwargs: Any,
) -> ProcessRunResult:
    """Run a process with the module-default :class:`ProcessRunner`."""
    return _default_runner.run(argv, **kwargs)


async def run_process_async(
    argv: Sequence[str] | ProcessSpec,
    /,
    **kwargs: Any,
) -> ProcessRunResult:
    """Async form of :func:`run_process`."""
    return await _default_runner.run_async(argv, **kwargs)


def stream_process(
    argv: Sequence[str] | ProcessSpec,
    /,
    **kwargs: Any,
) -> Iterator[Union[CLIEvent, ProcessRunResult]]:
    """Stream events then result using the module-default runner."""
    return _default_runner.stream(argv, **kwargs)


__all__ = [
    "DEFAULT_MAX_ENV_KEYS",
    "DEFAULT_MAX_ENV_KEY_CHARS",
    "DEFAULT_MAX_ENV_VALUE_CHARS",
    "DEFAULT_MAX_STDIN_BYTES",
    "DEFAULT_MAX_STDOUT_BYTES",
    "DEFAULT_MAX_STDERR_BYTES",
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_TERM_GRACE_SECONDS",
    "DEFAULT_KILL_WAIT_SECONDS",
    "REDACTED",
    "Clock",
    "PopenFactory",
    "CancelCheck",
    "ProcessBounds",
    "CancellationToken",
    "ProcessSpec",
    "ProcessRunResult",
    "ProcessRunner",
    "is_secret_env_key",
    "redact_env_mapping",
    "redact_prompt",
    "terminate_process_tree",
    "run_process",
    "run_process_async",
    "stream_process",
]
