"""Deterministic tests for the bounded shared CLI process runner."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from ipfs_accelerate_py.cli_runtime.errors import (
    BoundsExceededError,
    MalformedOutputError,
    NonzeroExitError,
    PolicyDeniedError,
    ProcessCancelledError,
    ProcessSpawnError,
    ProcessTimeoutError,
)
from ipfs_accelerate_py.cli_runtime.process_runner import (
    REDACTED,
    CancellationToken,
    ProcessBounds,
    ProcessRunner,
    ProcessSpec,
    is_secret_env_key,
    redact_env_mapping,
    redact_prompt,
    run_process,
    run_process_async,
    stream_process,
    terminate_process_tree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _py(*code_lines: str) -> list[str]:
    """Build an argv that runs a short Python snippet without a shell."""
    return [sys.executable, "-c", "\n".join(code_lines)]


class _ControllableClock:
    """Monotonic clock that tests can advance explicitly."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


class _RecordingPopen:
    """Wraps subprocess.Popen and records the kwargs used for spawn."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.processes: list[Any] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append({"argv": list(argv), **kwargs})
        assert kwargs.get("shell") is False
        proc = subprocess.Popen(argv, **kwargs)
        self.processes.append(proc)
        return proc


# ---------------------------------------------------------------------------
# Basic execution / argv integrity
# ---------------------------------------------------------------------------


def test_spaces_and_metacharacters_as_one_argv_item() -> None:
    dangerous = "hello world; rm -rf / && echo pwned | cat `id` $HOME"
    runner = ProcessRunner()
    result = runner.run(
        _py(
            "import sys",
            "print(repr(sys.argv[1]))",
        )
        + [dangerous]
    )
    assert result.ok
    assert result.exit_code == 0
    # The entire string must arrive as a single argv element — not expanded.
    assert dangerous in result.stdout
    assert "pwned" not in result.stdout or dangerous in result.stdout
    # Confirm shell was not used by checking the argv record shape via factory.
    recorder = _RecordingPopen()
    runner2 = ProcessRunner(popen_factory=recorder)
    runner2.run(_py("print('ok')",) + [dangerous])
    assert recorder.calls
    assert recorder.calls[0]["shell"] is False
    assert dangerous in recorder.calls[0]["argv"]


def test_never_uses_shell_true() -> None:
    recorder = _RecordingPopen()
    runner = ProcessRunner(popen_factory=recorder)
    runner.run(_py("print(1)"))
    assert recorder.calls[0]["shell"] is False
    # Platform process-group isolation must be requested.
    if os.name == "nt":
        flags = recorder.calls[0].get("creationflags", 0)
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert recorder.calls[0].get("start_new_session") is True


def test_large_stdin_round_trip() -> None:
    payload = ("A" * 50_000) + "\n" + ("B" * 50_000)
    runner = ProcessRunner()
    result = runner.run(
        _py(
            "import sys",
            "data = sys.stdin.read()",
            "print(len(data))",
            "print(data[:1], data[-1], sep='')",
        ),
        stdin=payload,
    )
    assert result.ok
    lines = result.stdout.strip().splitlines()
    assert lines[0] == str(len(payload))
    assert "A" in lines[1] and "B" in lines[1]


def test_output_truncation() -> None:
    bounds = ProcessBounds(max_stdout_bytes=64, max_stderr_bytes=32)
    runner = ProcessRunner(bounds=bounds)
    result = runner.run(
        _py(
            "import sys",
            "sys.stdout.write('X' * 10_000)",
            "sys.stderr.write('Y' * 10_000)",
        )
    )
    assert result.ok
    assert result.truncated_stdout is True
    assert result.truncated_stderr is True
    assert len(result.stdout.encode("utf-8")) <= bounds.max_stdout_bytes + 16
    assert len(result.stderr.encode("utf-8")) <= bounds.max_stderr_bytes + 16
    assert result.had_output is True


def test_argv_bounds_exceeded() -> None:
    bounds = ProcessBounds(max_argv_items=2, max_argv_item_chars=8)
    runner = ProcessRunner(bounds=bounds)
    with pytest.raises(BoundsExceededError):
        runner.run(["a", "b", "c"])
    with pytest.raises(BoundsExceededError):
        runner.run(["short", "toolongarg"])


def test_empty_argv_policy_denied() -> None:
    runner = ProcessRunner()
    with pytest.raises(PolicyDeniedError):
        runner.run([])


# ---------------------------------------------------------------------------
# Cancellation / timeout / TERM→KILL / orphans
# ---------------------------------------------------------------------------


def test_timeout_raises_and_kills() -> None:
    runner = ProcessRunner(
        bounds=ProcessBounds(term_grace_seconds=0.1, kill_wait_seconds=0.5)
    )
    with pytest.raises(ProcessTimeoutError) as excinfo:
        runner.run(
            _py("import time; time.sleep(30)"),
            timeout_seconds=0.2,
        )
    assert excinfo.value.code.value == "timeout"
    assert excinfo.value.details.get("process_started") == "true"


def test_concurrent_cancellation() -> None:
    token = CancellationToken()
    runner = ProcessRunner(
        bounds=ProcessBounds(term_grace_seconds=0.1, kill_wait_seconds=0.5)
    )
    errors: list[BaseException] = []
    done = threading.Event()

    def _worker() -> None:
        try:
            runner.run(
                _py("import time; time.sleep(30)"),
                cancel_token=token,
                timeout_seconds=10.0,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    time.sleep(0.15)
    token.cancel()
    assert done.wait(timeout=5.0)
    assert errors
    assert isinstance(errors[0], ProcessCancelledError)
    assert errors[0].code.value == "cancelled"


def test_term_to_kill_escalation(tmp_path: Path) -> None:
    """Child ignores SIGTERM; runner must escalate to SIGKILL."""
    if os.name != "posix":
        pytest.skip("SIGTERM ignore test requires POSIX")

    marker = tmp_path / "alive.marker"
    script = _py(
        "import os, signal, time, pathlib, sys",
        f"path = pathlib.Path({str(marker)!r})",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        "path.write_text(str(os.getpid()))",
        "time.sleep(60)",
    )
    runner = ProcessRunner(
        bounds=ProcessBounds(term_grace_seconds=0.15, kill_wait_seconds=1.0)
    )
    with pytest.raises(ProcessTimeoutError):
        runner.run(script, timeout_seconds=0.25)

    # Process should be gone (killed) shortly after.
    deadline = time.time() + 3.0
    while marker.exists() and time.time() < deadline:
        try:
            pid = int(marker.read_text().strip())
            os.kill(pid, 0)
            time.sleep(0.05)
        except (OSError, ValueError):
            break
    if marker.exists():
        try:
            pid = int(marker.read_text().strip())
            with pytest.raises(OSError):
                os.kill(pid, 0)
        except ValueError:
            pass


def test_orphan_prevention_kills_process_group(tmp_path: Path) -> None:
    """Descendants in the same process group must die on timeout."""
    if os.name != "posix":
        pytest.skip("process-group orphan test requires POSIX")

    child_pid_file = tmp_path / "child.pid"
    parent_script = _py(
        "import os, subprocess, sys, time, pathlib",
        f"child_path = pathlib.Path({str(child_pid_file)!r})",
        "child = subprocess.Popen(",
        "    [sys.executable, '-c', 'import time; time.sleep(60)'],",
        ")",
        "child_path.write_text(str(child.pid))",
        "time.sleep(60)",
    )
    runner = ProcessRunner(
        bounds=ProcessBounds(term_grace_seconds=0.2, kill_wait_seconds=1.0)
    )
    with pytest.raises(ProcessTimeoutError):
        runner.run(parent_script, timeout_seconds=0.3)

    deadline = time.time() + 3.0
    while not child_pid_file.exists() and time.time() < deadline:
        time.sleep(0.02)
    # Child pid file may or may not have been written before kill; if written,
    # the grandchild must not remain alive.
    if child_pid_file.exists():
        child_pid = int(child_pid_file.read_text().strip())
        dead = False
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                os.kill(child_pid, 0)
                time.sleep(0.05)
            except OSError:
                dead = True
                break
        assert dead, f"orphan child pid {child_pid} still alive"


def test_pre_spawn_cancellation() -> None:
    token = CancellationToken()
    token.cancel()
    runner = ProcessRunner()
    with pytest.raises(ProcessCancelledError) as excinfo:
        runner.run(_py("print('nope')"), cancel_token=token)
    assert excinfo.value.details.get("phase") == "pre_spawn"


# ---------------------------------------------------------------------------
# cwd policy / environment
# ---------------------------------------------------------------------------


def test_cwd_escape_denied(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    runner = ProcessRunner()
    with pytest.raises(PolicyDeniedError) as excinfo:
        runner.run(
            _py("import os; print(os.getcwd())"),
            cwd=str(outside),
            allowed_cwd_roots=[str(allowed)],
        )
    assert excinfo.value.code.value == "policy_denied"
    blob = f"{excinfo.value} {excinfo.value.details}".lower()
    assert "escape" in blob


def test_cwd_within_root_allowed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    runner = ProcessRunner()
    result = runner.run(
        _py("import os; print(os.getcwd())"),
        cwd=str(nested),
        allowed_cwd_roots=[str(root)],
    )
    assert result.ok
    assert str(nested.resolve()) in result.stdout


def test_cwd_missing_denied(tmp_path: Path) -> None:
    runner = ProcessRunner()
    with pytest.raises(PolicyDeniedError):
        runner.run(
            _py("print(1)"),
            cwd=str(tmp_path / "does-not-exist"),
        )


def test_environment_removal() -> None:
    base = {"KEEP": "1", "DROP_ME": "secret-value", "PATH": os.environ.get("PATH", "")}
    runner = ProcessRunner(base_env=base)
    result = runner.run(
        _py(
            "import os",
            "print('DROP=' + os.environ.get('DROP_ME', 'MISSING'))",
            "print('KEEP=' + os.environ.get('KEEP', 'MISSING'))",
        ),
        env={"DROP_ME": None},
        env_overlay=True,
    )
    assert result.ok
    assert "DROP=MISSING" in result.stdout
    assert "KEEP=1" in result.stdout


def test_environment_overlay_and_bounds() -> None:
    runner = ProcessRunner(bounds=ProcessBounds(max_env_keys=3))
    with pytest.raises(BoundsExceededError):
        runner.run(
            _py("print(1)"),
            env={"A": "1", "B": "2", "C": "3", "D": "4"},
            env_overlay=False,
        )


def test_secret_redaction() -> None:
    assert is_secret_env_key("API_KEY")
    assert is_secret_env_key("my_secret_token")
    assert not is_secret_env_key("HOME")
    assert redact_prompt("super secret prompt text") == REDACTED

    env = {
        "API_KEY": "sk-live-abc",
        "PASSWORD": "hunter2",
        "HOME": "/home/user",
        "PROMPT_TEXT": "do not leak",
        "GONE": None,
    }
    redacted = redact_env_mapping(env)
    assert redacted["API_KEY"] == REDACTED
    assert redacted["PASSWORD"] == REDACTED
    assert redacted["PROMPT_TEXT"] == REDACTED
    assert redacted["HOME"] == "/home/user"
    assert redacted["GONE"] == "[removed]"

    runner = ProcessRunner(base_env={"PATH": os.environ.get("PATH", "")})
    result = runner.run(
        _py("print('ok')"),
        env={"OPENAI_API_KEY": "sk-test", "SAFE_FLAG": "1"},
        env_overlay=True,
        stdin="this is a confidential prompt",
    )
    assert result.ok
    payload = result.to_dict()
    # Prompt content must never appear in diagnostics.
    assert "confidential prompt" not in str(payload)
    assert "sk-test" not in str(payload)
    assert payload["redacted_env"].get("OPENAI_API_KEY") == REDACTED
    assert "SAFE_FLAG" in payload["redacted_env"]


def test_error_details_never_include_prompt() -> None:
    runner = ProcessRunner()
    secret_prompt = "EXFILTRATE_ME_PROMPT_VALUE_42"
    with pytest.raises(NonzeroExitError) as excinfo:
        runner.run(
            _py("import sys; sys.exit(3)"),
            stdin=secret_prompt,
            check=True,
        )
    assert secret_prompt not in str(excinfo.value)
    assert secret_prompt not in str(excinfo.value.details)
    assert secret_prompt not in str(excinfo.value.to_dict())


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def test_spawn_failure() -> None:
    runner = ProcessRunner()
    with pytest.raises(ProcessSpawnError) as excinfo:
        runner.run(["/nonexistent/cli-runtime-binary-zzz", "--help"])
    assert excinfo.value.code.value == "spawn_failed"


def test_nonzero_exit_without_check_returns_result() -> None:
    runner = ProcessRunner()
    result = runner.run(_py("import sys; sys.exit(7)"), check=False)
    assert result.ok is False
    assert result.exit_code == 7
    assert result.error_code == "nonzero_exit"


def test_nonzero_exit_with_check_raises() -> None:
    runner = ProcessRunner()
    with pytest.raises(NonzeroExitError) as excinfo:
        runner.run(_py("import sys; sys.exit(9)"), check=True)
    assert excinfo.value.code.value == "nonzero_exit"


def test_malformed_output_strict_decode() -> None:
    runner = ProcessRunner()
    with pytest.raises(MalformedOutputError) as excinfo:
        runner.run(
            _py(
                "import sys",
                "sys.stdout.buffer.write(b'\\xff\\xfe not utf8')",
            ),
            decode_errors="strict",
        )
    assert excinfo.value.code.value == "malformed_output"


def test_policy_denied_nul_in_argv() -> None:
    runner = ProcessRunner()
    with pytest.raises(PolicyDeniedError):
        runner.run([sys.executable, "-c", "print(1)", "bad\x00arg"])


# ---------------------------------------------------------------------------
# Injected clock / factory / sync-async parity
# ---------------------------------------------------------------------------


def test_injected_subprocess_factory() -> None:
    recorder = _RecordingPopen()
    runner = ProcessRunner(popen_factory=recorder)
    result = runner.run(_py("print('via-factory')"))
    assert result.ok
    assert "via-factory" in result.stdout
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["shell"] is False


def test_injected_clock_used_for_elapsed() -> None:
    clock = _ControllableClock(start=500.0)

    class _AdvancingPopen(_RecordingPopen):
        def __call__(self, argv: list[str], **kwargs: Any) -> Any:
            clock.advance(1.25)
            return super().__call__(argv, **kwargs)

    runner = ProcessRunner(clock=clock, popen_factory=_AdvancingPopen())
    result = runner.run(_py("print('t')"))
    assert result.ok
    assert result.elapsed_seconds >= 1.25


def test_sync_async_result_parity() -> None:
    argv = _py(
        "import sys",
        "data = sys.stdin.read()",
        "print('OUT:' + data)",
        "print('ARG:' + sys.argv[1], file=sys.stderr)",
    ) + ["meta char $HOME; echo"]
    stdin = "parity-payload"
    env = {"CLI_RUNTIME_TEST_FLAG": "1"}

    sync_runner = ProcessRunner()
    sync_result = sync_runner.run(
        argv,
        stdin=stdin,
        env=env,
        env_overlay=True,
        check=False,
    )

    async def _async() -> Any:
        async_runner = ProcessRunner()
        return await async_runner.run_async(
            argv,
            stdin=stdin,
            env=env,
            env_overlay=True,
            check=False,
        )

    async_result = asyncio.run(_async())

    assert sync_result.exit_code == async_result.exit_code
    assert sync_result.stdout == async_result.stdout
    assert sync_result.stderr == async_result.stderr
    assert sync_result.ok == async_result.ok
    assert sync_result.had_output == async_result.had_output
    assert sync_result.truncated_stdout == async_result.truncated_stdout
    assert sync_result.truncated_stderr == async_result.truncated_stderr


def test_module_level_helpers_and_stream() -> None:
    result = run_process(_py("print('helper')"))
    assert result.ok
    assert "helper" in result.stdout

    items = list(stream_process(_py("print('streamed')")))
    assert any(getattr(i, "kind", None) is not None for i in items[:-1])
    assert items[-1].ok is True

    async def _go() -> Any:
        return await run_process_async(_py("print('async-helper')"))

    async_result = asyncio.run(_go())
    assert "async-helper" in async_result.stdout


def test_side_effect_tracking() -> None:
    runner = ProcessRunner()
    result = runner.run(
        _py("print('side')"),
        side_effecting=True,
    )
    assert result.ok
    assert result.process_started is True
    assert result.had_output is True
    assert result.had_side_effect_event is True
    cli = result.to_cli_result()
    assert cli.had_side_effect_event is True
    assert cli.side_effecting is True
    assert cli.cacheable is False


def test_had_output_false_when_silent() -> None:
    runner = ProcessRunner()
    result = runner.run(_py("pass"))
    assert result.ok
    assert result.had_output is False


def test_process_spec_object() -> None:
    spec = ProcessSpec(
        argv=_py("import os; print(os.environ.get('Z','-'))"),
        env={"Z": "zed"},
        env_overlay=True,
    )
    runner = ProcessRunner(base_env={"PATH": os.environ.get("PATH", "")})
    result = runner.run(spec)
    assert result.ok
    assert "zed" in result.stdout


def test_terminate_process_tree_helper() -> None:
    proc = subprocess.Popen(
        _py("import time; time.sleep(30)"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **(
            {"start_new_session": True}
            if os.name != "nt"
            else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        ),
    )
    try:
        assert terminate_process_tree(proc, grace_seconds=0.1, kill_wait_seconds=1.0)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


def test_cancel_check_callable() -> None:
    flag = {"cancel": False}

    def _check() -> bool:
        return flag["cancel"]

    runner = ProcessRunner(
        bounds=ProcessBounds(term_grace_seconds=0.1, kill_wait_seconds=0.5)
    )
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            runner.run(
                _py("import time; time.sleep(30)"),
                cancel_check=_check,
                timeout_seconds=10.0,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    time.sleep(0.15)
    flag["cancel"] = True
    thread.join(timeout=5.0)
    assert errors
    assert isinstance(errors[0], ProcessCancelledError)


def test_stdin_bounds_exceeded() -> None:
    bounds = ProcessBounds(max_stdin_bytes=16)
    runner = ProcessRunner(bounds=bounds)
    with pytest.raises(BoundsExceededError):
        runner.run(_py("import sys; print(sys.stdin.read())"), stdin="x" * 64)


def test_import_does_not_start_processes() -> None:
    """Importing the module must remain side-effect free (smoke check).

    Avoid ``importlib.reload``: reloading rebinds ``ProcessSpec`` while
    providers still hold the original class, breaking ``isinstance`` checks in
    ``ProcessRunner.run`` for the rest of the suite.
    """
    import ipfs_accelerate_py.cli_runtime.process_runner as mod

    assert hasattr(mod, "ProcessRunner")
    assert hasattr(mod, "ProcessSpec")
    assert callable(mod.run_process)
    assert callable(mod.stream_process)


# ---------------------------------------------------------------------------
# GOOSE-011 security matrix anchors (process runner surface)
# ---------------------------------------------------------------------------


def test_matrix_shell_false_with_metacharacters_and_secret_env() -> None:
    """Argv metacharacters never shell-expand; secret env is redacted."""
    dangerous = "a; rm -rf / && echo pwned"
    recorder = _RecordingPopen()
    runner = ProcessRunner(popen_factory=recorder)
    result = runner.run(_py("import sys; print(sys.argv[1])") + [dangerous])
    assert result.ok
    assert result.stdout.strip() == dangerous
    assert recorder.calls[0]["shell"] is False
    assert dangerous in recorder.calls[0]["argv"]

    redacted = redact_env_mapping(
        {
            "OPENAI_API_KEY": "matrix-cred-secret-value",
            "PROMPT_TEXT": "user private prompt",
            "PATH": "/usr/bin",
        }
    )
    assert redacted["OPENAI_API_KEY"] == REDACTED
    assert redacted["PROMPT_TEXT"] == REDACTED
    assert redacted["PATH"] == "/usr/bin"
    assert "matrix-cred-secret-value" not in str(redacted)
