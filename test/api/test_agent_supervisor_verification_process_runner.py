"""Tests for the admitted explicit-argv verification process runner."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive
from ipfs_accelerate_py.agent_supervisor.verification.contracts import TerminalStatus
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    PROCESS_RUNNER_SCHEMA,
    VerificationCancellation,
    VerificationCommand,
    VerificationProcessPolicyError,
    VerificationProcessRunner,
    VerificationProcessRunnerError,
    VerificationRunDisposition,
    VerificationSandboxIdentity,
    build_closed_sandbox,
    build_hermetic_environment,
    fence_process_tree,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _host() -> HostResourceSnapshot:
    return HostResourceSnapshot(
        worker_limit=16,
        available_worker_capacity=16,
        active_workers=0,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_available_bytes=8 * 1024 * 1024 * 1024,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        disk_total_bytes=64 * 1024 * 1024 * 1024,
        capabilities=("cpu",),
        resource_classes=(
            "cpu-validation",
            "cpu-proof-type-check",
            "cpu-small",
        ),
    )


def _sandbox(tmp_path: Path) -> VerificationSandboxIdentity:
    source = tmp_path / "source"
    artifacts = tmp_path / "artifacts"
    source.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return build_closed_sandbox(source_root=source, artifact_root=artifacts)


def _command(
    tmp_path: Path,
    argv: list[str],
    *,
    timeout_seconds: float = 10.0,
    max_stdout_bytes: int = 64 * 1024,
    max_stderr_bytes: int = 64 * 1024,
    environment: dict[str, str] | None = None,
    cwd: Path | None = None,
    sandbox: VerificationSandboxIdentity | None = None,
    stdin: bytes | str | None = None,
) -> VerificationCommand:
    box = sandbox or _sandbox(tmp_path)
    work = cwd or Path(box.source_root)
    env = environment if environment is not None else build_hermetic_environment(
        path=os.environ.get("PATH", "/usr/bin:/bin")
    )
    # Ensure Python can run child snippets without ambient secrets.
    env = {
        **env,
        "PATH": env.get("PATH") or os.environ.get("PATH", "/usr/bin:/bin"),
    }
    return VerificationCommand(
        argv=argv,
        cwd=str(work),
        environment=env,
        timeout_seconds=timeout_seconds,
        sandbox=box,
        network_policy=NETWORK_POLICY_DENY_ALL,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        stdin=stdin,
        lane_id=f"test-lane:{tmp_path.name}",
    )


def _py(*code_lines: str) -> list[str]:
    return [sys.executable, "-c", "\n".join(code_lines)]


def _runner(**kwargs: Any) -> VerificationProcessRunner:
    return VerificationProcessRunner(
        resource_scheduler=ResourceScheduler(),
        host_snapshot=_host(),
        **kwargs,
    )


class _RecordingPopen:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.processes: list[Any] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append({"argv": list(argv), **kwargs})
        assert kwargs.get("shell") is False
        proc = subprocess.Popen(argv, **kwargs)
        self.processes.append(proc)
        return proc


def _wait_until_dead(pid: int, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not pid_alive(pid), f"pid {pid} still alive"


# ---------------------------------------------------------------------------
# Construction / policy
# ---------------------------------------------------------------------------


def test_shell_string_argv_is_impossible(tmp_path: Path) -> None:
    box = _sandbox(tmp_path)
    with pytest.raises(VerificationProcessPolicyError) as excinfo:
        VerificationCommand(
            argv="echo hello",  # type: ignore[arg-type]
            cwd=str(box.source_root),
            environment=build_hermetic_environment(),
            timeout_seconds=1.0,
            sandbox=box,
        )
    assert excinfo.value.reason_code == "shell_interpolation_impossible"


def test_metacharacters_are_one_argv_item_not_interpolated(tmp_path: Path) -> None:
    dangerous = "hello world; rm -rf / && echo pwned | cat `id` $HOME"
    recorder = _RecordingPopen()
    runner = _runner(popen_factory=recorder)
    result = runner.run(
        _command(
            tmp_path,
            _py("import sys", "print(repr(sys.argv[1]))") + [dangerous],
        )
    )
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.exit_code == 0
    assert dangerous in result.stdout.preview
    assert recorder.calls
    assert recorder.calls[0]["shell"] is False
    assert dangerous in recorder.calls[0]["argv"]
    assert "pwned" not in result.stdout.preview or dangerous in result.stdout.preview


def test_never_enables_shell_true(tmp_path: Path) -> None:
    recorder = _RecordingPopen()
    runner = _runner(popen_factory=recorder)
    runner.run(_command(tmp_path, _py("print(1)")))
    assert recorder.calls[0]["shell"] is False
    if os.name == "nt":
        flags = recorder.calls[0].get("creationflags", 0)
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert recorder.calls[0].get("start_new_session") is True


def test_auto_install_argv_refused(tmp_path: Path) -> None:
    box = _sandbox(tmp_path)
    with pytest.raises(VerificationProcessPolicyError) as excinfo:
        VerificationCommand(
            argv=[sys.executable, "-m", "pip", "install", "requests"],
            cwd=str(box.source_root),
            environment=build_hermetic_environment(),
            timeout_seconds=1.0,
            sandbox=box,
        )
    assert excinfo.value.reason_code == "auto_install_denied"


def test_network_policy_must_be_deny_all(tmp_path: Path) -> None:
    box = _sandbox(tmp_path)
    with pytest.raises(VerificationProcessPolicyError):
        VerificationCommand(
            argv=_py("print(1)"),
            cwd=str(box.source_root),
            environment=build_hermetic_environment(),
            timeout_seconds=1.0,
            sandbox=box,
            network_policy="loopback_only",
        )


def test_forbidden_environment_keys_refused(tmp_path: Path) -> None:
    box = _sandbox(tmp_path)
    with pytest.raises(VerificationProcessPolicyError):
        VerificationCommand(
            argv=_py("print(1)"),
            cwd=str(box.source_root),
            environment={**build_hermetic_environment(), "OPENAI_API_KEY": "sk-test"},
            timeout_seconds=1.0,
            sandbox=box,
        )


def test_cwd_must_stay_inside_sandbox(tmp_path: Path) -> None:
    box = _sandbox(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(VerificationProcessPolicyError) as excinfo:
        VerificationCommand(
            argv=_py("print(1)"),
            cwd=str(outside),
            environment=build_hermetic_environment(),
            timeout_seconds=1.0,
            sandbox=box,
        )
    assert excinfo.value.reason_code == "cwd_escape"


def test_widened_sandbox_policy_refused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifacts = tmp_path / "artifacts"
    source.mkdir()
    artifacts.mkdir()
    with pytest.raises(VerificationProcessPolicyError):
        VerificationCommand(
            argv=_py("print(1)"),
            cwd=str(source),
            environment=build_hermetic_environment(),
            timeout_seconds=1.0,
            sandbox=VerificationSandboxIdentity(
                sandbox_policy={
                    "schema": "hermetic-sandbox-policy@1",
                    "network": "allow",
                    "auto_install": "deny",
                    "home_cache": "deny",
                    "auth_material": "deny",
                },
                filesystem_policy={
                    "schema": "verification-filesystem-policy@1",
                    "source": "read_only",
                    "artifacts": "private_writable",
                },
                source_root=str(source),
                artifact_root=str(artifacts),
            ),
        )


# ---------------------------------------------------------------------------
# Observation of explicit fields
# ---------------------------------------------------------------------------


def test_result_observes_executable_argv_cwd_sandbox_network_timeout(
    tmp_path: Path,
) -> None:
    box = _sandbox(tmp_path)
    command = _command(
        tmp_path,
        _py("import os,sys", "print(os.getcwd())", "print(sys.argv[0])"),
        sandbox=box,
        timeout_seconds=5.0,
    )
    result = _runner().run(command)
    assert result.schema == PROCESS_RUNNER_SCHEMA
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.exit_code == 0
    assert result.executable == str(Path(sys.executable).resolve())
    assert result.command_argv[0] == result.executable
    assert result.cwd == str(Path(box.source_root).resolve())
    assert result.network_policy == NETWORK_POLICY_DENY_ALL
    assert result.timeout_seconds == 5.0
    assert result.sandbox["sandbox_id"] == box.sandbox_id
    assert result.sandbox["filesystem_policy"]["source"] == "read_only"
    assert result.sandbox["sandbox_policy"]["network"] == "deny"
    assert result.sandbox["sandbox_policy"]["auto_install"] == "deny"
    assert result.process_started is True
    assert result.publication_allowed is True
    assert result.lease_id.startswith("resource-lease:")
    assert Path(box.source_root).resolve().as_posix() in result.stdout.preview.replace(
        "\\", "/"
    )


# ---------------------------------------------------------------------------
# Unavailable
# ---------------------------------------------------------------------------


def test_missing_executable_is_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "source" / "no-such-tool"
    (tmp_path / "source").mkdir(exist_ok=True)
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    # Bypass command executable existence check by pointing argv[0] absolute missing.
    # VerificationCommand does not require executable existence at construction;
    # the runner does at run time.
    box = build_closed_sandbox(
        source_root=tmp_path / "source",
        artifact_root=tmp_path / "artifacts",
    )
    command = VerificationCommand(
        argv=[str(missing), "--version"],
        cwd=str(box.source_root),
        environment=build_hermetic_environment(),
        timeout_seconds=1.0,
        sandbox=box,
    )
    result = _runner().run(command)
    assert result.terminal_status is TerminalStatus.UNAVAILABLE
    assert result.disposition is VerificationRunDisposition.UNAVAILABLE
    assert result.unavailable is True
    assert result.process_started is False
    assert result.publication_allowed is False
    assert "executable" in result.reason_codes[0] or result.reason_codes[0] in {
        "executable_missing",
        "executable_not_file",
        "executable_not_executable",
    }


def test_relative_executable_is_unavailable(tmp_path: Path) -> None:
    command = _command(tmp_path, ["python3", "-c", "print(1)"])
    # Force relative argv0 after construction by rebuilding with a relative path
    # that survives normalization (string is non-empty).
    object.__setattr__(command, "argv", ("python3", "-c", "print(1)"))
    result = _runner().run(command)
    assert result.terminal_status is TerminalStatus.UNAVAILABLE
    assert result.reason_codes[0] == "executable_not_absolute"


def test_missing_sandbox_roots_unavailable(tmp_path: Path) -> None:
    with pytest.raises(VerificationProcessRunnerError) as excinfo:
        build_closed_sandbox(
            source_root=tmp_path / "missing-source",
            artifact_root=tmp_path / "missing-artifacts",
        )
    assert excinfo.value.reason_code == "sandbox_unavailable"


# ---------------------------------------------------------------------------
# Timeout / cancel / late publication fence
# ---------------------------------------------------------------------------


def test_timeout_is_timeout(tmp_path: Path) -> None:
    result = _runner().run(
        _command(
            tmp_path,
            _py("import time", "time.sleep(30)"),
            timeout_seconds=0.2,
        )
    )
    assert result.terminal_status is TerminalStatus.TIMEOUT
    assert result.disposition is VerificationRunDisposition.TIMEOUT
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.publication_allowed is False
    assert "timeout" in result.reason_codes
    assert result.process_started is True
    if result.pid is not None:
        _wait_until_dead(result.pid)


def test_cancellation_is_cancelled_and_fences_late_publication(
    tmp_path: Path,
) -> None:
    cancel = VerificationCancellation(cancellation_id="cancel:test-late-success")
    started = threading.Event()

    def cancel_soon() -> None:
        started.wait(timeout=2.0)
        time.sleep(0.05)
        cancel.cancel(cancellation_id="cancel:test-late-success", reason="operator-abort")

    thread = threading.Thread(target=cancel_soon, daemon=True)
    thread.start()

    # Child writes a ready marker then sleeps; if cancellation is late after a
    # quick success path we still fence publication via the token check.
    marker = tmp_path / "source" / "ready.txt"
    result = _runner().run(
        _command(
            tmp_path,
            _py(
                "import pathlib, time, sys",
                f"pathlib.Path({str(marker)!r}).write_text('ready')",
                "time.sleep(30)",
            ),
            timeout_seconds=10.0,
        ),
        cancellation=cancel,
    )
    # Ensure cancel thread saw process start window; if process finished before
    # cancel, pre-mark the event so the helper thread exits.
    started.set()
    thread.join(timeout=2.0)

    assert result.terminal_status is TerminalStatus.CANCELLED
    assert result.disposition is VerificationRunDisposition.CANCELLED
    assert result.cancelled is True
    assert result.publication_allowed is False
    assert result.cancellation_id == "cancel:test-late-success"
    if result.pid is not None:
        _wait_until_dead(result.pid)


def test_pre_spawn_cancellation_fences_publication(tmp_path: Path) -> None:
    cancel = VerificationCancellation(cancellation_id="cancel:pre")
    cancel.cancel(cancellation_id="cancel:pre", reason="preempted")
    result = _runner().run(
        _command(tmp_path, _py("print('should-not-run')")),
        cancellation=cancel,
    )
    assert result.terminal_status is TerminalStatus.CANCELLED
    assert result.process_started is False
    assert result.publication_allowed is False
    assert result.reason_codes[0] == "cancelled_before_spawn"


def test_late_success_after_cancel_is_not_publishable(tmp_path: Path) -> None:
    """Even if the child exits 0, a set cancellation token fences publication."""

    cancel = VerificationCancellation(cancellation_id="cancel:fence-success")

    class _ImmediateCancelPopen:
        """Popen that cancels as soon as the child is created."""

        def __init__(self) -> None:
            self.inner: Any = None

        def __call__(self, argv: list[str], **kwargs: Any) -> Any:
            self.inner = subprocess.Popen(argv, **kwargs)
            # Cancel after spawn but before the runner observes completion.
            cancel.cancel(cancellation_id="cancel:fence-success", reason="fence")
            return self.inner

    factory = _ImmediateCancelPopen()
    runner = _runner(popen_factory=factory)
    result = runner.run(
        _command(
            tmp_path,
            _py("print('ok')"),
            timeout_seconds=5.0,
        ),
        cancellation=cancel,
    )
    assert result.cancelled is True
    assert result.terminal_status is TerminalStatus.CANCELLED
    assert result.publication_allowed is False
    # Exit code may be 0 if the child finished first; publication still fenced.
    if result.exit_code == 0:
        assert result.ok is False


def test_cancellation_identity_mismatch_is_ignored(tmp_path: Path) -> None:
    cancel = VerificationCancellation(cancellation_id="cancel:owner")
    assert cancel.cancel(cancellation_id="cancel:attacker") is False
    assert cancel.is_cancelled() is False
    result = _runner().run(
        _command(tmp_path, _py("print('alive')")),
        cancellation=cancel,
    )
    assert result.terminal_status is TerminalStatus.PASSED
    assert result.publication_allowed is True


# ---------------------------------------------------------------------------
# Deterministic truncation and digests
# ---------------------------------------------------------------------------


def test_stdout_stderr_truncate_deterministically_with_artifact_digests(
    tmp_path: Path,
) -> None:
    max_out = 64
    max_err = 32
    result = _runner().run(
        _command(
            tmp_path,
            _py(
                "import sys",
                "sys.stdout.buffer.write(b'X' * 10_000)",
                "sys.stderr.buffer.write(b'Y' * 10_000)",
            ),
            max_stdout_bytes=max_out,
            max_stderr_bytes=max_err,
        )
    )
    assert result.stdout.truncated is True
    assert result.stderr.truncated is True
    assert result.stdout.captured_byte_count == max_out
    assert result.stderr.captured_byte_count == max_err
    assert result.stdout.byte_count >= max_out
    assert result.stderr.byte_count >= max_err
    assert result.stdout.digest.startswith("sha256:")
    assert result.stderr.digest.startswith("sha256:")
    # Digests address captured bytes (deterministic for the retained artifact).
    assert result.stdout.digest == "sha256:" + hashlib.sha256(b"X" * max_out).hexdigest()
    assert result.stderr.digest == "sha256:" + hashlib.sha256(b"Y" * max_err).hexdigest()
    assert result.stdout.cid == cid_for_bytes(b"X" * max_out)
    assert result.stderr.cid == cid_for_bytes(b"Y" * max_err)
    assert "stdout_truncated" in result.reason_codes
    assert "stderr_truncated" in result.reason_codes

    # Second run yields identical digests for the same captured payload.
    again = _runner().run(
        _command(
            tmp_path,
            _py(
                "import sys",
                "sys.stdout.buffer.write(b'X' * 10_000)",
                "sys.stderr.buffer.write(b'Y' * 10_000)",
            ),
            max_stdout_bytes=max_out,
            max_stderr_bytes=max_err,
        )
    )
    assert again.stdout.digest == result.stdout.digest
    assert again.stderr.digest == result.stderr.digest
    assert again.stdout.cid == result.stdout.cid
    assert again.stderr.cid == result.stderr.cid


def test_untruncated_output_digest_matches_payload(tmp_path: Path) -> None:
    payload = b"deterministic-payload-42\n"
    result = _runner().run(
        _command(
            tmp_path,
            _py(
                "import sys",
                f"sys.stdout.buffer.write({payload!r})",
            ),
        )
    )
    assert result.stdout.truncated is False
    assert result.stdout.digest == "sha256:" + hashlib.sha256(payload).hexdigest()
    assert result.stdout.cid == cid_for_bytes(payload)
    assert payload.decode() in result.stdout.preview


# ---------------------------------------------------------------------------
# Process tree cancellation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="process-tree fencing requires Linux /proc sessions",
)
def test_cancellation_kills_child_grandchild_and_escaped_session(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    child_pid_path = source / "child.pid"
    grand_pid_path = source / "grand.pid"
    escaped_pid_path = source / "escaped.pid"

    # Layered scripts avoid nested quoting issues:
    # root -> child (same session) -> grandchild (same session) -> escaped (new session)
    escaped_script = source / "escaped.py"
    escaped_script.write_text(
        "import time\ntime.sleep(120)\n",
        encoding="utf-8",
    )
    grand_script = source / "grand.py"
    grand_script.write_text(
        "\n".join(
            [
                "import pathlib",
                "import subprocess",
                "import sys",
                "import time",
                f"escaped = subprocess.Popen([sys.executable, {str(escaped_script)!r}], start_new_session=True)",
                f"pathlib.Path({str(escaped_pid_path)!r}).write_text(str(escaped.pid))",
                f"pathlib.Path({str(grand_pid_path)!r}).write_text(str(__import__('os').getpid()))",
                "time.sleep(120)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    child_script = source / "child.py"
    child_script.write_text(
        "\n".join(
            [
                "import pathlib",
                "import subprocess",
                "import sys",
                "import time",
                f"grand = subprocess.Popen([sys.executable, {str(grand_script)!r}])",
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(__import__('os').getpid()))",
                "time.sleep(120)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    root_script = source / "root.py"
    root_script.write_text(
        "\n".join(
            [
                "import subprocess",
                "import sys",
                "import time",
                f"subprocess.Popen([sys.executable, {str(child_script)!r}])",
                "time.sleep(120)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cancel = VerificationCancellation(cancellation_id="cancel:tree")
    ready = threading.Event()

    def arm() -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (
                child_pid_path.exists()
                and grand_pid_path.exists()
                and escaped_pid_path.exists()
            ):
                ready.set()
                time.sleep(0.15)
                cancel.cancel(cancellation_id="cancel:tree", reason="tree-fence")
                return
            time.sleep(0.02)

    thread = threading.Thread(target=arm, daemon=True)
    thread.start()
    result = _runner().run(
        _command(
            tmp_path,
            [sys.executable, str(root_script)],
            timeout_seconds=25.0,
        ),
        cancellation=cancel,
    )
    thread.join(timeout=5.0)
    assert ready.is_set(), (
        "child process tree did not publish PIDs in time; "
        f"stderr={result.stderr.preview!r} reason={result.reason!r} "
        f"status={result.terminal_status!r}"
    )
    assert result.cancelled is True
    assert result.publication_allowed is False

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    grand_pid = int(grand_pid_path.read_text(encoding="utf-8"))
    escaped_pid = int(escaped_pid_path.read_text(encoding="utf-8"))
    assert len({child_pid, grand_pid, escaped_pid}) == 3
    for pid in filter(None, (result.pid, child_pid, grand_pid, escaped_pid)):
        _wait_until_dead(int(pid), timeout=8.0)


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="process-tree fencing requires Linux /proc sessions",
)
def test_fence_process_tree_helper_reaps_escaped_session(tmp_path: Path) -> None:
    escaped_path = tmp_path / "escaped.pid"
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen("
                "[sys.executable, '-c', 'import time; time.sleep(60)'], "
                "start_new_session=True)\n"
                f"pathlib.Path({str(escaped_path)!r}).write_text(str(child.pid))\n"
                "time.sleep(60)\n"
            ),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 3.0
        while not escaped_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        escaped_pid = int(escaped_path.read_text(encoding="utf-8"))
        assert fence_process_tree(parent, grace_seconds=0.2, require_gone=True)
        _wait_until_dead(escaped_pid)
        _wait_until_dead(parent.pid)
    finally:
        if parent.poll() is None:
            try:
                os.killpg(parent.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            parent.wait(timeout=1.0)


# ---------------------------------------------------------------------------
# Exit status / lease / no mock path
# ---------------------------------------------------------------------------


def test_nonzero_exit_is_failed_not_timeout(tmp_path: Path) -> None:
    result = _runner().run(
        _command(tmp_path, _py("import sys", "sys.exit(7)"))
    )
    assert result.terminal_status is TerminalStatus.FAILED
    assert result.disposition is VerificationRunDisposition.FAILED
    assert result.exit_code == 7
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.publication_allowed is True
    assert result.ok is False


def test_resource_lease_acquired_and_released(tmp_path: Path) -> None:
    scheduler = ResourceScheduler()
    runner = VerificationProcessRunner(
        resource_scheduler=scheduler,
        host_snapshot=_host(),
    )
    assert scheduler.active_leases == ()
    result = runner.run(_command(tmp_path, _py("print('leased')")))
    assert result.lease_id
    assert result.terminal_status is TerminalStatus.PASSED
    assert scheduler.active_leases == ()


def test_resource_lease_denial_is_unavailable(tmp_path: Path) -> None:
    exhausted = HostResourceSnapshot(
        worker_limit=0,
        available_worker_capacity=0,
        active_workers=0,
        memory_available_bytes=0,
        disk_available_bytes=0,
    )
    runner = VerificationProcessRunner(
        resource_scheduler=ResourceScheduler(),
        host_snapshot=exhausted,
    )
    result = runner.run(_command(tmp_path, _py("print(1)")))
    assert result.terminal_status is TerminalStatus.UNAVAILABLE
    assert result.reason_codes[0] == "resource_lease_denied"
    assert result.publication_allowed is False
    assert result.process_started is False


def test_result_to_dict_is_json_safe_and_explicit(tmp_path: Path) -> None:
    result = _runner().run(_command(tmp_path, _py("print('hi')")))
    payload = result.to_dict()
    assert payload["schema"] == PROCESS_RUNNER_SCHEMA
    assert payload["network_policy"] == NETWORK_POLICY_DENY_ALL
    assert payload["sandbox"]["sandbox_policy"]["auto_install"] == "deny"
    assert "mock" not in str(payload).lower()
    assert "install" not in payload["command_argv"]
    assert isinstance(payload["stdout"]["digest"], str)
    assert isinstance(payload["stderr"]["cid"], str)


def test_no_ambient_secret_inheritance(tmp_path: Path) -> None:
    """Child environment is only the explicit hermetic mapping."""

    result = _runner().run(
        _command(
            tmp_path,
            _py(
                "import os",
                "print('OPENAI' if 'OPENAI_API_KEY' in os.environ else 'clean')",
                "print(os.environ.get('LANG', ''))",
            ),
            environment=build_hermetic_environment(
                path=str(Path(sys.executable).parent)
                + os.pathsep
                + os.environ.get("PATH", "/usr/bin:/bin")
            ),
        )
    )
    assert "clean" in result.stdout.preview
    assert "C.UTF-8" in result.stdout.preview
    assert "OPENAI" not in result.stdout.preview or "clean" in result.stdout.preview


def test_stdin_round_trip(tmp_path: Path) -> None:
    result = _runner().run(
        _command(
            tmp_path,
            _py("import sys", "print(sys.stdin.read(), end='')"),
            stdin="payload-via-stdin\n",
        )
    )
    assert result.exit_code == 0
    assert "payload-via-stdin" in result.stdout.preview
