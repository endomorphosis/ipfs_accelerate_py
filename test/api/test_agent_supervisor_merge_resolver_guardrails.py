from __future__ import annotations

import builtins
import errno
import fcntl
import io
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations import (
    llm_merge_resolver_fallback as resolver,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor


def _git_init(path) -> None:
    path.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )


def test_merge_resolver_lock_timeout_fails_closed(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "repo"
    _git_init(workspace)
    common_dir = resolver._git_common_dir(workspace)
    lock_path = common_dir / "agent-llm-resolver.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setenv(resolver._LOCK_TIMEOUT_ENV, "0")

    try:
        with pytest.raises(
            resolver._GitLockAcquisitionError,
            match="checkout lock acquisition timed out",
        ):
            resolver._acquire_git_lock(workspace)
    finally:
        lock_handle.close()


def test_merge_resolver_common_dir_failure_fails_before_provider(
    tmp_path, monkeypatch, capsys
) -> None:
    provider_invoked = False

    def fail_provider(_prompt, _workspace):
        nonlocal provider_invoked
        provider_invoked = True
        return 0, False

    monkeypatch.delenv("AGENT_RESOLVER_LOCK_BYPASS", raising=False)
    monkeypatch.setattr(resolver, "_run_grok", fail_provider)
    monkeypatch.setattr(sys, "stdin", io.StringIO("resolve"))

    result = resolver.main([str(tmp_path / "not-a-repository")])

    assert result == resolver._LOCK_ACQUISITION_FAILURE_EXIT_CODE
    assert provider_invoked is False
    assert "no provider was invoked" in capsys.readouterr().err


def test_merge_resolver_lock_open_failure_is_not_a_bypass(
    tmp_path, monkeypatch
) -> None:
    common_dir_file = tmp_path / "common-dir-is-a-file"
    common_dir_file.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.delenv("AGENT_RESOLVER_LOCK_BYPASS", raising=False)
    monkeypatch.setattr(resolver, "_git_common_dir", lambda _workspace: common_dir_file)

    with pytest.raises(
        resolver._GitLockAcquisitionError,
        match="could not open the checkout lock",
    ):
        resolver._acquire_git_lock(tmp_path)


def test_merge_resolver_fcntl_import_failure_is_not_a_bypass(
    tmp_path, monkeypatch
) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("fcntl unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("AGENT_RESOLVER_LOCK_BYPASS", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(
        resolver._GitLockAcquisitionError,
        match="does not provide fcntl",
    ):
        resolver._acquire_git_lock(tmp_path)


def test_merge_resolver_flock_error_is_not_treated_as_contention(
    tmp_path, monkeypatch
) -> None:
    workspace = tmp_path / "repo"
    _git_init(workspace)
    monkeypatch.delenv("AGENT_RESOLVER_LOCK_BYPASS", raising=False)

    def fail_flock(_descriptor, _operation):
        raise OSError(errno.EIO, "simulated flock I/O failure")

    monkeypatch.setattr(fcntl, "flock", fail_flock)

    with pytest.raises(
        resolver._GitLockAcquisitionError,
        match="checkout flock failed",
    ):
        resolver._acquire_git_lock(workspace)


def test_merge_resolver_explicit_test_lock_bypass_remains_available(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_RESOLVER_LOCK_BYPASS", "1")
    monkeypatch.setattr(
        resolver,
        "_git_common_dir",
        lambda _workspace: pytest.fail("bypass attempted git discovery"),
    )

    assert resolver._acquire_git_lock(tmp_path) is None


def test_merge_resolver_provider_output_is_bounded_and_truncation_fails_closed(
    tmp_path,
) -> None:
    completed = resolver._run_tool(
        (
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write(b'x' * 400000); "
                "sys.stderr.buffer.write(b'y' * 400000)"
            ),
        ),
        prompt="bounded provider output",
        timeout=5,
    )

    assert completed.returncode == 0
    assert len(completed.stdout.encode("utf-8")) < (
        resolver._MAX_TOOL_OUTPUT_BYTES + 100
    )
    assert len(completed.stderr.encode("utf-8")) < (
        resolver._MAX_TOOL_OUTPUT_BYTES + 100
    )
    assert "stdout truncated" in completed.stdout
    assert "stderr truncated" in completed.stderr
    assert resolver._strict_grok_quota_exhaustion(
        completed.stderr,
        returncode=86,
    ) is False


@pytest.mark.parametrize(
    "cmdline",
    (
        (
            "/usr/bin/python3 -m "
            "ipfs_accelerate_py.agent_supervisor.integrations."
            "llm_merge_resolver_fallback /workspace"
        ),
        (
            "/usr/bin/python3 /checkout/ipfs_accelerate_py/agent_supervisor/"
            "integrations/llm_merge_resolver_fallback.py /workspace"
        ),
        (
            "/checkout/ipfs_accelerate_py/agent_supervisor/integrations/"
            "llm_merge_resolver_fallback.py /workspace"
        ),
    ),
)
def test_worker_watchdog_recognizes_packaged_merge_resolver(
    monkeypatch, cmdline
) -> None:
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: [{"pid": 8123, "cmdline": cmdline}],
    )

    status = supervisor.worktree_phase_worker_status(
        {
            "active_phase": "merge_resolver",
            "active_phase_started_at": (now - timedelta(minutes=10)).isoformat(),
        },
        daemon_pid=1234,
        threshold_seconds=60,
        now=now,
    )

    assert status["active_worker_count"] == 1
    assert status["active_worker_pids"] == [8123]
    assert status["stalled_without_active_worker"] is False
