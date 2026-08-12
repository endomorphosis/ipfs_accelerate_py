from __future__ import annotations

import builtins
import errno
import fcntl
import io
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations import (
    llm_merge_resolver_fallback as resolver,
)
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
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
    runner_invoked = False

    def fail_runner(_argv):
        nonlocal runner_invoked
        runner_invoked = True
        return 0

    monkeypatch.delenv("AGENT_RESOLVER_LOCK_BYPASS", raising=False)
    monkeypatch.setattr(grok_cli_runner, "main", fail_runner)
    monkeypatch.setattr(sys, "stdin", io.StringIO("resolve"))

    result = resolver.main([str(tmp_path / "not-a-repository")])

    assert result == resolver._LOCK_ACQUISITION_FAILURE_EXIT_CODE
    assert runner_invoked is False
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


def test_merge_resolver_main_delegates_once_to_canonical_marked_runner(
    tmp_path, monkeypatch
) -> None:
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    captured: list[tuple[list[str], str]] = []

    def fake_runner_main(argv) -> int:
        captured.append((list(argv), sys.stdin.read()))
        return 23

    monkeypatch.setenv("AGENT_RESOLVER_LOCK_BYPASS", "1")
    monkeypatch.setenv(resolver._INVOCATION_DEPTH_ENV, "0")
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(grok_cli_runner, "main", fake_runner_main)
    monkeypatch.setattr(sys, "stdin", io.StringIO("resolve this conflict"))

    assert resolver.main([str(tmp_path)]) == 23
    assert len(captured) == 1
    argv, prompt = captured[0]
    assert prompt == "resolve this conflict"
    assert argv[:2] == ["--workspace", str(tmp_path)]
    assert grok_cli_runner.CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG in argv
    fallback = json.loads(
        argv[argv.index("--codex-fallback-command-json") + 1]
    )
    assert fallback[0] == str(codex)
    assert fallback[1] == "exec"
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert (
        fallback[fallback.index("-c") + 1]
        == 'model_reasoning_effort="medium"'
    )
    assert "--dangerously-bypass-approvals-and-sandbox" not in fallback
    assert fallback[fallback.index("-s") + 1] == "workspace-write"

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
    now = datetime.now(UTC)
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
