from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.integrations import (
    llm_merge_resolver_fallback as resolver,
)


def test_explicit_lock_bypass_does_not_require_git_locking(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_RESOLVER_LOCK_BYPASS", "1")

    def unexpected_git_lookup(_workspace):
        raise AssertionError("explicit lock bypass must not inspect git state")

    monkeypatch.setattr(resolver, "_git_common_dir", unexpected_git_lookup)

    assert resolver._acquire_git_lock(tmp_path) is None


def test_main_fails_closed_when_git_lock_setup_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_RESOLVER_LOCK_BYPASS", raising=False)
    monkeypatch.setenv("_AGENT_RESOLVER_INVOCATION_DEPTH", "0")
    monkeypatch.setattr(resolver, "_git_common_dir", lambda _workspace: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("resolve this conflict"))

    def unexpected_resolver(*_args, **_kwargs):
        raise AssertionError("a resolver must not run without the git lock")

    monkeypatch.setattr(resolver, "_run_codex", unexpected_resolver)
    monkeypatch.setattr(resolver, "_run_copilot", unexpected_resolver)

    assert resolver.main([str(tmp_path)]) == resolver._LOCK_FAILURE_EXIT_CODE
    assert capsys.readouterr().err == (
        "error: merge resolver lock unavailable: "
        "could not determine the git common directory\n"
    )


def test_module_fails_closed_when_git_lock_times_out(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "-C", str(repo), "init", "--quiet"],
        text=True,
        capture_output=True,
        check=True,
    )
    lock_path = repo / ".git" / "agent-llm-resolver.lock"
    resolver_marker = tmp_path / "resolver-invoked"
    resolver_bin = tmp_path / "codex"
    resolver_bin.write_text(
        "#!/usr/bin/env bash\n"
        'printf invoked > "$RESOLVER_MARKER"\n',
        encoding="utf-8",
    )
    resolver_bin.chmod(0o755)

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "CODEX_BIN": str(resolver_bin),
        "RESOLVER_MARKER": str(resolver_marker),
        "AGENT_RESOLVER_LOCK_TIMEOUT_SECONDS": "0.05",
        "_AGENT_RESOLVER_INVOCATION_DEPTH": "0",
    }
    env.pop("AGENT_RESOLVER_LOCK_BYPASS", None)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ipfs_accelerate_py.agent_supervisor.integrations.llm_merge_resolver_fallback",
                str(repo),
            ],
            input="resolve this conflict",
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    assert completed.returncode == resolver._LOCK_FAILURE_EXIT_CODE
    assert completed.stderr == (
        "error: merge resolver lock unavailable: "
        "acquisition timed out after 0.05s\n"
    )
    assert not resolver_marker.exists()
