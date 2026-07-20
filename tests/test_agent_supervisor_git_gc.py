from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.git_gc import _run_git


def test_run_git_returns_captured_result(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)

    result = _run_git(["status", "--short"], cwd=tmp_path, timeout=5)

    assert result.returncode == 0
    assert result.stdout == ""


def test_run_git_timeout_terminates_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class TimedOutProcess:
        pid = 43210
        returncode = -15
        calls = 0

        def communicate(self, timeout: float | None = None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["git"], timeout or 0)
            return "partial-out", "partial-err"

    process = TimedOutProcess()
    popen_kwargs: dict[str, object] = {}
    signals: list[tuple[int, int]] = []

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("os.killpg", lambda pid, sig: signals.append((pid, sig)))

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _run_git(["gc", "--aggressive"], cwd=tmp_path, timeout=0.01)

    assert popen_kwargs["start_new_session"] is True
    assert signals == [(process.pid, signal.SIGTERM)]
    assert exc_info.value.output == "partial-out"
