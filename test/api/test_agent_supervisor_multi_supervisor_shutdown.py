"""Focused fail-closed shutdown tests for the multi-supervisor runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)


class _StillLiveProcess:
    pid = 424242

    def wait(self, *, timeout: float) -> None:
        raise subprocess.TimeoutExpired(cmd=("supervisor-wrapper",), timeout=timeout)

    def poll(self) -> None:
        return None


def test_stop_tracks_does_not_retire_a_still_live_wrapper_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An empty lifecycle snapshot cannot hide its live wrapper Popen."""

    track = runner.SupervisorTrack(
        name="lane-0",
        script_path=Path("wrapper.py"),
        log_path=Path("lane-0.log"),
        supervisor_pid_path=Path("lane-0.pid"),
        daemon_pid_path=Path("lane-0-daemon.pid"),
    )
    process = _StillLiveProcess()
    removed = []
    messages: list[str] = []
    monkeypatch.setattr(
        runner,
        "_terminate_managed_process",
        lambda *_args, **_kwargs: (True, (process.pid,)),
    )

    def unexpected_marker_removal(*_args, **_kwargs):
        removed.append((_args, _kwargs))
        return True

    monkeypatch.setattr(
        runner,
        "_remove_stale_pid_marker_if_unchanged",
        unexpected_marker_removal,
    )

    result = runner.stop_tracks(
        (track,),
        {track.name: process},
        repo_root=tmp_path,
        grace_seconds=0.01,
        output=messages.append,
    )

    assert result == {
        "stopped_pids": [],
        "stopped_count": 0,
        "all_trees_fenced": False,
        "removed_runtime_markers": [],
    }
    assert removed == []
    assert any("could not verify complete shutdown" in item for item in messages)
