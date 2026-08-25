"""Focused fail-closed shutdown tests for the multi-supervisor runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)


class _StillLiveProcess:
    pid = 424242

    def __init__(self) -> None:
        self.wait_timeouts: list[float] = []

    def wait(self, *, timeout: float) -> None:
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired(cmd=("supervisor-wrapper",), timeout=timeout)

    def poll(self) -> None:
        return None


class _ManagedProcess:
    def __init__(
        self,
        profile: runner.LifecycleProfile,
        identity: runner.ProcessIdentity | None,
        *,
        pid: int = 424242,
    ) -> None:
        self.pid = pid
        self.live = True
        self._agent_supervisor_lifecycle_profile = profile
        self._agent_supervisor_process_identity = identity

    def poll(self) -> int | None:
        return None if self.live else 0


class _RecordingAdapter:
    def __init__(
        self,
        *,
        profile: runner.LifecycleProfile,
        process: _ManagedProcess,
        marker_members: tuple[runner.ProcessIdentity, ...],
        alive_members: tuple[runner.ProcessIdentity, ...],
    ) -> None:
        self.profile = profile
        self.process = process
        self.marker_members = marker_members
        self.alive_pids = {item.pid for item in alive_members}
        self.terminated_tree: runner.ProcessTreeSnapshot | None = None
        self.termination_arguments: tuple[float, int] | None = None

    def snapshot(
        self,
        profile: runner.LifecycleProfile,
    ) -> runner.ProcessTreeSnapshot:
        assert profile == self.profile
        return runner.ProcessTreeSnapshot(
            profile_id=profile.profile_id,
            run_id=profile.run_id,
            members=tuple(
                item
                for item in self.marker_members
                if item.pid in self.alive_pids
            ),
            captured_at_ms=123,
        )

    def identity_alive(self, identity: runner.ProcessIdentity) -> bool:
        return identity.pid in self.alive_pids

    def terminate(
        self,
        tree: runner.ProcessTreeSnapshot,
        *,
        grace_seconds: float,
        deadline_ms: int,
    ) -> None:
        self.terminated_tree = tree
        self.termination_arguments = (grace_seconds, deadline_ms)
        self.alive_pids.difference_update(item.pid for item in tree.members)
        if self.process.pid in {item.pid for item in tree.members}:
            self.process.live = False


def _profile(tmp_path: Path) -> runner.LifecycleProfile:
    repository_root = tmp_path.resolve()
    state_root = repository_root / "state"
    return runner.LifecycleProfile(
        target_id="supervisor-track:lane-0",
        run_id="multi-supervisor:test-lane-0",
        configuration_root="sha256:test-configuration",
        repository_root=str(repository_root),
        state_root=str(state_root),
        run_root=str(state_root / "lifecycle-runs" / "lane-0"),
        argv=("/usr/bin/python3", "wrapper.py"),
        cwd=str(repository_root),
    )


def _identity(
    profile: runner.LifecycleProfile,
    *,
    pid: int,
    parent_pid: int,
    start_time_ticks: int,
) -> runner.ProcessIdentity:
    return runner.ProcessIdentity(
        pid=pid,
        start_time_ticks=start_time_ticks,
        parent_pid=parent_pid,
        process_group_id=424242,
        session_id=424242,
        boot_id="test-boot-id",
        argv=profile.argv,
        cwd=profile.cwd,
        executable="/usr/bin/python3",
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=0,
        configuration_root=profile.configuration_root,
    )


@pytest.mark.parametrize("marker_child_visible", (False, True))
def test_terminate_managed_process_recovers_marker_hidden_exact_wrapper(
    tmp_path: Path,
    monkeypatch,
    marker_child_visible: bool,
) -> None:
    """The captured wrapper birth supplements, but never widens, marker scope."""

    profile = _profile(tmp_path)
    wrapper = _identity(
        profile,
        pid=424242,
        parent_pid=1,
        start_time_ticks=100,
    )
    child = _identity(
        profile,
        pid=424243,
        parent_pid=wrapper.pid,
        start_time_ticks=101,
    )
    process = _ManagedProcess(profile, wrapper)
    marker_members = (child,) if marker_child_visible else ()
    alive_members = (wrapper, *marker_members)
    adapter = _RecordingAdapter(
        profile=profile,
        process=process,
        marker_members=marker_members,
        alive_members=alive_members,
    )
    monkeypatch.setattr(runner, "LinuxProcessAdapter", lambda: adapter)

    fenced, member_pids = runner._terminate_managed_process(
        process,
        grace_seconds=0.25,
    )

    expected_pids = (wrapper.pid, child.pid) if marker_child_visible else (wrapper.pid,)
    assert fenced is True
    assert member_pids == expected_pids
    assert adapter.terminated_tree is not None
    assert tuple(item.pid for item in adapter.terminated_tree.members) == expected_pids
    assert tuple(item.pid for item in adapter.terminated_tree.roots) == (wrapper.pid,)
    assert adapter.termination_arguments == (0.25, 1_250)
    assert process.poll() == 0


def test_terminate_managed_process_empty_snapshot_without_birth_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A live Popen PID alone never becomes signal authority."""

    profile = _profile(tmp_path)
    process = _ManagedProcess(profile, None)
    adapter = _RecordingAdapter(
        profile=profile,
        process=process,
        marker_members=(),
        alive_members=(),
    )
    monkeypatch.setattr(runner, "LinuxProcessAdapter", lambda: adapter)

    assert runner._terminate_managed_process(
        process,
        grace_seconds=0.25,
    ) == (False, ())
    assert adapter.terminated_tree is None
    assert process.poll() is None


def test_terminate_managed_process_rejects_mismatched_birth_before_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A birth for another PID cannot authorize marker or process signals."""

    profile = _profile(tmp_path)
    foreign_birth = _identity(
        profile,
        pid=525252,
        parent_pid=1,
        start_time_ticks=100,
    )
    process = _ManagedProcess(profile, foreign_birth)

    def unexpected_adapter():
        raise AssertionError("mismatched birth must fail before process scanning")

    monkeypatch.setattr(runner, "LinuxProcessAdapter", unexpected_adapter)

    with pytest.raises(
        runner.ProcessIdentityMismatch,
        match="does not match its lifecycle profile",
    ):
        runner._terminate_managed_process(process, grace_seconds=0.25)


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
    assert process.wait_timeouts == []
    assert removed == []
    assert any("could not verify complete shutdown" in item for item in messages)
