"""Focused fail-closed shutdown tests for the multi-supervisor runner."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
    vrif_runtime_settlement as runtime_settlement,
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


class _ReapableProcess:
    pid = 424242

    def __init__(self) -> None:
        self.live = True
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self.live else 0

    def terminate(self) -> None:
        self.terminated = True
        self.live = False

    def kill(self) -> None:
        self.killed = True
        self.live = False

    def wait(self, *, timeout: float) -> int:
        del timeout
        if self.live:
            raise subprocess.TimeoutExpired(cmd=("wrapper",), timeout=0.1)
        return 0


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


class _DirectChildStatAdapter:
    def __init__(
        self,
        observations: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        self.observations = list(observations)
        self.last: tuple[int, int, int, int] | None = None

    def _stat(self, _pid: int) -> tuple[int, int, int, int]:
        if self.observations:
            self.last = self.observations.pop(0)
        assert self.last is not None
        return self.last

    def _identity(
        self,
        _pid: int,
        _profile: runner.LifecycleProfile,
    ) -> runner.ProcessIdentity:
        raise PermissionError("credential-bearing child is non-dumpable")

    def identity_alive(self, identity: runner.ProcessIdentity) -> bool:
        assert self.last is not None
        return identity.start_time_ticks == self.last[3]


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


def test_capture_owned_popen_birth_uses_stable_direct_child_stat(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    process = _ManagedProcess(profile, None)
    observation = (os.getpid(), process.pid, process.pid, 12345)
    adapter = _DirectChildStatAdapter((observation, observation))

    identity = runner._capture_owned_popen_birth(
        process,
        profile,
        adapter=adapter,
    )

    assert identity.pid == process.pid
    assert identity.parent_pid == os.getpid()
    assert identity.process_group_id == process.pid
    assert identity.session_id == process.pid
    assert identity.start_time_ticks == 12345
    assert identity.argv == profile.argv
    assert identity.cwd == profile.cwd
    assert identity.profile_id == profile.profile_id
    assert identity.run_id == profile.run_id


def test_legacy_start_track_attaches_birth_when_environ_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("raise SystemExit(0)\n", encoding="utf-8")
    track = runner.SupervisorTrack(
        name="lane-0",
        script_path=wrapper,
        log_path=tmp_path / "lane-0.log",
        supervisor_pid_path=tmp_path / "lane-0.pid",
        daemon_pid_path=tmp_path / "lane-0-daemon.pid",
    )
    process = _StillLiveProcess()
    observation = (os.getpid(), process.pid, process.pid, 12345)
    adapter = _DirectChildStatAdapter((observation, observation))
    monkeypatch.setattr(runner, "LinuxProcessAdapter", lambda: adapter)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    launched = runner.start_track(
        track,
        repo_root=tmp_path.resolve(),
        common_args=(),
        output=lambda _message: None,
    )

    identity = getattr(launched, "_agent_supervisor_process_identity")
    assert isinstance(identity, runner.ProcessIdentity)
    assert identity.pid == process.pid
    assert identity.start_time_ticks == 12345
    assert track.supervisor_pid_path.read_text(encoding="utf-8") == "424242\n"


def test_legacy_start_track_reaps_unadmitted_birth_before_pid_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("raise SystemExit(0)\n", encoding="utf-8")
    track = runner.SupervisorTrack(
        name="lane-0",
        script_path=wrapper,
        log_path=tmp_path / "lane-0.log",
        supervisor_pid_path=tmp_path / "lane-0.pid",
        daemon_pid_path=tmp_path / "lane-0-daemon.pid",
    )
    process = _ReapableProcess()
    foreign = (1, process.pid, process.pid, 12345)
    adapter = _DirectChildStatAdapter((foreign, foreign))
    monkeypatch.setattr(runner, "LinuxProcessAdapter", lambda: adapter)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    with pytest.raises(
        runner.ProcessIdentityMismatch,
        match="direct_child_fenced=true",
    ):
        runner.start_track(
            track,
            repo_root=tmp_path.resolve(),
            common_args=(),
            output=lambda _message: None,
        )

    assert process.terminated is True
    assert process.killed is False
    assert not track.supervisor_pid_path.exists()


@pytest.mark.parametrize(
    ("first", "second", "message"),
    (
        (
            (1, 424242, 424242, 12345),
            (1, 424242, 424242, 12345),
            "exact direct dedicated-session child",
        ),
        (
            (0, 424241, 424242, 12345),
            (0, 424241, 424242, 12345),
            "exact direct dedicated-session child",
        ),
        (
            (0, 424242, 424242, 12345),
            (0, 424242, 424242, 12346),
            "changed during capture",
        ),
    ),
)
def test_capture_owned_popen_birth_rejects_unstable_or_foreign_child(
    tmp_path: Path,
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    message: str,
) -> None:
    profile = _profile(tmp_path)
    process = _ManagedProcess(profile, None)
    normalized_first = (
        os.getpid() if first[0] == 0 else first[0],
        *first[1:],
    )
    normalized_second = (
        os.getpid() if second[0] == 0 else second[0],
        *second[1:],
    )
    adapter = _DirectChildStatAdapter(
        (normalized_first, normalized_second)
    )

    with pytest.raises(runner.ProcessIdentityMismatch, match=message):
        runner._capture_owned_popen_birth(
            process,
            profile,
            adapter=adapter,
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


def test_stop_tracks_retains_markers_for_a_still_live_managed_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path)
    wrapper_pid = 424242
    daemon_pid = 424243
    track = runner.SupervisorTrack(
        name="lane-0",
        script_path=Path("wrapper.py"),
        log_path=Path("lane-0.log"),
        supervisor_pid_path=Path("state/lane-0.pid"),
        daemon_pid_path=Path("state/lane-0-daemon.pid"),
    )
    resolved = track.resolve(tmp_path)
    resolved.supervisor_pid_path.parent.mkdir(parents=True)
    resolved.supervisor_pid_path.write_text(f"{wrapper_pid}\n", encoding="utf-8")
    resolved.daemon_pid_path.write_text(f"{daemon_pid}\n", encoding="utf-8")
    process = _ManagedProcess(profile, None, pid=wrapper_pid)
    process.live = False
    removals: list[Path] = []
    messages: list[str] = []
    monkeypatch.setattr(
        runner,
        "_terminate_managed_process",
        lambda *_args, **_kwargs: (True, (wrapper_pid, daemon_pid)),
    )
    monkeypatch.setattr(runner, "pid_alive", lambda pid: pid == daemon_pid)
    monkeypatch.setattr(
        runner,
        "_remove_stale_pid_marker_if_unchanged",
        lambda path, _pid: removals.append(path) or True,
    )

    result = runner.stop_tracks(
        (track,),
        {track.name: process},
        repo_root=tmp_path,
        grace_seconds=0.01,
        output=messages.append,
    )

    assert result["all_trees_fenced"] is False
    assert result["stopped_count"] == 0
    assert result["removed_runtime_markers"] == []
    assert removals == []
    assert resolved.supervisor_pid_path.exists()
    assert resolved.daemon_pid_path.exists()
    assert any("managed daemon shutdown" in item for item in messages)


def _run_vrif_terminal_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    terminal: bool,
    checkpoint_error: Exception | None = None,
    remove_master: bool = True,
) -> tuple[dict[str, object], list[str]]:
    state = tmp_path / "data/agent_supervisor/residual_intelligence_foundry/state"
    lane = state / "lane-0"
    lane.mkdir(parents=True)
    track = runner.SupervisorTrack(
        name="vrif-lane-0",
        script_path=Path("wrapper.py"),
        log_path=lane / "lane.log",
        supervisor_pid_path=lane / "vrif_lane_0_supervisor.pid",
        daemon_pid_path=lane / "vrif_lane_0_managed_daemon.pid",
    )
    process = _StillLiveProcess()
    events: list[str] = []
    monkeypatch.setattr(runner, "start_track", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(runner, "daemon_pid_health_fields", lambda *_a, **_k: {})
    monkeypatch.setattr(
        runner,
        "supervisor_status_health_fields",
        lambda *_a, **_k: {"restart_supervisor": False},
    )
    monkeypatch.setattr(runner, "format_supervisor_status_fields", lambda _v: "")
    monkeypatch.setattr(runner, "format_daemon_heartbeat_fields", lambda _v: "")
    monkeypatch.setattr(
        runner,
        "terminal_task_state_fields",
        lambda *_a, **_k: {"terminal_quiescent": terminal},
    )

    def stop(*_args, **_kwargs):
        events.append("stop")
        return {
            "stopped_count": 1,
            "all_trees_fenced": True,
            "removed_runtime_markers": [],
        }

    def checkpoint(*_args, **_kwargs):
        events.append("checkpoint")
        if checkpoint_error is not None:
            raise checkpoint_error
        return {"schema": "test-checkpoint", "checkpointed": True}

    def remove(*_args, **_kwargs):
        events.append("remove-master")
        return remove_master

    monkeypatch.setattr(runner, "stop_tracks", stop)
    monkeypatch.setattr(
        runtime_settlement,
        "checkpoint_vrif_terminal_sidecars",
        checkpoint,
    )
    monkeypatch.setattr(runner, "_remove_owned_pid_projection", remove)
    result = runner.run_supervisor_tracks(
        (track,),
        repo_root=tmp_path,
        common_args=(
            "--merge-target-branch",
            "codex/verified-residual-intelligence-foundry-v1",
        ),
        duration_seconds=0.01,
        heartbeat_interval_seconds=0.001,
        stop_grace_seconds=0.01,
        master_pid_path=(
            state.relative_to(tmp_path) / "configured-board-master.pid"
        ),
        label=runtime_settlement.VRIF_PROGRAM_IDENTIFIER,
        exit_when_all_tracks_terminal=True,
        output=lambda _message: None,
    )
    return result, events


def test_terminal_checkpoint_runs_after_fencing_before_master_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, events = _run_vrif_terminal_hook(
        tmp_path,
        monkeypatch,
        terminal=True,
    )

    assert events == ["stop", "checkpoint", "remove-master"]
    assert result["completed"] is True
    assert result["terminal_quiescent"] is True
    assert result["terminal_sidecar_checkpoint"] == {
        "schema": "test-checkpoint",
        "checkpointed": True,
    }


def test_terminal_checkpoint_failure_downgrades_terminal_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, events = _run_vrif_terminal_hook(
        tmp_path,
        monkeypatch,
        terminal=True,
        checkpoint_error=RuntimeError("checkpoint unavailable"),
    )

    assert events == ["stop", "checkpoint", "remove-master"]
    assert result["completed"] is False
    assert result["terminal_quiescent"] is False
    assert "terminal sidecar checkpoint failed" in str(result["blocked"])
    assert result["terminal_sidecar_checkpoint"] == {
        "checkpointed": False,
        "error_type": "RuntimeError",
        "error": "checkpoint unavailable",
    }


def test_terminal_checkpoint_is_not_run_for_duration_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, events = _run_vrif_terminal_hook(
        tmp_path,
        monkeypatch,
        terminal=False,
    )

    assert events == ["stop", "remove-master"]
    assert result["terminal_quiescent"] is False
    assert result["terminal_sidecar_checkpoint"] is None


def test_terminal_checkpoint_requires_master_marker_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, events = _run_vrif_terminal_hook(
        tmp_path,
        monkeypatch,
        terminal=True,
        remove_master=False,
    )

    assert events == ["stop", "checkpoint", "remove-master"]
    assert result["completed"] is False
    assert result["terminal_quiescent"] is False
    assert result["master_pid_removed"] is False
    assert result["blocked"] == "VRIF terminal master PID marker retirement failed"


def test_non_plan_runner_exit_code_propagates_fail_closed_result() -> None:
    assert runner._supervisor_run_exit_code(
        {"completed": False, "all_trees_fenced": True},
        plan_bound_wave=False,
    ) == 2
    assert runner._supervisor_run_exit_code(
        {"completed": True, "all_trees_fenced": False},
        plan_bound_wave=False,
    ) == 2
    assert runner._supervisor_run_exit_code(
        {"completed": True, "all_trees_fenced": True},
        plan_bound_wave=False,
    ) == 0
