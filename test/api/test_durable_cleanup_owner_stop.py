"""Owner STOP retains exact cleanup custody before declaring process absence.

Process and Docker observations are authored doubles. These tests do not start
containers, invoke a provider, or qualify the disabled native launch contract.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import signal
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as runner
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    LifecycleProfile, ProcessIdentity, ProcessTreeSnapshot,
)


@pytest.fixture
def owner(tmp_path, monkeypatch):
    profile = LifecycleProfile(
        target_id="supervisor-track:cleanup-owner-test", run_id="cleanup-owner-run",
        configuration_root="cleanup-owner-configuration", repository_root=str(tmp_path),
        state_root=str(tmp_path / "state"), run_root=str(tmp_path / "state/run"),
        argv=(sys.executable, "owned-supervisor.py"), cwd=str(tmp_path),
    )

    def identity(pid, parent, *, watchdog=False):
        return ProcessIdentity(
            pid=pid, start_time_ticks=pid + 100, parent_pid=parent,
            process_group_id=pid, session_id=pid, boot_id="owned-boot",
            argv=(sys.executable, "-m", "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
                  "--internal-docker-cleanup-watchdog") if watchdog else profile.argv,
            cwd=profile.cwd, executable=sys.executable, run_id=profile.run_id,
            profile_id=profile.profile_id, target_id=profile.target_id,
            repository_root=profile.repository_root, state_root=profile.state_root,
            run_root=profile.run_root, fencing_epoch=0,
            configuration_root=profile.configuration_root,
        )

    root = identity(810001, 1)
    provider = identity(810002, root.pid)
    watchdog = identity(810003, 1, watchdog=True)
    events = []
    clock = [10.0]

    class Process:
        pid = root.pid
        returncode = None
        _agent_supervisor_lifecycle_profile = profile
        _agent_supervisor_process_identity = root

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append(("root-term", self.pid))

        def kill(self):
            events.append(("root-kill", self.pid))
            self.returncode = -signal.SIGKILL

    process = Process()
    anchor = object()
    process._agent_supervisor_cleanup_directory_anchor = anchor

    class Adapter:
        members = (root, provider, watchdog)

        def snapshot(self, selected):
            assert selected is profile
            return ProcessTreeSnapshot(profile_id=profile.profile_id, run_id=profile.run_id,
                                       members=tuple(self.members))

        def identity_alive(self, member):
            return any(item.identity_id == member.identity_id for item in self.members)

        def terminate(self, tree, *, grace_seconds, deadline_ms):
            assert grace_seconds >= 0 and deadline_ms > 0
            selected = {member.pid for member in tree.members}
            assert watchdog.pid not in selected, "cleanup watchdog must survive provider fencing"
            events.append(("fence", tuple(sorted(selected))))
            self.members = tuple(member for member in self.members if member.pid not in selected)
            if root.pid in selected:
                process.returncode = 0
            clock[0] += 0.01

    adapter = Adapter()
    state = SimpleNamespace(complete=True, complete_calls=[], observe_calls=[],
                            protected=frozenset({(watchdog.pid, watchdog.start_time_ticks, watchdog.boot_id)}),
                            observation_error=None)

    class Observer:
        def __init__(self, selected, selected_anchor, *, fencing_epoch=0):
            assert selected is profile and selected_anchor is anchor
            assert fencing_epoch == 0

        def observe(self, tree):
            state.observe_calls.append(tuple(item.pid for item in tree.members))
            if state.observation_error is not None:
                raise state.observation_error
            return state.protected

        def complete(self, *, deadline):
            state.complete_calls.append((clock[0], deadline))
            assert process.poll() is not None, "cleanup cannot finish before the owner exits"
            assert provider not in adapter.members, "cleanup cannot finish before the provider exits"
            if state.complete:
                adapter.members = tuple(item for item in adapter.members if item.pid != watchdog.pid)
            return state.complete

    monkeypatch.setattr(runner, "LinuxProcessAdapter", lambda: adapter)
    monkeypatch.setattr(runner, "ManagedCleanupObserver", Observer)
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + max(.01, seconds)))
    return SimpleNamespace(profile=profile, process=process, root=root, provider=provider,
                           watchdog=watchdog, adapter=adapter, observer=state,
                           events=events, clock=clock, anchor=anchor)


def test_stop_preserves_exact_watchdog_until_provider_exit(owner):
    fenced, pids = runner._terminate_managed_process(owner.process, grace_seconds=.2)
    assert fenced is True
    assert set(pids) == {owner.root.pid, owner.provider.pid, owner.watchdog.pid}
    assert owner.observer.complete_calls
    assert not owner.adapter.members
    assert all(owner.watchdog.pid not in event[1] for event in owner.events if event[0] == "fence")


def test_stop_unknown_docker_absence_keeps_cleanup_incomplete_with_one_deadline(owner):
    owner.observer.complete = False
    started = owner.clock[0]
    fenced, _ = runner._terminate_managed_process(owner.process, grace_seconds=.2)
    assert fenced is False
    assert owner.watchdog in owner.adapter.members
    assert owner.observer.complete_calls
    deadlines = {deadline for _, deadline in owner.observer.complete_calls}
    assert len(deadlines) == 1
    assert next(iter(deadlines)) <= started + 1.2 + 1e-9
    assert owner.clock[0] <= started + 1.21


def test_stop_crash_after_process_markers_disappear_still_checks_cleanup(owner):
    owner.process.returncode = 137
    owner.adapter.members = ()
    owner.observer.complete = False
    fenced, _ = runner._terminate_managed_process(owner.process, grace_seconds=.1)
    assert fenced is False
    assert owner.observer.complete_calls
    assert owner.events == []


def test_stop_record_observation_failure_never_signals_cleanup_watchdog(owner):
    owner.observer.observation_error = ValueError("private binding changed")
    try:
        fenced, _ = runner._terminate_managed_process(owner.process, grace_seconds=.1)
    except ValueError:
        # The public owner records a typed failed STOP for observer errors.
        fenced = False
    assert fenced is False
    assert owner.watchdog in owner.adapter.members
    assert not owner.observer.complete_calls


def test_stop_watchdog_pid_reuse_does_not_inherit_saved_exclusion(owner):
    owner.adapter.members = (owner.root, owner.provider, replace(
        owner.watchdog, start_time_ticks=owner.watchdog.start_time_ticks + 1, identity_id=""))
    try:
        fenced, _ = runner._terminate_managed_process(owner.process, grace_seconds=.1)
    except (ValueError, runner.ProcessIdentityMismatch):
        fenced = False
    assert fenced is False
    assert not [event for event in owner.events if event[0] == "fence"]


def test_stop_unrecorded_native_watchdog_cannot_be_signalled_as_ordinary_worker(owner):
    owner.observer.protected = frozenset()
    try:
        fenced, _ = runner._terminate_managed_process(owner.process, grace_seconds=.1)
    except (ValueError, runner.ProcessIdentityMismatch):
        fenced = False
    assert fenced is False
    assert not [event for event in owner.events if event[0] == "fence"]


def test_stop_ordinary_non_docker_processes_keep_existing_fence(owner):
    del owner.process._agent_supervisor_cleanup_directory_anchor
    owner.adapter.members = (owner.root, owner.provider)
    fenced, pids = runner._terminate_managed_process(owner.process, grace_seconds=.1)
    assert fenced is True
    assert set(pids) == {owner.root.pid, owner.provider.pid}
    assert owner.observer.complete_calls == []
    assert owner.process.poll() is not None


def test_stop_tracks_keeps_markers_and_false_completion_when_cleanup_unknown(owner, tmp_path):
    owner.observer.complete = False
    supervisor_pid = tmp_path / "supervisor.pid"
    daemon_pid = tmp_path / "daemon.pid"
    supervisor_pid.write_text(str(owner.root.pid) + "\n")
    daemon_pid.write_text(str(owner.provider.pid) + "\n")
    track = runner.SupervisorTrack(name="cleanup", script_path=tmp_path / "supervisor.py",
                                  log_path=tmp_path / "supervisor.log",
                                  supervisor_pid_path=supervisor_pid, daemon_pid_path=daemon_pid)
    result = runner.stop_tracks((track,), {track.name: owner.process}, repo_root=tmp_path,
                                grace_seconds=.1, output=lambda _: None)
    assert result["all_trees_fenced"] is False
    assert result["removed_runtime_markers"] == []
    assert result["stopped_pids"] == []
    assert result["stop_failure_receipts"][0]["task_completion_authority"] is False
    assert supervisor_pid.read_text() == str(owner.root.pid) + "\n"
    assert daemon_pid.read_text() == str(owner.provider.pid) + "\n"
