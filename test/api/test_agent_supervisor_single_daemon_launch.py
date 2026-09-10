"""One composed daemon birth must own every descriptor and custody marker."""

from types import SimpleNamespace
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as module,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    events = []
    supervisor = object.__new__(module.PortalImplementationSupervisor)
    supervisor.config = SimpleNamespace(
        database_program=None,
        repo_root=tmp_path,
        state_dir=tmp_path,
        state_prefix="test",
        configured_board_live_context=None,
        plan_bound_dispatch=False,
        accepted_control_plane_descriptor=-1,
        state_owner_bootstrap_fd=31,
    )
    supervisor.ensure_managed_daemon_pid_file = lambda: {"blocked": False}
    supervisor._build_daemon_command = lambda **kwargs: ["python", "exact-daemon"]
    supervisor._write_managed_daemon_identity = lambda **kwargs: events.append(
        ("identity", kwargs)
    )
    supervisor._fence_recorded_managed_daemon = lambda **kwargs: (
        events.append(("fence", kwargs)) or {"fenced": True}
    )
    monkeypatch.setattr(
        module, "_requires_eaaef_implementation_daemon_birth", lambda p: False
    )
    monkeypatch.setattr(
        module,
        "_managed_daemon_child_environment",
        lambda **k: {"MANAGED": "yes", "PYTHONPATH": "poison"},
    )
    monkeypatch.setattr(module, "state_authority_pass_fds", lambda env: (32,))
    monkeypatch.setattr(
        runner, "apply_sealed_native_dependency_to_child_environment", lambda env: (33,)
    )
    monkeypatch.setattr(
        module, "write_text_atomic", lambda path, value: events.append(("pid", value))
    )

    def forbidden(*args, **kwargs):
        pytest.fail("direct Popen bypasses the one composed launch")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    process = SimpleNamespace(pid=45678)
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["pre_popen_verify"]()
        kwargs["before_authority_handoff"](process)
        events.append(("handoff", process.pid))
        return process

    monkeypatch.setattr(module, "launch_process_child", launch)
    return supervisor, events, calls, process


def test_ordinary_birth_launches_once_and_publishes_one_identity_then_pid(harness):
    supervisor, events, calls, process = harness
    assert supervisor._start_daemon() is process
    assert len(calls) == 1
    options = calls[0][1]
    assert options["start_new_session"] is True
    assert options["process_group"] is None
    assert options["pass_fds"] == (31, 32, 33)
    assert options["inherit_environment"] is False
    assert [item[0] for item in events] == ["identity", "handoff", "pid"]
    assert events[0][1]["pid"] == process.pid
    assert events[-1][1] == f"{process.pid}\n"


def test_plan_bound_birth_retains_native_inputs_and_inherited_session_group(
    harness, monkeypatch
):
    supervisor, events, calls, process = harness
    supervisor.config.plan_bound_dispatch = True
    supervisor.config.accepted_control_plane_descriptor = 34
    retained_fd = os.open(os.devnull, os.O_RDONLY)
    retained = SimpleNamespace(
        descriptor=retained_fd, executable_path=f"/proc/self/fd/{retained_fd}"
    )
    native = SimpleNamespace(
        descriptor=SimpleNamespace(descriptor=35),
        accepted_authorization_id="native:test",
    )
    monkeypatch.setattr(
        runner, "admit_sealed_native_dependency_environment", lambda env: (native, "[]")
    )
    monkeypatch.setattr(
        runner, "retain_control_plane_interpreter", lambda executable: retained
    )
    monkeypatch.setattr(
        runner,
        "sealed_native_dependency_environment",
        lambda *a, **k: {"NATIVE": "yes"},
    )
    build_calls = []
    supervisor._build_daemon_command = lambda **kwargs: (
        build_calls.append(kwargs) or ["sealed-python", "exact-daemon"]
    )
    assert supervisor._start_daemon() is process
    assert len(calls) == len(build_calls) == 1
    assert build_calls[0]["retained_interpreter"] is retained
    assert build_calls[0]["native_dependency"] is native
    options = calls[0][1]
    assert options["start_new_session"] is False
    assert options["process_group"] == 0
    assert options["executable"] == retained.executable_path
    assert set(options["pass_fds"]) == {retained_fd, 31, 32, 34, 35}
    assert options["env"] == {
        "MANAGED": "yes",
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "NATIVE": "yes",
    }
    with pytest.raises(OSError):
        os.fstat(retained_fd)
    assert [item[0] for item in events] == ["identity", "handoff", "pid"]


def test_live_capsule_preserves_projected_environment_and_bootstrap(harness):
    supervisor, _, calls, _ = harness
    context = SimpleNamespace(pass_fds=(40, 41))
    supervisor.config.configured_board_live_context = context
    supervisor._lgcvf_live_managed_daemon_environment = lambda command: (
        {"PROJECTED": "yes"},
        context,
    )
    supervisor._start_daemon()
    assert len(calls) == 1
    assert calls[0][1]["env"] == {"PROJECTED": "yes"}
    assert calls[0][1]["pass_fds"] == (31, 32, 40, 41)
    assert calls[0][1]["start_new_session"] is True


def test_live_context_change_refuses_birth_before_identity_or_pid(harness):
    supervisor, events, _, _ = harness
    context = SimpleNamespace(pass_fds=(40, 41))
    supervisor.config.configured_board_live_context = context
    results = iter(
        [({"PROJECTED": "original"}, context), ({"PROJECTED": "changed"}, context)]
    )
    supervisor._lgcvf_live_managed_daemon_environment = lambda command: next(results)
    with pytest.raises(module.SupervisorSchedulerConfigError, match="bindings changed"):
        supervisor._start_daemon()
    assert not events


def test_identity_failure_cannot_publish_pid_or_start_second_child(harness):
    supervisor, events, calls, _ = harness

    def fail_identity(**kwargs):
        raise OSError("identity unavailable")

    supervisor._write_managed_daemon_identity = fail_identity
    with pytest.raises(OSError, match="identity unavailable"):
        supervisor._start_daemon()
    assert len(calls) == 1
    assert not events


def test_pid_publication_failure_fences_the_same_recorded_child(harness, monkeypatch):
    supervisor, events, calls, process = harness

    def fail_pid(*args):
        raise OSError("pid write unavailable")

    monkeypatch.setattr(module, "write_text_atomic", fail_pid)
    with pytest.raises(OSError, match="pid write unavailable"):
        supervisor._start_daemon()
    assert len(calls) == 1
    assert [item[0] for item in events] == ["identity", "handoff", "fence"]
    assert events[-1][1]["pid"] == process.pid


def test_ownership_blocker_prevents_command_and_birth(harness):
    supervisor, events, calls, _ = harness
    supervisor.ensure_managed_daemon_pid_file = lambda: {
        "blocked": True,
        "reason": "existing exact child",
    }
    with pytest.raises(RuntimeError, match="existing exact child"):
        supervisor._start_daemon()
    assert not events and not calls
