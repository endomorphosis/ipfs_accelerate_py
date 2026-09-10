"""Regressions for native custody composed with scoped generic launch gates."""

import hashlib
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime import process_security


def test_gate_routes_before_any_credential_capture(monkeypatch):
    def forbidden():
        pytest.fail("the unreleased gate must not receive native credentials")

    monkeypatch.setattr(
        process_security, "capture_state_authority_credentials", forbidden
    )
    monkeypatch.setattr(process_security, "harden_state_authority_process", forbidden)
    monkeypatch.setattr(runner, "_run_plan_bound_launch_gate", lambda argv: 23)
    assert runner.main([runner.PLAN_BOUND_LAUNCH_GATE_MARKER, "test"]) == 23


def test_legacy_gate_layout_is_refused_before_interpreter_admission(monkeypatch):
    monkeypatch.setattr(
        runner,
        "admit_retained_control_plane_interpreter",
        lambda **kwargs: pytest.fail("unretained layout"),
    )
    assert (
        runner._run_plan_bound_launch_gate(
            ["3", "/root", "pin", "4", "-", "--", "python", "script"]
        )
        == 78
    )


@pytest.mark.parametrize("wrong_executable", [False, True])
def test_gate_holds_exact_interpreter_and_closes_it_on_return(
    monkeypatch, wrong_executable
):
    descriptor = os.open(
        os.devnull if wrong_executable else "/proc/self/exe", os.O_RDONLY
    )
    retained = SimpleNamespace(
        descriptor=descriptor, argv0="python", sha256="sha256:" + "a" * 64
    )
    monkeypatch.setattr(
        runner, "admit_retained_control_plane_interpreter", lambda **kwargs: retained
    )
    calls = []
    monkeypatch.setattr(
        runner,
        "_run_plan_bound_launch_gate_with_interpreter",
        lambda argv, **kwargs: calls.append(kwargs) or 23,
    )
    argv = [
        "3",
        "/root",
        "pin",
        "4",
        "-",
        str(descriptor),
        "python",
        retained.sha256,
        "--",
        "python",
        "script",
    ]
    assert runner._run_plan_bound_launch_gate(argv) == (78 if wrong_executable else 23)
    assert len(calls) == (0 if wrong_executable else 1)
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize("release", [b"Y", b"N"])
def test_native_gate_preserves_retained_exec_and_handoff_without_early_release(
    tmp_path, monkeypatch, release
):
    pin = SimpleNamespace(source_head="a" * 40, source_tree="b" * 40)
    retained = SimpleNamespace(
        descriptor=100,
        argv0="python",
        sha256="sha256:" + "c" * 64,
        executable_path="/proc/self/fd/100",
    )
    native = SimpleNamespace(
        pin=SimpleNamespace(python_executable_sha256=retained.sha256),
        accepted_authorization_id="native:test",
    )
    monkeypatch.setattr(runner, "parse_accepted_control_plane_pin", lambda value: pin)
    monkeypatch.setattr(
        runner, "verify_agent_implementation_sealed_control_plane", lambda *args: None
    )
    monkeypatch.setattr(runner, "_canonical_accepted_tree_root", lambda root: tmp_path)
    monkeypatch.setattr(
        runner, "admit_sealed_native_dependency_environment", lambda env: (native, "[]")
    )
    prefix_calls = []
    monkeypatch.setattr(
        runner,
        "build_sealed_control_plane_module_command",
        lambda **kwargs: prefix_calls.append(kwargs) or ["python", "sealed"],
    )
    validations = []
    monkeypatch.setattr(
        runner,
        "_validate_plan_bound_accepted_tree",
        lambda **kwargs: validations.append(kwargs),
    )
    monkeypatch.setattr(
        runner, "_plan_bound_positive_child_environment", lambda env: {}
    )
    monkeypatch.setattr(
        runner,
        "sealed_native_dependency_environment",
        lambda *args, **kwargs: {"SEALED": "native"},
    )
    handoff_name = next(iter(process_security.STATE_AUTHORITY_HANDOFF_ENV_NAMES))
    monkeypatch.setenv(handoff_name, "non-secret-owned-handoff")
    executions = []
    monkeypatch.setattr(runner.os, "execve", lambda *args: executions.append(args))
    read_fd, write_fd = os.pipe()
    os.write(
        write_fd, runner.PLAN_BOUND_LAUNCH_GATE_SUCCESS if release == b"Y" else release
    )
    os.close(write_fd)
    child = [
        "python",
        "sealed",
        "--plan-bound-dispatch",
        "--plan-bound-source-head",
        pin.source_head,
        "--plan-bound-source-tree",
        pin.source_tree,
        "--plan-bound-accepted-tree-root",
        str(tmp_path),
        "--plan-revision-store-path",
        "store",
        "--plan-bound-revision-cid",
        "revision",
        "--plan-bound-slice-id",
        "slice",
        "--plan-bound-lane-id",
        "lane",
        "--state-prefix",
        "lane",
    ]
    argv = [
        str(read_fd),
        str(tmp_path),
        "pin",
        "101",
        "-",
        "100",
        "python",
        retained.sha256,
        "--",
        *child,
    ]
    assert (
        runner._run_plan_bound_launch_gate_with_interpreter(
            argv, retained_interpreter=retained
        )
        == 78
    )
    assert prefix_calls[0]["retained_interpreter"] is retained
    assert prefix_calls[0]["native_dependency_launch"] is native
    assert len(executions) == len(validations) == (1 if release == b"Y" else 0)
    if executions:
        assert executions[0] == (
            retained.executable_path,
            child,
            {handoff_name: "non-secret-owned-handoff", "SEALED": "native"},
        )
    with pytest.raises(OSError):
        os.fstat(read_fd)


@pytest.fixture
def scoped_board(tmp_path):
    def git(*args):
        return (
            subprocess.check_output(
                [
                    "/usr/bin/git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    *args,
                ],
                cwd=tmp_path,
            )
            .decode()
            .strip()
        )

    git("init", "--quiet")

    def make(namespace):
        payload = {
            "schema": f"ipfs_accelerate_py.agent_supervisor.{namespace}.scheduler_config@1",
            "program_identifier": namespace,
            "board_namespace": namespace,
        }
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
        (tmp_path / "board.json").write_bytes(raw)
        git("add", "board.json")
        git("commit", "--quiet", "-m", "bind board")
        pin = SimpleNamespace(
            source_head=git("rev-parse", "HEAD"),
            source_tree=git("rev-parse", "HEAD^{tree}"),
        )
        argv = [
            "--scheduler-config",
            "board.json",
            "--plan-bound-configuration-root",
            scheduler._identity({"bytes_sha256": hashlib.sha256(raw).hexdigest()}),
            "--board-namespace",
            namespace,
            "--plan-bound-accepted-tree-root",
            str(tmp_path),
            "--plan-bound-source-head",
            pin.source_head,
            "--plan-bound-source-tree",
            pin.source_tree,
        ]
        return pin, argv

    return make, git


def test_qualified_non_eaaef_scope_uses_accepted_config_after_source_progress(
    scoped_board,
):
    make, git = scoped_board
    pin, argv = make("agent-supervisor-efficiency-and-state-hardening-v1")
    assert runner.sealed_implementation_native_scope_allowed(pin=pin, argv=argv)
    git("commit", "--quiet", "--allow-empty", "-m", "accepted task source progress")
    assert runner.sealed_implementation_native_scope_allowed(pin=pin, argv=argv)


def test_eaaef_scope_cannot_be_bypassed_by_namespace_flag(scoped_board, monkeypatch):
    make, _ = scoped_board
    pin, argv = make("external-agent-autonomous-execution-fabric-v1")
    receipts = []
    monkeypatch.setattr(
        runner,
        "_eaaef_host_receipt_admitted",
        lambda *args, **kwargs: receipts.append((args, kwargs)) or False,
    )
    assert not runner.sealed_implementation_native_scope_allowed(pin=pin, argv=argv)
    assert len(receipts) == 1
    argv[argv.index("--board-namespace") + 1] = (
        "agent-supervisor-efficiency-and-state-hardening-v1"
    )
    assert not runner.sealed_implementation_native_scope_allowed(pin=pin, argv=argv)
    assert len(receipts) == 1


@pytest.mark.parametrize(
    "change",
    ["absent", "wrong_cid", "duplicate_namespace", "unknown_path", "wrong_source"],
)
def test_unknown_or_mismatched_board_scope_is_denied(scoped_board, change):
    make, _ = scoped_board
    pin, argv = make("agent-supervisor-efficiency-and-state-hardening-v1")
    if change == "absent":
        argv = []
    elif change == "wrong_cid":
        argv[argv.index("--plan-bound-configuration-root") + 1] = "sha256:" + "0" * 64
    elif change == "duplicate_namespace":
        argv.extend(["--board-namespace", argv[argv.index("--board-namespace") + 1]])
    elif change == "unknown_path":
        argv[argv.index("--scheduler-config") + 1] = "missing.json"
    elif change == "wrong_source":
        argv[argv.index("--plan-bound-source-tree") + 1] = "0" * 40
    assert not runner.sealed_implementation_native_scope_allowed(pin=pin, argv=argv)


def test_plan_track_composes_one_retained_birth_gate_handoff_and_kernel_custody(
    tmp_path, monkeypatch
):
    import sys
    from ipfs_accelerate_py.agent_supervisor.control import plan_execution_store
    from ipfs_accelerate_py.agent_supervisor.task_sources import plan_revision_store

    entry = tmp_path / runner.PLAN_BOUND_ACCEPTED_ENTRY_PATH
    entry.parent.mkdir(parents=True)
    entry.write_text("# test entry\n")
    state = tmp_path / "state/lane"
    state.mkdir(parents=True)
    pin = SimpleNamespace(source_head="a" * 40, source_tree="b" * 40)
    extra = (
        "--plan-bound-dispatch",
        "--plan-bound-accepted-tree-root",
        str(tmp_path),
        "--plan-bound-configuration-root",
        "config",
        "--plan-revision-store-path",
        str(state.parent / "store"),
        "--plan-bound-source-head",
        pin.source_head,
        "--plan-bound-source-tree",
        pin.source_tree,
        "--plan-bound-revision-cid",
        "revision",
        "--plan-bound-slice-id",
        "slice",
        "--plan-bound-lane-id",
        "lane",
        "--state-dir",
        str(state),
        "--state-prefix",
        "lane",
    )
    track = runner.SupervisorTrack(
        name="lane",
        script_path=entry,
        log_path=state / "lane.log",
        supervisor_pid_path=state / "supervisor.pid",
        daemon_pid_path=state / "daemon.pid",
        extra_args=extra,
    )
    retained_fd = os.open(os.devnull, os.O_RDONLY)
    retained = SimpleNamespace(
        descriptor=retained_fd,
        argv0=sys.executable,
        sha256="sha256:" + "c" * 64,
        executable_path=f"/proc/self/fd/{retained_fd}",
    )
    native = SimpleNamespace(
        descriptor=SimpleNamespace(descriptor=202),
        accepted_authorization_id="native:test",
    )
    monkeypatch.setattr(
        runner, "verify_agent_implementation_sealed_control_plane", lambda *args: None
    )
    monkeypatch.setattr(
        runner, "_validate_plan_bound_accepted_tree", lambda **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_plan_bound_repository_identity",
        lambda root: (pin.source_head, pin.source_tree),
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
        lambda *args, **kwargs: {"SEALED": "native"},
    )
    monkeypatch.setattr(
        runner, "accepted_control_plane_pin_json", lambda pin: "pin-json"
    )
    monkeypatch.setattr(plan_revision_store, "PlanRevisionStore", lambda path: object())
    monkeypatch.setattr(
        plan_execution_store,
        "ProductionParallelPlanAdapter",
        lambda store: SimpleNamespace(load_execution_lease=lambda **kwargs: None),
    )
    commands = []

    def command(**kwargs):
        commands.append(kwargs)
        return [sys.executable, "sealed", *kwargs["argv"]]

    monkeypatch.setattr(runner, "build_sealed_control_plane_module_command", command)
    events = []
    identity_type = type("Identity", (), {})
    identity = identity_type()
    monkeypatch.setattr(runner, "ProcessIdentity", identity_type)
    monkeypatch.setattr(
        runner,
        "LinuxProcessAdapter",
        lambda: SimpleNamespace(
            _identity=lambda pid, profile: events.append("birth") or identity
        ),
    )
    monkeypatch.setattr(
        runner,
        "_persist_plan_bound_process_birth",
        lambda **kwargs: events.append("persist") or "birth:cid",
    )
    fence = object()
    monkeypatch.setattr(
        runner,
        "_managed_daemon_kernel_fence_for_track",
        lambda *args, **kwargs: events.append("kernel_fence") or fence,
    )
    handoff_calls = []
    handoff = SimpleNamespace(
        pass_fds=(201,),
        deliver=lambda process, **kwargs: (
            handoff_calls.append(kwargs) or events.append("handoff")
        ),
        close=lambda: events.append("close_handoff"),
    )
    monkeypatch.setattr(
        process_security,
        "prepare_state_authority_child_handoff",
        lambda *args, **kwargs: handoff,
    )
    captured = []
    process = SimpleNamespace(pid=456789, poll=lambda: None)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda command, **kwargs: captured.append((command, kwargs)) or process,
    )
    real_write = os.write

    def write(fd, data):
        if data == runner.PLAN_BOUND_LAUNCH_GATE_SUCCESS:
            events.append("release")
            return len(data)  # The fake child has no inherited pipe reader.
        return real_write(fd, data)

    monkeypatch.setattr(runner.os, "write", write)
    result = runner.start_track(
        track,
        repo_root=tmp_path,
        common_args=(),
        python_executable=sys.executable,
        accepted_control_plane_pin=pin,
        accepted_control_plane_descriptor=203,
        output=lambda value: None,
    )
    try:
        assert result is process
        assert len(captured) == 1 and len(commands) == 2
        gate = list(commands[1]["argv"])
        assert gate[:1] == [runner.PLAN_BOUND_LAUNCH_GATE_MARKER]
        assert gate[6:11] == [
            str(retained_fd),
            retained.argv0,
            retained.sha256,
            "-",
            "--",
        ]
        for call in commands:
            assert (
                call["retained_interpreter"] is retained
                and call["native_dependency_launch"] is native
            )
        assert captured[0][1]["executable"] == retained.executable_path
        assert {201, 202, 203, retained_fd} <= set(captured[0][1]["pass_fds"])
        assert events[:5] == ["birth", "persist", "release", "handoff", "kernel_fence"]
        assert handoff_calls[0]["expected_argv"] == [
            sys.executable,
            "sealed",
            *commands[0]["argv"],
        ]
        assert handoff_calls[0]["expected_executable_descriptor"] == retained_fd
        assert process._agent_supervisor_managed_daemon_kernel_fence is fence
        assert (state / "supervisor.pid").read_text() == f"{process.pid}\n"
        with pytest.raises(OSError):
            os.fstat(retained_fd)
    finally:
        os.close(process._agent_supervisor_cleanup_directory_anchor.descriptor)


def test_failed_handoff_preserves_unfenced_tree_and_cleanup_anchor(
    tmp_path, monkeypatch
):
    script = tmp_path / "ordinary.py"
    script.write_text("# isolated fake launch\n")
    track = runner.SupervisorTrack(
        name="ordinary",
        script_path=script,
        log_path=tmp_path / "ordinary.log",
        supervisor_pid_path=tmp_path / "ordinary.pid",
        daemon_pid_path=tmp_path / "daemon.pid",
    )
    process = SimpleNamespace(pid=456790, poll=lambda: None)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        runner, "optional_active_sealed_native_dependency", lambda env: None
    )
    monkeypatch.setattr(
        runner, "_fence_failed_owned_process_birth", lambda child: False
    )

    def refuse(*args, **kwargs):
        raise RuntimeError("credential exchange refused")

    handoff = SimpleNamespace(pass_fds=(), deliver=refuse, close=lambda: None)
    monkeypatch.setattr(
        process_security,
        "prepare_state_authority_child_handoff",
        lambda *args, **kwargs: handoff,
    )
    with pytest.raises(runner.UnadmittedSupervisorProcessBirthError) as rejected:
        runner.start_track(
            track, repo_root=tmp_path, common_args=(), output=lambda value: None
        )
    try:
        assert rejected.value.process is process
        assert rejected.value.marker_published is True
        assert (tmp_path / "ordinary.pid").read_text() == f"{process.pid}\n"
        assert process._agent_supervisor_birth_admission_failed is True
        assert (
            process._agent_supervisor_lifecycle_profile.target_id
            == "supervisor-track:ordinary"
        )
        os.fstat(process._agent_supervisor_cleanup_directory_anchor.descriptor)
    finally:
        os.close(process._agent_supervisor_cleanup_directory_anchor.descriptor)
