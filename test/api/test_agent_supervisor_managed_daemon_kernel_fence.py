from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    LifecycleProfile,
    ProcessIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    process_security,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SupervisedChildIdentity,
)
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)

_SCOPE_BOOT_ID = "01234567-89ab-cdef-0123-456789abcdef"


def _scope_stat(pid: int, start_time_ticks: int) -> str:
    fields = ["S", *("0" for _ in range(18)), str(start_time_ticks), "0"]
    return f"{pid} (scope init) " + " ".join(fields) + "\n"


def _fake_linux_process_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], Path, Path, int, int]:
    proc_root = tmp_path / "proc"
    boot_id_path = tmp_path / "boot_id"
    mountinfo_path = tmp_path / "mountinfo"
    cgroup_mount = tmp_path / "cgroup2"
    cgroup_directory = cgroup_mount / "docker" / "scope"
    pid = 4242
    start_ticks = 88001

    process_directory = proc_root / str(pid)
    namespace_directory = process_directory / "ns"
    namespace_directory.mkdir(parents=True)
    (process_directory / "stat").write_text(
        _scope_stat(pid, start_ticks),
        encoding="ascii",
    )
    (process_directory / "cgroup").write_text(
        "0::/docker/scope\n",
        encoding="ascii",
    )
    (namespace_directory / "pid").write_text("pid namespace\n", encoding="ascii")
    cgroup_directory.mkdir(parents=True)
    (cgroup_directory / "cgroup.events").write_text(
        "populated 1\nfrozen 0\n",
        encoding="ascii",
    )
    boot_id_path.write_text(_SCOPE_BOOT_ID + "\n", encoding="ascii")
    mountinfo_path.write_text(
        f"38 28 0:32 / {cgroup_mount} rw,nosuid,nodev,noexec - "
        "cgroup2 cgroup2 rw\n",
        encoding="ascii",
    )
    monkeypatch.setattr(process_security, "_PROC_ROOT", proc_root)
    monkeypatch.setattr(process_security, "_BOOT_ID_PATH", boot_id_path)
    monkeypatch.setattr(process_security, "_MOUNTINFO_PATH", mountinfo_path)

    record = process_security.capture_linux_process_scope(pid)
    assert process_security.validate_linux_process_scope(record) == record
    assert set(record) == {
        "schema",
        "pid",
        "start_time_ticks",
        "boot_id",
        "pid_namespace_device",
        "pid_namespace_inode",
        "cgroup_relative_path",
        "cgroup_mount_device",
        "cgroup_mount_inode",
        "cgroup_directory_device",
        "cgroup_directory_inode",
        "cgroup_events_device",
        "cgroup_events_inode",
        "identity_id",
    }
    return record, process_directory, cgroup_directory, pid, start_ticks


def _root_identity(tmp_path: Path) -> ProcessIdentity:
    return ProcessIdentity(
        pid=7001,
        start_time_ticks=101,
        parent_pid=6001,
        process_group_id=7001,
        session_id=7001,
        boot_id="boot-test",
        argv=("python", "lane.py", "--plan-bound-dispatch"),
        cwd=str(tmp_path),
        executable="/usr/bin/python3",
        run_id="run-test",
        profile_id="profile-test",
        target_id="target-test",
        repository_root=str(tmp_path),
        state_root=str(tmp_path / "state"),
        run_root=str(tmp_path / "state" / "run"),
        fencing_epoch=1,
        configuration_root="sha256:configuration",
    )


def _binding(tmp_path: Path, root: ProcessIdentity) -> runner._ManagedDaemonKernelFence:
    state_dir = tmp_path / "state"
    return runner._ManagedDaemonKernelFence(
        pid_path=state_dir / "lane_managed_daemon.pid",
        identity_path=state_dir / "lane_managed_daemon.identity.json",
        owner_scope={
            "repo_root": str(tmp_path),
            "state_dir": str(state_dir),
            "state_prefix": "lane",
            "todo_path": str(tmp_path / "todo.md"),
            "daemon_entrypoint": (
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_daemon"
            ),
            "lifecycle_session_id": str(root.session_id),
            "process_group_policy": "dedicated_group_inherited_session",
        },
        root_session_id=root.session_id,
        state_dir_option=str(state_dir),
        state_prefix="lane",
        todo_path_option=str(tmp_path / "todo.md"),
        daemon_entrypoint=(
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_daemon"
        ),
    )


def _daemon_identity(
    binding: runner._ManagedDaemonKernelFence,
    *,
    owner_scope: dict[str, str] | None = None,
) -> SupervisedChildIdentity:
    command = (
        "/usr/bin/python3",
        "-m",
        binding.daemon_entrypoint,
        "--state-dir",
        binding.state_dir_option,
        "--state-prefix",
        binding.state_prefix,
        "--todo-path",
        binding.todo_path_option,
    )
    return SupervisedChildIdentity(
        process_birth=ProcessBirthIdentity(
            pid=7101,
            start_time_ticks=202,
            boot_id="boot-test",
            parent_pid=1,
        ),
        command=command,
        owner_scope=dict(owner_scope or binding.owner_scope),
        created_at="2026-08-24T00:00:00+00:00",
    )


def _write_markers(
    binding: runner._ManagedDaemonKernelFence,
    identity: SupervisedChildIdentity,
) -> None:
    binding.pid_path.parent.mkdir(parents=True, exist_ok=True)
    binding.pid_path.write_text(
        f"{identity.process_birth.pid}\n",
        encoding="ascii",
    )
    binding.identity_path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_reparented_nondumpable_daemon_session_prevents_false_death(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root_identity(tmp_path)
    profile = LifecycleProfile(
        target_id=root.target_id,
        run_id=root.run_id,
        configuration_root=root.configuration_root,
        repository_root=root.repository_root,
        state_root=root.state_root,
        run_root=root.run_root,
        argv=root.argv,
        cwd=root.cwd,
    )
    monkeypatch.setattr(
        runner.LinuxProcessAdapter,
        "_stat",
        staticmethod(
            lambda _pid: (_ for _ in ()).throw(FileNotFoundError())
        ),
    )
    monkeypatch.setattr(
        runner,
        "_kernel_session_members_once",
        lambda _session, **_kwargs: (
            "alive",
            ((7101, 202, 7101),),
        ),
    )

    state, snapshot = runner._strict_plan_bound_process_fence_observation(
        profile,
        root,
    )

    assert state == "alive"
    assert snapshot is None


def test_forged_managed_daemon_sidecar_is_unknown_and_never_signalled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root_identity(tmp_path)
    binding = _binding(tmp_path, root)
    forged_scope = {**dict(binding.owner_scope), "state_prefix": "forged"}
    identity = _daemon_identity(binding, owner_scope=forged_scope)
    _write_markers(binding, identity)
    monkeypatch.setattr(
        runner,
        "_kernel_session_members_once",
        lambda _session, **_kwargs: (
            "alive",
            ((identity.process_birth.pid, 202, identity.process_birth.pid),),
        ),
    )
    monkeypatch.setattr(
        runner,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail(
            "forged daemon identity became signal authority"
        ),
    )

    state, observed = runner._managed_daemon_kernel_fence_observation(
        binding,
        root_identity=root,
    )

    assert state == "unknown"
    assert observed is None
    assert not runner._fence_managed_daemon_from_kernel_binding(
        binding,
        root_identity=root,
        grace_seconds=0,
    )


def test_exact_kernel_bound_daemon_is_fenced_by_birth_and_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root_identity(tmp_path)
    binding = _binding(tmp_path, root)
    identity = _daemon_identity(binding)
    _write_markers(binding, identity)
    stopped = False

    def session_members(_session: int, **_kwargs: object):
        return (
            ("dead", ())
            if stopped
            else (
                "alive",
                (
                    (
                        identity.process_birth.pid,
                        identity.process_birth.start_time_ticks,
                        identity.process_birth.pid,
                    ),
                ),
            )
        )

    monkeypatch.setattr(runner, "_kernel_session_members_once", session_members)
    monkeypatch.setattr(
        runner,
        "supervised_child_identity_liveness",
        lambda _identity: (
            OwnerLiveness.DEAD if stopped else OwnerLiveness.ALIVE
        ),
    )
    monkeypatch.setattr(
        runner.LinuxProcessAdapter,
        "_stat",
        staticmethod(
            lambda _pid: (
                1,
                identity.process_birth.pid,
                root.session_id,
                identity.process_birth.start_time_ticks,
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "read_process_command_argv",
        lambda _pid: identity.command,
    )
    calls: list[dict[str, object]] = []

    def terminate(pid: int, **kwargs: object) -> bool:
        nonlocal stopped
        calls.append({"pid": pid, **kwargs})
        stopped = True
        return True

    monkeypatch.setattr(runner, "terminate_pid_tree", terminate)

    assert runner._fence_managed_daemon_from_kernel_binding(
        binding,
        root_identity=root,
        grace_seconds=0.25,
    )
    assert calls == [
        {
            "pid": identity.process_birth.pid,
            "grace_seconds": 0.25,
            "freeze_first": True,
            "require_gone": True,
            "owned_process_group_id": identity.process_birth.pid,
            "expected_root_start_time_ticks": (
                identity.process_birth.start_time_ticks
            ),
        }
    ]


def test_linux_process_scope_rejects_live_exact_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, _process, _cgroup, pid, _start = _fake_linux_process_scope(
        tmp_path,
        monkeypatch,
    )
    observed: list[int] = []

    def alive(observed_pid: int) -> bool:
        observed.append(observed_pid)
        return False

    monkeypatch.setattr(process_security, "_pidfd_reports_exit", alive)

    assert not process_security.linux_process_scope_quiescent(record)
    assert observed == [pid]


def test_linux_process_scope_rejects_dead_init_with_namespace_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, process_directory, cgroup_directory, _pid, _start = (
        _fake_linux_process_scope(tmp_path, monkeypatch)
    )
    descendant = process_directory.parent / "5252"
    (descendant / "ns").mkdir(parents=True)
    os.link(
        process_directory / "ns" / "pid",
        descendant / "ns" / "pid",
    )
    (descendant / "cgroup").write_text(
        "0::/docker/scope\n",
        encoding="ascii",
    )
    shutil.rmtree(process_directory)
    shutil.rmtree(cgroup_directory)

    assert not process_security.linux_process_scope_quiescent(record)


def test_linux_process_scope_rejects_namespace_descendant_migrated_outside_cgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, process_directory, cgroup_directory, _pid, _start = (
        _fake_linux_process_scope(tmp_path, monkeypatch)
    )
    descendant = process_directory.parent / "5353"
    (descendant / "ns").mkdir(parents=True)
    os.link(
        process_directory / "ns" / "pid",
        descendant / "ns" / "pid",
    )
    (descendant / "cgroup").write_text(
        "0::/migrated/outside\n",
        encoding="ascii",
    )
    shutil.rmtree(process_directory)
    (cgroup_directory / "cgroup.events").write_text(
        "populated 0\nfrozen 0\n",
        encoding="ascii",
    )

    assert not process_security.linux_process_scope_quiescent(record)


def test_linux_process_scope_rejects_unreadable_extant_proc_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, process_directory, cgroup_directory, _pid, _start = (
        _fake_linux_process_scope(tmp_path, monkeypatch)
    )
    unreadable = process_directory.parent / "5454"
    unreadable.mkdir()
    shutil.rmtree(process_directory)
    (cgroup_directory / "cgroup.events").write_text(
        "populated 0\nfrozen 0\n",
        encoding="ascii",
    )

    assert not process_security.linux_process_scope_quiescent(record)


def test_linux_process_scope_rejects_populated_cgroup_after_init_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, process_directory, _cgroup, _pid, _start = (
        _fake_linux_process_scope(tmp_path, monkeypatch)
    )
    shutil.rmtree(process_directory)

    assert not process_security.linux_process_scope_quiescent(record)


def test_linux_process_scope_accepts_empty_exact_cgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, process_directory, cgroup_directory, _pid, _start = (
        _fake_linux_process_scope(tmp_path, monkeypatch)
    )
    shutil.rmtree(process_directory)
    (cgroup_directory / "cgroup.events").write_text(
        "populated 0\nfrozen 0\n",
        encoding="ascii",
    )

    assert process_security.linux_process_scope_quiescent(record)


def test_linux_process_scope_rejects_pid_birth_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, process_directory, _cgroup, pid, start_ticks = (
        _fake_linux_process_scope(tmp_path, monkeypatch)
    )
    (process_directory / "stat").write_text(
        _scope_stat(pid, start_ticks + 1),
        encoding="ascii",
    )
    monkeypatch.setattr(
        process_security,
        "_pidfd_reports_exit",
        lambda _pid: pytest.fail("a substituted birth reached pidfd admission"),
    )

    assert not process_security.linux_process_scope_quiescent(record)


def test_linux_process_scope_rejects_cgroup_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, process_directory, cgroup_directory, _pid, _start = (
        _fake_linux_process_scope(tmp_path, monkeypatch)
    )
    shutil.rmtree(process_directory)
    displaced = cgroup_directory.with_name("scope-displaced")
    cgroup_directory.rename(displaced)
    cgroup_directory.mkdir()
    (cgroup_directory / "cgroup.events").write_text(
        "populated 0\nfrozen 0\n",
        encoding="ascii",
    )

    assert not process_security.linux_process_scope_quiescent(record)


def test_linux_process_scope_accepts_absent_cgroup_only_after_namespace_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, process_directory, cgroup_directory, _pid, _start = (
        _fake_linux_process_scope(tmp_path, monkeypatch)
    )
    shutil.rmtree(process_directory)
    shutil.rmtree(cgroup_directory)

    assert process_security.linux_process_scope_quiescent(record)
