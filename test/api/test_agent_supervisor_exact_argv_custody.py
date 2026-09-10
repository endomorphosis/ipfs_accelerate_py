"""Exact process arguments preserve custody without granting task authority."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor as workers
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as implementation,
)
from test.api.test_agent_supervisor_database_portal_reload_gate import (
    _authenticated_watchdog_projection,
    _changed_control_plane_projection,
    _config,
)


GROK = ("/usr/bin/python3", "-m", "ipfs_accelerate_py.agent_supervisor.grok_cli_runner")


@pytest.mark.parametrize(
    "argv",
    [
        GROK,
        ("/usr/bin/grok",),
        ("node", "/opt/grok"),
        ("python3", "-u", "/opt/grok_cli_runner.py"),
        (*GROK, "--prompt", ""),
        ("/usr/bin/grok", "--prompt", ""),
        ("python3", "-uB", "-m", GROK[2]),
    ],
)
def test_exact_argv_recognizes_worker_when_ps_text_is_missing(monkeypatch, argv):
    item = {"pid": 41002, "cmdline": "", "argv": argv, "start_ticks": 900}
    monkeypatch.setattr(workers, "descendant_processes", lambda pid: [item])
    assert workers.active_codex_exec_workers(41001) == [item]


@pytest.mark.parametrize(
    "argv",
    [
        ("python3", "-c", "-m", GROK[2]),
        ("echo", "grok_cli_runner"),
        ("python3", "-m", "pytest", GROK[2]),
        ("python-not-an-interpreter", "-m", GROK[2]),
        "grok",
        (),
        ("grok", None),
        ("grok", "bad\0argument"),
    ],
)
def test_exact_argv_cannot_be_overridden_by_worker_text(monkeypatch, argv):
    item = {"pid": 41002, "cmdline": "/usr/bin/grok", "argv": argv}
    monkeypatch.setattr(workers, "descendant_processes", lambda pid: [item])
    assert workers.active_codex_exec_workers(41001) == []


@pytest.mark.parametrize("ps_text", ["", " ".join(GROK)])
def test_exact_argv_does_not_replace_active_attempt_receipt(monkeypatch, ps_text):
    item = {"pid": 41002, "cmdline": ps_text, "argv": GROK, "start_ticks": 900}
    monkeypatch.setattr(workers, "descendant_processes", lambda pid: [item])
    status = {
        "active_task_id": "TASK-1",
        "active_attempt": 1,
        "active_task_cid": "cid:revision",
        "active_worktree_path": "/tmp/workspace",
        "active_phase": "implementing",
        "implementation_in_progress": True,
    }
    assert workers.active_codex_exec_workers(41001, status) == []


def test_procfs_argument_reader_preserves_empty_trailing_argument(monkeypatch):
    argv = (*GROK, "--prompt", "")
    raw = b"\0".join(part.encode() for part in argv) + b"\0"

    def read(path):
        assert path == Path("/proc/41002/cmdline")
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    assert workers._process_command_argv(41002) == argv


def _patch_descendant(monkeypatch):
    item = {"pid": 41002, "cmdline": "", "argv": GROK, "start_ticks": 900}
    observe = lambda root: [item] if root == 41001 else []
    monkeypatch.setattr(workers, "descendant_processes", observe)
    monkeypatch.setattr(implementation, "descendant_processes", observe)
    return item


def test_idle_projection_reload_preserves_exact_argv_worker(tmp_path, monkeypatch):
    supervisor = implementation.PortalImplementationSupervisor(_config(tmp_path))
    _patch_descendant(monkeypatch)
    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        _changed_control_plane_projection,
    )
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)
    monkeypatch.setattr(
        supervisor, "_active_agent_worker_processes", lambda *_a, **_k: []
    )
    monkeypatch.setattr(
        supervisor, "_active_validation_subprocess_exists", lambda: False
    )

    @contextmanager
    def portal_fence():
        yield supervisor.config.database_program

    monkeypatch.setattr(
        supervisor, "_database_portal_reload_mutation_fence", portal_fence
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _: _authenticated_watchdog_projection(active=False),
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_a, **_k: pytest.fail("live worker must not be quiesced"),
    )
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    result = supervisor._supervisor_loop_watchdog_decision(
        loop, SimpleNamespace(pid=41001), {}
    )
    assert result.action == "continue"
    assert loop.config.status_extra_fields["control_plane_update_pending"] is True
    assert loop.config.status_extra_fields["control_plane_reload_deferred"] is True
    assert (
        loop.config.status_extra_fields["control_plane_reload_quiescence"]["attempted"]
        is False
    )


def test_termination_preserves_exact_argv_worker_without_task_projection(
    tmp_path, monkeypatch
):
    supervisor = implementation.PortalImplementationSupervisor(_config(tmp_path))
    _patch_descendant(monkeypatch)
    monkeypatch.setattr(supervisor, "_recorded_managed_daemon_pid", lambda: 41001)
    monkeypatch.setattr(implementation, "process_is_running", lambda pid: pid == 41001)
    monkeypatch.setattr(
        supervisor,
        "_fence_recorded_managed_daemon",
        lambda **_k: pytest.fail("live worker must not be fenced"),
    )
    result = supervisor._terminate_managed_daemon_tree()
    assert result["terminated"] is False
    assert result["quiesced"] is False
    assert result["markers_removed"] is False
    assert result["daemon_fence"]["safe_to_restart"] is False


def test_dirty_file_identity_drift_is_pending_even_with_unchanged_git_tree(
    tmp_path, monkeypatch
):
    supervisor = implementation.PortalImplementationSupervisor(_config(tmp_path))
    loaded = {
        "source_id": "cid:loaded",
        "control_plane_tree_id": "a" * 40,
        "repository_revision": "b" * 40,
    }
    current = {**loaded, "source_id": "cid:changed-dirty-file"}
    supervisor._loaded_control_plane_source = loaded
    supervisor._last_control_plane_source_probe_monotonic = 0
    monkeypatch.setattr(supervisor, "_control_plane_source_snapshot", lambda: current)
    result = supervisor._control_plane_status_projection()
    assert result["control_plane_update_pending"] is True
    assert result["control_plane_source_id"] == "cid:loaded"
    assert result["control_plane_current_source_id"] == "cid:changed-dirty-file"
    assert result["control_plane_update_detected_at"]
