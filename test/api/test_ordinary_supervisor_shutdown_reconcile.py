"""Signal shutdown may reconcile state only after native daemon quiescence."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTaskState, parse_task_file,
)
from test.api.test_ordinary_supervisor_child_fence import _markers, _native_owner


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


@pytest.mark.parametrize("identity_fault", ["missing", ""])
def test_actual_signal_finally_reconciles_only_after_native_daemon_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity_fault: str,
) -> None:
    with _native_owner(tmp_path, identity_fault=identity_fault) as (supervisor, process, _):
        repo = supervisor.config.repo_root
        board = supervisor.config.todo_path
        board.write_text(board.read_text().replace("- Status: completed", "- Status: in_progress"))
        (repo / ".gitignore").write_text("state/\n", encoding="utf-8")
        source = repo / "answer.py"
        source.write_text("ORIGINAL = True\n", encoding="utf-8")
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, "config", "user.name", "Native Supervisor Test")
        _git(repo, "config", "user.email", "native-supervisor@example.invalid")
        _git(repo, "add", ".gitignore", "answer.py", board.name, "authored_daemon.py")
        _git(repo, "commit", "-q", "-m", "Private native shutdown fixture")
        _git(repo, "branch", "implementation/tasks")
        task = parse_task_file(board, supervisor.config.task_prefix)[0]
        # A selected attempt may precede worker/worktree allocation. Its live
        # managed daemon still owns the ability to continue this attempt.
        state = PortalTaskState(
            active_task_id=task.task_id, active_task_key=task.canonical_task_key,
            active_task_cid=task.canonical_task_cid, active_attempt=2,
            active_phase="implementation", implementation_in_progress=True,
            task_identities={task.task_id: {
                "canonical_task_key": task.canonical_task_key,
                "canonical_task_cid": task.canonical_task_cid,
            }},
        )
        state.save(supervisor.config.state_path)
        before_state = asdict(PortalTaskState.load(supervisor.config.state_path))
        before_markers = _markers(supervisor)
        before_head = _git(repo, "rev-parse", "HEAD")
        before_source, before_board = source.read_bytes(), board.read_bytes()

        def deliver_sigterm_at_scheduler_entry() -> None:
            os.kill(os.getpid(), signal.SIGTERM)
            pytest.fail("the native supervisor signal handler did not interrupt scheduler entry")

        # Only scheduler entry is injected. Signal handling, cleanup, native
        # state storage and recovery/mutation owners execute unchanged.
        monkeypatch.setattr(supervisor, "_run_forever_loop", deliver_sigterm_at_scheduler_entry)
        with pytest.raises(SystemExit) as stopped:
            supervisor.run_forever()
        assert stopped.value.code == 128 + signal.SIGTERM
        status = json.loads(supervisor._supervisor_status_path().read_text())
        cleanup = status["managed_daemon_cleanup"]
        reconciliation = status["interrupted_implementation_reconciliation"]
        after_state = PortalTaskState.load(supervisor.config.state_path)
        assert _git(repo, "rev-parse", "HEAD") == before_head
        assert source.read_bytes() == before_source and board.read_bytes() == before_board
        if identity_fault:
            assert process.poll() is None
            assert cleanup["quiesced"] is False and status["status"] == "termination_blocked"
            assert _markers(supervisor) == before_markers
            assert asdict(after_state) == before_state
            assert reconciliation["reconciled"] is False and reconciliation["blocked"] is True
        else:
            assert cleanup["quiesced"] is True and status["status"] == "stopped"
            process.wait(timeout=3)
            assert reconciliation["reconciled"] is True
            assert after_state.implementation_in_progress is False
            assert after_state.active_task_id == "" and after_state.active_task_cid == ""
            assert after_state.active_attempt == 0 and after_state.active_phase == ""
            assert after_state.implementation_attempts_by_cid[task.canonical_task_cid] == 2
            assert all(value is None for value in _markers(supervisor).values())
