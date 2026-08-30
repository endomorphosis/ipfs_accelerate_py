from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    SupervisorTrack,
    parse_track_spec,
    run_supervisor_tracks,
    supervisor_status_health_fields,
)


def _track(repo_root: Path) -> SupervisorTrack:
    return SupervisorTrack(
        name="T",
        script_path=repo_root / "worker.py",
        log_path=repo_root / "logs" / "wrapper.log",
        supervisor_pid_path=repo_root / "state" / "example_supervisor.pid",
        daemon_pid_path=repo_root / "state" / "example_managed_daemon.pid",
    )


def _write_stale_status(
    repo_root: Path,
    *,
    daemon_pid: int | None,
    log_path: Path,
    fallback_enabled: bool = True,
) -> None:
    state_dir = repo_root / "state"
    (state_dir / "example_supervisor.pid").write_text(
        f"{os.getpid()}\n", encoding="utf-8"
    )
    if daemon_pid is not None:
        (state_dir / "example_managed_daemon.pid").write_text(
            f"{daemon_pid}\n", encoding="utf-8"
        )
    (state_dir / "example_supervisor_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "updated_at": "2000-01-01T00:00:00+00:00",
                "repo_root": str(repo_root),
                "supervisor_pid": os.getpid(),
                "daemon_pid": daemon_pid,
                "run_id": "current-run",
                "log_path": str(log_path.relative_to(repo_root)),
                "current_status_path": "state/missing_task_state.json",
                "watchdog_accept_fresh_child_log": fallback_enabled,
            }
        ),
        encoding="utf-8",
    )


def test_stale_supervisor_accepts_exact_fresh_process_bound_child_log(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    child_log_path = state_dir / "example_implementation_daemon_run.log"
    child_log_path.write_text("implementation still advancing\n", encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
    )
    try:
        _write_stale_status(
            tmp_path,
            daemon_pid=child.pid,
            log_path=child_log_path,
        )

        fields = supervisor_status_health_fields(
            _track(tmp_path),
            repo_root=tmp_path,
            stale_seconds=60.0,
            expected_supervisor_pid=os.getpid(),
            supervisor_status_not_before_epoch_seconds=0.0,
        )

        assert fields["supervisor_status"] == "stale_child_log_live"
        assert fields["supervisor_child_log_fresh"] is True
        assert fields["supervisor_child_log_process_bound"] is True
        assert fields["supervisor_child_log_daemon_pid"] == child.pid
        assert fields["restart_supervisor"] is False
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_stale_supervisor_restarts_when_fresh_log_is_not_child_bound(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    child_log_path = state_dir / "example_implementation_daemon_run.log"
    child_log_path.write_text("unbound fresh bytes\n", encoding="utf-8")
    _write_stale_status(
        tmp_path,
        daemon_pid=os.getpid(),
        log_path=child_log_path,
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=0.0,
    )

    assert fields["supervisor_status"] == "stale"
    assert fields["supervisor_child_log_fresh"] is False
    assert fields["supervisor_child_log_process_bound"] is False
    assert fields["restart_supervisor"] is True


def test_stale_idle_supervisor_still_restarts_with_fallback_enabled(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    child_log_path = state_dir / "old_child.log"
    child_log_path.write_text("old idle pass\n", encoding="utf-8")
    _write_stale_status(
        tmp_path,
        daemon_pid=None,
        log_path=child_log_path,
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=0.0,
    )

    assert fields["supervisor_status"] == "stale"
    assert fields["restart_supervisor"] is True


def test_multi_runner_still_restarts_stale_idle_supervisor(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json",
                "import signal",
                "import sys",
                "import time",
                "from pathlib import Path",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "Path('state').mkdir(exist_ok=True)",
                "Path('state/task_state.json').write_text(",
                "    json.dumps({'active_task_id': '', "
                "'implementation_in_progress': False}),",
                "    encoding='utf-8',",
                ")",
                "Path('state/example_supervisor_status.json').write_text(",
                "    json.dumps({'updated_at': "
                "'2000-01-01T00:00:00+00:00', "
                "'current_status_path': 'state/task_state.json'}),",
                "    encoding='utf-8',",
                ")",
                "while True:",
                "    time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    track = parse_track_spec(
        "T|worker.py|logs/{stamp}.log|state/example_supervisor.pid|"
        "state/example_managed_daemon.pid",
        stamp="RUN",
    )
    output: list[str] = []

    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.25,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="test runner",
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) >= 2
    assert any("supervisor_status=stale" in line for line in output)
    assert any("restart_supervisor=true" in line for line in output)
    assert any("restarting stale T supervisor" in line for line in output)


def test_stale_supervisor_rejects_fresh_child_log_outside_state_root(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    escaped_log_path = tmp_path / "foreign.log"
    escaped_log_path.write_text("unconfined fresh bytes\n", encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
    )
    try:
        _write_stale_status(
            tmp_path,
            daemon_pid=child.pid,
            log_path=escaped_log_path,
        )

        fields = supervisor_status_health_fields(
            _track(tmp_path),
            repo_root=tmp_path,
            stale_seconds=60.0,
            expected_supervisor_pid=os.getpid(),
            supervisor_status_not_before_epoch_seconds=0.0,
        )

        assert fields["supervisor_status"] == "stale"
        assert fields["supervisor_child_log_fresh"] is False
        assert fields["supervisor_child_log_process_bound"] is False
        assert fields["restart_supervisor"] is True
    finally:
        child.terminate()
        child.wait(timeout=5)
