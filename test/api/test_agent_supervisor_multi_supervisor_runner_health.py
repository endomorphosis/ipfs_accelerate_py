from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner as runner

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    SupervisorTrack,
    build_configured_multi_supervisor_cli_runner,
    parse_track_spec,
    run_supervisor_tracks,
    start_track,
    supervisor_status_health_fields,
)


def test_cli_runner_forwards_outer_supervisor_startup_grace(tmp_path):
    args = build_configured_multi_supervisor_cli_runner(
        repo_root=tmp_path,
        supervisor_status_startup_grace_seconds=3.5,
    ).args()

    assert (
        args[args.index("--supervisor-status-startup-grace-seconds") + 1]
        == "3.5"
    )


def test_start_track_birth_fence_precedes_immediate_child_status(
    tmp_path, monkeypatch
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "example_supervisor_status.json"

    class ImmediateStatusProcess:
        pid = os.getpid()

    def immediate_status_popen(*_args, **_kwargs):
        status_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "supervisor_pid": os.getpid(),
                }
            ),
            encoding="utf-8",
        )
        return ImmediateStatusProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", immediate_status_popen)
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")

    process = start_track(
        _track(tmp_path),
        repo_root=tmp_path,
        common_args=[],
        python_executable=sys.executable,
        output=lambda _line: None,
    )
    started_at = process._agent_supervisor_started_at_epoch_seconds
    fields = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=started_at,
        supervisor_status_startup_grace_seconds=30.0,
        supervisor_status_started_monotonic_seconds=(
            process._agent_supervisor_started_at_monotonic_seconds
        ),
    )

    assert started_at <= status_path.stat().st_mtime
    assert fields["supervisor_status"] == "live"


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
    supervisor_pid: int | None = None,
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
                "supervisor_pid": (
                    os.getpid() if supervisor_pid is None else supervisor_pid
                ),
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


def test_prior_generation_status_waits_for_current_birth_during_grace(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    child_log_path = state_dir / "old_child.log"
    child_log_path.write_text("old idle pass\n", encoding="utf-8")
    _write_stale_status(
        tmp_path,
        daemon_pid=None,
        log_path=child_log_path,
        supervisor_pid=999_999,
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time(),
        supervisor_status_startup_grace_seconds=30.0,
    )

    assert fields["supervisor_status"] == "awaiting_current_generation"
    assert fields["supervisor_prior_generation_status"] == (
        "stale_prior_generation"
    )
    assert fields["supervisor_status_pid_mismatch"] is True
    assert fields["supervisor_status_predates_process"] is True
    assert fields["supervisor_startup_grace_expired"] is False
    assert fields["restart_supervisor"] is False


def test_prior_generation_status_restarts_after_startup_grace(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    child_log_path = state_dir / "old_child.log"
    child_log_path.write_text("old idle pass\n", encoding="utf-8")
    _write_stale_status(
        tmp_path,
        daemon_pid=None,
        log_path=child_log_path,
        supervisor_pid=999_999,
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time() - 5.0,
        supervisor_status_startup_grace_seconds=1.0,
    )

    assert fields["supervisor_status"] == "stale_prior_generation"
    assert fields["supervisor_startup_grace_expired"] is True
    assert fields["restart_supervisor"] is True


def test_current_timestamp_without_exact_supervisor_pid_is_not_live(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "example_supervisor_status.json"
    started_at = time.time() - 0.01
    for recorded_pid in (None, True, "123", 0):
        payload = {
            "status": "running",
            "updated_at": "2999-01-01T00:00:00+00:00",
        }
        if recorded_pid is not None:
            payload["supervisor_pid"] = recorded_pid
        status_path.write_text(json.dumps(payload), encoding="utf-8")

        fields = supervisor_status_health_fields(
            _track(tmp_path),
            repo_root=tmp_path,
            stale_seconds=60.0,
            expected_supervisor_pid=os.getpid(),
            supervisor_status_not_before_epoch_seconds=started_at,
            supervisor_status_startup_grace_seconds=30.0,
        )

        assert fields["supervisor_status"] == "awaiting_current_generation"
        assert fields["supervisor_status_pid_mismatch"] is True
        assert fields["restart_supervisor"] is False


def test_missing_status_restarts_only_after_startup_grace(tmp_path):
    (tmp_path / "state").mkdir()

    waiting = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time(),
        supervisor_status_startup_grace_seconds=30.0,
    )
    expired = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time() - 5.0,
        supervisor_status_startup_grace_seconds=1.0,
    )

    assert waiting["supervisor_status"] == "awaiting_current_generation"
    assert waiting["supervisor_prior_generation_status"] == "missing"
    assert waiting["restart_supervisor"] is False
    assert expired["supervisor_status"] == "missing"
    assert expired["supervisor_startup_grace_expired"] is True
    assert expired["restart_supervisor"] is True


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
        duration_seconds=0.75,
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


def test_new_supervisor_survives_prior_status_until_current_generation(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "task_state.json").write_text(
        json.dumps(
            {
                "active_task_id": "",
                "implementation_in_progress": False,
            }
        ),
        encoding="utf-8",
    )
    old_log_path = state_dir / "old_child.log"
    old_log_path.write_text("old idle pass\n", encoding="utf-8")
    _write_stale_status(
        tmp_path,
        daemon_pid=None,
        log_path=old_log_path,
        supervisor_pid=999_999,
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import signal",
                "import sys",
                "import time",
                "from datetime import datetime, timezone",
                "from pathlib import Path",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "time.sleep(0.2)",
                "Path('state/example_supervisor_status.json').write_text(",
                "    json.dumps({",
                "        'status': 'running',",
                "        'updated_at': datetime.now(timezone.utc).isoformat(),",
                "        'supervisor_pid': os.getpid(),",
                "        'current_status_path': 'state/task_state.json',",
                "    }),",
                "    encoding='utf-8',",
                ")",
                "while True:",
                "    time.sleep(0.02)",
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
        duration_seconds=0.9,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=2.0,
        supervisor_status_startup_grace_seconds=0.6,
        stop_grace_seconds=0.3,
        python_executable=sys.executable,
        label="test runner",
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) == 1
    assert not any("restarting stale T supervisor" in line for line in output)
    assert any(
        "supervisor_status=awaiting_current_generation" in line
        for line in output
    )
    assert any("supervisor_status=live" in line for line in output)


def test_restarted_supervisor_receives_a_fresh_generation_grace(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import signal",
                "import sys",
                "import time",
                "from datetime import datetime, timezone",
                "from pathlib import Path",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "counter = Path('state/generation.txt')",
                "generation = int(counter.read_text() or '0') + 1 "
                "if counter.exists() else 1",
                "counter.write_text(str(generation), encoding='utf-8')",
                "status = Path('state/example_supervisor_status.json')",
                "if generation == 1:",
                "    status.write_text(json.dumps({",
                "        'status': 'running',",
                "        'updated_at': '2000-01-01T00:00:00+00:00',",
                "        'supervisor_pid': os.getpid(),",
                "    }), encoding='utf-8')",
                "else:",
                "    time.sleep(0.2)",
                "    status.write_text(json.dumps({",
                "        'status': 'running',",
                "        'updated_at': datetime.now(timezone.utc).isoformat(),",
                "        'supervisor_pid': os.getpid(),",
                "    }), encoding='utf-8')",
                "while True:",
                "    time.sleep(0.02)",
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
        duration_seconds=1.4,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=2.0,
        supervisor_status_startup_grace_seconds=0.5,
        stop_grace_seconds=0.3,
        python_executable=sys.executable,
        label="test runner",
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) == 2
    assert sum("restarting stale T supervisor" in line for line in output) == 1
    assert any("supervisor_status=live" in line for line in output)
    assert (state_dir / "generation.txt").read_text(encoding="utf-8") == "2"


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
