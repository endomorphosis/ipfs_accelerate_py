"""A persisted active task cannot indefinitely disable supervisor recovery."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    parse_track_spec,
    run_supervisor_tracks,
    supervisor_status_health_fields,
)


def _health(tmp_path: Path, *, child: dict, generation_age: float = 900):
    now = datetime.now(UTC)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (state / "child.json").write_text(json.dumps(child), encoding="utf-8")
    (state / "lane_supervisor_status.json").write_text(
        json.dumps(
            {
                "supervisor_pid": 123,
                "updated_at": (now - timedelta(seconds=120)).isoformat(),
                "current_status_path": "state/child.json",
            }
        ),
        encoding="utf-8",
    )
    track = parse_track_spec(
        "lane|worker.py|logs/{stamp}.log|state/lane_supervisor.pid|state/daemon.pid",
        stamp="RUN",
    ).resolve(tmp_path)
    fields = supervisor_status_health_fields(
        track,
        repo_root=tmp_path,
        stale_seconds=60,
        expected_supervisor_pid=123,
        generation_started_at_epoch_seconds=(
            now - timedelta(seconds=generation_age)
        ).timestamp(),
    )
    return fields


@pytest.mark.parametrize(
    "heartbeat",
    (None, "invalid", "2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00"),
)
def test_stale_active_projection_does_not_disable_restart(tmp_path, heartbeat):
    child = {"active_task_id": "TASK-001", "implementation_in_progress": True}
    if heartbeat is not None:
        child["heartbeat_at"] = heartbeat
    fields = _health(tmp_path, child=child)

    assert fields["supervisor_status_generation"] == "valid"
    assert fields["supervisor_status"] == "stale"
    assert fields["restart_supervisor"] is True
    assert fields["supervisor_child_heartbeat_fresh"] is False
    # Health diagnosis preserves the task projection for normal recovery.
    assert json.loads((tmp_path / "state/child.json").read_text()) == child


@pytest.mark.parametrize("timestamp_field", ("heartbeat_at", "updated_at"))
def test_active_child_with_fresh_progress_preserves_supervisor(tmp_path, timestamp_field):
    fields = _health(
        tmp_path,
        child={
            "active_task_id": "TASK-001",
            "implementation_in_progress": True,
            timestamp_field: datetime.now(UTC).isoformat(),
        },
    )

    assert fields["supervisor_status"] == "stale_active"
    assert fields["restart_supervisor"] is False
    assert fields["supervisor_child_heartbeat_fresh"] is True


def test_fresh_idle_projection_does_not_protect_stalled_supervisor(tmp_path):
    fields = _health(
        tmp_path, child={"heartbeat_at": datetime.now(UTC).isoformat()}
    )

    assert fields["supervisor_status"] == "stale"
    assert fields["restart_supervisor"] is True


def test_recent_child_from_prior_generation_cannot_extend_startup(tmp_path):
    now = datetime.now(UTC)
    fields = _health(
        tmp_path,
        child={
            "active_task_id": "TASK-001",
            "implementation_in_progress": True,
            "heartbeat_at": (now - timedelta(seconds=30)).isoformat(),
        },
        generation_age=10,
    )

    assert fields["restart_supervisor"] is True
    assert fields["supervisor_status_generation_reason"] == "status_predates_process_generation"


def test_active_heartbeat_exemption_expires_when_progress_stops(tmp_path):
    child = {
        "active_task_id": "TASK-001",
        "implementation_in_progress": True,
        "heartbeat_at": datetime.now(UTC).isoformat(),
    }
    assert _health(tmp_path, child=child)["restart_supervisor"] is False
    child["heartbeat_at"] = (datetime.now(UTC) - timedelta(seconds=61)).isoformat()

    assert _health(tmp_path, child=child)["restart_supervisor"] is True


def test_runner_restarts_stalled_active_generation(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, os, signal, sys, time\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "state = Path('state')\n"
        "state.mkdir(exist_ok=True)\n"
        "stamp = datetime.now(timezone.utc).isoformat()\n"
        "(state / 'child.json').write_text(json.dumps({\n"
        "    'active_task_id': 'TASK-001', 'implementation_in_progress': True,\n"
        "    'heartbeat_at': stamp}))\n"
        "(state / 'lane_supervisor_status.json').write_text(json.dumps({\n"
        "    'supervisor_pid': os.getpid(), 'updated_at': stamp,\n"
        "    'current_status_path': 'state/child.json'}))\n"
        "while True:\n"
        "    time.sleep(0.01)\n",
        encoding="utf-8",
    )
    track = parse_track_spec(
        "lane|worker.py|logs/{stamp}.log|state/lane_supervisor.pid|state/daemon.pid",
        stamp="RUN",
    )
    output = []
    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=("--watchdog-startup-grace-seconds", "0.25"),
        duration_seconds=0.65,
        heartbeat_interval_seconds=0.025,
        supervisor_status_stale_seconds=0.15,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        output=output.append,
    )

    assert result["completed"] is True
    assert any("restarting stale lane supervisor" in line for line in output)
    assert sum("started lane supervisor" in line for line in output) >= 2
    assert json.loads((tmp_path / "state/child.json").read_text())["active_task_id"] == "TASK-001"
