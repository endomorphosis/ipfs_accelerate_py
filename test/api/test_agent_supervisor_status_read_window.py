"""Transient status reads must not fence and restart a healthy lane."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as runner


def unavailable(reason: str = "status_projection_missing") -> dict[str, object]:
    return {"supervisor_status": "unknown", "supervisor_status_reason": reason,
            "restart_supervisor": True}


def test_sustained_unavailable_observations_eventually_allow_recovery() -> None:
    window = runner.SupervisorStatusReadWindow()
    process = object()
    fields = unavailable()
    for now in (0, 5, 10, 20, 29):
        result = window.observe(fields, process=process, now=now)
        assert result["restart_supervisor"] is False
        assert result["supervisor_status"] == "unknown"
    result = window.observe(unavailable("status_timestamp_invalid"), process=process, now=30)
    assert result["restart_supervisor"] is True
    assert result["supervisor_status_unavailable_seconds"] == 30
    assert fields == unavailable()


def test_good_observation_resets_unavailable_window() -> None:
    window = runner.SupervisorStatusReadWindow()
    process = object()
    window.observe(unavailable(), process=process, now=0)
    good = {"supervisor_status": "live"}
    assert window.observe(good, process=process, now=20) == good
    result = window.observe(unavailable(), process=process, now=30)
    assert result["restart_supervisor"] is False
    assert result["supervisor_status_unavailable_seconds"] == 0


@pytest.mark.parametrize("interruption", ["new_launch", "sample_gap", "backward_clock", "nan", "inf"])
def test_observation_discontinuity_cannot_authorize_restart(interruption: str) -> None:
    window = runner.SupervisorStatusReadWindow()
    process = object()
    window.observe(unavailable(), process=process, now=10)
    window.observe(unavailable(), process=process, now=35)
    if interruption == "new_launch":
        process = object()
        now = 40
    elif interruption == "sample_gap":
        now = 70
    elif interruption == "backward_clock":
        now = 5
    else:
        now = float(interruption)
    result = window.observe(unavailable(), process=process, now=now)
    assert result["restart_supervisor"] is False
    if interruption in {"nan", "inf"}:
        assert result["supervisor_status_recovery_deferred"] == "observation_clock_invalid"
    else:
        assert result["supervisor_status_unavailable_seconds"] == 0


@pytest.mark.parametrize("reason", [None, "supervisor_pid_mismatch", "status_predates_process_birth"])
def test_other_restart_evidence_is_not_deferred(reason: str | None) -> None:
    window = runner.SupervisorStatusReadWindow()
    process = object()
    window.observe(unavailable(), process=process, now=0)
    fields = {"supervisor_status": "stale", "supervisor_status_reason": reason,
              "restart_supervisor": True}
    assert window.observe(fields, process=process, now=1) == fields


@pytest.mark.parametrize("bad_read", ["missing", "invalid_timestamp"])
def test_one_bad_read_does_not_restart_real_live_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_read: str
) -> None:
    script = tmp_path / "lane.py"
    script.write_text(
        "import datetime, json, os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "p = pathlib.Path(__file__).with_name('lane-status.json')\n"
        "while True:\n"
        "    q = p.with_suffix('.tmp')\n"
        "    q.write_text(json.dumps({'supervisor_pid': os.getpid(), 'status': 'running',\n"
        "        'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}))\n"
        "    q.replace(p)\n"
        "    time.sleep(0.01)\n",
        encoding="utf-8",
    )
    status_path = tmp_path / "lane-status.json"
    original_read = runner._read_json_dict
    original_health = runner.supervisor_status_health_fields
    injected = False
    good_after = False
    classified: list[dict[str, object]] = []

    def health(track: runner.SupervisorTrack, **kwargs: object) -> dict[str, object]:
        nonlocal injected, good_after
        # Exercise the real classifier's read, after generation-bound startup
        # grace. Other scheduler reads must not consume the injected failure.
        value = original_read(status_path)
        inject = (not injected and bool(value)
                  and kwargs.get("expected_supervisor_pid") == value.get("supervisor_pid")
                  and kwargs.get("startup_grace_remaining_seconds") == 0)
        if inject:
            def unavailable_read(path: Path) -> dict[str, object]:
                if Path(path) == status_path:
                    return {} if bad_read == "missing" else {**value, "updated_at": "invalid"}
                return original_read(path)
            with monkeypatch.context() as scoped:
                scoped.setattr(runner, "_read_json_dict", unavailable_read)
                fields = original_health(track, **kwargs)
            injected = True
            classified.append(dict(fields))
        else:
            fields = original_health(track, **kwargs)
        if injected and fields.get("supervisor_status") == "live":
            good_after = True
        return fields

    monkeypatch.setattr(runner, "supervisor_status_health_fields", health)
    track = runner.SupervisorTrack(
        name="lane", script_path=script, log_path=tmp_path / "lane.log",
        supervisor_pid_path=tmp_path / "lane.pid", daemon_pid_path=tmp_path / "daemon.pid",
        supervisor_status_path=status_path,
    )
    output: list[str] = []
    result = runner.run_supervisor_tracks(
        (track,), repo_root=tmp_path, common_args=(), duration_seconds=0.5,
        heartbeat_interval_seconds=0.05, supervisor_status_stale_seconds=10,
        supervisor_startup_grace_seconds=0.02, stop_grace_seconds=0.2,
        python_executable=sys.executable, output=output.append,
    )
    assert result["completed"] is True
    assert injected and good_after
    assert len(classified) == 1 and classified[0]["restart_supervisor"] is True
    assert classified[0]["supervisor_status_reason"] == (
        "status_projection_missing" if bad_read == "missing" else "status_timestamp_invalid"
    )
    assert sum("started lane supervisor" in line for line in output) == 1
    assert any("projection_unavailable_grace" in line for line in output)
    assert any("supervisor_status=live" in line for line in output)
    assert not any("restarting stale lane supervisor" in line for line in output)
    pid = json.loads(status_path.read_text())["supervisor_pid"]
    assert not runner.pid_alive(pid)
