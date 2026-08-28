"""Generation-bound wrapper-status regressions for the multi-supervisor."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    _track_supervisor_status_startup_grace_seconds,
    parse_track_spec,
    run_supervisor_tracks,
    supervisor_status_health_fields,
)


def _track():
    return parse_track_spec(
        "T|worker.py|logs/{stamp}.log|state/example_supervisor.pid|state/daemon.pid",
        stamp="RUN",
    )


def _write_prior_status(root: Path, *, supervisor_pid: int) -> None:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "task_state.json").write_text(
        json.dumps(
            {
                "active_task_id": "PCAR-000",
                "implementation_in_progress": True,
            }
        ),
        encoding="utf-8",
    )
    (state / "example_supervisor_status.json").write_text(
        json.dumps(
            {
                "updated_at": "2000-01-01T00:00:00+00:00",
                "supervisor_pid": supervisor_pid,
                "current_status_path": "state/task_state.json",
            }
        ),
        encoding="utf-8",
    )


def _write_worker(root: Path) -> None:
    (root / "worker.py").write_text(
        "\n".join(
            (
                "import signal",
                "import sys",
                "import time",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "while True:",
                "    time.sleep(0.05)",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_prior_generation_is_pending_then_restarts_without_old_child_state(
    tmp_path: Path,
) -> None:
    _write_prior_status(tmp_path, supervisor_pid=111)
    spawn_epoch = time.time()

    starting = supervisor_status_health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=1.0,
        expected_supervisor_pid=222,
        generation_started_at_epoch_seconds=spawn_epoch,
        startup_grace_seconds=300.0,
    )
    assert starting["supervisor_status"] == "starting"
    assert starting["supervisor_status_generation"] == "pending"
    assert starting["supervisor_status_generation_reason"] == (
        "supervisor_pid_mismatch"
    )
    assert starting["restart_supervisor"] is False
    assert starting["supervisor_within_startup_grace"] is True

    timed_out = supervisor_status_health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=1.0,
        expected_supervisor_pid=222,
        generation_started_at_epoch_seconds=spawn_epoch - 301.0,
        startup_grace_seconds=300.0,
    )
    assert timed_out["supervisor_status"] == "stale"
    assert timed_out["restart_supervisor"] is True
    assert timed_out["supervisor_within_startup_grace"] is False
    assert "supervisor_active_task_id" not in timed_out


def test_generation_requires_matching_pid_and_post_spawn_timestamp(
    tmp_path: Path,
) -> None:
    spawn_epoch = time.time()
    status_path = tmp_path / "state" / "example_supervisor_status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "supervisor_pid": 111,
            }
        ),
        encoding="utf-8",
    )
    wrong_pid = supervisor_status_health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=600.0,
        expected_supervisor_pid=222,
        generation_started_at_epoch_seconds=spawn_epoch - 1.0,
        startup_grace_seconds=300.0,
    )
    assert wrong_pid["supervisor_status_generation_reason"] == (
        "supervisor_pid_mismatch"
    )

    status_path.write_text(
        json.dumps(
            {
                "updated_at": "2000-01-01T00:00:00+00:00",
                "supervisor_pid": 222,
            }
        ),
        encoding="utf-8",
    )
    old_timestamp = supervisor_status_health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=600.0,
        expected_supervisor_pid=222,
        generation_started_at_epoch_seconds=spawn_epoch,
        startup_grace_seconds=300.0,
    )
    assert old_timestamp["supervisor_status_generation_reason"] == (
        "status_predates_process_generation"
    )


def test_missing_current_generation_status_restarts_after_grace(
    tmp_path: Path,
) -> None:
    fields = supervisor_status_health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=600.0,
        expected_supervisor_pid=222,
        generation_started_at_epoch_seconds=time.time() - 2.0,
        startup_grace_seconds=1.0,
    )

    assert fields["supervisor_status"] == "stale"
    assert fields["supervisor_status_generation_reason"] == "status_missing"
    assert fields["restart_supervisor"] is True


@pytest.mark.parametrize(
    "common_args",
    (
        (
            "--watchdog-startup-grace-seconds",
            "300",
            "--watchdog-startup-grace-seconds",
            "120",
        ),
        ("--watchdog-startup-grace-seconds", "nan"),
        ("--watchdog-startup-grace-seconds", "-1"),
        ("--watchdog-startup-grace-seconds", "invalid"),
    ),
)
def test_launch_profile_startup_grace_fails_closed(
    common_args: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="startup grace"):
        _track_supervisor_status_startup_grace_seconds(
            _track(),
            common_args=common_args,
            fallback_seconds=600.0,
        )


def test_runner_does_not_recycle_prior_status_during_declared_startup_grace(
    tmp_path: Path,
) -> None:
    _write_prior_status(tmp_path, supervisor_pid=999_999_999)
    _write_worker(tmp_path)

    output: list[str] = []
    result = run_supervisor_tracks(
        [_track()],
        repo_root=tmp_path,
        common_args=("--watchdog-startup-grace-seconds", "0.3"),
        duration_seconds=0.16,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="generation test runner",
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) == 1
    assert not any("restarting stale T supervisor" in line for line in output)
    assert any(
        "supervisor_status=starting" in line
        and "supervisor_status_generation=pending" in line
        and "supervisor_within_startup_grace=true" in line
        for line in output
    )


def test_runner_recycles_prior_status_after_declared_startup_grace(
    tmp_path: Path,
) -> None:
    _write_prior_status(tmp_path, supervisor_pid=999_999_999)
    _write_worker(tmp_path)

    output: list[str] = []
    result = run_supervisor_tracks(
        [_track()],
        repo_root=tmp_path,
        common_args=("--watchdog-startup-grace-seconds", "0.01"),
        duration_seconds=0.16,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=600.0,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="generation timeout test runner",
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) >= 2
    assert any("restarting stale T supervisor" in line for line in output)
    assert any(
        "supervisor_status=stale" in line
        and "supervisor_status_generation=pending" in line
        and "restart_supervisor=true" in line
        for line in output
    )


def _run_status_observation_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observations: list[dict[str, object]],
) -> tuple[dict[str, object], list[str]]:
    _write_worker(tmp_path)
    observed = iter(observations)

    def status_fields(*args, **kwargs):
        del args, kwargs
        return next(
            observed,
            {
                "supervisor_status": "live",
                "restart_supervisor": False,
            },
        )

    monkeypatch.setattr(
        runner_module,
        "supervisor_status_health_fields",
        status_fields,
    )
    output: list[str] = []
    result = run_supervisor_tracks(
        [_track()],
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0.21,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="status observation sequence runner",
        output=output.append,
    )
    return result, output


def test_one_transient_status_miss_after_live_does_not_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, output = _run_status_observation_sequence(
        tmp_path,
        monkeypatch,
        [
            {"supervisor_status": "live", "restart_supervisor": False},
            {
                "supervisor_status": "stale",
                "supervisor_status_generation": "pending",
                "supervisor_status_generation_reason": "status_missing",
                "restart_supervisor": True,
            },
            {"supervisor_status": "live", "restart_supervisor": False},
        ],
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) == 1
    assert not any("restarting stale T supervisor" in line for line in output)
    assert any(
        "supervisor_status_restart_deferred=true" in line for line in output
    )


def test_sustained_status_misses_after_live_restart_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, output = _run_status_observation_sequence(
        tmp_path,
        monkeypatch,
        [
            {"supervisor_status": "live", "restart_supervisor": False},
            {
                "supervisor_status": "stale",
                "supervisor_status_generation": "pending",
                "supervisor_status_generation_reason": "status_missing",
                "restart_supervisor": True,
            },
            {
                "supervisor_status": "stale",
                "supervisor_status_generation": "pending",
                "supervisor_status_generation_reason": "status_missing",
                "restart_supervisor": True,
            },
            {"supervisor_status": "live", "restart_supervisor": False},
        ],
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) == 2
    assert sum(
        "restarting stale T supervisor" in line for line in output
    ) == 1


class _StrictProjectionProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.alive = True

    def poll(self) -> int | None:
        return None if self.alive else 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.alive = False
        return 0


def _strict_projection_restart_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    substitute_projection: bool,
) -> tuple[dict[str, object], list[_StrictProjectionProcess], list[str], Path]:
    track = replace(_track(), extra_args=("--plan-bound-dispatch",))
    pid_path = track.resolve(tmp_path).supervisor_pid_path
    processes: list[_StrictProjectionProcess] = []
    substituted = False

    def strict_start_track(*args, **kwargs):
        del args
        resolved = track.resolve(Path(kwargs["repo_root"]))
        resolved.supervisor_pid_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, identity = runner_module._reserve_owned_pid_projection(
            resolved.supervisor_pid_path
        )
        process = _StrictProjectionProcess(810_000 + len(processes))
        try:
            runner_module._publish_reserved_pid_projection(
                resolved.supervisor_pid_path,
                descriptor,
                identity,
                process.pid,
            )
        finally:
            os.close(descriptor)
        processes.append(process)
        return process

    def terminate(process, *, grace_seconds):
        nonlocal substituted
        del grace_seconds
        process.alive = False
        if substitute_projection and not substituted:
            pid_path.write_text("999999\n", encoding="ascii")
            pid_path.chmod(0o600)
            substituted = True
        return True, (process.pid,)

    def status_fields(*args, **kwargs):
        del args, kwargs
        restart = len(processes) == 1
        return {
            "supervisor_status": "stale" if restart else "ok",
            "supervisor_status_generation": "current",
            "supervisor_status_generation_reason": "status_missing",
            "supervisor_status_age_seconds": 999.0 if restart else 0.0,
            "restart_supervisor": restart,
        }

    monkeypatch.setattr(runner_module, "start_track", strict_start_track)
    monkeypatch.setattr(
        runner_module,
        "daemon_pid_health_fields",
        lambda *args, **kwargs: {
            "daemon_pid": None,
            "daemon_status": "missing",
        },
    )
    monkeypatch.setattr(
        runner_module,
        "supervisor_status_health_fields",
        status_fields,
    )
    monkeypatch.setattr(
        runner_module,
        "pid_alive",
        lambda pid: any(item.pid == pid and item.alive for item in processes),
    )
    monkeypatch.setattr(runner_module, "_terminate_managed_process", terminate)
    monkeypatch.setattr(
        runner_module,
        "stop_tracks",
        lambda *args, **kwargs: {
            "stopped_count": 0,
            "all_trees_fenced": True,
            "removed_runtime_markers": [],
        },
    )

    output: list[str] = []
    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0.11,
        heartbeat_interval_seconds=0.01,
        supervisor_status_stale_seconds=0.01,
        stop_grace_seconds=0.01,
        python_executable=sys.executable,
        label="strict projection generation test runner",
        output=output.append,
    )
    return result, processes, output, pid_path


def test_strict_projection_is_retired_before_same_track_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, processes, output, pid_path = _strict_projection_restart_harness(
        tmp_path,
        monkeypatch,
        substitute_projection=False,
    )

    assert result["completed"] is True
    assert len(processes) == 2
    assert pid_path.read_text(encoding="ascii") == f"{processes[1].pid}\n"
    assert sum("started T supervisor" in line for line in output) == 0
    assert any("restarting stale T supervisor" in line for line in output)


def test_strict_projection_substitution_blocks_same_track_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, processes, _output, pid_path = _strict_projection_restart_harness(
        tmp_path,
        monkeypatch,
        substitute_projection=True,
    )

    assert result["completed"] is False
    assert "could not retire fenced T supervisor PID projection" in str(
        result["interrupted"]
    )
    assert len(processes) == 1
    assert pid_path.read_text(encoding="ascii") == "999999\n"
