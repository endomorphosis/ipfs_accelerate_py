"""Generation-bound wrapper-status regressions for the multi-supervisor."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    _SupervisorStatusGenerationBinding,
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


def _write_current_status(root: Path, *, supervisor_pid: int) -> Path:
    status_path = root / "state" / "example_supervisor_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "supervisor_pid": supervisor_pid,
            }
        ),
        encoding="utf-8",
    )
    return status_path


def _write_status_gap_worker(
    root: Path,
    *,
    gap_seconds: float,
    recover: bool,
) -> None:
    (root / "worker.py").write_text(
        "\n".join(
            (
                "import json",
                "import os",
                "import signal",
                "import sys",
                "import time",
                "from datetime import datetime, timezone",
                "from pathlib import Path",
                "status = Path('state/example_supervisor_status.json')",
                "def publish():",
                "    temporary = status.with_suffix('.json.tmp')",
                "    payload = {'supervisor_pid': os.getpid(),",
                "               'updated_at': datetime.now(timezone.utc).isoformat()}",
                "    temporary.write_text(json.dumps(payload), encoding='utf-8')",
                "    temporary.replace(status)",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "status.parent.mkdir(parents=True, exist_ok=True)",
                "publish()",
                "time.sleep(0.12)",
                "status.unlink(missing_ok=True)",
                f"time.sleep({gap_seconds!r})",
                *(('publish()',) if recover else ()),
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


def test_post_live_bad_reads_share_fresh_gap_until_generation_recovers(
    tmp_path: Path,
) -> None:
    expected_pid = 222
    binding = _SupervisorStatusGenerationBinding(
        expected_supervisor_pid=expected_pid,
        generation_started_at_epoch_seconds=time.time() - 3600.0,
        startup_grace_seconds=30.0,
    )
    status_path = _write_current_status(
        tmp_path,
        supervisor_pid=expected_pid,
    )

    live = binding.health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
    )
    binding.record_observation(live)
    assert live["supervisor_status"] == "live"
    assert binding.live_status_observed is True

    status_path.unlink()
    missing = binding.health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
    )
    binding.record_observation(missing)
    gap_started_at = binding.status_gap_started_at_epoch_seconds
    assert gap_started_at is not None
    assert missing["supervisor_status_generation"] == "status_gap"
    assert missing["supervisor_status_generation_reason"] == "status_missing"
    assert missing["restart_supervisor"] is False
    assert "supervisor_startup_age_seconds" not in missing

    status_path.write_text("{invalid", encoding="utf-8")
    invalid = binding.health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
    )
    binding.record_observation(invalid)
    assert invalid["supervisor_status_generation_reason"] == (
        "status_missing_or_invalid"
    )
    assert binding.status_gap_started_at_epoch_seconds == gap_started_at
    assert invalid["restart_supervisor"] is False

    _write_current_status(tmp_path, supervisor_pid=999)
    mismatched = binding.health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
    )
    binding.record_observation(mismatched)
    assert mismatched["supervisor_status_generation_reason"] == (
        "supervisor_pid_mismatch"
    )
    assert binding.status_gap_started_at_epoch_seconds == gap_started_at
    assert mismatched["restart_supervisor"] is False

    _write_current_status(tmp_path, supervisor_pid=expected_pid)
    recovered = binding.health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
    )
    binding.record_observation(recovered)
    assert recovered["supervisor_status_generation_valid"] is True
    assert binding.status_gap_started_at_epoch_seconds is None


def test_post_live_status_gap_restarts_after_its_own_bound(
    tmp_path: Path,
) -> None:
    now = time.time()
    binding = _SupervisorStatusGenerationBinding(
        expected_supervisor_pid=222,
        generation_started_at_epoch_seconds=now - 3600.0,
        startup_grace_seconds=1.0,
        live_status_observed=True,
        status_gap_started_at_epoch_seconds=now - 2.0,
    )

    fields = binding.health_fields(
        _track().resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
    )

    assert fields["supervisor_status"] == "stale"
    assert fields["supervisor_status_generation"] == "status_gap"
    assert fields["supervisor_status_gap_age_seconds"] >= 2.0
    assert fields["restart_supervisor"] is True
    assert "supervisor_startup_age_seconds" not in fields


def test_status_gap_state_fails_closed_outside_exact_generation(
    tmp_path: Path,
) -> None:
    now = time.time()
    with pytest.raises(ValueError, match="gap binding"):
        _SupervisorStatusGenerationBinding(
            expected_supervisor_pid=222,
            generation_started_at_epoch_seconds=now - 2.0,
            startup_grace_seconds=1.0,
            live_status_observed=False,
            status_gap_started_at_epoch_seconds=now - 1.0,
        )

    binding = _SupervisorStatusGenerationBinding(
        expected_supervisor_pid=222,
        generation_started_at_epoch_seconds=now - 2.0,
        startup_grace_seconds=1.0,
    )
    with pytest.raises(ValueError, match="escaped its process generation"):
        binding.record_observation(
            {
                "supervisor_status": "starting",
                "expected_supervisor_pid": 333,
            }
        )
    with pytest.raises(ValueError, match="valid status escaped"):
        binding.record_observation(
            {
                "supervisor_status": "live",
                "supervisor_status_generation_valid": True,
                "expected_supervisor_pid": 222,
                "observed_supervisor_pid": 333,
            }
        )

    with pytest.raises(ValueError, match="prior valid live status"):
        supervisor_status_health_fields(
            _track().resolve(tmp_path),
            repo_root=tmp_path,
            stale_seconds=60.0,
            expected_supervisor_pid=222,
            generation_started_at_epoch_seconds=now - 2.0,
            startup_grace_seconds=1.0,
            status_gap_started_at_epoch_seconds=now - 1.0,
        )


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


def test_runner_preserves_live_popen_across_transient_post_live_gap(
    tmp_path: Path,
) -> None:
    _write_status_gap_worker(tmp_path, gap_seconds=0.08, recover=True)

    output: list[str] = []
    result = run_supervisor_tracks(
        [_track()],
        repo_root=tmp_path,
        common_args=("--watchdog-startup-grace-seconds", "0.2"),
        duration_seconds=0.4,
        heartbeat_interval_seconds=0.03,
        supervisor_status_stale_seconds=1.0,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="transient status-gap runner",
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) == 1
    assert not any("restarting stale T supervisor" in line for line in output)
    assert any(
        "supervisor_status_generation=status_gap" in line
        and "supervisor_within_status_gap_grace=true" in line
        for line in output
    )


def test_runner_restarts_live_popen_after_persistent_post_live_gap(
    tmp_path: Path,
) -> None:
    _write_status_gap_worker(tmp_path, gap_seconds=5.0, recover=False)

    output: list[str] = []
    result = run_supervisor_tracks(
        [_track()],
        repo_root=tmp_path,
        common_args=("--watchdog-startup-grace-seconds", "0.06"),
        duration_seconds=0.36,
        heartbeat_interval_seconds=0.03,
        supervisor_status_stale_seconds=1.0,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="persistent status-gap runner",
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) >= 2
    assert any("restarting stale T supervisor" in line for line in output)
    assert any(
        "supervisor_status_generation=status_gap" in line
        and "restart_supervisor=true" in line
        for line in output
    )


def test_adopt_master_pid_quarantines_dead_legacy_projection(tmp_path: Path) -> None:
    import os

    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        _adopt_or_create_current_master_pid_projection,
    )

    pid_path = tmp_path / "configured-board-master.pid"
    dead_pid = 999_999_999
    pid_path.write_bytes(f"{dead_pid}\n".encode("ascii"))
    os.chmod(pid_path, 0o600)

    _adopt_or_create_current_master_pid_projection(pid_path)

    assert pid_path.read_bytes() == f"{os.getpid()}\n".encode("ascii")
    quarantines = list(tmp_path.glob(".configured-board-master.pid.stale-*.quarantine"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == f"{dead_pid}\n".encode("ascii")
