from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner as runner

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
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


def _database_program(
    *,
    store_generation: str = "generation-38",
) -> DatabaseProgramConfig:
    return DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="handle:test-quack",
        quack_endpoint="quack:127.0.0.1:24068",
        store_id="control.duckdb",
        store_generation=store_generation,
        schema_revision="datasets-authoritative-operational-v1",
        failover_policy="fail_closed",
    )


def _track(
    repo_root: Path,
    *,
    store_generation: str = "",
) -> SupervisorTrack:
    return SupervisorTrack(
        name="T",
        script_path=repo_root / "worker.py",
        log_path=repo_root / "logs" / "wrapper.log",
        supervisor_pid_path=repo_root / "state" / "example_supervisor.pid",
        daemon_pid_path=repo_root / "state" / "example_managed_daemon.pid",
        database_program=(
            _database_program(store_generation=store_generation)
            if store_generation
            else None
        ),
    )


def _shared_authority_terminal_payload(
    *,
    supervisor_pid: int,
    repo_root: Path,
    store_generation: str = "generation-38",
) -> dict[str, object]:
    return {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "todo_implementation_supervisor.supervisor"
        ),
        "status": "shared_authority_terminal",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "supervisor_pid": supervisor_pid,
        "run_id": "terminal-run",
        "shared_authority_terminal": True,
        "terminal_kind": "shared_database_authority_unavailable",
        "configured_store_generation": store_generation,
        "attempt_budget_consumed": False,
        "provider_invocation_consumed": False,
        "task_completion_authority": False,
        "generation_restart_authorized": False,
        "control_plane_reload_authorized": False,
        "operator_successor_required": True,
        "database_authority_watchdog": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-authority-watchdog@1"
            ),
            "state": "terminal",
            "terminal": True,
        },
        "database_authority_terminal_guard": {
            "safe": True,
            "blockers": [],
            "attempt_budget_consumed": False,
            "provider_invocation_consumed": False,
            "task_completion_authority": False,
            "generation_restart_authorized": False,
            "operator_successor_required": True,
        },
    }


def test_fresh_process_bound_shared_authority_terminal_forbids_restart(
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "example_supervisor_status.json"
    started_at = time.time() - 0.01
    status_path.write_text(
        json.dumps(
            _shared_authority_terminal_payload(
                supervisor_pid=os.getpid(),
                repo_root=tmp_path,
            )
        ),
        encoding="utf-8",
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path, store_generation="generation-38"),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=started_at,
    )

    assert fields["supervisor_status"] == "shared_authority_terminal"
    assert fields["shared_authority_terminal"] is True
    assert fields["shared_authority_terminal_kind"] == (
        "shared_database_authority_unavailable"
    )
    assert fields["configured_store_generation"] == "generation-38"
    assert fields["supervisor_terminal_process_bound"] is True
    assert fields["supervisor_terminal_fresh"] is True
    assert fields.get("restart_supervisor") is not True


def test_shared_authority_terminal_rejects_wrong_process_binding(tmp_path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "example_supervisor_status.json"
    status_path.write_text(
        json.dumps(
            _shared_authority_terminal_payload(
                supervisor_pid=999_999,
                repo_root=tmp_path,
            )
        ),
        encoding="utf-8",
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path, store_generation="generation-38"),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time() - 0.01,
    )

    assert fields["supervisor_status"] == "live"
    assert fields.get("shared_authority_terminal") is not True


def test_shared_authority_terminal_rejects_foreign_store_generation(
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "example_supervisor_status.json"
    status_path.write_text(
        json.dumps(
            _shared_authority_terminal_payload(
                supervisor_pid=os.getpid(),
                repo_root=tmp_path,
                store_generation="generation-37",
            )
        ),
        encoding="utf-8",
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path, store_generation="generation-38"),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time() - 0.01,
    )

    assert fields["supervisor_status"] == "live"
    assert fields.get("shared_authority_terminal") is not True


def test_shared_authority_terminal_rejects_symlink_status_artifact(
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    target = state_dir / "untrusted.json"
    target.write_text(
        json.dumps(
            _shared_authority_terminal_payload(
                supervisor_pid=os.getpid(),
                repo_root=tmp_path,
            )
        ),
        encoding="utf-8",
    )
    (state_dir / "example_supervisor_status.json").symlink_to(target)

    fields = supervisor_status_health_fields(
        _track(tmp_path, store_generation="generation-38"),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time() - 0.01,
    )

    assert fields["supervisor_status"] == "unsafe"
    assert fields.get("shared_authority_terminal") is not True


def test_process_bound_pending_authority_terminal_is_visible_but_not_actionable(
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "example_supervisor_status.json"
    payload = _shared_authority_terminal_payload(
        supervisor_pid=os.getpid(),
        repo_root=tmp_path,
    )
    payload.update(
        {
            "status": "running",
            "shared_authority_terminal": False,
            "shared_authority_terminal_pending": True,
        }
    )
    payload["database_authority_terminal_guard"] = {
        "safe": False,
        "blockers": ["implementation_worker_active"],
        "attempt_budget_consumed": False,
        "provider_invocation_consumed": False,
        "task_completion_authority": False,
        "generation_restart_authorized": False,
        "operator_successor_required": True,
    }
    started_at = time.time() - 0.01
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    fields = supervisor_status_health_fields(
        _track(tmp_path, store_generation="generation-38"),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=started_at,
    )

    assert fields["supervisor_status"] == "live"
    assert fields["shared_authority_terminal_pending"] is True
    assert fields.get("shared_authority_terminal") is not True
    assert fields.get("restart_supervisor") is not True
    assert "shared_authority_terminal_pending=true" in (
        runner.format_supervisor_status_fields(fields)
    )


def test_multi_runner_fences_wave_on_shared_authority_terminal(tmp_path) -> None:
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
                "payload = "
                + repr(
                    _shared_authority_terminal_payload(
                        supervisor_pid=0,
                        repo_root=tmp_path,
                    )
                ),
                "payload['supervisor_pid'] = os.getpid()",
                "payload['updated_at'] = datetime.now(timezone.utc).isoformat()",
                "Path('state/example_supervisor_status.json').write_text(",
                "    json.dumps(payload), encoding='utf-8')",
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
    track = replace(track, database_program=_database_program())
    output: list[str] = []

    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=1.0,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=60.0,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="test runner",
        output=output.append,
    )

    assert result["completed"] is False
    assert result["all_trees_fenced"] is True
    assert result["blocked"] == (
        "shared database authority unavailable; every live lane reached a "
        "process-bound safe terminal and exact generation restart requires "
        "a sealed operator successor"
    )
    assert result["shared_authority_terminals"] == [
        {
            "track": "T",
            "terminal_kind": "shared_database_authority_unavailable",
            "configured_store_generation": "generation-38",
            "task_completion_authority": False,
            "generation_restart_authorized": False,
            "control_plane_reload_authorized": False,
            "operator_successor_required": True,
            "supervisor_status_path": str(
                state_dir / "example_supervisor_status.json"
            ),
            "supervisor_terminal_process_bound": True,
            "supervisor_terminal_fresh": True,
        }
    ]
    assert sum("started T supervisor" in line for line in output) == 1
    assert not any("restarting exited T supervisor" in line for line in output)
    assert any(
        "shared database authority unavailable" in line for line in output
    )
    assert any(
        "ended at typed shared-authority terminal" in line for line in output
    )
    assert "completed requested run window" not in output


def test_multi_runner_waits_for_every_live_lane_authority_terminal(
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    def write_worker(
        *,
        name: str,
        pid_stem: str,
        delay_seconds: float,
    ) -> Path:
        worker_path = tmp_path / f"worker_{name.lower()}.py"
        payload = _shared_authority_terminal_payload(
            supervisor_pid=0,
            repo_root=tmp_path,
        )
        worker_path.write_text(
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
                    f"time.sleep({delay_seconds!r})",
                    "payload = " + repr(payload),
                    "payload['supervisor_pid'] = os.getpid()",
                    "payload['updated_at'] = datetime.now(timezone.utc).isoformat()",
                    f"Path('state/{pid_stem}_supervisor_status.json').write_text(",
                    "    json.dumps(payload), encoding='utf-8')",
                    f"Path('state/{name.lower()}_published').write_text(",
                    "    'published', encoding='utf-8')",
                    "while True:",
                    "    time.sleep(0.02)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return worker_path

    write_worker(name="A", pid_stem="a", delay_seconds=0.0)
    write_worker(name="B", pid_stem="b", delay_seconds=0.35)
    tracks: list[SupervisorTrack] = []
    for name, stem in (("A", "a"), ("B", "b")):
        track = parse_track_spec(
            f"{name}|worker_{name.lower()}.py|logs/{name}.log|"
            f"state/{stem}_supervisor.pid|state/{stem}_managed_daemon.pid",
            stamp="RUN",
        )
        tracks.append(replace(track, database_program=_database_program()))
    output: list[str] = []
    started = time.monotonic()

    result = run_supervisor_tracks(
        tracks,
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=2.0,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=60.0,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="two-lane authority drain",
        output=output.append,
    )

    assert time.monotonic() - started >= 0.3
    assert result["completed"] is False
    assert result["all_trees_fenced"] is True
    assert [item["track"] for item in result["shared_authority_terminals"]] == [
        "A",
        "B",
    ]
    assert result["shared_authority_fenced_tracks"] == []
    assert (state_dir / "b_published").read_text(encoding="utf-8") == "published"
    assert any("shared-authority drain pending track=B" in line for line in output)
    assert not any("restarting" in line for line in output)


def test_multi_runner_prescans_later_terminal_before_earlier_exit_action(
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker_a.py").write_text(
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    payload = _shared_authority_terminal_payload(
        supervisor_pid=0,
        repo_root=tmp_path,
    )
    (tmp_path / "worker_b.py").write_text(
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
                "payload = " + repr(payload),
                "payload['supervisor_pid'] = os.getpid()",
                "payload['updated_at'] = datetime.now(timezone.utc).isoformat()",
                "Path('state/b_supervisor_status.json').write_text(",
                "    json.dumps(payload), encoding='utf-8')",
                "while True:",
                "    time.sleep(0.02)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tracks: list[SupervisorTrack] = []
    for name, stem in (("A", "a"), ("B", "b")):
        track = parse_track_spec(
            f"{name}|worker_{name.lower()}.py|logs/{name}.log|"
            f"state/{stem}_supervisor.pid|state/{stem}_managed_daemon.pid",
            stamp="RUN",
        )
        tracks.append(replace(track, database_program=_database_program()))
    output: list[str] = []

    result = run_supervisor_tracks(
        tracks,
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=1.0,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=60.0,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="terminal pre-scan ordering",
        output=output.append,
    )

    assert result["completed"] is False
    assert result["all_trees_fenced"] is True
    assert [item["track"] for item in result["shared_authority_terminals"]] == [
        "B"
    ]
    assert [
        item["track"] for item in result["shared_authority_fenced_tracks"]
    ] == ["A"]
    assert sum("started A supervisor" in line for line in output) == 1
    assert not any("restarting exited A supervisor" in line for line in output)
    assert not any("reassigned" in line for line in output)


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


def test_stale_lane_fence_failure_does_not_stop_other_lanes(
    tmp_path, monkeypatch
):
    """One unfenceable stale lane must not interrupt the remaining live lanes."""

    def _worker_source(*, stale: bool) -> str:
        status_name = (
            "stale_supervisor_status.json" if stale else "live_supervisor_status.json"
        )
        updated_at = (
            "'2000-01-01T00:00:00+00:00'"
            if stale
            else "datetime.now(timezone.utc).isoformat()"
        )
        return "\n".join(
            [
                "import json",
                "import os",
                "import signal",
                "import sys",
                "import time",
                "from datetime import datetime, timezone",
                "from pathlib import Path",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "Path('state').mkdir(exist_ok=True)",
                "Path('state/task_state.json').write_text(",
                "    json.dumps({'active_task_id': '', "
                "'implementation_in_progress': False}),",
                "    encoding='utf-8',",
                ")",
                f"Path('state/{status_name}').write_text(",
                "    json.dumps({",
                "        'status': 'running',",
                f"        'updated_at': {updated_at},",
                "        'supervisor_pid': os.getpid(),",
                "        'current_status_path': 'state/task_state.json',",
                "    }),",
                "    encoding='utf-8',",
                ")",
                "while True:",
                "    time.sleep(0.05)",
            ]
        ) + "\n"

    (tmp_path / "stale.py").write_text(_worker_source(stale=True), encoding="utf-8")
    (tmp_path / "live.py").write_text(_worker_source(stale=False), encoding="utf-8")
    stale_track = parse_track_spec(
        "S|stale.py|logs/{stamp}-s.log|state/stale_supervisor.pid|"
        "state/stale_managed_daemon.pid",
        stamp="RUN",
    )
    live_track = parse_track_spec(
        "L|live.py|logs/{stamp}-l.log|state/live_supervisor.pid|"
        "state/live_managed_daemon.pid",
        stamp="RUN",
    )
    output: list[str] = []

    def refuse_fence(process, *, grace_seconds):
        del process, grace_seconds
        return False, ()

    monkeypatch.setattr(runner, "_terminate_managed_process", refuse_fence)

    result = run_supervisor_tracks(
        [stale_track, live_track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.8,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="test runner",
        output=output.append,
    )

    assert result["interrupted"] == ""
    assert result["completed"] is True
    assert not any(
        "interrupted: could not fence stale S process tree" in line
        for line in output
    )
    assert any(
        "could not fence stale S process tree; leaving remaining lanes running"
        in line
        for line in output
    )
    stale_restart_indexes = [
        index
        for index, line in enumerate(output)
        if "could not fence stale S process tree; leaving remaining lanes running"
        in line
    ]
    assert stale_restart_indexes
    assert any(
        index > stale_restart_indexes[0] and "heartbeat L " in line
        for index, line in enumerate(output)
    )


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
