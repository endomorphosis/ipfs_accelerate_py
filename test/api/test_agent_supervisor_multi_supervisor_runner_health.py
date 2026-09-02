from __future__ import annotations

import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner as runner
import ipfs_accelerate_py.agent_supervisor.todo_daemon.core as daemon_core

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
    SupervisorTrack,
    build_configured_multi_supervisor_cli_runner,
    parse_track_spec,
    run_supervisor_tracks,
    start_track,
    supervisor_status_health_fields,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor import (
    SupervisorStatusContext,
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


def test_status_reader_retries_one_atomic_replacement_race(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "example_supervisor_status.json"
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
    original = runner._read_stable_regular_json
    calls = 0

    def collide_once(path, *, max_bytes=1_048_576):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise runner._StableArtifactChangedError(
                "simulated atomic status replacement"
            )
        return original(path, max_bytes=max_bytes)

    monkeypatch.setattr(runner, "_read_stable_regular_json", collide_once)

    fields = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time() - 1.0,
        supervisor_status_startup_grace_seconds=0.1,
    )

    assert calls == 2
    assert fields["supervisor_status"] == "live"
    assert fields["supervisor_status_read_retries"] == 1
    assert fields.get("restart_supervisor") is not True


def test_status_reader_does_not_retry_structural_unsafe_artifact(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    calls = 0

    def structurally_unsafe(_path, *, max_bytes=1_048_576):
        nonlocal calls
        calls += 1
        raise runner._StableArtifactReadError("simulated symbolic link")

    monkeypatch.setattr(
        runner,
        "_read_stable_regular_json",
        structurally_unsafe,
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time() - 1.0,
        supervisor_status_startup_grace_seconds=0.1,
    )

    assert calls == 1
    assert fields["supervisor_status"] == "unsafe"
    assert fields["restart_supervisor"] is True


def test_status_reader_retries_two_persistent_atomic_replacements_then_fails_closed(
    tmp_path,
    monkeypatch,
):
    calls = 0

    def continually_replaced(_path, *, max_bytes=1_048_576):
        nonlocal calls
        calls += 1
        raise runner._StableArtifactChangedError("simulated replacement")

    monkeypatch.setattr(
        runner,
        "_read_stable_regular_json",
        continually_replaced,
    )

    fields = supervisor_status_health_fields(
        _track(tmp_path),
        repo_root=tmp_path,
        stale_seconds=60.0,
        expected_supervisor_pid=os.getpid(),
        supervisor_status_not_before_epoch_seconds=time.time() - 1.0,
        supervisor_status_startup_grace_seconds=0.1,
    )

    assert calls == 2
    assert fields["supervisor_status"] == "unsafe"
    assert fields["supervisor_status_read_failures"] == 2
    assert fields["restart_supervisor"] is True


def test_supervisor_status_publication_replaces_complete_json_atomically(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "supervisor-status.json"
    status_path.write_text('{"generation":"prior"}\n', encoding="utf-8")
    spec = ManagedDaemonSpec(
        name="atomic-status",
        schema="test.atomic-status",
        repo_root=tmp_path,
        daemon_dir=state_dir,
        runner=(sys.executable, "worker.py"),
        status_path=state_dir / "daemon-status.json",
        supervisor_status_path=status_path,
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure-status.json",
        ensure_check_path=state_dir / "ensure-check.json",
    )
    original_replace = daemon_core.os.replace
    observations: list[tuple[dict[str, object], dict[str, object]]] = []

    def inspect_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        observations.append(
            (
                json.loads(destination_path.read_text(encoding="utf-8")),
                json.loads(source_path.read_text(encoding="utf-8")),
            )
        )
        original_replace(source, destination)

    monkeypatch.setattr(daemon_core.os, "replace", inspect_replace)

    SupervisorStatusContext(spec).write("running", supervisor_pid=os.getpid())

    assert len(observations) == 1
    assert observations[0][0] == {"generation": "prior"}
    assert observations[0][1]["status"] == "running"
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == (
        "running"
    )
    assert list(state_dir.glob(".supervisor-status.json.*.tmp")) == []


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


def _write_restart_admission_worker(tmp_path: Path) -> None:
    (tmp_path / "worker.py").write_text(
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
                "updated_at = ('2000-01-01T00:00:00+00:00' "
                "if generation == 1 else "
                "datetime.now(timezone.utc).isoformat())",
                "Path('state/example_supervisor_status.json').write_text(",
                "    json.dumps({",
                "        'status': 'running',",
                "        'updated_at': updated_at,",
                "        'supervisor_pid': os.getpid(),",
                "    }), encoding='utf-8')",
                "while True:",
                "    if generation > 1:",
                "        time.sleep(0.01)",
                "        status_path = Path('state/example_supervisor_status.json')",
                "        temporary_path = status_path.with_suffix('.tmp')",
                "        temporary_path.write_text(json.dumps({",
                "                'status': 'running',",
                "                'updated_at': datetime.now(timezone.utc).isoformat(),",
                "                'supervisor_pid': os.getpid(),",
                "            }), encoding='utf-8')",
                "        os.replace(temporary_path, status_path)",
                "        continue",
                "    time.sleep(0.02)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_plan_bound_stale_restart_retries_inside_one_admission_boundary(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    _write_restart_admission_worker(tmp_path)
    plan_track = replace(
        _track(tmp_path),
        extra_args=("--plan-bound-dispatch",),
    )
    original_start = runner.start_track
    starts = 0

    def transient_second_admission(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise runner.ConfiguredBoardLiveCapsuleError(
                "simulated stale-restart source replacement"
            )
        safe_track = replace(args[0], extra_args=())
        return original_start(safe_track, *args[1:], **kwargs)

    monkeypatch.setattr(runner, "start_track", transient_second_admission)
    monkeypatch.setattr(
        runner,
        "supervisor_status_health_fields",
        lambda *_args, **_kwargs: {
            "supervisor_status": "stale" if starts < 3 else "live",
            "supervisor_status_age_seconds": 3600.0,
            "restart_supervisor": starts < 3,
        },
    )
    result = run_supervisor_tracks(
        [plan_track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=3.5,
        heartbeat_interval_seconds=2.0,
        supervisor_status_stale_seconds=0.01,
        supervisor_status_startup_grace_seconds=0.0,
        stop_grace_seconds=0.2,
        restart_admission_failure_limit=2,
        output=lambda _line: None,
    )

    assert starts == 3
    assert result["all_trees_fenced"] is True
    assert any(
        receipt["cause"] == "stale_supervisor"
        and receipt["retry_authority"] is True
        for receipt in result["restart_failure_receipts"]
    )


def test_runtime_restart_admission_recovers_after_transient_failure(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    _write_restart_admission_worker(tmp_path)
    original = runner.start_track
    starts = 0

    def fail_first_restart(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise runner.ConfiguredBoardLiveCapsuleError(
                "simulated sealed restart admission race"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "start_track", fail_first_restart)
    output: list[str] = []

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=["--max-restarts", "3"],
        duration_seconds=0.9,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.5,
        supervisor_status_startup_grace_seconds=0.2,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="restart admission retry",
        output=output.append,
    )

    assert result["completed"] is True, result
    assert result["blocked"] == ""
    assert starts == 3, output
    assert len(result["restart_failure_receipts"]) == 1
    receipt = result["restart_failure_receipts"][0]
    assert receipt == {
        "track": "T",
        "cause": "stale_supervisor",
        "failure_count": 1,
        "failure_limit": 3,
        "error_type": "ConfiguredBoardLiveCapsuleError",
        "error_digest": receipt["error_digest"],
        "process_started": False,
        "all_trees_fenced": True,
        "task_completion_authority": False,
        "retry_authority": True,
        "operator_repair_required": False,
    }
    assert str(receipt["error_digest"]).startswith("sha256:")
    assert any("restart admission deferred" in line for line in output)
    assert any("restart admission recovered" in line for line in output)
    assert (tmp_path / "state/generation.txt").read_text(
        encoding="utf-8"
    ) == "2"


def test_runtime_restart_admission_reaches_typed_bounded_terminal(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    _write_restart_admission_worker(tmp_path)
    original = runner.start_track
    starts = 0

    def reject_restarts(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts > 1:
            raise runner.ConfiguredBoardLiveCapsuleError(
                "simulated dirty accepted source"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "start_track", reject_restarts)
    output: list[str] = []

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=["--max-restarts", "2"],
        restart_admission_failure_limit=2,
        duration_seconds=1.0,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        supervisor_status_startup_grace_seconds=0.0,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="restart admission terminal",
        output=output.append,
    )

    assert result["completed"] is False, result
    assert result["all_trees_fenced"] is True
    assert result["blocked"] == (
        "supervisor restart admission failed after 2 bounded attempts for "
        "track T; accepted source or runtime controls require operator repair"
    )
    assert starts == 3
    assert [
        receipt["failure_count"]
        for receipt in result["restart_failure_receipts"]
    ] == [1, 2]
    assert result["restart_failure_receipts"][-1][
        "operator_repair_required"
    ] is True
    assert result["restart_failure_receipts"][-1]["retry_authority"] is False
    assert sum(
        "restart admission deferred" in line for line in output
    ) == 2
    assert any(
        "blocked: supervisor restart admission failed" in line
        for line in output
    )


def test_run_window_never_starts_a_replacement_at_or_after_deadline(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text(
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
        "while True: time.sleep(0.02)\n",
        encoding="utf-8",
    )
    (state_dir / "example_supervisor_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "updated_at": "2000-01-01T00:00:00+00:00",
                "supervisor_pid": 999_999,
            }
        ),
        encoding="utf-8",
    )
    original = runner.start_track
    starts = 0

    def count_starts(*args, **kwargs):
        nonlocal starts
        starts += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "start_track", count_starts)

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.02,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.0,
        supervisor_status_startup_grace_seconds=0.0,
        stop_grace_seconds=0.2,
        output=lambda _line: None,
    )

    assert starts == 1
    assert result["completed"] is True
    assert result["restart_failure_total"] == 0


def test_start_track_rejects_elapsed_birth_deadline_before_popen(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("late process birth was attempted"),
    )

    with pytest.raises(runner.SupervisorRunWindowExpired):
        start_track(
            _track(tmp_path),
            repo_root=tmp_path,
            common_args=[],
            birth_deadline_monotonic_seconds=time.monotonic() - 1.0,
            output=lambda _line: None,
        )


def test_start_track_fences_process_when_popen_crosses_birth_deadline(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    (tmp_path / "worker.py").write_text(
        "import time\nwhile True: time.sleep(0.02)\n",
        encoding="utf-8",
    )
    original_popen = runner.subprocess.Popen
    spawned: list[subprocess.Popen[bytes]] = []

    def delayed_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        spawned.append(process)
        time.sleep(0.05)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", delayed_popen)

    with pytest.raises(runner.SupervisorTrackStartError) as caught:
        start_track(
            _track(tmp_path),
            repo_root=tmp_path,
            common_args=[],
            birth_deadline_monotonic_seconds=time.monotonic() + 0.02,
            output=lambda _line: None,
        )

    assert len(spawned) == 1
    assert caught.value.cause_type == "SupervisorRunWindowExpired"
    assert caught.value.all_trees_fenced is True
    assert not runner.pid_alive(spawned[0].pid)


def test_start_track_treats_positive_infinite_deadline_as_unbounded(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")

    process = start_track(
        _track(tmp_path),
        repo_root=tmp_path,
        common_args=[],
        birth_deadline_monotonic_seconds=float("inf"),
        output=lambda _line: None,
    )

    assert process.wait(timeout=5) == 0


def test_pending_restart_admission_at_finite_deadline_is_not_completion(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text(
        "pass\n",
        encoding="utf-8",
    )
    (state_dir / "example_supervisor_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "updated_at": "2000-01-01T00:00:00+00:00",
                "supervisor_pid": 999_999,
            }
        ),
        encoding="utf-8",
    )
    original = runner.start_track
    starts = 0

    def reject_replacements(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts > 1:
            raise runner.ConfiguredBoardLiveCapsuleError(
                "simulated finite-window admission delay"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "start_track", reject_replacements)

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.18,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.0,
        supervisor_status_startup_grace_seconds=0.0,
        stop_grace_seconds=0.1,
        restart_admission_failure_limit=5,
        output=lambda _line: None,
    )

    assert result["completed"] is False
    assert "remained pending at the finite run-window terminal" in str(
        result["blocked"]
    )
    assert 1 <= result["restart_failure_total"] < 5
    assert result["all_trees_fenced"] is True


def test_exited_supervisor_restart_admission_reaches_bounded_terminal(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    original = runner.start_track
    starts = 0

    def reject_replacements(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts > 1:
            raise runner.ConfiguredBoardLiveCapsuleError(
                "simulated exited-supervisor admission delay"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "start_track", reject_replacements)

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.5,
        heartbeat_interval_seconds=0.05,
        stop_grace_seconds=0.1,
        restart_admission_failure_limit=2,
        output=lambda _line: None,
    )

    assert starts == 3
    assert result["completed"] is False
    assert result["terminal_kind"] == "restart_admission_blocked"
    assert [
        receipt["cause"] for receipt in result["restart_failure_receipts"]
    ] == ["exited_supervisor", "exited_supervisor"]
    assert result["all_trees_fenced"] is True


def test_initial_admission_retries_are_bounded_and_persist_a_terminal_receipt(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    starts = 0

    def reject_initial(*_args, **_kwargs):
        nonlocal starts
        starts += 1
        raise runner.ConfiguredBoardLiveCapsuleError(
            "simulated transient initial admission failure"
        )

    monkeypatch.setattr(runner, "start_track", reject_initial)
    master_pid = tmp_path / "state" / "master.pid"

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=1.0,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.1,
        restart_admission_failure_limit=3,
        master_pid_path=master_pid,
        output=lambda _line: None,
    )

    assert starts == 3
    assert result["completed"] is False
    assert result["all_trees_fenced"] is True
    assert result["master_pid_removed"] is True
    assert result["terminal_kind"] == "restart_admission_blocked"
    assert result["terminal_receipt_written"] is True
    assert not master_pid.exists()
    receipt_path = Path(str(result["terminal_receipt_path"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_cid = receipt.pop("terminal_receipt_cid")
    assert runner.content_identity(receipt) == receipt_cid
    assert receipt["completed"] is False
    assert receipt["all_trees_fenced"] is True
    assert receipt["task_completion_authority"] is False


def test_new_run_archives_prior_terminal_and_publishes_active_generation(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    master_pid = state_dir / "master.pid"
    current_path = runner._terminal_receipt_path(master_pid)
    prior_body = {
        "schema": runner.MULTI_SUPERVISOR_TERMINAL_RECEIPT_SCHEMA,
        "label": "prior",
        "master_pid": 123456,
        "run_started_at_epoch_nanoseconds": 1,
        "all_trees_fenced": True,
        "task_completion_authority": False,
    }
    prior = {
        **prior_body,
        "terminal_receipt_cid": runner.content_identity(prior_body),
    }
    current_path.write_text(
        json.dumps(prior, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    observed_active: dict[str, object] = {}

    def inspect_active_then_reject(*_args, **_kwargs):
        observed_active.update(
            json.loads(current_path.read_text(encoding="utf-8"))
        )
        raise runner.ConfiguredBoardLiveCapsuleError(
            "simulated bounded initial admission failure"
        )

    monkeypatch.setattr(runner, "start_track", inspect_active_then_reject)
    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.1,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.1,
        restart_admission_failure_limit=1,
        master_pid_path=master_pid,
        label="current",
        output=lambda _line: None,
    )

    assert observed_active["schema"] == (
        runner.MULTI_SUPERVISOR_ACTIVE_BINDING_SCHEMA
    )
    assert observed_active["master_pid"] == os.getpid()
    assert observed_active["task_completion_authority"] is False
    active_body = dict(observed_active)
    active_cid = active_body.pop("active_binding_cid")
    assert runner.content_identity(active_body) == active_cid
    assert result["active_binding_cid"] == active_cid
    archive_path = Path(str(result["archived_run_artifact_path"]))
    assert json.loads(archive_path.read_text(encoding="utf-8")) == prior
    terminal = json.loads(current_path.read_text(encoding="utf-8"))
    assert terminal["schema"] == runner.MULTI_SUPERVISOR_TERMINAL_RECEIPT_SCHEMA
    assert terminal["active_binding_cid"] == active_cid


@pytest.mark.parametrize("prior_kind", ["active", "unfenced_terminal"])
def test_new_run_rejects_prior_generation_without_fenced_terminal_proof(
    tmp_path,
    monkeypatch,
    prior_kind,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    master_pid = state_dir / "master.pid"
    current_path = runner._terminal_receipt_path(master_pid)
    if prior_kind == "active":
        prior_body = {
            "schema": runner.MULTI_SUPERVISOR_ACTIVE_BINDING_SCHEMA,
            "label": "prior",
            "active": True,
            "master_pid": 123456,
            "run_started_at_epoch_nanoseconds": 1,
            "task_completion_authority": False,
        }
        identity_field = "active_binding_cid"
    else:
        prior_body = {
            "schema": runner.MULTI_SUPERVISOR_TERMINAL_RECEIPT_SCHEMA,
            "label": "prior",
            "master_pid": 123456,
            "run_started_at_epoch_nanoseconds": 1,
            "all_trees_fenced": False,
            "task_completion_authority": False,
        }
        identity_field = "terminal_receipt_cid"
    prior = {
        **prior_body,
        identity_field: runner.content_identity(prior_body),
    }
    current_path.write_text(
        json.dumps(prior, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    starts = 0

    def forbidden_start(*_args, **_kwargs):
        nonlocal starts
        starts += 1
        raise AssertionError("unfenced predecessor must block every child birth")

    monkeypatch.setattr(runner, "start_track", forbidden_start)
    with pytest.raises(
        ValueError,
        match="prior run generation has no fully fenced terminal proof",
    ):
        run_supervisor_tracks(
            [_track(tmp_path)],
            repo_root=tmp_path,
            common_args=[],
            duration_seconds=0.1,
            heartbeat_interval_seconds=0.01,
            stop_grace_seconds=0.1,
            master_pid_path=master_pid,
            output=lambda _line: None,
        )

    assert starts == 0
    assert json.loads(current_path.read_text(encoding="utf-8")) == prior
    assert not master_pid.exists()


def test_terminal_post_replace_failure_preserves_master_marker(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    master_pid = state_dir / "master.pid"
    current_path = runner._terminal_receipt_path(master_pid)
    original_write = runner.write_json_atomic

    def replace_then_fail(path, payload, *, sync_directory=False):
        original_write(
            path,
            payload,
            sync_directory=sync_directory,
        )
        if (
            Path(path) == current_path
            and payload.get("schema")
            == runner.MULTI_SUPERVISOR_TERMINAL_RECEIPT_SCHEMA
        ):
            raise OSError("simulated directory fsync failure after replace")

    monkeypatch.setattr(runner, "write_json_atomic", replace_then_fail)
    monkeypatch.setattr(
        runner,
        "start_track",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.ConfiguredBoardLiveCapsuleError("bounded admission failure")
        ),
    )
    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.1,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.1,
        restart_admission_failure_limit=1,
        master_pid_path=master_pid,
        output=lambda _line: None,
    )

    assert result["completed"] is False
    assert result["terminal_receipt_written"] is False
    assert result["master_pid_removed"] is False
    assert master_pid.read_text(encoding="ascii") == f"{os.getpid()}\n"
    visible = json.loads(current_path.read_text(encoding="utf-8"))
    assert visible["master_pid"] == os.getpid()
    assert visible["active_binding_cid"] == result["active_binding_cid"]


def test_master_marker_removal_reports_success_only_after_directory_fsync(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    master_pid = state_dir / "master.pid"
    original_fsync_directory = runner._fsync_directory

    def fail_marker_directory_fsync(path):
        if Path(path) == state_dir:
            raise OSError("simulated marker-directory fsync failure")
        return original_fsync_directory(path)

    monkeypatch.setattr(runner, "_fsync_directory", fail_marker_directory_fsync)
    monkeypatch.setattr(
        runner,
        "start_track",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.ConfiguredBoardLiveCapsuleError("bounded admission failure")
        ),
    )
    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.1,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.1,
        restart_admission_failure_limit=1,
        master_pid_path=master_pid,
        output=lambda _line: None,
    )

    assert result["terminal_receipt_written"] is True
    assert result["master_pid_removed"] is False
    assert not master_pid.exists()


def test_teardown_defers_signal_until_children_and_receipt_are_fenced(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_restart_admission_worker(tmp_path)
    master_pid = state_dir / "master.pid"
    original_stop_tracks = runner.stop_tracks
    prior_term = signal.getsignal(signal.SIGTERM)

    def unsafe_prior_handler(_signum, _frame):
        raise RuntimeError("prior handler ran before teardown completed")

    def signal_during_stop(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return original_stop_tracks(*args, **kwargs)

    signal.signal(signal.SIGTERM, unsafe_prior_handler)
    monkeypatch.setattr(runner, "stop_tracks", signal_during_stop)
    try:
        result = run_supervisor_tracks(
            [_track(tmp_path)],
            repo_root=tmp_path,
            common_args=[],
            duration_seconds=0.05,
            heartbeat_interval_seconds=0.01,
            stop_grace_seconds=0.2,
            master_pid_path=master_pid,
            output=lambda _line: None,
        )
        assert signal.getsignal(signal.SIGTERM) is unsafe_prior_handler
    finally:
        signal.signal(signal.SIGTERM, prior_term)

    assert result["completed"] is False
    assert result["interrupted"] == f"received signal {signal.SIGTERM}"
    assert result["all_trees_fenced"] is True
    assert result["terminal_receipt_written"] is True
    assert result["master_pid_removed"] is True


def test_teardown_keeps_one_combined_handler_without_replacement_window(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    master_pid = state_dir / "master.pid"
    original_signal = signal.signal
    prior_term = signal.getsignal(signal.SIGTERM)
    prior_int = signal.getsignal(signal.SIGINT)
    installed_handlers: list[tuple[int, str]] = []

    def record_handler_install(signum, handler):
        name = getattr(handler, "__name__", repr(handler))
        installed_handlers.append((int(signum), name))
        assert name != "defer_teardown_signal"
        return original_signal(signum, handler)

    monkeypatch.setattr(
        runner.signal,
        "signal",
        record_handler_install,
    )
    monkeypatch.setattr(
        runner,
        "start_track",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.ConfiguredBoardLiveCapsuleError("bounded admission failure")
        ),
    )
    try:
        result = run_supervisor_tracks(
            [_track(tmp_path)],
            repo_root=tmp_path,
            common_args=[],
            duration_seconds=0.1,
            heartbeat_interval_seconds=0.01,
            stop_grace_seconds=0.1,
            restart_admission_failure_limit=1,
            master_pid_path=master_pid,
            output=lambda _line: None,
        )
        assert signal.getsignal(signal.SIGTERM) is prior_term
        assert signal.getsignal(signal.SIGINT) is prior_int
    finally:
        original_signal(signal.SIGTERM, prior_term)
        original_signal(signal.SIGINT, prior_int)

    assert result["completed"] is False
    assert result["interrupted"] == ""
    assert result["all_trees_fenced"] is True
    assert result["terminal_receipt_written"] is True
    assert result["master_pid_removed"] is True
    assert installed_handlers[:2] == [
        (signal.SIGTERM, "_handle_signal"),
        (signal.SIGINT, "_handle_signal"),
    ]


def test_teardown_never_blocks_on_a_stale_pending_signal_snapshot(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    master_pid = state_dir / "master.pid"
    pending_reads = 0
    timed_waits: list[tuple[set[signal.Signals], float]] = []

    def stale_pending_snapshot():
        nonlocal pending_reads
        pending_reads += 1
        return {signal.SIGTERM} if pending_reads == 1 else set()

    def signal_was_consumed_elsewhere(signals, timeout):
        timed_waits.append((set(signals), float(timeout)))
        return None

    monkeypatch.setattr(runner.signal, "sigpending", stale_pending_snapshot)
    monkeypatch.setattr(
        runner.signal,
        "sigtimedwait",
        signal_was_consumed_elsewhere,
    )
    monkeypatch.setattr(
        runner,
        "start_track",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.ConfiguredBoardLiveCapsuleError("bounded admission failure")
        ),
    )

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.1,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.1,
        restart_admission_failure_limit=1,
        master_pid_path=master_pid,
        output=lambda _line: None,
    )

    assert result["interrupted"] == f"received signal {signal.SIGTERM}"
    assert result["all_trees_fenced"] is True
    assert result["terminal_receipt_written"] is True
    assert timed_waits == [({signal.SIGTERM}, 0.0)]


def test_signal_at_terminal_outcome_freeze_matches_exact_receipt(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_restart_admission_worker(tmp_path)
    master_pid = state_dir / "master.pid"
    terminal_path = runner._terminal_receipt_path(master_pid)
    source_lines, first_line = inspect.getsourcelines(run_supervisor_tracks)
    boundary_line = first_line + next(
        index
        for index, line in enumerate(source_lines)
        if line.strip() == "terminal_outcome_frozen = True"
    )
    prior_term = signal.getsignal(signal.SIGTERM)
    caller_deliveries: list[int] = []
    signal_sent = False

    def caller_handler(signum, _frame):
        caller_deliveries.append(int(signum))

    def trace_boundary(frame, event, _arg):
        nonlocal signal_sent
        if (
            not signal_sent
            and event == "line"
            and frame.f_code is run_supervisor_tracks.__code__
            and frame.f_lineno == boundary_line
        ):
            signal_sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return trace_boundary

    signal.signal(signal.SIGTERM, caller_handler)
    sys.settrace(trace_boundary)
    try:
        result = run_supervisor_tracks(
            [_track(tmp_path)],
            repo_root=tmp_path,
            common_args=[],
            duration_seconds=0.02,
            heartbeat_interval_seconds=0.01,
            stop_grace_seconds=0.1,
            master_pid_path=master_pid,
            output=lambda _line: None,
        )
    finally:
        sys.settrace(None)
        signal.signal(signal.SIGTERM, prior_term)

    receipt = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert signal_sent is True
    assert receipt["completed"] is result["completed"]
    assert receipt["terminal_kind"] == result["terminal_kind"]
    if caller_deliveries:
        assert caller_deliveries == [signal.SIGTERM]
        assert result["completed"] is True
        assert result["interrupted"] == ""
        assert receipt["interrupted"] is False
        assert receipt["interrupted_digest"] == ""
    else:
        assert result["completed"] is False
        assert result["interrupted"] == f"received signal {signal.SIGTERM}"
        assert result["terminal_kind"] == "interrupted"
        assert receipt["interrupted"] is True
        assert receipt["interrupted_digest"] == (
            "sha256:"
            + hashlib.sha256(
                f"received signal {signal.SIGTERM}".encode("utf-8")
            ).hexdigest()
        )


def test_signal_during_terminal_publication_preserves_frozen_receipt(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_restart_admission_worker(tmp_path)
    master_pid = state_dir / "master.pid"
    current_path = runner._terminal_receipt_path(master_pid)
    original_write = runner.write_json_atomic
    terminal_signal_sent = False
    terminal_write_count = 0
    prior_term = signal.getsignal(signal.SIGTERM)
    delivered: list[int] = []

    def caller_handler(signum, _frame):
        delivered.append(int(signum))

    def signal_during_terminal_write(path, payload, *, sync_directory=False):
        nonlocal terminal_signal_sent, terminal_write_count
        if (
            Path(path) == current_path
            and payload.get("schema")
            == runner.MULTI_SUPERVISOR_TERMINAL_RECEIPT_SCHEMA
        ):
            terminal_write_count += 1
            if not terminal_signal_sent:
                terminal_signal_sent = True
                os.kill(os.getpid(), signal.SIGTERM)
        return original_write(
            path,
            payload,
            sync_directory=sync_directory,
        )

    signal.signal(signal.SIGTERM, caller_handler)
    monkeypatch.setattr(runner, "write_json_atomic", signal_during_terminal_write)
    try:
        result = run_supervisor_tracks(
            [_track(tmp_path)],
            repo_root=tmp_path,
            common_args=[],
            duration_seconds=0.02,
            heartbeat_interval_seconds=0.01,
            stop_grace_seconds=0.2,
            master_pid_path=master_pid,
            output=lambda _line: None,
        )
        assert signal.getsignal(signal.SIGTERM) is caller_handler
    finally:
        signal.signal(signal.SIGTERM, prior_term)

    assert terminal_signal_sent is True
    assert terminal_write_count == 1
    assert delivered == [signal.SIGTERM]
    assert result["completed"] is True
    assert result["interrupted"] == ""
    receipt = json.loads(current_path.read_text(encoding="utf-8"))
    assert receipt["terminal_kind"] == "run_window_complete"
    assert receipt["interrupted"] is False
    assert receipt["completed"] is True
    receipt_cid = receipt.pop("terminal_receipt_cid")
    assert runner.content_identity(receipt) == receipt_cid


def test_signal_during_post_commit_marker_cleanup_uses_caller_policy(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_restart_admission_worker(tmp_path)
    master_pid = state_dir / "master.pid"
    terminal_path = runner._terminal_receipt_path(master_pid)
    original_remove = runner._remove_owned_pid_projection
    prior_term = signal.getsignal(signal.SIGTERM)
    delivered: list[int] = []
    signal_sent = False

    def caller_handler(signum, _frame):
        delivered.append(int(signum))

    def signal_during_marker_cleanup(path, expected_pid):
        nonlocal signal_sent
        if not signal_sent and Path(path) == master_pid:
            signal_sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return original_remove(path, expected_pid)

    signal.signal(signal.SIGTERM, caller_handler)
    monkeypatch.setattr(
        runner,
        "_remove_owned_pid_projection",
        signal_during_marker_cleanup,
    )
    try:
        result = run_supervisor_tracks(
            [_track(tmp_path)],
            repo_root=tmp_path,
            common_args=[],
            duration_seconds=0.02,
            heartbeat_interval_seconds=0.01,
            stop_grace_seconds=0.2,
            master_pid_path=master_pid,
            output=lambda _line: None,
        )
        assert signal.getsignal(signal.SIGTERM) is caller_handler
    finally:
        signal.signal(signal.SIGTERM, prior_term)

    receipt = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert signal_sent is True
    assert delivered == [signal.SIGTERM]
    assert result["completed"] is True
    assert result["interrupted"] == ""
    assert result["master_pid_removed"] is True
    assert receipt["completed"] is True
    assert receipt["interrupted"] is False
    assert receipt["terminal_commit_boundary"] == (
        "durable_terminal_receipt_before_master_marker_cleanup"
    )
    assert receipt["terminal_signal_boundary"] == (
        "outcome_frozen_under_term_int_barrier_before_publication"
    )
    assert receipt["post_commit_signal_policy"] == (
        "restore_caller_disposition_and_replay_post_outcome_signal"
    )


def test_teardown_aggregates_one_track_failure_and_fences_later_tracks(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    first = _track(tmp_path)
    second = replace(
        first,
        name="U",
        log_path=tmp_path / "logs" / "second.log",
        supervisor_pid_path=state_dir / "second_supervisor.pid",
        daemon_pid_path=state_dir / "second_daemon.pid",
    )
    master_pid = state_dir / "master.pid"
    prior_term = signal.getsignal(signal.SIGTERM)
    started = 0
    terminated: list[int] = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(*_args, **_kwargs):
            return 0

    def fake_start(*_args, **_kwargs):
        nonlocal started
        started += 1
        return FakeProcess(500000 + started)

    def aggregate_termination(process, *, grace_seconds):
        del grace_seconds
        terminated.append(process.pid)
        if len(terminated) == 1:
            raise runner.ProcessIdentityMismatch(
                "simulated first-track identity drift"
            )
        return True, (process.pid,)

    monkeypatch.setattr(runner, "start_track", fake_start)
    monkeypatch.setattr(
        runner,
        "_terminate_managed_process",
        aggregate_termination,
    )
    result = run_supervisor_tracks(
        [first, second],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.0,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.1,
        master_pid_path=master_pid,
        output=lambda _line: None,
    )

    assert terminated == [500001, 500002]
    assert result["completed"] is False
    assert result["all_trees_fenced"] is False
    assert result["master_pid_removed"] is False
    assert master_pid.exists()
    assert signal.getsignal(signal.SIGTERM) is prior_term
    receipt = json.loads(
        Path(str(result["terminal_receipt_path"])).read_text(encoding="utf-8")
    )
    assert receipt["blocked"] is True
    assert receipt["all_trees_fenced"] is False
    assert [
        item["track"] for item in receipt["stop_failure_receipts"]
    ] == [first.name]


def test_detached_parent_rejects_stale_binding_and_preserves_dead_child_marker(
    tmp_path,
    monkeypatch,
):
    class ExitedChild:
        pid = 424242

        @staticmethod
        def poll():
            return 1

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: ExitedChild(),
    )
    args = runner.build_arg_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--stamp",
            "detached-preserve",
        ]
    )
    _log, marker = runner._master_paths(args)
    prior_body = {
        "schema": runner.MULTI_SUPERVISOR_ACTIVE_BINDING_SCHEMA,
        "label": "prior",
        "active": True,
        "master_pid": 123456,
        "run_started_at_epoch_nanoseconds": 1,
        "task_completion_authority": False,
    }
    prior = {
        **prior_body,
        "active_binding_cid": runner.content_identity(prior_body),
    }
    runner.write_json_atomic(
        runner._terminal_receipt_path(marker),
        prior,
        sync_directory=True,
    )

    with pytest.raises(
        ValueError,
        match="exited before active-generation binding",
    ):
        runner.launch_detached(args, [])

    assert marker.read_text(encoding="ascii") == "424242\n"
    assert json.loads(
        runner._terminal_receipt_path(marker).read_text(encoding="utf-8")
    ) == prior


def test_detached_launch_returns_only_after_active_binding_ack(
    tmp_path,
    monkeypatch,
):
    class LiveChild:
        pid = 424243

        @staticmethod
        def poll():
            return None

    acknowledgements: list[tuple[Path, int, int]] = []

    def acknowledge(master_pid, process, *, not_before_epoch_nanoseconds):
        acknowledgements.append(
            (
                Path(master_pid),
                process.pid,
                int(not_before_epoch_nanoseconds),
            )
        )
        return {"active": True}

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: LiveChild(),
    )
    monkeypatch.setattr(
        runner,
        "_wait_for_detached_active_binding",
        acknowledge,
    )
    args = runner.build_arg_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--stamp",
            "detached-ack",
        ]
    )
    result = runner.launch_detached(args, [])

    assert len(acknowledgements) == 1
    assert acknowledgements[0][1] == 424243
    assert acknowledgements[0][2] > 0
    assert Path(str(result["master_pid_file"])).read_text(
        encoding="ascii"
    ) == "424243\n"


def test_generic_detach_rejects_plan_bound_wave_before_pid_reservation(
    tmp_path,
):
    args = runner.build_arg_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--stamp",
            "plan-bound-detach",
            "--plan-bound-wave",
        ]
    )
    _log, marker = runner._master_paths(args)

    with pytest.raises(ValueError, match="cannot use the generic detached"):
        runner.launch_detached(args, [])

    assert not marker.exists()


def test_unknown_restart_failure_never_reports_fenced_or_removes_master_marker(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    _write_restart_admission_worker(tmp_path)
    original = runner.start_track
    starts = 0

    def fail_unknown_after_first_start(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts > 1:
            raise RuntimeError("simulated unclassified launch boundary")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "start_track", fail_unknown_after_first_start)
    master_pid = tmp_path / "state" / "master.pid"

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.5,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        supervisor_status_startup_grace_seconds=0.0,
        stop_grace_seconds=0.2,
        master_pid_path=master_pid,
        output=lambda _line: None,
    )

    assert starts == 2
    assert result["completed"] is False
    assert result["all_trees_fenced"] is False
    assert result["master_pid_removed"] is False
    assert master_pid.exists()
    assert result["restart_failure_receipts"][-1]["process_started"] == (
        "unknown"
    )
    receipt = json.loads(
        Path(str(result["terminal_receipt_path"])).read_text(encoding="utf-8")
    )
    assert receipt["completed"] is False
    assert receipt["all_trees_fenced"] is False


def test_restart_interruption_is_not_converted_into_retry_authority(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    _write_restart_admission_worker(tmp_path)
    original = runner.start_track
    starts = 0

    def interrupt_restart(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts > 1:
            raise runner.SupervisorRunInterrupted("simulated operator stop")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "start_track", interrupt_restart)

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.5,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        supervisor_status_startup_grace_seconds=0.0,
        stop_grace_seconds=0.2,
        output=lambda _line: None,
    )

    assert starts == 2
    assert result["completed"] is False
    assert result["interrupted"] == "simulated operator stop"
    assert result["restart_failure_total"] == 0
    assert result["restart_failure_receipts"] == []
    assert result["all_trees_fenced"] is True


def test_deferred_signal_does_not_hide_unfenced_start_terminal(
    tmp_path,
    monkeypatch,
):
    import signal

    state_root = tmp_path / "state"
    state_root.mkdir()
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    profile = runner.LifecycleProfile(
        target_id="supervisor-track:T",
        run_id="test-unfenced-start",
        configuration_root="test-unfenced-start-config",
        repository_root=str(tmp_path.resolve()),
        state_root=str(state_root.resolve()),
        run_root=str((state_root / "lifecycle-runs" / "T").resolve()),
        argv=(sys.executable, str((tmp_path / "worker.py").resolve())),
        cwd=str(tmp_path.resolve()),
    )

    def signal_then_report_unfenced(*_args, **_kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        raise runner.SupervisorTrackStartError(
            "simulated unproved post-birth fence",
            pid=999_999,
            profile=profile,
            all_trees_fenced=False,
            cause_type="InjectedPublicationError",
        )

    monkeypatch.setattr(runner, "start_track", signal_then_report_unfenced)
    master_pid = state_root / "master.pid"

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=1.0,
        heartbeat_interval_seconds=0.01,
        stop_grace_seconds=0.1,
        master_pid_path=master_pid,
        output=lambda _line: None,
    )

    assert result["completed"] is False
    assert result["interrupted"] == ""
    assert result["terminal_kind"] == "restart_admission_blocked"
    assert result["all_trees_fenced"] is False
    assert result["master_pid_removed"] is False
    assert master_pid.exists()
    assert result["restart_failure_receipts"][-1]["all_trees_fenced"] is False
    assert result["restart_failure_receipts"][-1]["retry_authority"] is False


def test_signal_after_popen_birth_waits_for_process_registry_then_fences(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    (tmp_path / "worker.py").write_text(
        "import signal,time\n"
        "from pathlib import Path\n"
        "def stop(*_):\n"
        "    Path('state/term.marker').write_text('term', encoding='utf-8')\n"
        "    exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "Path('state/ready.marker').write_text('ready', encoding='utf-8')\n"
        "while True: time.sleep(0.02)\n",
        encoding="utf-8",
    )
    original_popen = runner.subprocess.Popen
    spawned: list[subprocess.Popen[bytes]] = []

    def signal_at_popen_return(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        spawned.append(process)
        ready = tmp_path / "state" / "ready.marker"
        ready_deadline = time.monotonic() + 2.0
        while not ready.exists() and time.monotonic() < ready_deadline:
            time.sleep(0.01)
        assert ready.exists()
        os.kill(os.getpid(), signal.SIGTERM)
        return process

    import signal

    monkeypatch.setattr(runner.subprocess, "Popen", signal_at_popen_return)

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=1.0,
        heartbeat_interval_seconds=0.05,
        stop_grace_seconds=0.2,
        output=lambda _line: None,
    )

    assert len(spawned) == 1
    assert result["completed"] is False
    assert result["interrupted"] == f"received signal {signal.SIGTERM}"
    assert result["all_trees_fenced"] is True
    assert not runner.pid_alive(spawned[0].pid)
    assert (tmp_path / "state" / "term.marker").read_text(
        encoding="utf-8"
    ) == "term"


def test_signal_before_start_track_return_waits_for_registry_then_fences(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "state").mkdir()
    (tmp_path / "worker.py").write_text(
        "import signal,time\n"
        "from pathlib import Path\n"
        "def stop(*_):\n"
        "    Path('state/term.marker').write_text('term', encoding='utf-8')\n"
        "    exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "Path('state/ready.marker').write_text('ready', encoding='utf-8')\n"
        "while True: time.sleep(0.02)\n",
        encoding="utf-8",
    )
    original_start = runner.start_track
    spawned: list[subprocess.Popen[bytes]] = []

    def signal_before_return(*args, **kwargs):
        process = original_start(*args, **kwargs)
        spawned.append(process)
        ready = tmp_path / "state" / "ready.marker"
        ready_deadline = time.monotonic() + 2.0
        while not ready.exists() and time.monotonic() < ready_deadline:
            time.sleep(0.01)
        assert ready.exists()
        os.kill(os.getpid(), signal.SIGTERM)
        return process

    import signal

    monkeypatch.setattr(runner, "start_track", signal_before_return)

    result = run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=1.0,
        heartbeat_interval_seconds=0.05,
        stop_grace_seconds=0.2,
        output=lambda _line: None,
    )

    assert len(spawned) == 1
    assert result["completed"] is False
    assert result["interrupted"] == f"received signal {signal.SIGTERM}"
    assert result["all_trees_fenced"] is True
    assert not runner.pid_alive(spawned[0].pid)
    assert (tmp_path / "state" / "term.marker").read_text(
        encoding="utf-8"
    ) == "term"


def test_post_popen_publication_failure_fences_child_before_raising(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "worker.py").write_text(
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
        "while True: time.sleep(0.02)\n",
        encoding="utf-8",
    )

    def reject_publication(_line):
        raise OSError("simulated log publication failure")

    with pytest.raises(runner.SupervisorTrackStartError) as caught:
        start_track(
            _track(tmp_path),
            repo_root=tmp_path,
            common_args=[],
            python_executable=sys.executable,
            output=reject_publication,
        )

    error = caught.value
    assert error.all_trees_fenced is True
    assert error.cause_type == "OSError"
    assert not runner.pid_alive(error.pid)
    assert not (tmp_path / "state" / "example_supervisor.pid").exists()


def test_restart_admission_limit_validation_precedes_master_pid_publication(
    tmp_path,
):
    master_pid = tmp_path / "state" / "master.pid"

    with pytest.raises(ValueError, match="restart admission failure limit"):
        run_supervisor_tracks(
            [_track(tmp_path)],
            repo_root=tmp_path,
            common_args=[],
            duration_seconds=0.0,
            restart_admission_failure_limit=0,
            master_pid_path=master_pid,
            output=lambda _line: None,
        )

    assert not master_pid.exists()


def test_normal_cli_returns_nonzero_for_typed_runtime_blocker(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_supervisor_tracks",
        lambda *_args, **_kwargs: {
            "completed": False,
            "all_trees_fenced": True,
            "shared_authority_terminals": [],
            "replan_required": False,
        },
    )

    exit_code = runner.main(
        [
            "--repo-root",
            str(tmp_path),
            "--track",
            "T|worker.py|logs/run.log|state/supervisor.pid|state/daemon.pid",
            "--duration-seconds",
            "0",
            "--master-dir",
            "state",
        ]
    )

    assert exit_code == 2


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
